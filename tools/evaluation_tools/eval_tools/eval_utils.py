import os
from copy import deepcopy
from collections import namedtuple, OrderedDict, defaultdict
from tqdm import tqdm
import logging as logger
import json

import numpy as np



def bbox_overlaps(bboxes1,
                  bboxes2,
                  mode='iou',
                  eps=1e-6,
                  use_legacy_coordinate=False):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (ndarray): Shape (n, 4)
        bboxes2 (ndarray): Shape (k, 4)
        mode (str): IOU (intersection over union) or IOF (intersection
            over foreground)
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.

    Returns:
        ious (ndarray): Shape (n, k)
    """

    assert mode in ['iou', 'iof']
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
            y_end - y_start + extra_length, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None,
                 use_legacy_coordinate=False,
                 **kwargs):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be
            evaluated, in the format [(min1, max1), (min2, max2), ...].
            Default: None.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """

    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.

    num_valid_gts = gt_bboxes.shape[0]

    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    if gt_bboxes.shape[0] == 0:
        gt_bboxes = gt_bboxes_ignore
    elif gt_bboxes_ignore.shape[0] == 0:
        pass
    else:
        gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    # BUG
    # if num_valid_gts != 0:
    #     if gt_bboxes_ignore.shape[0] != 0:
    #         gt_ignore_inds = np.concatenate(
    #             (np.zeros(gt_bboxes.shape[0], dtype=bool),
    #             np.ones(gt_bboxes_ignore.shape[0], dtype=bool)))
    #     # stack gt_bboxes and gt_bboxes_ignore for convenience
    #         gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
    #     else:
    #         gt_ignore_inds = np.zeros(gt_bboxes.shape[0], dtype=bool)

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    scores = np.zeros((num_scales, num_dets), dtype=np.float32)
    gt_covered = np.zeros((num_scales, num_gts), dtype=int)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives

    if num_dets == 0:
        return tp, fp, scores, gt_covered[:,:num_valid_gts]

    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
            if num_dets > 0:
                scores[...] = det_bboxes[:,-1]
        else:
            det_areas = (
                det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (
                    det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp, scores, gt_covered[:,:num_valid_gts]


    ious = bbox_overlaps(
        det_bboxes, gt_bboxes, use_legacy_coordinate=use_legacy_coordinate)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    
    for k, (min_area, max_area) in enumerate(area_ranges):
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + extra_length) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1] + extra_length)
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)

        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if gt_covered[k, matched_gt] == 0:
                        gt_covered[k, matched_gt] = i + 1
                        tp[k, i] = 1
                        scores[k, i] = det_bboxes[i][-1]
                    else:
                        fp[k, i] = 1
                        scores[k, i] = det_bboxes[i][-1]
                # otherwise ignore this detected bbox, tp = 0, fp = 0

            elif min_area is None:
                fp[k, i] = 1
                scores[k, i] = det_bboxes[i][-1]
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0] + extra_length) * (
                    bbox[3] - bbox[1] + extra_length)
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
                    scores[k, i] = det_bboxes[i][-1]
    return tp, fp, scores, gt_covered[:,:num_valid_gts]


def get_recall_at_fa(pos_scores, neg_scores, total_num_gt, frame_num, target_fa=None):
    total_neg_num = neg_scores.shape[0]
    total_pos_num = pos_scores.shape[0]
    logger.info('total_fp_num: {} , total_tp_num: {}, total_frame_num:{}, total_gt_bbox_num:{}'
                .format(total_neg_num, total_pos_num, frame_num, total_num_gt))
    
    if total_num_gt == 0: 
        logger.info('There is no GT')
        return [(0,0,0)]

    if total_neg_num == 0:
        return [(total_pos_num/total_num_gt,0,0)]
    
    if target_fa is None:
        max_neg_log10 = np.floor(np.log10(frame_num)).astype(np.int32)
        min_neg_log10 = 0
        fa_list = np.arange(min_neg_log10, max_neg_log10, 0.25)
        if (max_neg_log10 < 1):
            logger.warning("The number of fp in testset is too small: %d" % frame_num)
            return [(0, 0, 1)]
        real_min_neg = -np.log10(total_neg_num / frame_num)

        if real_min_neg > fa_list[0]:
            for ii in range(len(fa_list)):
                if fa_list[ii] > -np.log10(total_neg_num / frame_num):
                    fa_list = np.concatenate((np.array([-np.log10(total_neg_num / frame_num)]), np.array(fa_list[ii:])))
                    break
                elif ii == len(fa_list) - 1:
                    fa_list = np.array([-np.log10(total_neg_num / frame_num)])

    else:
        fa_list = []
        target_fa = deepcopy(target_fa)
        target_fa.sort(reverse=True)
        for idx, fa in enumerate(target_fa):
            if fa > total_neg_num / frame_num:
                logger.warning("Target fa {} is larger than maximum fa({}) of the testset".format(fa, total_neg_num / frame_num))
                logger.info("change to calculate Recall@fa={}".format(total_neg_num / frame_num))
                fa_list.append(-np.log10(total_neg_num / frame_num))
                target_fa[idx] = total_neg_num / frame_num
                # target_fa = target_fa[:idx + 1]
                continue
            fa_list.append(-np.log10(fa))

    eval_result = []

    if target_fa is None:
        logger.info('Searching from top {} fp/frame bboxes...'.format(1 / float(np.power(10, fa_list[0]))))
    else:
        logger.info('Searching for top {} fp/frame bboxes...'.format(target_fa))
    current_used_negs = neg_scores[:]
    sorted_negs_idxs = np.argsort(neg_scores)
    logger.info("Done")


    pos_scores = list(pos_scores)
    for idx, i in enumerate(fa_list):
        neg_num = int(10 ** (-i) * frame_num)
        current_used_negs = np.partition(current_used_negs, -neg_num)[-neg_num:]
        sorted_negs_idxs = sorted_negs_idxs[-neg_num:]
        thres = current_used_negs[0]
        pos_num = np.count_nonzero(pos_scores > thres)
        if total_num_gt > 0:
            recall = float(pos_num) / total_num_gt
        else:
            recall = 0
        if target_fa is None:
            eval_result.append((recall, 1 / float(np.power(10, i)), thres))
        else:
            eval_result.append((recall, target_fa[idx], thres))


    return eval_result


def get_recall_at_thresh(pos_scores, neg_scores, total_num_gt, frame_num, target_thresh=None):
    total_neg_num = neg_scores.shape[0]
    total_pos_num = pos_scores.shape[0]
    logger.info('total_fp_num: {} , total_tp_num: {}, total_frame_num:{}, total_gt_bbox_num:{}'
                .format(total_neg_num, total_pos_num, frame_num, total_num_gt))
    
    if total_num_gt == 0: 
        logger.info('There is no GT')
        return [(0,0,0)]

    if total_neg_num == 0:
        return [(total_pos_num/total_num_gt,0,0)]
    
    if target_thresh == None:
        target_thresh = np.arange(0.1, 1, 0.1)
        logger.info('Searching from threshold 0.1 ...')
    else:
        target_thresh.sort()
        logger.info('Searching threshold in {} ...'.format(target_thresh))

    result = []
    pos_scores = list(pos_scores)
    for idx, o_thres in enumerate(target_thresh):
        new_fa_num = sum(np.sum(sub_neg_scores > o_thres) for sub_neg_scores in neg_scores)
        new_recall_num = np.count_nonzero(pos_scores > np.float32(o_thres))
        result.append((new_recall_num/total_num_gt, new_fa_num/frame_num, o_thres))

    return result


def rearrange_list(lst):
    if len(lst) != 4:
        raise ValueError("The list must contain exactly 4 elements.")
    return [lst[1], lst[0], lst[3], lst[2]]


class Evaluator:
    def __init__(self, infer_res, gt_res, out_path, mode, iou_thresh=0.5, overwrite=False, targets=None, total_classes=[]):
        assert mode in ['fa' ,'thresh']

        if len(infer_res) != len(gt_res):
            use_infer_res = dict()
            use_gt_res = dict()
            for k in gt_res.keys():
                if k in infer_res:
                    use_infer_res[k] = infer_res[k]
                else:
                    use_infer_res[k] = dict(
                                        content_ann = dict(
                                        bboxes = list(),
                                        scores = list(),
                                        labels = list(),
                                        ),
                                    )
            self.infer_res = use_infer_res
            self.gt_res = gt_res
        else:
            self.infer_res = infer_res
            self.gt_res = gt_res
        self.out_path = out_path
        self.overwrite = overwrite
        self.frame_num = len(self.gt_res)
        self.mode = mode
        self.iou_thresh = iou_thresh

        
        if targets is None or targets[0]=='-1':
            target_list = None
        else:
            target_list = [float(tt) for tt in targets]

        if mode == 'fa':
            self.target_fa = target_list
        else:
            self.target_thresh = target_list

        self.total_classes = total_classes
        

    def get_recall(self):
        # variables for globle info
        tp_scores = np.array([])
        fp_scores = np.array([])
        total_res = []
        
        missed_bbox_num = 0
        total_num_gt = 0
        total_info_dict = OrderedDict()
        non_covered_gt = OrderedDict()
        for idx_cls, cls in enumerate(self.total_classes):
            # variables for each class
            cur_cls_num_gt = 0
            cur_cls_tp_scores = np.array([])
            cur_cls_fp_scores = np.array([])

            for key in self.infer_res.keys():
                # process infer res
                cur_infer = []
                infer_bboxes = self.infer_res[key]['content_ann']['bboxes']
                infer_scores = self.infer_res[key]['content_ann']['scores']
                infer_labels = self.infer_res[key]['content_ann']['labels']

                for i in range(len(infer_bboxes)):
                    if infer_labels[i][0] == idx_cls:
                        # adhoc: For reversed input bbox
                        # info = rearrange_list(infer_bboxes[i]) + infer_scores[i]

                        info = infer_bboxes[i] + infer_scores[i]
                        cur_infer.append(np.array(info))
                cur_infer = np.array(cur_infer)

                # process gt infomation
                cur_gt = []
                cur_gt_ignore = []
                cur_gt_bboxes = []
                cur_cares = []
                gt_bboxes = self.gt_res[key]['content_ann']['bboxes']
                gt_labels = self.gt_res[key]['content_ann']['labels']
                cares = self.gt_res[key]['content_ann']['cares']
                for i in range(len(gt_labels)):
                    if self.gt_res[key]['content_ann']['labels'][i][0] == cls:
                        cur_gt_bboxes.append(gt_bboxes[i][:])
                        cur_cares.append(cares[i])
                                      
                total_num_gt += sum(cur_cares)
                for i, care in enumerate(cur_cares):
                    info = np.array(cur_gt_bboxes[i])
                    
                    if care:
                        cur_gt.append(info)
                        cur_cls_num_gt += 1
                    else:
                        cur_gt_ignore.append(info)
                
                cur_gt = np.array(cur_gt)
                cur_gt_ignore = np.array(cur_gt_ignore)

                # calculate tp&fp scores and pairs
                eval_res = tpfp_default(cur_infer, cur_gt, cur_gt_ignore, iou_thr=self.iou_thresh)

                tp, fp, scores, gt_covered = eval_res

                # tp mask
                tp_score_idxes = np.argwhere(tp)
                cur_tp_scores = scores[tp_score_idxes[:, 0], tp_score_idxes[:, 1]]
                # fp mask
                fp_score_idxes = np.argwhere(fp)
                cur_fp_scores = scores[fp_score_idxes[:, 0], fp_score_idxes[:, 1]]

                cur_match = []
                if cur_tp_scores.shape[0] != 0:
                    for gt_i, det_i in enumerate(np.squeeze(gt_covered, 0)):
                        if det_i != 0:
                            tmp = [list(map(int, cur_gt[gt_i])), list(map(float, cur_infer[det_i - 1]))]
                            cur_match.append(tmp[:])

                if cur_fp_scores.shape[0] != 0:
                    cur_fp = cur_infer[fp_score_idxes[:, 1]][:]
                else:
                    cur_fp = np.array([])
                        
                gt_non_covered_idxes = np.argwhere(np.logical_not(gt_covered))
                cur_non_covered_gt = cur_gt[gt_non_covered_idxes[:, 1]]
                missed_bbox_num += np.count_nonzero(np.logical_not(gt_covered))

                # save absolute missed GT
                if cur_non_covered_gt.shape[0] > 0:
                    if key not in non_covered_gt:
                        non_covered_gt[key] = dict()
                    non_covered_gt[key][cls] = [list(map(int, bbox)) for bbox in cur_non_covered_gt]

                # current class tp&fp scores
                cur_cls_tp_scores = np.concatenate((cur_cls_tp_scores, np.array(cur_tp_scores)))
                cur_cls_fp_scores = np.concatenate((cur_cls_fp_scores, np.array(cur_fp_scores)))

                # total class tp&fp scores
                tp_scores = np.concatenate((tp_scores, np.array(cur_tp_scores)))
                fp_scores = np.concatenate((fp_scores, np.array(cur_fp_scores)))

                # save match, miss and fas results
                if key not in total_info_dict:
                    total_info_dict[key] = dict()
                total_info_dict[key][cls] = dict()
                total_info_dict[key][cls]['match'] = cur_match[:]
                total_info_dict[key][cls]['miss'] = [list(map(int, bbox)) for bbox in cur_non_covered_gt]
                total_info_dict[key][cls]['fas'] = [list(map(float, item)) for item in cur_fp]

            logger.info('Calculating Recall@FA metric of class [{}] ...'.format(cls))
            if self.mode == 'thresh':
                res = get_recall_at_thresh(cur_cls_tp_scores, cur_cls_fp_scores, cur_cls_num_gt, self.frame_num, self.target_thresh)
            else:
                res = get_recall_at_fa(cur_cls_tp_scores, cur_cls_fp_scores, cur_cls_num_gt, self.frame_num, self.target_fa)
            total_res.append(deepcopy(res))
            
        logger.info('Calculating Recall@FA metric for all classes ...')
        if self.mode == 'thresh':
            res = get_recall_at_thresh(tp_scores, fp_scores, total_num_gt, self.frame_num, self.target_thresh)
        else:
            res = get_recall_at_fa(tp_scores, fp_scores, total_num_gt, self.frame_num, self.target_fa)

        total_res.append(deepcopy(res))

        return total_res, total_info_dict, non_covered_gt, missed_bbox_num

    def run(self):
        logger.info('Begin evaluating detection result')

        eval_path = os.path.join(self.out_path, 'eval.txt')
        miss_bbox_path = os.path.join(self.out_path, 'miss.json')
        total_info_path = os.path.join(self.out_path, 'total_info.json')

        if os.path.exists(eval_path):
            size = os.path.getsize(eval_path)
            if not self.overwrite and size != 0:
                logger.warning('{} exists'.format(eval_path))
                os.system('cat {}'.format(eval_path))
                return
            else:
                os.remove(eval_path)
                if os.path.exists(total_info_path):
                    os.remove(total_info_path)
                if os.path.exists(miss_bbox_path):
                    os.remove(miss_bbox_path)

        writer = open(eval_path, 'a')
        
        eval_res, total_info_dict, miss_bbox_dict, missed_gt_num = self.get_recall()

        logger.info('Got {} missed bboxes'.format(missed_gt_num))
        if missed_gt_num > 0:
            with open(miss_bbox_path, 'w', encoding='utf-8') as f:
                json.dump(miss_bbox_dict, f, ensure_ascii=False, indent=2)
        logger.info('Done {}'.format(miss_bbox_path))

        with open(total_info_path, 'w', encoding='utf-8') as f:
            json.dump(total_info_dict, f, ensure_ascii=False, indent=2)
        logger.info('Done {}'.format(total_info_path))

        PERF_1v1_TYPE = namedtuple('Perf_1v1', ['Recall', 'FA', 'Threshold'])
        for j in range(len(eval_res)):
            perf_1v1_list = list(map(
                    lambda x: PERF_1v1_TYPE._make([x[0], x[1], x[2]]),eval_res[j]))
            
            
            if j < len(self.total_classes):
                writer.write(f"=======================Class [{self.total_classes[j]}]==========================\n")
            else:
                writer.write(f"=======================Global result==========================\n")
            writer.write("FA\tRecall\tThreshold\n")

            for i in range(0, len(perf_1v1_list)):
                recall = str(format(perf_1v1_list[i].Recall*100, '.6f'))
                fa = str(format(perf_1v1_list[i].FA, '.6f'))
                thresh = str(format(float(perf_1v1_list[i].Threshold), '.6f'))
                writer.write('{}\t{}\t{}\n'.format(fa, recall, thresh))
            writer.write('\n')

        writer.close()

        os.system('cat {}'.format(eval_path))
        logger.info('Done {}'.format(eval_path))
