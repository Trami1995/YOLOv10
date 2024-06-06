# Copyright (c) OpenMMLab. All rights reserved.
import copy
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict, namedtuple
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import mmengine
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from mmdet.structures.mask import encode_mask_results
from mmdet.evaluation.functional import eval_recalls, eval_map

from mmyolo.registry import METRICS

logger: MMLogger = MMLogger.get_current_instance()

@METRICS.register_module()
class DavarCustomMetric(BaseMetric):
    """Evaluation metric for DavarCustomMetric.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', 'proposal', and 'proposal_fast'.
            Defaults to 'bbox'.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (100, 300, 1000).
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        sort_categories (bool): Whether sort categories in annotations. Only
            used for `Objects365V1Dataset`. Defaults to False.
        use_mp_eval (bool): Whether to use mul-processing evaluation
    """
    default_prefix: Optional[str] = 'custom'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'mAP',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False,
                 use_mp_eval: bool = False,
                 classes_config: str = None,
                 targets: List = [],
                 ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # evaluation metrics, currently only support one metric
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mAP', 'recall', 'fa', 'threshold']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'mAP', 'recall', 'fa', 'threshold', "
                    f"but got {metric}.")

        if self.metrics[0] not in ['fa', 'threshold'] and len(targets) > 0:
            logger.warning(f'targets is not useful under {self.metrics[0]} evaluation')
            self.targets = None
        else:
            self.targets = targets

        # do class wise evaluation, default False
        self.classwise = classwise
        # whether to use multi processing evaluation, default False
        self.use_mp_eval = use_mp_eval

        # proposal_nums used to compute recall or precision.
        self.proposal_nums = list(proposal_nums)

        # iou_thrs used to compute recall or precision.
        if iou_thrs is None:
            iou_thrs = [0.5, 0.75, 0.95]
        self.iou_thrs = iou_thrs
        self.metric_items = metric_items
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix

        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )

        # handle dataset lazy init
        self.cat_ids = None
        self.img_ids = None

        if classes_config is not None:
            self.classes_config = mmengine.load(classes_config)
        else:
            self.classes_config = None
        self.cat_ids = self.classes_config['classes']
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}


    def _bbox_overlaps(self,
                    bboxes1,
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


    def _tpfp_default(self,
                    det_bboxes,
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


        ious = self._bbox_overlaps(
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
    
    def _get_cls_results(self, det_results, annotations, class_id):
        """Get det results and gt information of a certain class.

        Args:
            det_results (list[list]): Same as `eval_map()`.
            annotations (list[dict]): Same as `eval_map()`.
            class_id (int): ID of a specific class.

        Returns:
            tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
        """
        cls_dets = [img_res[class_id] for img_res in det_results]
        cls_gts = []
        cls_gts_ignore = []
        for ann in annotations:
            gt_inds = ann['labels'] == class_id
            cls_gts.append(ann['bboxes'][gt_inds, :])

            if ann.get('labels_ignore', None) is not None:
                ignore_inds = ann['labels_ignore'] == class_id
                cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
            else:
                cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))

        return cls_dets, cls_gts, cls_gts_ignore
    
    def _get_recall_at_fa(self, pos_scores, neg_scores, total_num_gt, frame_num, target_fa=None):
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
            target_fa = copy.deepcopy(target_fa)
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
    
    def _get_recall_at_thresh(self, pos_scores, neg_scores, total_num_gt, frame_num, target_thresh=None):
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
    
    def _get_recall(self, annotations, det_results, iou_thr=0.5, mode='fa', target_fa=None, target_thresh=None):
        # variables for globle info
        tp_scores = np.array([])
        fp_scores = np.array([])
        total_res = []
        
        total_num_gt = 0

        assert len(det_results) == len(annotations)

        num_imgs = len(det_results)

        for idx_cls, cls in enumerate(self.cat_ids):
            # variables for each class
            cur_cls_num_gt = 0
            cur_cls_tp_scores = np.array([])
            cur_cls_fp_scores = np.array([])

            cls_dets, cls_gts, cls_gts_ignore = self._get_cls_results(det_results, annotations, idx_cls)

            for det, gt, gt_ign in zip(cls_dets, cls_gts, cls_gts_ignore):
                eval_res = self._tpfp_default(det, gt, gt_ign, iou_thr=iou_thr)
                cur_cls_num_gt += gt.shape[0]
                tp, fp, scores, gt_covered = eval_res

                # tp mask
                tp_score_idxes = np.argwhere(tp)
                cur_tp_scores = scores[tp_score_idxes[:, 0], tp_score_idxes[:, 1]]
                # fp mask
                fp_score_idxes = np.argwhere(fp)
                cur_fp_scores = scores[fp_score_idxes[:, 0], fp_score_idxes[:, 1]]

                # current class tp&fp scores
                cur_cls_tp_scores = np.concatenate((cur_cls_tp_scores, np.array(cur_tp_scores)))
                cur_cls_fp_scores = np.concatenate((cur_cls_fp_scores, np.array(cur_fp_scores)))

                # total class tp&fp scores
                tp_scores = np.concatenate((tp_scores, np.array(cur_tp_scores)))
                fp_scores = np.concatenate((fp_scores, np.array(cur_fp_scores)))

            logger.info('Calculating Recall@FA metric of class [{}] ...'.format(cls))
            if mode == 'threshold':
                res = self._get_recall_at_thresh(cur_cls_tp_scores, cur_cls_fp_scores, cur_cls_num_gt, num_imgs, target_thresh)
            else:
                res = self._get_recall_at_fa(cur_cls_tp_scores, cur_cls_fp_scores, cur_cls_num_gt, num_imgs, target_fa)
            total_res.append(copy.deepcopy(res))
            total_num_gt += cur_cls_num_gt
            
        logger.info('Calculating Recall@FA metric for all classes ...')
        if mode == 'threshold':
            res = self._get_recall_at_thresh(tp_scores, fp_scores, total_num_gt, num_imgs, target_thresh)
        else:
            res = self._get_recall_at_fa(tp_scores, fp_scores, total_num_gt, num_imgs, target_fa)


        total_res.append(copy.deepcopy(res))

        return total_res


    def _custom_print(self, eval_res):
        PERF_1v1_TYPE = namedtuple('Perf_1v1', ['Recall', 'FA', 'Threshold'])
        for j in range(len(eval_res)):
            perf_1v1_list = list(map(
                    lambda x: PERF_1v1_TYPE._make([x[0], x[1], x[2]]),eval_res[j]))
            if j < len(self.cat_ids):
                print_log(f"=======================Class [{self.cat_ids[j]}]==========================")
            else:
                print_log(f"=======================Global result==========================")
            print_log("FA\tRecall\tThreshold")

            for i in range(0, len(perf_1v1_list)):
                recall = str(format(perf_1v1_list[i].Recall*100, '.6f'))
                fa = str(format(perf_1v1_list[i].FA, '.6f'))
                thresh = str(format(float(perf_1v1_list[i].Threshold), '.6f'))
                print_log('{}\t{}\t{}'.format(fa, recall, thresh))


    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']

            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            # gt['img_id'] = data_sample['img_id']

            # TODO: Need to refactor to support LoadAnnotations
            assert 'gt_instances' in data_sample and 'ignored_instances' in data_sample, \
                'ground truth is required for evaluation when ' \
                '`ann_file` is not provided'
            gt['gt_anns'] = data_sample['gt_instances']
            gt['ignored_anns'] = data_sample['ignored_instances']
            gt['img_path'] = data_sample['img_path']
            # add converted result to the results list
            self.results.append((gt, result))


    def results2json(self, results: Sequence[dict], annotations: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        assert len(results) == len(annotations)
        bbox_json_results = dict()
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_path = annotations[idx]['img_path']
            bbox_json_results[image_path] = dict()
            bbox_json_results[image_path]['width'] = annotations[idx]['width']
            bbox_json_results[image_path]['height'] = annotations[idx]['height']
            bbox_json_results[image_path]['content_ann'] = dict()
            bbox_json_results[image_path]['content_ann']['bboxes'] = []
            bbox_json_results[image_path]['content_ann']['labels'] = []
            bbox_json_results[image_path]['content_ann']['scores'] = []

            for label_idx, pred_infos in enumerate(result):
                for pred_info in pred_infos:
                    bbox_json_results[image_path]['content_ann']['bboxes'].append(pred_info[:4])
                    bbox_json_results[image_path]['content_ann']['labels'].append([label_idx])
                    bbox_json_results[image_path]['content_ann']['scores'].append([pred_info[4]])

            if segm_json_results is None:
                continue

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.json'
        dump(bbox_json_results, result_files['bbox'])

        return result_files


    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)
        assert len(gts) == len(preds)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        num_classes = len(self.classes_config['classes'])
        tmp_results = []
        for res in preds:
            points = np.array(res['bboxes']).reshape(-1, 4)
            scores = np.array(res['scores']).reshape(-1, 1)
            labels = np.array(res['labels'])
            bboxes = np.concatenate([points, scores], axis=-1)
            tmp_results.append([bboxes[labels == i, :] for i in range(num_classes)])
        preds_results = tmp_results

        annotations = []
        for res in gts:
            img_info = dict()
            gt_res = res['gt_anns']
            ignored_res = res['ignored_anns']
            gt_points = np.array(gt_res['bboxes']).reshape(-1, 4)
            gt_labels = np.array(gt_res['labels'])
            ignored_points = np.array(ignored_res['bboxes']).reshape(-1, 4)
            ignored_labels = np.array(ignored_res['labels'])

            img_info['img_path'] = res['img_path']
            img_info['width'] = res['width']
            img_info['height'] = res['height']
            img_info['bboxes'] = gt_points
            img_info['bboxes_ignore'] = ignored_points
            img_info['labels'] = gt_labels
            img_info['labels_ignore'] = ignored_labels

            annotations.append(copy.deepcopy(img_info))
            
        eval_results = OrderedDict()
        if self.format_only:
            result_files = self.results2json(preds_results, annotations, outfile_prefix)
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        iou_thrs = [self.iou_thrs] if isinstance(self.iou_thrs, float) else self.iou_thrs

        if self.metrics[0] == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for tmp_iou_thre in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {tmp_iou_thre}{"-" * 15}')
                mean_ap, _ = eval_map(
                    preds_results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=tmp_iou_thre,
                    logger=logger)
                mean_aps.append(mean_ap)
                # eval_results[f'AP{int(tmp_iou_thre * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        #BUG: have bugs while evaluating
        elif self.metrics[0] == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, preds_results, self.proposal_nums, iou_thrs, logger=logger)
            for i, num in enumerate(self.proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                arr = recalls.mean(axis=1)
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = arr[i]
        #TODO: need to support target_fa and target_thresh
        elif self.metrics[0] == 'fa' or self.metrics[0] == 'threshold':
            iou_thr = 0.5
            if self.metrics[0] == 'fa':
                recall_at_fas = self._get_recall(annotations, preds_results, iou_thr=iou_thr, mode=self.metrics[0], target_fa=self.targets)
            else:
                recall_at_fas = self._get_recall(annotations, preds_results, iou_thr=iou_thr, mode=self.metrics[0], target_thresh=self.targets)
            
            self._custom_print(recall_at_fas)
        else:
            error_message = f"[{self.metrics[0]}] is not a valid metric..."
            logger.error(error_message)
            raise ValueError(error_message)
        return eval_results
