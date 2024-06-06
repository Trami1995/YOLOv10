# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union, Optional

import mmengine
from mmdet.datasets import BaseDetDataset
from mmengine.fileio import join_path, list_from_file, load

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS


@DATASETS.register_module()
class DavarCustomDataset(BatchShapePolicyDataset, BaseDetDataset):
    """ Implementation of the common dataset of davar group, which supports tasks of
        Object Detection, Classification, Semantic Segmentation, OCR, etc. Properties in 'content_ann' can be chosen
        according to different tasks.

       train_datalist.json:                                                        # file name
        {
            "###": "Comment",                                                      # The meta comment
            "Images/train/img1.jpg": {                                             # Relative path of images
                "height": 534,                                                     # Image height
                "width": 616,                                                      # Image width
                "frame_id": 1,                                                     # for tracking
                "content_ann": {                                                   # Following lists have same lengths.
                    "bboxes": [[161, 48, 563, 195, 552, 225, 150, 79],             # Bounding boxes in shape of [2 * N]
                                [177, 178, 247, 203, 240, 224, 169, 198],          # where N >= 2. N=2 means the
                                [263, 189, 477, 267, 467, 296, 252, 218],          # axis-alignedrect bounding box
                                [167, 211, 239, 238, 232, 256, 160, 230],
                                [249, 227, 389, 278, 379, 305, 239, 254],
                                [209, 280, 382, 343, 366, 384, 194, 321]],
                    "ignore_area": [[161, 48, 563, 195, 552, 225, 150, 79], ...]
                    "cbboxes": [ [[...],[...]], [[...],[...],[...]],               # Character-wised bounding boxes
                    "cares": [1, 1, 1, 1, 1, 0],                                   # If the bboxes will be cared
                    "labels": [['title'], ['code'], ['num'], ['value'], ['other]], # Labels for classification/detection
                                                                                   # task, can be int or string.
                    "texts": ['apple', 'banana', '11', '234', '###'],              # Transcriptions for text recognition
                }
                "content_ann2":{                                                   # Second-level annotations
                    "labels": [[1],[2],[1]]
                }
                "answer_ann":{                                                  # Structure information k-v annotations
                    "keys": ["title", "code", "num","value"],                   # used in end-to-end evaluation
                    "value": [["apple"],["banana"],["11"],["234"]]
                }
            },
            ....
        }
    """

    METAINFO = {
        'classes':
        ("vehicle", "pedestrian", "cyclist", "cyclist_person", "another_vehicle"),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228)]
    }


    def __init__(self,
                 *args,
                 seg_map_suffix: str = '.jpg',
                 proposal_file: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 return_classes: bool = False,
                 classes_config=None,
                 ignore_bbox_threshold=0,
                 small_bbox_as_ignore_area=True,
                 **kwargs):
        self.seg_map_suffix = seg_map_suffix
        self.proposal_file = proposal_file
        self.backend_args = backend_args
        self.return_classes = return_classes
        self.ignore_bbox_threshold = ignore_bbox_threshold
        self.small_bbox_as_ignore_area = small_bbox_as_ignore_area

        if classes_config is not None:
            self.classes_config = mmengine.load(classes_config)
        else:
            self.classes_config = None

        self.cat_ids = self.classes_config['classes']
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        super().__init__(*args, **kwargs)


    def _cvt_list(self, img_info):
        """ Convert JSON dict into a list.

        Args:
            img_info(dict): annotation information in a json obj

        Returns:
            list(dict): converted list of annotations, in form of
                       [{"filename": "xxx", width: 120, height: 320, ann: {}, ann2: {}},...]
        """

        result_dict = []

        # Remove the meta comment in json
        if "###" in img_info.keys():
            del img_info["###"]

        for key in img_info.keys():
            data_info = dict()
            data_info["img_path"] = key
            data_info["height"] = img_info[key]["height"]
            data_info["width"] = img_info[key]["width"]
            anns = img_info[key]["content_ann"]
            data_info["ignore_area"] = anns.get("ignore_area", list())
            
            assert "bboxes" in anns

            # Filter bboxes which get more than 50% area in ignore areas
            if len(data_info["ignore_area"]) > 0:
                anns = self._ignore_area_filter(data_info["ignore_area"], anns)
            instances = []

            for i, bbox in enumerate(anns["bboxes"]):
                instance = {}

                x_min = min(bbox[0::2])
                x_max = max(bbox[0::2])
                y_min = min(bbox[1::2])
                y_max = max(bbox[1::2])
                w = x_max - x_min
                h = y_max - y_min

                inter_w = max(0, min(x_min + w, img_info[key]['width']) - max(x_min, 0))
                inter_h = max(0, min(y_min + h, img_info[key]['height']) - max(y_min, 0))
                if inter_w * inter_h == 0:
                    continue
                # if ann['area'] <= 0 or w < 1 or h < 1:
                #     continue
                if anns['labels'][i][0] not in self.cat_ids:
                    continue
                bbox = [x_min, y_min, x_max, y_max]

                if anns['cares'][i] == 0:
                    instance['ignore_flag'] = 1
                else:
                    instance['ignore_flag'] = 0
                instance['bbox'] = bbox
                instance['bbox_label'] = self.cat2label[anns['labels'][i][0]]

                instances.append(instance)

            data_info['instances'] = instances

            # data_info = self._data_info_postprocess(data_info)

            result_dict.append(data_info)

        return result_dict
    

    def _ignore_area_filter(self, ignore_area, anns):
        import copy

        anns_new = copy.deepcopy(anns)
        for i, bbox in enumerate(anns["bboxes"]):
            if anns['cares'][i] == 0: continue
            x_min = min(bbox[0::2])
            x_max = max(bbox[0::2])
            y_min = min(bbox[1::2])
            y_max = max(bbox[1::2])
            bbox_area = max(0, x_max - x_min) * max(0, y_max - y_min)
            if bbox_area < 1:  
                anns_new['cares'][i] = 0
                continue

            for ign_area in ignore_area:
                ign_x_min, ign_x_max = min(ign_area[0::2]), max(ign_area[0::2])
                ign_y_min, ign_y_max = min(ign_area[1::2]), max(ign_area[1::2])

                inter_x1 = max(x_min, ign_x_min)
                inter_y1 = max(y_min, ign_y_min)
                inter_x2 = min(x_max, ign_x_max)
                inter_y2 = min(y_max, ign_y_max)

                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                if inter_area / bbox_area > 0.5:
                    anns_new['cares'][i] = 0
        return anns_new


    #TODO: not useful currently, might be deleted later
    def _data_info_postprocess(self, img_info):
        # 过滤过小的 bbox
        ann = img_info['ann']
        img_height = img_info['height']
        img_width = img_info['width']

        bboxes_list = list()
        care_list = list()
        label_list = list()
        bboxes_ignore_list = list()
        ori_ignore_area = ann.get("ignore_area", list())
        assert "bboxes" in ann
        for i, bbox in enumerate(ann["bboxes"]):
            x_min = min(bbox[0::2])
            x_max = max(bbox[0::2])
            y_min = min(bbox[1::2])
            y_max = max(bbox[1::2])
            if (x_max - x_min) * (y_max - y_min) / img_height / img_width < self.ignore_bbox_threshold:
                bboxes_ignore_list.append(bbox)
            elif len(ori_ignore_area) > 0:
                flag = False
                for ign_area in ori_ignore_area:
                    ign_x_min, ign_x_max = min(ign_area[0::2]), max(ign_area[0::2])
                    ign_y_min, ign_y_max = min(ign_area[1::2]), max(ign_area[1::2])
                    # Union
                    if ign_x_min <= x_min and ign_y_min <= y_min and ign_x_max >= x_max and ign_y_max >= y_max:
                        flag = True
                        break

                if flag:
                    bboxes_ignore_list.append(bbox)
                else:
                    bboxes_list.append(bbox)
                    care_list.append(ann["cares"][i])
                    label_list.append(ann["labels"][i])
            else:
                bboxes_list.append(bbox)
                care_list.append(ann["cares"][i])
                label_list.append(ann["labels"][i])
        ann['bboxes'] = bboxes_list
        ann['cares'] = care_list
        ann['labels'] = label_list

        # 过小的 bbox 会直接归入 ignore area
        if self.small_bbox_as_ignore_area:
            for bbox in ann.get("ignore_area", list()):
                    bboxes_ignore_list.append(bbox)
            ann['ignore_area'] = bboxes_ignore_list
        img_info['ann'] = ann
        return img_info
    

    def full_init(self) -> None:
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - load_proposals: Load proposals from proposal file, if
              `self.proposal_file` is not None.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
            ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        # get proposals from file
        if self.proposal_file is not None:
            self.load_proposals()
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()

        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True



    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  
        # noqa: E501
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')

        # load and parse data_infos.
        data_list = self._cvt_list(annotations)

        return data_list
    

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        valid_data_infos = []
        for i, img_info in enumerate(self.data_list):
            # Not support gif
            if img_info['img_path'].split(".")[-1].upper() == 'GIF':
                continue

            instances = img_info['instances']

            if len(instances) == 0: continue
            filter_flag = True
            for j, instance in enumerate(instances):
                # Filter images with empty ground-truth
                if instance is not None and filter_empty_gt:
                    if ('bboxes' in instance and len(instance['bboxes']) == 0) or \
                        ('ignore_flag' in instance and instance['ignore_flag'] == 1):
                        continue

                    filter_flag = False

            if filter_flag and min(img_info['width'], img_info['height']) >= min_size:
                valid_data_infos.append(img_info)
            
        return valid_data_infos
    
