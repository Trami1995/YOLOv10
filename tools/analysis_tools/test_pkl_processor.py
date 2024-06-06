"""
将模型输出转换成固定格式的 JSON 文件
"""

import os
import cv2
import json
import pickle
import argparse
import collections
import multiprocessing

import numpy as np

from functools import partial
from PIL import Image
from tqdm import tqdm


class DefaultResultReader:
    """
    模型输出格式已经经过了 format
    """

    @classmethod
    def load(cls, pred_pkl_path, **kwargs):
        # 读取 detections 结果
        with open(pred_pkl_path, "br") as f:
            return pickle.load(f)

class CustomDatasetResultReader:
    """
    模型输出格式已经经过了 format
    """

    @classmethod
    def load(cls, pred_pkl_path, **kwargs):
        # 读取 detections 结果
        with open(pred_pkl_path, "r") as f:
            return json.load(f)


class DetectionResultReader:
    """
    模型输出格式：
    [
        # instance one
        [
            [(N, 5)], # class one, N 是 bbox detection 数量, 分别是 xmin, ymin, xmax, ymax, score
            ...
        ], ...
    ]
    """

    @classmethod
    def load(cls, data_json_path, pred_pkl_path, **kwargs):
        assert data_json_path is not None

        # 读取 detections 结果
        with open(pred_pkl_path, "br") as f:
            detections = pickle.load(f)

        # 读取数据集信息
        with open(data_json_path, 'r', encoding='utf-8') as load_f:
            ann = json.load(load_f, object_pairs_hook=collections.OrderedDict)

        result = dict()
        for (data_key, data_ann), data_detection in zip(ann.items(), detections):
            result[data_key] = dict()
            result[data_key]["height"] = data_ann["height"]
            result[data_key]["width"] = data_ann["width"]

            result[data_key]["content_ann"] = dict()
            result[data_key]["content_ann"]["bboxes"] = list()
            result[data_key]["content_ann"]["labels"] = list()
            result[data_key]["content_ann"]["scores"] = list()

            for label_idx, label_ann in enumerate(data_detection):
                bbox_size = label_ann.shape[0]
                result[data_key]["content_ann"]["bboxes"].extend(label_ann[:, :-1].tolist())
                result[data_key]["content_ann"]["labels"].extend([[label_idx] for _  in range(bbox_size)])
                result[data_key]["content_ann"]["scores"].extend(label_ann[:, -1:].tolist())

        return result


class TrackingResultReader:
    """
    模型输出格式：
    [
        {
            "bboxes": torch.tensor([[xmin, ymin, xmax, ymax], ...]),
            "labels": torch.tensor([label_1, label_2, ...]),              # label_1, label_2 是数字, 从 0 开始
            "scores": torch.tensor([score_1, score_2, ...]),
            "instances_id": torch.tensor([instance_1, instance_2, ...]),
        },
        ...
    ]
    """

    @classmethod
    def load(cls, data_json_path, pred_pkl_path, **kwargs):
        assert data_json_path is not None

        # 读取 detections 结果
        with open(pred_pkl_path, "br") as f:
            detections = pickle.load(f)

        # 读取数据集信息
        with open(data_json_path, 'r', encoding='utf-8') as load_f:
            ann = json.load(load_f, object_pairs_hook=collections.OrderedDict)

        # 这里 detection 是已经经过排序的，我们将 ann 也进行排序
        sorted_ann = [(data_key, data_ann) for data_key, data_ann in ann.items()]
        sorted_ann.sort(key=lambda x: x[1].get("frame_id", -1))

        result = dict()
        for (data_key, data_ann), data_detection in zip(sorted_ann, detections):
            result[data_key] = dict()
            result[data_key]["height"] = data_ann["height"]
            result[data_key]["width"] = data_ann["width"]
            result[data_key]["frame_id"] = data_ann.get("frame_id", -1)

            bboxes = data_detection["bboxes"].detach().cpu().numpy().tolist()
            labels = data_detection["labels"].detach().cpu().numpy().tolist()
            scores = data_detection["scores"].detach().cpu().numpy().tolist()
            instances_id = data_detection["instances_id"].detach().cpu().numpy().tolist()

            result[data_key]["content_ann"] = dict()
            result[data_key]["content_ann"]["bboxes"] = bboxes
            result[data_key]["content_ann"]["labels"] = [[l] for l in labels]
            result[data_key]["content_ann"]["scores"] = [[s] for s in scores]
            result[data_key]["content_ann"]["track_ids"] = instances_id

        return result


class CustomDatasetWriter:
    """
    JSON 格式：
    custom dataset format:
    {
        "image_name": {
            "height": 534,
            "width": 616,
            "content_ann": {
                "bboxes": [[xmin, ymin, xmax, ymax], ...],
                "labels": [[label_1], [label_2], ...],      # label_1, label_2 是数字, 从 0 开始
                "scores": [[score_1], [score_2], ...],
                "track_ids": [track_1, track_2, ...]         # optional, 只有 tracking 才有
            },
        },
        ...
    }
    """

    @classmethod
    def dump(cls, custom_dataset_data, output_path, **kwargs):
        pred_data = dict()
        for dd in custom_dataset_data:
            img_path = dd['img_path']
            height, width = dd['ori_shape']
            pred_bboxes = dd['pred_instances']['bboxes'].tolist()
            pred_labels = [[int(lb)] for lb in dd['pred_instances']['labels']]
            pred_scores = [[float(sc)] for sc in dd['pred_instances']['scores']]

            pred_data[img_path] = dict()
            pred_data[img_path]['width'] = width
            pred_data[img_path]['height'] = height
            pred_data[img_path]['content_ann'] = dict()
            pred_data[img_path]['content_ann']['bboxes'] = pred_bboxes
            pred_data[img_path]['content_ann']['labels'] = pred_labels
            pred_data[img_path]['content_ann']['scores'] = pred_scores

        with open(os.path.join(output_path, "custom_dataset.json"), "w") as f:
            json.dump(pred_data, f, indent=2)


class TrackingWriter:
    """
    JSON 格式：
    tracking format:
    {
        "track_id": {
            "height": height,
            "width": width,
            "content_ann": {
                "image_name": [image_name_1, ...],
                "bboxes": [[xmin, ymin, xmax, ymax], ...],
                "labels": [label_1, ...],
                "scores": [score_1, ...],
            }
        }
    }
    """

    @classmethod
    def dump(cls, custom_dataset_data, output_path, **kwargs):
        assert "track_ids" in list(custom_dataset_data.values())[0]["content_ann"]

        # custom dataset 转换为 tracking 格式
        tracking_data = dict()
        for image_name, image_data in custom_dataset_data.items():
            height = image_data["height"]
            width = image_data["width"]
            if "frame_id" in image_data:
                frame_id = image_data["frame_id"]
            else:
                frame_id = int(image_name.split(".")[0].split("_")[-1])

            image_ann = image_data["content_ann"]
            for bbox, label, score, track_id in zip(image_ann["bboxes"], image_ann["labels"], image_ann["scores"], image_ann["track_ids"]):
                if track_id not in tracking_data:
                    tracking_data[track_id] = dict()
                    tracking_data[track_id]["height"] = height
                    tracking_data[track_id]["width"] = width

                    tracking_data[track_id]["content_ann"] = dict()
                    tracking_data[track_id]["content_ann"]["image_name"] = list()
                    tracking_data[track_id]["content_ann"]["bboxes"] = list()
                    tracking_data[track_id]["content_ann"]["labels"] = list()
                    tracking_data[track_id]["content_ann"]["scores"] = list()

                tracking_data[track_id]["content_ann"]["image_name"].append(image_name)
                tracking_data[track_id]["content_ann"]["bboxes"].append(bbox)
                tracking_data[track_id]["content_ann"]["labels"].append(label)
                tracking_data[track_id]["content_ann"]["scores"].append(score)

        with open(os.path.join(output_path, "tracking.json"), "w") as f:
            json.dump(tracking_data, f, indent=4)


class VideoWriter:
    """
    将 tracking 结果保存到 video 中
    """

    colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

    @classmethod
    def dump(cls, data_image_path, custom_dataset_data, output_path, fps=6.0, **kwargs):
        assert "track_ids" in list(custom_dataset_data.values())[0]["content_ann"]

        if "filename" in list(custom_dataset_data.values())[0]:
            has_abs_path = True
        else:
            has_abs_path = False
            assert data_image_path is not None

        # 初始化视屏流
        if has_abs_path:
            sample_image_path = list(custom_dataset_data.values())[0]["filename"]
        else:
            sample_image_path = os.path.join(data_image_path, list(custom_dataset_data.keys())[0])
        sample_image = Image.open(sample_image_path)
        vw = sample_image.size[0]
        vh = sample_image.size[1]
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        outvideo = cv2.VideoWriter(os.path.join(output_path, "bbox.avi"), fourcc, fps, (vw, vh), True)

        tasks = list(custom_dataset_data.items())
        tasks.sort(
            key=lambda x: x[1]["frame_id"] if "frame_id" in x[1] else int(x[0].split(".")[0].split("_")[-1])
        )

        pool = multiprocessing.Pool(48)
        for img in tqdm(
            pool.imap(
                partial(
                    cls._dump_helper,
                    data_image_path = data_image_path,
                    has_abs_path = has_abs_path,
                ),
                tasks,
            ),
            total=len(tasks)
        ):
            outvideo.write(img)
        pool.close()
        pool.join()

        cv2.destroyAllWindows()
        outvideo.release()

    @classmethod
    def _dump_helper(cls, data, data_image_path, has_abs_path):
        image_name, image_data = data

        if has_abs_path:
            image_path = image_data["filename"]
        else:
            image_path = os.path.join(data_image_path, image_name)
        image = Image.open(image_path)
        img = np.array(image)

        bboxes = image_data["content_ann"]["bboxes"]
        scores = image_data["content_ann"]["scores"]
        track_ids = image_data["content_ann"]["track_ids"]
        if "land_bboxes" in image_data["content_ann"]:
            land_bboxes = image_data["content_ann"]["land_bboxes"]
        else:
            land_bboxes = [None for _ in bboxes]

        for bbox, score, track_id, land_bbox in zip(bboxes, scores, track_ids, land_bboxes):
            x1, y1, x2, y2 = bbox
            color = cls.colors[int(track_id) % len(cls.colors)]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 2)
            if land_bbox is not None:
                cv2.polylines(img, [np.array([[int(p[0]), int(p[1])] for p in land_bbox])], True, (255,0, 0), 2)
            cv2.putText(
                img,
                "vehicle" + "-" + str(int(track_id)) + "-{:.4f}".format(score[0]),
                (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1,
            )

        return img


def parse_args():
    parser = argparse.ArgumentParser(description="Convert to predefined Json format")
    parser.add_argument("-i", "--data_image_path", help="input image root path")
    parser.add_argument("-d", "--data_json_path", help="input json path")
    parser.add_argument("-p", "--pred_pkl_path", help="input pkl path")
    parser.add_argument("-o", "--output_path", help="outout path")
    parser.add_argument(
        "-r", "--reader",
        choices=["default", "custom", "det", "track"], default="default",
        help="reader type: default, det, track"
    )
    parser.add_argument(
        "-w", "--writer",
        choices=["custom", "track", "video"], nargs='+', default=["custom"],
        help="writer type: custom, track, video"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.data_image_path is not None and not os.path.exists(args.data_image_path):
        print("data image path" + args.data_image_path + " not exist")
        return

    if args.data_json_path is not None and not os.path.exists(args.data_json_path):
        print("data json path" + args.data_json_path + " not exist")
        return

    if not os.path.exists(args.pred_pkl_path):
        print("pred json path" + args.pred_pkl_path + " not exist")
        return

    os.makedirs(args.output_path, exist_ok=True)

    if args.reader == "default":
        reader = DefaultResultReader
    elif args.reader == "det":
        reader = DetectionResultReader
    elif args.reader == "track":
        reader = TrackingResultReader
    elif args.reader == "custom":
        reader = CustomDatasetResultReader
    else:
        raise NotImplementedError

    custom_dataset_data = reader.load(
        data_json_path = args.data_json_path,
        pred_pkl_path = args.pred_pkl_path,
    )

    for writer_type in args.writer:
        if  writer_type== "custom":
            writer = CustomDatasetWriter
        elif writer_type == "track":
            writer = TrackingWriter
        elif writer_type == "video":
            writer = VideoWriter
        else:
            raise NotImplementedError

        writer.dump(
            data_image_path = args.data_image_path,
            custom_dataset_data = custom_dataset_data,
            output_path = args.output_path,
        )


if __name__ == '__main__':
    main()
