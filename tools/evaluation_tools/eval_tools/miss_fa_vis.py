import os
import random
import json
import argparse
from PIL import Image, ImageDraw, ImageFont


def plot_bbox(draw, box_tmp, color):
    point_colors = ["yellow", "blue", "purple", "orange"]

    for i in range(len(box_tmp)):
        draw.line([box_tmp[i], box_tmp[(i+1)%len(box_tmp)], box_tmp[(i+2)%len(box_tmp)], box_tmp[(i+3)%len(box_tmp)]], fill=color, width=5)
        draw.ellipse((box_tmp[i][0]-6, box_tmp[i][1]-6, box_tmp[i][0]+6, box_tmp[i][1]+6), fill=point_colors[i], outline=point_colors[i])


def main(
    json_file_path,
    output_folder="output_json_images",
    debug_num=100,
    debug_dir=None,
    pre="/",
    score_thres=0.5,
    selected_image_name=None,
    gt_json_path=None,
    spot_json_path=None,
    debug_class=None
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取json文件
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    # 读取 GT 文件
    gts = None
    if gt_json_path is not None:
        with open(gt_json_path, "r") as f:
            gts = json.load(f)

    # 读取 spot 文件
    spots_info = None
    if spot_json_path is not None:
        with open(spot_json_path, "r") as f:
            spots = json.load(f)

        spots_info = dict()
        for spot_id, spot_info in spots.items():
            spots_info[spot_id] = list()
            for poly in spot_info["park_number"].values():
                spots_info[spot_id].append([[p["x"], p["y"]] for p in poly])

    ignore_area_color = (0, 0, 255)  # 蓝色
    gt_color = (0, 0, 255)  # 蓝色
    bbox_color = (0, 255, 0)  # 绿色
    landpoint_color = (255, 0, 0)  # 红色
    spot_color = (255, 255, 0) # 黄色
    small_threshold = 0.001
    font = ImageFont.truetype("arialbi.ttf", 20)#设置字体

    # 图片挑选逻辑
    debug_json_data = list()
    if selected_image_name is not None:
        debug_json_data = [(image_name, json_data[image_name]) for image_name in selected_image_name]
    elif debug_dir is not None:
        debug_image_name = set()
        for image_name in os.listdir(debug_dir):
            debug_image_name.add(image_name)

        for img_path, img_info in json_data.items():
            img_name = os.path.basename(img_path)
            if img_name in debug_image_name:
                debug_json_data.append((img_path, img_info))
    elif len(json_data) > debug_num:
        debug_json_data = list(json_data.items())
        debug_json_data.sort(key = lambda x: x[0])
        # random.seed(1)
        debug_json_data = random.sample(debug_json_data, debug_num)
    else:
        debug_json_data = list(json_data.items())

    for img_path, img_info in debug_json_data:
        img_tmp = img_path
        img_full_path = os.path.join(pre, img_tmp)
        if os.path.exists(img_full_path):
            img = Image.open(img_full_path)
            width, height = img.size
            draw = ImageDraw.Draw(img)
            flag = False

            for cls_idx, cur_cls in enumerate(debug_class):
                assert cur_cls in img_info
                cur_info = img_info[cur_cls]
                cur_thrsh = score_thres[cls_idx]
                if cur_thrsh == 0: continue
                
                if "match" in cur_info and len(cur_info["match"]) > 0:
                    for match in cur_info["match"]:
                        # if match[1][4] < score_thres and match[1][3] > height/5:
                        if match[1][4] < cur_thrsh:
                            flag = True
                            x_min = min(match[1][0:4:2])
                            x_max = max(match[1][0:4:2])
                            y_min = min(match[1][1:4:2])
                            y_max = max(match[1][1:4:2])
                            box_tmp = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                            plot_bbox(draw, box_tmp, bbox_color)

                if "miss" in cur_info and len(cur_info["miss"]) > 0:
                    
                    for bbox_idx, box in enumerate(cur_info["miss"]):
                        # if box[3] > height/5:
                        flag = True
                        x_min = min(box[0::2])
                        x_max = max(box[0::2])
                        y_min = min(box[1::2])
                        y_max = max(box[1::2])
                        box_tmp = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                        plot_bbox(draw, box_tmp, gt_color)

                if "fas" in cur_info and len(cur_info["fas"]) > 0:
                    for box in cur_info["fas"]:
                        # if box[4] < score_thres or box[3] <= height/5:
                        if box[4] < cur_thrsh:
                            continue
                        flag = True
                        x_min = min(box[0], box[2])
                        x_max = max(box[0], box[2])
                        y_min = min(box[1], box[3])
                        y_max = max(box[1], box[3])
                        box_tmp = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                        plot_bbox(draw, box_tmp, landpoint_color)
                        draw.text((x_min, y_min), cur_cls + str(box[4]), 'fuchsia', font)

                if "content_ann" in cur_info and "crop_bboxes" in cur_info["content_ann"]:
                    crop_bboxes = cur_info["content_ann"]["crop_bboxes"]
                    land_bboxes = cur_info["content_ann"]["land_bboxes"]
                    for bbox_idx, (crop_bbox, land_bbox) in enumerate(zip(crop_bboxes, land_bboxes)):
                        if "scores" in cur_info["content_ann"]:
                            if cur_info["content_ann"]["scores"][bbox_idx][0] < score_thres: continue
                        x_min = min(crop_bbox[0::2])
                        x_max = max(crop_bbox[0::2])
                        y_min = min(crop_bbox[1::2])
                        y_max = max(crop_bbox[1::2])
                        box_tmp = [
                            [crop_bbox[0] + land_bbox[0][0], crop_bbox[1] + land_bbox[0][1]],
                            [crop_bbox[0] + land_bbox[1][0], crop_bbox[1] + land_bbox[1][1]],
                            [crop_bbox[0] + land_bbox[2][0], crop_bbox[1] + land_bbox[2][1]],
                            [crop_bbox[0] + land_bbox[3][0], crop_bbox[1] + land_bbox[3][1]],
                        ]
                        for i in range(4):
                            box_tmp[i][0] = max(min(box_tmp[i][0], x_max), x_min)
                            box_tmp[i][1] = max(min(box_tmp[i][1], y_max), y_min)
                            box_tmp[i] = tuple(box_tmp[i])
                        plot_bbox(draw, box_tmp, landpoint_color)

                if spots_info is not None:
                    image_spots = None
                    for spot_id, spot_info in spots_info.items():
                        if spot_id in img_path.split("_"):
                            image_spots = spot_info
                    assert image_spots is not None
                    for box in image_spots:
                        box_tmp = [tuple(p) for p in box]
                        plot_bbox(draw, box_tmp, spot_color)

                if gts is not None:
                    gt_info = gts[img_path]
                    boxes = gt_info["content_ann"]["bboxes"]
                    for bbox_idx, box in enumerate(boxes):
                        x_min = min(box[0::2])
                        x_max = max(box[0::2])
                        y_min = min(box[1::2])
                        y_max = max(box[1::2])
                        box_tmp = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                        plot_bbox(draw, box_tmp, gt_color)

            if flag:
                img_name = '__'.join(img_full_path.rsplit('/', 2)[-2:])
                img.save(os.path.join(output_folder, img_name))
                # 输出文件名
                print(img_name)



    print("操作完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--input', type=str, help='path to the JSON file')
    parser.add_argument('-o', '--output_folder', type=str, default='output_json_images', help='folder to save the output images')
    parser.add_argument('-n', '--debug_num', type=int, default=100000000, help='debug image number')
    parser.add_argument('--debug_dir', type=str, default=None, help='debug image name dir')
    parser.add_argument('-p', '--image_prefix', type=str, default='/', help='file prefix')
    parser.add_argument('-g', '--gt_json_path', type=str, default=None, help='file prefix')
    parser.add_argument('--score_thres', type=float, nargs='+', default=[0.65,0.3,0.3,0.3,0.3], help='threshold for debug')
    parser.add_argument('-k', '--image_name', nargs='+', help='selected debug image name, multiple input supported')
    parser.add_argument('-s', '--spot_json_path', type=str, help='parking spot json file')
    parser.add_argument('-c', '--debug_class', type=str, nargs='+', 
                        default=['vehicle',"pedestrian","cyclist","cyclist_person","another_vehicle"], 
                        help='class to draw debug images')

    args = parser.parse_args()
    assert len(args.score_thres) == len(args.debug_class)

    main(
        json_file_path = args.input,
        output_folder = args.output_folder,
        debug_num = args.debug_num,
        debug_dir = args.debug_dir,
        pre = args.image_prefix,
        score_thres = args.score_thres,
        selected_image_name = args.image_name,
        gt_json_path = args.gt_json_path,
        spot_json_path = args.spot_json_path,
        debug_class = args.debug_class,
    )

