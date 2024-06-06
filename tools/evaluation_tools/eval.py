import os
import logging
import json
import argparse


from eval_tools.eval_utils import Evaluator

logging.basicConfig(
    format='[%(asctime)s %(name)-8s'
           '%(levelname)s %(process)d '
           '%(filename)s:%(lineno)-5d]'
           ' %(message)s',
    level=logging.INFO
)


def main(args):
    infer_json = args.infer_json
    assert os.path.exists(infer_json)

    gt_json = args.gt_json
    assert os.path.exists(gt_json)

    out_path = args.out_dir
    os.makedirs(out_path, exist_ok=True)
    
    mode = args.mode
    targets = args.targets
    iou_thresh = args.iou_thresh
    overwrite = True if args.overwrite else False

    # classes for evaluation
    total_classes = args.classes

    with open(infer_json) as f:
        infer_res = json.load(f)
    
    with open(gt_json) as f:
        gt_res = json.load(f)
    
    evaluator = Evaluator(infer_res, gt_res, out_path, mode, iou_thresh, overwrite, targets, total_classes)
    evaluator.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get Recall@FA metric of detection results")
    parser.add_argument("--infer_json", type=str, help="infer json path")
    parser.add_argument("--gt_json", type=str, help="gt json path")
    parser.add_argument("--out_dir", type=str, help="output path")
    parser.add_argument("--classes", type=str, nargs='+', default=["vehicle", "pedestrian", "cyclist", "cyclist_person", "another_vehicle"], 
                        help="classes type, multiple classes input should be seperated by blank space")
    parser.add_argument(
        "--mode",
        choices=["fa", "thresh"], default="fa",
        help="evaluate mode: fa or thresh"
    )
    parser.add_argument("--targets", type=str, nargs='+', default=None, help='targets to evaluate, \
                        actual targets depend on evaluation mode, multiple target should be divided by blank space')
    parser.add_argument("--iou_thresh", type=float, default=0.5, help='IOU thresh used dering evaluation')
    parser.add_argument('--overwrite', action='store_true', help='do overwrite')
    args = parser.parse_args()

    main(args)


   
