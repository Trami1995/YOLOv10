#!/bin/bash

INFER=$1 # detection result json file in yitu custom format
OUT=$2 # output directory for evaluation

XYQT06_60PX="/mnt/CEPH_CHANNEL/nzhou/projects/jiaojing/datas/v3/XYQT06/simple_60px/data_keys_modified.json"
XYQT06_90PX="/mnt/CEPH_CHANNEL/nzhou/projects/jiaojing/datas/v3/XYQT06/simple_90px/data_keys_modified.json"

# 获取脚本的相对路径
SCRIPT_PATH="${BASH_SOURCE[0]}"

# 获取脚本所在的目录
SCRIPT_DIR="$( cd "$( dirname "$SCRIPT_PATH" )" && pwd )"

EVAL_DIR="$( dirname "$SCRIPT_DIR" )"

PYTHONPATH="${EVAL_DIR}":$PYTHONPATH \
python ${EVAL_DIR}/eval.py \
    --infer_json ${INFER} \
    --gt_json ${XYQT06_60PX} \
    --out_dir ${OUT}/XYQT06_sp_60px \
    --targets 0.054 0.062 0.018 0.015 \
    ${@:3}


PYTHONPATH="${EVAL_DIR}":$PYTHONPATH \
python ${EVAL_DIR}/eval.py \
    --infer_json ${INFER} \
    --gt_json ${XYQT06_90PX} \
    --out_dir ${OUT}/XYQT06_sp_90px \
    --targets 0.054 0.062 0.018 0.015 \
    ${@:3}