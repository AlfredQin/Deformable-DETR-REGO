#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_rego
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --use_rego \
    --with_box_refine \
    --two_stage \
    --coco_path '/home/qinc/Dataset/ISAID/iSAID_patches' \
    --num_workers 12 \
    --batch_size 2 \
    --lr 5e-5 \
    --dataset_file 'isaid' \
    --pretrained_coco ./ckpt/r50-deformable-detr-plus-plus-rego.pth \
    --mixed_selection \
    --look_forward_twice \
    ${PY_ARGS}
