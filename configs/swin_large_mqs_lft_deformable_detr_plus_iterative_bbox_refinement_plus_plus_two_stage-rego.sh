#!/usr/bin/env bash

set -x

EXP_DIR=exps/swin_large_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_rego
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
    --mixed_selection \
    --look_forward_twice \
    --dim_feedforward 2048 \
    --backbone swin_large \
    --pretrained_backbone_path ./ckpt/pretrained_backbone/swin_large_patch4_window7_224_22kto1k.pth \
    --pretrained_coco ./ckpt/swin_large_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth \
    ${PY_ARGS}

    ##     --pretrained_coco ./ckpt/r50-deformable-detr-plus-plus-rego.pth \
