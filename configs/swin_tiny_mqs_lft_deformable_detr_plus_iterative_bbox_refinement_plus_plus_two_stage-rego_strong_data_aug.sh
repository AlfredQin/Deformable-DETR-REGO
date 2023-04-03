#!/usr/bin/env bash

set -x

EXP_DIR=exps/swin_tiny_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_rego_strong_data_aug
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
    --backbone swin_tiny \
    --pretrained_backbone_path ./ckpt/pretrained_backbone/swin_tiny_patch4_window7_224.pth \
    --pretrained_coco ./ckpt/swin_tiny_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth \
    --strong_aug \
    ${PY_ARGS}

    ##     --pretrained_coco ./ckpt/r50-deformable-detr-plus-plus-rego.pth \
