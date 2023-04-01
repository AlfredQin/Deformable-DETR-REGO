#!/usr/bin/env bash

set -x

EXP_DIR=exps/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage
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
    --pretrained_coco ./ckpt/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth \
    --num_queries_one2one 300 \
    --num_queries_one2many 1500 \
    --k_one2many 6 \
    --lambda_one2many 1.0 \
    ${PY_ARGS}

