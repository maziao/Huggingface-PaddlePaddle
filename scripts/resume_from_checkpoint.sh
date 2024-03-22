#!/bin/bash

# ========== metadata ========== #
cuda=$1
pretrained_model_path=$2
dataset=$3
work_dir=$4
# ========== metadata ========== #

mkdir -p $work_dir/$model/$dataset

CUDA_VISIBLE_DEVICES=$cuda python3 train.py \
    --tokenizer ${pretrained_model_path} \
    --pretrained-model-path ${pretrained_model_path} \
    --dataset ${dataset} \
    --criterion cross_entropy \
    --save-dir $work_dir/$model/$dataset
