#!/bin/bash

# ========== metadata ========== #
cuda=$1
model=Qwen-1.5-0.5B
dataset=$2
work_dir=$3
pretrained_model=$4
# ========== metadata ========== #

mkdir -p $work_dir/$model/$dataset

CUDA_VISIBLE_DEVICES=$cuda python3 train.py \
    --model-config ./config/model_config/qwen-1_5.yaml \
    --model-name $model \
    --tokenizer $pretrained_model \
    --dataset $dataset \
    --criterion cross_entropy \
    --pretrained-model-path $pretrained_model \
    --save-dir $work_dir/$model/$dataset
