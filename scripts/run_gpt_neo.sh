#!/bin/bash

# ========== metadata ========== #
cuda=$1
model=$2
dataset=$3
work_dir=$4
pretrained_model=$5
# ========== metadata ========== #

mkdir -p $work_dir/$model/$dataset

CUDA_VISIBLE_DEVICES=$cuda python3 train.py \
    --model-config ./config/model_config/gpt-neo.yaml \
    --model-name $model \
    --tokenizer $pretrained_model \
    --dataset $dataset \
    --criterion cross_entropy \
    --pretrained-model-path $pretrained_model \
    --save-dir $work_dir/$model/$dataset
