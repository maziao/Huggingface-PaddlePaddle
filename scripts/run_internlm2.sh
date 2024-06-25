#!/bin/bash

# ========== metadata ========== #
cuda=$1
model=internlm/internlm2-1_8b
dataset=$2
work_dir=$3
pretrained_model=$4
# ========== metadata ========== #

mkdir -p $work_dir/$model/$dataset

CUDA_VISIBLE_DEVICES=$cuda python3 train.py \
    --model-config ./config/model_config/internlm2.yaml \
    --model-name $model \
    --tokenizer $pretrained_model \
    --dataset $dataset \
    --criterion cross_entropy \
    --pretrained-model-path $pretrained_model \
    --save-dir $work_dir/$model/$dataset
