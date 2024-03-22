#!/bin/bash

# ========== metadata ========== #
cuda=$1
model_config=$2
model_name=$3
tokenizer=$4
dataset=$5
work_dir=$6
# ========== metadata ========== #

mkdir -p $work_dir/$model/$dataset

CUDA_VISIBLE_DEVICES=$cuda python3 train.py \
    --model-config ${model_config} \
    --model-name ${model_name} \
    --tokenizer ${tokenizer} \
    --dataset ${dataset} \
    --criterion cross_entropy \
    --save-dir $work_dir/$model/$dataset
