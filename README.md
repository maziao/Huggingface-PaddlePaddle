# Huggingface-PaddlePaddle
Convert HuggingFace code and pretrained models to a PaddlePaddle supported format.

## Procedures

### Step 1. Download HuggingFace model checkpoints
```commandline
huggingface-cli download --resume-download PRETRAINED_MODEL_NAME --cache-dir CACHE_DIR
```

### Step 2. Transform HuggingFace checkpoints (PyTorch) to PaddlePaddle
```commandline
python transform_checkpoint/transform_xxx.py --hf-repo CACHE_DIR --pd-repo TARGET_DIR
```

### [Optional] Step 3. Check correctness of transformation
```commandline
python check_correctness.py --hf-repo CACHE_DIR --pd-repo TARGET_DIR
```

### Step 4. Generate new config file
```commandline
python generate_config.py --src-config PATH_TO_SRC_CONFIG --mode-name MODEL_NAME --tgt-dir TARGET_DIR
```

### Step 5. Continue training from pretrained checkpoints
```commandline
CUDA_VISIBLE_DEVICES={x} python train.py \
    --model-config MODEL_CONFIG \
    --model-name MODEL_NAME \
    --tokenizer TOKENIZER \
    --dataset DATASET \
    --criterion CRITERION \
    --pretrained-model-path TARGET_DIR \
    --save-dir SAVE_DIR
```

### Step 6. Evaluation
```commandline
CUDA_VISIBLE_DEVICES={x} python test.py --dataset DATASET --pretrained-model-path SAVE_DIR
```
