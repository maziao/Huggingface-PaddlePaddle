# Huggingface-PaddlePaddle

Convert HuggingFace code and pretrained models to a PaddlePaddle supported format.

## Supported models

| Family             | Converted Checkpoints                                                                                                        | Article                                                                                     |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| GPT2               | [gpt2](https://huggingface.co/DataHammer/PaddlePaddle-GPT2)                                                                  | Language Models are Unsupervised Multitask Learners                                         |
| GPT-Neo            | [EleutherAI/gpt-neo-125m](https://huggingface.co/DataHammer/PaddlePaddle-GPT-Neo-125M)                                       | GPT-NeoX-20B: An Open-Source Autoregressive Language Model                                  |
| OPT                | [facebook/opt-125m](https://huggingface.co/DataHammer/PaddlePaddle-OPT-125M)                                                 | OPT: Open Pre-trained Transformer Language Models                                           |
| BLOOM              | [YeungNLP/bloom-396m-zh](https://huggingface.co/DataHammer/PaddlePaddle-BLOOM-396M-zh)                                       | BLOOM: A 176B-Parameter Open-Access Multilingual Language Model                             |
| LLaMa              | [TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T](https://huggingface.co/DataHammer/PaddlePaddle-TinyLlama-1.1B) | LLaMA: Open and Efficient Foundation Language Models                                        |
| DITTO              | [Finetuned](https://huggingface.co/DataHammer/PaddlePaddle-DITTO)                                                            | Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation |
| ScaleGrad          | [Finetuned](https://huggingface.co/DataHammer/PaddlePaddle-ScaleGrad)                                                        | Straight to the Gradient: Learning to Use Novel Tokens for Neural Text Generation           |
| SimCTG             | [Finetuned](https://huggingface.co/DataHammer/PaddlePaddle-SimCTG)                                                           | A Contrastive Framework for Neural Text Generation                                          |
| Unlikelihood-Token | [Finetuned](https://huggingface.co/DataHammer/PaddlePaddle-UnlikelihoodTraining-Token-Level)                                 | Neural Text Generation with Unlikelihood Training                                           |
| Unlikelihood-Seq   | [Finetuned](https://huggingface.co/DataHammer/PaddlePaddle-UnlikelihoodTraining-Sequence-Level)                              | Neural Text Generation with Unlikelihood Training                                           |
| Qwen-1.5           | [Qwen/Qwen1.5-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B)                                                                | Qwen Technical Report                                                                       |

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
