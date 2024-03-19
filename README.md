# Huggingface-PaddlePaddle

Convert HuggingFace code and pretrained models to a PaddlePaddle supported format.

## Supported models

| ID  | Family             | Converted Checkpoints                                                                                                        | Article                                                                                     |
| --- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 1   | GPT2               | [gpt2](https://huggingface.co/DataHammer/PaddlePaddle-GPT2)                                                                  | Language Models are Unsupervised Multitask Learners                                         |
| 2   | GPT-Neo            | [EleutherAI/gpt-neo-125m](https://huggingface.co/DataHammer/PaddlePaddle-GPT-Neo-125M)                                       | GPT-NeoX-20B: An Open-Source Autoregressive Language Model                                  |
| 3   | OPT                | [facebook/opt-125m](https://huggingface.co/DataHammer/PaddlePaddle-OPT-125M)                                                 | OPT: Open Pre-trained Transformer Language Models                                           |
| 4   | BLOOM              | [YeungNLP/bloom-396m-zh](https://huggingface.co/DataHammer/PaddlePaddle-BLOOM-396M-zh)                                       | BLOOM: A 176B-Parameter Open-Access Multilingual Language Model                             |
| 5   | LLaMa              | [TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T](https://huggingface.co/DataHammer/PaddlePaddle-TinyLlama-1.1B) | LLaMA: Open and Efficient Foundation Language Models                                        |
| 6   | DITTO              | [Finetuned](https://huggingface.co/DataHammer/PaddlePaddle-DITTO)                                                            | Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation |
| 7   | ScaleGrad          | [Finetuned](https://huggingface.co/DataHammer/PaddlePaddle-ScaleGrad)                                                        | Straight to the Gradient: Learning to Use Novel Tokens for Neural Text Generation           |
| 8   | SimCTG             | [Finetuned](https://huggingface.co/DataHammer/PaddlePaddle-SimCTG)                                                           | A Contrastive Framework for Neural Text Generation                                          |
| 9   | Unlikelihood-Token | [Finetuned](https://huggingface.co/DataHammer/PaddlePaddle-UnlikelihoodTraining-Token-Level)                                 | Neural Text Generation with Unlikelihood Training                                           |
| 10  | Unlikelihood-Seq   | [Finetuned](https://huggingface.co/DataHammer/PaddlePaddle-UnlikelihoodTraining-Sequence-Level)                              | Neural Text Generation with Unlikelihood Training                                           |
| 11  | Qwen-1.5           | [Qwen/Qwen1.5-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B)                                                                | Qwen Technical Report                                                                       |
| 12  | GPT-SW3            | [AI-Sweden-Models/gpt-sw3-126m](https://huggingface.co/AI-Sweden-Models/gpt-sw3-126m)                                        | GPT-SW3: An Autoregressive Language Model for the Nordic Languages                          |
| 13  | Galactica          | [facebook/galactica-125m](https://huggingface.co/facebook/galactica-125m)                                                    | Galactica: A Large Language Model for Science                                               |
| 14  | DeepSeek LLM       | [deepseek-ai/deepseek-coder-1.3b-base](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base)                          | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism                          |
| 15  | InternLM2          | [internlm/internlm2-1_8b](https://huggingface.co/internlm/internlm2-1_8b)                                                    | [InternLM - GitHub Repo](https://github.com/InternLM/InternLM)                              |

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
