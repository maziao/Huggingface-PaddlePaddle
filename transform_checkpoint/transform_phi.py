import paddle
import os.path
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def transform_phi(state_dict):
    new_state_dict = dict()
    for key, value in state_dict.items():
        split_key = key.split('.')
        new_key = None

        if split_key[0] == 'model':
            if split_key[1] == 'embed_tokens':
                new_key = f'transformer.embed.token_embed.{split_key[2]}'
            elif split_key[1] == 'layers':
                if split_key[3] == 'self_attn':
                    if split_key[4] == 'q_proj':
                        new_key = f'transformer.decoder.{split_key[2]}.self_attn.c_attn.{split_key[5]}'
                        q = value.T
                        k = state_dict[key.replace('q_proj', 'k_proj')].T
                        v = state_dict[key.replace('q_proj', 'v_proj')].T
                        fused_qkv = torch.cat([q, k, v], dim=-1)
                        new_state_dict[new_key] = paddle.Tensor(fused_qkv.numpy())
                        continue
                    elif split_key[4] == 'dense':
                        new_key = f'transformer.decoder.{split_key[2]}.self_attn.c_proj.{split_key[5]}'
                    else:
                        continue
                elif split_key[3] == 'mlp':
                    if split_key[4] == 'fc1':
                        new_key = f'transformer.decoder.{split_key[2]}.mlp.fc_in.{split_key[5]}'
                    elif split_key[4] == 'fc2':
                        new_key = f'transformer.decoder.{split_key[2]}.mlp.fc_out.{split_key[5]}'
                elif split_key[3] == 'input_layernorm':
                    new_key = f'transformer.decoder.{split_key[2]}.ln_1.{split_key[4]}'
            elif split_key[1] == 'final_layernorm':
                new_key = f'transformer.ln_f.{split_key[2]}'
            else:
                continue
        elif split_key[0] == 'lm_head':
            new_key = f'lm_head.fc_pred.{split_key[1]}'

        if split_key[-1] == 'weight' and split_key[1] != 'embed_tokens' and len(value.shape) > 1:
            new_state_dict[new_key] = paddle.Tensor(value.T.numpy())
        else:
            new_state_dict[new_key] = paddle.Tensor(value.numpy())

    for key, value in new_state_dict.items():
        print(key, value.shape)
    return new_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-repo', type=str, default='microsoft/phi-1_5')
    parser.add_argument('--pd-repo', type=str, required=True)
    args = parser.parse_args()

    print(f"[!] Loading tokenizer from {args.hf_repo} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_repo, use_fast=False)
    print(f"[!] Loading pretrained model from {args.hf_repo} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_repo)
    print(f"[!] Perform transformation ...")
    pd_state_dict = transform_phi(hf_model.state_dict())
    paddle.save(obj=pd_state_dict, path=os.path.join(args.pd_repo, 'paddle_model.pdparams'))
    tokenizer.save_pretrained(args.pd_repo)
    print(f"[!] Transformation completed.")
