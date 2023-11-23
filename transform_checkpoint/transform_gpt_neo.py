import os.path
import paddle
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def transform_bloom(state_dict: dict):
    new_state_dict = dict()
    for key, value in state_dict.items():
        split_key = key.split('.')[1:]
        new_key = None

        if split_key[0] == 'wte':
            new_key = f'transformer.embed.token_embed.{split_key[1]}'
        elif split_key[0] == 'wpe':
            new_key = f'transformer.embed.pos_embed.{split_key[1]}'
        elif split_key[0] == 'ln_f':
            new_key = f'transformer.ln_f.{split_key[1]}'
        elif split_key[0] == 'h':
            if split_key[2] == 'attn':
                if split_key[4] in ['k_proj', 'v_proj']:
                    continue
                elif split_key[4] == 'q_proj':
                    new_key = (
                        f'transformer.decoder.{split_key[1]}.self_attn.c_attn.{split_key[5]}'
                    )
                    if split_key[5] == 'weight':
                        value = torch.cat(
                            [
                                state_dict[f"transformer.h.{split_key[1]}.attn.attention.{name}.weight"].T
                                for name in ['q_proj', 'k_proj', 'v_proj']
                            ],
                            dim=-1
                        )
                    elif split_key[5] == 'bias':
                        value = torch.cat(
                            [
                                state_dict[f"transformer.h.{split_key[1]}.attn.attention.{name}.bias"]
                                for name in ['q_proj', 'k_proj', 'v_proj']
                            ]
                        )
                elif split_key[4] == 'out_proj':
                    new_key = (
                        f'transformer.decoder.{split_key[1]}.self_attn.c_proj.{split_key[5]}'
                    )
                    if split_key[5] == 'weight':
                        value = value.T
            elif split_key[2] == 'ln_1':
                new_key = (
                    f'transformer.decoder.{split_key[1]}.ln_1.{split_key[3]}'
                )
            elif split_key[2] == 'ln_2':
                new_key = (
                    f'transformer.decoder.{split_key[1]}.ln_2.{split_key[3]}'
                )
            elif split_key[2] == 'mlp':
                if split_key[3] == 'c_fc':
                    new_key = (
                        f'transformer.decoder.{split_key[1]}.mlp.fc_in.{split_key[4]}'
                    )
                    if split_key[4] == 'weight':
                        value = value.T
                elif split_key[3] == 'c_proj':
                    new_key = (
                        f'transformer.decoder.{split_key[1]}.mlp.fc_out.{split_key[4]}'
                    )
                    if split_key[4] == 'weight':
                        value = value.T
        else:
            continue

        new_state_dict[new_key] = paddle.Tensor(value.numpy())

    for key, value in new_state_dict.items():
        print(key, value.shape)
    return new_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-repo', type=str, default='EleutherAI/gpt-neo-125m')
    parser.add_argument('--pd-repo', type=str)
    args = parser.parse_args()

    print(f"[!] Loading tokenizer from {args.hf_repo} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_repo, use_fast=False)
    print(f"[!] Loading pretrained model from {args.hf_repo} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_repo)
    print(f"[!] Perform transformation ...")
    pd_state_dict = transform_bloom(hf_model.state_dict())
    paddle.save(obj=pd_state_dict, path=os.path.join(args.pd_repo, 'paddle_model.pdparams'))
    tokenizer.save_pretrained(args.pd_repo)
    print(f"[!] Transformation completed.")
