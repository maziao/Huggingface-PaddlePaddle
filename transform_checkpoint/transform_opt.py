import os.path
import paddle
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def transform_bloom(state_dict: dict):
    new_state_dict = dict()
    for key, value in state_dict.items():
        split_key = key.split('.')[2:]
        if len(split_key) == 0:
            continue
        if split_key[0] == 'embed_tokens':
            new_key = f'transformer.embed.token_embed.{split_key[1]}'
        elif split_key[0] == 'embed_positions':
            new_key = f'transformer.embed.pos_embed.{split_key[1]}'
        elif split_key[0] == 'final_layer_norm':
            new_key = f'transformer.ln_f.{split_key[1]}'
        elif split_key[0] == 'layers':
            if split_key[2] == 'self_attn':
                if split_key[3] in ['k_proj', 'v_proj']:
                    continue
                elif split_key[3] == 'q_proj':
                    new_key = (
                        f'transformer.decoder.{split_key[1]}.self_attn.c_attn.{split_key[4]}'
                    )
                    if split_key[4] == 'weight':
                        value = torch.cat(
                            [
                                state_dict[f"model.decoder.layers.{split_key[1]}.self_attn.{name}.weight"].T
                                for name in ['q_proj', 'k_proj', 'v_proj']
                            ],
                            dim=-1
                        )
                    elif split_key[4] == 'bias':
                        value = torch.cat(
                            [
                                state_dict[f"model.decoder.layers.{split_key[1]}.self_attn.{name}.bias"]
                                for name in ['q_proj', 'k_proj', 'v_proj']
                            ]
                        )
                elif split_key[3] == 'out_proj':
                    new_key = (
                        f'transformer.decoder.{split_key[1]}.self_attn.c_proj.{split_key[4]}'
                    )
                    if split_key[4] == 'weight':
                        value = value.T
            elif split_key[2] == 'self_attn_layer_norm':
                new_key = (
                    f'transformer.decoder.{split_key[1]}.ln_1.{split_key[3]}'
                )
            elif split_key[2] == 'fc1':
                new_key = (
                    f'transformer.decoder.{split_key[1]}.mlp.fc_in.{split_key[3]}'
                )
                if split_key[3] == 'weight':
                    value = value.T
            elif split_key[2] == 'fc2':
                new_key = (
                    f'transformer.decoder.{split_key[1]}.mlp.fc_out.{split_key[3]}'
                )
                if split_key[3] == 'weight':
                    value = value.T
            elif split_key[2] == 'final_layer_norm':
                new_key = (
                    f'transformer.decoder.{split_key[1]}.ln_2.{split_key[3]}'
                )
        else:
            continue

        new_state_dict[new_key] = paddle.Tensor(value.numpy())
    for key, value in new_state_dict.items():
        print(key, value.shape)
    return new_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-repo', type=str, default='facebook/opt-125m')
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
