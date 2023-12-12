import paddle
import os.path
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def transform_llama(state_dict):
    new_state_dict = dict()
    for key, value in state_dict.items():
        split_key = key.split('.')
        new_key = None

        if split_key[0] == 'model':
            if split_key[1] == 'embed_tokens':
                new_key = f'transformer.embed.token_embed.{split_key[2]}'
            elif split_key[1] == 'layers':
                if split_key[3] == 'self_attn':
                    if split_key[4] in ['q_proj', 'k_proj', 'v_proj']:
                        new_key = f'transformer.decoder.{split_key[2]}.self_attn.{split_key[4]}.{split_key[5]}'
                    elif split_key[4] == 'o_proj':
                        new_key = f'transformer.decoder.{split_key[2]}.self_attn.c_proj.{split_key[5]}'
                elif split_key[3] == 'mlp':
                    if split_key[4] == 'gate_proj':
                        new_key = f'transformer.decoder.{split_key[2]}.mlp.fc_in_1.{split_key[5]}'
                    elif split_key[4] == 'up_proj':
                        new_key = f'transformer.decoder.{split_key[2]}.mlp.fc_in_2.{split_key[5]}'
                    elif split_key[4] == 'down_proj':
                        new_key = f'transformer.decoder.{split_key[2]}.mlp.fc_out.{split_key[5]}'
                elif split_key[3] == 'input_layernorm':
                    new_key = f'transformer.decoder.{split_key[2]}.ln_1.{split_key[4]}'
                elif split_key[3] == 'post_attention_layernorm':
                    new_key = f'transformer.decoder.{split_key[2]}.ln_2.{split_key[4]}'
            elif split_key[1] == 'norm':
                new_key = f'transformer.ln_f.{split_key[2]}'
            else:
                continue
        elif split_key[0] == 'lm_head':
            new_key = f'lm_head.fc_pred.weight'

        if split_key[-1] == 'weight' and split_key[1] != 'embed_tokens' and len(value.shape) > 1:
            new_state_dict[new_key] = paddle.Tensor(value.T.numpy())
        else:
            new_state_dict[new_key] = paddle.Tensor(value.numpy())

    for key, value in new_state_dict.items():
        print(key, value.shape)
    return new_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-repo', type=str, default='TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T')
    parser.add_argument('--pd-repo', type=str)
    args = parser.parse_args()

    print(f"[!] Loading tokenizer from {args.hf_repo} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_repo, use_fast=False)
    print(f"[!] Loading pretrained model from {args.hf_repo} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_repo)
    print(f"[!] Perform transformation ...")
    pd_state_dict = transform_llama(hf_model.state_dict())
    paddle.save(obj=pd_state_dict, path=os.path.join(args.pd_repo, 'paddle_model.pdparams'))
    tokenizer.save_pretrained(args.pd_repo)
    print(f"[!] Transformation completed.")
