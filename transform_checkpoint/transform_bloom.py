import os.path
import paddle
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def transform_bloom(state_dict: dict):
    new_state_dict = dict()
    for key, value in state_dict.items():
        split_key = key.split('.')[1:]
        if split_key[0] == 'word_embeddings':
            new_key = f'transformer.embed.token_embed.{split_key[1]}'
        elif split_key[0] == 'word_embeddings_layernorm':
            new_key = f'transformer.embed.ln.{split_key[1]}'
        elif split_key[0] == 'h':
            if split_key[2] == 'input_layernorm':
                new_key = (
                    f'transformer.decoder.{split_key[1]}.ln_1.{split_key[3]}'
                )
            elif split_key[2] == 'post_attention_layernorm':
                new_key = (
                    f'transformer.decoder.{split_key[1]}.ln_2.{split_key[3]}'
                )
            elif split_key[2] == 'self_attention':
                if split_key[3] == 'query_key_value':
                    new_key = (
                        f'transformer.decoder.{split_key[1]}.self_attn.c_attn.{split_key[4]}'
                    )
                elif split_key[3] == 'dense':
                    new_key = (
                        f'transformer.decoder.{split_key[1]}.self_attn.c_proj.{split_key[4]}'
                    )
                if split_key[4] == 'weight':
                    value = value.T
            elif split_key[2] == 'mlp':
                if split_key[3] == 'dense_h_to_4h':
                    new_key = (
                        f'transformer.decoder.{split_key[1]}.mlp.fc_in.{split_key[4]}'
                    )
                elif split_key[3] == 'dense_4h_to_h':
                    new_key = (
                        f'transformer.decoder.{split_key[1]}.mlp.fc_out.{split_key[4]}'
                    )
                if split_key[4] == 'weight':
                    value = value.T
        elif split_key[0] == 'ln_f':
            new_key = f'transformer.ln_f.{split_key[1]}'
        else:
            continue

        new_state_dict[new_key] = paddle.Tensor(value.numpy())
    return new_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-repo', type=str, default='bigscience/bloom-560m')
    parser.add_argument('--pd-repo', type=str)
    args = parser.parse_args()

    print(f"[!] Loading tokenizer from {args.hf_repo} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_repo, use_fast=True)
    print(f"[!] Loading pretrained model from {args.hf_repo} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_repo)
    print(f"[!] Perform transformation ...")
    pd_state_dict = transform_bloom(hf_model.state_dict())
    paddle.save(obj=pd_state_dict, path=os.path.join(args.pd_repo, 'paddle_model.pdparams'))
    tokenizer.save_pretrained(args.pd_repo)
    print(f"[!] Transformation completed.")
