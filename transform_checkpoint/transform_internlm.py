import paddle
import os.path
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def transform_llama(state_dict):
    new_state_dict = dict()
    for key, value in state_dict.items():
        split_key = key.split(".")
        new_key = None

        if split_key[0] == "model":
            if split_key[1] == "tok_embeddings":
                new_key = f"transformer.embed.token_embed.{split_key[2]}"
            elif split_key[1] == "layers":
                if split_key[3] == "attention":
                    if split_key[4] == "wqkv":
                        value = value.T
                        n_key_value_head = 8
                        n_query_head_group = 2
                        head_size = 128
                        assert value.size(-1) == n_key_value_head * (n_query_head_group + 2) * head_size
                        value = value.reshape([value.size(0), n_key_value_head, n_query_head_group + 2, head_size])
                        new_state_dict[
                            f"transformer.decoder.{split_key[2]}.self_attn.q_proj.weight"
                        ] = paddle.Tensor(value[..., :n_query_head_group, :].reshape([value.size(0), -1]).numpy())
                        new_state_dict[
                            f"transformer.decoder.{split_key[2]}.self_attn.k_proj.weight"
                        ] = paddle.Tensor(value[..., -2, :].reshape([value.size(0), -1]).numpy())
                        new_state_dict[
                            f"transformer.decoder.{split_key[2]}.self_attn.v_proj.weight"
                        ] = paddle.Tensor(value[..., -1, :].reshape([value.size(0), -1]).numpy())
                        continue
                    elif split_key[4] == "wo":
                        new_key = f"transformer.decoder.{split_key[2]}.self_attn.c_proj.{split_key[5]}"
                elif split_key[3] == "feed_forward":
                    if split_key[4] == "w1":
                        new_key = f"transformer.decoder.{split_key[2]}.mlp.fc_in_1.{split_key[5]}"
                    elif split_key[4] == "w3":
                        new_key = f"transformer.decoder.{split_key[2]}.mlp.fc_in_2.{split_key[5]}"
                    elif split_key[4] == "w2":
                        new_key = f"transformer.decoder.{split_key[2]}.mlp.fc_out.{split_key[5]}"
                elif split_key[3] == "attention_norm":
                    new_key = f"transformer.decoder.{split_key[2]}.ln_1.{split_key[4]}"
                elif split_key[3] == "ffn_norm":
                    new_key = f"transformer.decoder.{split_key[2]}.ln_2.{split_key[4]}"
            elif split_key[1] == "norm":
                new_key = f"transformer.ln_f.{split_key[2]}"
            else:
                continue
        elif split_key[0] == "output":
            new_key = f"lm_head.fc_pred.weight"

        if (
            split_key[-1] == "weight"
            and split_key[1] != "tok_embeddings"
            and len(value.shape) > 1
        ):
            new_state_dict[new_key] = paddle.Tensor(value.T.numpy())
        else:
            new_state_dict[new_key] = paddle.Tensor(value.numpy())

    for key, value in new_state_dict.items():
        print(key, value.shape)
    return new_state_dict


if __name__ == "__main__":
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
