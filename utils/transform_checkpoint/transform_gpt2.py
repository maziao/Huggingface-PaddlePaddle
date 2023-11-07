import paddle
import torch


def transform_gpt2(state_dict):
    new_state_dict = dict()
    print(state_dict.keys())
    for key, value in state_dict.items():
        split_key = key.split('.')[1:]
        if split_key[0] == 'wte':
            new_key = f'transformer.embed.token_embed.{split_key[1]}'
        elif split_key[0] == 'wpe':
            new_key = f'transformer.embed.pos_embed.{split_key[1]}'
        elif split_key[0] == 'h':
            if split_key[2] in ['ln_1', 'ln_2']:
                new_key = (
                    f'transformer.decoder.{split_key[1]}.{split_key[2]}.{split_key[3]}'
                )
            elif split_key[2] == 'attn':
                if split_key[3] in ['bias', 'masked_bias']:
                    continue
                new_key = (
                    f'transformer.decoder.{split_key[1]}.self_attn.{split_key[3]}.{split_key[4]}'
                )
            elif split_key[2] == 'mlp':
                if split_key[3] == 'c_fc':
                    new_key = (
                        f'transformer.decoder.{split_key[1]}.mlp.fc_in.{split_key[4]}'
                    )
                elif split_key[3] == 'c_proj':
                    new_key = (
                        f'transformer.decoder.{split_key[1]}.mlp.fc_out.{split_key[4]}'
                    )
        elif split_key[0] == 'ln_f':
            new_key = f'transformer.ln_f.{split_key[1]}'
        else:
            continue

        # if split_key[-1] == 'weight' and split_key[0] not in ['wte', 'wpe'] and len(value.shape) != 1:
        #     value = value.T
        new_state_dict[new_key] = paddle.Tensor(value.numpy())
    for key, value in new_state_dict.items():
        print(key, value.shape)
    paddle.save(obj=new_state_dict, path='/home/mza/model-zoo/paddle/gpt2-added-sep.pdparams')


if __name__ == '__main__':
    state_dict = torch.load(
        '/home/mza/model-zoo/gpt2-added-sep/gpt2/pytorch_model.bin')
    transform_gpt2(state_dict)
