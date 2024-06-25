import paddle
import argparse
from tqdm import tqdm


def cast_checkpoint(state_dict):
    new_state_dict = dict()

    for key, value in tqdm(state_dict.items()):
        new_state_dict[key] = value.cast(paddle.float16)
    return new_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pd-ckpt', type=str, required=True)
    args = parser.parse_args()

    pd_state_dict = paddle.load(args.pd_ckpt)
    print(f"[!] Initial checkpoint loaded.")
    paddle.save(obj=cast_checkpoint(pd_state_dict), path=args.pd_ckpt)
    print(f"[!] Transformation completed.")
