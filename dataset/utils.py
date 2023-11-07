import paddle


def generate_mask(ids, pad_token_idx=0):
    mask = paddle.ones_like(ids)
    mask[ids == pad_token_idx] = 0
    return mask
