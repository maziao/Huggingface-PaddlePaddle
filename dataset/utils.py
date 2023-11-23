import numpy as np
from typing import Union, List, Tuple


def generate_mask(ids, pad_token_idx=0):
    mask = np.ones_like(ids)
    mask[ids == pad_token_idx] = 0
    return mask


def pad_sequence(
        sequences: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]],
        batch_first: bool = False,
        padding_value: float = 0.0
):
    if isinstance(sequences, np.ndarray):
        return np.reshape(sequences, (1,) + sequences.shape)
    else:
        padded_sequences = []
        dtype = sequences[0].dtype
        max_len = np.max([len(sequence) for sequence in sequences])
        for sequence in sequences:
            padded_sequence = np.pad(
                array=sequence,
                pad_width=(0, max_len - len(sequence)),
                mode="constant",
                constant_values=(0, np.array(padding_value, dtype=dtype))
            )
            padded_sequences.append(padded_sequence)
        if batch_first:
            return np.stack(padded_sequences)
        else:
            return np.stack(padded_sequences).T


if __name__ == '__main__':
    a = [
        np.array([1, 2, 3]),
        np.array([4, 5]),
        np.array([6, 7, 8, 9])
    ]
    # a = np.array([1, 2, 3])
    print(pad_sequence(a, padding_value=1.5))
