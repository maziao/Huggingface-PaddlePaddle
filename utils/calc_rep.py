import json

import numpy as np
from tqdm import tqdm
from nltk import ngrams


# rep-w, counting the current token occurs in previous prefix (overall prefix sequence) in the generations
def calculate_rep_w(text_list, w=16):
    # code borrowed from paper: A Theoretical Analysis of the Repetition Problem in Text Generation
    # tokens are the BPE tokens from this paper: NEURAL TEXT DEGENERATION WITH UNLIKELIHOOD TRAINING
    rep_w = []
    for tokens in tqdm(text_list, desc=f"Calculating rep-w"):
        rep_w_single = 0
        for idx in range(1, len(tokens)):
            t = tokens[idx]
            prefix = set(tokens[max(0, idx - w):idx])
            if t in prefix:
                rep_w_single += 1
        if len(tokens) <= 1:
            continue
        rep_w_single /= len(tokens) - 1
        rep_w.append(rep_w_single)
    rep_w = np.mean(rep_w) * 100
    return rep_w


# code borrowed from paper: A Theoretical Analysis of the Repetition Problem in Text Generation
# https://github.com/fuzihaofzh/repetition-problem-nlg/blob/f0f80ea986d288fb5a76f48d4d16ddb60cace575/src/eval_metrics.py#L133
def calculate_rep_r(text_list):
    rep_r_list = []
    for tokens in tqdm(text_list, desc=f"Calculating rep-r"):
        if len(tokens) < 2:
            rep_r_list.append(0)
        counter = {}
        for j in range(len(tokens) - 1):
            gm = '%s %s' % (tokens[j], tokens[j + 1])
            counter[gm] = counter[gm] + 1 if gm in counter else 1
        label = [0] * len(tokens)
        for i in range(1, len(tokens)):
            if counter['%s %s' % (tokens[i - 1], tokens[i])] > 1:
                label[i - 1] = label[i] = 1
        if len(label) == 0:
            continue
        ratio = sum(label) / len(label)
        rep_r_list.append(ratio)
    rep_r = np.mean(rep_r_list) * 100
    return rep_r


def calculate_rep_n(text_list):
    ngram_list = [2, 3, 4]
    results = {i: {'num_rep': [], 'num_total': []} for i in ngram_list}
    for tokens in tqdm(text_list, desc=f"Calculating repetition ratio"):
        rest_dict = compute_instance(tokens, ngram_list)
        for n, (num_rep, num_total) in rest_dict.items():
            results[n]['num_rep'].append(num_rep)
            results[n]['num_total'].append(num_total)
    final = {f"rep-{i}": -1 for i in ngram_list}
    for n, item in results.items():
        a = sum(item['num_rep'])
        b = sum(item['num_total'])
        final[f"rep-{n}"] = 100 * a / b
    return final


def calculate_rep(text_list):
    result = calculate_rep_n(text_list)
    result['rep-w'] = calculate_rep_w(text_list)
    result['rep-r'] = calculate_rep_r(text_list)
    return result


def compute_instance(tokens, ngram_list):
    res_dict = {}
    for n in ngram_list:
        num_rep, num_total = eval_text(tokens, n)
        res_dict[n] = (num_rep, num_total)
    return res_dict


def eval_text(tokens, ngram):
    ngram_list = list(ngrams(tokens, ngram))
    ngram_set = set()
    counter = 0
    for item in ngram_list:
        if item not in ngram_set:
            ngram_set.add(item)
        else:
            counter += 1
    if len(ngram_list) > 0:
        return counter, len(ngram_list)
    else:
        return 0, 0


if __name__ == '__main__':
    import argparse
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True, type=str)
    parser.add_argument('--tokenizer', required=False, type=str, default='gpt2')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)

    generations = []
    with open(args.data_file, 'r', encoding="utf-8") as f:
        data = json.load(f)
    for sample in data:
        generations.append(tokenizer.decode(sample))

    rep_results = calculate_rep_n(generations)
    rep_w = calculate_rep_w(generations)
    rep_r = calculate_rep_r(generations)
    print(rep_results)
    print(f"rep_w: {round(rep_w, 4)}")
    print(f"rep_r: {round(rep_r, 4)}")
