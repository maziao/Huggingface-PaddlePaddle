import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-repo', type=str, required=True)
    parser.add_argument('--tgt-dir', type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_repo, use_fast=False)
    tokenizer.save_pretrained(args.tgt_dir)
    model = AutoModelForCausalLM.from_pretrained(args.hf_repo)
    model.save_pretrained(args.tgt_dir)
