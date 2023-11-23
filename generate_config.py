import argparse

from config import AutoConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-config', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--tgt-dir', type=str, required=True)
    args = parser.parse_args()

    cfg = AutoConfig.from_yaml(args.src_config, model_name=args.model_name)
    print(cfg)
    cfg.save_pretrained(args.tgt_dir)
    print(f"[!] Config generation completed.")
