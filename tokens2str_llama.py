from transformers import LlamaTokenizerFast
import numpy as np
import argparse


model_path = "./models/minicpmv-0_9b-guiagent-0305-ckpt13500/"


def convert(token_path):
    tokens = np.fromfile(token_path, dtype=np.int32)

    tokenizer = LlamaTokenizerFast.from_pretrained(model_path)

    out_str = tokenizer.decode(tokens)

    print(out_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('token_path', help='prompt')
    args = parser.parse_args()

    convert(args.token_path)
