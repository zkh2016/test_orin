from transformers import AutoTokenizer
import numpy as np

path = "/DATA/disk1/zhangkaihuo/minicpmv-0_9b-guiagent-0305-ckpt13500/"


def convert(token_path):
    tokens = np.fromfile(token_path, dtype=np.int32)

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    out_str = tokenizer.decode(tokens)

    print(out_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('token_path', help='prompt')
    args = parser.parse_args()

    convert(args.prompt)
