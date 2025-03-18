from transformers import AutoTokenizer
path = "/DATA/disk1/zhangkaihuo/minicpmv-0_9b-guiagent-0305-ckpt13500/"


def convert(prompt):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\nassistant"
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    tokens += tokenizer.encode("<image>", add_special_tokens=False)

    img_placehoder = [0 for _ in range(64)]
    tokens += img_placehoder
    tokens += tokenizer.encode("</image>", add_special_tokens=False)

    print(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('prompt', help='prompt')
    args = parser.parse_args()

    convert(args.prompt)
