from transformers import LlamaTokenizerFast
import numpy as np
import argparse


model_path = "./models/minicpmv-0_9b-guiagent-0305-ckpt13500-llama-format/"

def read_integers_from_file(file_path):
    """读取逗号分隔的整数文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        # 一次性读取所有内容并合并换行符
        content = f.read().replace('\n', ',')
        # 分割并过滤空字符串
        str_numbers = [s.strip() for s in content.split(',') if s.strip()]
        # 转换为整数列表
        return [int(num) for num in str_numbers]

# 使用示例
#numbers = read_integers_from_file('data.txt')
#print(numbers)  # 输出示例：[1, 2, 3, 4, 5]



def convert(token_path):
    #tokens = np.fromfile(token_path, dtype=np.int32)
    tokens = read_intergers_from_file(token_path)
    print(tokens)

    tokenizer = LlamaTokenizerFast.from_pretrained(model_path)

    out_str = tokenizer.decode(tokens)

    print(out_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('token_path', help='prompt')
    args = parser.parse_args()

    convert(args.token_path)
