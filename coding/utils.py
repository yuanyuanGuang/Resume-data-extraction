from cProfile import label
import json
import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad

def load_dict(dict_path):
    """获取label对应的取值

    Args:
        dict_path (str): 文件路径 eg:./dataset/label_config.json

    Returns:
        dict: {"label":1,......}
    """
    vocab = {}
    with open(dict_path, "r", encoding="utf-8") as fp:
        labels_json = json.loads(fp.read()) # type(labels_json)==list [{"id": 19,"text": "EDUCATIONAL INSTITUTE",    "prefixKey": null,"suffixKey": "e","backgroundColor": "#209CEE","textColor": "#ffffff"}]
        i = 0
        for label_dic in labels_json:
            key1 = label_dic['text'] + '-B'
            key2 = label_dic['text'] + '-I'
            vocab[key1] = i
            i += 1
            vocab[key2] = i
            i += 1
        
        vocab['O'] = i
        return vocab

def convert_example(example, tokenizer, label_vocal):
    tokens, labels = example
    tokenized_input = tokenizer(
        tokens, return_length=True, is_split_into_words=True
    )
    # Token '[CLS]' and '[SEP]' will get label 'O'
    labels = ['O'] + labels + ['O']
    tokenized_input['labels'] = [label_vocal[x] for x in labels]
    return tokenized_input['input_ids'], tokenized_input['token_type_ids'],\
            tokenized_input['seq_len'], tokenized_input['labels']


