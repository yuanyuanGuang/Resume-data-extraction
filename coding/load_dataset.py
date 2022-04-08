
from cProfile import label
import os
from functools import partial
from matplotlib.pyplot import axis
import paddle
from paddle.io import DataLoader, BatchSampler
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from paddlenlp.metrics import ChunkEvaluator
from utils import load_dict, convert_example

import json



# 获取当前.py所在目录
current_file_dirname = os.path.dirname(os.path.abspath(__file__))
# print(current_file_dirname)
'''
# 使用序列模型完成快递单信息抽取任务
def download_dataset():
    """
        下载快递单数据集
    """
    from paddle.utils.download import get_path_from_url
    URL = "https://paddlenlp.bj.bcebos.com/paddlenlp/datasets/waybill.tar.gz"
    get_path_from_url(URL, current_file_dirname)

data_file = current_file_dirname +"/data/"  # 下载数据集放在与当前.py文件相同目录下的data文件中
# print(os.path.exists(data_path))
if not os.path.exists(data_file):
    # 如果数据集不存在则下载
    download_dataset()
    print("Download Successful !!!")

'''

# 使用MapDataset()自定义数据集。
def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                data_json = json.loads(line) # 读取得json文件的数据转化为dict的对象,{'id':'1','data':"xxxxx",'label':[['xxx','xxxx','xxxx'], ['xxx','xxxx','xxxx'],......]}
                
                text = data_json['data']
                label_list = data_json['label']
                words = list(text)
                labels = ['O']*len(words)  # 将所有label初始化为['O','O','O',......]
                for label in label_list:
                    start, end, tag = label
                    labels[start] = tag + '-B'
                    # print(len(words),end)
                    for i in range(start+1,end):
                        if i >= len(labels): # 标注软件有bug导致
                            print(i, len(labels), len(text), label, data_json)
                            break
                        labels[i] = tag + '-I'
                yield words, labels

    if isinstance(datafiles,str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles,list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]

# Create dataset, tokenizer and dataloader.
train_ds, dev_ds, test_ds = load_dataset(datafiles=(
        current_file_dirname + '/dataset/train.jsonl', current_file_dirname + '/dataset/admin.jsonl', current_file_dirname + '/dataset/admin.jsonl'))

# for i in range(3):
#     print(train_ds[i])
#     # print(f"text:{train_ds[i][0]}\tlabel:{train_ds[i][1]}")

label_vocal = load_dict(current_file_dirname + '/dataset/label_config.json')
print(label_vocal)
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

# 偏函数
trans_func = partial(convert_example, tokenizer=tokenizer, label_vocal=label_vocal)

# print(train_ds[0])

train_ds.map(trans_func)
dev_ds.map(trans_func)
test_ds.map(trans_func)

print(train_ds[0])
'''
""" 
    数据读入：
        数据批处理
        使用paddle.io.DataLoader接口多线程异步加载数据。 
"""
ignore_label = -1

batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id), # token_type_ids
    Stack(), # seq_len
    Pad(axis=0, pad_val=ignore_label) # label
): fn(samples)

train_batch_sample = BatchSampler(train_ds, batch_size=36, shuffle=True)

train_data_loader = DataLoader(dataset=train_ds, 
                                batch_sampler=train_batch_sample, 
                                return_list=True,
                                collate_fn=batchify_fn)
dev_data_loader = DataLoader(dataset=dev_ds,
                                batch_sampler=train_batch_sample, 
                                return_list=True,
                                collate_fn=batchify_fn)
test_data_loader = DataLoader(dataset=test_ds,
                                batch_sampler=train_batch_sample, 
                                return_list=True,
                                collate_fn=batchify_fn)

print(type(train_data_loader))

'''
