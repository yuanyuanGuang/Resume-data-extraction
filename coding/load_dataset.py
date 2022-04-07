
import os
from functools import partial
import paddle
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from paddlenlp.metrics import ChunkEvaluator
from utils import load_dict


# 使用序列模型完成快递单信息抽取任务
def download_dataset():
    """
        下载快递单数据集
    """
    from paddle.utils.download import get_path_from_url
    URL = "https://paddlenlp.bj.bcebos.com/paddlenlp/datasets/waybill.tar.gz"
    get_path_from_url(URL, "./")

data_file = "./data/"
# print(os.path.exists(data_path))
if not os.path.exists(data_file):
    # 如果数据集不存在则下载
    download_dataset()
    print("Download Successful !!!")


# 使用MapDataset()自定义数据集。
def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, "r", encoding="utf-8") as fp:
            next(fp)   # 跳过最开始的header
            for line in fp.readlines():
                texts, labels = line.strip('\n').split('\t')
                texts = texts.split('\002')
                labels = labels.split('\002')
                yield texts, labels

    if isinstance(datafiles,str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles,list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


# Create dataset, tokenizer and dataloader.
train_ds, dev_ds, test_ds = load_dataset(datafiles=(
        './data/train.txt', './data/dev.txt', './data/test.txt'))


# for i in range(3):
#     print(f"text:{train_ds[i][0]}\tlabel:{train_ds[i][1]}")

label_vocal = load_dict('./data/tag.dic')
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')


