
import json
from spacy.lang.zh import Chinese
from matplotlib.pyplot import text
from paddlenlp.datasets import MapDataset

def load_dict(dic_path):
    """加载.json中的label,使变成BIO形式

    Args:
        dic_path (str): .json文件的路径

    Returns:
        dict: labels的{}
    """
    print(dic_path)
    vocab = {}
    i = 0
    with open(dic_path, 'r', encoding='utf-8') as fp:
        labels_json = json.loads(fp.read())
        for label in labels_json:
            vocab[label["text"] + "-B"] = i
            i += 1
            vocab[label["text"] + "-I"] = i
            i += 1

        vocab["O"] = i
        print(vocab)
    return vocab


def load_dataset(datafiles):
    """加载数据集

    Args:
        datafile (str):数据集文件的路径
    """
    nlp = Chinese()
    # char
    cfg = {"segmenter": "char"}
    nlp = Chinese.from_config({"nlp": {"tokenizer": cfg}})
    nlp.add_pipe('sentencizer')
    def read(data_path):
        with open(data_path, "r", encoding='utf-8') as fp:
            for line in fp.readlines():
                data_json = json.loads(line)
                text = data_json['data']
                words = list(text)
                labels_list = data_json['label']
                labels = ['O'] * len(words)
                for tag_tuple in labels_list:
                    start, end, tag = tag_tuple
                    labels[start] = tag + "-B"
                    for i in range(start+1, end):
                        if i >= len(labels): # 标注软件有bug导致
                            print(i, len(labels), len(text), tag_tuple, data_json)
                            break
                        labels[i] = tag + "-I"
                doc = nlp(text)
                for s in doc.sents:
                    # print(s, s.start, s.end)
                    yield words[s.start:s.end], labels[s.start:s.end]
                # break
    
    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


def parse_decodes(sentences, predictions, lengths, label_vocab):
    predictions = [x for batch in predictions for x in batch]
    lengths = [x for batch in lengths for x in batch]
    
    # 将label_vocab={keys:values} ==> id_label={values:key}
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))

    outputs = []
    for idx, end in enumerate(lengths):
        sent = sentences[idx][:end]
        tags = [id_label[x] for x in predictions[idx][:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in list(zip(sent, tags)):
            if t.endswith('-B') or t == 'O':
                if len(words):
                    sent_out.append(words)
                tags_out.append(t.split('-')[0])
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs

# label_vocab = load_dict("./dataset/label_config.json")
# print(type(label_vocab), label_vocab)
# id_label = dict(zip(label_vocab.values(), label_vocab.keys()))
# print(id_label)
# train_dataset = load_dataset('coding/dataset/train.jsonl')
# # print(test_dataset.data[0][0])

# print('data: ',train_dataset[0][0])
# print('label: ',train_dataset[0][1])