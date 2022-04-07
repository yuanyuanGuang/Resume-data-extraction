

from re import I


def load_dict(dict_path):
    vocab = {}
    i = 0
    with open(dict_path, "r", encoding="utf-8") as fp:
        for line in fp.readlines():
            key = line.strip("\n")
            vocab[key] = i
            i += 1
        return vocab


