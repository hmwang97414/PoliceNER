import random
from os.path import join
from codecs import open
import numpy as np
from loadEmbedding import pretrainedEmbedding


# # 读取npz文件中保存的词向量
# embedding = np.load('pretrained_embedding.npz', allow_pickle=True)
# # 嵌入的维度
# embedding_dim = embedding['embedding_dim']
# # w2i,i2w
# vocab = embedding['vocab'][()]
# # 嵌入矩阵
# matrix = embedding['matrix']
# # 词典大小
# vocab_size = len(vocab["w2i"])
# # 词典
# word2idx = vocab["w2i"]


def build_corpus(split, make_vocab=True, data_dir="./data/Police2000"):
    """读取数据"""
    assert split in ['train', 'dev', 'test_tag', 'test']
    word_lists = []
    tag_lists = []
    with open(join(data_dir, split + ".txt"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f.readlines():
            try:
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            except:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        # 换成百度百科训练的向量
        word2id = pretrainedEmbedding.word2idx
        tag2id = build_map(tag_lists)
        word2id['<unk>'] = len(word2id)
        word2id['<pad>'] = len(word2id)
        tag2id['<unk>'] = len(tag2id)
        tag2id['<pad>'] = len(tag2id)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def get_preTag(split, data_dir="./data/Police2000"):
    """读取数据"""
    assert split in ['train_tag', 'dev_tag', 'test_tag']
    tag_lists = []
    with open(join(data_dir, split + ".txt"), 'r', encoding='utf-8') as f:
        tag_list = []
        for line in f.readlines():
            try:
                word, tag = line.strip('\n').strip("\r").split("\t")
                tag_list.append(tag)
            except:
                tag_lists.append(tag_list)
                tag_list = []
    return tag_lists

def get_pinyin(split, data_dir="./data/Police2000"):
    assert split in ['train_pinyin', 'dev_pinyin', 'test_pinyin']
    pinyin_lists = []
    with open(join(data_dir, split + ".txt"), 'r', encoding='utf-8') as f:
        pinyin_list = []
        for line in f.readlines():
            try:
                word, pinyin = line.strip('\n').strip("\r").split("\t")

                pinyin_list.append(pinyin)
            except:
                pinyin_lists.append(pinyin_list)
                pinyin_list = []
    pinyin2id = build_map(pinyin_lists)
    pinyin2id['<unk>'] = len(pinyin2id)
    pinyin2id['<pad>'] = len(pinyin2id)

    return pinyin_lists, pinyin2id




def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps


def sort_by_lengths(word_lists, tag_lists):
    pairs = list(zip(word_lists, tag_lists))
    # pairs = [[word0,tag0],[word1,tag1],[word2,tag2]...]
    # 根据word降序排序
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]
    # pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    word_lists, tag_lists = list(zip(*pairs))

    return word_lists, tag_lists, indices


if __name__ == '__main__':
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    print(train_word_lists)
    print(train_tag_lists)
    #
    # word_lists, tag_lists, indices = sort_by_lengths(train_word_lists,train_tag_lists)
    # print(word_lists)
    # print(tag_lists)
    # print(indices)

    print(tag2id)
