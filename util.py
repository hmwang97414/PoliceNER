import torch
import torch.nn.functional as F


def getPreTag(filename):
    file = open(filename, "r", encoding="utf-8")
    lists = []
    list = []
    for lines in file.readlines():
        if lines != "\n":
            lines = lines.split("\t")
            list.append(lines[1])
        else:
            lists.append(list)
            list = []
    return lists


def tensorized(batch, maps):
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')
    # tests_word = ["在北航被盗钱包"]
    # 每个batch的长度以这一批次的最大长度为准
    max_len = len(batch[0])
    batch_size = len(batch)

    # 初始化一个batch_tensor，用于存放数字表示的输入
    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            batch_tensor[i][j] = maps.get(e, UNK)
    # batch各个元素的长度
    lengths = [len(l) for l in batch]

    return batch_tensor, lengths


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


def sort_by_lengths_preTag(word_lists, tag_lists, pre_tag, position_tag):
    pairs = list(zip(word_lists, tag_lists, pre_tag, position_tag))
    # pairs = [[word0,tag0],[word1,tag1],[word2,tag2]...]
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]
    # pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    word_lists, tag_lists, pre_tag, position_tag = list(zip(*pairs))

    return word_lists, tag_lists, pre_tag, position_tag, indices


def sort_lists(word_lists):
    #     word_lists = list(word_lists)
    indices = sorted(range(len(word_lists)), key=lambda k: len(word_lists[k]), reverse=True)
    list = [word_lists[i] for i in indices]
    return list



def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list


def cal_loss(logits, targets, tag2id):

    PAD = tag2id.get('<pad>')
    assert PAD is not None

    mask = (targets != PAD)  # [B, L]
    targets = targets[mask]
    out_size = logits.size(2)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, out_size)
    ).contiguous().view(-1, out_size)

    assert logits.size(0) == targets.size(0)
    loss = F.cross_entropy(logits, targets)

    return loss