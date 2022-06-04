import gc
import logging
import sys
import time

import torch

from data.data_preprocess import build_corpus, toList, get_preTag, get_pinyin
from train import evaluate_BiLSTM_CRF
from evaluating import Metrics
import os

from util import sort_lists, getPreTag, sort_by_lengths


def eval_model(model_name):
    # train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    # dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    # # 真实标签
    # test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)
    # # 预识别的的tag
    # pre_test_word_lists, pre_test_tag_lists = build_corpus("test_tag", make_vocab=False)
    # print(pre_test_word_lists)
    # print(pre_test_tag_lists)

    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    train_pre_tag, train_position_tag, position2id = get_preTag("train_id_tag")
    dev_pre_tag, dev_position_tag, _ = get_preTag("dev_id_tag")
    test_pre_tag, test_position_tag, _ = get_preTag("test_id_tag")

    train_pinyin, pinyin2id = get_pinyin("train_pinyin")
    dev_pinyin, _ = get_pinyin("dev_pinyin")
    test_pinyin, _ = get_pinyin("test_pinyin")


    # preTag = True

    model = evaluate_BiLSTM_CRF(word2id, tag2id, pinyin2id)

    # 开始训练模型
    print("训练模型:", model_name)
    print("模型参数")
    print("epoch:", model.epoch)
    print("batch_size:", model.batch_size)
    print("learning_rate:", model.learning_rate)

    model.train(train_word_lists, train_tag_lists, train_pinyin, train_pre_tag, dev_word_lists, dev_tag_lists,
                dev_pre_tag, dev_pinyin)
    # 没有预识别的tag
    # model.train(train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists)
    print("开始测试")
    # test_tag_lists, pred_tag_lists = model.test(test_word_lists, test_tag_lists)
    test_tag_lists, pred_tag_lists = model.test(test_word_lists, test_tag_lists, test_pre_tag, test_pinyin)
    print("预测标签")
    print(pred_tag_lists)
    metrics = Metrics(test_tag_lists, pred_tag_lists, True)
    print("模型结果")
    p, r, f = metrics.report_scores()
    return p, r, f


if __name__ == '__main__':

    p, r, f = eval_model("bilstm_crf")
    print("p:", p)
    print("r:", r)
    print("f1:", f)
