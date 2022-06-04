import gc
import logging
import sys
import time

import torch

from data.data_preprocess import build_corpus, get_preTag, get_pinyin
from train import evaluate_BiLSTM_CRF
from evaluating import Metrics
import os

from util import sort_lists, getPreTag, sort_by_lengths


def eval_model(model_name):

    # get data
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # get pre_recognized labels
    train_pre_tag = get_preTag("train_tag")
    dev_pre_tag= get_preTag("dev_tag")
    test_pre_tag = get_preTag("test_tag")
    # get phonetic 'pinyin'
    train_pinyin, pinyin2id = get_pinyin("train_pinyin")
    dev_pinyin, _ = get_pinyin("dev_pinyin")
    test_pinyin, _ = get_pinyin("test_pinyin")

    model = evaluate_BiLSTM_CRF(word2id, tag2id, pinyin2id)

    # begin training
    print("begin training")
    print("model name:", model_name)
    print("parameters")
    print("epoch:", model.epoch)
    print("batch_size:", model.batch_size)
    print("learning_rate:", model.learning_rate)

    model.train(train_word_lists, train_tag_lists, train_pinyin, train_pre_tag, dev_word_lists, dev_tag_lists,
                dev_pre_tag, dev_pinyin)
    print("begin testing")
    test_tag_lists, pred_tag_lists = model.test(test_word_lists, test_tag_lists, test_pre_tag, test_pinyin)
    print("predicting")
    print(pred_tag_lists)
    metrics = Metrics(test_tag_lists, pred_tag_lists, True)
    print("result")
    p, r, f = metrics.report_scores()
    return p, r, f


if __name__ == '__main__':

    p, r, f = eval_model("bilstm_crf")
    print("p:", p)
    print("r:", r)
    print("f1:", f)
