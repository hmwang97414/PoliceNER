from copy import deepcopy

import torch.nn.utils

from model.BiLSTM_CRF import BiLSTM_CRF
from util import sort_by_lengths, tensorized, cal_loss, sort_lists, sort_by_lengths_preTag
from torch.optim import Adam
from config import config
import torch.nn as nn

import numpy as np
from loadEmbedding import pretrainedEmbedding

word2idx = pretrainedEmbedding.word2idx
embedding_dim = pretrainedEmbedding.embedding_dim.tolist()


class evaluate_BiLSTM_CRF(object):
    def __init__(self, word2id, tag2id, pinyin2id):
        # print(type(word2idx))
        self.word2id = word2idx
        self.tag2id = tag2id
        self.pinyin2id = pinyin2id
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.epoch = config.epoch
        self._best_val_loss = 1e18
        self.best_model = None
        # print(type(embedding_dim))
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        # print(type(self.device))
        self.model = BiLSTM_CRF(len(self.word2id), char_embedding_dim=embedding_dim,
                                tag_embedding_dim=config.tag_embedding_dim,
                                pinyin_embedding_dim=config.pinyin_embedding_dim,
                                hidden_size=config.hidden_size,
                                pinyin=len(self.pinyin2id),
                                num_tags=len(self.tag2id), device=self.device).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, train_word_lists, train_tag_lists, train_pinyin, train_pre_tag, dev_word_lists, dev_tag_lists,
              dev_pre_tag, dev_pinyin):
        # print(train_position_tag)
        # 1.将word_lists按照长度排序
        train_lists, train_tags, train_pre_tag, train_pinyin, _ = sort_by_lengths_preTag(train_word_lists,
                                                                                         train_tag_lists,
                                                                                         train_pre_tag,
                                                                                         train_pinyin)
        dev_lists, dev_tags, dev_pre_tag, dev_pinyin, _ = sort_by_lengths_preTag(dev_word_lists, dev_tag_lists,
                                                                                 dev_pre_tag, dev_pinyin)

        # print(word_lists)
        # print(tag_lists)

        # model.train()
        for i in range(self.epoch):
            self.model.train()
            step = 0
            loss = 0
            train_lists = [x for x in train_lists if len(x) >= 1]
            train_tags = [x for x in train_tags if len(x) >= 1]
            # 预识别的tag
            pre_tags = [x for x in train_pre_tag if len(x) >= 1]
            pinyin_tags = [x for x in train_pinyin if len(x) >= 1]
            total_step = len(train_lists) // self.batch_size + 1
            for j in range(0, len(train_lists), self.batch_size):
                step += 1
                # 每一个batch的数据和tag
                batch_sents = train_lists[j:j + self.batch_size]
                batch_tags = train_tags[j:j + self.batch_size]
                batch_pre_tags = pre_tags[j:j + self.batch_size]
                batch_pinyin = pinyin_tags[j:j + self.batch_size]
                # 设置为相同长度，不足的补pad,并转为tensor
                # print(batch_sents)
                word_lists_tensor, lengths = tensorized(batch_sents, self.word2id)
                tag_lists_tensor, lengths = tensorized(batch_tags, self.tag2id)
                pre_tag_tensor, lengths = tensorized(batch_pre_tags, self.tag2id)
                pinyin_tensor, lengths = tensorized(batch_pinyin, self.pinyin2id)
                # print(word_lists_tensor.shape)
                # print("===========================")
                self.optimizer.zero_grad()
                # print(word_lists_tensor)
                # print(word_lists_tensor.shape)
                # [batch_size,seq_len]
                # mask = (x != sent_vocab.stoi["<pad>"])

                mask = (word_lists_tensor != self.word2id['<pad>'])
                # print(mask.shape)
                # print(mask)
                word_lists_tensor = word_lists_tensor.to(self.device)
                tag_lists_tensor = tag_lists_tensor.to(self.device)
                pre_tag_tensor = pre_tag_tensor.to(self.device)
                pinyin_tensor = pinyin_tensor.to(self.device)
                mask = mask.to(self.device)
                # print("执行到这里了")
                loss = self.model(word_lists_tensor, tag_lists_tensor, pre_tag_tensor, pinyin_tensor, mask).to(
                    self.device)
                loss.backward()
                self.optimizer.step()
                # if step % 10 == 0:
                # print(
                #     "Epoch {},step/total_step: {}/{} Loss:{:.4f}".format(i + 1, step, total_step, loss.item() / 10))
            # 每轮结束测试在验证集上的性能，保存最好的一个
            val_loss = self.validate(dev_lists, dev_tags, dev_pre_tag, dev_pinyin)
            # print("Epoch {}, Val Loss:{:.4f}".format(i + 1, val_loss))

    def validate(self, dev_word_lists, dev_tag_lists, dev_pre_tag, dev_pinyin):
        # 用验证集测试模型的效果
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1
                # 准备batch数据
                batch_sents = dev_word_lists[ind:ind + self.batch_size]
                batch_tags = dev_tag_lists[ind:ind + self.batch_size]
                batch_pre_tags = dev_pre_tag[ind:ind + self.batch_size]
                batch_pinyin_tags = dev_pinyin[ind:ind + self.batch_size]
                # 张量化，
                tensorized_sents, lengths = tensorized(batch_sents, self.word2id)
                targets, lengths = tensorized(batch_tags, self.tag2id)
                tensorized_pre_tags, lengths = tensorized(batch_pre_tags, self.tag2id)

                tensorized_pinyin_tags, lengths = tensorized(batch_pinyin_tags, self.pinyin2id)
                # forward
                mask = (tensorized_sents != self.word2id['<pad>'])
                # print(mask.shape)
                # print(mask)
                tensorized_sents = tensorized_sents.to(self.device)
                tensorized_pre_tags = tensorized_pre_tags.to(self.device)
                tensorized_pinyin_tags = tensorized_pinyin_tags.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)

                loss = self.model(tensorized_sents, targets, tensorized_pre_tags, tensorized_pinyin_tags, mask).to(
                    self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self._best_val_loss:
                print("保存模型...")
                torch.save(self.model, "saved_model/bilstm_crf.pth")
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss

            return val_loss

    def test(self, test_word_lists, test_tag_lists, test_pre_tag, test_pinyin):
        test_lists, test_tags, test_pre_tag, test_pinyin, indices = sort_by_lengths_preTag(test_word_lists,
                                                                                           test_tag_lists,
                                                                                           test_pre_tag,
                                                                                           test_pinyin)
        tests_word_tensor, lengths = tensorized(test_lists, self.word2id)
        test_pre_tag, lengths = tensorized(test_pre_tag, self.tag2id)
        test_pinyin_tag, lengths = tensorized(test_pinyin, self.pinyin2id)

        id2tag = dict((id_, tag) for tag, id_ in self.tag2id.items())
        # 用best_model进行预测
        self.best_model.eval()
        # self.model.load_state_dict(torch.load("saved_model/bilstm_crf.pth"))
        # self.model.eval()
        with torch.no_grad():
            mask = (tests_word_tensor != self.word2id['<pad>'])
            tests_word_tensor = tests_word_tensor.to(self.device)
            pre_tag_tensor = test_pre_tag.to(self.device)
            test_pinyin_tensor = test_pinyin_tag.to(self.device)

            mask = mask.to(self.device)
            pred = self.best_model.predict(tests_word_tensor, pre_tag_tensor, test_pinyin_tensor, mask)
        # print(lengths)
        pred_tag_lists = []
        for i, ids in enumerate(pred):
            tag_list = []
            for j in range(lengths[i]):
                tag_list.append(id2tag[ids[j]])
            pred_tag_lists.append(tag_list)

        # print("原始文本:", test_lists[121])
        # print("测试数据真实标签：", test_tags[121])
        # print("预测标签：", pred_tag_lists[121])
        # print(len(test_tags))
        # print(len(pred_tag_lists))

        return test_tags, pred_tag_lists
