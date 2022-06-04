import torch.nn as nn
from torch.autograd import Variable
from torchcrf import CRF
import torch
import numpy as np
from loadEmbedding import pretrainedEmbedding

# 嵌入矩阵
embedding_dim = pretrainedEmbedding.embedding_dim
matrix = pretrainedEmbedding.matrix
# 补充<pad>和<unk>两行嵌入矩阵
pad = np.zeros((2, embedding_dim))
new_matrix = np.row_stack((matrix, pad))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, char_embedding_dim, tag_embedding_dim, pinyin_embedding_dim, pinyin, hidden_size,
                 num_tags, device):
        super(BiLSTM_CRF, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embedding_dim).to(self.device)
        self.tag_embed = nn.Embedding(num_tags, tag_embedding_dim)
        self.pinyin_embed = nn.Embedding(pinyin, pinyin_embedding_dim)
        weights = torch.from_numpy(new_matrix).cuda()
        self.embed.weight.data.copy_(torch.from_numpy(new_matrix))
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(embedding_dim + pinyin_embedding_dim + tag_embedding_dim, hidden_size, bidirectional=True,
                            batch_first=True)

        self.hidden2tag = nn.Linear(hidden_size * 2, num_tags)
        self.crf = CRF(num_tags=num_tags, batch_first=True)

    # 计算损失loss
    def forward(self, x, y, pre_tags, pinyin_tags, mask=None):
        emissions = self.get_emissions(x, pre_tags, pinyin_tags)
        loss = -self.crf(emissions=emissions, tags=y, mask=mask)
        return loss

    # 找出最可能的标签序列，使用维特比解码
    def predict(self, x, pre_tags, pinyin_tags, mask=None):
        emissions = self.get_emissions(x, pre_tags, pinyin_tags)
        # emissions维度[batch,seq_len, num_tags]
        preds = self.crf.decode(emissions, mask)
        # for i in range(len(preds)):
        #     print(len(preds[i]))
        return preds

    # lstm输出，即发射分数
    def get_emissions(self, x, pre_tags, pinyin_tags):

        batch_size, seq_len = x.shape
        pre_tags = pre_tags.long()
        char_embedded = self.embed(x)
        # 预识别的标签embedding
        tag_embedded = self.tag_embed(pre_tags)

        pinyin_embedded = self.pinyin_embed(pinyin_tags)
        h0, c0 = torch.zeros(2, batch_size, self.hidden_size).to(self.device), torch.zeros(2, batch_size,
                                                                                           self.hidden_size).to(
            self.device)
        char_embedded = self.dropout(char_embedded)
        tag_embedded = self.dropout(tag_embedded)
        pinyin_embedded = self.dropout(pinyin_embedded)
        embedded = torch.cat((char_embedded, pinyin_embedded,tag_embedded), 2)
        out, (_, _) = self.lstm(embedded, (h0, c0))
        out = self.dropout(out)
        emissions = self.hidden2tag(out)
        return emissions
