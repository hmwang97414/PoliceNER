import numpy as np


class pretrainedEmbedding(object):
    # 读取npz文件中保存的词向量
    embedding = np.load('pretrained_embedding.npz', allow_pickle=True)
    # 嵌入的维度
    embedding_dim = embedding['embedding_dim']
    # w2i,i2w
    vocab = embedding['vocab'][()]
    # 嵌入矩阵
    matrix = embedding['matrix']
    # 词典大小
    vocab_size = len(vocab["w2i"])
    # 词典
    word2idx = vocab["w2i"]
