import numpy as np


class pretrainedEmbedding(object):
    embedding = np.load('pretrained_embedding.npz', allow_pickle=True)
    embedding_dim = embedding['embedding_dim']
    vocab = embedding['vocab'][()]
    matrix = embedding['matrix']
    vocab_size = len(vocab["w2i"])
    word2idx = vocab["w2i"]
