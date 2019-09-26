import os
import jieba
import numpy as np
import torch
from config.Const import *
from vocab.Vocab import LangVocab
from collections import Counter


# 加载语料
def load_data(path) -> list:
    assert os.path.exists(path)
    sents = []
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if line != '':
                sents.append(line)
    return sents


# 处理语料
def tokenize(sent, need_cut=False):
    if need_cut:
        tokens = jieba.lcut(sent)
    else:
        tokens = sent.split(' ')

    return [token for token in tokens if token != '']


def prepare_data(src_corpus, tgt_corpus, reverse=False, exchange=False):
    pairs = []
    src_counter = Counter()
    tgt_counter = Counter()

    if exchange:  # 互换翻译顺序
        src_corpus, tgt_corpus = tgt_corpus, src_corpus

    for src_sent, tgt_sent in zip(src_corpus, tgt_corpus):
        src_tokens, tgt_tokens = tokenize(src_sent), tokenize(tgt_sent)
        if reverse:  # 逆置source inputs
            src_tokens = list(reversed(src_tokens))

        src_tokens.append(EOS)
        tgt_tokens.insert(0, BOS)
        tgt_tokens.append(EOS)
        pairs.append((src_tokens, tgt_tokens))
        src_counter.update(src_tokens)
        tgt_counter.update(tgt_tokens)

    print('prepare_data:', pairs[:2])
    return pairs, LangVocab(src_counter), LangVocab(tgt_counter)


class Batch(object):
    def __init__(self, src, tgt=None, non_pad_mask=None):
        self.src_idxs = src  # 输入值
        self.tgt_idxs = tgt  # 标签值
        self.non_pad_mask = non_pad_mask


# paris: [(src1, tgt1), (src2, tgt2), ..., (srcn, tgtn)]
def batch_iter(pairs, args, src_vocab, tgt_vocab, shuffle=False):
    if shuffle:
        np.random.shuffle(pairs)

    batch_size = args.batch_size
    nb_batch = int(np.ceil(len(pairs) / batch_size))

    for i in range(nb_batch):
        batch_data = pairs[i*batch_size: (i+1)*batch_size]
        # 将source inputs的长度进行降序排列
        batch_data = sorted(batch_data, key=lambda k: len(k[0]), reverse=True)

        yield batch_variable(batch_data, src_vocab, tgt_vocab, args.device)


# 词 -> 索引
def batch_variable(batch_pairs, src_vocab, tgt_vocab, device=torch.device('cpu')):
    src_batch, tgt_batch = zip(*batch_pairs)
    assert len(src_batch) == len(tgt_batch)
    src_max_len = max(len(src_seq) for src_seq in src_batch)
    tgt_max_len = max(len(tgt_seq) for tgt_seq in tgt_batch)

    batch_size = len(batch_pairs)
    src_wd_idxs = torch.zeros((batch_size, src_max_len), dtype=torch.long, device=device)
    tgt_wd_idxs = torch.zeros((batch_size, tgt_max_len), dtype=torch.long, device=device)

    for i, (src_seq, tgt_seq) in enumerate(batch_pairs):
        src_wd_idxs[i, :len(src_seq)] = torch.tensor(src_vocab.word2index(src_seq), device=device)
        tgt_wd_idxs[i, :len(tgt_seq)] = torch.tensor(tgt_vocab.word2index(tgt_seq), device=device)

    src_mask = src_wd_idxs.ne(src_vocab.pad).float()
    tgt_mask = tgt_wd_idxs.ne(tgt_vocab.pad).float()

    tgt_wd_idxs.transpose_(0, 1)
    tgt_mask.transpose_(0, 1)

    return Batch(src_wd_idxs, non_pad_mask=src_mask), \
           Batch(tgt_wd_idxs, non_pad_mask=tgt_mask)







