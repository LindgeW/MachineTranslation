from collections import Counter
from config.Const import *
import os
import numpy as np


# 语言词表
class LangVocab(object):
    def __init__(self, wd_counter: Counter, min_count=5):
        super(LangVocab, self).__init__()
        self.pad = 0
        self.unk = 1
        self.bos = 2
        self.eos = 3

        self._wd2freq = dict((wd, freq) for wd, freq in wd_counter.items() if freq >= min_count)

        self._wd2idx = {
            PAD: self.pad,
            UNK: self.unk,
            BOS: self.bos,
            EOS: self.eos
        }

        for wd in self._wd2freq.keys():
            if wd not in self._wd2idx:
                self._wd2idx[wd] = len(self._wd2idx)
        self._idx2wd = dict((idx, wd) for wd, idx in self._wd2idx.items())

        self._extwd2idx = dict()
        self._extidx2wd = dict()

    def get_embedding_weights(self, embed_path):
        if not os.path.exists(embed_path):
            return None

        vec_size = 0
        wd_tab = dict()
        with open(embed_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                tokens = line.strip().split(' ')
                wd, vec = tokens[0], tokens[1:]
                if vec_size == 0:
                    vec_size = len(vec)
                wd_tab[wd] = np.asarray(vec, dtype=np.float32)
        print('embed dim:', vec_size)

        oov = 0
        for wd in self._wd2idx.keys():
            if wd not in wd_tab:
                oov += 1
        print('oov ratio: %.3f%%' % (100 * oov / len(self._wd2idx)))

        self._extwd2idx = {
            PAD: self.pad,
            UNK: self.unk,
            BOS: self.bos,
            EOS: self.eos
        }

        for wd in wd_tab.keys():
            if wd not in self._extwd2idx:
                self._extwd2idx[wd] = len(self._extwd2idx)
        self._extidx2wd = dict((idx, wd) for wd, idx in self._extwd2idx.items())

        vocab_size = len(self._extwd2idx)
        embed_weights = np.zeros((vocab_size, vec_size), dtype=np.float32)
        for idx, wd in self._extidx2wd.items():
            if wd in wd_tab:
                embed_weights[idx] = wd_tab[wd]

        if BOS not in wd_tab:
            embed_weights[self.bos] = np.random.uniform(-0.35, 0.35, vec_size)
        if EOS not in wd_tab:
            embed_weights[self.eos] = np.random.uniform(-0.35, 0.35, vec_size)

        embed_weights[self.unk] = np.mean(embed_weights, 0) / np.std(embed_weights)
        print(embed_weights[:4])

        return embed_weights

    @property
    def exits_pretrain(self):
        return len(self._extwd2idx) != 0

    @property
    def vocab_size(self):
        return len(self._extwd2idx) if self.exits_pretrain else len(self._wd2idx) 

    def word2index(self, wds):
        assert wds is not None
        wd2idx = self._extwd2idx if self.exits_pretrain else self._wd2idx
        if isinstance(wds, list):
            return [wd2idx.get(wd, self.unk) for wd in wds]
        else:
            return wd2idx.get(wds, self.unk)

    def index2word(self, idxs):
        assert idxs is not None
        idx2wd = self._extidx2wd if self.exits_pretrain else self._idx2wd
        if isinstance(idxs, list):
            return [idx2wd.get(i, UNK) for i in idxs]
        else:
            return idx2wd.get(idxs, UNK)


# # 创建语料词表
# def create_vocab(token_lst: list) -> LangVocab:
#     wd_counter = Counter()
#     for tokens in token_lst:
#         wd_counter.update(tokens)
#
#     return LangVocab(wd_counter)
