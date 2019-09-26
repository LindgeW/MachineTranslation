from datautil.dataloader import batch_iter
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nn_utils
import time
import torch
import numpy as np
from config.Const import *


class NMT(object):
    def __init__(self, encoder, decoder):
        super(NMT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def summary(self):
        print('encoder:', self.encoder)
        print('decoder:', self.decoder)

    # 训练一轮
    def train(self, train_pairs, enc_optimizer, dec_optimizer, args, src_vocab, tgt_vocab):
        train_loss = 0
        for src_batch, tgt_batch in batch_iter(train_pairs, args, src_vocab, tgt_vocab):
            loss = 0
            # enc_out: (batch_size, seq_len, hidden_size * nb_directions)
            # enc_hidden: (num_layers * nb_directions, batch_size, hidden_size)
            enc_out, enc_hidden = self.encoder(src_batch.src_idxs, mask=src_batch.non_pad_mask)

            self.encoder.zero_grad()
            self.decoder.zero_grad()

            dec_hidden = enc_hidden
            dec_input = tgt_batch.src_idxs[0].unsqueeze(1)
            if np.random.uniform(0, 1) <= args.teacher_force:
                # print('以目标作为下一个输入')
                for i in range(1, tgt_batch.src_idxs.size(0)):
                    dec_out, dec_hidden = self.decoder(dec_input, dec_hidden, enc_out)
                    dec_hidden *= tgt_batch.non_pad_mask[i].unsqueeze(1).repeat(1, dec_hidden.size(-1))
                    loss += self.calc_loss(dec_out, tgt_batch.src_idxs[i])
                    train_loss += loss.data.item()

                    dec_input = tgt_batch.src_idxs[i].unsqueeze(1)
            else:
                # print('以网络的预测输出作为下一个输入')
                for i in range(1, tgt_batch.src_idxs.size(0)):
                    dec_out, dec_hidden = self.decoder(dec_input, dec_hidden, enc_out)
                    dec_hidden *= tgt_batch.non_pad_mask[i].unsqueeze(1).repeat(1, dec_hidden.size(-1))
                    loss += self.calc_loss(dec_out, tgt_batch.src_idxs[i])
                    train_loss += loss.data.item()

                    _, top_i = dec_out.data.topk(1)
                    dec_input = top_i  # (batch_size, 1)

            loss.backward()

            nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.encoder.parameters()), max_norm=5.0)
            nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.decoder.parameters()), max_norm=5.0)

            enc_optimizer.step()
            dec_optimizer.step()

        return train_loss / len(train_pairs)

    # 训练多轮
    def train_iter(self, train_pairs, args, src_vocab, tgt_vocab):
        self.encoder.train()
        self.decoder.train()
        enc_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.encoder.parameters()), lr=args.lr)
        dec_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.decoder.parameters()), lr=args.lr)
        enc_lr_scheduler = optim.lr_scheduler.LambdaLR(enc_optimizer, lambda ep: max(0.95**ep, 1e-4))
        dec_lr_scheduler = optim.lr_scheduler.LambdaLR(dec_optimizer, lambda ep: max(0.95**ep, 1e-4))
        # enc_lr_scheduler = optim.lr_scheduler.LambdaLR(enc_optimizer, lambda ep: max(1 - 0.75 * ep / args.epoch, 1e-4))
        # dec_lr_scheduler = optim.lr_scheduler.LambdaLR(dec_optimizer, lambda ep: max(1 - 0.75 * ep / args.epoch, 1e-4))

        for i in range(args.epoch):
            enc_lr_scheduler.step()
            dec_lr_scheduler.step()
            t1 = time.time()
            train_loss = self.train(train_pairs, enc_optimizer, dec_optimizer, args, src_vocab, tgt_vocab)
            t2 = time.time()
            print('[Epoch %d] train loss: %.3f' % (i+1, train_loss))
            print('encoder lr:', enc_lr_scheduler.get_lr())
            print('decoder lr:', dec_lr_scheduler.get_lr())
            print('time cost: %.2fs' % (t2 - t1))

    def calc_loss(self, pred, tgt):
        return F.nll_loss(pred, tgt, ignore_index=0)

    # def evaluate(self, test_pairs, args, src_vocab, tgt_vocab):
    #     self.encoder.eval()
    #     self.decoder.eval()
    #     pred_wds, tgt_wds = [], []
    #     for src_batch, tgt_batch in batch_iter(test_pairs, args, src_vocab, tgt_vocab):
    #         batch_pred_wds, batch_tgt_wds = [], []
    #         enc_out, enc_hidden = self.encoder(src_batch.src_idxs, mask=src_batch.non_pad_mask)
    #
    #         dec_hidden = enc_hidden
    #         dec_input = tgt_batch.src_idxs[0]
    #         for i in range(1, tgt_batch.src_idxs.size(0)):
    #             dec_out, dec_hidden = self.decoder(dec_input, dec_hidden, enc_out)
    #
    #             dec_hidden *= tgt_batch.non_pad_mask[i].unsqueeze(1).repeat(1, dec_hidden.size(-1))
    #             tgt_idxs = tgt_batch.src_idxs[i]
    #             # greedy search
    #             pred_idxs = dec_out.data.argmax(dim=1)
    #             batch_pred_wds.append(tgt_vocab.index2word(pred_idxs.tolist()))
    #             batch_tgt_wds.append(tgt_vocab.index2word(tgt_idxs.tolist()))
    #             dec_input = pred_idxs
    #
    #         pred_wds.extend(self.extract_valid(np.asarray(batch_pred_wds).T.tolist()))
    #         tgt_wds.extend(self.extract_valid(np.asarray(batch_tgt_wds).T.tolist()))
    #
    #     print('BLEU:', self.corpus_bleu(pred_wds, tgt_wds))

    # beam search
    '''
        执行过程：设beam size = 3
        1、选择t1时刻输出的概率分数最大的3个词
        2、分别将t-1时刻选择的3个词作为当前时刻的输入
        3、求t时刻累积的（序列）概率分数(历史所选择词的对数似然和)，选择分数值最大的3个词
        4、重复2-3过程，直到到达最大长度（或遇到<eos>）
    '''
    def evaluate(self, test_pairs, args, src_vocab, tgt_vocab):
        self.encoder.eval()
        self.decoder.eval()
        # pred_wds, tgt_wds = [], []
        for src_batch, tgt_batch in batch_iter(test_pairs, args, src_vocab, tgt_vocab):
            # batch_pred_wds, batch_tgt_wds = [], []
            enc_out, enc_hidden = self.encoder(src_batch.src_idxs, mask=src_batch.non_pad_mask)

            # 保存历史分数
            seq_len, batch_size = tgt_batch.src_idxs.size()
            # (bz, beam_size)
            hist_score = torch.zeros((batch_size, args.beam_size), device=args.device)
            # (beam_size, bz, vocab_size)
            beam_score = torch.zeros((args.beam_size, batch_size, tgt_vocab.vocab_size), device=args.device)
            # (bz, beam_size, max_len)
            best_paths = torch.zeros((MAX_LEN, batch_size, args.beam_size), device=args.device)

            dec_hidden = enc_hidden
            dec_input = tgt_batch.src_idxs[0].unsqueeze(1)
            for i in range(1, min(MAX_LEN, seq_len)):
                if i == 1:
                    # dec_input: (bz, 1)
                    # dec_out: (bz, vocab_size)
                    dec_out, dec_hidden = self.decoder(dec_input, dec_hidden, enc_out)
                    dec_hidden *= tgt_batch.non_pad_mask[i].unsqueeze(1).repeat(1, dec_hidden.size(-1))
                    # (bz, beam_size)
                    top_prob, top_idxs = dec_out.data.topk(args.beam_size, dim=1)
                    hist_score = top_prob
                    best_paths[i] = top_idxs
                    # (bz, beam_size)
                    dec_input = top_idxs
                else:
                    # dec_input: (bz, beam_size) -> (beam_size, bz)
                    dec_input = dec_input.transpose(0, 1)
                    for j in range(args.beam_size):
                        # dec_out: (bz, vocab_size)
                        dec_out, dec_hidden = self.decoder(dec_input[j].unsqueeze(1), dec_hidden, enc_out)
                        dec_hidden *= tgt_batch.non_pad_mask[i].unsqueeze(1).repeat(1, dec_hidden.size(-1))
                        beam_score[j] = dec_out
                    # (bz, beam_size, 1) -> (bz, beam_size, vocab_size)
                    hist_score = hist_score.unsqueeze(-1).expand((-1, -1, tgt_vocab.vocab_size))
                    hist_score += beam_score.transpose(0, 1)  # (bz, beam_size, vocab_size)
                    # (bz, beam_size * vocab_size)
                    hist_score = hist_score.reshape((batch_size, -1))
                    # (bz, beam_size)
                    top_prob, top_idxs = hist_score.topk(args.beam_size, dim=1)
                    hist_score = top_prob
                    top_idxs %= tgt_vocab.vocab_size
                    best_paths[i] = top_idxs
                    dec_input = top_idxs

            # pred_wds.extend(self.extract_valid(np.asarray(batch_pred_wds).T.tolist()))
            # tgt_wds.extend(self.extract_valid(np.asarray(batch_tgt_wds).T.tolist()))

    # 提取序列的非填充部分
    def extract_valid(self, seqs: list):
        return list(map(lambda x: x[:x.index(EOS)] if EOS in x else x, seqs))

    # 统计ngram数目
    def count_ngram(self, cand: list, ref: list, n=1) -> int:
        assert len(cand) != 0 and len(ref) != 0

        total_count = 0
        for i in range(len(cand) - n + 1):
            cand_count, ref_count = 1, 0
            ngram = cand[i: i + n]
            # 统计ngram在机器翻译译文中出现的次数
            for j in range(i + n, len(cand) - n + 1):
                if ngram == cand[j: j + n]:
                    cand_count += 1
            # 统计ngram在人工译文中出现的次数
            for k in range(len(ref) - n + 1):
                if ngram == ref[k: k + n]:
                    ref_count += 1
            total_count += min(cand_count, ref_count)

        return total_count

    # 计算单句话的BLEU值，取值在[0, 1]之间，越大越好
    def sentence_bleu(self, cand: list, ref: list, N=4) -> float:
        '''
        :param cand: sentence_tokens
        :param ref: sentence_tokens
        :return:
        '''
        assert len(cand) != 0 and len(ref) != 0
        # n-gram中n的取值在[1, 4]之间
        res = 0
        cand_len, ref_len = len(cand), len(ref)
        for n in range(1, N+1):
            cand_gram = max(0, cand_len - n + 1)
            res += 0.25 * np.log(self.count_ngram(cand, ref, n) / cand_gram)
        # 短译句惩罚因子
        # bp = np.exp(1 - max(1., len(ref) / len(cand)))
        return np.exp(res + min(0., 1 - ref_len / cand_len))

    # 计算多句话的BLEU值(注:不是直接对sentence bleu求和求平均)
    def corpus_bleu(self, cands: list, refs: list, N=4) -> float:
        '''
        :param cands: [sentence_tokens1, sentence_tokens2]
        :param refs: [sentence_tokens1, sentence_tokens2]
        :return:
        '''
        assert len(cands) != 0 and len(cands) == len(refs)

        ref_len, cand_len = 0, 0
        for cand, ref in zip(cands, refs):
            ref_len += len(ref)
            cand_len += len(cand)

        res = 0
        for n in range(1, N+1):
            n_match, n_grams = 0, 0
            for cand, ref in zip(cands, refs):
                n_match += self.count_ngram(cand, ref, n)
                n_grams += max(0, len(cand) - n + 1)
            res += 0.25 * np.log(n_match / n_grams + 1e-8)

        return np.exp(res + min(0., 1 - ref_len / cand_len))
