import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
'''
编码器：将原输入压缩成固定维度的tensor
'''


class MTEncoder(nn.Module):
    def __init__(self, args, embedding_weights=None):
        super(MTEncoder, self).__init__()

        self.args = args

        if embedding_weights is not None:
            args.src_embedding_dim = embedding_weights.shape[1]
            self.embed_layer = nn.Embedding.from_pretrained(torch.from_numpy(embedding_weights))
        else:
            self.embed_layer = nn.Embedding(num_embeddings=args.src_vocab_size,
                                            embedding_dim=args.src_embedding_dim,
                                            padding_idx=0)
            nn.init.xavier_uniform_(self.embed_layer.weight)
            # nn.init.kaiming_uniform_(self.embed_layer.weight)

        self.embed_drop_layer = nn.Dropout(args.embed_drop)

        self._bidirectional = args.bidirectional
        self.nb_directions = 2 if self._bidirectional else 1

        self.encoder = nn.GRU(input_size=args.src_embedding_dim,
                              hidden_size=args.hidden_size,
                              num_layers=args.num_layers,
                              batch_first=True,
                              dropout=(0. if args.num_layers == 1 else args.rnn_drop),
                              bidirectional=self._bidirectional)

    def _init_hidden(self, batch_size: int, device=torch.device('cpu')):
        h0 = torch.zeros((self.args.num_layers * self.nb_directions, batch_size, self.args.hidden_size),
                         requires_grad=True,
                         device=device)
        return h0

    def forward(self, inputs, init_hidden=None, mask=None):
        '''
        :param inputs:  (bz, seq_len)
        :param mask: (bz, seq_len)
        :param init_hidden:  (nb_layers, bz, hidden_size)
        :return:
        '''
        batch_size = inputs.shape[0]

        # (bz, seq_len, embed_dim)
        embed = self.embed_layer(inputs)

        if self.training:
            embed = self.embed_drop_layer(embed)

        if init_hidden is None:
            init_hidden = self._init_hidden(batch_size, inputs.device)

        # enc_out: (batch_size, seq_len, hidden_size * nb_directions)
        # final_hidden: (num_layers * nb_directions, batch_size, hidden_size)
        input_lens = mask.sum(dim=1).tolist()
        packed_inputs = pack_padded_sequence(embed, lengths=input_lens, batch_first=True)
        rnn_out, final_hidden = self.encoder(packed_inputs, init_hidden)
        enc_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        if self._bidirectional:
            # (batch_size, seq_len, hidden_size)
            fw_rnn, bw_rnn = enc_out.chunk(chunks=2, dim=-1)
            enc_out = fw_rnn + bw_rnn

            fw_hiddens = final_hidden[0::2]  # 每层前向传播的隐层状态
            bw_hiddens = final_hidden[1::2]  # 每层反向传播的隐层状态
            # (num_layers, batch_size, hidden_size)
            final_hidden = fw_hiddens + bw_hiddens

        return enc_out, final_hidden

