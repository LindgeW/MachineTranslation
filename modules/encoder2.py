import torch
import torch.nn as nn
from .rnn_encoder import RNNEncoder
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

        self._encoder = RNNEncoder(input_size=args.src_embedding_dim,
                                   hidden_size=args.hidden_size,
                                   num_layers=args.num_layers,
                                   batch_first=True,
                                   bidirectional=self._bidirectional,
                                   dropout=(0. if args.num_layers == 1 else args.rnn_drop),
                                   rnn_type='gru')

    def _init_hidden(self, batch_size: int, device=torch.device('cpu')):
        h0 = torch.zeros((batch_size, self.args.hidden_size),
                         requires_grad=True,
                         device=device)
        return h0

    def forward(self, inputs, init_hidden=None, mask=None):
        '''
        :param inputs:  (bz, seq_len)
        :param init_hidden:  (nb_layers, bz, hidden_size)
        :param mask: (bz, seq_len)
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
        # final_hidden: (num_layers, batch_size, hidden_size * nb_directions)
        enc_out, final_hidden = self._encoder(embed, init_hidden, mask=mask)

        if self._bidirectional:
            # (batch_size, seq_len, hidden_size)
            fw_rnn, bw_rnn = enc_out.chunk(chunks=2, dim=-1)
            enc_out = fw_rnn + bw_rnn

            fw_hidden, bw_hidden = final_hidden.chunk(chunks=2, dim=-1)
            # (num_layers, batch_size, hidden_size)
            final_hidden = fw_hidden + bw_hidden

        return enc_out, final_hidden

