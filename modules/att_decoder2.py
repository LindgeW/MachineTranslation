'''
含注意力机制的decoder:
    query: t-1时刻decoder的输出
    key-values: encoder输出
    key + query 得到注意力分数，施加到value上，得到特定部分的encoder输出
    将encoder输出整合decoder输出
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# with attention
class AttMTDecoder(nn.Module):
    def __init__(self, args, embedding_weights=None):
        super(AttMTDecoder, self).__init__()

        self.args = args

        if embedding_weights is not None:
            args.tgt_embedding_dim = embedding_weights.shape[1]
            self.embed_layer = nn.Embedding.from_pretrained(torch.from_numpy(embedding_weights))
        else:
            self.embed_layer = nn.Embedding(num_embeddings=args.tgt_vocab_size,
                                            embedding_dim=args.tgt_embedding_dim,
                                            padding_idx=0)

            nn.init.xavier_uniform_(self.embed_layer.weight)
            # nn.init.kaiming_uniform_(self.embed_layer.weight)

        self.embed_drop_layer = nn.Dropout(args.embed_drop)

        self._decoder = nn.GRU(input_size=args.tgt_embedding_dim,
                               hidden_size=args.hidden_size,
                               num_layers=args.num_layers,
                               batch_first=True,
                               dropout=(0. if args.num_layers == 1 else args.rnn_drop)
                               )

        self.att_combine = nn.Linear(in_features=args.hidden_size * 2,
                                     out_features=args.hidden_size)

        self.linear = nn.Linear(in_features=args.hidden_size,
                                out_features=args.tgt_vocab_size)

        self.softmax = nn.LogSoftmax(-1)

    def forward(self, inputs, init_hidden, encoder_outputs=None):
        '''
        :param inputs: (bz, 1)  t-1时刻decoder输出
        :param init_hidden: (nb_layers, bz, hidden_size)
        :param encoder_outputs:
                encoder全部输出  (batch_size, seq_len, hidden_size)
        :return:
        '''

        assert init_hidden is not None

        # (bz, 1) -> (bz, 1, embed_dim)
        embed = self.embed_layer(inputs)

        if self.training:
            embed = self.embed_drop_layer(embed)

        # dec_out: (bz, seq_len, hidden_size * nb_directions)
        # final_hidden: (num_layers * nb_directions, bz, hidden_size)
        rnn_out, final_hidden = self._decoder(embed, init_hidden)

        # 计算encoder输出的注意力权重分数
        # (bz, 1, hidden_size) * (bz, hidden_size, seq_len) -> (bz, 1, seq_len) -> (bz, seq_len)
        att_weights = torch.bmm(rnn_out, encoder_outputs.transpose(1, 2))
        att_weights /= math.sqrt(encoder_outputs.size(-1))
        soft_att_weights = F.softmax(att_weights.squeeze(), dim=1)
        # 将权重分数施加到encoder输出上
        # (bz, 1, seq_len) * (bz, seq_len, hidden_size) -> (bz, 1, hidden_size) -> (bz, hidden_size)
        dec_context = torch.bmm(soft_att_weights.unsqueeze(dim=1), encoder_outputs).squeeze()
        # 整合t-1时刻decoder的输出和encoder的局部输出
        # (bz, hidden_size * 2) -> (bz, hidden_size)
        dec_out = self.att_combine(torch.cat((rnn_out.squeeze(), dec_context), dim=1)).relu()

        # (bz, hidden_size) -> (bz, tgt_vocab_size)
        out = self.linear(dec_out)

        return self.softmax(out), final_hidden


