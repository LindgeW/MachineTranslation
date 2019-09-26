import torch
import torch.nn as nn

'''
解码器：基于语言模型生成目标语言
'''


# without attention
class MTDecoder(nn.Module):
    def __init__(self, args, embedding_weights=None):
        super(MTDecoder, self).__init__()

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

        self._bidirectional = False
        self.nb_directions = 2 if self._bidirectional else 1

        self._decoder = nn.GRU(input_size=args.tgt_embedding_dim,
                              hidden_size=args.hidden_size,
                              num_layers=args.num_layers,
                              batch_first=True,
                              dropout=(0. if args.num_layers == 1 else args.rnn_drop),
                              bidirectional=self._bidirectional)

        self.activate = nn.ReLU()

        self.linear = nn.Linear(in_features=args.hidden_size * self.nb_directions,
                                out_features=args.tgt_vocab_size)

        self.softmax = nn.LogSoftmax(-1)

    def forward(self, inputs, init_hidden):
        '''
        :param inputs: (bz, 1)
        :param init_hidden:  (nb_layers, bz, hidden_size)
        :return:
        '''
        assert init_hidden is not None

        # (bz, 1) -> (bz, 1, embed_dim)
        embed = self.embed_layer(inputs)

        # if self.training:
        #     embed = self.embed_drop_layer(embed)

        # dec_out: (bz, seq_len, hidden_size * nb_directions)
        # final_hidden: (num_layers * nb_directions, bz, hidden_size)
        dec_out, final_hidden = self._decoder(self.activate(embed), init_hidden)

        # (bz, 1, hidden_size) -> (bz, hidden_size) -> (bz, tgt_vocab_size)
        out = self.linear(dec_out.squeeze(1))

        return self.softmax(out), final_hidden
