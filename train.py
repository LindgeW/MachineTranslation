import torch
import random
import numpy as np
from config.conf import arg_config, path_conf
from datautil.dataloader import load_data, prepare_data
from modules.nmt import NMT
from modules.encoder import MTEncoder
from modules.att_decoder import AttMTDecoder


if __name__ == '__main__':
    torch.manual_seed(1453)
    torch.cuda.manual_seed_all(1453)
    np.random.seed(2343)
    random.seed(1347)

    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())

    args = arg_config()
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')

    data_path = path_conf('./config/data_path.json')

    train_src_corpus = load_data(data_path['train']['src_lang'])
    train_tgt_corpus = load_data(data_path['train']['tgt_lang'])
    test_src_corpus = load_data(data_path['test']['src_lang'])
    test_tgt_corpus = load_data(data_path['test']['tgt_lang'])

    train_pairs, src_vocab, tgt_vocab = prepare_data(train_src_corpus, train_tgt_corpus)
    test_pairs, _, _ = prepare_data(test_src_corpus, test_tgt_corpus)

    src_embed_weights = src_vocab.get_embedding_weights(data_path['embedding']['src_embedding'])
    tgt_embed_weights = tgt_vocab.get_embedding_weights(data_path['embedding']['tgt_embedding'])
    print('source lang vocab size:', src_vocab.vocab_size)
    print('target lang vocab size:', tgt_vocab.vocab_size)

    args.src_vocab_size = src_vocab.vocab_size
    args.tgt_vocab_size = tgt_vocab.vocab_size
    encoder = MTEncoder(args, src_embed_weights).to(args.device)
    # decoder = MTDecoder(args, tgt_embed_weights).to(args.device)
    decoder = AttMTDecoder(args, tgt_embed_weights).to(args.device)

    nmt = NMT(encoder, decoder)
    print(nmt.summary())

    nmt.train_iter(train_pairs, args, src_vocab, tgt_vocab)

    nmt.evaluate(test_pairs, args, src_vocab, tgt_vocab)
