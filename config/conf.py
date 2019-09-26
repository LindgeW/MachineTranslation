import os
import json
import argparse


def path_conf(data_path):
    assert os.path.exists(data_path)

    with open(data_path, 'r', encoding='utf-8') as fin:
        opts = json.load(fin)
    return opts


def arg_config():
    parser = argparse.ArgumentParser('Neural Machine Translation')

    # 训练参数
    parser.add_argument('--cuda', type=int, default=-1, help='cuda device, default cpu')
    parser.add_argument('-bz', '--batch_size', type=int, default=64, help='batch size of source inputs')
    parser.add_argument('-ep', '--epoch', type=int, default=20, help='number of training')
    parser.add_argument('-tr', '--teacher_force', type=float, default=0.5, help='teaching rate in training')
    parser.add_argument('-bs', '--beam_size', type=int, default=3, help='beam width when decoder search')

    # 模型参数
    parser.add_argument('-sed', '--src_embedding_dim', type=int, default=200, help='feature size of source inputs')
    parser.add_argument('-ted', '--tgt_embedding_dim', type=int, default=200, help='feature size of target inputs')
    parser.add_argument('-hz', '--hidden_size', type=int, default=128, help='feature size of hidden layer')
    parser.add_argument('-nl', '--num_layers', type=int, default=1, help='number of rnn layers')
    parser.add_argument('-bi', '--bidirectional', type=bool, default=False, help='is encoder bidirectional?')
    parser.add_argument('-ed', '--embed_drop', type=float, default=0.5, help='drop rate of embedding layer')
    parser.add_argument('-rd', '--rnn_drop', type=float, default=0.3, help='drop rate of rnn layer')

    # 优化器参数
    parser.add_argument('-lr', '--lr', type=float, default=3e-3, help='learning rate')

    args = parser.parse_args()

    print(vars(args))

    return args



