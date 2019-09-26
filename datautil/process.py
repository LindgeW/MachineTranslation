import sys
import opencc
import re
import jieba
'''
语料预处理
'''


# 分离中英文语料
def separate(path):
    src_txt = 'cmn_en.txt'
    dest_txt = 'cmn_ch.txt'
    src_file = open(src_txt, 'a', encoding='utf-8')
    dest_file = open(dest_txt, 'a', encoding='utf-8')
    with open(path, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin):
            src, dest = line.split('\t')
            src_file.write(src+'\n')
            dest_file.write(dest)
            # if i % 2 == 0:
            #     src_file.write(line)
            # else:
            #     dest_file.write(line)
    src_file.close()
    dest_file.close()


# 在英文标点前添加空格
def process_eng(path):
    dots = [',', '.', '!', '?']
    with open(path, 'r', encoding='utf-8') as fin:
        with open('en.txt', 'a', encoding='utf-8') as fout:
            for line in fin:
                for dot in dots:
                    if dot in line:
                        line = line.replace(dot, ' ' + dot)
                # if '"' in line:
                #     line = re.sub(r'"\w+"', line, '" '+line[1:-1]+' "')
                fout.write(line)

# 中文繁体转简体
def process_ch(path):
    cc = opencc.OpenCC('t2s')
    f = open(path, 'r', encoding='utf-8')
    mapper = map(lambda x: cc.convert(x), f.readlines())
    with open("ch.txt", "a", encoding='utf-8') as fout:
        for line in mapper:
            fout.write(' '.join(jieba.lcut(line)))
    f.close()


if __name__ == '__main__':
    # separate('../data/cmn.txt')
    process_eng('../data/cmn/test/en.txt')
    # process_ch('../data/cmn/test/ch.txt')
    # print(sys.argv[1])
