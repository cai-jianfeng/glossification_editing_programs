# _*_ coding:utf-8 _*_
"""
@Software: (已读)glossification
@FileName: BPE.py
@Date: 2023/11/30 23:04
@Author: caijianfeng
"""
import collections
import re


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    '''re.escape 表示将其中的所有转义字符进行转义
    例如, pair = ('o', 'w'), 则 ' '.join(pair) = 'o w',
    可以看到，其中的空白字符是通过转义字符而来的, 即其原始的字符串为 'o\ w';
    然后 escape 对这个字符串进行再转义, 其中的 \ 是由转义字符而来, 最终获得 'o\\ w'。'''
    bigram = re.escape(' '.join(pair))
    # TODO: 但是我不了解为什么不直接使用下面的形式，好像不用将其转化也可以进行匹配
    # bigram = ' '.join(pair)
    '''为了防止出现一下情况：
    word = 'lo w', pair = ('o', 'w'),
    如果使用 re.compile(bigram), 则会将 word 中的 'o w' 替换为 'ow', 这并不是我们想要的。
    因此，除了满足 bigram, 还必须满足前后有空白字符来说明其是一个 subword; 
    但是当它是第一个 / 最后一个时，其前面 / 后面是没有空白字符的, 需要处理这 2 种特殊情况。
    所以最终的匹配模式为: (?<!\S)' + bigram + r'(?!\S)'''
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        # 将 word 中所有匹配项 (匹配 p 的模式: (?<!\S)' + re.escape(' '.join(pair)) + r'(?!\S)')使用 ''.join(pair) 替换
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


if __name__ == '__main__':

    vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
    num_merges = 10
    for i in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(best)
    print(vocab)  # {'low</w>': 5, 'low e r </w>': 2, 'newest</w>': 6, 'wi d est</w>': 3}

    final_vocab = {}
    for word, index in vocab.items():
        subwords = word.split()
        for subword in subwords:
            final_vocab.setdefault(subword, 0)
            final_vocab[subword] += index
    print(final_vocab)  # {'low</w>': 5, 'low': 2, 'e': 2, 'r': 2, '</w>': 2, 'newest</w>': 6, 'wi': 3, 'd': 3, 'est</w>': 3}
