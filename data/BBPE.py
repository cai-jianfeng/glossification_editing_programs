# _*_ coding:utf-8 _*_
"""
@Software: (已读)glossification
@FileName: BBPE.py
@Date: 2023/12/1 15:48
@Author: caijianfeng
"""
import collections
import re

def convert(vocab):
    # vocab = {'l o w </w>': 5 ...}
    new_vocab = {}
    for word, freq in vocab.items():
        symbols = word.split()
        new_word = ''
        for symbol in symbols:
            for s in symbol:
                new_word += hex(ord(s))[2:]
                new_word += ' '
        new_vocab[new_word.strip()] = freq
        # TODO: 为什么下面这种方式不行
        '''
        for symbol in symbols:
            for s in symbol:
                word.replace(s, str(ord(s)))
        new_vocab[word] = freq
        '''
    return new_vocab


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
    '''为了防止出现以下情况：
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


def dynamic_progress(byte_list):
    """
    将给定的字节序列使用 DP 算法转化为原始的单词
    :param byte_list: type: list(str)
    :return: type: tuple(int, str)
    """
    f = [0] * (len(byte_list) + 1)
    bt = [-1] * len(f)
    for k in range(1, len(f)):
        for t in range(1, 5):
            if k - t >= 0 and f[k-t] + valid(k-t, k, byte_list)[0] > f[k]:
                f[k] = f[k-t] + valid(k-t+1, k, byte_list)[0]
                bt[k] = k-t
    bt_num = bt[-1]
    cur_num = len(byte_list)
    result = ''
    while bt_num != -1 and cur_num > 0:
        result = valid(bt_num, cur_num, byte_list)[1] + ' ' + result
        cur_num = bt_num
        bt_num = bt[cur_num]
    result = result.strip()
    return f[-1], result


def valid(i, j, byte_list):
    """
    判断当前 byte_list 中 的 i ~ j 的十六进制数对应的 UTF-8 的正确性
    :param i: type: int
    :param j: type: int
    :param byte_list: type: list(str)
    :return: int, str
    """
    # TODO: 后续需根据 UTF-8 的编码特性来进行改进
    number = 0
    for k in range(i, j):
        number = number * 256 + int(byte_list[k], 16)
    try:
        ch = chr(number)
    except ValueError:
        return 0, ' '
    return 1, ch


if __name__ == '__main__':

    vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
    vocab = convert(vocab)
    print(vocab)  # {'6c 6f 77 3c 2f 77 3e': 5, '6c 6f 77 65 72 3c 2f 77 3e': 2, '6e 65 77 65 73 74 3c 2f 77 3e': 6, '77 69 64 65 73 74 3c 2f 77 3e': 3}
    num_merges = 10
    for i in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(best)
    print(vocab)  # {'6c6f77 3c2f773e': 5, '6c6f77 65 72 3c2f773e': 2, '6e6577 6573743c2f773e': 6, '77 69 64 6573743c2f773e': 3}

    final_vocab = {}
    for word, index in vocab.items():
        subwords = word.split()
        for subword in subwords:
            final_vocab.setdefault(subword, 0)
            final_vocab[subword] += index
    print(final_vocab)  # {'6c6f77': 7, '3c2f773e': 7, '65': 2, '72': 2, '6e6577': 6, '6573743c2f773e': 9, '77': 3, '69': 3, '64': 3}

    a = ['l', 'o', 'w', '中']
    b = [hex(ord(ch))[2:] for ch in a]
    b = [s[i:i + 2] for s in b for i in range(0, len(s), 2)]
    print(b)  # ['6c', '6f', '77', '4e', '2d']
    lens, result = dynamic_progress(b)
    print(lens, result)  # len = 5, result =  'l o w N -'，代码将 '中' 解析为 'N' 和 '-'
    # TODO: 还需要在 dynamic_progress 函数中添加一个语言类型的 arg (type: str) 来指示当前的语种，这样就不会出现上述问题