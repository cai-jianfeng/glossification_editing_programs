# _*_ coding:utf-8 _*_
"""
@Software: (已读)glossification
@FileName: extract_editoring_program.py
@Date: 2023/11/1 17:58
@Author: caijianfeng
"""
import numpy as np
from tqdm import tqdm


def min_edit_distance(sentence, gloss):
    """
    编辑距离，即给定 2 个序列 A 和 B(假设序列的元素是字符串)，至少需要多少次编辑操作可以使得序列 A 变成序列 B
    其中合法的编辑操作包括：
    1. 增加 Add，即在 A 的任意位置增加一个任意元素
    2. 删除 Del，即删除 A 的任意位置的元素
    3. 替换 Sub，即将 A 的任意位置的元素替换为另一元素(注意，该代码实现中没有 替换)
    """
    # len_sentence = sentence.shape[0]
    # len_gloss = gloss.shape[0]
    len_sentence = len(sentence)
    len_gloss = len(gloss)

    # 状态矩阵: state[i][j] 表示由序列 A 的前 i 个组成的子序列 Ai 和序列 B 的前 j 个组成的子序列 Bi的最小编辑距离
    state = np.zeros([len_sentence + 1, len_gloss + 1])

    # 初始化状态矩阵
    for i in range(len_sentence + 1):
        state[i][0] = i
    for j in range(len_gloss + 1):
        state[0][j] = j

    '''
    DP
    state[i][j] 可由 state[i-1][j] 和 state[i][j-1] 进行 +1 得来
    或者由 state[i-1][j-1] + 1 进行得来(注意，没有 Sub，所以没有不能由 state[i-1][j-1] +1 得到, 同时, Copy 也算一次编辑操作)
    '''
    for i in range(1, len_sentence + 1):
        for j in range(1, len_gloss + 1):
            state[i][j] = min(state[i - 1][j], state[i][j - 1]) + 1
            if sentence[i - 1] == gloss[j - 1]:
                state[i][j] = min(state[i][j], state[i - 1][j - 1] + 1)

    # 最后结果，即最短编辑距离为 state[len_sentence][len_gloss]
    # print(state[len_sentence][len_gloss])
    return state


def min_edit_trajectory(state, sentence, gloss):
    """
    从后往前回溯确定 edit trajectory (Add 优先级 > Del)
    """
    # len_sentence = sentence.shape[0]
    # len_gloss = gloss.shape[0]
    len_sentence = len(sentence)
    len_gloss = len(gloss)

    edit_program = []
    i, j = len_sentence, len_gloss
    while i > 0 and j > 0:
        # print(i, '; ', j)
        if state[i - 1][j] + 1 == state[i][j]:
            # edit_program.insert(0, "Del %d" % i)
            edit_program.insert(0, "Del")
            i -= 1
        elif state[i][j - 1] + 1 == state[i][j]:
            edit_program.insert(0, "Add %s" % gloss[j - 1])
            j -= 1
        elif state[i - 1][j - 1] + 1 == state[i][j]:
            # edit_program.insert(0, "Copy %d" % i)
            edit_program.insert(0, "Copy")
            i -= 1
            j -= 1
    while i > 0:
        # edit_program.insert(0, "Del %d" % i)
        edit_program.insert(0, "Del")
        i -= 1
    while j > 0:
        edit_program.insert(0, "Add %s" % gloss[j - 1])
        j -= 1

    return edit_program


def compress_trajectory(edit_program):
    """
    双指针算法压缩 edit program
    """
    length = len(edit_program)
    i, j = 0, 0
    new_edit_program = []
    num = 0
    while i < length and j < length:
        if edit_program[i] == edit_program[j]:
            num += 1
            j += 1
        elif num > 1:
            new_edit_program.append(edit_program[i] + " %d" % num)
            i = j
            num = 0
        else:
            new_edit_program.append(edit_program[i])
            i = j
            num = 0
    new_edit_program.append("Skip")
    return new_edit_program


def min_edit_program_parallel(sentences, glosses):
    edit_programs = []
    for i, (sentence, gloss) in tqdm(enumerate(zip(sentences, glosses))):
        # print(i, '; ', sentence, '; ', gloss)
        state = min_edit_distance(sentence, gloss)
        edit_program = min_edit_trajectory(state, sentence, gloss)
        # print(edit_program)
        compress_edit_program = compress_trajectory(edit_program)
        edit_programs.append(compress_edit_program)
    return edit_programs


if __name__ == '__main__':
    # 序列 A
    # sentence = np.array(["montag", "und", "dienstag", "wechselhaft", "hier", "und", "da", "zeigt", "sich", "aber", "auch", "die", "sonne"])
    # # 序列 B
    # gloss = np.array(["montag", "dienstag", "wechselhaft", "mal", "auch", "sonne"])
    #
    # state = min_edit_distance(sentence, gloss)
    # # print(state)
    #
    # edit_program = min_edit_trajectory(state, sentence, gloss)
    # print(edit_program)
    #
    # compress_edit_program = compress_trajectory(edit_program)
    # print(compress_edit_program)
    seq1 = np.array([1, 1, 2, 1, 1, 1, 1, 1])
    seq2 = np.array([2, 1, 1, 1, 1, 1])
    state = min_edit_distance(seq1, seq2)
    print(state)

    edit_program = min_edit_trajectory(state, seq1, seq2)
    print(edit_program)
