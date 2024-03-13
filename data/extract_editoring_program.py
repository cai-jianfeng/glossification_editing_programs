# _*_ coding:utf-8 _*_
"""
@Project Name: glossification
@FileName: extract_editoring_program.py
@Begin Date: 2023/11/1 17:58
@End Date: 2024/03/13 09:55
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
    3. 替换 Sub，即将 A 的任意位置的元素替换为另一元素(注意，该代码实现中没有 替换);
       但是增加了 复制 Copy 操作, 即将 A 序列的元素复制到 B 序列
    :param sentence: type: str 序列 A
    :param gloss: type: str 序列 B
    :return: type: np.array(shape = sentence_len + 1, gloss_len + 1) 状态矩阵,
             其中第 (i, j) 个元素表示由序列 A 的前 i-1 个元素转化为序列 B 的前 j-1 个元素所需的最少编辑操作
    """
    # len_sentence = sentence.shape[0]
    # len_gloss = gloss.shape[0]
    len_sentence = len(sentence)
    len_gloss = len(gloss)

    # 状态矩阵: state[i][j] 表示由序列 A 的前 i 个组成的子序列 Ai 和序列 B 的前 j 个组成的子序列 Bi的最小编辑距离
    state = np.zeros([len_sentence + 1, len_gloss + 1])

    # 初始化状态矩阵: 由序列 A 的前 i 个元素转化为序列 B 的前 0 个元素的唯一解为 i 次删除操作;
    #               由序列 A 的前 0 个元素转化为序列 B 的前 j 个元素的唯一解为 j 次删除增加
    for i in range(len_sentence + 1):
        state[i][0] = i
    for j in range(len_gloss + 1):
        state[0][j] = j

    '''
    DP 算法
    state[i][j] 可由 state[i-1][j] 和 state[i][j-1] 进行 +1 得来, 
    表示可以通过删除序列 A 的第 i 个元素转化为 state[i-1][j] 的状态, 
    也可以通过增加序列 B 的第 j 个元素转化为 state[i][j-1] 的状态,
    或者由 state[i-1][j-1] + 1 进行得来,
    表示若序列 A 的第 i 个元素和序列 B 的第 j 个元素相同，
    则可以通过将序列 A 的第 i 个元素复制到序列 B 将其转化为 state[i-1][j-1] 的状态
    (注意，没有 Sub, 同时, Copy 也算一次编辑操作)
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
    通过状态矩阵 state 从后往前回溯确定 edit trajectory (Add 优先级 > Del)
    :param state: type: np.array(shape = sentence_len + 1, gloss_len + 1) DP 算法计算最小编辑距离的状态矩阵
    :param sentence: type: str 序列 A
    :param gloss: type: str 序列 B
    :return: type: list(str) 由序列 A　通过最少编辑操作得到序列　B　的具体编辑操作序列, 即 editing program
    """
    # len_sentence = sentence.shape[0]
    # len_gloss = gloss.shape[0]
    len_sentence = len(sentence)
    len_gloss = len(gloss)

    edit_program = []
    i, j = len_sentence, len_gloss
    while i > 0 and j > 0:
        # print(i, '; ', j)
        if state[i][j - 1] + 1 == state[i][j]:
            # edit_program.insert(0, "Add %s" % gloss[j - 1])
            edit_program.insert(0, "添加 %s" % gloss[j - 1])
            j -= 1
        elif state[i - 1][j] + 1 == state[i][j]:
            # edit_program.insert(0, "Del %d" % i)
            # edit_program.insert(0, "Del")
            edit_program.insert(0, "删除")
            i -= 1
        elif state[i - 1][j - 1] + 1 == state[i][j]:
            # edit_program.insert(0, "Copy %d" % i)
            # edit_program.insert(0, "Copy")
            edit_program.insert(0, "复制")
            i -= 1
            j -= 1
    while i > 0:
        # edit_program.insert(0, "Del %d" % i)
        # edit_program.insert(0, "Del")
        edit_program.insert(0, "删除")
        i -= 1
    while j > 0:
        # edit_program.insert(0, "Add %s" % gloss[j - 1])
        edit_program.insert(0, "添加 %s" % gloss[j - 1])
        j -= 1

    return edit_program


def compress_trajectory(edit_program):
    """
    双指针算法压缩 edit program, 即原文中的 For 操作, 将连续 N 次的编辑操作 A 压缩为 A N.
    :param edit_program: type: list(str) 给定的未压缩的编辑操作序列
    :return: type: list(str) 压缩后的编辑操作序列
    """
    length = len(edit_program)
    i, j = 0, 0
    new_edit_program = []
    num = 0
    while i < length and j <= length:
        if j >= length:
            if num > 1:
                new_edit_program.append(edit_program[i] + " %d" % num)
            else:
                # if 'Add' in edit_program[i]:
                if '添加' in edit_program[i]:
                    new_edit_program.append(edit_program[i])
                else:
                    new_edit_program.append(edit_program[i] + ' 1')
            break
        elif edit_program[i] == edit_program[j]:
            num += 1
            j += 1
        elif num > 1:
            new_edit_program.append(edit_program[i] + " %d" % num)
            i = j
            num = 0
        else:
            # if 'Add' in edit_program[i]:
            if '添加' in edit_program[i]:
                new_edit_program.append(edit_program[i])
            else:
                new_edit_program.append(edit_program[i] + ' 1')
            i = j
            num = 0
    # 如果最后一个编辑操作是删除, 则可以直接省略, 使用 Skip 结束
    # if 'Del' in new_edit_program[-1]:
    if '删除' in new_edit_program[-1]:
        new_edit_program.pop()
    # new_edit_program.append("Skip")
    new_edit_program.append("跳过")
    return new_edit_program


def min_edit_program_parallel(sentences, glosses):
    """
    将 min_edit_trajectory 操作扩展为批处理
    TODO: 能否将其重写为并行生成的模式？
    :param sentences: list(list(str)) 一个小批量(mini batch) 的序列 A
    :param glosses: list(list(str)) 一个小批量(mini batch) 的序列 B
    :return:  list(list(str)) 返回给定小批量(mini batch) 中的每对序列的 editing program
    """
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
    # seq1 = np.array([1, 1, 2, 1, 1, 1, 1, 1])
    # seq2 = np.array([2, 1, 1, 1, 1, 1])
    # state = min_edit_distance(seq1, seq2)
    # print(state)

    # edit_program = min_edit_trajectory(state, seq1, seq2)
    # print(edit_program)

    # seq1 = ['Copy', 'Copy', 'Copy', 'Copy', 'Add love', 'Del', 'Del', 'Del', 'Add you']
    seq1 = ['复制', '复制', '复制', '复制', '添加 你', '删除', '删除', '删除', '添加 好']
    seq = compress_trajectory(seq1)
    print(seq)
