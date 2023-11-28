# _*_ coding:utf-8 _*_
"""
@Software: (已读)glossification
@FileName: word_error_rate.py
@Date: 2023/11/28 15:41
@Author: caijianfeng
"""
import numpy as np


def min_edit_distance(sentence1, sentence2):
    """
    编辑距离，即给定 2 个序列 A 和 B(假设序列的元素是字符串)，至少需要多少次编辑操作可以使得序列 A 变成序列 B
    其中合法的编辑操作包括：
    1. 增加 Add，即在 A 的任意位置增加一个任意元素
    2. 删除 Del，即删除 A 的任意位置的元素
    3. 替换 Sub，即将 A 的任意位置的元素替换为另一元素
    """
    if type(sentence1) is list:
        sentence1 = np.array(sentence1)
    assert type(sentence1) is np.ndarray, "the sentence1 must be the np.array or list"
    if type(sentence2) is list:
        sentence2 = np.array(sentence2)
    assert type(sentence2) is np.ndarray, "the sentence2 must be the np.array or list"

    len_sentence1 = sentence1.shape[0]
    len_sentence2 = sentence2.shape[0]

    # 状态矩阵: state[i][j] 表示由序列 A 的前 i 个组成的子序列 Ai 和序列 B 的前 j 个组成的子序列 Bi的最小编辑距离
    state = np.zeros([len_sentence1 + 1, len_sentence2 + 1])

    # 初始化状态矩阵
    for i in range(len_sentence1 + 1):
        state[i][0] = i
    for j in range(len_sentence2 + 1):
        state[0][j] = j

    '''
    DP
    state[i][j] 可由 state[i-1][j] 和 state[i][j-1] 进行 +1 得来
    或者由 state[i-1][j-1] / state[i-1][j-1] + 1 进行得来
    '''
    for i in range(1, len_sentence1 + 1):
        for j in range(1, len_sentence2 + 1):
            state[i][j] = min(state[i - 1][j], state[i][j - 1]) + 1
            if sentence1[i - 1] == sentence2[j - 1]:
                state[i][j] = min(state[i][j], state[i - 1][j - 1])
            else:
                state[i][j] = min(state[i][j], state[i - 1][j - 1] + 1)

    # 最后结果，即最短编辑距离为 state[len_sentence1][len_sentence2]
    # print(state[len_sentence1][len_sentence2])
    return state, state[len_sentence1][len_sentence2]


def word_error_rate(predicts, ground_truths):
    """
    This is an implementation of Word Error Rate Score in the corpus level.
    :param predicts: the prediction of the model -> type: list(list(str))
    :param ground_truths: the ground-truth corresponding to the predicts -> type: list(list(str))
                          list(str) according to a sentence;
                          list(list(str)) according to the total ground-truth in the corpus level.
    :return: word error rate in the total corpus -> type: float
    """
    wer_score = 0.0
    assert len(predicts) == len(ground_truths), "the number of predicts is must equal to the one of ground_truths"
    for predict, ground_truth in zip(predicts, ground_truths):
        _, edit_distance = min_edit_distance(predict, ground_truth)
        wer_score += edit_distance / len(ground_truth)
    wer_score /= len(predicts)
    return wer_score


if __name__ == '__main__':
    ground_truths = [
        ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
         'ensures', 'that', 'the', 'military', 'will', 'forever',
         'heed', 'Party', 'commands']
    ]

    predicts = [
        ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
         'ensures', 'that', 'the', 'military', 'always',
         'obeys', 'the', 'commands', 'of', 'the', 'party']
    ]
    wer_score = word_error_rate(predicts, ground_truths)
    print("Word Error Rate: ", wer_score)
