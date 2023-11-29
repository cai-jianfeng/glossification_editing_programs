# _*_ coding:utf-8 _*_
"""
@Software: (已读)glossification
@FileName: ROUGE-L.py
@Date: 2023/11/28 21:54
@Author: caijianfeng
"""
import numpy as np


def longest_common_subsequence_length(sequence1, sequence2):
    """
    使用 DP 算法实现 2 个序列的最长公共子序列的长度求解
    :param sequence1: type: list / array
    :param sequence2: type: list / array
    :return: tuple (np.ndarray, int)
    """
    len1 = len(sequence1)
    len2 = len(sequence2)
    state = np.zeros([len1 + 1, len2 + 1])
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            # 当 s^1_i == s^2_j 时，这说明它相比 state[i-1][j-1] 的位置多了一对字符可以构成公共子序列，因此 state[i][j] = state[i-1][j-1] + 1；
            # 否则，state[i][j] = state[i-1][j-1]
            state[i][j] = state[i - 1][j - 1] + 1 if sequence1[i - 1] == sequence2[j - 1] else state[i - 1][j - 1]
            # 同时，state[i][j] 也可以舍弃 s^1_i 或者 s^2_j 回到上一状态 state[i-1][j] / state[i][j-1]
            state[i][j] = max([state[i][j], state[i - 1][j], state[i][j - 1]])
            '''
            由于 state[i - 1][j - 1] <= state[i - 1][j]/state[i][j - 1]，所以不用比较 state[i - 1][j - 1]
            只有当 state[i - 1][j - 1] == state[i - 1][j] & state[i][j - 1] 时，state[i][j] 才有可能 = state[i - 1][j - 1] + 1
            if sequence1[i - 1] == sequence2[j - 1]:
                state[i][j] = state[i - 1][j - 1] + 1
            state[i][j] = max([state[i][j], state[i - 1][j], state[i][j - 1]])
            '''
    return state, state[len1][len2]


def longest_common_subsequence(sequence1, sequence2):
    len1 = len(sequence1)
    len2 = len(sequence2)
    state, max_subseq_len = longest_common_subsequence_length(sequence1, sequence2)
    result = [state, max_subseq_len]
    track = []
    while max_subseq_len:
        '''
        由 longest_common_subsequence_length 的注释可知，只有当 state[i - 1][j - 1] == state[i - 1][j] & state[i][j - 1] 时，
        state[i][j] 才有可能由 state[i - 1][j - 1] 更新而来。此时，state[i][j] = state[i - 1][j - 1] + 1 > state[i - 1][j] & state[i][j - 1]
        因此，当 state[i][j] 的左边 state[i][j-1] / 上面 state[i-1][j] 和其的值一致时，
        表示其不是由 state[i-1][j-1] 过渡而来，而是由 state[i][j-1] / state[i-1][j] 过渡而来；
        若都不一致，则表示其是由左上角 state[i-1][j-1] 过渡而来
        '''
        if state[len1][len2 - 1] == state[len1][len2]:
            len2 -= 1
        elif state[len1 - 1][len2] == state[len1][len2]:
            len1 -= 1
        else:
            track.insert(0, sequence1[len1-1])
            len1, len2 = len1 - 1, len2 - 1
            max_subseq_len -= 1

    result.append(track)
    return result


def ROUGE_L(predict, reference):
    """
    X = reference, Y = predict
    R_lcs = \dfrac{LCS(X, Y)}{m};
    P_lcs = \dfrac{LCS(X, Y)}{n};
    beta = P_lcs / R_lcs or 1;
    F_lcs = \dfrac{(1 + \beta^2) * R_lcs * P_lcs}{R_lcs + \beta^2 * P_lcs}
    :param predict: type: list / array
    :param reference: type: list / array
    :return: tuple(float, float, float)
    """
    _, lcs = longest_common_subsequence_length(predict, reference)
    m, n = len(reference), len(predict)
    R_lcs = lcs / m
    P_lcs = lcs / n
    beta = P_lcs / R_lcs
    F_cls = (1 + beta ** 2) * R_lcs * P_lcs / (R_lcs + beta ** 2 * P_lcs)
    return R_lcs, P_lcs, F_cls


def ROUGE_L_multi_ref(predict, references):
    R_lcs_max, P_lcs_max = 0, 0
    for reference in references:
        R_lcs, P_lcs, _ = ROUGE_L(predict, reference)
        R_lcs_max = max(R_lcs_max, R_lcs)
        P_lcs_max = max(P_lcs_max, P_lcs)
    beta = P_lcs_max / R_lcs_max
    F_lcs_multi = (1 + beta ** 2) * R_lcs_max * P_lcs_max / (R_lcs_max + beta ** 2 * P_lcs_max)
    return R_lcs_max, P_lcs_max, F_lcs_multi


if __name__ == '__main__':
    # test LCS and LCS len
    # sequence1 = [1, 2, 10, 3, 22, 4, 5, 6, 34, 8]
    # sequence2 = [2, 0, 3, 4, 5, 6, 7, 8, 9]
    # state, max_subseq_len, max_subseq = longest_common_subsequence(sequence1, sequence2)
    # print(state)
    # print(max_subseq_len)
    # print(max_subseq)
    # ground_truth = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    #                 'ensures', 'that', 'the', 'military', 'will', 'forever',
    #                 'heed', 'Party', 'commands']
    #
    # predict = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    #            'ensures', 'that', 'the', 'military', 'always',
    #            'obeys', 'the', 'commands', 'of', 'the', 'party']
    # state, max_subseq_len, max_subseq = longest_common_subsequence(predict, ground_truth)
    # print(state)
    # print(max_subseq_len)
    # print(max_subseq, ';', len(max_subseq))
    # seq1 = 'abcdefghijklmjnojp'
    # seq2 = 'abcdefihijkrstun'
    # state, max_subseq_len, max_subseq = longest_common_subsequence(seq1, seq2)
    # print(state)
    # print(max_subseq_len)
    # print(max_subseq, ';', len(max_subseq))

    # test ROUGE-L
    predict1 = ['police', 'kill', 'the', 'gunman']
    predict2 = ['the', 'gunman', 'kill', 'people']
    reference = ['police', 'killed', 'the', 'gunman']
    rouge_l1 = ROUGE_L(predict1, reference)
    rouge_l2 = ROUGE_L(predict2, reference)
    print(rouge_l1, ';', rouge_l2)
