# _*_ coding:utf-8 _*_
"""
@Software: (已读)glossification
@FileName: bleu_score.py
@Date: 2023/11/27 23:03
@Author: caijianfeng
"""
from collections import Counter
import math
from itertools import tee

def bleu_score(predicts, references, weight=None):
    """
    This is an implementation of BLEU Score
    referring to 'nltk.translate.bleu_score'
    in the corpus level.
    :param predicts: the candidate, i.e. the prediction of the model -> type: list(list(str))
    :param references: the references, i.e. the ground-truth corresponding to the candidate -> type: list(list(list(str)))
                      list(str) according to a sentence;
                      list(list(str)) according to all ground-truth corresponding to a candidate;
                      list(list(list(str))) according to the total ground-truth in the corpus level.
    :param weight: the average weight, which can also indict the n of the order of n-gram (default [0.25] * 4) -> type: list(float)
    :return: BLEU Score -> type: float
    """
    if weight is None:
        weight = [0.25] * 4
    assert len(predicts) == len(references), "the number of predicts is must equal to the one of references"

    p_count = Counter()
    p_count_clip = Counter()
    # p_n = {}
    predict_lens, reference_lens = 0, 0
    for predict, reference in zip(predicts, references):  # compute the n-gram in each data
        for i, _ in enumerate(weight, start=1):  # compute the i-gram in the range of 1 \sim n
            p_i_numerator, p_i_denominator = modified_precision(predict, reference, i)  # compute the count_clip and count number in the i-gram
            p_count[i] = p_count[i] + p_i_denominator
            p_count_clip[i] = p_count_clip[i] + p_i_numerator
            # equal to the follow code:
            # from nltk.translate.bleu_score import modified_precision
            # p_i = modified_precision(reference, predict, i)
            # p_count[i] = p_count[i] + p_i.denominator
            # p_count_clip[i] = p_count_clip[i] + p_i.numerator
        predict_lens += len(predict)
        reference_lens += closest_reference_lens(len(predict), reference)
    bp = brevity_penalty(reference_lens, predict_lens)  # compute brevity penalty
    if p_count_clip[1] == 0:
        return 0
    p_n = [p_count_clip_i/p_count_i for p_count_i, p_count_clip_i in zip(p_count.values(), p_count_clip.values())]
    # equal to the follow code:
    # from fractions import Fraction
    # p_n = [
    #     Fraction(p_count_clip[i], p_count[i], _normalize=False)
    #     for i, _ in enumerate(weight, start=1)
    # ]
    bleu = bp * math.exp(math.fsum([w * math.log(p_i) for w, p_i in zip(weight, p_n)]))
    return bleu


def modified_precision(predict, reference, n):
    """
    This is the function of compute the count_clip number and count number given the prediction(candidate) and reference.
    The important (and difficult) calculation is the count_clip
    :param predict: type: list(list(str))
    :param reference: type: list(list(str))
    :param n: type: int
    :return: type -> (int, int)
    """
    pre_n_gram_count = Counter(n_gram(predict, n)) if len(predict) >= n else Counter()
    max_n_gram = {}
    for ref in reference:
        ref_n_gram_count = Counter(n_gram(ref, n)) if len(ref) >= n else Counter()
        for ng in pre_n_gram_count:
            max_n_gram[ng] = max(max_n_gram.get(ng, 0), ref_n_gram_count[ng])
    clip_n_gram_count = {ng: min(count, max_n_gram[ng]) for ng, count in pre_n_gram_count.items()}
    numerator = sum(clip_n_gram_count.values())
    denominator = max(1, sum(pre_n_gram_count.values()))
    return numerator, denominator


def closest_reference_lens(predict_len, reference):
    if predict_len == 0:
        return 0
    ref_lens = [len(ref) for ref in reference]
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - predict_len), ref_len))
    return closest_ref_len


def brevity_penalty(r, c):
    if c >= r:
        return 1
    elif c == 0:
        return 0
    else:
        return math.exp(1 - r / c)


def n_gram(sentence, n):
    sentence = iter(sentence)
    sentence_n = tee(sentence, n)
    for i, item in enumerate(sentence_n):
        for _ in range(i):
            next(item, None)
    return zip(*sentence_n)


if __name__ == '__main__':
    from nltk.translate.bleu_score import corpus_bleu

    references = [
        [
            ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
             'ensures', 'that', 'the', 'military', 'will', 'forever',
             'heed', 'Party', 'commands'],
            ['It', 'is', 'the', 'guiding', 'principle', 'which',
             'guarantees', 'the', 'military', 'forces', 'always',
             'being', 'under', 'the', 'command', 'of', 'the', 'Party'],
            ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
             'army', 'always', 'to', 'heed', 'the', 'directions',
             'of', 'the', 'party']
        ]
    ]

    predict = [
        ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
         'ensures', 'that', 'the', 'military', 'always',
         'obeys', 'the', 'commands', 'of', 'the', 'party']
    ]

    bleu = bleu_score(predict, references)
    print(bleu)
    bleu = corpus_bleu(references, predict)
    print(bleu)