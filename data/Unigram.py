# _*_ coding:utf-8 _*_
"""
@Software: (已读)glossification
@FileName: Unigram.py
@Date: 2023/12/2 13:33
@Author: caijianfeng
"""
from transformers import AutoTokenizer
from collections import defaultdict
from math import log
import copy


def symbol_frequence(tokenizer, corpus):
    """
    统计经过初始化分词后的 corpus 中每个句子的每个分词的词频
    :param tokenizer: 对 corpus 进行初始化分词的分词器 -> type: AutoTokenizer
    :param corpus: 语料库，即训练数据 -> type: list(str)
    :return: 初始化分词后的每个分词词频 -> type: dict(str, int)
    """
    word_freqs = defaultdict(int)
    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)  # 使用预分词器进行初始化分词
        new_words = [word for word, offset in words_with_offsets]  # 得到 text 的对应分词序列
        for word in new_words:  # 统计每个分词的词频
            word_freqs[word] += 1
    return word_freqs


def init_vocabulary(word_freqs, num=300):
    """
    使用初始化分词词频来初始化符号词汇表
    :param num: 初始化符号词汇表的大小 -> type: int
    :param word_freqs: 初始化分词词频 -> type: dict(str, int)
    :return: 初始化符号词汇表 -> type: dict(str, int)
    """
    char_freqs = defaultdict(int)  # 单个字符符号的词汇表
    subwords_freqs = defaultdict(int)  # 多个字符组成的子词符号的词汇表
    for word, freq in word_freqs.items():
        for i in range(len(word)):
            char_freqs[word[i]] += freq  # 首先统计每个词的单个字符
            # Loop through the subwords of length at least 2
            for j in range(i + 2, len(word) + 1):  # 然后统计每个词的所有可能子词
                subwords_freqs[word[i:j]] += freq

    # Sort subwords by frequency
    sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)  # 按照词频进行排序

    # 选择保留所有单个字符符号和剩下排序靠前的子词符号，注意，需要保证最终的符号数量等于预定义的 num
    token_freqs = list(char_freqs.items()) + sorted_subwords[: num - len(char_freqs)]
    # 其中 freq 就是每个符号出现的词频，即 freq(x_i)
    token_freqs = {token: freq for token, freq in token_freqs}

    # 统计任意符号出现的频率，即 freq(any)
    # 所以每个符号的概率 = 该符号出现的词频 / 任意符号出现的频率，即 freq(x_i) / freq(any) -> Unigram LM 的计算方程
    total_sum = sum([freq for token, freq in token_freqs.items()])
    # 计算每个符号的对数似然，方便后续计算 loss
    model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}
    return token_freqs, model


def encode_word(word, model):
    """
    Viterbi Algorithm
    :param word: 给定的词，需要使用 model 中的符号对其进行 tokenize 变成符号序列 -> type: str
    :param model: 给定的符号词汇表 -> type: dict(str, int)
    :return: 分词结果 和 其对应的对数似然 -> type: tuple(list, float)
    """
    best_segmentations = [{"start": 0, "score": 1}] + [{"start": None, "score": None} for _ in range(len(word))]
    for start_idx in range(len(word)):
        # This should be properly filled by the previous steps of the loop
        best_score_at_start = best_segmentations[start_idx]["score"]
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                # If we have found a better segmentation ending at end_idx, we update
                if (
                    best_segmentations[end_idx]["score"] is None
                    or best_segmentations[end_idx]["score"] > score
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}

    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        # We did not find a tokenization of the word -> unknown
        return ["<unk>"], None

    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)
    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score

def compute_loss(model, word_freqs):
    """
    计算所有符号的累计损失 L
    \mathcal{L} = \sum_{i=1}^{|\mathcal{D}|}log(\sum_{\vec{x} \in S(X^{(s)})}\prod_{i=1}^{M_{\vec{x}}}p(x_i))
    :param model: 给定的符号词汇表 -> type: dict(str, int)
    :param word_freqs: 给定的初始符号词频 -> type: dict(str, int)
    :return: 计算的 loss -> type: float
    """
    loss = 0
    for word, freq in word_freqs.items():
        _, word_loss = encode_word(word, model)
        loss += freq * word_loss
    return loss


def compute_scores(model, word_freqs):
    """
    计算所有符号的 loss，即删除该符号后的 L 减少量
    :param model: 给定的符号词汇表 -> type: dict(str, int)
    :param word_freqs: 给定的初始符号词频 -> type: dict(str, int)
    :return: 所有符号的 loss -> type: dict(str, float)
    """
    scores = {}
    model_loss = compute_loss(model, word_freqs)
    for token in model.keys():
        # We always keep tokens of length 1
        if len(token) == 1:
            continue
        model_without_token = copy.deepcopy(model)
        _ = model_without_token.pop(token)
        scores[token] = compute_loss(model_without_token, word_freqs) - model_loss
    return scores


def EM_iterate(model, word_freqs, token_freqs, num=100, percent_to_remove=0.1):
    """
    EM 算法迭代更新每个符号的概率 p(x_i) 和符号词汇表 V
    :param percent_to_remove: 每次迭代时删除的符号比例 -> type: float
    :param num: 预定义的符号词汇表的大小 -> type: int
    :param model: 给定的符号词汇表 -> type: dict(str, int)
    :param word_freqs: 给定的初始符号词频 -> type: dict(str, int)
    :param token_freqs: 计算得到的符号词频 -> type: dict(str, int)
    :return: 更新的符号词汇表 -> type: dict(str, int)
    """
    while len(model) > num:
        scores = compute_scores(model, word_freqs)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        # Remove percent_to_remove tokens with the lowest scores.
        for i in range(int(len(model) * percent_to_remove)):
            _ = token_freqs.pop(sorted_scores[i][0])

        total_sum = sum([freq for token, freq in token_freqs.items()])
        model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}
    return model


def tokenize(text, model):
    """
    使用已经训练好的符号词汇表对给定的 text 进行分词
    :param text: 给定的句子 -> type: str
    :param model: 训练好的符号词汇表 -> type: dict(str, int)
    :return: 分词结果 type: list(str)
    """
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in words_with_offsets]
    encoded_words = [encode_word(word, model)[0] for word in pre_tokenized_text]
    return sum(encoded_words, [])


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]
    word_freqs = symbol_frequence(tokenizer, corpus)
    token_freqs, init_model = init_vocabulary(word_freqs)
    model = EM_iterate(init_model, word_freqs, token_freqs)

    tokenize("This is the Hugging Face course.", model)