# _*_ coding:utf-8 _*_
"""
@Project Name: (已用已读)glossification
@FileName: metrics.py
@Begin Date: 2024/3/23 14:32
@End Date: 
@Author: caijianfeng
"""
import utils

# utils.set_proxy()

import nltk
from nltk.translate.bleu_score import corpus_bleu

reference = [['this', 'is', 'a', 'test'], ['this', 'is', 'test']]
candidate = ['this', 'is', 'a', 'test']

bleu_score = corpus_bleu(reference, candidate)
print("BLEU Score:", bleu_score)
