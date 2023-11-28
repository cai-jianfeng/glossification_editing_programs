# _*_ coding:utf-8 _*_
"""
@Software: (已读)glossification
@FileName: metrics.py
@Date: 2023/11/27 15:29
@Author: caijianfeng
"""
from bleu_score import bleu_score
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
