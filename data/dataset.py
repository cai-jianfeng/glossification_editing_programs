# _*_ coding:utf-8 _*_
"""
@Project Name: glossification
@FileName: dataset.py
@Begin Date: 2023/11/10 20:29
@Author: caijianfeng
"""
from torch.utils.data import Dataset
# from torchtext.data import Dataset
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE

class CSL_Dataset(Dataset):
    def __init__(self, dataset_file, editing_casual_mask_file, pre_trained_tokenizer=True, tokenizer_name=None):
        """
        给定数据集文件路径, 读取数据
        :param dataset_file: type: str -> the filename of dataset
        :param editing_casual_mask_file: type: int -> the filename of editing casual mask of the dataset
        :return None
        """
        if pre_trained_tokenizer:
            assert tokenizer_name, 'if pre_trained_tokenizer is True, you must offer the tokenizer_name !'
            self.tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        with open(dataset_file, 'r', encoding='utf-8') as f:
            data_lines = f.readlines()
        editing_casual_mask = np.load(editing_casual_mask_file)

        self.data_lines = data_lines
        self.data_lines_split = [data_line.split('|') for data_line in self.data_lines[1:]]
        self.data_lines_gloss = [data_line[3] for data_line in self.data_lines_split]
        self.data_lines_sentence = [data_line[4] for data_line in self.data_lines_split]
        self.data_lines_editing_program = [data_line[7] for data_line in self.data_lines_split]
        self.editing_casual_mask = editing_casual_mask

        self.data_sentence_token = self.tokenize(self.data_lines_sentence)
        self.data_gloss_token = self.tokenize(self.data_lines_gloss)
        self.data_editing_program_token = self.tokenize(self.data_lines_editing_program)

    def __getitem__(self, item):
        """
        根据 item (即 index) 获取对应标号的数据, 包括 (原始句子, gloss, editing program, editing casual mask)
        :param item: type: int -> the index of data in dataset
        :return: tuple(list(int), list(int), list(int), np.array(data_num, max_program_length, max_gloss_length)) -> (sentence token indexes, gloss token indexes, editing program token indexes, editing casual mask)
        """
        return self.data_sentence_token[item].ids, self.data_gloss_token[item].ids, self.data_editing_program_token[item].ids, self.editing_casual_mask[item]

    def __len__(self):
        return len(self.data_sentence_token)

    def tokenize(self, text):
        text_token = self.tokenizer.encode_batch(text)
        return text_token

    def get_vocab(self):
        """
        获得当前数据集使用的 tokenizer 的 vocab
        :return: token vocab: type: dict -> key 是 汉字, value 是 token index
        """
        return self.tokenizer.get_vocab()

    def decode(self, text_token_ids):
        """
        根据给定的 token id 序列, 将其转化为原始的字符串
        :param text_token_ids: type: list(list(int)) -> 表示一个 batch 的 ids 列表
        :return: type: list(str) -> 表示对应的一个 batch 的字符串
        """
        return self.tokenizer.decode_batch(text_token_ids)