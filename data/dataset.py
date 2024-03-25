# _*_ coding:utf-8 _*_
"""
@Project Name: glossification
@FileName: dataset.py
@Begin Date: 2023/11/10 20:29
@Author: caijianfeng
"""
import torch
from torch.utils.data import Dataset
# from torchtext.data import Dataset
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE


class CSL_Dataset(Dataset):
    def __init__(self, dataset_file=None, editing_casual_mask_file=None, pre_trained_tokenizer=True, tokenizer_name=None):
        """
        给定数据集文件路径, 读取数据
        :param dataset_file: type: str -> the filename of dataset
        :param editing_casual_mask_file: type: int -> the filename of editing casual mask of the dataset
        :param pre_trained_tokenizer: type: bool -> using the pre_trained tokenizer. If it is True, must offer the tokenizer_name
        :param tokenizer_name: type: str -> the name of pre_trained tokenizer
        :return None
        """
        if pre_trained_tokenizer:
            assert tokenizer_name, 'if pre_trained_tokenizer is True, you must offer the tokenizer_name !'
            self.tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        if dataset_file is not None:
            # assert editing_casual_mask_file is not None, 'the editing casual mask file must provided !'
            with open(dataset_file, 'r', encoding='utf-8') as f:
                data_lines = f.readlines()
            # editing_casual_mask = np.load(editing_casual_mask_file)

            self.data_lines = data_lines
            self.data_lines_split = [data_line.split('|') for data_line in self.data_lines[1:]]
            self.data_lines_gloss = [data_line[3].replace(' ', '') for data_line in self.data_lines_split]
            self.data_lines_sentence = [data_line[4].replace(' ', '') for data_line in self.data_lines_split]
            self.data_lines_editing_program = [data_line[7].replace(' ', '') for data_line in self.data_lines_split]
            # self.editing_casual_mask = editing_casual_mask

            self.data_sentence_token = [token.ids for token in self.tokenize(self.data_lines_sentence)]
            self.data_gloss_token = [token.ids for token in self.tokenize(self.data_lines_gloss)]
            self.data_editing_program_token = [token.ids for token in self.tokenize(self.data_lines_editing_program)]
            # for i, data_edit in enumerate(self.data_editing_program_token):
            #     if data_edit[-1] != self.get_pad_id():
            #         print(i, '; ', len(data_edit))

    def __getitem__(self, item):
        """
        根据 item (即 index) 获取对应标号的数据, 包括 (原始句子, gloss, editing program, editing casual mask)
        :param item: type: int -> the index of data in dataset
        :return: dict(tensor(int), tensor(int), tensor(int), np.array(data_num, max_program_length, max_gloss_length)) -> (sentence token indexes, gloss token indexes, editing program token indexes, editing casual mask)
        """
        return {
            'src': np.array(self.data_sentence_token[item], dtype=np.int64),
            'trg': np.array(self.data_gloss_token[item], dtype=np.int64),
            'pro': np.array(self.data_editing_program_token[item], dtype=np.int64),
            # 'editing_casual_mask': self.editing_casual_mask[item]
        }

    def __len__(self):
        return len(self.data_sentence_token)

    def tokenize(self, text, enable_padding=True):
        """
        将给定的一个 batch 的文字数据转化为 token id 序列
        :param enable_padding: type: bool -> 表示是否将给定的 batch 的数据填充到同一长度(以最长序列为准)
        :param text: list(str) -> 表示一个 batch 的文字数据
        :return: list(tokenizers.Encoding) -> 表示对应的 token id 序列
        """
        if enable_padding:
            self.tokenizer.enable_padding()  # 允许 tokenizer 按照给定的 batch 中的最长序列对其余序列进行 [PAD] 填充
        text_token = self.tokenizer.encode_batch(text)
        return text_token

    def decode(self, text_token_ids):
        """
        根据给定的 token id 序列, 将其转化为原始的字符串
        :param text_token_ids: type: list(list(int)) -> 表示一个 batch 的 ids 列表
        :return: type: list(str) -> 表示对应的一个 batch 的字符串
        """
        return self.tokenizer.decode_batch(text_token_ids)

    def get_vocab(self):
        """
        获得当前数据集使用的 tokenizer 的 vocab
        :return: token vocab: type: dict -> key 是 汉字, value 是 token index
        """
        return self.tokenizer.get_vocab()

    def get_vocab_size(self):
        """
        获得当前数据集使用的 tokenizer 的 vocab size
        :return: token vocab size: type: int
        """
        return self.tokenizer.get_vocab_size()

    def get_token_id(self, token):
        """
        给定 token, 获得当前数据集使用的 tokenizer 的 token id
        :param token: type: str -> 给定的 token
        :return: token id: type: int or None -> 其中 None 表示 tokenizer 的 vocab 中没有该 token
        """
        return self.tokenizer.token_to_id(token)

    def get_pad_id(self, pad_token='[PAD]'):
        """
        获得当前数据集使用的 tokenizer 的 pad token id
        :param pad_token: type: str -> tokenizer 使用的 pad 的符号, 默认为 [PAD]
        :return: pad token id: type: int
        """
        return self.get_token_id(pad_token)

    def get_token_from_id(self, id):
        """
        给定 id, 获得当前数据集使用的 tokenizer 的对应 token
        :param id: type: int -> token id
        :return: token: type: str -> 给定 id 所对应的 token
        """
        return self.tokenizer.id_to_token(id)

    def get_data_len(self):
        return {
            'src': self.data_sentence_token[0].shape[0],
            'trg': self.data_gloss_token[0].shape[0],
            'pro': self.data_editing_program_token[0].shape[0],
        }
