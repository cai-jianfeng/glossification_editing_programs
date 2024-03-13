# _*_ coding:utf-8 _*_
"""
@Project Name: glossification
@FileName: dataset.py
@Begin Date: 2023/11/10 20:29
@Author: caijianfeng
"""
from torch.utils.data import Dataset
# from torchtext.data import Dataset

class CSL(Dataset):
    def __init__(self, dataset_file, editing_casual_mask_file):
        pass

    def __getitem__(self, item):
        """
        根据 item (即 index) 获取对应标号的数据, 包括 (原始句子, gloss, editing program, editing casual mask)
        :param item: type: int -> the index of data in dataset
        :return: tuple(list(int), list(int), list(int), np.array(data_num, max_program_length, max_gloss_length)) -> (sentence token indexes, gloss token indexes, editing program token indexes, editing casual mask)
        """
        pass

    def __len__(self):
        pass

    def tokenizer(self, text):
        pass