# _*_ coding:utf-8 _*_
"""
@Project Name: glossification
@FileName: test.py
@Begin Date: 2023/12/2 11:10
@Author: caijianfeng
"""
from utils import set_proxy
from dataset import CSL_Dataset
from torch.utils.data import DataLoader

set_proxy()

dataset_file = '../CSL_data/CSL-Daily_editing_chinese_test.txt'
model_name = "bert-base-chinese"

CSL_dataset = CSL_Dataset(dataset_file=dataset_file,
                          pre_trained_tokenizer=True,
                          tokenizer_name=model_name)

print(CSL_dataset[-1]['pro'])
print(CSL_dataset[-1]['editing_casual_mask'])