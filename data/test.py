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

# # Load the pre-trained model and tokenizer
# model = AutoModel.from_pretrained(model_name)
# # Get the word embeddings layer
# embeddings_table = model.get_input_embeddings()
# CSL_dataloader = DataLoader(CSL_dataset, 2)
# for data in CSL_dataloader:
#     print(data['src'], '\n', data['trg'], '\n', data['pro'])
#     print(type(data['src']))
#     break
# test_data = CSL_dataset[0]
# print(test_data)

# print(CSL_dataset.get_token_id('加'))
# print(CSL_dataset.get_token_id('删'))
# print(CSL_dataset.get_token_id('贴'))
# print(CSL_dataset.get_token_id('过'))

# print(CSL_dataset.decode([[101, 6585, 125, 1160, 123, 6585, 123, 1160, 128, 6585, 122, 1160, 127, 1217, 1914, 1217, 1069, 1217, 6637, 1217, 752, 1217, 2658, 1217, 1139, 123, 6814, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
print(CSL_dataset.decode([[101, 6585, 125, 1217, 102, 1217, 125, 102]]))
# print(CSL_dataset.get_token_id(' '))