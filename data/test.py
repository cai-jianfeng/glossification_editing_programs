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

dataset_file = '../CSL_data/CSL-Daily_editing_chinese.txt'
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

print(CSL_dataset.get_pad_id())