# _*_ coding:utf-8 _*_
"""
@Project Name: glossification
@FileName: test.py
@Begin Date: 2023/12/2 11:10
@Author: caijianfeng
"""
from utils import set_proxy
from dataset import CSL_Dataset
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader

set_proxy()

dataset_file = './CSL-Daily_editing_chinese.txt'
editing_casual_mask_file = './editing_casual_mask_CSL_50_40.npy'
model_name = "bert-base-chinese"

CSL_dataset = CSL_Dataset(dataset_file=dataset_file,
                          editing_casual_mask_file=editing_casual_mask_file,
                          pre_trained_tokenizer=True,
                          tokenizer_name=model_name)

# # Load the pre-trained model and tokenizer
# model = AutoModel.from_pretrained(model_name)
# # Get the word embeddings layer
# embeddings_table = model.get_input_embeddings()
CSL_dataloader = DataLoader(CSL_dataset, 1)
for data in CSL_dataloader:
    print(len(data['src']), '; ', len(data['trg']), '; ', len(data['pro']), '; ', len(data['editing_casual_mask']))
    print(data['src'].dtype)
    break

# print(CSL_dataset[0]['src'])