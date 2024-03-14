# _*_ coding:utf-8 _*_
"""
@Project Name: glossification
@FileName: test.py
@Begin Date: 2023/12/2 11:10
@Author: caijianfeng
"""
from utils import set_proxy
from dataset import CSL_Dataset

set_proxy()

dataset_file = './CSL-Daily_editing_chinese.txt'
editing_casual_mask_file = './editing_casual_mask_CSL_50_40.npy'
tokenizer_name = "bert-base-chinese"

CSL_dataset = CSL_Dataset(dataset_file=dataset_file,
                          editing_casual_mask_file=editing_casual_mask_file,
                          pre_trained_tokenizer=True,
                          tokenizer_name=tokenizer_name)

# index = 143
# test_data = CSL_dataset[index][0]
# print(CSL_dataset.data_sentence_token[index].tokens)
# origin_data = CSL_dataset.decode([test_data])
# print(origin_data)
# input_text = ["谢谢你！", "不客气！"]
# outputs = CSL_dataset.tokenizer.encode_batch(input_text)
# outputs_ids = [output.ids for output in outputs]
# decode_outputs = CSL_dataset.decode(outputs_ids)
# print(decode_outputs)
# print(CSL_dataset.get_token_id(token='1'))
