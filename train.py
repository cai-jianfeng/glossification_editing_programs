# _*_ coding:utf-8 _*_
"""
@Project Name: glossification
@FileName: train.py
@Begin Date: 2024/3/14 11:08
@End Date: 
@Author: caijianfeng
"""
from torch.utils.data import DataLoader
from optimizer import LRScheduler
from transformers import AutoModel

from models.model import Glossification
from data.dataset import CSL_Dataset

from utils import set_proxy

set_proxy()

def train(model, dataloader, optimizer):
    pass

def main():
    dataset_file = './CSL-Daily_editing_chinese.txt'
    editing_casual_mask_file = './editing_casual_mask_CSL_50_40.npy'

    tokenizer_model_name = "bert-base-chinese"

    CSL_dataset = CSL_Dataset(dataset_file=dataset_file,
                              editing_casual_mask_file=editing_casual_mask_file,
                              pre_trained_tokenizer=True,
                              tokenizer_name=tokenizer_model_name)

    batch_size = 4
    CSL_dataloader = DataLoader(CSL_dataset, batch_size)

    # tokenizer = CSL_dataset.tokenizer
    # Load the pre-trained model and tokenizer
    tokenizer_model = AutoModel.from_pretrained(tokenizer_model_name)
    # Get the word embeddings layer
    embeddings_table = tokenizer_model.get_input_embeddings().weight

    head_num = 10
    inner_size = 1024
    dropout_rate = 0.1
    generator_encoder_n_layers = 3
    generator_decoder_n_layers = 1
    executor_encoder_n_layers = 1
    share_target_embeddings = True
    use_pre_trained_embedding = True

    model = Glossification(CSL_dataset.get_vocab_size(),
                           CSL_dataset.get_vocab_size(),
                           CSL_dataset.get_vocab_size(),
                           src_pad_idx=CSL_dataset.get_pad_id(),
                           trg_pad_idx=CSL_dataset.get_pad_id(),
                           pro_pad_idx=CSL_dataset.get_pad_id(),
                           head_num=head_num,
                           hidden_size=embeddings_table.shape[1],
                           inner_size=inner_size,
                           dropout_rate=dropout_rate,
                           generator_encoder_n_layers=generator_encoder_n_layers,
                           generator_decoder_n_layers=generator_decoder_n_layers,
                           executor_encoder_n_layers=executor_encoder_n_layers,
                           share_target_embeddings=share_target_embeddings,
                           use_pre_trained_embedding=use_pre_trained_embedding,
                           pre_trained_embedding=embeddings_table)

    Adam_optimizer = LRScheduler(parameters=model.parameters(),
                                 hidden_size=embeddings_table.shape[1],
                                 warmup=100)

    train(model, CSL_dataloader, Adam_optimizer)


