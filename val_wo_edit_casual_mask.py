# _*_ coding:utf-8 _*_
"""
@Project Name: (已用已读)glossification
@FileName: val.py
@Begin Date: 2024/3/28 9:38
@End Date: 
@Author: caijianfeng
"""
import torch
from tensorboardX import SummaryWriter
from train_wo_edit_casual_mask import validation
from data.dataset import CSL_Dataset
from torch.utils.data import DataLoader

import argparse
import os

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--train_args_path', type=str, default='./output_wo_mask/train.json')
    args.add_argument('--trained_model', type=str, default='./output_wo_mask/last/models/best_model.pt')
    args.add_argument('--dataset_path', type=str, default='./CSL_data/CSL-Daily_editing_chinese_test.txt')
    args.add_argument('--tokenizer_name', type=str, default='bert-base-chinese')
    args.add_argument('--batch_size', type=int, default=64)

    val_opt = args.parse_args()

    model = torch.load(val_opt.trained_model, map_location=torch.device('cpu'))

    dataset = CSL_Dataset(dataset_file=val_opt.dataset_path,
                          pre_trained_tokenizer=True,
                          tokenizer_name=val_opt.tokenizer_name)

    dataloader = DataLoader(dataset, val_opt.batch_size, shuffle=True)

    val_opt.p_vocab_size = dataset.get_vocab_size()
    val_opt.pro_pad_idx = dataset.get_pad_id()

    val_opt.edit_num = 4
    val_opt.edit_op = [dataset.get_token_id('加'),
                       dataset.get_token_id('删'),
                       dataset.get_token_id('贴'),
                       dataset.get_token_id('过')]
    val_opt.label_smoothing = 0.1

    val_writer = SummaryWriter(logdir=os.path.join('./output_wo_mask', 'last/val'))

    val_loss, total_bleu3, total_bleu4, total_rouge_l, total_acc = validation(dataloader, model, 0,
                                                                              val_writer, val_opt, device='cpu')
    print('total best bleu3: ', total_bleu3)
    print('total best bleu4: ', total_bleu4)
    print('total best rouge-l: ', total_rouge_l)
    print('total best accuracy: ', total_acc)

    with open(os.path.join('./output_wo_mask', 'result.txt'), 'w', encoding='utf-8') as f:
        f.write(f'total best bleu3: {total_bleu3}\n')
        f.write(f'total best bleu4: {total_bleu4}\n')
        f.write(f'total best rouge-l: {total_rouge_l}\n')
        f.write(f'total best accuracy: {total_acc}\n')
