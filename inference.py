# _*_ coding:utf-8 _*_
"""
@Project Name: (已用已读)glossification
@FileName: inference.py
@Begin Date: 2024/3/23 15:13
@End Date: 
@Author: caijianfeng
"""
import torch

# from models.model import Glossification
from data.dataset import CSL_Dataset
import utils

import json
import argparse


utils.set_proxy()


def execute(input, part_program, target, add_id, copy_id, zero_id, p):
    """
    通过执行部分生成的 programs 生成对应的 gloss
    :param input: sentence -> shape = [i_len]
    :param part_program: min editing programs -> shape = [b, p_len]
    :param target:
    :param add_id:
    :param copy_id:
    :param zero_id:
    :param p:
    :return: corresponding glosses -> shape = [b, t_len]
    """
    if part_program[-2] in [add_id, copy_id]:
        if zero_id <= part_program[-1] <= zero_id + 9:
            target += input[p: p+part_program[-1]] if p+part_program[-1] < len(input) else input[:]
            p += min(part_program[-1], len(input) - p)
        else:
            target += input[p]
            p += 1
    return target, p


def inference(model, inputs, max_output_len, dataset):
    input_tokens = torch.tensor(dataset.tokenize([inputs])[0].ids).unsqueeze(0)
    # print('input shape: ', input_tokens.shape)
    model.eval()
    # print(inputs.shape, '; ', targets.shape, '; ', programs.shape)
    # predict = model.forward_origin(inputs, targets, programs)  # [b, p_len, p_vocab_size]
    begin_id = dataset.get_token_id('[CLS]')
    end_id = dataset.get_token_id('[SEP]')
    programs = [dataset.tokenize([''])[0].ids]
    target = ''
    p = 0
    for _ in range(max_output_len):
        if p >= len(inputs):
            break
        programs_tensor = torch.tensor(programs)
        target_tokens = torch.tensor(dataset.tokenize([target])[0].ids).unsqueeze(0)
        # print('program shape:', programs_tensor.shape, '; target shape: ', target_tokens.shape)
        predict = model(input_tokens, programs_tensor, target_tokens)[0]  # [p_len, p_vocab_size]
        pred = torch.argmax(predict[-1], dim=-1).item()
        # print(p, '; ', input_tokens.shape[-1])
        if pred == end_id:
            break
        programs[0].append(pred)
        target, p = execute(inputs, programs[0], target, dataset.get_token_id('加'), dataset.get_token_id('制'), dataset.get_token_id('0'), p)

    return dataset.decode(programs)[0], target


def main():
    args = argparse.ArgumentParser()

    args.add_argument('--train_args_path', type=str, default='./output/train.json')
    args.add_argument('--trained_model', type=str, default='./output/last/models/best_model.pt')
    args.add_argument('--input', type=str, required=True)
    args.add_argument('--dataset_path', type=str, default='./CSL_data/CSL-Daily_editing_chinese_test.txt')
    args.add_argument('--editing_casual_mask_file', type=str, default='./CSL_data/editing_casual_mask_CSL_174_40_test.npy')
    args.add_argument('--tokenizer_name', type=str, default='bert-base-chinese')
    args.add_argument('--max_output_len', type=int, default=20)
    infere_opt = args.parse_args()

    # with open(infere_opt.train_args_path, 'r', encoding='utf-8') as f:
    #     opt = json.load(f)

    # model = Glossification(
    #     opt.i_vocab_size,
    #     opt.p_vocab_size,
    #     opt.t_vocab_size,
    #     src_pad_idx=opt.src_pad_idx,
    #     trg_pad_idx=opt.trg_pad_idx,
    #     pro_pad_idx=opt.pro_pad_idx,
    #     head_num=opt.head_num,
    #     hidden_size=opt.embeddings_table.weight.shape[1],
    #     inner_size=opt.inner_size,
    #     dropout_rate=opt.dropout_rate,
    #     generator_encoder_n_layers=opt.generator_encoder_n_layers,
    #     generator_decoder_n_layers=opt.generator_decoder_n_layers,
    #     executor_encoder_n_layers=opt.executor_encoder_n_layers,
    #     share_target_embeddings=opt.share_target_embeddings,
    #     use_pre_trained_embedding=opt.use_pre_trained_embedding,
    #     pre_trained_embedding=opt.embeddings_table
    # )

    model = torch.load(infere_opt.trained_model, map_location=torch.device('cpu'))
    CSL_dataset = CSL_Dataset(dataset_file=infere_opt.dataset_path,
                              editing_casual_mask_file=infere_opt.editing_casual_mask_file,
                              pre_trained_tokenizer=True,
                              tokenizer_name=infere_opt.tokenizer_name)
    src = infere_opt.input

    print('model load succeed !')
    max_output_len = infere_opt.max_output_len
    program, target = inference(model, src, max_output_len, CSL_dataset)
    print('the predicted editing program: ', program)
    print('the predicted gloss: ', target)


if __name__ == '__main__':
    main()