# _*_ coding:utf-8 _*_
"""
@Project Name: (已用已读)glossification
@FileName: inference.py
@Begin Date: 2024/3/23 15:13
@End Date: 
@Author: caijianfeng
"""
import torch

from data.dataset import CSL_Dataset
import utils

import argparse

utils.set_proxy()


def execute(input: str, program: str):
    """
    通过执行预测的 program 生成最终的 gloss
    :param input: sentence: shape: [i_len] -> type: str
    :param program: editing program: shape: [p_len] -> type: str
    :return: target gloss: shape: [t_len] -> type: str
    """
    program = program.replace(' ', '')
    input = input.replace(' ', '')
    # print(program, '; ', input)
    i, p = 0, 0
    target = ''
    while i < len(program) and p < len(input):
        if program[i] == '加':
            target += program[i+1]
            i += 2
        elif program[i] == '删':
            p += int(program[i + 1])
            i += 2
        elif program[i] == '贴':
            num = min(int(program[i + 1]), len(input) - p)
            target += input[p:p + num]
            p += num
            i += 2
        elif program[i] == '过':
            break
    return target


def inference(model, inputs, max_output_len, dataset, opt):
    # [1, i_len]
    input_tokens = torch.tensor(dataset.tokenize([inputs])[0].ids).unsqueeze(0)
    model.eval()
    begin_id = dataset.get_token_id('[CLS]')
    end_id = dataset.get_token_id('[SEP]')
    programs = [dataset.get_token_id('[CLS]')]  # programs 初始时是占位符
    for i in range(max_output_len):
        programs_tensor = torch.tensor(programs).unsqueeze(0)
        # print('program shape:', programs_tensor.shape, '; target shape: ', target_tokens.shape)
        pred_edit_op, pred_edit_num = model(input_tokens, programs_tensor)  # [1, p_len, edit_num/p_vocab_size]
        pred_edit_op, pred_edit_num = pred_edit_op[0], pred_edit_num[0]  # [p_len, edit_num/p_vocab_size]
        if i % 2 == 1:
            pred = torch.argmax(pred_edit_op[-1], dim=-1).item()
            pred = opt.edit_op[pred]
        else:
            pred = torch.argmax(pred_edit_num[-1], dim=-1).item()
            if i != 0:
                if pred == begin_id:
                    pred = opt.edit_num[0]
                elif pred not in opt.edit_num and programs[-2] != opt.edit_op[0]:
                    pred = opt.edit_num[0]
        programs.insert(-1, pred)
        if pred in [end_id, opt.edit_op[-1]]:
            break
    programs.pop()  # programs 的最后一个是我们的占位符
    return dataset.decode([programs[:]])[0]


def main():
    args = argparse.ArgumentParser()

    args.add_argument('--train_args_path', type=str, default='./output/train.json')
    args.add_argument('--trained_model', type=str, default='./output/last/models/best_model.pt')
    args.add_argument('--input', type=str, required=True)
    args.add_argument('--dataset_path', type=str, default='./CSL_data/CSL-Daily_editing_chinese_test.txt')
    args.add_argument('--tokenizer_name', type=str, default='bert-base-chinese')
    args.add_argument('--max_output_len', type=int, default=20)
    infere_opt = args.parse_args()

    model = torch.load(infere_opt.trained_model, map_location=torch.device('cpu'))
    # print(type(model))
    CSL_dataset = CSL_Dataset(dataset_file=infere_opt.dataset_path,
                              pre_trained_tokenizer=True,
                              tokenizer_name=infere_opt.tokenizer_name)
    src = infere_opt.input.replace(' ', '')
    infere_opt.edit_op = [CSL_dataset.get_token_id('加'),
                          CSL_dataset.get_token_id('删'),
                          CSL_dataset.get_token_id('贴'),
                          CSL_dataset.get_token_id('过')]
    infere_opt.edit_num = [CSL_dataset.get_token_id('1'),
                           CSL_dataset.get_token_id('2'),
                           CSL_dataset.get_token_id('3'),
                           CSL_dataset.get_token_id('4'),
                           CSL_dataset.get_token_id('5'),
                           CSL_dataset.get_token_id('6'),
                           CSL_dataset.get_token_id('7'),
                           CSL_dataset.get_token_id('8'),
                           CSL_dataset.get_token_id('9')]
    print('model load succeed !')
    print('the origin sentence: ', src)
    max_output_len = infere_opt.max_output_len
    program = inference(model, src, max_output_len, CSL_dataset, infere_opt)
    print('the predicted editing program: ', program)
    target = execute(src, program)
    print('the predicted gloss: ', target)


if __name__ == '__main__':
    main()
