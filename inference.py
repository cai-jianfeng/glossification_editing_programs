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
    while i < len(program):
        if program[i] == '加' and i < len(program) - 1:
            target += program[i+1]
            i += 2
        elif program[i] == '删' and p < len(input):
            p += int(program[i + 1])
            i += 2
        elif program[i] == '贴' and p < len(input):
            num = min(int(program[i + 1]), len(input) - p)
            target += input[p:p + num]
            p += num
            i += 2
        elif program[i] == '过':
            break
        else:
            i += 2
    return target


def inference(model, inputs, max_output_len, dataset, opt):
    # [1, i_len]
    input_tokens = torch.tensor(dataset.tokenize([inputs])[0].ids).unsqueeze(0)
    model.eval()
    begin_id = dataset.get_token_id('[CLS]')
    end_id = dataset.get_token_id('[SEP]')
    programs = [dataset.get_token_id('[CLS]')]  # programs 初始时是占位符
    target = [dataset.get_token_id('[CLS]')]
    for i in range(max_output_len):
        programs_tensor = torch.tensor(programs).unsqueeze(0)
        target_tensor = torch.tensor(target).unsqueeze(0)
        # print('program shape:', programs_tensor.shape, '; target shape: ', target_tokens.shape)
        pred_edit_op, pred_edit_num = model(input_tokens, programs_tensor, target_tensor, torch.zeros([1, len(programs), len(target)]) == 1)  # [1, p_len, edit_num/p_vocab_size]
        pred_edit_op, pred_edit_num = pred_edit_op[0], pred_edit_num[0]  # [p_len, edit_num/p_vocab_size]
        if i % 2 == 1:
            pred = torch.argmax(pred_edit_op[-1], dim=-1).item()
            pred = opt.edit_op[pred]
        else:
            if i != 0:
                # program[-1] 是占位符
                if programs[-2] != opt.edit_op[0]:  # 当前 edit op = '删'/'贴'
                    pred = opt.edit_num[torch.argmax(pred_edit_num[-1][opt.edit_num], dim=-1).item()]  # 只取 1 ~ 9 中的概率最大值
                else:  # 当前 edit op = '加'
                    pred = torch.argmax(pred_edit_num[-1], dim=-1).item()
                    if pred in [begin_id, end_id]:  # 当 加 后的结果为 [CLS] 和 [SEP]
                        _, indices = torch.topk(pred_edit_num[-1], k=3, dim=-1)  # 取除了 [CLS] 和 [SEP] 的其他结果
                        pred = indices[-1] if indices[1] in [begin_id, end_id] else indices[1]  # 前两个结果均为 [CLS] 和 [SEP] 时取第三个结果
                    #     pred = opt.edit_num[0]
                    #     programs[-2] = opt.edit_op[2] if programs[-2] == opt.edit_op[0] else programs[-2]
                    # elif pred not in opt.edit_num and programs[-2] != opt.edit_op[0]:
                    #     pred = opt.edit_num[0]
                    # elif programs[-2] == opt.edit_op[0] and pred in opt.edit_num:
                    #     programs[-2] = opt.edit_op[2]
            else:  # i=0 时是预测 [CLS]
                pred = torch.argmax(pred_edit_num[-1], dim=-1).item()
        programs.insert(-1, pred)
        if pred in [end_id, opt.edit_op[-1]]:
            break
        if i and i % 2 == 0:
            programs.pop()  # 执行时需要剔除占位符
            target = execute(inputs, dataset.decode([programs[:]])[0])
            programs.append(dataset.get_token_id('[CLS]'))  # 执行结束后需要加回占位符
            target = dataset.tokenize(['[CLS]' + target])[0].ids  # 将执行得到的 target 添加 [CLS] 并 tokenize

    programs.pop()  # programs 的最后一个是我们的占位符
    return dataset.decode([programs[:]])[0]


def main():
    args = argparse.ArgumentParser()

    args.add_argument('--train_args_path', type=str, default='./output/train.json')
    args.add_argument('--trained_model', type=str, default='./output/last/models/best_model.pt')
    args.add_argument('--input', type=str, required=True)
    args.add_argument('--dataset_path', type=str, default='./CSL_data/CSL-Daily_editing_chinese_test.txt')
    args.add_argument('--tokenizer_name', type=str, default='bert-base-chinese')
    args.add_argument('--max_output_len', type=int, default=50)
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
