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

import evaluate

utils.set_proxy()


def execute(targets, programs, add_id, copy_id):
    """
    通过执行 programs 生成 editing casual attention 的 mask
    :param targets: glosses -> shape = [b, t_len]
    :param programs: min editing programs -> shape = [b, p_len]
    :return: editing casual attention mask -> shape = [b, p_len, t_len]
    """
    mask = torch.zeros(programs.shape[0], programs.shape[1], targets.shape[1])
    for i, program in enumerate(programs):
        generator_pointer = 0
        for j, edit in enumerate(program):
            if add_id or copy_id in edit:
                index = int(edit.split(' ')[1])
                mask[i][generator_pointer: generator_pointer + index] = -1e-8
                generator_pointer += index
    return mask

def inference(model, inputs, targets, programs, edit_mask, dataset, evaluator):
    model.eval()
    # print(inputs.shape, '; ', targets.shape, '; ', programs.shape)
    # predict = model.forward_origin(inputs, targets, programs)  # [b, p_len, p_vocab_size]
    predict = model(inputs, targets, programs, edit_mask)
    pred = torch.argmax(predict, dim=-1)  # [b, p_len]
    pred_str = dataset.decode(pred.tolist())  # [b , str(len=p_len)]
    program_str = dataset.decode(programs.tolist())  # [b , str(len=p_len)]
    # print(pred_str, '; ', program_str)
    programs_str = [[pro] for pro in program_str]  # [b, 1, str(len=p_len)]
    # predictions is list[str]; references is list[list[str]]
    results = evaluator.compute(predictions=pred_str, references=programs_str)
    return results


def main():
    args = argparse.ArgumentParser()

    args.add_argument('--train_args_path', type=str, default='./output/train.json')
    args.add_argument('--trained_model', type=str, default='./output/last/models/best_model.pt')

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
    CSL_dataset = CSL_Dataset(dataset_file='./CSL_data/CSL-Daily_editing_chinese_test.txt',
                              editing_casual_mask_file='./CSL_data/editing_casual_mask_CSL_174_40_test.npy',
                              pre_trained_tokenizer=True,
                              tokenizer_name='bert-base-chinese')
    test_data = CSL_dataset[-1]
    src, trg, pro, edit_mask = test_data.values()
    src, trg, pro, edit_mask = src.unsqueeze(0), trg.unsqueeze(0), pro.unsqueeze(0), torch.from_numpy(edit_mask).unsqueeze(0)
    evaluator_bleu = evaluate.load("bleu")

    print('model load succeed !')

    metric_results = inference(model, src, trg, pro, edit_mask, CSL_dataset, evaluator_bleu)
    bleu = metric_results['bleu']
    scores = metric_results['precisions']
    print(f'average bleu scores: {bleu}; each bleu score (from 1 to 4): {scores}')


if __name__ == '__main__':
    main()