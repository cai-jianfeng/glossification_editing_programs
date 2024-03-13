# _*_ coding:utf-8 _*_
"""
@Software: (已读)glossification
@FileName: utils.py
@Date: 2023/11/3 8:23
@Author: caijianfeng
"""


import shutil
import math

import torch
import torch.nn.functional as F


def execute(programs, targets):
    # programs.shape = [b, p_len]
    # targets.shape = [b, t_len]
    mask = torch.zeros([programs.size(), targets.size()[1]], dtype=torch.uint8,
                       device=targets.device)  # [b, p_len, t_len]
    for i, program in enumerate(programs):  # [p_len,]
        k = 0
        for j, action in enumerate(program):
            mask[i, j, k:] = 1
            if "ADD" in action or "COPY" in action:
                k = k + 1
    return mask  # [b, p_len, t_len]


def get_loss(pred, ans, vocab_size, label_smoothing, pad):
    """
    compute loss
    :param pred: [b * t_len, t_vocab_size]
    :param ans: [b * t_len]
    :param vocab_size: int, 表示预测的序列的 vocabulary size
    :param label_smoothing: float, 对 ground-truth 的 one-hot 向量进行 label smoothing
    :param pad: int, 表示 vocabulary 中 的 '<pad>' 的 index
    :return: float, 损失值
    """
    # took this "normalizing" from tensor2tensor. We subtract it for
    # readability. This makes no difference on learning.
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / float(vocab_size - 1)
    normalizing = -(
        confidence * math.log(confidence) + float(vocab_size - 1) *
        low_confidence * math.log(low_confidence + 1e-20))
    # scatter_(·,·,·): 第一个参数表示按照哪个维度进行映射 -> 1 表示按照行
    # 第二个参数表示映射的索引 -> 其中第 i 个元素值 ans_i 表示映射到 one_hot 第 i 行的第 ans_i 列
    # 第三个参数表示映射值 -> 将 one_hot 第 i 行的第 ans_i 列设置为 1
    one_hot = torch.zeros_like(pred).scatter_(1, ans.unsqueeze(1), 1)  # [b * t_len, t_vocab_size]
    one_hot = one_hot * confidence + (1 - one_hot) * low_confidence  # [b * t_len, t_vocab_size]
    log_prob = F.log_softmax(pred, dim=1)  # [b * t_len, t_vocab_size]

    xent = -(one_hot * log_prob).sum(dim=1)  # [b * t_len,]
    # mask_select: 取出 xent 中对应 mask 为 True 的值，注意最后返回的张量是 1 维的
    xent = xent.masked_select(ans != pad)  # 选择不是 '<pad>' 填充的预测结果
    loss = (xent - normalizing).mean()
    return loss


def get_accuracy(pred, ans, pad):
    pred = pred.max(1)[1]
    n_correct = pred.eq(ans)
    n_correct = n_correct.masked_select(ans != pad)
    return n_correct.sum().item() / n_correct.size(0)


def save_checkpoint(model, filepath, global_step, is_best):
    model_save_path = filepath + '/last_model.pt'
    torch.save(model, model_save_path)
    torch.save(global_step, filepath + '/global_step.pt')
    if is_best:
        best_save_path = filepath + '/best_model.pt'
        shutil.copyfile(model_save_path, best_save_path)


def load_checkpoint(model_path, device, is_eval=True):
    if is_eval:
        model = torch.load(model_path + '/best_model.pt')
        model.eval()
        return model.to(device=device)

    model = torch.load(model_path + '/last_model.pt')
    global_step = torch.load(model_path + '/global_step.pt')
    return model.to(device=device), global_step


def create_pad_mask(t, pad):
    mask = (t == pad).unsqueeze(-2)
    return mask


def create_trg_self_mask(target_len, device=None):
    # Prevent leftward information flow in self-attention.
    ones = torch.ones(target_len, target_len, dtype=torch.uint8,
                      device=device)  # [target_len, target_len]
    # torch.triu(·): 以 diagonal 为对角线返回上三角矩阵
    t_self_mask = torch.triu(ones, diagonal=1).unsqueeze(0)  # [1, target_len, target_len]

    return t_self_mask == 1  # [1, target_len, target_len]

def set_proxy(port=10809):
    import os
    os.environ['http_proxy'] = f'http://127.0.0.1:{port}'
    os.environ['https_proxy'] = f'http://127.0.0.1:{port}'

class Fraction:
    def __init__(self, numerator, denominator):
        self.numerartor = numerator
        self.denominator = denominator