# _*_ coding:utf-8 _*_
"""
@Project Name: glossification
@FileName: editing_casual_attention_mask.py
@Begin Date: 2024/3/12 19:19
@End Date: 2024/3/13 9:55
@Author: caijianfeng
"""
import numpy as np


def generate_editing_casual_mask(editing_programs, max_program_len, max_glosses_len):
    """
    给定 editing program 生成对应的 editing casual mask (批处理)
    主要是用于给定数据集，提前生成所有数据对(sentence, gloss)对应的 editing casual mask,
    后续使用时可以直接读取使用
    由于 每对数据对应的 mask 的维度为 [len_program, len_gloss], 而每对数据的长度都不一致;
    为了后续方便处理, 这里将 mask 的维度统一为 [max_program_len, max_glosses_len].
    TODO：原文中使用指针方式序列化生成, 能否实现并行化生成？
    :param editing_programs: type: list(list(str)) 一个小批量(mini batch) 的 editing program
    :param max_program_len: type: int 数据集中最长的 editing program 的长度
    :param max_glosses_len: type: int 数据集中最长的 gloss 的长度
    :return: editing casual mask: type: np.array(shape = [batch, max_program_len, max_glosses_len])
                                  给定小批量数据对应的 editing casual mask
    """
    mask = np.zeros([len(editing_programs), max_program_len, max_glosses_len])

    for i, editing_program in enumerate(editing_programs):
        index = 0
        for j, edit in enumerate(editing_program):
            if 'Copy' in edit:
                repeat_num = int(edit.split(' ')[1])
                mask[i, j, index + repeat_num:] = -1e-8
                index += repeat_num
            elif 'Add' in edit:
                mask[i, j, index + 1:] = -1e-8
                index += 1
            else:
                mask[i, j, index:] = -1e-8
    return mask


if __name__ == '__main__':
    # the example in origin paper
    max_program_len, max_glosses_len = 14, 9
    editing_programs = [
        ['Add 1', 'Add 2', 'Add 3', 'DEL 5', 'DEL 5', 'Copy 1', 'DEL 5', 'DEL', 'Copy 1', 'Add 4', 'Add 5', 'Add 6',
         'Add 7', 'Skip']]
    # assert len(editing_programs[0]) == max_program_len, f'the length of editing program is {len(editing_programs[0])}'

    mask = generate_editing_casual_mask(editing_programs,
                                        max_program_len,
                                        max_glosses_len)
    print(mask[0, :, :])
