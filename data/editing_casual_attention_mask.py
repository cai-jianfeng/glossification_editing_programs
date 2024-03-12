# _*_ coding:utf-8 _*_
"""
@Software: (已用已读)glossification
@FileName: editing_casual_attention_mask.py
@Date: 2024/3/12 19:19
@Author: caijianfeng
"""
import numpy as np


def generate_editing_casual_mask(editing_programs, max_program_len, max_glosses_len):
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
    # data_file = 'CSL-Daily_preporcess.txt'  # the max length of editing program in CSL dataset is 42.
    # with open(data_file, 'r', encoding='utf-8') as f:
    #     data_lines = f.readlines()
    # editing_programs = [data_line.replace('\n', '').split('; ') for data_line in data_lines]
    # the example in origin paper
    max_program_len, max_glosses_len = 14, 9
    editing_programs = [
        ['Add 1', 'Add 2', 'Add 3', 'DEL 5', 'DEL 5', 'Copy 1', 'DEL 5', 'DEL', 'Copy 1', 'Add 4', 'Add 5', 'Add 6',
         'Add 7', 'SKIP']]
    assert len(editing_programs[0]) == max_program_len, f'the length of editing program is {len(editing_programs[0])}'

    mask = generate_editing_casual_mask(editing_programs,
                                        max_program_len,
                                        max_glosses_len)
    print(mask[0, :, :])
