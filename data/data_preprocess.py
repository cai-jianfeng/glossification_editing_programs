# _*_ coding:utf-8 _*_
"""
@Project Name: glossification
@FileName: data_preprocess.py
@Begin Date: 2024/3/12 10:40
@End Date: 2024/3/13 9:54
@Author: caijianfeng
"""
import numpy as np
from extract_editoring_program import min_edit_distance, min_edit_trajectory, compress_trajectory, \
    min_edit_program_parallel
from editing_casual_attention_mask import generate_editing_casual_mask

data_file_name = 'CSL-Daily.txt'

with open(data_file_name, 'r', encoding='utf-8') as file:
    data_lines = file.readlines()

# data_lines_gloss = np.array([[c for c in data_line.split('|')[3].replace(' ', '')] for data_line in data_lines])
data_lines_gloss = [[c for c in data_line.split('|')[3].replace(' ', '')] for data_line in data_lines][1:]
# data_lines_sentence = np.array([[c for c in data_line.split('|')[4].replace(' ', '')] for data_line in data_lines])
data_lines_sentence = [[c for c in data_line.split('|')[4].replace(' ', '')] for data_line in data_lines][1:]

compress_edit_trajectorys = min_edit_program_parallel(sentences=data_lines_sentence,
                                                      glosses=data_lines_gloss)
compress_edit_trajectorys_lines = ['; '.join(compress_edit_trajectory) + '\n' for compress_edit_trajectory in
                                   compress_edit_trajectorys]

# data_preprocess_file_name = 'CSL-Daily_preprocess.txt'

# with open(data_preprocess_file_name, 'w', encoding='utf-8') as f:
#     f.writelines(compress_edit_trajectorys_lines)

# data_preprocess_file_name = 'CSL-Daily_editing_chinese.txt'
#
# with open(data_preprocess_file_name, 'w', encoding='utf-8') as f:
#     # f.writelines(compress_edit_trajectorys_lines)
#     f.write(data_lines[0].replace('\n', '') + '|editing program\n')
#     for i, (origin_data, compress_edit_trajectory) in enumerate(zip(data_lines[1:], compress_edit_trajectorys_lines)):
#         f.write('|'.join([origin_data.replace('\n', ''), compress_edit_trajectory]))

editing_casual_mask_file = 'editing_casual_mask_CSL_174_40.npy'  # 命名规则: editing_casual_mask + _数据集名称 + _max editing program length + _max gloss length
max_program_len, max_glosses_len = 174, 40  # the max length of editing program and glosses in CSL dataset are 42 and 38.

editing_casual_mask = generate_editing_casual_mask(editing_programs=compress_edit_trajectorys,
                                                   max_program_len=max_program_len,
                                                   max_glosses_len=max_glosses_len)

np.save(editing_casual_mask_file, editing_casual_mask)

print('save end !')
