# _*_ coding:utf-8 _*_
"""
@Project Name: glossification
@FileName: test.py
@Begin Date: 2023/12/2 11:10
@Author: caijianfeng
"""
import numpy as np
from extract_editoring_program import min_edit_distance, min_edit_trajectory, compress_trajectory, min_edit_program_parallel

data_file_name = 'CSL-Daily.txt'

with open(data_file_name, 'r', encoding='utf-8') as file:
    data_lines = file.readlines()

# data_lines_gloss = np.array([[c for c in data_line.split('|')[3].replace(' ', '')] for data_line in data_lines])
data_lines_gloss = [[c for c in data_line.split('|')[3].replace(' ', '')] for data_line in data_lines][1:]
# data_lines_sentence = np.array([[c for c in data_line.split('|')[4].replace(' ', '')] for data_line in data_lines])
data_lines_sentence = [[c for c in data_line.split('|')[4].replace(' ', '')] for data_line in data_lines][1:]

index = 16
print(data_lines_sentence[index], '; ', data_lines_gloss[index])

edit_state = min_edit_distance(sentence=data_lines_sentence[index],
                               gloss=data_lines_gloss[index])
print(edit_state)

edit_trajectory = min_edit_trajectory(state=edit_state,
                                      sentence=data_lines_sentence[index],
                                      gloss=data_lines_gloss[index])
print(edit_trajectory)

compress_edit_trajectory = compress_trajectory(edit_program=edit_trajectory)

print(compress_edit_trajectory)