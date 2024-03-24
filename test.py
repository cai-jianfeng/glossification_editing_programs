# _*_ coding:utf-8 _*_
"""
@Project Name: (已用已读)glossification
@FileName: test.py
@Begin Date: 2024/3/23 15:23
@End Date: 
@Author: caijianfeng
"""
import argparse

arg = argparse.ArgumentParser()
arg.add_argument('--test', type=str, default='hello')
arg.add_argument('--test2', type=int, default=4)
arg.add_argument('--test3', action='store_true')

opt = arg.parse_args()

print(opt.__dict__)