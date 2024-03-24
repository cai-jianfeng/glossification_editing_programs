# _*_ coding:utf-8 _*_
"""
@Project Name: (已用已读)glossification
@FileName: metrics.py
@Begin Date: 2024/3/23 14:32
@End Date: 
@Author: caijianfeng
"""
from rouge import Rouge

# 实例化 ROUGE 计算器
rouge = Rouge()

# 假设你有参考摘要和生成摘要
reference_summary = ["对不起！", "你好！"]
generated_summary = ["对不起！", "你好！"]

# 计算 ROUGE 分数
scores = rouge.get_scores(generated_summary, reference_summary)

# 输出 ROUGE 分数
print(scores)
