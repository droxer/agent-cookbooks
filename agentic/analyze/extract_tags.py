import jieba
import jieba.analyse
from rich.pretty import pprint

content = """确知信号
³ 概念：取值在任何时间都是确定的和可预知的信号；
³ 可以用数学公式表示它在任何时间的取值。
³ 分类：
® 周期信号与非周期信号；
® 能量信号与功率信号。
 随机信号
³ 概念：取值在任何时间都是随机的和不可预知的信号；
³ 无法用数学公式表示它的取值。
"""
top_k = 10

# TF-IDF algorithm
tags = jieba.analyse.extract_tags(content, topK=top_k, withWeight=True)
pprint(tags)

# TextRank algorithm
tags = jieba.analyse.textrank(content, topK=top_k, withWeight=True)
pprint(tags)