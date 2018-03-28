#如果我们调用df.mean(axis=1),我们将得到按行计算的均值
>>> df.mean(axis=1)
0    1
1    2
2    3

# 如果我们调用 df.drop((name, axis=1),我们实际上删掉了一列，而不是一行
>>> df.drop("col4", axis=1)
   col1  col2  col3
0     1     1     1
1     2     2     2
2     3     3     3
