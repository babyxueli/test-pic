just for a good list 
# Python使用numpy和pandas模拟转盘抽奖游戏
# encoding = utf-8
import pandas as pd
import numpy as np
data = np.random.ranf(100000)
category = (0.0,0.08,0.3,1.0)
labels = ('一等奖','二等奖','三等奖')
results = pd.cut(data,category,labels=labels)
results = pd.value_counts(results)
print results

