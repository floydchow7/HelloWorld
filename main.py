import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#https://ajeyvenkataraman.com/2020/03/04/dealing-with-heteroscedasticity-in-python/
df = pd.read_csv('./data/data1.csv')
x = df.loc[:,['x']]
y = df.loc[:,['y']]
plt.figure()
plt.scatter(x,y)
plt.show()
import statsmodels.formula.api as sm

simple = sm.ols(formula='y~x',data=df).fit()
#print(simple.summary())
df2 = pd.concat([x,y,simple.resid],axis=1)


import statsmodels.stats.diagnostic
white_test=statsmodels.stats.diagnostic.het_white(simple.resid,simple.model.exog)
#define labels to use for output of White's test
labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']

#print results of White's test
print(dict(zip(labels, white_test)))
#通过white检验可以发现p值小于0.05，因为拒绝零假设，异方差性存在
plt.figure(figsize=(20,10))
plt.scatter(df['x'],simple.resid)
plt.xlabel('x value')
plt.ylabel('residual')
plt.show()

#下面对异方差性进行处理,根据残差图可以假设残差的平方与x成正比
#在损失函数上使用L1损失函数以及huber损失函数来进行估计

