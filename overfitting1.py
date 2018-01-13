# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 19:28:21 2018

@author: Administrator
"""

from sklearn.learning_curve import learning_curve
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
#导入数据预处理，包括标准化处理或正则处理
from sklearn import preprocessing
#样本平均测试，评分更加
from sklearn.cross_validation import cross_val_score
 
from sklearn import datasets
#导入knn分类器
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


#excel文件名
fileName="data.xlsx"
#读取excel
df=pd.read_excel(fileName)
# data为Excel前几列数据
x1=df[df.columns[:4]]
#标签为Excel最后一列数据
y1=df[df.columns[-1:]]
 
#把dataframe 格式转换为阵列
x1=np.array(x1)
y1=np.array(y1)
#数据预处理，否则计算出错
y1=[i[0] for i in y1]
y1=np.array(y1)

#创建一个knn分类器
knn=KNeighborsClassifier() 
#svc=SVC()

train_sizes,train_loss,test_loss=learning_curve(knn,x1,y1,cv=10,
                                                scoring='accuracy', train_sizes=[0.1,0.25,0.5,0.75]) 
                                            
 
train_loss_mean=-np.mean(train_loss,axis=1)
test_loss_mean=-np.mean(test_loss,axis=1)

plt.plot(train_sizes,train_loss_mean,'o-',color='r',label='Training')
plt.plot(train_sizes,test_loss_mean,'o-',color='g',label='Cross-validation')

plt.xlabel("Traing examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
