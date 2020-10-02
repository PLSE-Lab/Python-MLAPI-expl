# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_d=pd.read_csv('../input/train_date.csv',nrows=180000)
train_d=train_d.dropna(axis=1,thresh=50000)
train_d=train_d.dropna(thresh=140)
train_n=pd.read_csv('../input/train_numeric.csv',nrows=180000)
train_n=train_n.dropna(axis=1,thresh=50000)
train_n=train_n.dropna(thresh=140)
train=pd.merge(train_n,train_d,on='Id')
train= train[np.isfinite(train['Response'])]
train=train.dropna(axis=1)
train1=train.loc[:,train.var()>15]
train1[['Response']]=train[['Response']]
print(list(train1.columns))
plt.plot(train1.ix[:,1],train1['Response'])
plt.tight_layout()
plt.title('Evolution by Province')
plt.show()
for num in range(10,20):
    print (num)
#print(train1['Response'])
#print(train.ix[0:1,:])
#print(train1.ix[0:1,:])
#print(train1['Response'].value_counts())
#test_d=pd.read_csv('../input/test_date.csv',nrows=100000)
#test_n=pd.read_csv('../input/test_numeric.csv',nrows=100000)
#test_d1=pd.read_csv('../input/test_date.csv',names=['Id','L3_S30_D3531', 'L3_S29_D3371', 'L3_S30_D3736',
#       'L3_S29_D3489', 'L3_S29_D3325'],skiprows=300000,dtype=object)
#print('3')
#test=pd.merge([test_n,test_d],on='Id')
#from sklearn import tree
#x=train1.ix[:, train1.columns != 'Response']
#y=train1['Response']
#factor = tree.DecisionTreeClassifier()
#factor=factor.fit(x, y)
#factor.score(x, y)
#with open("factor.dot", 'w') as decisiontree:
#   decisiontree= tree.export_graphviz(factor,out_file=decisiontree,
#   feature_names=x.columns)
#print('ok')
#rank = np.argsort(factor.feature_importances_)[::-1]
#print(rank[:7])
#print(x.columns[rank[:7]])
#from sklearn.linear_model import LogisticRegression
# Create logistic regression object
#model = LogisticRegression()
# Train the model using the training sets and check score
#x1=train1[['L3_S30_D3531', 'L3_S29_D3371', 'L3_S30_D3736',
#       'L3_S29_D3489', 'L3_S29_D3325']]
#y1=train1['Response']
#model.fit(x1, y1)
#model.score(x1, y1)
#Equation coefficient and Intercept
#print('Coefficient: \n', model.coef_)
#print('Intercept: \n', model.intercept_)
#Predict Output
#import matplotlib.pyplot as plt

#data0=train1[train1.Response==0]
#plt.scatter(data0['L3_S29_D3428'], data0['L3_S29_D3492'], color='b')
#data1=train1[train1.Response==1]
#plt.scatter(data1['L3_S29_D3428'], data1['L3_S29_D3492'], color='r')
#plt.legend()
#plt.savefig('fig.png')
#print(test[['L3_S30_D3531', 'L3_S29_D3371', 'L3_S30_D3736',
#       'L3_S29_D3489', 'L3_S29_D3325']])
#predicted= model.predict(test[['L3_S30_D3531', 'L3_S29_D3371', 'L3_S30_D3736',
#       'L3_S29_D3489', 'L3_S29_D3325']])
#test_d=pd.read_csv('../input/test_date.csv',nrows=10000)
#test_n=pd.read_csv('../input/test_numeric.csv',nrows=10000)
#import os
#os.unlink("factor.dot")
#open("factor.dot")
#import pydotplus
#dot_data = tree.export_graphviz(factor, out_file=None) 
#graph = pydotplus.graph_from_dot_data(dot_data) 
#rank = np.argsort(factor.feature_importances_)[::-1]






