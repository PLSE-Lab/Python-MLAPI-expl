#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # using SVM to predict if a stock will rise based on previous information

# In[ ]:


#using svm to predict stock
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm,preprocessing 
from sklearn.metrics import classification_report
stock_prices = pd.read_csv(r'../input/prices.csv')
symbols = list(set(stock_prices['symbol']))


# In[ ]:


msft_prices = stock_prices[stock_prices['symbol']== 'MSFT']
msft_prices = msft_prices[['date','open','low','high','close','volume']]
msft_prices.to_csv('msft_prices.csv',sep='\t')
msft_dates = [pd.Timestamp(date) for date in msft_prices['date']]


# In[ ]:


msft_close = np.array(msft_prices['close'],dtype='float')
import matplotlib.pyplot as plt
plt.title('MSFT')
plt.scatter(msft_dates,msft_close)
plt.show()


# In[ ]:


msft_prices = msft_prices.set_index('date')


# In[ ]:


def get_x_and_y(price,window_length=7,predict_day_length=1):
    '''get train and test set
    every time get window from price and
    '''
    m = len(price.iloc[0])
    n = len(price) - window_length
    m = window_length * m

    x = np.ones((n,m))
    y = np.ones((n,1))

    for i in range(len(price)-window_length):
        ans = [list(price.iloc[j] for j in range(i,i+window_length))]
        ans = np.array(ans).flatten()
        x[i] = ans
        y[i] = 1 if price.close[i+window_length+predict_day_length-1] - price.close[i+window_length-1] >0 else 0
    return [x,y]


# In[ ]:


def train_and_test(price,window_length,accurarys,reports):
    x,y = get_x_and_y(msft_prices,window_length=window_length)
    y = y.flatten()
    scaler = preprocessing.StandardScaler()
    scaler.fit_transform(x)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=233)
    for kernel_arg in ['rbf','poly','linear']:
        clf = svm.SVC(kernel=kernel_arg,max_iter=5000)
        clf.fit(x_train,y_train)
        y_predict = clf.predict(x_test)

        accurary = clf.score(x_test,y_test)
        report = classification_report(y_test,y_predict,target_names = ['drop','up'])
        if window_length in accurarys:
            accurarys[window_length].append(accurary)
            reports[window_length].append(report)
        else: 
            accurarys[window_length] = [accurary]
            reports[window_length] = [report]
        print('The Accurary of %s : %f'%(kernel_arg,clf.score(x_test,y_test)))
        print(report)


# In[ ]:


window_lengths = [7,14,21,30,60,90,120,150,180]
accurarys = {}
reports ={}

for l in window_lengths:
    print('window_length:',l)
    train_and_test(msft_prices,l,accurarys,reports)


# we can see the accurary of svm is about 50%~60%
# I don't think it's a good way to predict, but as we know there is no way can predict stock market well since it was influenced by many factors which not just history price.

# In[ ]:




