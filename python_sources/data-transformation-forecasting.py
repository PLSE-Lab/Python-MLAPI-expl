#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')
test = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])


# In[ ]:


train.head(5)


# In[ ]:


train.tail(5)


# In[ ]:


train['Date'] = train['Date'].astype('int64')
test['Date'] = test['Date'].astype('int64')


# In[ ]:


train.tail(5)


# In[ ]:


train.info()


# ## date output seggregation

# In[ ]:


train.iloc[:,-3].sample(3)


# In[ ]:


X = train.iloc[:,-3]
print(X.shape)
X = np.array(X).reshape(-1,1)
print(X.shape)


# In[ ]:


Y = train.iloc[:,-2:]
print(Y.shape)
Y.sample(3)


# ## train-test split

# In[ ]:


from sklearn.model_selection import train_test_split 
trainX , valX, trainY, valY = train_test_split(X, Y, random_state=1)


# In[ ]:


y1Train = trainY.iloc[:,0]
print(y1Train.shape)
y1Train.sample(3)


# In[ ]:


y2Train = trainY.iloc[:,1]
y2Train.sample(3)


# In[ ]:


y1Val = valY.iloc[:,0]
y1Val.sample(3)


# In[ ]:


y2Val = valY.iloc[:,1]
y2Val.sample(3)


# In[ ]:


print(trainX.shape)


# ## model1 training : ConfirmedCases

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
lrModel1 = DecisionTreeRegressor(random_state = 27)
get_ipython().run_line_magic('time', 'lrModel1.fit(trainX, y1Train)')


# In[ ]:


get_ipython().run_line_magic('time', 'y1Pred = lrModel1.predict(valX)')
print(y1Pred[:,])


# In[ ]:


from sklearn.metrics import mean_absolute_error

print("Accuracy in train set : ", lrModel1.score(trainX, y1Train))
print("RMSE : ", mean_absolute_error(y1Val, y1Pred)**(0.5))


# ## model2 training : Fatalities

# In[ ]:


lrModel2 = DecisionTreeRegressor(random_state = 27)
get_ipython().run_line_magic('time', 'lrModel2.fit(trainX.reshape(-1, 1), y2Train)')

get_ipython().run_line_magic('time', 'y2Pred = lrModel2.predict(valX)')

print("Accuracy in train set : ", lrModel2.score(trainX, y2Train))
print("RMSE : ", mean_absolute_error(y2Val, y2Pred)**(0.5))


# ## taking on test data

# In[ ]:


print(test.shape)
test.sample(3)


# In[ ]:


forecastID = test.iloc[:,0]


# In[ ]:


test.iloc[:,-1].sample(3)


# In[ ]:


test = np.array(test.iloc[:,-1]).reshape(-1,1)


# In[ ]:


get_ipython().run_line_magic('time', 'finalPred1 = lrModel1.predict(test)')
print(finalPred1[:,])


# In[ ]:


get_ipython().run_line_magic('time', 'finalPred2 = lrModel2.predict(test)')
print(finalPred2[:,])


# In[ ]:


outputFile = pd.DataFrame({"ForecastId": forecastID,
                           "ConfirmedCases": (finalPred1+0.5).astype('int'),
                           "Fatalities": (finalPred2+0.5).astype('int')})


# In[ ]:


outputFile.sample(3)


# In[ ]:


outputFile.to_csv("submission.csv", index=False)

