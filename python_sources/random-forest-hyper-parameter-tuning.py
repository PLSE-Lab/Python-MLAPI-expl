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
os.listdir('../input/learn-together')

# Any results you write to the current directory are saved as output.


# In[ ]:


traindata = pd.read_csv('../input/learn-together/train.csv')
traindata.head()


# In[ ]:


np.where(np.isnan(traindata))


# In[ ]:


y = traindata['Cover_Type']
X = traindata
del X['Cover_Type']
del X['Id']

column_name = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
X[column_name] = X[column_name].apply(lambda x:(x - x.min()) / (x.max() - x.min()))
X.columns


# In[ ]:


from sklearn import svm
clf = svm.SVC()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=640
                            ,n_jobs = -1,max_features=.2,oob_score=True)


# In[ ]:


clf.fit(X,y)


# In[ ]:


clf.oob_score_


# In[ ]:


test = pd.read_csv('../input/learn-together/test.csv')
test.columns


# In[ ]:


column_name = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
test[column_name] = test[column_name].apply(lambda x:(x - x.min()) / (x.max() - x.min()))


# In[ ]:


test_X = test['Id']
del test['Id']
predicted = clf.predict(test)


# In[ ]:


# test['Id']


# In[ ]:


my_submission = pd.DataFrame({'Id': pd.read_csv('../input/learn-together/test.csv')['Id'], 'Cover_Type': predicted})

my_submission.to_csv('submission_data_cleaned.csv', index=False)


# In[ ]:


my_submission.tail()

