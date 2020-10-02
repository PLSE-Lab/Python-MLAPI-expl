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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data =pd.read_csv('../input/voice.csv')
data.head() # will give top five rows 


# In[ ]:


#import libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data.tail() # give last five rows 


# In[ ]:


data.shape # print shape(row, column)


# In[ ]:


data.describe()


# In[ ]:


data.info()  # check what kind of data are 


# In[ ]:


data.isnull().values.any()  # check data has any values null/nan or not


# In[ ]:


column_names = data.columns.tolist()
column_names   # print columns name


# In[ ]:


correlation = data.corr()
correlation


# In[ ]:


# plot correlation matrix
f, ax = plt.subplots(figsize = (8,8))
# Draw the heatmap using seaborn
sns.heatmap(correlation, square = True)
plt.show()


# In[ ]:


#Box plot see comparision in labels by other features
data.boxplot(column= 'meanfreq', by='label', grid=False)


# In[ ]:


#Check how many datas are of male and female
data[data['label'] == 'male'].shape[0]    
#male


# In[ ]:


#female
data[data['label'] == 'female'].shape[0] 


# In[ ]:


# for checking difference between male and female
a = data[data['label'] == 'male'].mean()
a


# In[ ]:


b = data[data['label'] == 'female'].mean()
b


# In[ ]:


#Distribution of male and female
sns.FacetGrid(data, hue="label", size=6)    .map(sns.kdeplot, "meanfreq")    .add_legend()
plt.show()


# In[ ]:


sns.FacetGrid(data, hue="label", size=6)    .map(sns.kdeplot, "IQR")    .add_legend()
plt.show()


# In[ ]:


data.plot(kind='scatter', x='meanfreq', y='dfrange')
data.plot(kind='kde', y='meanfreq')


# In[ ]:


from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


# In[ ]:


# convert srting data into numberic eg. male 1, female 0
df = data
df = df.drop(['label'],axis = 1)
X = df.values
y = data['label'].values

# only one column has object type so we encode it

encoder = LabelEncoder()
y = encoder.fit_transform(y)


# In[ ]:


# 70-30% of train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30)


# In[ ]:


#Random forest with default hyperparameter
rand_forest = RandomForestClassifier()
rand_forest.fit(Xtrain, ytrain)
y_pred = rand_forest.predict(Xtest)


# In[ ]:


print(metrics.accuracy_score(ytest, y_pred))


# In[ ]:




