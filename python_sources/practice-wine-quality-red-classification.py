#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd


# In[ ]:


dataset = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')


# In[ ]:


dataset.head()


# In[ ]:


dataset.shape


# In[ ]:


dataset.info()


# In[ ]:


dataset.describe()
#In the below you can see there are outliers present in the dataset. Let's say for total sulfur dioxide, 75% percent of data is below 62 and max is 289


# In[ ]:


dataset.isnull().sum()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
dataset.hist(bins = 50, figsize = (16,10))
#checking the distribution
#dataset.plot(kind = 'box',subplots = True,layout = (4,3),figsize = (16,10))


# In[ ]:


#There is a outlier present checking with the boxplot
sns.boxplot(dataset['total sulfur dioxide'])


# In[ ]:


from scipy import stats


# In[ ]:


z = np.abs(stats.zscore(dataset))
filtered_entries = (z < 3).all(axis=1)
new_df = dataset[filtered_entries]


# In[ ]:


new_df.shape
#1599


# In[ ]:


(1599-1322)/1599
#17% percent of the dataset are outliers so it wont be a good idea to elimate the outliers


# In[ ]:


correlation = dataset.corr()
correlation['quality']


# In[ ]:


plt.figure(figsize = (15,8))
sns.heatmap(correlation,annot = True)


# In[ ]:


dataset.quality.value_counts()


# In[ ]:


bins = (2, 6.5, 8)
group_names = ['bad', 'good']
dataset['quality'] = pd.cut(dataset['quality'], bins = bins, labels = group_names)


# In[ ]:


sns.countplot(dataset.quality)


# In[ ]:


X= dataset.iloc[:,:-1]
y= dataset.iloc[:,-1]


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_quality = LabelEncoder()
y =label_quality.fit_transform(y)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C= 25, max_iter= 100, penalty= 'l2',)
classifier.fit(X_train,y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test,y_pred)*100
#This can be misleading as there is the dataset is not balanced


# In[ ]:


#Lets look at the classification report
#Precision = True positive/Actual results
#Recall = True positive/predicted results
#We have to decide based on the situation what to choose


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


#Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
clf = LogisticRegression()
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25],'max_iter':[100,1000,10000]}
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,scoring = 'accuracy')
grid_clf_acc.fit(X_train, y_train)

#Predict values based on new parameters
y_pred_acc = grid_clf_acc.predict(X_test)


# In[ ]:


grid_clf_acc.best_params_


# In[ ]:


#!pip install imblearn 


# In[ ]:


print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) 
  
# import SMOTE module from imblearn library 
# pip install imblearn (if you don't have imblearn in your system) 
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_res, y_res = sm.fit_sample(X, y) 

X_train_res,X_test,y_train_res,y_test_res = train_test_split(X_res,y_res,random_state = 0)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0))) 


# In[ ]:


classifier = LogisticRegression(C= 25, max_iter= 100, penalty= 'l2',)
classifier.fit(X_train_res,y_train_res)


# In[ ]:


y_pred = classifier.predict(X_test_res)


# In[ ]:


accuracy_score(y_test,y_pred)*100


# In[ ]:


#Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
clf = LogisticRegression()
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25],'max_iter':[100,1000,10000]}
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,scoring = 'accuracy')
grid_clf_acc.fit(X_train_res, y_train_res)

#Predict values based on new parameters
y_pred_acc = grid_clf_acc.predict(X_test_res)


# In[ ]:


grid_clf_acc.best_params_


# In[ ]:




