#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data= pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


data.head()


# In[ ]:


data.drop(['sl_no','specialisation','mba_p'],axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


import seaborn as sns
from matplotlib.pyplot import show
ag=sns.countplot(x="gender",data=data)


# In[ ]:


sns.countplot(x='gender', hue='status', data=data)


# In[ ]:


sns.countplot(x="gender",data=data)


# In[ ]:


ax=sns.countplot(x="gender",hue='status',data=data)
for p in ax.patches:
    height = p.get_height()
    width=p.get_x()+p.get_width()/2.
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height),
            ha="center") 



       
show()



#### Label


# In[ ]:


ax=sns.countplot(x="ssc_b",hue='status',data=data)
for p in ax.patches:
    height = p.get_height()
    width=p.get_x()+p.get_width()/2.
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height),
            ha="center") 
       
show()

######## one hot


# In[ ]:


ax=sns.countplot(x="hsc_s",hue='status',data=data)
for p in ax.patches:
    height = p.get_height()
    width=p.get_x()+p.get_width()/2.
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height),
            ha="center") 
       
show()


##########One hot #######


# In[ ]:


ax=sns.countplot(x="degree_t",hue='status',data=data)
for p in ax.patches:
    height = p.get_height()
    width=p.get_width()/2.
    ax.text(p.get_x()+width,
            height + 1,
            '{:1.2f}'.format(height),
            ha="center") 
       
show()

#### one hot


# In[ ]:


ax=sns.countplot(x="workex",hue='status',data=data)
for p in ax.patches:
    height = p.get_height()
    width=p.get_x()+p.get_width()/2.
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height),
            ha="center") 
show()

### label


# In[ ]:


total_count=data['gender'].groupby(data['gender']).count()
total_male_count=total_count['M']
total_female_count=total_count['F']


# In[ ]:


data['status'].unique().tolist()


# In[ ]:


plt.figure(figsize=(40,15))
ax=sns.countplot(x='salary', data=data, order = data['salary'].value_counts().index)
for p, label in zip(ax.patches, data["salary"].value_counts().index):
    ax.annotate(label, (p.get_x()-0.1, p.get_height()+0.15))


# In[ ]:


data['salary']=data['salary'].fillna(data['salary'].mean())


# In[ ]:


label=['status']
hot=['workex','gender','degree_t','ssc_b','hsc_s','hsc_b']
tr=['workex','gender','degree_t','ssc_b','hsc_s','hsc_b']


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data[label]=data[label].apply(le.fit_transform)
data[hot]=data[hot].apply(le.fit_transform)


# In[ ]:


fig, axes = plt.subplots(2,2)
axes[0,0].plot('ssc_b')
axes[0.1].plot('hsc_b')


# In[ ]:





# In[ ]:





# In[ ]:


pred=data[['status','salary']]


# In[ ]:


pred1=data[['status']]
pred2=data[['salary']]


# In[ ]:


X_data=data.drop(['status','salary'],axis=1)


# In[ ]:


dummy_data=X_data.join(pred1)


# In[ ]:


dummy_data.corr().sort_values(by='status',ascending=False)['status']


# In[ ]:


preprocess=['ssc_p','hsc_p','degree_p','etest_p']


# In[ ]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_data[preprocess] = min_max_scaler.fit_transform(X_data[preprocess])


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb


X_train, X_test, Y_train, Y_test = train_test_split(X_data, pred1, test_size=0.3)
clf = DecisionTreeClassifier(random_state=10)
clf.fit(X_train,Y_train)


# In[ ]:


from sklearn.model_selection import GridSearchCV

clf2 = xgb.XGBClassifier(random_state=10)
clf2.fit(X_train, Y_train)


# In[ ]:


X_test


# In[ ]:


from sklearn.metrics import precision_score, recall_score, accuracy_score

preds1 = clf.predict(X_test)

print("DT Precision = {}".format(precision_score(Y_test,preds1, average='macro')))
print("DT Recall = {}".format(recall_score(Y_test, preds1, average='macro')))
print("DT Accuracy = {}".format(accuracy_score(Y_test, preds1)))


# In[ ]:


from sklearn.metrics import precision_score, recall_score, accuracy_score

preds = clf2.predict(X_test)

print("XGB Precision = {}".format(precision_score(Y_test,preds, average='macro')))
print("XGB Recall = {}".format(recall_score(Y_test, preds, average='macro')))
print("XGB Accuracy = {}".format(accuracy_score(Y_test, preds)))


# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

auc = roc_auc_score(Y_test, preds)
print('XGB ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(Y_test, preds)
print("XGB Matrix", matrix)


# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

auc = roc_auc_score(Y_test, preds1)
print('DT ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(Y_test, preds1)
print("DT Matrix ",matrix)


# In[ ]:





# In[ ]:




