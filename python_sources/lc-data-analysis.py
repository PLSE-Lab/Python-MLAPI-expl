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


# In[ ]:


import numpy as np
import pandas as pd
import os


# In[ ]:


data = pd.read_csv('../input/lc_2016_2017.csv')


# In[ ]:


data.head(10)


# In[ ]:


data.tail()


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


data.loan_status.value_counts()


# In[ ]:


data.loan_status.unique()


# In[ ]:


data.loan_status.nunique()


# In[ ]:


do ={'Late (31-120 days)':1,'Late (16-30 days)':1,'Default':1,'Current':0,'Fully Paid':0,'Charged Off':0,'In Grace Period':0}


# In[ ]:


do


# In[ ]:


data['Target'] = data['loan_status'].map(do)


# In[ ]:


data.Target.value_counts()


# In[ ]:


def Tar(var):
    if var == 'Late (31-120 days)' or var == 'Late (16-30 days)' or var == 'Default':
        return(1)
    else:
        return(0)


# In[ ]:


data.apply(lambda x: sum(x.isnull()))


# In[ ]:


data_1 = data.apply(lambda x: sum(x.isnull()))


# In[ ]:


type(data_1)


# In[ ]:


data.apply(lambda x: sum(x.isnull()))/data.shape[0]


# In[ ]:


data.apply(lambda x: sum(x.isnull()))/data.shape[0] * 100


# In[ ]:


data_2 = data.apply(lambda x: sum(x.isnull()))/data.shape[0]


# In[ ]:


data_2 > [.6] 


# In[ ]:


data_2[data_2 > .6] 


# In[ ]:


data_2[data_2 > .6].index


# In[ ]:


miss_gt_60 =list(data_2[data_2 > .6].index)


# In[ ]:


miss_gt_60


# In[ ]:


#data.drop('member_id',axis = 1,inplace = True)


# In[ ]:


def miss(var):
    if var == var:
        return(0)
    else:
        return(1)


# In[ ]:


miss(np.NaN)


# In[ ]:


#data['desc_1']= data['desc'].apply(fun)


# In[ ]:


for i in miss_gt_60:
    print(i)


# In[ ]:


for i in miss_gt_60:
     data[i+'_1']= data[i].apply(miss)


# In[ ]:


data[miss_gt_60].isnull().sum()


# In[ ]:


tr =[]
for i in miss_gt_60:
    tr.append(i+'_1')


# In[ ]:


data[tr].sum()


# In[ ]:


data.drop(miss_gt_60,axis=1,inplace=True)


# In[ ]:


miss = data.apply(lambda x: sum(x.isnull()))/data.shape[0]
miss_vars =list(miss[miss> 0].index)


# In[ ]:


miss


# In[ ]:


miss_vars


# In[ ]:


cat_miss = []
num_miss = []
for i in miss_vars:
        if data[i].dtype == 'O':
            cat_miss.append(i)
        else:
            num_miss.append(i)


# In[ ]:


cat_miss


# In[ ]:


num_miss


# In[ ]:


for i in num_miss:
    data[i].fillna(0,inplace=True)
    


# In[ ]:


data[num_miss].isnull().sum()


# In[ ]:


for i in cat_miss:
    data[i].fillna('missing',inplace=True)


# In[ ]:


data[cat_miss].isnull().sum()


# In[ ]:


data.isnull().sum()/data.shape[0]


# In[ ]:


data.describe(include ='all',percentiles =[.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99])


# In[ ]:


data.describe(include ='all',percentiles =[.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99]).transpose().to_csv('sec.csv')


# In[ ]:


cat_ = []
num_ = []
for i in data.columns:
        if data[i].dtype == 'O':
            cat_.append(i)
        else:
            num_.append(i)


# In[ ]:


cat_


# In[ ]:


num_


# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.heatmap(data[num_].corr())


# In[ ]:


data[num_].corr()


# In[ ]:


data[num_].corr().to_csv('seccorr.csv')


# In[ ]:


cat_lt10=[]
cat_gt10=[]
for i in cat_:
    print(i,':',data[i].nunique())
    if data[i].nunique() <=10:
        cat_lt10.append(i)
    else:
        cat_gt10.append(i)
    


# In[ ]:


cat_lt10


# In[ ]:


cat_gt10


# In[ ]:


data[cat_lt10].head(10)


# In[ ]:


data[cat_gt10].head(10)


# In[ ]:


grade = pd.get_dummies(data['grade'])


# In[ ]:


grade


# In[ ]:


grade = pd.get_dummies(data['grade'],drop_first=True)


# In[ ]:


grade.head()


# In[ ]:


for i in cat_lt10:
    print('processed:',i)
    i = pd.get_dummies(data[i],drop_first=True)
    


# In[ ]:


grade.head()


# In[ ]:


data.shape


# In[ ]:


for i in cat_lt10:
    print('processed:',i)
    data = pd.concat([data,pd.get_dummies(data[i],drop_first=True)],axis=1)


# In[ ]:


data.shape


# In[ ]:


data.drop(cat_lt10,axis =1,inplace=True)


# In[ ]:


data.shape


# In[ ]:


len(cat_lt10)


# In[ ]:


cat_gt10


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in cat_gt10:
    print('processing:',i)
    data[i] = le.fit_transform(data[i])


# In[ ]:


data[cat_gt10].head()


# In[ ]:


def fun (x):
    if x > 5000:
        return(5000)
    else:
        return(x)


# In[ ]:


data.info()


# In[ ]:


data['Target'].value_counts()


# In[ ]:


data.drop('id',axis=1,inplace=True)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data.drop('Target',axis =1),data['Target'],test_size=.3, random_state = 2018)


# In[ ]:


print(x_train.shape,x_test.shape)


# In[ ]:


print(y_train.shape,y_test.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()


# In[ ]:


lm.fit(x_train,y_train)


# In[ ]:


pred_train = lm.predict(x_train)
pred_test = lm.predict(x_test)

proba_train = lm.predict_proba(x_train)[:,1]
proba_test = lm.predict_proba(x_test)[:,1]


# In[ ]:


pred_train.shape


# In[ ]:


pred_test.shape


# In[ ]:


pred_test[:5]


# In[ ]:


proba_test[:5]


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,auc,roc_curve


# In[ ]:


print ('*********************************')
print ('train')
print ('*********************************')
print ('Accuracy')
print (accuracy_score(pred_train,y_train))
print ('P R F1')
print (classification_report(y_train,pred_train))
print ('CM')
print (confusion_matrix(y_train,pred_train))
print ('roc')
tpr,fpr, th = (roc_curve(y_train,proba_train))
print (auc(tpr,fpr))


# In[ ]:


th


# In[ ]:


len(th)


# In[ ]:


print ('*********************************')
print ('test')
print ('*********************************')
print ('Accuracy')
print (accuracy_score(pred_test,y_test))
print ('P R F1')
print (classification_report(y_test,pred_test))
print ('CM')
print (confusion_matrix(y_test,pred_test))
print ('roc')
tpr,fpr, th = (roc_curve(y_test,proba_test))
print (auc(tpr,fpr))


# In[ ]:


x_train['proba'] = proba_train
for i in th[:100]:
    temp =x_train['proba'].apply(lambda x:1 if x > i else 0)
    print(accuracy_score(temp,y_train))


# In[ ]:


x_train['proba'] = proba_train
for i in th[:100]:
    temp =x_train['proba'].apply(lambda x:1 if x > i else 0)
    print(i,':', accuracy_score(temp,y_train))


# In[ ]:




