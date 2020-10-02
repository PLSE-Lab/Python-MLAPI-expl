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





# # Import Data

# In[ ]:


ad=pd.read_csv('../input/adult.csv')
ad


# # Data Visualization

# In[ ]:


ad.hist()


# In[ ]:


ad.dtypes


# In[ ]:


import seaborn as sns
sns.catplot(x="capital.loss", y="sex",  hue="sex",data=ad)


# In[ ]:


sns.catplot(x="education.num", y="sex",  hue="sex",data=ad)


# In[ ]:


import matplotlib.pyplot as plt
fig_dims = (15, 5)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x="hours.per.week", y="education",  hue="sex",data=ad,ax=ax)


# In[ ]:


fig_dims = (15, 5)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x="age", y="race",  hue="sex",data=ad,ax=ax)


# In[ ]:


fig_dims = (15, 5)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x="age", y="income",  hue="sex",data=ad,ax=ax)


# In[ ]:


fig_dims = (15, 5)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x="capital.gain", y="income",  hue="sex",data=ad,ax=ax)


# In[ ]:


fig_dims = (15, 5)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x="fnlwgt", y="education",  hue="sex",data=ad,ax=ax)


# In[ ]:


sns.violinplot(x=ad['fnlwgt'],y=ad['income'],hue='sex',data=ad)


# In[ ]:


sns.violinplot(x=ad['workclass'],y=ad['age'],data=ad)


# In[ ]:


plt.scatter(x=ad['age'],y=ad['race'])


# In[ ]:


sns.violinplot(x=ad['education.num'],y=ad['sex'],data=ad)


# In[ ]:


sns.violinplot(x=ad['age'],y=ad['income'],data=ad)


# In[ ]:


plt.rcParams["figure.figsize"] = 12,8
plt.scatter(ad['age'],ad['native.country'])


# In[ ]:


sns.pairplot(ad)


# In[ ]:


sns.heatmap(ad.corr(), annot=True)


# # Data Preprocessing

# In[ ]:


ad = ad[(ad.astype(str) != '?').all(axis=1)]
ad=ad.reset_index(drop=True)
ad


# In[ ]:


ad.isna().sum().sum()


# In[ ]:


ad['education']=ad.education.str.replace('-','')


# In[ ]:


ad['marital.status'] =  ad['marital.status'].str.replace('-','')


# In[ ]:


ad['occupation'] =  ad['occupation'].str.replace('-','')


# In[ ]:


ad['relationship'] =  ad['relationship'].str.replace('-','')


# In[ ]:


ad['native.country'] =  ad['native.country'].str.replace('-','')


# In[ ]:


ad


# In[ ]:


ad.workclass.value_counts()


# In[ ]:


ad.education.value_counts()


# ### Label Encoding

# In[ ]:


from sklearn import preprocessing
#label Encoder
category_col =['workclass', 'race', 'education','Marital_Status', 'occupation',
               'relationship', 'sex', 'native.country'] 
labelEncoder = preprocessing.LabelEncoder()


# In[ ]:


ad= ad.rename(columns={'marital.status':'Marital_Status','education.num':'education_num'})
ad


# In[ ]:


mapping_dict={}
for col in category_col:
    ad[col] = labelEncoder.fit_transform(ad[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col]=le_name_mapping
print(mapping_dict)


# In[ ]:


ad


# In[ ]:


plt.rcParams["figure.figsize"] = 12,8
import matplotlib.pyplot as plt
fig,axes = plt.subplots(2,3)


axes[0,0].set_title("fnlwgt")
axes[0,0].boxplot(ad['fnlwgt'])
axes[0,1].set_title("capital.loss")
axes[0,1].boxplot(ad['capital.loss'])
axes[0,2].set_title("hours.per.week")
axes[0,2].boxplot(ad['hours.per.week'])



axes[1,0].set_title("education")
axes[1,0].boxplot(ad['education'])
axes[1,1].set_title("occupation")
axes[1,1].boxplot(ad['occupation'])
axes[1,2].set_title("native.country")
axes[1,2].boxplot(ad['native.country'])


# # Power Transformation

# In[ ]:



a=ad
a=a.select_dtypes(exclude="object")
a = a.drop(['sex'],axis=1)


# In[ ]:


from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
a=pt.fit_transform(a)
a


# In[ ]:


a = pd.DataFrame(a)
a


# In[ ]:


a = a.rename(columns={0:'age',1:'workclass',2:'fnlwgt',3:'education',4:'education_num',5:'Marital_Status',6:'occupation',7:'relationship',8:'race',9:'Capital_gain',10:'Capital_loss',11:'hours_per_week',12:'native_country'})
a


# In[ ]:


dd=ad['income']
dd = pd.DataFrame(dd)
dd


# In[ ]:


a = pd.concat([a,dd],axis=1)
a


# In[ ]:


a.dtypes


# In[ ]:


a.income.value_counts()


# In[ ]:


def encode_target(x):
    if x == "<=50K":
        return 1
    elif x == ">50K":
        return 0
a['income']=a['income'].apply(encode_target)
a


# In[ ]:


plt.rcParams["figure.figsize"] = 12,8
import matplotlib.pyplot as plt
fig,axes = plt.subplots(2,3)


axes[0,0].set_title("fnlwgt")
axes[0,0].boxplot(a['fnlwgt'])
axes[0,1].set_title("capital.loss")
axes[0,1].boxplot(a['Capital_loss'])
axes[0,2].set_title("hours_per_week")
axes[0,2].boxplot(a['hours_per_week'])



axes[1,0].set_title("education")
axes[1,0].boxplot(a['education'])
axes[1,1].set_title("occupation")
axes[1,1].boxplot(a['occupation'])
axes[1,2].set_title("native_country")
axes[1,2].boxplot(a['native_country'])


# ## Finding Outliers

# In[ ]:


def outlier(x):
    high=0
    q1 = x.quantile(.25)
    q3 = x.quantile(.75)
    iqr = q3-q1
    low = q1-1.5*iqr
    high += q3+1.5*iqr
    outlier = (x.loc[(x < low) | (x > high)])
    return(outlier)


# In[ ]:


outlier(a['hours_per_week']).count()


# In[ ]:


outlier(a['fnlwgt']).count()


# In[ ]:


outlier(a['native_country']).count()


# In[ ]:


outlier(a['income']).count()


# # Removing Outliers

# In[ ]:


q1 =a.quantile(.25)
q3 =a.quantile(.75)
iqr = q3-q1


# In[ ]:


df_new = a[~((a < (q1 - 1.5 *iqr))  |  (a > (q3+ 1.5 * iqr)))]


# In[ ]:


df_new.isna().sum()


# In[ ]:


modeval=int(df_new['workclass'].mode())
df_new.workclass = df_new.workclass.fillna(modeval)


# In[ ]:


modeval=int(df_new['fnlwgt'].mode())
df_new.fnlwgt = df_new.fnlwgt.fillna(modeval)


# In[ ]:


modeval=int(df_new['education'].mode())
df_new.education = df_new.education.fillna(modeval)


# In[ ]:


modeval=int(df_new['education_num'].mode())
df_new.education_num = df_new.education_num.fillna(modeval)


# In[ ]:


modeval=int(df_new['race'].mode())
df_new.race = df_new.race.fillna(modeval)


# In[ ]:


modeval=int(df_new['Capital_gain'].mode())
df_new.Capital_gain = df_new.Capital_gain.fillna(modeval)


# In[ ]:


modeval=int(df_new['Capital_loss'].mode())
df_new.Capital_loss = df_new.Capital_loss.fillna(modeval)


# In[ ]:


modeval=int(df_new['hours_per_week'].mode())
df_new.hours_per_week = df_new.hours_per_week.fillna(modeval)


# In[ ]:


modeval=int(df_new['native_country'].mode())
df_new.native_country = df_new.native_country.fillna(modeval)


# In[ ]:


df_new=df_new.fillna(0)


# In[ ]:



df_new.isna().sum()


# In[ ]:


df_new.shape


# In[ ]:


plt.rcParams["figure.figsize"] = 12,8
import matplotlib.pyplot as plt
fig,axes = plt.subplots(2,3)


axes[0,0].set_title("fnlwgt")
axes[0,0].boxplot(df_new['fnlwgt'])
axes[0,1].set_title("capital.loss")
axes[0,1].boxplot(df_new['Capital_loss'])
axes[0,2].set_title("hours_per_week")
axes[0,2].boxplot(df_new['hours_per_week'])



axes[1,0].set_title("education")
axes[1,0].boxplot(df_new['education'])
axes[1,1].set_title("occupation")
axes[1,1].boxplot(df_new['occupation'])
axes[1,2].set_title("native_country")
axes[1,2].boxplot(df_new['native_country'])


# In[ ]:


df_new.income = df_new.income.astype(int)

df_new


# # Separation Of x and y

# In[ ]:


x = df_new.drop(['income'],axis=1)
y = df_new['income']


# # Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)


# # Model PipeLine

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


pipeline_lr=Pipeline([('lr_classifier',LogisticRegression(random_state=0))])
pipeline_dt=Pipeline([('dt_classifier',DecisionTreeClassifier())])
pipeline_rf=Pipeline([('rf_classifier',RandomForestClassifier())])
pipeline_Nb=Pipeline([('Nb_Gaussion',GaussianNB())])
pipeline_XGb=Pipeline([('XGb_classifier',XGBClassifier())])


# In[ ]:


pipelines = [pipeline_lr, pipeline_dt, pipeline_rf,pipeline_Nb,pipeline_XGb]


# In[ ]:


best_accuracy=0.0
best_classifier=0
best_pipeline=""
pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest',3:'GaussianNB',4:'XGBClassifier'}


# In[ ]:


for pipe in pipelines:
    pipe.fit(x_train, y_train)


# In[ ]:


for i,model in enumerate(pipelines):
    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(x_test,y_test)))


# In[ ]:


for i,model in enumerate(pipelines):
    if model.score(x_test,y_test)>best_accuracy:
        best_accuracy=model.score(x_test,y_test)
        best_pipeline=model
        best_classifier=i
print('Classifier with best accuracy:{}'.format(pipe_dict[best_classifier]))


# # Over Sampling With SMOTE

# In[ ]:


from imblearn.over_sampling import SMOTE

smote = SMOTE()
x_train_smote, y_train_smote = smote.fit_sample(x_train.astype('float'),y_train)

from collections import Counter
print("Before smote:",Counter(y_train))
print("After smote:",Counter(y_train_smote))


# # Model PipeLine After SMOTE

# In[ ]:


pipeline_lr1=Pipeline([('lr_classifier',LogisticRegression(random_state=0))])
pipeline_dt1=Pipeline([('dt_classifier',DecisionTreeClassifier())])
pipeline_rf1=Pipeline([('rf_classifier',RandomForestClassifier())])
pipeline_Nb1=Pipeline([('Nb_Gaussion',GaussianNB())])
pipeline_XGb1=Pipeline([('XGb_classifier',XGBClassifier())])


# In[ ]:


pipelines1 = [pipeline_lr1, pipeline_dt1, pipeline_rf1,pipeline_Nb1,pipeline_XGb1]


# In[ ]:


best_accuracy1=0.0
best_classifier1=0
best_pipeline1=""
pipe_dict1 = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest',3:'GaussianNB',4:'XGBClassifier'}


# In[ ]:


for pipe in pipelines1:
    pipe.fit(x_train_smote, y_train_smote)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix
for i,model in enumerate(pipelines1):
    print("{} Test Accuracy: {}".format(pipe_dict1[i],model.score(x_test,y_test)))


# In[ ]:


for i,model in enumerate(pipelines1):
    if model.score(x_test,y_test)>best_accuracy1:
        best_accuracy1=model.score(x_test,y_test)
        best_pipeline1=model
        best_classifier1=i
print('Classifier with best accuracy:{}'.format(pipe_dict1[best_classifier1]))


# # Classification Report

# In[ ]:


Logisclassifier = LogisticRegression(random_state = 95)
Logisclassifier.fit(x_train_smote, y_train_smote)
LOGy_pred = Logisclassifier.predict(x_test)
print("LG:",classification_report(LOGy_pred,y_test),confusion_matrix(y_test, LOGy_pred))


# In[ ]:


Desclassifier = DecisionTreeClassifier()
Desclassifier.fit(x_train,y_train)
y_preddtree = Desclassifier.predict(x_test)
print("DTREE:",classification_report(y_preddtree,y_test),confusion_matrix(y_test, y_preddtree))


# In[ ]:


Classmodel = RandomForestClassifier(n_estimators = 100, random_state = 42)
Classmodel.fit(x_train_smote, y_train_smote)
y_pred = Classmodel.predict(x_test)
print("RF:",classification_report(y_pred,y_test),confusion_matrix(y_test, y_pred))


# In[ ]:


classifier = GaussianNB()
classifier.fit(x_train_smote, y_train_smote)
y_predNB = classifier.predict(x_test)
print("NB:",classification_report(y_predNB,y_test),confusion_matrix(y_test, y_predNB))


# In[ ]:


XGBModel = XGBClassifier(n_estimators=110,learning_rate=0.05,max_depth=3)
XGBModel.fit(x_train_smote, y_train_smote)
y_predXg = XGBModel.predict(x_test)
print("XGB:",classification_report(y_predXg,y_test),confusion_matrix(y_test, y_predXg))


# In[ ]:




