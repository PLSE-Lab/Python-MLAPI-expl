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


#read Csv File
data=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head()


# In[ ]:


data.describe()


# In[ ]:


#to Find any missing Value
data.isnull().sum()


# As we can see there are 67 missing values in salary attribute.So, we can say that there are total 67 students which doesn't get placement.We can overcome this problem by filling these missing value with 0
# 

# In[ ]:


##To fill the missing values with 0

data = data.fillna(0)

data.isnull().sum()


# Importing userful libraries
# 

# In[ ]:


import matplotlib.pyplot as mp
import seaborn as sb
import plotly.express as px
sb.set()


# In[ ]:


data.head()


# In[ ]:


mp.pie(data['gender'].value_counts(),labels=data['gender'].value_counts().index,autopct='%1.1f%%')


# In[ ]:


grp1=data.groupby(['gender','specialisation'])[['salary']].mean().reset_index()
fig=px.bar(grp1[['gender','specialisation','salary']].sort_values('salary',ascending=False),x='gender',y='salary',color='specialisation',template='ggplot2',log_y=True)
fig.show()


# In[ ]:


grp2 = data.groupby(["degree_t"])[["degree_p"]].mean().reset_index()
fig2=px.pie(grp2,values='degree_p',names='degree_t',template='seaborn')
fig2.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig2.show()


# In[ ]:


grp1


# In[ ]:


grp2


# In[ ]:


D=data.copy()
def plot_wrt_feature(feature):
     return sb.countplot(x=feature,data=D,hue='status')


# In[ ]:


plot_wrt_feature('ssc_b')


# In[ ]:


plot_wrt_feature('degree_t')


# In[ ]:


mp.barh(data['status'].value_counts().index,data['status'].value_counts(),tick_label =data['status'].value_counts())


# In[ ]:


plot_wrt_feature('workex')


# In[ ]:


plot_wrt_feature('specialisation')


# In[ ]:


mp.bar(data['specialisation'].value_counts().index,data['specialisation'].value_counts())


# In[ ]:


sb.heatmap(data.corr(),annot=True)


# In[ ]:


data=data.assign(outcome=(data['status']=='Placed').astype(int))
data.head()


# In[ ]:


Y=data['outcome']
variables=['ssc_p','hsc_b','degree_p','degree_t','etest_p','mba_p']
X=data[variables]


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=.30, random_state=1)

#numerical_cols = [col for col in X_train.columns if X_train[col].isnull.any]
s = (X_train.dtypes == 'object')
categorical_cols = list(s[s].index)

#PreProcessing fro numerical data. 

#numerical_transformer = SimpleImputer(strategy='constant')

#PreProcessing for categorcial data. 

categorical_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols)
    ])

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=1000, random_state=42)

from sklearn.metrics import mean_absolute_error

#bundle preprocessing and modeling code

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

#Preprocessing of training data, fit model.
my_pipeline.fit(X_train,y_train)

preds = my_pipeline.predict(X_test)
score = mean_absolute_error(y_test,preds)
print(score)


# In[ ]:


from sklearn.linear_model import LogisticRegression
model2=LogisticRegression()
my_pipeline2=Pipeline(steps=[('preprocessor',preprocessor),('model2',model2)])

my_pipeline2.fit(X_train,y_train)
preds2=my_pipeline2.predict(X_test)
score2=mean_absolute_error(y_test,preds2)
score2


# In[ ]:


variable2=['ssc_p','hsc_p','degree_p','workex','etest_p','mba_p']
x=data[variable2]
x


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.1,random_state=1)

label_x_train=x_train.copy()
label_x_test=x_test.copy()
#to generate Categorical Columns
s=(x_train.dtypes=='object')
cols=list(s[s].index)

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in cols:
    label_x_train[col]=labelencoder.fit_transform(x_train[col])
    label_x_test[col]=labelencoder.fit_transform(x_test[col])


# In[ ]:


from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

model.fit(label_x_train,y_train)
model.predict(label_x_test)


# In[ ]:


print(model.score(label_x_test,y_test))


# In[ ]:




