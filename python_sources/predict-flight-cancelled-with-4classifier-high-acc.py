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


df_2019 = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2019_ontime.csv')
df_2020 = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2020_ontime.csv')


# In[ ]:


df_2020.head()


# In[ ]:


df_2020.tail()


# In[ ]:


print(df_2020.shape)
print(df_2020['Unnamed: 21'].isnull().sum())


# # **Unnamed: 21 already empty. We will clean this column at preprocessing step.**

# In[ ]:


def bar_plot(variable):
    var = df_2020[variable] # get feature
    varValue = var.value_counts() # count number of categorical variable(value/sample)
    
    plt.figure(figsize = (9,6))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{} \n {}".format(variable,varValue))


# In[ ]:


bar_plot('CANCELLED')


# In[ ]:


print(df_2020.columns)
print(df_2020.shape[1])


# # **22 column and their names**

# In[ ]:


df_2020.info()


# # **How much NaN Value**

# In[ ]:


column_names = df_2020.columns
j=0
for i in df_2020.columns:
    print("  {} has got {} Null Sample " .format(df_2020.columns[j],df_2020[i].isnull().sum()))
    j=j+1


# In[ ]:


import missingno as msno
plt.figure(figsize=(4,4))
msno.bar(df_2020)


# In[ ]:


msno.heatmap(df_2020) 


# # **-O-O- PREPROCESSING -0-0-**

# # **Unnamed: 21**

# In[ ]:


#Data Preprocessing
df_2020 = df_2020.drop(['Unnamed: 21'],axis=1)
df_2020.shape


# # TAIL NUM

# In[ ]:


#Drop NaN TAIL_NUM rows
df_2020 = df_2020.dropna(subset=['TAIL_NUM'])
print(df_2020['TAIL_NUM'].isna().sum())
print(df_2020.shape)


# # **DEP_DEL15**

# # **if not type 15min delay :  we filled with 0**

# In[ ]:


df_2020['DEP_DEL15'] = df_2020['DEP_DEL15'].replace(np.NaN,0)
df_2020['DEP_DEL15'].isnull().sum()


# # **ARR_DEL15**

# # if not type 15 min delay : we filled 0

# In[ ]:


df_2020['ARR_DEL15'] = df_2020['ARR_DEL15'].replace(np.NaN,0)
df_2020['ARR_DEL15'].isnull().sum()


# # **DEP_TIME and ARR_TIME**

# In[ ]:


from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
#DEP_TIME

df_2020['DEP_TIME'] = imp_mean.fit_transform(df_2020[['DEP_TIME']])
#ARR_TIME

df_2020['ARR_TIME'] = imp_mean.fit_transform(df_2020[['ARR_TIME']])


# # **We filled NaN values with column's mean**

# # **CHECK NaN VALUES**

# In[ ]:


column_names = df_2020.columns
j=0
for i in df_2020.columns:
    print("  {} has got {} NaN Sample " .format(df_2020.columns[j],df_2020[i].isnull().sum()))
    j=j+1


# In[ ]:


df_2020.shape


# # We cleaned data but we lost (607346 - 600271 = 7075) sample

# # **--------------------------------------------------------------------------------------------------------**

# # CORRELATION MATRIX

# In[ ]:


import seaborn as sns
f,ax= plt.subplots(figsize=(15,15))
sns.heatmap(df_2020.corr(),linewidths=.5,annot=True,fmt='.4f',ax=ax)
plt.show()


# # DEST_AIRPORT_ID - DEST_AIRPORT_SEQ_ID  and  ORIGIN_AIRPORT_ID - ORIGIN_AIRPORT_SEQ_ID  They are looking same so I gonna drop each one of them

# In[ ]:


df_2020 = df_2020.drop(['DEST_AIRPORT_SEQ_ID'],axis=1)
df_2020 = df_2020.drop(['ORIGIN_AIRPORT_SEQ_ID'],axis=1)
print(df_2020.shape)


# # **I'm looking my target feature**

# In[ ]:


bar_plot('CANCELLED')


# # Data set is imbalance. So we can't trust accuracy metric. We will check other metrics.

# # TRAIN - TEST SPLIT

# In[ ]:


y = df_2020.CANCELLED
df_2020 = df_2020.drop('CANCELLED',axis=1)
X = df_2020


# In[ ]:


categorical_columns = ['OP_CARRIER','OP_UNIQUE_CARRIER','TAIL_NUM','ORIGIN','DEST','DEP_TIME_BLK']
for col in categorical_columns:
    X_encoded = pd.get_dummies(X[col],prefix_sep = '_')
    df_2020 = df_2020.drop([col],axis=1)

df_2020 = pd.concat([df_2020, X_encoded], axis=1)


# In[ ]:


X = df_2020


# # **We applied One-Hot Encoder for categorical columns**

# In[ ]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,shuffle=True,random_state=42)


# # **Seperated train and test data**

# # **Decision Tree Classifier**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(random_state = 0)
model_dt = clf_dt.fit(X_train, y_train) 


# In[ ]:


from sklearn import tree
tree.plot_tree(model_dt) 


# In[ ]:


from sklearn import metrics
y_pred = model_dt.predict(X_test)
print(metrics.classification_report(y_test,y_pred))


# In[ ]:


y_test.value_counts()


# # **Random Forest Classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(max_depth=50)
model_rf = clf_rf.fit(X_train, y_train)


# In[ ]:


from sklearn import metrics
y_pred = model_rf.predict(X_test)
print(metrics.classification_report(y_test,y_pred))


# # **Ada Boost Classifier**

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
clf_ab = RandomForestClassifier()
model_ab = clf_ab.fit(X_train, y_train)


# In[ ]:


from sklearn import metrics
y_pred = model_ab.predict(X_test)
print(metrics.classification_report(y_test,y_pred))


# # **XGBoost Classifier**

# In[ ]:


import xgboost as xgb
clf_xgb = xgb.XGBClassifier()
model_xgb = clf_xgb.fit(X_train, y_train)


# In[ ]:


from sklearn import metrics
y_pred = model_xgb.predict(X_test)
print(metrics.classification_report(y_test,y_pred))


# # Please report your suggestions. These are important for me to see my mistakes and improve my self.
