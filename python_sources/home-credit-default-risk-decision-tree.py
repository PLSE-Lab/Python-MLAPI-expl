#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import gc
import warnings
import time
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
#app_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')


# ### Considering basic numeric features

# In[ ]:


app_num_basic_col = [
'SK_ID_CURR',
'TARGET',
'CNT_CHILDREN',
'AMT_INCOME_TOTAL',
'AMT_CREDIT',
'AMT_ANNUITY',
'AMT_GOODS_PRICE',
'REGION_POPULATION_RELATIVE',
'DAYS_BIRTH',
'DAYS_EMPLOYED',
'DAYS_REGISTRATION',
'DAYS_ID_PUBLISH',
'CNT_FAM_MEMBERS',
'REGION_RATING_CLIENT',
'REGION_RATING_CLIENT_W_CITY',
'REG_REGION_NOT_LIVE_REGION',
'REG_REGION_NOT_WORK_REGION',
'LIVE_REGION_NOT_WORK_REGION',
'REG_CITY_NOT_LIVE_CITY',
'REG_CITY_NOT_WORK_CITY',
'LIVE_CITY_NOT_WORK_CITY']


# In[ ]:


app_cat_basic_col = ['NAME_CONTRACT_TYPE',
'FLAG_OWN_CAR',
'FLAG_OWN_REALTY',
'CODE_GENDER',
'NAME_TYPE_SUITE',
'NAME_INCOME_TYPE',
'NAME_EDUCATION_TYPE',
'NAME_FAMILY_STATUS',
'NAME_HOUSING_TYPE',
'OCCUPATION_TYPE',
'ORGANIZATION_TYPE']


# In[ ]:


len(app_num_basic_col)


# In[ ]:


len(app_cat_basic_col)


# - Creating dataframe with required columns only

# In[ ]:


df = df[app_num_basic_col + app_cat_basic_col]


# In[ ]:


df.shape


# In[ ]:


df['TARGET'].value_counts()


# In[ ]:


def find_missing(data):
    ## Number of missing values
    missing_cnt = data.isnull().sum().values
    ## Total
    total = data.shape[0]
    ##Percentage of Missing values
    percentage = missing_cnt/total * 100
    missing_df = pd.DataFrame(data={'Total': total, 'Missing Count' : missing_cnt,'Percentage' : percentage}, 
                              index=data.columns.values)
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    return missing_df


# In[ ]:


find_missing(df[app_num_basic_col])


# ### Handling missing values

# In[ ]:


df['AMT_GOODS_PRICE']=df['AMT_GOODS_PRICE'].fillna(df['AMT_GOODS_PRICE'].median())
df['AMT_ANNUITY']=df['AMT_ANNUITY'].fillna(df['AMT_ANNUITY'].median())
df['CNT_FAM_MEMBERS']=df['CNT_FAM_MEMBERS'].fillna(df['CNT_FAM_MEMBERS'].median())


# In[ ]:


find_missing(df[app_num_basic_col])


# In[ ]:


find_missing(df[app_cat_basic_col])


# In[ ]:


df.OCCUPATION_TYPE.unique()


# In[ ]:


df.NAME_TYPE_SUITE.unique()


# In[ ]:


app_cat_basic_col.remove('OCCUPATION_TYPE')


# In[ ]:


df.drop('OCCUPATION_TYPE',inplace=True, axis=1)


# In[ ]:


df.shape


# In[ ]:


df['NAME_TYPE_SUITE']=df['NAME_TYPE_SUITE'].fillna('NTS_XNA')


# ### creating combined basic features from numerical and categorical

# In[ ]:


basic_features = app_num_basic_col + app_cat_basic_col 


# In[ ]:


len(basic_features)


# In[ ]:


find_missing(df[basic_features])


# ### Handling Outlier

# In[ ]:


sns.boxplot(data=df['DAYS_EMPLOYED'])


# In[ ]:


df['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');


# - found that DAYS_EMPLOYED has some anomalies
# - Around 18% of data amongs all data has some '365243' value in this fields
# - as its not make sence to current data so we need to handle it somehow
# - so i am replacing this value with np.nan
# - creating new column called DAYS_EMPLOYED_ANOM Anomalous flag which will have True or False value based on this field

# In[ ]:


# Create an anomalous flag column
#df['DAYS_EMPLOYED_ANOM'] = df["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

df['DAYS_EMPLOYED']=df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median())

df['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');


# After removing anomalies we can see above histogram that DAYS_EMPLOYED has maximum as 49 years and minimum is 0 year as discribe below

# In[ ]:


len(basic_features)


# In[ ]:


#basic_features.append('DAYS_EMPLOYED_ANOM')


# In[ ]:


len(basic_features)


# In[ ]:


df[df['DAYS_EMPLOYED'] / -365 > 8]['DAYS_EMPLOYED'].count()


# In[ ]:


(df['DAYS_BIRTH'] / -365).describe()


# In[ ]:


df[df['CODE_GENDER'] == 'XNA']


# In[ ]:


df = df[df['CODE_GENDER'] != 'XNA']


# In[ ]:


#df.drop('DAYS_EMPLOYED_ANOM',inplace=True,axis=1)
df.shape


# ### Lable encoding for categorical features whose values are binary like Y/N, Yes/No, True/False, M/F etc.

# In[ ]:


df[['SK_ID_CURR','CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'DAYS_EMPLOYED']].head(10)


# In[ ]:


# Categorical features with Binary encode (0 or 1; two categories)
for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'DAYS_EMPLOYED']:
    df[bin_feature], uniques = pd.factorize(df[bin_feature])


# In[ ]:


df[['SK_ID_CURR','CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'DAYS_EMPLOYED']].head(10)


# out of above basic categorical features we already encoded binary 
# - FLAG_OWN_CAR
# - FLAG_OWN_REALITY
# - CODE_GENDER
# - DAYS_EMPLYED_ANOM
# 
# Now doing one hot encoding for remaining features
# - NAME_CONTRACT_TYPE
# - NAME_TYPE_SUITE
# - NAME_INCOME_TYPE
# - NAME_EDUCATION_TYPE
# - NAME_FAMILY_STATUS
# - NAME_HOUSING_TYPE
# - ORGANIZATION_TYPE

# In[ ]:


one_hot_encode_col = ['NAME_CONTRACT_TYPE',
'NAME_TYPE_SUITE',
'NAME_INCOME_TYPE',
'NAME_EDUCATION_TYPE',
'NAME_FAMILY_STATUS',
'NAME_HOUSING_TYPE',
'ORGANIZATION_TYPE']


# In[ ]:


dummy_df = pd.get_dummies(df[one_hot_encode_col], dummy_na=False, drop_first=True)


# In[ ]:


len(dummy_df.columns)


# In[ ]:


df.shape


# In[ ]:


len(basic_features)


# In[ ]:


df.drop(one_hot_encode_col, axis=1,inplace=True)


# In[ ]:


for f in one_hot_encode_col:
    basic_features.remove(f)


# In[ ]:


len(basic_features)


# In[ ]:


df.shape


# ### creating final dataframe with required features

# In[ ]:


len(df[basic_features].columns)


# In[ ]:


len(dummy_df.columns)


# In[ ]:


df = pd.concat([df[basic_features], dummy_df], axis=1)


# In[ ]:


del dummy_df
gc.collect()


# In[ ]:


df.shape
df.head()


# In[ ]:


df.describe()


# ## Creating baseline model DecisionTree

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


X = df.drop('TARGET',axis=1)
y = df['TARGET']
print(X.shape)
print(y.shape)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=27)
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# **Decision Tree**

# In[ ]:



dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=27)
print(X_train.shape[1])
print(X_train.shape[0])


# **Random forests**

# In[ ]:


dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions))


# In[ ]:



rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))


# Balacing data

# In[ ]:


pd.value_counts(y_train).plot.bar()
plt.title('histogram')
plt.xlabel('TARGET')
plt.ylabel('Frequency')
df['TARGET'].value_counts()


# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


sm = SMOTE(random_state=27)
X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)
pd.value_counts(y_train_sm).plot.bar()
plt.title('histogram')
plt.xlabel('TARGET')
plt.ylabel('Frequency')

print("Number transactions X_train dataset: ", X_train_sm.shape)
print("Number transactions y_train dataset: ", y_train_sm.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# In[ ]:


dtree.fit(X_train_sm,y_train_sm)
predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions))


# In[ ]:


rfc.fit(X_train_sm, y_train_sm)

rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))


# Standardization
# 

# In[ ]:


#Standardization
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.transform(X_test)
df_x = pd.DataFrame(X_test_std)
df_x.head()


# In[ ]:


sm = SMOTE(random_state=27)
X_train_sm, y_train_sm = sm.fit_sample(X_train_std, y_train)
pd.value_counts(y_train_sm).plot.bar()
plt.title('histogram')
plt.xlabel('TARGET')
plt.ylabel('Frequency')


# In[ ]:


dtree.fit(X_train_sm,y_train_sm)
predictions = dtree.predict(X_test_std)
print(classification_report(y_test,predictions))


# In[ ]:


rfc.fit(X_train_sm, y_train_sm)

rfc_pred = rfc.predict(X_test_std)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))


# In[ ]:




