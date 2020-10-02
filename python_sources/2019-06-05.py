#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


app_train = pd.read_csv('../input/application_train.csv')
print('Training data shape: ', app_train.shape)
app_train.head()


# In[3]:


app_test = pd.read_csv('../input/application_test.csv')
print('Testing data shape: ', app_test.shape)
app_test.head()


# In[4]:


app_train['TARGET'].value_counts()


# In[5]:


app_train.dtypes.value_counts()


# In[6]:


app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[7]:


le = LabelEncoder()
le_count = 0


# In[8]:


for col in app_train:
    if app_train[col].dtype == 'object':
        if len(list(app_train[col].unique())) <= 2:
            le.fit(app_train[col])
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            le_count += 1


# In[9]:


app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)


# In[10]:


app_train.shape


# In[11]:


app_test.shape


# In[12]:


train_labels = app_train['TARGET']

app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)


# In[13]:


app_train['TARGET'] = train_labels


# In[14]:


print(app_train.shape)
print(app_test.shape)


# In[15]:


(app_train['DAYS_BIRTH'] / -365).describe()


# In[16]:


app_train['DAYS_EMPLOYED'].describe()


# In[17]:


app_train['DAYS_EMPLOYES_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');


# In[18]:


app_test['DAYS_EMPLOYED_ANOM'] = app_test['DAYS_EMPLOYED'] == 365243
app_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
print(app_test['DAYS_EMPLOYED_ANOM'].sum(), len(app_test))


# In[19]:


correlations = app_train.corr()['TARGET'].sort_values()
print(correlations.tail(15), correlations.head(15))


# In[20]:


app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_train['DAYS_BIRTH'].corr(app_train['TARGET'])


# In[22]:


plt.style.use('fivethirtyeight')
plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');


# In[25]:


age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_data.head()


# In[26]:


age_groups = age_data.groupby('YEARS_BINNED').mean()


# In[27]:


age_groups


# In[28]:


plt.figure(figsize = (8, 8))
plt.bar(age_groups.index.astype(str), 100*age_groups['TARGET'])
plt.xticks(rotation = 75);
plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay(%)')
plt.title('Failure to Repay by age Group')


# In[30]:


ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
ext_data_corrs


# In[31]:


poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]


# In[32]:


from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')
poly_target = poly_features['TARGET']
poly_features = poly_features.drop(columns = ['TARGET'])


# In[33]:


poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)


# In[34]:


from sklearn.preprocessing import PolynomialFeatures


# In[35]:


poly_transformer = PolynomialFeatures(degree = 3)


# In[36]:


poly_transformer.fit(poly_features)

poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)


# In[37]:


poly_features.shape


# In[38]:


poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]


# In[39]:


poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))
poly_features['TARGET'] = poly_target

poly_corrs = poly_features.corr()['TARGET'].sort_values()


# In[40]:


print(poly_corrs.head(10))
print(poly_corrs.tail(10))


# In[41]:


poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))

poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
app_train_poly = app_train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)


# In[42]:


print(app_train_poly.shape)
print(app_test_poly.shape)


# In[43]:


app_train_domain = app_train.copy()
app_test_domain = app_test.copy()

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']


# In[44]:


app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']


# In[45]:


from sklearn.preprocessing import MinMaxScaler, Imputer


# In[46]:


if 'TARGET' in app_train:
    train = app_train.drop(columns = ['TARGET'])
else:
    train = app_train.copy()


# In[47]:


features = list(train.columns)
test = app_test.copy()
imputer = Imputer(strategy = 'median')
scaler = MinMaxScaler(feature_range = (0, 1))
imputer.fit(train)


# In[48]:


train = imputer.transform(train)
test = imputer.transform(app_test)


# In[49]:


scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)


# In[50]:


print(train.shape)
print(test.shape)


# In[51]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C = 0.0001)
log_reg.fit(train, train_labels)


# In[52]:


log_reg_pred = log_reg.predict_proba(test)[:, 1]


# In[53]:


submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()


# In[54]:


submit.to_csv('log_reg_baseline.csv', index = False)


# In[ ]:




