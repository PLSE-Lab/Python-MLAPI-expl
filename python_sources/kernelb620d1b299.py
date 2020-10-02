#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train_data = pd.read_csv('../input/train.csv')


# In[4]:


train_data.shape


# In[5]:


train_data.select_dtypes(include=[np.int64]).nunique().value_counts().sort_index().plot.bar(color='blue');
plt.xlabel('Number of Unique Values')
plt.ylabel('Count')
plt.title('Count of Unique values in Integer Columns')


# In[6]:


train_data.select_dtypes(include = ['object']).head()
    


# dependency: Dependency rate, calculated = (No. of members of the household younger than 19 or older tahn 64)/(No of 
#             members of household between 19 and 64)
# edjefe: years of education of male head of household, "yes" = 1 and "no" = 0
# edjefa: years of education of female head of household, "yes" = 1 and "no" = 0
# 
# Replace 'yes' with 1 and 'no' with 0 in all the above 3 columns for 'train' and 'test' data

# In[7]:


def ProcessObjectTypeColumns(df):
    replacements = {'yes':1,'no':0}
    df['dependency'].replace(replacements,inplace=True)
    df['edjefe'].replace(replacements,inplace=True)
    df['edjefa'].replace(replacements,inplace=True)
    #Now all the 'object' columns have 'float' data. So, convert these columns to 'float' datatype
    df['dependency'] = pd.to_numeric(df['dependency'])
    df['edjefe'] = pd.to_numeric(df['edjefe'])
    df['edjefa'] = pd.to_numeric(df['edjefa'])
    return df


# In[8]:


train_data = ProcessObjectTypeColumns(train_data)


# Distribution of poverty levels in the training data

# In[9]:


train_data['Target'].value_counts().sort_index().plot.bar(color='blue')


# The poverty level should be same for each individual in a household. Let's check whether each member of the household has same poverty level. As per the given data, the 'idhogar' field is a unique identifier for each household. We will make use of this field to identify if there are any anomolies

# In[10]:


all_equal = train_data.groupby('idhogar')['Target'].apply(lambda x:x.nunique() == 1)
not_equal = all_equal[all_equal != True]
print('There are {} households where the poverty level is not same for all members'.format(len(not_equal)))


# For all these households we can take the poverty level of head of the household as poverty level. But before that let's check if there are any households without a head of the household

# In[11]:


head_of_household  = train_data.groupby('idhogar')['parentesco1'].sum()
households_no_head = train_data.loc[train_data['idhogar'].isin(head_of_household[head_of_household == 0].index),:]
print('There are {} households without a head of household'.format(households_no_head['idhogar'].nunique()))


# For the records where there is a difference in poverty level within the household members, lets update them with the poverty level of head of the household

# In[13]:


for household in not_equal.index:
    actual_target = int(train_data[(train_data['idhogar'] == household)&(train_data['parentesco1']) == 1.0]['Target'])
    train_data.loc[train_data['idhogar']==household,'Target'] = actual_target


# Find the missing values in each column

# In[14]:


missing_values = train_data.isnull().sum().sort_values(ascending = False)
missing_values[missing_values != 0]


# The above data shows missing values are in the columns 'rez_esc', 'v18q1', 'v2a1', 'meaneduc', 'SQBmeaned'
# 'rez_esc': Years behind in school
# 'v18q1': number of tablets household owns
# 'v2a1': Monthly rent payment
# 'meaneduc': average years of education for adults (18+)
# 'SQBmeaned': square of the mean years of education of adults (>=18) in the household

# In[15]:


def ReplaceMissingDatawithMean(df):
    df['rez_esc'].fillna((df['rez_esc'].mean()),inplace=True)
    df['v18q1'].fillna((df['v18q1'].mean()),inplace=True)
    df['v2a1'].fillna((df['v2a1'].mean()),inplace=True)
    df['meaneduc'].fillna((df['meaneduc'].mean()),inplace=True)
    df['SQBmeaned'].fillna((df['SQBmeaned'].mean()),inplace=True)
    return df


# In[16]:


train_data = ReplaceMissingDatawithMean(train_data)


# We can neglect the features 'Id' and 'idhogar' as they are just identifiers

# In[17]:


X = train_data[train_data.columns.difference(['Id','idhogar','Target'])].copy()
y = train_data['Target']


# In[18]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[19]:


from sklearn.decomposition import PCA
pca = PCA().fit(X)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('PCA')
plt.show()


# From the above plot it is clear that if we take 80 components we can preserve atleast 90 % variance. So we will choose components

# In[20]:


pca = PCA(n_components = 80)
X = pca.fit_transform(X)


# In[21]:


df = pd.DataFrame(data = X)


# In[22]:


from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(df,y,test_size=0.3)


# In[23]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# **Gradient Boosting Classifier:**

# In[24]:


from sklearn.ensemble import GradientBoostingClassifier
gbclf = GradientBoostingClassifier()
gbclf.fit(X_train,y_train)
y_gbclf = gbclf.predict(X_val)
cm_gbclf = confusion_matrix(y_val,y_gbclf)
cm_gbclf


# In[25]:


score_gbclf = accuracy_score(y_val,y_gbclf)
score_gbclf


# **Random Forest Classifier:**

# In[26]:


from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier()
rfclf.fit(X_train,y_train)
y_rfclf = rfclf.predict(X_val)
cm_rfclf = confusion_matrix(y_val,y_rfclf)
cm_rfclf


# In[27]:


score_rfclf = accuracy_score(y_val,y_rfclf)
score_rfclf


# **KNeighbors Classifier:**

# In[28]:


from sklearn.neighbors import KNeighborsClassifier
knclf = KNeighborsClassifier()
knclf.fit(X_train,y_train)
y_knclf = knclf.predict(X_val)
cm_knclf = confusion_matrix(y_val,y_knclf)
cm_knclf


# In[29]:


score_knclf = accuracy_score(y_val,y_knclf)
score_knclf


# **Extra Tress Classifier:**

# In[30]:


from sklearn.ensemble import ExtraTreesClassifier
etclf = ExtraTreesClassifier()
etclf.fit(X_train,y_train)
y_etclf = etclf.predict(X_val)
cm_etclf = confusion_matrix(y_val,y_etclf)
cm_etclf


# In[31]:


score_etclf = accuracy_score(y_val,y_etclf)
score_etclf


# **XGBoost Classifier:**

# In[32]:


from xgboost import XGBClassifier
xgclf = XGBClassifier()
xgclf.fit(X_train,y_train)
y_xgclf = xgclf.predict(X_val)
cm_xgclf = confusion_matrix(y_val,y_xgclf)
cm_xgclf


# In[33]:


score_xgclf = accuracy_score(y_val,y_xgclf)
score_xgclf


# **Light GBM Classifier:**

# In[44]:


import lightgbm as lgb
d_train = lgb.Dataset(X_train,label=y_train)
params={}
lgbclf = lgb.train(params,d_train,100)
y_lgbclf = lgbclf.predict(X_val)


# **Predicting Poverty levels for Test Data**

# In[45]:


test_data = pd.read_csv('../input/test.csv')


# In[47]:


test_data = ProcessObjectTypeColumns(test_data)


# In[48]:


test_data = ReplaceMissingDatawithMean(test_data)


# In[49]:


X_test = test_data[test_data.columns.difference(['Id','idhogar'])].copy()
X_test = scaler.fit_transform(X_test)
X_test = pca.fit_transform(X_test)
X_test = pd.DataFrame(data = X_test)


# We will choose KNeighbors Classifier for preducting the test data poverty levels as it has given 75 % accuracy

# In[50]:


y_test_knclf = knclf.predict(X_test)


# In[52]:


test_data['Target'] = pd.Series(y_test_knclf)


# In[53]:


test_data.head()


# In[ ]:




