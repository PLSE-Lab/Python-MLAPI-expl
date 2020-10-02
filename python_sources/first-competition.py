#!/usr/bin/env python
# coding: utf-8

# In[33]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import f1_score , precision_score , recall_score , mean_squared_error 
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score
from xgboost import XGBClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Reading Data Files

# In[34]:


train = pd.read_csv("../input/train.csv")
test= pd.read_csv("../input/test.csv")


# In[35]:


train.head()


# In[36]:


train.info()


# In[37]:


train.drop(['Alley' , 'MiscFeature', 'Fence' , 'PoolQC'] , axis=1 , inplace=True)


# ## Defining TARGET variable & all the data

# In[38]:


train_labels = train['SalePrice']
#train.drop(['SalePrice'] , axis=1 , inplace=True)
All_data = pd.concat([train.loc[: , 'MSSubClass':'SaleCondition'] , test.loc[:,'MSSubClass':'SaleCondition']] , sort=False)

cat_data = list(train.select_dtypes(include='object').columns.values)
num_data = train._get_numeric_data().columns.drop(['Id','SalePrice'])


# In[39]:


All_data.hist(bins=50 , figsize=(20,20))
plt.show()


# # Data Preprocessing
# 
# 1. First , we will normalize the numerical data. The data is skewed. We will take log(feature+1).
# 2. Secondly , We will fill up the missing data by their mean.
# 3. And, Create dummies for categorical columns.

# In[40]:


fig = plt.figure(figsize=(12,6))
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()
plt.show()


# # Normalizing the data

# In[41]:


skewed = All_data[num_data].apply(lambda x : skew(x.dropna()))
skewed = skewed[skewed>0.75]
#skewed = skewed.index
    
train_labels = np.log1p(train_labels)


# # Handling Missing Values

# In[42]:


#Filling NAN values with median values
# imputer = SimpleImputer(strategy="median")
# imputer.fit_transform(All_data[num_data])
All_data.fillna(All_data.mean() , inplace=True)
All_data[num_data] = np.log1p(All_data[num_data])

#Filling up NAN values in Categorical Columns
All_data[cat_data] = All_data[cat_data].bfill()


# In[43]:


All_data.head()


# # Correlation between Target and rest of the features

# # Feature Selection ( Filter method )
# **Methods used here for feature selection is explained [here](https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b)**

# In[44]:


corr = train.corr()
corr_mat = corr['SalePrice'].sort_values(ascending=False)
corr_mat.index[1:11]


# In[45]:


fig = plt.figure(figsize=(18,36))
for i , cols in enumerate(num_data,1):
    plt.subplot(9,4 ,i)
    plt.scatter(x=train[cols] , y=train_labels)
    plt.xlabel(cols)
plt.show()


# In[46]:


fig = plt.figure(figsize=(12,10))
sns.heatmap(corr , cmap=plt.cm.Reds)


# # Feature Selection (Wrapper Method : Backward Elimination )

# In[47]:


import statsmodels.api as sm

train.fillna(train.mean() , inplace=True)
pmax = 1
cols = list(num_data)

while (len(cols)>0):
    
    p = []
    X_1 =  train[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(train_labels , X_1).fit()
    p = pd.Series(model.pvalues.values[1:] , index=cols)
    pmax = max(p)
    feature_with_max_p = p.idxmax()
    if (pmax > 0.5):
        cols.remove(feature_with_max_p)
    else:
        break

print(cols )


# # data preparation

# In[48]:


training_data = All_data[cols][:train.shape[0]]
testing_data = All_data[cols][train.shape[0]:]


# ## Training and Evaluating on the Training set

# In[49]:


lin = LinearRegression(normalize=False)
lin.fit(train[cols] , train_labels)


# In[50]:


np.expm1(lin.predict(testing_data[cols]))


# In[51]:


dec = DecisionTreeRegressor()
dec.fit(training_data[cols] , train_labels)


# In[52]:


np.expm1(dec.predict(testing_data[cols]))


# In[53]:


scores = cross_val_score(dec , np.expm1(training_data[cols]) , np.expm1(train_labels) , cv=10 , scoring='neg_mean_squared_error')


# In[54]:


rmse_score = np.sqrt(-scores)
rmse_score


# In[55]:


from sklearn.ensemble import RandomForestRegressor

ran = RandomForestRegressor(n_estimators=100)
ran.fit(training_data[cols] , train_labels)


# In[56]:


ran_scores = cross_val_score(ran , np.expm1(training_data[cols]) , np.expm1(train_labels) , cv=10 , scoring='neg_mean_squared_error')


# In[57]:


ran_scores_rmse = np.sqrt(-ran_scores)
ran_scores_rmse


# In[58]:


predict = np.expm1(ran.predict(testing_data[cols]))


# In[59]:


xgb =  XGBClassifier(n_estimators=100)
xgb.fit(training_data[cols] , train_labels)


# In[60]:


xgb_scores = cross_val_score(xgb , np.expm1(training_data[cols]) , np.expm1(train_labels) , cv=10 , scoring='neg_mean_squared_error')


# In[64]:


predict = np.expm1(xgb.predict((testing_data[cols])))


# In[65]:


data = { 'Id': test['Id'] , 'SalePrice':predict}
my_model = pd.DataFrame(data= data )


# In[66]:


my_model.to_csv('submission.csv' , index=False)


# In[ ]:




