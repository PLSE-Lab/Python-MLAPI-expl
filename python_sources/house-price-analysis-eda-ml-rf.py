#!/usr/bin/env python
# coding: utf-8

# ### About Data 
# The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset. 
# 
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[3]:


train_data.head()


# In[4]:


test_data.head()


# In[5]:


#shape of Train Data
train_data.shape


# There are **1460** rows and **81** columns in the **train_data**

# In[8]:


test_data.shape


# There are **1460** rows and **80* columns in the **train_data**

# In[9]:


# train dataframe info
train_data.info() 


# In[10]:


test_data.info()


# In[11]:


#Defining Fuction for dropping columns
def  drop_null (data):
    for column in data:
        if data[column].count() / len(data) <= 0.3:
            data.drop(column, axis=1, inplace=True)
            print('Dropped Column', column)


# This fuction will eliminate all columns having more then **30%** null values.

# In[13]:


#Calling function created for column having null values
#for Train dataset
drop_null(train_data)

#dropping id
train_data.drop('Id', axis=1,  inplace=True)
print(train_data.shape)


# Above column had more then **30** missing values hence are removed from the both the dataframe

# In[14]:


#Calling function created for column having null values
#for Train dataset
drop_null(test_data)

#dropping id
test_data.drop('Id', axis=1,  inplace=True)
print(test_data.shape)


# Above column had more then **30** missing values hence are removed from the both the dataframe

# #### Dealing with null values
# 
# Defining Categorical and Numerical columns and also defining conditions to fill null values.

# In[15]:


# Filling "NA" others missing data from train data
missing_data_stats = train_data.isnull().sum()
cols = missing_data_stats[missing_data_stats>0].index.tolist()
cat_cols = train_data.select_dtypes(exclude=['int64', 'float64']).columns


for c in cols:
    if c in cat_cols:
        mode = train_data[c].mode()[0]
        train_data[c] = train_data[c].fillna(mode)
    else:
        median = train_data[c].median()
        train_data[c] = train_data[c].fillna(median)


# In[16]:


#Filling "NA" others missing data from test data
missing_data_stats = test_data.isnull().sum()
cols = missing_data_stats[missing_data_stats>0].index.tolist()
cat_cols = test_data.select_dtypes(exclude=['int64', 'float64']).columns


for c in cols:
    if c in cat_cols:
        mode = test_data[c].mode()[0]
        test_data[c] = test_data[c].fillna(mode)
    else:
        median = test_data[c].median()
        test_data[c] = test_data[c].fillna(median)


# In[17]:


train_data.isnull().sum()


# In[18]:


test_data.isnull().sum()


# #### Descriptive statistics of numerical values

# In[19]:


train_data.describe()


# ### Visualization
# 
# I only used train dataSet for visualization and EDA exploration.............

# In[21]:


print(train_data['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(train_data['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.6});


# In[22]:


plt.figure(figsize=(16,8))
sns.boxplot(y='SalePrice', x='OverallQual', data=train_data)
plt.title('Sales Price comaparison with Overall Qality of house')
plt.show()


# In[23]:


plt.figure(figsize=(16,8))
sns.boxplot(y='SalePrice', x='OverallCond', data=train_data)
plt.title('Sales Price comaparison with Overall condition of house')
plt.show()


# In[24]:


plt.figure(figsize=(16,8))
sns.scatterplot(y='SalePrice', x='LotArea', data=train_data)
plt.show()


# In[25]:


plt.figure(figsize=(16,6))
sns.barplot(y='SalePrice', x='YearBuilt', data=train_data)
plt.xticks(rotation=90)
plt.title('Sales Price distribution against Year Built')
plt.show()


# In[26]:


#Numerical data distribution
df_num= train_data.select_dtypes(include = ['float64', 'int64'])

df_num.head()


# In[27]:


df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); 


# In[28]:


df_corr = train_data.corr()


# In[29]:


plt.figure(figsize=(16,8))
sns.heatmap(df_corr, cmap='viridis')
plt.show()


# In[30]:


df_num_corr = df_num.corr()['SalePrice'][:-1] 
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
golden_features_list


# These all are strongly correlated with Sales Price.....

# In[32]:


import operator

individual_features_df = []
for i in range(0, len(df_num.columns) - 1): 
    temp = df_num[[df_num.columns[i], 'SalePrice']]
    temp = temp[temp[df_num.columns[i]] != 0]
    individual_features_df.append(temp)

all_correlations = {feature.columns[0]: feature.corr()['SalePrice'][0] for feature in individual_features_df}
all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))
for (key, value) in all_correlations:
    print("{:>15}: {:>15}".format(key, value))


# In[33]:


corr = df_num.drop('SalePrice', axis=1).corr()
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# ### Encoding Data

# In[34]:


#Importing Library for Encoding Data
from sklearn.preprocessing import LabelEncoder

#Defining instance for Label Encoder
encode_data = LabelEncoder()


# In[35]:


# Defining Function For encoding All Categorical columns
def CaTorigical_data(data):
    for c in cat_cols:
        data[c] = encode_data.fit_transform(data[c])


# In[36]:


#Applying Encoding function to train data
CaTorigical_data(train_data)

#Validating encoding
train_data.head()


# In[37]:


#Applying Encoding function to test data
CaTorigical_data(test_data)

#Validating encoding
test_data.head()


# In[38]:


# Importing Libraries 

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score


# In[39]:


X= train_data.drop('SalePrice', axis=1)
ytrain = train_data['SalePrice']
x = test_data


# In[40]:


# Scalling data with standard scaler
sc = StandardScaler()
X_ = sc.fit_transform(X)
x_= sc.fit_transform(x)

X = pd.DataFrame(data=X_, columns = X.columns)
x = pd.DataFrame(data=x_, columns = x.columns)
X.head()


# In[41]:


x.head() 


# In[42]:


xtrain = X
xtest= x
print(xtrain.shape, ytrain.shape, xtest.shape)


# #### Random Forrest Regressor
# 

# In[43]:


rf_regr = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=100)


# In[44]:


rf_regr.fit(xtrain, ytrain)


# In[45]:


rf_pred = rf_regr.predict(xtest)


# In[46]:


rf_regr.score (xtrain, ytrain)


# #### Tunning Random Forrests

# In[48]:


features_tuple=list(zip(X.columns,rf_regr.feature_importances_))
feature_imp=pd.DataFrame(features_tuple,columns=["Feature Names","Importance"])
feature_imp=feature_imp.sort_values("Importance",ascending=False)


# In[49]:


plt.figure(figsize=(20,4))
sns.barplot(x="Feature Names",y="Importance", data=feature_imp, color='g')
plt.xlabel("House Price Features")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.title("Random Forest Regressor - Features Importance")
plt.show()


# #### Hyperparameter Tuning using GridSearchCV

# In[50]:


param_grid1 = {"n_estimators" : [9, 18, 27, 36, 45, 54, 63, 72, 81, 90],
           "max_depth" : [1, 5, 10, 15, 20, 25, 30],
           "min_samples_leaf" : [1, 2, 4, 6, 8, 10]}

RF = RandomForestRegressor(random_state=0)
# Instantiate the GridSearchCV object: logreg_cv
RF_cv1 = GridSearchCV(RF, param_grid1, cv=5,scoring='r2',n_jobs=4)

# Fit it to the data
RF_cv1.fit(xtrain,ytrain)

#RF_cv1.cv_results_, 
RF_cv1.best_params_, RF_cv1.best_score_


# In[51]:


param_grid2 = {"n_estimators" : [45,48,51,54,57,60,63],
           "max_depth" : [16,17,18,19,20,21,22,23,24],
           "min_samples_leaf" : [1,2,3,4]} 

RF = RandomForestRegressor(random_state=0)
# Instantiate the GridSearchCV object: logreg_cv
RF_cv2 = GridSearchCV(RF, param_grid2, cv=5,scoring='r2',n_jobs=4)

# Fit it to the data
RF_cv2.fit(xtrain,ytrain)

#RF_cv2.grid_scores_, 
RF_cv2.best_params_, RF_cv2.best_score_


# #### Tuned Random Forrest

# In[52]:


RF_tuned = RF_cv2.best_estimator_


# In[53]:


RF_tuned.fit(xtrain, ytrain)
RF_tpred = RF_tuned.predict(xtest)


# In[54]:


RF_tuned.score(xtrain,ytrain)


# In[55]:


pred = RF_tuned.predict(xtrain)


# Now it's yours turn...... The Analysis almost 99% completed, you just find the scores and compare them......

# In[ ]:




