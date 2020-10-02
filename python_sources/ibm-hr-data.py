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


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# # Importing Data

# In[ ]:


data=pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.head()


# # Data Processing
# 

# Removing EmployeeNumber column as it's ID/ we can also set it as index

# In[ ]:


data.drop('EmployeeNumber',axis=1,inplace=True)


# **Removing columns which have only one value**

# In[ ]:


for i in data.columns:
    if data[i].nunique()==1:
        data.drop(i,axis=1,inplace=True)


# In[ ]:


data.info()


# **Let's print details of columns which have Cataorical Data**

# In[ ]:


for i in data.columns:
    if data[i].dtype=='O':
        print(i)
        print(data[i].unique())
        print()


# * We can covert categorical data into a suitable data format( integer ) for fitting the ML model. For this, we can use the pandas ***get_dummies* **Function. I'll prefer to use ***LabelEncoder*** because it does not increase the dimensionality of data but get_dummies do increase the dimensionality of data which may lead to a complex ML model.
# * Here I have not used LabelEncoder instead i have written a code that is doing the same as LabelEncoder.

# In[ ]:


for col in data.columns:
    if data[col].dtype=='O':
        un = data[col].unique()
        var=0
        for i in un:
            data[col].replace(i,var,inplace=True)
            var+=1


# In[ ]:


data.head()


#  # Identifying as well as Separating Quantitative and Qualitative data
# *  cat contains the name of column which have name of ***Qualitative(categorical)*** data

# In[ ]:


cat=['Attrition','BusinessTravel','Department','Education','EducationField','EnvironmentSatisfaction',
     'Gender','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','OverTime','WorkLifeBalance',
     'StockOptionLevel','RelationshipSatisfaction','PerformanceRating'] 


# In[ ]:


cat_data= data[cat]
cat_data.head()


# # Heatmap for Qualitative Data
# 1. * For checking of highly correlated columns.
# 1. * we'll drop one of two highly correlated columns(90%) which will help us to reduce the dimensionality of data 

# In[ ]:


fig_dims = (20,15)
fig, ax = plt.subplots(figsize=fig_dims)

#mask=np.triu(np.ones_like(cat_data.corr(),dtype=bool))

cmap=sns.diverging_palette(h_neg=15,h_pos=240,as_cmap=True)
sns.heatmap(cat_data.corr(),center=0,cmap=cmap,linewidths=1,annot=True,fmt='.2f',ax=ax);


# # Ploting countplot for Qualitative Data
# To visualise distribution  

# In[ ]:


for a in cat_data.columns:
    sns.countplot(cat_data[a])
    plt.show()


# # Preparing Quantitative data

# In[ ]:


quan=data.columns.to_list()
for i in cat:
    quan.remove(i)


# In[ ]:


quan_data=data[quan]
quan_data.head()


# # Heatmap for Quantitative Data
# * For checking of highly correlated columns.
# * we'll drop one of two highly correlated columns(90%) which will help us to reduce the dimensionality of data 

# In[ ]:


fig_dims = (20,15)
fig, ax = plt.subplots(figsize=fig_dims)

#mask=np.triu(np.ones_like(quan_data.corr(),dtype=bool))

cmap=sns.diverging_palette(h_neg=15,h_pos=240,as_cmap=True)
sns.heatmap(quan_data.corr(),center=0,cmap=cmap,linewidths=1,annot=True,fmt='.2f',ax=ax);


# # Pairplot for Quantitative Data
# * For understanding correlation of quantitative data visually 

# In[ ]:


sns.pairplot(quan_data);


# # Checking data Distrubution of Quantitative Data by ploting distplot and ECDF
# * we will check for normal distrubution using this plot 
# * If continues data is not in the normal form we'll try to make it normal by necessary transformation
# * Only if we are getting low Accuracy/Score

# In[ ]:


for a in quan_data.columns:
    sns.distplot(quan_data[a])
    plt.show()


# In[ ]:


def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y


# In[ ]:


for a in quan_data.columns:
    x1,y1=ecdf(quan_data[a])
    x2,y2=ecdf(np.random.normal(np.mean(quan_data[a]),np.std(quan_data[a]),size=10000))
    plt.plot(x1,y1,marker='.',linestyle=None)
   
    plt.xlabel(a)
    plt.plot(x2,y2)
    plt.legend(['Real', 'Theory'])
    plt.show()


# # Observation from ECDF and Distplot
# * Most of columns are discrete in nature and very few columns are continues
# * As of know we'll not be making our continues data Normal
# * If the features or columns are selected during the Feature selection with that our accuracy/score is less then we'll go for making our data normal

# In[ ]:


MonthlyIncome=quan_data['MonthlyIncome']
quan_data.drop('MonthlyIncome',axis=1,inplace=True)


# # StandardScaler
# * With StandardScaler we'll be bringing all feature on the same scale
# * Scalling data always result in better Score

# In[ ]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler().fit(quan_data)

quan_data_col_name=quan_data.columns
quan_data_col_name


# In[ ]:


quan_pre_data=pd.DataFrame(ss.transform(quan_data),columns=quan_data_col_name)


# **We can observe that after scaling our distribution remain the same because scaling never change the distribution**

# In[ ]:


for a in quan_pre_data.columns:
    sns.distplot(quan_pre_data[a])
    plt.show()


# In[ ]:


quan_pre_data.head()


# In[ ]:


cat_data.head()


# In[ ]:


MonthlyIncome.head()


# # Feture Selection for reducing complexcity of our ML model
# 
# * As we know Lasso, RandomForestRegressor, GradientBoostingRegressor work on by the principal of features selection or features importance. 
# * We can use this information for selecting important columns for the prediction.
# 
# * We have used LassoCV which Lassos regressor with the implementation ofgrid search for Hypertuning well know as L2 regularization

# In[ ]:


from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from math import sqrt


# **Bring Quantitative and Qualitative data together for Preparing the ML model**
# * and diving data into training and testing

# In[ ]:


y=MonthlyIncome
x=pd.concat([quan_pre_data,cat_data],axis=1)

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=12)
X_train.iloc[:,:13]


# *Fitting data to L2 regularization *

# In[ ]:


lcv=LassoCV().fit(X_train,y_train)
lcv.alpha_


# In[ ]:


print(lcv.score(X_train,y_train))
print(lcv.score(X_test,y_test))


# Masking mask of columns that have coefficients more than Zero. ie, Important features

# In[ ]:


lcv_mask=lcv.coef_!=0
lcv_mask


# # Trainng RandomForestRegressor, GradientBoostingRegressor for feature Selection 
# For creating a mask of important feature by both algorithm

# In[ ]:


rfr=RandomForestRegressor().fit(X_train,y_train)
gbr=GradientBoostingRegressor().fit(X_train,y_train)


# In[ ]:


print(rfr.score(X_train,y_train))
print(rfr.score(X_test,y_test))
print()
print(gbr.score(X_train,y_train))
print(gbr.score(X_test,y_test))


# In[ ]:


rfr.feature_importances_


# # Plotting for Important feature by RandomForestRegressor 

# In[ ]:


importances_rf=pd.Series(rfr.feature_importances_,index=X_train.columns)
importances_rf_sort=importances_rf.sort_values()
importances_rf_sort.plot(kind='barh',figsize=(10,10));


# # Plotting for Important feature by GradientBoostingRegressor 

# In[ ]:


importances_gbr=pd.Series(gbr.feature_importances_,index=X_train.columns)
importances_gbr_sort=importances_gbr.sort_values()
importances_gbr_sort.plot(kind='barh',figsize=(10,10),color='red');


# # Plotting for Important feature by Lasso 

# In[ ]:


importances_lcv=pd.Series(lcv.coef_,index=X_train.columns)
importances_lcv_sort=importances_lcv.sort_values()
importances_lcv_sort.plot(kind='barh',figsize=(10,10),color='green');


# # Selecting top 15 feature from RandomFroestRegressor and GradientBoostingRegressor with RFE
# we are using RFE (Recursive feature elimination) for more control on feture elimination

# In[ ]:


rfe_rfr=RFE(estimator=RandomForestRegressor(), n_features_to_select=15, step=3, verbose=1).fit(X_train,y_train)


# In[ ]:


print(rfe_rfr.score(X_train,y_train))
print(rfe_rfr.score(X_test,y_test))


# In[ ]:


sqrt(mean_squared_error(y_test,rfe_rfr.predict(X_test)))


# In[ ]:


rfr_mask=rfe_rfr.support_


# In[ ]:


rfe_gbr=RFE(estimator=GradientBoostingRegressor(), n_features_to_select=15, step=3, verbose=1)
rfe_gbr.fit(X_train,y_train)


# In[ ]:


print(rfe_gbr.score(X_train,y_train))
print(rfe_gbr.score(X_test,y_test))


# In[ ]:


gbr_mask=rfe_gbr.support_
votes=np.sum([lcv_mask,gbr_mask,rfr_mask],axis=0)
votes


# # Voting for Feature selection by all three algorithm

# In[ ]:


mask = votes>=2
mask


# In[ ]:


mask_data=X_train.loc[:,mask]
mask_data.head()


# # Applying PCA for feature extraction
# Let's apply Principal component analysis and let's try to reduce more complexcity of our data
# PCA help our model to not get overfit
# Note:- Before applying PCA make sure you data is Scaled otherwise our model will underfit

# In[ ]:


from sklearn.decomposition import PCA
pca=PCA().fit(mask_data)


# In[ ]:


pca=PCA().fit(mask_data)

print(pca.explained_variance_ratio_)
print()
print(print(pca.explained_variance_ratio_.cumsum()))


# ***As we can observe we need all columns to cover the full variability of our data***
# if we okay to remove few columns to make our model less complex, we can remove them but it may lead to a bit less accuracy (this is generally used to a data set which has too many columns)
# But in our data we have very less no. of columns and we want all feature for full variblity in this case. so, we'll be not Using PCA

# In[ ]:


len(pca.components_)


# # Plotting for elbow
# Plotting for elbow point which determine no. of column of PCA which can be used for maximum variance in data
# This is genrally used for Data set which have too many columns like more then 100.

# In[ ]:


plt.plot(pca.explained_variance_ratio_)


# 
# here elbow point comes on 3 that means for maximum variability by least no.of feature is 3+1=4. plus 1 because of plot stats from 0 which means 0 is also considered as column similar to indexing

# In[ ]:


del data,quan_data,cat_data,quan_pre_data


# # Fitting Final ML model and Hyper parameter tuning Using **RandomizedSearchCV**
# Here we'll be fitting the ML model which contains only important data. ie, mask_data(data from feature Selection)

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist={'loss':['ls', 'lad', 'huber', 'quantile'],
           'n_estimators':randint(100,200),
           'max_depth':randint(1,5)
           }
cv=GradientBoostingRegressor()

final_modelCV=RandomizedSearchCV(cv,param_dist,cv=10,verbose=True,n_jobs=-1)


# In[ ]:


final_modelCV.fit(mask_data,y_train)


# In[ ]:


final_modelCV.score(mask_data,y_train)


# In[ ]:


final_modelCV.best_params_


# In[ ]:


final_modelCV.score(mask_data,y_train)


# In[ ]:


final_modelCV.score(X_test.loc[:,mask],y_test)


# In[ ]:


sqrt(mean_squared_error(y_train,final_modelCV.predict(X_train.loc[:,mask])))


# In[ ]:




