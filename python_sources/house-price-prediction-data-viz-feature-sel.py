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


import pandas as pd

#def load_data(path, filename, codec='utf-8'):
  #csv_path = os.path.join(path, filename)
  #print(csv_path)
  #return pd.read_csv(csv_path, encoding=codec)


# In[ ]:



path=('../input/train.csv')


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import Imputer,LabelEncoder
from scipy.stats import norm, skew
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


#reading of train data
df= pd.read_csv(path)


# In[ ]:


#checking the contents of table
df.head()


# In[ ]:


df.columns


# In[ ]:


fig = plt.figure(2)
ax1 = fig.add_subplot(2, 2, 1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax4=fig.add_subplot(2,2,4)
df.plot.scatter(x='LotFrontage',y='SalePrice',ax=ax1)
df.plot.scatter(x='LotArea',y='SalePrice',ax=ax2)
df.plot.scatter(x='MSSubClass',y='SalePrice',ax=ax3)
df.plot.scatter(x='OverallQual',y='SalePrice',ax=ax4)
plt.show()


# In[ ]:


#Cheking for nulls in target
df['SalePrice'].isnull().sum()


# In[ ]:


#assigning target to y
y=df['SalePrice']


# In[ ]:


sns.set()
cols=list(df.columns)
sns.pairplot(df[cols],size=2.5)
plt.show()


# In[ ]:


#dropping target from rest of features
X=df.drop('SalePrice',axis=1)


# In[ ]:


#checking  correlation between the independent features
sns.heatmap(df.corr())


# In[ ]:


#selecting continous features
X_con=X.select_dtypes(exclude='object')
#X_test_con=df_test.select_dtypes(exclude='object')


# In[ ]:


X_con_col=list(X_con.columns)


# In[ ]:


#outlier detection and replacing them with mean
def outlier_detect(df):
    for i in df.describe().columns:
        Q1=df.describe().at['25%',i]
        Q3=df.describe().at['75%',i]
        IQR=Q3 - Q1
        LTV=Q1 - 1.5 * IQR
        UTV=Q3 + 1.5 * IQR
        x=np.array(df[i])
        p=[]
        for j in x:
            if j < LTV or j>UTV:
                p.append(df[i].median())
            else:
                p.append(j)
        df[i]=p
    return df


# In[ ]:


#calling outlier dectectfunction
X_con=outlier_detect(X_con)


# In[ ]:


#selecting categorical data
X_cat=X.select_dtypes(include='object')


# In[ ]:


X_cat_col=list(X_cat.columns)


# In[ ]:


#checking for nulls in categorical features
X_cat.isnull().sum()


# In[ ]:


#checking for nulls in continous data
X_con.isnull().sum()


# In[ ]:


#dropping the features which have maximum values missing
X_cat.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[ ]:


X_cat.drop('Alley',1,inplace=True)


# In[ ]:


#again checking for null values----data with no missing values
X_cat.isnull().sum()


# In[ ]:


#filling missing value of categorical data  with mode
for col in X_cat.columns:
       
    X_cat[col]=X_cat[col].fillna(X_cat[col].mode()[0])


# In[ ]:


for cols in X_cat.columns:
    sns.set(style="whitegrid")
    ax = sns.barplot(x=cols, y="SalePrice", data=df)
    plt.show()


# In[ ]:


#again checking for nulls
X_cat.isnull().sum()


# In[ ]:


#checking nulls in continous features
X_con.isnull().sum()


# In[ ]:


#replacing NaN with 0 for continous features
for col in X_con.columns:
       
    X_con[col]=X_con[col].replace(to_replace=np.nan,value=0)


# In[ ]:



for cols in X_con.columns:
    sns.set()
    sns.distplot(X_con[cols])
    plt.show()


# In[ ]:


#checking for skewness of data and replacing it by sqrt if skewness>1
for feature in X_con.columns:
    if (X_con[feature].skew())>1.0:
        X_con[feature]=np.sqrt(X_con[feature])


# In[ ]:


X_con.head()


# In[ ]:


from sklearn import preprocessing


# In[ ]:


#converting the categorical values into continous values with the hep of label encoding
le = preprocessing.LabelEncoder()


# In[ ]:


for feat in X_cat:
    le.fit(X_cat[feat])
    X_cat[feat]=le.transform(X_cat[feat])


# In[ ]:


#merging continous and categorical features
XX=pd.concat([X_con,X_cat],1)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


#splitting of data
X_train,X_test,y_train,y_test=train_test_split(XX,y,test_size=.2,random_state=0)


# In[ ]:


#creating object of linear model
linear=LinearRegression()


# In[ ]:


#fitting model on X_train and y_train
linear.fit(X_train,y_train)


# In[ ]:


#predicting the value of X_test
y_pred=linear.predict(X_test)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


lin_score=r2_score(y_test,y_pred)


# In[ ]:


lin_score


# In[ ]:


###applying regularization--Lasso
from sklearn import linear_model
lasso = linear_model.Lasso(alpha=0.1)


# In[ ]:


# Ride---
ridge=linear_model.Ridge(alpha=0.1)


# In[ ]:


lasso.fit(X_train,y_train)


# In[ ]:


y_lasso=lasso.predict(X_test)


# In[ ]:


r2_score(y_test,y_lasso)


# In[ ]:


ridge.fit(X_train,y_train)
y_ridge=lasso.predict(X_test)
r2_score(y_test,y_ridge)


# ### regularization does'nt increase the accuracy...

# In[ ]:


#apply feature selection
from sklearn.model_selection import cross_validate


# In[ ]:


from sklearn.feature_selection import chi2

from sklearn.feature_selection import SelectKBest


# In[ ]:


feat_sel = SelectKBest(score_func=chi2, k=60)
X_train=feat_sel.fit_transform(X_train,y_train)
X_test=feat_sel.transform(X_test)
model=LinearRegression()
model.fit(X_train,y_train)
chi2_score=model.score(X_test,y_test)
print(chi2_score)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor(random_state=0)


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


clf.score(X_test,y_test)


# By applying decision tree accuracy is decreased...try random forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# Code starts here
rf_reg=RandomForestRegressor(random_state=9)
rf_reg.fit(X_train,y_train)

score_rf=rf_reg.score(X_test,y_test)
print(score_rf)


# HyperTuning

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


parameter_grid={'n_estimators': [20,40,60,80],'max_depth': [8,10],'min_samples_split': [8,10]}


# In[ ]:


rf_reg=RandomForestRegressor(random_state=9)
rf_reg.fit(X_train,y_train)
grid_search=GridSearchCV(estimator=rf_reg,param_grid=parameter_grid)


# In[ ]:


grid_search.fit(X_train,y_train)
score_gs=grid_search.score(X_test,y_test)
print(score_gs)


# In[ ]:


#score_gs


# In[ ]:




