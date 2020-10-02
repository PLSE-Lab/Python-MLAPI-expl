#!/usr/bin/env python
# coding: utf-8

# ![](https://media.giphy.com/media/2ilpqokCAJSwj9MDbz/giphy.gif)

# # Introduction

# Hi all!
# This kernel is about Predicting the money spent by People on Health.<br>
# Hoping you will like it...
# 
# Please UPVOTE if my hope works :)

# ![](https://cdn.pixabay.com/photo/2016/02/02/13/10/balloons-1175297_960_720.jpg)

# # EDA

# >Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling as pp
from scipy.stats import kurtosis,skew


# > Reading Data

# In[ ]:


dtypes={'sex':'category','smoker':'category','region':'category'}
data= pd.read_csv('../input/insurance.csv',dtype=dtypes)


# > Top 5 rows

# In[ ]:


data.head()


# > Overview of Data

# In[ ]:


pp.ProfileReport(data)


# > Removing all duplicate rows

# In[ ]:


data=data.drop_duplicates(data.columns).reset_index(drop=True)


# > Information about columns

# In[ ]:


data.info(memory_usage='deep')


# > Statistical Analysis of Numeric Columns

# In[ ]:


data.describe()


# > Number of Null Values

# In[ ]:


data.isna().sum()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,square=True,cmap='coolwarm')


# ## Univariate Analysis

# #### 1) AGE

# In[ ]:


sns.distplot(data.age, rug=True, rug_kws={"color": "g"},
                  kde_kws={"color": "r", "lw": 3, "label": "KDE"},
                  hist_kws={"histtype": "step", "linewidth": 2,
                            "alpha": 0.5, "color": "g"})


# **Age is having almost a Uniform Distribution**<br>
# **More 18-19 years old are there**

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(data.age)


# **Most insurances are done in age 18-19**

# #### 2) BMI

# In[ ]:


sns.distplot(data.bmi, rug=True, rug_kws={"color": "g"},
                  kde_kws={"color": "y", "lw": 3, "label": "KDE"},
                  hist_kws={"histtype": "step", "linewidth": 2,
                            "alpha": 0.5, "color": "b"})


# **Pure Normal Distribution(approximately) of BMI**

# ![](https://media.giphy.com/media/BuwdPoYAnoyCk/giphy.gif)

# #### 3) Children

# In[ ]:


sns.countplot(data.children)


# #### 4) SMOKER

# ![](https://media.giphy.com/media/HtyM9FYa9koCY/giphy.gif)

# In[ ]:


data.smoker.value_counts()


# In[ ]:


sns.countplot(data.smoker)


# **Huge number of non-smokers...**

# #### 5) SEX

# In[ ]:


data.sex.value_counts()


# In[ ]:


sns.countplot(data.sex)


# **Almost equal number of males and females**

# #### 6) Region

# In[ ]:


data.region.value_counts()


# In[ ]:


sns.countplot(data.region)


# **Nearly equal frequencies of each category...**

# #### 7) Charges

# In[ ]:


print(kurtosis(data.charges))
print(skew(data.charges))


# **If skewness is less than -1 or greater than 1, the distribution is highly skewed.**<br>

# In[ ]:


sns.distplot(data.charges, rug=True, rug_kws={"color": "g"},
                  kde_kws={"color": "red", "lw": 3, "label": "KDE"},
                  hist_kws={"histtype": "step", "linewidth": 2,
                            "alpha": 0.3, "color": "y"})


# ## Bivariate Analysis

# #### 1) AGE

# In[ ]:


data[data.columns].corr()['age'][:]


# In[ ]:


sns.jointplot(data.age,data.charges,kind='kde')


# #### 2) BMI 

# In[ ]:


data[data.columns].corr()['bmi'][:]


# In[ ]:


sns.regplot(data.bmi,data.charges,color='r',marker='+')


# ### Categorical Data 

# #### 1) Sex

# In[ ]:


data.sex.value_counts()


# In[ ]:


data.sex = np.where(data.sex=='male', 0, data.sex)
data.sex = np.where(data.sex=='female', 1, data.sex)


# In[ ]:


data.sex=data.sex.apply(pd.to_numeric,errors='coerce')


# #### 2) Children

# In[ ]:


data.children.value_counts()


# In[ ]:


sns.boxplot(data.children,data.charges)


# In[ ]:


'''data.children = np.where(data.children==3, 3, data.children)
data.children = np.where(data.children==4, 3, data.children)
data.children = np.where(data.children==5, 3, data.children)'''


# In[ ]:


df=pd.get_dummies(data.children,drop_first=True)
data=pd.concat([df,data],axis=1)
del data['children']


# #### 3) Region

# In[ ]:


data.region.value_counts()


# In[ ]:


sns.violinplot(data.region,data.charges)


# In[ ]:


data.region=data.region.astype('object')


# In[ ]:


df=pd.get_dummies(data.region,drop_first=True)
data=pd.concat([df,data],axis=1)
del data['region']


# #### 4) Smoker

# In[ ]:


data.smoker.value_counts()


# In[ ]:


sns.stripplot(data.smoker,data.charges,jitter=True)


# ![](https://media.giphy.com/media/SqmkZ5IdwzTP2/giphy.gif)

# In[ ]:


data.smoker = np.where(data.smoker=='no', 0, data.smoker)
data.smoker = np.where(data.smoker=='yes', 1, data.smoker)


# In[ ]:


data.smoker=data.smoker.apply(pd.to_numeric,errors='coerce')


# In[ ]:


data.info()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,cmap='coolwarm')


# ## Modelling

# In[ ]:


X= data.iloc[:,:-1].values
y= data.iloc[:,-1].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[ ]:


import xgboost
from xgboost import XGBRegressor
lr=XGBRegressor()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
y_pred_train=lr.predict(X_train)


# In[ ]:


from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
print(r2_score(y_train,y_pred_train))


# ![](https://media.giphy.com/media/WnIu6vAWt5ul3EVcUE/giphy.gif)
