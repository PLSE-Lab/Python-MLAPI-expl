#!/usr/bin/env python
# coding: utf-8

# ## Students Performance  in Exams
# ### Objective:
# We are going to predict Student performance in exam based on basic information about them. In this dataset we can able to apply Regression and Classification model. 
# 
# This kernel I am going to explain basic recipe for regression model.
# 
# ### Content
# 1. Load Libraries
# 2. Getting Data
# 3. Understanding Basic Info
# 4. Data Visualization / EDA
# 5. Missing Value Treatment
# 6. Preprocessing 
# 7. Model Creation
# 8. Select Best Model
# 9. Visualize our Prediction 
# 
# ### Load Libraries

# In[ ]:


#Basic Libraries
import numpy as np
import pandas as pd

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mnso

#Data PreProcessing
from sklearn.preprocessing import MinMaxScaler

#Train Test Split
from sklearn.model_selection import KFold, cross_val_score, train_test_split

#Model
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

#metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

import os
print(os.listdir("../input"))


# ### Getting Data

# In[ ]:


student_data = pd.read_csv('../input/StudentsPerformance.csv')


# ### Understanding Basic Info 
# 1. Dimesions (Rows/Columns Count)
# 2. Columns Names
# 3. Data types
# 4. Unique Values 
# 5. Missing Values

# In[ ]:


student_data.head()


# In[ ]:


student_data.shape, student_data.columns


# In[ ]:


student_data.info()


# In[ ]:


student_data.nunique()


# In[ ]:


mnso.matrix(student_data)


# From this basic information below points we observed
# 1. It contains only 1000 recordset with 8 Fields
# 2. 5 Fields are categorical variables and 3 are continuous fields
# 3. There is no missing values in this dataset. So missing value treatment is not required
# 4. There is no over all score. So we have to create Over all score Field and make it as Target variable 

# In[ ]:


pd.set_option('precision', 2)
student_data['Over_all_score'] = (student_data['math score']+student_data['reading score']+student_data['writing score'])/3
student_data.head()


# ### Data Visualization / EDA
# 1. Univarient Analysis
# 2. Bivarient Analysis

# In[ ]:


#Separate Categorical and Continous Variables
categ = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']
conti = ['math score','reading score','writing score']


# #### Univarient Analysis 

# In[ ]:


plt.figure(figsize = (10,30))
i=0
for cate in categ:
    plt.subplot(5,2,i+1)
    sns.set_style('whitegrid')
    sns.countplot(x = cate, data = student_data);
    plt.tight_layout()
    plt.xticks(rotation =90)
    i +=1
for cont in conti:
    plt.subplot(5,2,i+1)
    sns.distplot(student_data[cont])
    i+=1
plt.show()


# #### Bivarient Analysis

# In[ ]:


plt.figure(figsize = (15,30))
i = 0
for cat in categ:
    plt.subplot(5,2,i+1)
    sns.set_style('whitegrid')
    sns.barplot(x = cat, y = 'Over_all_score', data = student_data, hue = 'gender');
    plt.tight_layout()
    i+=1
plt.show()


# ### Missing Value Treatment
# Here there is no missing value in record set.
# 
# ### Pre processing 
# 1. Separate Individual and Target Variable
# 2. Scaling Data (MinMaxScale)
# 3. One Hot Encoding
# 4. Train and Test Split
# 
# #### Separate Individual and Target Variable

# In[ ]:


ind_var = student_data.iloc[:,0:8]
target_var = student_data['Over_all_score']


# #### Scaling Data

# In[ ]:


pre_data = ind_var
scale = MinMaxScaler()
pre_data[['math score','reading score','writing score']] = scale.fit_transform(pre_data[['math score','reading score','writing score']])
pre_data.head()


# ####  One Hot Encoding
# 

# In[ ]:


pre_data = pd.get_dummies(pre_data, drop_first = True)
pre_data.head()


# #### Train Test Split

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(pre_data, target_var, test_size = 0.20, random_state = 10)

print('x_train Shape is :', x_train.shape)
print('y_train Shape is :', y_train.shape)
print('x_test Shape is :', x_test.shape)
print('y_test Shape is :', y_test.shape)


# ### Model Creation

# In[ ]:


#Model
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor


# In[ ]:


models = []
models.append(['LR', LinearRegression()])
models.append(['Lasso', Lasso()])
models.append(['tree', DecisionTreeRegressor()])
models.append(['knn', KNeighborsRegressor()])
models.append(['GBM', GradientBoostingRegressor()])
models.append(['ada', AdaBoostRegressor()])


# In[ ]:


results = []
names =[]

for name, model in models:    
    kfold = KFold(n_splits = 10, random_state = 7)
    cv_result = cross_val_score(model, x_train, y_train, cv =kfold, scoring = 'r2')
    results.append(cv_result)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_result.mean(), cv_result.std())
    print(msg)


# ### Select Best Model

# In[ ]:


plt.figure(figsize = (10,5))
sns.boxplot(x = names, y = results)
plt.show()


# Linear Regression seems to be give good prediction 

# In[ ]:


lr = LinearRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)


# In[ ]:


rms = sqrt(mean_squared_error(y_test, pred))
print("Root Mean Squre Error is : %.20f" % rms)


# Root Mean Squre Error is close to 0. which means our model fits enough in training and test set.
# Now we are going to visualize our prediction
# ### Visualize our Prediction 

# In[ ]:


plt.figure(figsize = (7,7))
sns.regplot(y_test, pred)
plt.show()


# ### Conclusion
# This recipe is showing basic steps involved in Regression Model. I hope you learn some hints from this model.  
