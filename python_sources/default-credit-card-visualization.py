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


# 
# ****<h1/> Logistic Regression Model to find credit Card defaulter  </h1>****

# 
# ### 1. Dataset Description 
# 
# The dataset consists of 10000 individuals and whether their credit card has defaulted or not. Below are the column description:
# - **default** : Whether the individual has defaulted
# - **student** : Whether the individual is student
# - **balance** : The balance in individual's account
# - **income** : Income of individual

# ### 2. Importing the packages and dataset  

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas_profiling as pds
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#Load  and reading Credit Default datasets
cred_df = pd.read_csv("../input/attachment_default.csv")


# ### Performing "Profiling of Datasets" which depicts the colinearity of column variable and other important insights

# In[ ]:


cred_df.head()
detail_report= pds.ProfileReport(cred_df)
detail_report.to_file("default_card.html")


# ### 3. Exploratory Data Analysis and finding its colinearity between the column features  <a id='eda'>

# In[ ]:


cred_df.info()
sns.heatmap(cred_df.corr(),annot=True,cmap= "YlGnBu")


# #### The following figure shows there is having colinearity between balance and income to be -0.15 and same hold vice-Versa

# ** Observation ** : There are no missing values

# In[ ]:


# Relation between balance and default
sns.boxplot(x='default', y='balance', data=cred_df,palette="Set2")
#sns.catplot(x='balance', y='income',col='student',data=cred_df, kind= "box")
plt.show()


# In[ ]:


# Relation between income and default
sns.boxplot(x='default', y='income', data=cred_df)
plt.show()


# In[ ]:


# Relation between balance and income and whether they have defaulted or not 

sns.lmplot(x='balance', y='income', hue = 'default', data=cred_df, col='student',aspect=1.5, fit_reg = False)

sns.catplot(x='default', y='income', data=cred_df,hue='default',col='student', kind='boxen')


#plt.figure(figsize=(6,8))
#g=sns.FacetGrid(cred_df, row='balance',col='income')
#g=g.map(plt.scatter,"default")

plt.show()


# In[ ]:


# Relation between Student and default value representation

pd.crosstab(cred_df['default'], cred_df['student'], rownames=['Default'], colnames=['Student'])


# ### 4. Feature Engineering :  [ Converting Categorial Variable to Numerical Veriable]  <a id='feature'>

# In[ ]:


# Convert Categorical to Numerical for default column

default_dummies = pd.get_dummies(cred_df.default, prefix='default', drop_first= True)
cred_df = pd.concat([cred_df, default_dummies], axis=1)

cred_df.head()


# In[ ]:


# Convert Categorical to Numerical for student column

student_dummies = pd.get_dummies(cred_df.student, prefix='student', drop_first= True)
cred_df = pd.concat([cred_df, student_dummies], axis=1)
cred_df.head()


# In[ ]:


# Removing repeat columns
cred_df.drop(['default', 'student'], axis=1, inplace=True)


# ### 5. Building and Evaluating Models  <a id='modelling'>

# ### 5.1 Simple Linear Regression  <a id='linear'>

# In[ ]:


# Try simple linear regression on the data between balance and default

sns.lmplot(x='balance', y='default_Yes', data=cred_df, aspect=1.5, fit_reg = True)


# ### Buiding the Machine Learning Model : Linear Regression

# In[ ]:


# Building Linear Regression Model and determining the coefficients
from sklearn.linear_model import LinearRegression

x= cred_df[['balance']]
y= cred_df[['default_Yes']]

linreg= LinearRegression()
linreg.fit(x,y)
print(linreg.coef_)
print(linreg.intercept_)


# ### 5.2 Buiding the Machine Learning Model : Logistic Regression  <a id='logistic'>

# In[ ]:


# Building the Logistic Regression Model

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1e42)                            # Set Large C value for low regularization
logreg.fit(x, y)

print(logreg.coef_)                                            # Coefficients for Logistic Regression
print(logreg.intercept_)

y_pred = logreg.predict_proba(x) 
plt.scatter(x.values, y_pred[:,0])                             # Visualization
plt.scatter(x.values, y)
plt.show()


# In[ ]:


cred_df.head()


# #### Spliting X and Y for Train Test Split in  Logistics Regression

# In[ ]:


# splitting the features and labels

X= cred_df.drop('default_Yes', axis=1)
y = cred_df['default_Yes']

# splitting the data into train and test with 70:30 ratio
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train,y_test= train_test_split(X,y, random_state= 123,test_size=0.30)

# calling logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression(C=.1)


# In[ ]:


# fitting the model with x and y attributes of train data
# in this it is goin to learn the pattern
logreg.fit(x_train,y_train)


# In[ ]:


# now applying our learnt model on test and also on train data
y_pred_test = logreg.predict(x_test)
y_pred_train = logreg.predict(x_train)


# ### 5.2.1. Metrics for Logistic Regression<a id="matrix">

# In[ ]:


# comparing the metrics of predicted lebel and real label of test data
metrics.accuracy_score(y_test, y_pred_test)


# In[ ]:


# comparing the metrics of predicted lebel and real label of test data
metrics.accuracy_score(y_train, y_pred_train)


# ### 5.2.2. Representing The Confusion Metrics<a id="conf">

# In[ ]:


# creating a confusion matrix to understand the classification
conf = metrics.confusion_matrix(y_test, y_pred_test)


# In[ ]:


cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(conf,cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,
            fmt='d')


# In[ ]:




