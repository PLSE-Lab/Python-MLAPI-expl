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


# # Fundamental Analysis for Graduate Admissions for Universities

# In[ ]:


df1 = pd.read_csv('../input/Admission_Predict.csv')
df2 = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df=pd.concat([df1,df2])


# In[ ]:


df.head()


# In[ ]:


df.shape


# ## The Objective is to study the given dataset and draw inferences based on the initial hypothesis and create a predictive model which can predict the chances of getting admission to an aspirant

# ## Hypothesis
# 
# 1. H1 : Each variable in the dataset is capable of predicting the chance of admission of an aspirant 
# 2. H2 : There is a positive impact of research work in increasing the chances of getting admit.
# 3. H3 : Test scores and university ranking are negatively correlated
# 4. H4 : Good SOP increases the chance of admit significantly 
# 5. H5 : Students who are good in GRE are also good in TOEFL.

# ## Data Cleaning and Data Preparation

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df.columns


# In[ ]:


# Visual manner to check the null values in the dataset
df.drop(['Serial No.'], inplace=True,axis='columns')
sns.heatmap(data=df.isnull(),cmap='viridis')


# ## Exploratory Analysis
# ### Univariate Analysis 
# 
# Considering Individual variables and there distribution 

# In[ ]:


sns.set()
sns.distplot(tuple(df['Chance of Admit ']), color='green', bins=40)


# #### The distribution of our dependent variable : Chance of Admit seem to be slightly skweed and not completely a bimaodal distribution. On increasing the bin size it appears normal. with a wider standard deviation

# In[ ]:


sns.distplot(df['University Rating'])


# In[ ]:


sns.distplot(df['University Rating'],kde=False)


# ### Bivariate Analysis 
# Comparing two variables at a time for the given dataset

# In[ ]:


def modiffy(row):
    if row['Chance of Admit '] >0.7 :
        return 1
    else :
        return 0
df['Admit'] = df.apply(modiffy,axis=1)
dftemp = df.drop(['Chance of Admit '], axis=1)
sns.pairplot(dftemp,hue='Admit')
del dftemp


# * pairplot view gives us an idea that variables GRE score, TOEFL score CGPA has a strong linear relation with chance of admit
# * The Research column seem interesting ; It appears all those with research tend to get admission

# In[ ]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:


sns.scatterplot(data=df,x='GRE Score',y='TOEFL Score',hue='Research')


# Applicants who have done Research work are the ones who have high and above average GRE scores and TOEFL Scores

# In[ ]:


sns.scatterplot(data=df, y='Chance of Admit ', x='CGPA', hue='Research')


# In[ ]:


sns.boxplot(data=df,x='SOP',y='Chance of Admit ', hue ='Research')


# ### Logistic Regression 
# 
# Now it's time to do a train test split, and train our model!
# Split the data into training set and testing set using train_test_split
# 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA','Research']]
Y = df['Admit']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# Fitting the model by using the training data


# ### Predictions and Evaluations 

# In[ ]:


# Creating the predictions using our logistic regression model 
predictions = logmodel.predict(X_test)


# In[ ]:


# Creating the classigication report for the model to check the sensetivity and specificity
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# ### Linear Regression 
# Since we have the chance of admit in continous scale we have the liberty to perform a linear regression 

# In[ ]:


from sklearn.linear_model import LinearRegression
X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA','Research']]
Y = df['Chance of Admit ']


# In[ ]:


lm = LinearRegression()
lm.fit(X_train,y_train)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[ ]:


# Pringing The Coefficients 
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# #### Predictions and Evaluations ( Linear Regression )

# In[ ]:


predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# #### Evaluating the Model 

# In[ ]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# #### Residuals
# Plot a histogram of the residuals and make sure it looks normally distributed.

# In[ ]:


sns.distplot((y_test-predictions),bins=20)


# In[ ]:




