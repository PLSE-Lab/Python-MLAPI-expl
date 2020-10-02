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


# # OBJECTIVE:
# To predict whether a customer will buy a credit card or not.

# In[ ]:


# IMPORTING PACKAGES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Importing file
df = pd.read_csv("../input/svm-classification/UniversalBank.csv")


# In[ ]:


# To display the top 5 rows.
df.head()


# In[ ]:


# Total number of rows and columns.
df.shape


# The dataset contains 5000 rows and 14 columns.

# In[ ]:


# Display column names.
df.columns


# ![](http://)The column names are displayed above.

# In[ ]:


df.info()


# In[ ]:


# Statistics.
df.describe()


# In[ ]:


# Finding the null values.
df.isnull().sum()


# There are no missing values.

# In[ ]:


# Inspecting unique values in each column.
df.nunique(axis=0)


# # UNIVARIATE ANALYSIS

# In[ ]:


# Histogram
df.hist(figsize=(20, 20));


# In[ ]:


df_num=df.loc[:,'ID':'Mortgage']
for i in df_num.columns:
    sns.distplot(df_num[i])
    plt.show()


# OBSERVATION :       
# 1) Personal Loan, Securities Account, CD Account, Online, CreditCard are binary features.    
# 2) Age and Experience show normal distribution.       
# 3) CCAvg, Income, Mortage are right skewed.    
# 4) ID and ZIP Code do not provide any relevant information.     

# In[ ]:


del df['ID']
del df["ZIP Code"]


# # BIVARIATE ANALYSIS

# In[ ]:


sns.pairplot(data = df);


# The plot is difficult to understand. Pairplot is not suitable when many features are present.

# In[ ]:


plt.subplots(figsize=(10,8))
sns.heatmap(df.corr());


# 1) Maximum correlation can be seen between Age & Experience.      
# 2) CCAvg and Income show correlation of around 0.60.     
# 3) Personal Loan and Income also show correlation of around 0.50.       

# # MODELLING

# ****Decision Tree Model****

# In[ ]:


X = df.iloc[:,:-2].values
y = df.iloc[:, -1].values


# In[ ]:


from sklearn.model_selection import train_test_split
trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.20)


# In[ ]:


from sklearn import tree
from sklearn.metrics import accuracy_score, mean_absolute_error

# Fit / train the model
dtc = tree.DecisionTreeClassifier()
dtc.fit(trainx,trainy)


# In[ ]:


# Get the prediction for both train and test
pred_train = dtc.predict(trainx)
pred_test = dtc.predict(testx)

# Measure the accuracy of the model for both train and test sets
print("Accuracy on train is:",accuracy_score(trainy,pred_train))
print("Accuracy on test is:",accuracy_score(testy,pred_test))


# In[ ]:


# Max_depth = 3

dtc_2 = tree.DecisionTreeClassifier(max_depth=3)
dtc_2.fit(trainx,trainy)

pred_train2 = dtc_2.predict(trainx)
pred_test2 = dtc_2.predict(testx)

print("Accuracy on train is:",accuracy_score(trainy,pred_train2))
print("Accuracy on test is:",accuracy_score(testy,pred_test2))


# Accuracy obtained on test-set is 76%.

# ****SVM****

# In[ ]:


from sklearn.svm import SVC
# Fit
svc.fit(trainx,trainy)

# Get the prediction for both train and test
train_predictions = svc.predict(trainx)
test_predictions = svc.predict(testx)


# In[ ]:


# Measure the accuracy of the model for both train and test sets
print(accuracy_score(trainy,train_predictions))
print(accuracy_score(testy,test_predictions))


# Accuracy obtained on test-set is 71.9%.

# COMPARISON :      
# Decision Tree Model is better suited for modelling as compared to SVM.
