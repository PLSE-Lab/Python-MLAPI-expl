#!/usr/bin/env python
# coding: utf-8

# This "Graduate Admissions" kernel will help you to understand the current admission trend and also how your chance of admission depends on every factor such as GRE Score, LOP etc.
# 
# This is my first try.
# So, please provide your valuable feedback.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import utils
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


ap = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
ap.head()


# 1.Now lets see if the data types of all the colums are same are not

# In[ ]:


print(ap.dtypes)


# 2. Basic data clean 
# 
# Let see if there is any missing data in all the columns
# 

# In[ ]:


ap.info()


# It looks like there is no missing data.
# Some of the columns have space between them.
# Let us rename them.

# In[ ]:


#rename the labels

ap.rename(columns={"Serial No.":"Serial_No.","GRE Score":"GRE_Score","TOEFL Score":"TOEFL_Score","University Rating":"University_Rating", "Chance of Admit":"COA"},inplace = True)


# In[ ]:


ap.head(2)


# In[ ]:


display(ap.University_Rating.nunique())
ap.University_Rating.unique()


# In[ ]:


display(ap.Research.nunique())
ap.Research.unique()


# Building the corrleation  matrix

# In[ ]:


sns.heatmap(ap.corr())


# Now let us plot graphs
# 
# 1. GRE Score vs. TOEFL Score
# 

# In[ ]:


Graph = sns.regplot(x="GRE_Score", y="TOEFL_Score", data=ap)
plt.title("GRE Score vs TOEFL Score")
plt.show()


# * So, It looks like more the GRE score more is the TOEFL Score.
# We can deduce that the students who gets high GRE Score also get high TOEFL Score.

# In[ ]:


Graph = sns.regplot(x="GRE_Score", y="CGPA", data=ap)
plt.title("GRE Score vs CGPA")
plt.show()


# We can see the same result as we have seen before

# In[ ]:


ap.groupby('University_Rating')['GRE_Score'].nunique().plot(kind='bar')
plt.show()


# So, it looks like University with 4 rating gets most of the admission

# How about some machine learning models:
#     
# Lets split the data
# 

# In[ ]:


from sklearn.model_selection import train_test_split

X=ap.iloc[:,:-1]
y=ap.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[ ]:


#Applying L.R

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print("score : ",r2_score(y_test, y_pred))


# So, Applying linear regression we are getting a score of 78%
