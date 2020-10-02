#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Let's load the dataset and see the **attributes**

# In[ ]:


data=pd.read_csv('../input/nyc-jobs.csv')
data.head()


# Create a bar graph showing average anual and hourly salaries for a given agency

# In[ ]:


label=['min_salary','max_salary']
def salary_for_agencies(agency,salary_freq):
    try:
        posting_type_data=data[data['Salary Frequency']==salary_freq]
        agency_data=posting_type_data[posting_type_data['Agency']==agency]
        min_salary=agency_data['Salary Range From']
        max_salary=agency_data['Salary Range To']
        avg_max_salary=sum(max_salary)/len(max_salary)
        print("Count of "+salary_freq+" Job poosition in the agency:",len(max_salary))
        avg_min_salary=sum(min_salary)/len(min_salary)
        print("Minimum Avg salary for agency:",avg_min_salary)
        print("Maximum Avg salary for agency:",avg_max_salary)
        plt.bar(label,[avg_min_salary,avg_max_salary])
        plt.title("Average Min Max "+salary_freq+" Salary for "+agency)
        plt.show()
    except:
        print("No data")


# Lets check our function for agency **DEPARTMENT OF BUSINESS SERV.**

# In[ ]:


salary_for_agencies('DEPARTMENT OF BUSINESS SERV.','Annual')


# Checking hourly salries for the job postion of same company

# In[ ]:


salary_for_agencies('DEPARTMENT OF BUSINESS SERV.','Hourly')


# Checking for one more agency **NYC HOUSING AUTHORITY**

# In[ ]:


salary_for_agencies('NYC HOUSING AUTHORITY','Annual')


# Import the following libraries for finding out features of interest whick will **affect the result most** 

# In[ ]:


from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn import preprocessing


# Let's remove NaN values from column **Job Category**.The question arises why not remove NaN from column **Full-Time/Part-Time indicator**. Answer : Full-Time/Part-Time indicator contains NaN values in 245 such rows and we dont know with what value shall we replace it (if it was a int type column then we could have replaced NaN with mean,most_occured value etc but it's a character type here and I dont want to give false data).
# 

# In[ ]:


data=data.dropna(subset = ['Job Category'])
print(data.shape)
data.head()


# Label encoding those attributes which contains string datatypes

# In[ ]:


X_temp = data.iloc[:,[1,2,4,5,7,12]]
y = data.iloc[:,[10,11]]

X=X_temp.apply(preprocessing.LabelEncoder().fit_transform)


# In[ ]:


X.head()


# We have left some features which were not string type, so let's include them. Also , I am not considering columns after **"Salary Frequency"** column as some of them contains text data,too much of NaN data. **"Title Code No"** is also not relevant so leaving it.
# 
# If you think they are useful then you can include them. 

# In[ ]:


X['# Of Positions']=data.iloc[:,3]

print(X.shape)
X.head()


# In[ ]:


model = XGBClassifier()
model.fit(X, y.iloc[:,0])
# plot feature importance
plot_importance(model)
pyplot.show()


# As you can see **# Of Position** and **Posting type** are also nut much useful, So we are creating train and test set wihout taking these two columns

# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X.iloc[:,[0,2,3,4,6]],y)


# Using Linear Regression first

# In[ ]:


from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,Y_train)
predict=lr.predict(X_test)


# In[ ]:


lr.score(X_test,Y_test)


# As we can see the accuracy is poor. So we will go for RandomForest.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, Y_train);


# In[ ]:


rf.score(X_test,Y_test)


# **Future Scope**
# Use count vectorizer to list words liks skills required for a particular job. 
# 
# 

# In[ ]:




