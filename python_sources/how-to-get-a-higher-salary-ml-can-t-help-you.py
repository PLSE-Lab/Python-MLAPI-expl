#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# It is extremely important to get as high a salary as possible right after college, to pay off loans or other immediate needs. However, getting a better salary is not as straightforward as getting placed (see my kernel for getting placed: https://www.kaggle.com/yushg123/eda-and-key-insights). Thus, we need to identify what you can do to get a higher salary. 
# 
# I will also be making a simple linear regression model and Random Forest using the insights found.
# 
# Side Note: The models are not very predictive, meaning that finding the salary is not easy, and depends largely on factors not included in the data. My guess is that it is related to the extra-curricular work you do, so that you can show off your skills to the employer.
# 
# Thanks to Ben Roshan D for sharing this dataset. 
# 
# Also, please upvote, as I am trying to advance on the tiers. Feel free to also give me feedback, as it helps me improve.

# First, let us import the necessary modules and the data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


data


# # Salary Distribution
# 
# Before diving into the factors affecting the salary, let us explore the distribution of the salary itself.

# In[ ]:


plt.hist(data['salary'], bins = 50)


# So, it looks like most people have a salary between 200,000 and 300,000. A sizeable number of people also have a salary above 300,000 but below 500,000. Above that, it is very unlikely (unless you are a prodigy :). 

# In[ ]:


ax = sns.boxplot(x=data["salary"])


# Yeah, so even the box and whisker plot says the same. Most salaries above 400,000 are outliers.
# 
# The interquartile range (middle 50% of salary) lies in between the 250,000 and 300,000.
# 
# 
# 
# 
# # Numerical Features - Grades
# 
# Now, lets move on to see how grades affect salary (because grades were very important when it came to getting placed).

# In[ ]:


plt.figure(figsize=(14,12))
data2 = data.loc[:,data.columns != 'Id']
sns.heatmap(data2.corr(), linewidth=0.2, cmap="YlGnBu", annot=True)


# Not a high correlation for the numerical features with the salary. Let us plot the scatter plots to see what is going on.

# In[ ]:


columns = data.columns
for col in columns:
    if data[col].dtype != 'object' and col != 'sl_no' and col != 'salary':
        plt.scatter(data[col], data['salary'])
        plt.xlabel(col)
        plt.ylabel('Salary')
        plt.show()


# The graph is almost flat. So getting grades is important to get placed. But good grades DOES NOT mean good salary.
# 
# So getting good grades is important to get placed, but after that, companies don't care about grades. This removes all of the numerical features, as there is no significant correlation anyway. (However, if we are using a more advanced model (like LightGBM, or XGBoost, we will take these scores into account. Otherwise, this will just confuse a simple linear regression model.)
# 
# 
# # Categorical Features
# Let's have a look at some categorical features now.

# In[ ]:


ax = sns.barplot(x="gender", y="salary", data=data)


# Companies will pay a male more than a female, but there isn't anything a person can do about that. We will include this in our model, but it is not a factor a student can control.

# In[ ]:


ax = sns.barplot(x="workex", y="salary", data=data)


# Again: Go for an internship. Having work experience makes you more valuable material for companies, and you can get a better salary.

# In[ ]:


ax = sns.barplot(x="ssc_b", y="salary", data=data)


# In[ ]:


ax = sns.barplot(x="hsc_b", y="salary", data=data)


# Companies don't care about which board you took in School for salary. I guess the reason for this is that none of the boards teach anyone about work, so no board has an advantage.

# In[ ]:


ax = sns.barplot(x="degree_t", y="salary", data=data)


# In[ ]:


ax = sns.barplot(x="hsc_s", y="salary", data=data)


# In[ ]:


ax = sns.barplot(x="specialisation", y="salary", data=data)


# Specializations matter. Companies look at the 12th grade stream (Take science or commerce, companies don't like arts), college degree (take science and tech, companies appreciate the knowledge) and MBA specialisation (Take Marketing and Finance, companies don't appreciate HR).

# # Random Forest and Linear Regresion for Salary
# 
# So, as we saw, a simple linear regression model won't be able to handle the numerical features (We can try to use a more complex Regression, where we should include the scores). For now, we are just going to use the categorical features, as all of them are extremely important.

# We will be using Sklearn Label Encoding here. I find this better as there is a certain ordinal nature to the categorical variables. Also, using OneHotEncoding will create too many columns for such a shallow dataset.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data = data[data['status'] == 'Placed']

cols_to_use = ['gender', 'workex', 'degree_t', 'hsc_s', 'specialisation']
for col in data.columns:
    if data[col].dtype == 'object':
        le.fit(data[col])
        data[col] = le.transform(data[col])


# In[ ]:




y = data['salary']
#x = data[cols_to_use]
x = data.drop(['sl_no', 'salary'], axis=1)
x = pd.get_dummies(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

lr = LinearRegression()

lr.fit(x_train, y_train)

rf = RandomForestRegressor()
rf.fit(x_train, y_train)

print('R2 score of linear regression is ' + str(lr.score(x_test, y_test)))
print('R2 score of random forest is ' + str(rf.score(x_test, y_test)))


# One reason for the bad score are the outliers which I have included. Removing them would give a much better result.

# # ML can't predict your salary 
# Both of the models perform very poorly, showing that none of the features in the dataset are very predictive. This means that you need to think outside the box.. or the data.
# 
# Extra-curriculars play a big role, so you need to show the interviewer that you have done work and can add value to the company.

# # Conclusion
# 
# Here are a few things to keep in mind:
# 1. Specialisations Matter. Choose the right one.
# 2. Go for Internship. Work Experience helps.
# 3. Don't worry about grades for salary (although you need them to get placed).
# 
# 
# Please upvote this notebook if you find it helpful.

# In[ ]:




