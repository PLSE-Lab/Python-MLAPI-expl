#!/usr/bin/env python
# coding: utf-8

# This is my first Kaggle submission. In this Kernel I will working on Exploratory Data Analysis and prediction using Random Forest Classifier. Please comment if you think I could have made any improvements.

# ### Problem Statement and Data Description
# 
# Why are our best and most experienced employees leaving prematurely? Have fun with this database and try to predict which valuable employees will leave next. Fields in the dataset include:
# 
# * Satisfaction Level
# * Last evaluation
# * Number of projects
# * Average monthly hours
# * Time spent at the company
# * Whether they have had a work accident
# * Whether they have had a promotion in the last 5 years
# * Departments (column sales)
# * Salary
# * Whether the employee has left

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


from scipy.stats import chisquare


# In[ ]:


import plotly.figure_factory as ff
from plotly.graph_objs import graph_objs


# In[ ]:


hr_data = pd.read_csv('../input/HR_comma_sep.csv')


# In[ ]:


hr_data.info()


# Sales and Salary seems to be containing textual data. All others are numeric data.

# In[ ]:


hr_data.head()


# So sales and salary are categorical variables.
# I am suspecting work_accident and promotion_last_5years is also a binary variable. Lets check.

# In[ ]:


def getUnique(colName):
    return hr_data.loc[:,colName].unique()


# In[ ]:


arr = ['Work_accident', 'promotion_last_5years']
for cols in arr:
    print(cols, getUnique(cols))


# As suspected both are binary variables and categorical in nature.
# Let's check for unique values from sales and salary

# In[ ]:


arr = ['sales', 'salary']
for cols in arr:
    print(cols, getUnique(cols))


# Sales - contains 10 unique factors <br>
# Salary - contains 3 unique factors

# In[ ]:


hr_data.isnull().sum()


# There are no null values in dataset. In this dataset we don't need to impute any null values.

# ### Attrition Ratio

# In[ ]:


hr_data[hr_data.left == 1].shape[0]  /  hr_data.shape[0]


# Within this data 23.81% of people have left

# ## Exploratory Data Analysis - Categorical Variables

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def edaCat(col):
    hr_left = hr_data[hr_data.left == 1][col].value_counts()
    hr_noleft = hr_data[hr_data.left == 0][col].value_counts()
    rate = hr_left / (hr_left + hr_noleft) 
    df = pd.DataFrame({'total':hr_left+ hr_noleft,'1': hr_left, '0' : hr_noleft, 'rate':rate})
    df = df.sort_values(by='rate', ascending=False)
    print('Cross Tab')
    print(df)
    #Plotting graph of rate and total
    df['rate'].plot(secondary_y = True, label='rate',colormap=plt.get_cmap('jet') , title='Rate vs Total - ' + col)
    df['total'].plot(kind='bar', label='total_count')


# In[ ]:


edaCat('sales')


# Peopl who are in HR and accounting departments are more likely to leave the company. People in RandD and management department are less likely to leave.

# In[ ]:


edaCat('salary')


# People who have low salary are more likely to leave that medium and high salary. As obvious people with higher salary are very less likely to leave company.

# In[ ]:


edaCat('Work_accident')


# In this case people who didn't had any accidents are more likely to leave. It is not much making sense, but we will try this variable in our model for accuracy as it is significant feature.

# In[ ]:


edaCat('promotion_last_5years')


# Here it is an obvious thing as people who didn't had promotion in last 5 years are very much likely to leave than who had.

# ## Exploratory Data Analysis - Numerical Variables

# In[ ]:


num_features = ['satisfaction_level', u'last_evaluation', u'number_project', u'average_montly_hours']


# We will first perform t-test on this data to identify features which are significant.

# In[ ]:


from scipy.stats import ttest_ind


# In[ ]:


def edaNumeric(cols):
    dist_left = pd.DataFrame(hr_data[hr_data.left==1][cols].describe())
    dist_stay = pd.DataFrame(hr_data[hr_data.left==0][cols].describe())
    
    ttest = ttest_ind(hr_data[hr_data.left == 1][cols], hr_data[hr_data.left == 0][cols])
    print("{} T test p-value {}".format(cols,ttest.pvalue))
    
    print(' ')
    
    print(dist_stay.merge(dist_left,left_index=True, right_index=True))
    sns.violinplot(x="left", y=cols, data=hr_data)


# In[ ]:


edaNumeric("satisfaction_level")


# As per above table and violin distribution graph we can see a obvious thing as people who left were not much satisfied. Most of the people who stay (75%) had satisfaction level more than 0.5. Whereas people who left are categorized into three groups. First who had very less satisfaction level. Second group had satisfaction level around 0.4 and last group had around 0.9.

# In[ ]:


edaNumeric('last_evaluation')


# T-test score is not significant. Distribution for people who left shows bi-modal graph. Here some people had very low evaluation, so ofcourse they left. Few people had higher evaluation but still left. More exploration is needed in this case.

# In[ ]:


edaNumeric('number_project')


# People who had very less projects (2) left and even people who had many projects left due to over pressure.

# In[ ]:


edaNumeric('average_montly_hours')


# Some people were working very hard and spending extra monthly hours left due to work pressure.

# ### Train Test Data Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


np.random.seed(42)
train_set, test_set = train_test_split(hr_data, test_size = 0.2)


# We are splitting our data into 80 , 20 ratio.

# In[ ]:


print(train_set.shape)
print(test_set.shape)


# In[ ]:


print('Train Set left ratio {}'.format(train_set[train_set.left == 1].shape[0] / train_set.shape[0]))
print('Test  Set left ratio {}'.format(test_set[test_set.left == 1].shape[0] / test_set.shape[0]))


# Above ratio of 23% is very similar to our whole data set left ratio.

# ### Transform Variables

# In[ ]:


cat_vars = ['Work_accident', 'promotion_last_5years', 'sales', 'salary']
num_vars = num_features
label = 'left'


# In[ ]:


hr_data.loc[hr_data.Work_accident == 0, 'Work_accident'] = 'No'
hr_data.loc[hr_data.Work_accident == 1, 'Work_accident'] = 'Yes'

hr_data.loc[hr_data.promotion_last_5years == 0, 'promotion_last_5years'] = 'No'
hr_data.loc[hr_data.promotion_last_5years == 1, 'promotion_last_5years'] = 'Yes'


# In[ ]:


hr_data[cat_vars].head()


# In[ ]:


temp = pd.get_dummies(train_set[cat_vars])
train_label = train_set['left']
train_set = train_set.drop('left',axis=1)
temp1 = train_set.drop(cat_vars,axis=1)
train_set = temp1.merge(temp, right_index=True, left_index=True)


# In[ ]:


train_set.head()


# In[ ]:


temp = pd.get_dummies(test_set[cat_vars])
test_label = test_set['left']
test_set = test_set.drop('left',axis=1)
temp1 = test_set.drop(cat_vars,axis=1)
test_set = temp1.merge(temp, right_index=True, left_index=True)


# In[ ]:


test_set.head()


# ### Model Training

# We are going to use RandomForestClassifier for modelling. We will be using RandomizedSearchCV for getting best possible hyperparameters. Accuracy is the scoring method used in RandomizedSearchCV.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dists = {
    'n_estimators' : randint(low=1, high=200),
    'max_features' : randint(low=2, high=10)
}

forest_cls = RandomForestClassifier(random_state=42)
rnd_search = RandomizedSearchCV(forest_cls , param_distributions = param_dists, n_iter = 10, cv=8, scoring='accuracy', random_state=42)

rnd_search.fit(train_set, train_label)


# In[ ]:


cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.average(mean_score), params)


# We have highest accuracy of 0.9912 for two cases. We will be using 'max_features' as 9 and 'n_estimators' as 152 to train our model.

# In[ ]:


random_fr = RandomForestClassifier(max_features=9, n_estimators=152)
random_fr.fit(train_set, train_label)


# In[ ]:


pred = random_fr.predict(test_set)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(pred, test_label)


# In[ ]:


pd.crosstab(pred, test_label)


# We are getting accuracy measure of 98.90% on our test set of data. Which is really a good measure.

# In[ ]:




