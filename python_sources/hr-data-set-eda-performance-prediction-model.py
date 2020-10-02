#!/usr/bin/env python
# coding: utf-8

# # **HR Data Set - EDA & Performance prediction model**
# 
# This is my first project attempting to analyze a data set without any real guidelines. Feedback would be greatly appreciated! 
# 
# For my project, I've decided to try to create a employee performance prediction model.

# In[ ]:


#Importing libraries for EDA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Data Exploration

# In[ ]:


df = pd.read_csv('../input/human-resources-data-set/HRDataset_v13.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe().transpose()


# In[ ]:


#Exploring null values in the dataset, there seems to be quite a few
df.isnull().sum() 


# In[ ]:


#Getting column names for individual exploration
df.columns


# Here I start to check for typos in value names for further data cleaning

# In[ ]:


df['Sex'].unique()


# In[ ]:


df['Department'].unique()


# In[ ]:


df['Position'].unique()

# 'Data Analyst' is written in two different ways


# In[ ]:


df['Position'].replace('Data Analyst ','Data Analyst',inplace=True)


# In[ ]:


df['RecruitmentSource'].unique()


# In[ ]:


df['Department'].unique()


# In[ ]:


#Dropping null values
df.dropna(how='all', inplace=True)


# In[ ]:


#Exploring performance score and it's ID
df[['PerformanceScore','PerfScoreID']]


# In[ ]:


print(df['PerformanceScore'].unique())
print(df['PerfScoreID'].unique())


# Let's now explore some metrics within the company

# In[ ]:


#Male vs Female distribution
sns.countplot(x='Sex',data=df,palette='RdBu',saturation=1)


# In[ ]:


#Distribution of female and male employees among the different departments of the company
plt.figure(figsize=(12,6))
sns.countplot(x='Department',data=df,hue='Sex',palette='RdBu',saturation=1)


# In[ ]:


#Distribution of female and male employees among the different departments of the company
plt.figure(figsize=(12,6))
sns.countplot(x='EmpSatisfaction',data=df,saturation=1)
print(df['EmpSatisfaction'].mean())
#overall decent employee satisfaction with a mean of 3.89


# Let's now drop some columns that are irrelevant

# In[ ]:


df.drop(['DaysLateLast30','LastPerformanceReview_Date',
         'DateofTermination','TermReason','DaysLateLast30','Zip'],axis=1,inplace=True)


# In[ ]:


plt.figure(figsize=(16,16))
sns.heatmap(df.corr(),annot=True)


# there seems to be slight correlation between Employee Satisfaction and the score they get on their Performance

# In[ ]:


#Exploring how each department scores on their performances

plt.figure(figsize=(12,6))
sns.countplot(data=df,x='Department',hue='PerfScoreID')


# ### Creating the machine learning model
# 
# Let's now start creating a simple linear regression model to predict employee performance 

# In[ ]:


from sklearn.model_selection import train_test_split


# For the features I chose the following: EmpSatisfaction, PayRate, ManagerID, SpecialProjectsCount, and EngagementSurvey
# 
# Employee Satisfaction had the highest correlation with employee performance. Payrate seemed like it would be worth exploring even though it had negligable correlation with employee performance. As for the managers, they also seemed to not have any correlation with employee performance.

# In[ ]:


X = df[['EmpSatisfaction','PayRate','SpecialProjectsCount','ManagerID']]
y = df['PerfScoreID']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# Here we are getting an error. It seems that one of the dimensions chosen has missing values. Let's explore.

# In[ ]:


df[['EmpSatisfaction','PayRate','SpecialProjectsCount','ManagerID']].isna().sum()

#looks like there are 8 missing values in the ManagerID column


# In[ ]:


df[df['ManagerID'].isna()]

#the manager Webster Butler has no ID assigned to him in 8 different instances


# In[ ]:


df[df['ManagerName']=='Webster Butler'][['ManagerName','ManagerID']]

#it seems like the 8 rows with no ManagerID have been left empty by accident as Webster Butler
#has the ID 39 assigned to him, let's quickly fix that


# In[ ]:


df['ManagerID'] = df['ManagerID'].replace(np.nan, 39.0)


# In[ ]:


df[df['ManagerName']=='Webster Butler'][['ManagerName','ManagerID']]

#All fixed now


# In[ ]:


#Let's try this again

X = df[['EmpSatisfaction','PayRate','SpecialProjectsCount','ManagerID']]
y = df['PerfScoreID']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


#fitting the corrected data
lm.fit(X_train,y_train)


# In[ ]:


#Predicting values

predictions = lm.predict(X_test)


# ### Evaluating the Model
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).

# In[ ]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ## Residuals
# 
# Let's quickly explore the residuals to make sure everything was okay with our data. 

# In[ ]:


sns.distplot((y_test-predictions),bins=50);

#Normally distributed residuals, so we're good.


# In[ ]:


coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# ### Interpreting the coefficients:
# 
# - Holding all other features fixed, a 1 unit increase in **EmpSatisfaction** is associated with an **increase of performance score by ~ 0.25 points**.
# - Holding all other features fixed, a 1 unit increase in **PayRate** is **not** associated with an **increase of performance score**.
# - Holding all other features fixed, a 1 unit increase in **SpecialProjectsCount** is **not** associated with an **increase of performance score**.
# - ManagerID, as suspected, does not affect performance. It also does not make sense to say that increasing a unit in ManagerID would lead to an increase in performance score

# ## Concluding points
# 
# It seems that employee satisfaction is the most important factor in predicting employee performance, and the two are positively correlated, albeit slightly. The company might be interested in improving employee performance through increasing satisfaction using researched strategies, perhaps to increase the satisfaction of those who scored a 3.0 or less on the Employee Satisfaction survey. 
# 
# This was a straightforward project, however I am definitely going to come back to it and explore further.

# In[ ]:




