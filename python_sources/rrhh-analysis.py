#!/usr/bin/env python
# coding: utf-8

# 
# This is my first analisys after I have been learning for many hours and doing some homeworks about exploratory and predictive methods in Data Mining. So I thing it could be a little poor, but I'll do my best.
# 
# 
# The goal is to predict whereas an employee will leave the company or not, for that we have a dataset with the following data of 14999 employees:
# 
# Here are the viables:
# 
# * **satisfaction level**
# * ** last evaluation**: Score achived in the last evaluation
# * **number project:** Number of proyects assign to the employee
# * **average monthy hours**: Average of hours worked per month.
# * **time spend company** How years the worker has worked in the company
# * **Work accident**: Did the employe suffer an accident? (1 = yes, 0 = No)
# * **promotion last 5years**: Did the employee has a promotion in the last 5 years? (1 = yes, 0 = no)
# * **sales**: Department where the employee works. Categorical [marketing, support, sales, technical, management, accounting, product_mng, IT, hr]
# * **salary**: Salary Level, Categorical variable [high, medium, low]
# * **left**: Did the employee left the company?. Categorical variable [1 = yes, 0 = no]

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../input/recursos_humanos.csv')
df.columns.values


# In[ ]:


df.head()


# In[ ]:


df.info()


# We can see that there are 8 numerical variables and 2 categorical, but we know that 'Work_accident', 'left', 'promotion_last_5years' are binary variables, so, actually, they are categorical variables. Salary has an order, so I think can be useful to have a numeric version of this variable. 
# 

# In[ ]:


df['promotion_last_5years'] = df['promotion_last_5years'].astype('object')
df['Work_accident'] = df['Work_accident'].astype('object')
df['left'] = df['left'].astype('object')
df['salary_num'] = df['salary'].apply(lambda s: 0 if s == 'low' else 1 if s == 'medium' else 2) # Salary numerical variable
df.info()


# In[ ]:


df.describe()


# - Most employees (%75) have been working in the company between 2 and 4 years.
# - Only 25% of employees are highly satisfied  (+0.81)
# - On average, people work 201 hours monthly, 50% of them work more than 200 hours and the rest 50% less than 200 hours.
# - Each employee has 4 projects on average..
# - On average,  last evaluation results are 0.7,  25% of evaluations are over 0.87 other 25% under 0.56 and over 0.36

# In[ ]:


df.corr()


# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# There some positive correlations in number_project and average_montly_hours, it means, employees with more hours have more proyects assigned. Let see a bit deeper.

# In[ ]:


res = df[['number_project', 'average_montly_hours']].groupby(['number_project'], as_index=False).mean().sort_values(by='average_montly_hours', ascending=False)
plt.plot(res['number_project'], res['average_montly_hours'])


# In[ ]:


res = df[['number_project', 'last_evaluation']].groupby(['number_project'], as_index=False).mean().sort_values(by='last_evaluation', ascending=False)
plt.plot(res['number_project'], res['last_evaluation'])


# In[ ]:


res = df[['time_spend_company', 'satisfaction_level']].groupby(['time_spend_company'], as_index=False).mean().sort_values(by='time_spend_company', ascending=False)
plt.plot(res['time_spend_company'], res['satisfaction_level'])


# In[ ]:


res = df[['time_spend_company', 'number_project']].groupby(['time_spend_company'], as_index=False).mean().sort_values(by='time_spend_company', ascending=False)
plt.plot(res['time_spend_company'], res['number_project'])


# In[ ]:


res = df[['number_project', 'time_spend_company']].groupby(['number_project'], as_index=False).mean().sort_values(by='time_spend_company', ascending=False)
res


# In[ ]:


res = df[['number_project', 'time_spend_company']].groupby(['number_project'], as_index=False).mean().sort_values(by='number_project', ascending=False)
plt.plot(res['number_project'], res['time_spend_company'])


# In[ ]:


res = df[['average_montly_hours', 'last_evaluation']].groupby(['average_montly_hours'], as_index=False).mean().sort_values(by='average_montly_hours', ascending=False)
plt.plot(res['average_montly_hours'], res['last_evaluation'])


# In[ ]:


res = df[['average_montly_hours', 'satisfaction_level']].groupby(['average_montly_hours'], as_index=False).mean().sort_values(by='average_montly_hours', ascending=False)
plt.plot(res['average_montly_hours'], res['satisfaction_level'])


# In[ ]:


res = df[['time_spend_company', 'satisfaction_level']].groupby(['time_spend_company'], as_index=False).mean().sort_values(by='time_spend_company', ascending=False)
plt.plot(res['time_spend_company'], res['satisfaction_level'])


# In[ ]:


res = df[['number_project', 'satisfaction_level']].groupby(['number_project'], as_index=False).mean().sort_values(by='number_project', ascending=False)
plt.plot(res['number_project'], res['satisfaction_level'])


# - The more projects a worker have assigned, the more monthly hours he works.
# - If a worker have a good evaluation, he will have more proyects assigned.
# - El nivel de satisfaccion de los empleados baja abruptamente alrededor de los 4 anios de trabajar en la misma y luego vuelve a subir.
# - When a worker has been worked around 4 years, his satisfaction level falls down a lot.  
# - When a worker has been worked for an average on  275 monthly hours, his satisfaction level falls down a lot.
# - Satisfaction level keeps hight while employee have assigned 3, 4 or  5 projects, otherwise, it falls down.
# - La maxima carga laboral la tiene la gente que esta entre 4 y 6 anios en la empresa.
# - People who are in the company between 4 and 6 years, they have the maximum workload. 
# 
# Now, categorical vars:

# In[ ]:


df.describe(include=['O'])


# - 85% of employees didn't suffer an accident.
# - 76% of employees didn't leave the company.
# - 98% of workers hasn't gotten a promotion .
# - There are 10 departments,  28% of people work in Sales.
# - 45% of workers have a low salary.

# In[ ]:


df['left'] = df['left'].astype('int64')
ret = df[['salary', 'left']].groupby(['salary'], as_index=False).mean().sort_values(by='left', ascending=False)
ret.plot.bar(x="salary", y="left", legend=False )


# In[ ]:


ret=df[['salary', 'satisfaction_level']].groupby(['salary'], as_index=False).mean().sort_values(by='satisfaction_level', ascending=False)
ret.plot.bar(x="salary", y="satisfaction_level", legend=False )


# In[ ]:


df[['sales', 'left']].groupby(['sales'], as_index=False).mean().sort_values(by='left', ascending=False)


# In[ ]:


ret=df[['sales', 'satisfaction_level']].groupby(['sales'], as_index=False).mean().sort_values(by='satisfaction_level', ascending=False)
ret.plot.bar(x="sales", y="satisfaction_level", legend=False )


# In[ ]:


df[['Work_accident', 'left']].groupby(['Work_accident'], as_index=False).mean().sort_values(by='left', ascending=False)


# In[ ]:


df[['promotion_last_5years', 'left']].groupby(['promotion_last_5years'], as_index=False).mean().sort_values(by='left', ascending=False)


# In[ ]:


df[['number_project', 'left']].groupby(['number_project'], as_index=False).mean().sort_values(by='left', ascending=False)


# In[ ]:


res = df[['time_spend_company', 'left']].groupby(['time_spend_company'], as_index=False).mean().sort_values(by='time_spend_company', ascending=False)
plt.plot(res['time_spend_company'], res['left'])


# In[ ]:


res = df[['average_montly_hours', 'left']].groupby(['average_montly_hours'], as_index=False).mean().sort_values(by='average_montly_hours', ascending=False)
plt.plot(res['average_montly_hours'], res['left'])


# In[ ]:


res = df[['number_project', 'left']].groupby(['number_project'], as_index=False).mean().sort_values(by='number_project', ascending=False)
plt.plot(res['number_project'], res['left'])


# In[ ]:


res = df[['time_spend_company', 'salary_num']].groupby(['time_spend_company'], as_index=False).mean().sort_values(by='time_spend_company', ascending=False)
plt.plot(res['time_spend_company'], res['salary_num'])


# - The probability that someone with a low salary leaves the company is 30% while someone with a high salary will leave the company with a probability of 6%
# - People with high salary get more satisfaction.
# - RRHH's people are the most happy and Accounting 'people are the less.
# - People who are between in theirs 4 and 6 years in the company have more probability to leave the company.
# - Probability of someone leave the company grow strongly when the worker overpass 257 hours of work.
# - Probability of someone leave the company grow when the worker has more than 3 projects assigned.
# - People who are between in theirs 4 and 6 years in the company have a low salary, if the worker has been working more than 6 year in the company the salary grows strongly.
# 
# We can guess that after the third year working in the company, the load of work raises becouse the number of work and average of hours, also, the salary does not raise, so the employees have to do much more work for the same amount of money, finally he decides to leave the company.
# 
# More graphics:

# In[ ]:


g = sns.FacetGrid(df, col='left')
g.map(plt.hist, 'time_spend_company')


# Most of employees are in between theirs 2 and 4 years in the company

# In[ ]:


grid = sns.FacetGrid(df, col='left', row='time_spend_company', size=2.2, aspect=1.6)
grid.map(plt.hist, 'average_montly_hours', alpha=.5)
grid.add_legend();


# Another interesting thing:
# We can also see that there is a strong desertion over the thrid year of work.
# 
# **Now, lets use some calssifiation methods.**
# 
# 
# 

# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# **70/30 split data** for trainning and testing
# Based in the previus study, we select 4 predictor variables,  "satisfaction_level","number_project","average_montly_hours","time_spend_company", these variables, has a higth correlation with "left", this is the which we want to predict. 

# In[ ]:


X = df.drop('left', axis=1)
labels = df['left']
X_train, X_test, labels_train, labels_test = train_test_split(X, labels, random_state=1, test_size = 0.3)
predictors = ["satisfaction_level","number_project","average_montly_hours","time_spend_company"]


# **Decision Tree**

# In[ ]:


model = DecisionTreeClassifier()
model.fit(X_train[predictors], labels_train)
labels_predict = model.predict(X_test[predictors])
accuracy_score(labels_test, labels_predict)


# Global Prediction of 97%, not bad. Let see the Confusion Matrix

# In[ ]:


pd.DataFrame(
    confusion_matrix(labels_test, labels_predict),
    columns=['Predicted Not Left', 'Predicted Left'],
    index=['True No Left', 'True Left']
)


# This model is good predicting negatives and positives cases.
# 
# **Random Forest**

# In[ ]:


model = RandomForestClassifier()
model.fit(X_train[predictors], labels_train)
labels_predict = model.predict(X_test[predictors])
accuracy_score(labels_test, labels_predict)


# A little better than previus model.
# 
# But what about precision classes? let see

# In[ ]:


pd.DataFrame(
    confusion_matrix(labels_test, labels_predict),
    columns=['Predicted Not Left', 'Predicted Left'],
    index=['True No Left', 'True Left']
)


# Much better for negative cases, but, we need to predict who will left the company, so I prefer using Clasification Tree so far.
# 
# Lets see mre important variables according to Random Forest:

# In[ ]:


pd.Series(model.feature_importances_, index=predictors).sort_values(ascending=False)


# **K Near Neighbors**

# In[ ]:


model = KNeighborsClassifier()
model.fit(X_train[predictors], labels_train)
labels_predict = model.predict(X_test[predictors])
accuracy_score(labels_test, labels_predict)


# In[ ]:


pd.DataFrame(
    confusion_matrix(labels_test, labels_predict),
    columns=['Predicted Not Left', 'Predicted Left'],
    index=['True No Left', 'True Left']
)


# It's not the best model, we drop it.

# **Logistic Regression**

# In[ ]:


model = LogisticRegression()
model.fit(X_train[predictors], labels_train)
labels_predict = model.predict(X_test[predictors])
accuracy_score(labels_test, labels_predict)


# Very low global score. We also drop it.

# In[ ]:


model = SVC(kernel='linear')
model.fit(X_train[predictors], labels_train)
labels_predict = model.predict(X_test[predictors])
accuracy_score(labels_test, labels_predict)


# - I decided Decision Tree as the best method of clasiffication according to the tests, this is the best mothods for predict who empoyeers will leave the company. 
# - The three more important variables for prediction are:  
#     - satisfaction_level 
#     - time_spend_company 
#     - average_montly_hours
