#!/usr/bin/env python
# coding: utf-8

# Predicting Satisfaction level of the employee.
# ---------------------------------------------

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/HR_comma_sep.csv')
df.describe()


# In[ ]:


df.isnull().any()


# In[ ]:



#The meanings of all the columns are quite understandable except 'sales' which actually represents 
#the department of the employee
df=df.rename(columns={'sales':'job'})


# In[ ]:


sns.heatmap(df.corr(), vmax=.8, square=True,annot=True,fmt='.2f')


# In[ ]:


#satisfaction level comes out as the most correlated feature

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder 


le=LabelEncoder()
df['job']= le.fit_transform(df['job'])
df['salary']= le.fit_transform(df['salary'])


# In[ ]:


X= np.array(df.drop('left',1))
y=np.array(df['left'])

model= ExtraTreesClassifier()
model.fit(X,y)

feature_list= list(df.drop('left',1).columns)

feature_importance_dict= dict(zip(feature_list,model.feature_importances_))

print(sorted(feature_importance_dict.items(), key=lambda x: x[1],reverse=True))


# Since 'satisfaction level' turns out to be the most important metric for the target variable 'left' lets predict satisfaction level of the employee.
# ------------------------------------------------------------------------

# In[ ]:


sns.barplot(df['left'],df['satisfaction_level'])
#More the satisfaction level lesser the chance of an employee to leave.


# In[ ]:


facet = sns.FacetGrid(df, hue="left",aspect=3)
facet.map(sns.kdeplot,'satisfaction_level',shade= True)
facet.set(xlim=(0, 1))
facet.add_legend()

#3 peaks for left=1 indicates 3 types of trends for an employee to leave.


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor

model= ExtraTreesRegressor()

X=df.drop(['left','satisfaction_level'],axis=1)
y=df['satisfaction_level']
model.fit(X,y)

feature_list= list(df.drop(['left','satisfaction_level'],1).columns)

feature_importance_dict= dict(zip(feature_list,model.feature_importances_))

print(sorted(feature_importance_dict.items(), key=lambda x: x[1],reverse=True))


# **Let's analyse the 3 most important features: number_project,average_monthly_hours and last_evaluation***

# In[ ]:


#sns.swarmplot(x=[df['average_montly_hours'],df['last_evaluation']], y=df['satisfaction_level'])
plt.scatter(df['satisfaction_level'],df['average_montly_hours'])
plt.ylabel('average_montly_hours')
plt.xlabel('satisfaction_level')


# In[ ]:


plt.scatter(df['satisfaction_level'],df['last_evaluation'])
plt.xlabel('satisfaction_level')
plt.ylabel('last_evaluation')


# In[ ]:


sns.pointplot(df['number_project'],df['satisfaction_level'])


# In[ ]:


projects=df['number_project'].unique()
projects=sorted(projects)
for i in projects:
    mean_satisfaction_level=df['satisfaction_level'][df['number_project']==i].mean()
    print('project_total',i,':',mean_satisfaction_level)

#Expected reuslt


# In[ ]:


df1=df.copy()

group_name=list(range(20))
df1['last_evaluation']=pd.cut(df1['last_evaluation'],20,labels=group_name)
df1['average_montly_hours']=pd.cut(df1['average_montly_hours'],20,labels=group_name)

#average_monthly_hours bins:
"""
{0: '(149.5, 160.2]', 1: '(256.5, 267.2]', 2: '(267.2, 277.9]', 3: '(213.7, 224.4]', 4: '(245.8, 256.5]', 5: '(138.8, 149.5]',
 6: '(128.1, 138.8]', 7: '(299.3, 310]', 8: '(224.4, 235.1]', 9: '(277.9, 288.6]', 10: '(235.1, 245.8]'
 , 11: '(117.4, 128.1]', 12: '(288.6, 299.3]', 13: '(181.6, 192.3]', 14: '(160.2, 170.9]',
 15: '(170.9, 181.6]', 16: '(192.3, 203]', 17: '(203, 213.7]', 18: '(106.7, 117.4]',
 19: '(95.786, 106.7]'}
 """


# In[ ]:


sns.pointplot(df1['last_evaluation'],df1['satisfaction_level'])


# In[ ]:


#3 types of employees: 
#last_evaluation(0-3): satisfaction level is pretty low--> possibly not able to perform well 
#last_evaluation(7-12): satisfaction level is high---> possibly getting appreciated for their work 
#last_evaluation(13-18):satisfaction level is low again---> Not getting enough appreciated for their work**


# In[ ]:


sns.pointplot(df1['average_montly_hours'],df1['satisfaction_level'])


# In[ ]:


sns.pointplot(df['number_project'],df['last_evaluation'])
#As the number of projects increase last_evaluation score also increases.


# In[ ]:


sns.pointplot(df1['last_evaluation'],df['average_montly_hours'])

#more the hours you work higher is your last_evaluation score.


# In[ ]:


#Let's check some other features also.


# In[ ]:


sns.barplot(df['Work_accident'],df['satisfaction_level'])


# In[ ]:


sns.barplot(df['salary'],df['satisfaction_level'])


# In[ ]:


sns.barplot(df['job'],df['satisfaction_level'])


# Conclusion:
# Last_evaluation: 
# Better evaluation scores if you work for more hours and on more number of projects.
# Satisfaction_Level:
# There should be a balance in number of hours and number of projects that a employee is working on.
# **number_of_projects should lie between 3 to 5.**
# 
# **average_monthly_hours b/w  5 to 16 i.e (138.8,203] hours.**  
# 
# 
# 

# In[ ]:


from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


X=df.drop(['left','satisfaction_level'],axis=1)
y=df['satisfaction_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=7)

kfold=KFold(n_splits=10,random_state=7)
models=[['LR',LinearRegression()],['CART',DecisionTreeRegressor()],['RF',RandomForestRegressor()]]
scoring='neg_mean_squared_error'
result_list=[]
for names,model in models:
    results= cross_val_score(model, X,y, cv=kfold,scoring=scoring)
    print(names,results.mean())

    
#RandomForest performs the best. 
    
    


# In[ ]:


#Let's take a small example

test_dict={'last_evaluation':[0.2,0.6,0.7,0.8],'number_project':[1,3,4,6],'average_montly_hours':[110,180,190,250],
           'time_spend_company':[3,4,5,6],'Work_accident':[0,1,1,0],'promotion_last_5years':[0,0,1,1],'job':[0,1,2,3],
           'salary':[0,1,1,0]}

#1st employee is the stuggling one.
#2nd and 3rd fullfill all the required criterias for satisfaction level to be high.
#4th employee is a high performer.

df_test= pd.DataFrame(test_dict)

test_X= np.array(df_test)

model= RandomForestRegressor()
model.fit(X_train,y_train)
model.predict(test_X)

