#!/usr/bin/env python
# coding: utf-8

# In[67]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[68]:


df = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")


# In[69]:


df.head()


# In[70]:


df.shape


# In[71]:


df.Attrition.value_counts()


# In[72]:


df["BusinessTravel"].value_counts()


# In[73]:


df.columns


# In[74]:


df.drop(columns=["Department","EducationField","Education","EmployeeCount","BusinessTravel",                 "HourlyRate","Over18","MonthlyIncome","OverTime","StockOptionLevel","StandardHours"], inplace = True)


# In[75]:


plt.hist(df.Age)


# In[76]:


df.head()


# In[77]:


after_df = df.groupby(["Attrition","Gender","MaritalStatus"]).mean().reset_index().drop(columns = ["EmployeeNumber"])


# In[78]:


after_df.sample(5)


# In[79]:


sns.barplot(x="Attrition", y = "WorkLifeBalance", hue = "Gender", data = after_df)


# In[80]:


def plotmatrix(start,end):
    fig, axs = plt.subplots(nrows = 2, ncols=2)
    i = 0
    cols = after_df.columns[start:end]
    fig.set_size_inches(14, 10)
    for indi in range(2):
        for indj in range(2):
                sns.barplot(x="Attrition",y=str(cols[i]),data = after_df,ax = axs[indi][indj],hue = "MaritalStatus")                .set_title("affect of "+str(cols[i]))
                print("column : "+str(cols[i]))
                i+=1


# In[81]:


plotmatrix(3,7)


# From the above figures, we can conclude that, 
# 1. Average Age of employees those who left the IBM is less compared to others.
# 2. Average Daily Rate of employees those who left the IBM is less compared to others.
# 3. Average Distance from Home of employees those who left the IBM is more compared to others.
# 4. Average Environment Satisfaction of employees those who left the IBM is less compared to others

# In[82]:


plotmatrix(8,12)


# From the above figures, we can conclude that, 
# 1. Average Job Level of employees those who left the IBM is less compared to others.
# 2. Average Job Satisfaction of employees those who left the IBM is less compared to others.
# 3. seems like Monthly rate doesn't affect much for the attrition rate
# 4. Average No.of Companies worked by the employees those who left the IBM is more compared to others

# In[83]:


plotmatrix(13,17)


# From the above figures, we can conclude that, 
# 1. Seems like only Total Working Years is affecting the attrition rate.

# In[84]:


plotmatrix(18,22)


# From the above figures, we can conclude that, 
# 1. Average years at IBM of those who left the IBM is less compared to others.
# 2. Average years in current Role at IBM of employees those who left the IBM is less compared to others.
# 3. Average years since LastPromotion at IBM of employees those who left the IBM is less compared to others.
# 4. Average years with Current Manager at IBM of employees those who left the IBM is less compared to others

# (we have to build the model based on the selected features from the above analysis)(under editing*)

# In[85]:


df["Gender"] = df.Gender.map({"Female" : 0, "Male" : 1})
df["MaritalStatus"] = df.MaritalStatus.map({"Divorced":0,"Married":1,"Single":2})


# In[86]:


final_df = df[after_df.drop(columns=["MonthlyRate","PerformanceRating"]).columns]
final_df.head()
final_X = final_df.drop(columns=["Attrition"])
final_Y = final_df["Attrition"]


# 

# In[96]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(final_X, final_Y,test_size = 0.1)


# In[104]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
model = LogisticRegression()
model.fit(train_x,train_y)
target = model.predict(test_x)
print("accuracy score : ",accuracy_score(target, test_y))


# In[106]:


from xgboost import XGBClassifier
for i in range(2,7):
    model = XGBClassifier(max_depth=i)
    model.fit(train_x,train_y)
    target = model.predict(test_x)
    print("accuracy score : ",accuracy_score(target, test_y))
print(confusion_matrix(test_y,target))


# In[107]:





# In[110]:


pd.Series(test_y).value_counts()


# In[111]:


pd.Series(target).value_counts()


# So, If we observe closely, Accuracy is good. But in reality, Precision, Sensitivity, recall is less. 

# So, I would like to know how to improve these type of models?
