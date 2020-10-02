#!/usr/bin/env python
# coding: utf-8

# Reference: https://www.kaggle.com/hidede/graduate-admissions/notebook

# Features in the dataset:
# 
# GRE Scores (290 to 340)
# 
# TOEFL Scores (92 to 120)
# 
# University Rating (1 to 5)
# 
# Statement of Purpose (1 to 5)
# 
# Letter of Recommendation Strength (1 to 5)
# 
# Undergraduate CGPA (6.8 to 9.92)
# 
# Research Experience (0 or 1)
# 
# Chance of Admit (0.34 to 0.97)

# In[47]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import os
print(os.listdir("../input"))


# In[48]:


df = pd.read_csv("../input/Admission_Predict.csv", sep=",")


# In[49]:


print("there are {} columns in the dataset.".format(len(df.columns)))
print(df.columns)
print("there are {} samples in the dataset.".format(df.shape[0]))


# In[50]:


df = df.rename(columns={'Chance of Admit ':'Chance of Admit'})
print(df.columns)


# In[51]:


df.info()


# In[52]:


df.head(5)


# In[53]:


fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), ax=ax, annot=True)
plt.show()


# from correlation matrix, we can analyse that which feature strongly influnce to Chance of Admit.
# 
# most influncer feature: CGPA
# 
# least influncer feature: Research

# **Data Visualization**

# Having Research or not:
# 
# The majority of the candidates in the dataset have research experience.
# 
# Therefore, the Research will be a unimportant feature for the Chance of Admit. The correlation between Chance of Admit and Research was already lower than other correlation values.

# In[54]:


print("Not having research : ", len(df[df.Research == 0]))
print("having research : ", len(df[df.Research == 1]))


# In[55]:


x = ["Not having research", "Having research"]
y = np.array( [len(df[df.Research==0]), len(df[df.Research==1])] )
plt.bar(x,y)
plt.xlabel("candidate type")
plt.ylabel("number of candidate")
plt.show()


# lets get some idea about TOEFL and GRE scores

# In[56]:


df["TOEFL Score"].plot(kind='hist', bins=200, figsize=(4,4))
plt.show()


# In[57]:


df['GRE Score'].plot(kind='hist', bins=200, figsize=(4,4))
plt.show()


# In[58]:


# university ratings VS cgpa analysis

plt.scatter(df["University Rating"], df['CGPA'])
plt.show()


# In[59]:


# cgpa VS gre score

plt.scatter(df["GRE Score"], df["CGPA"])
plt.show()


# In[60]:


# cgpa VS toefl score

plt.scatter(df["TOEFL Score"], df["CGPA"])
plt.show()


# In[61]:


# gre score VS toefl score

plt.scatter(df["TOEFL Score"], df["GRE Score"])
plt.show()


# from above three plots, we can analyse that guys having high gre score and toefl score have high cgpa and vice versa.

# **REGRESSION ALGORITHMS (SUPERVISED MACHINE LEARNING ALGORITHMS)**

# In[62]:


# preparing dataset for regression
# the column "Serial No." is of no use, so we will delete it for now

df = pd.read_csv("../input/Admission_Predict.csv", sep=',')

# save serial no. for future use (if in case)
serialNo = df["Serial No."].values

df.drop(['Serial No.'], axis=1, inplace=True)

df = df.rename(columns={"Chance of Admit ": "Chance of Admit"})


# In[63]:


df.columns


# In[64]:


y = df["Chance of Admit"].values
x = df.drop(["Chance of Admit"], axis=1)


# In[75]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# In[76]:


import warnings
warnings.filterwarnings("ignore")


# In[88]:


# normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train, x_test = sc.fit_transform(x_train), sc.fit_transform(x_test)


# In[90]:


# Linear Regression

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

from sklearn.metrics import r2_score

print("r2 score of test set is : ", r2_score(y_test, y_pred_lr))
print("real value of y_test[0] is : " + str(y_test[0]) + " and predicted is : " + str(lr.predict(x_test[[0],:])))
print("real value of y_test[1] is : " + str(y_test[1]) + " and predicted is : " + str(lr.predict(x_test[[1],:])))


# In[91]:


# Random Forest Regression

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

print("r2 score of test set is : ", r2_score(y_test, y_pred_rf))
print("real value of y_test[0] is : " + str(y_test[0]) + " and predicted is : " + str(rf.predict(x_test[[0],:])))
print("real value of y_test[1] is : " + str(y_test[1]) + " and predicted is : " + str(rf.predict(x_test[[1],:])))


# In[92]:


# Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()

dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)

print("r2 score of test set is : ", r2_score(y_test, y_pred_dt))
print("real value of y_test[0] is : " + str(y_test[0]) + " and predicted is : " + str(dt.predict(x_test[[0],:])))
print("real value of y_test[1] is : " + str(y_test[1]) + " and predicted is : " + str(dt.predict(x_test[[1],:])))


# **Comparision of Regression Algorithms**

# In[93]:


x = ["LinearReg", "RandomForestReg", "DecisionTreeReg"]
y = np.array([ r2_score(y_test,y_pred_lr), r2_score(y_test,y_pred_rf), r2_score(y_test,y_pred_dt) ])

plt.bar(x,y)
plt.xlabel("type of regression algorithm")
plt.ylabel("r2 score")
plt.title("comparision of regression algorithms")
plt.show()


# In[98]:


# plot of comparision of predicted output from all 3 regression algorithms with the actual value for the indexes 0,10,20,30,40 etc.

red = plt.scatter(np.arange(0,80,10), y_pred_lr[0:80:10], color='red')
blue = plt.scatter(np.arange(0,80,10), y_pred_rf[0:80:10], color='blue')
black = plt.scatter(np.arange(0,80,10), y_pred_dt[0:80:10], color='black')
green = plt.scatter(np.arange(0,80,10), y_test[0:80:10], color='green')

plt.xlabel("index of candidate")
plt.ylabel("chances of admit")
plt.title("comparision of chances of admit from all 3 regression algorithms with actual value of chances")
plt.legend((red,blue,black,green), ("LR","RF","DT","Actual"))
plt.show()


# In[ ]:




