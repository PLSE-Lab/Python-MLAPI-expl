#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #Gets rid of warnings
df = pd.read_csv('../input/BlackFriday.csv')
df.shape


# In[ ]:


df.head() #Shows the head of the data 


# In[ ]:


df.tail() #Shows the last part of the data


# In[ ]:


df.describe() 


# In[ ]:


df.info() 


# In[ ]:


sns.countplot('Gender',data=df); #Count the number of females and males


# In[ ]:


sns.barplot(x='Gender',y='Purchase',data=df); #Generates a barplot with Gender against Purchase


# In[ ]:


sns.boxplot(x="Gender", y="Purchase", data=df); 
#Generates a boxplot of purchases for both males and females


# In[ ]:


sns.barplot(x='Age',y='Purchase',data=df); #Generates a barplot with Age against Purchase


# In[ ]:


sns.boxplot(x="Age", y="Purchase", data=df);#Genetates a boxplot of purchase for each age group


# In[ ]:


#count the number of females and males for each age group
splitAsGender = []   
for i in ["M","F"]: 
    splitAsGender.append(df[df.Gender == i])
    tmp = [i["Age"] for i in splitAsGender]

plt.xlabel("Age")

plt.hist(tmp, histtype="barstacked", bins=10, rwidth=0.5, color=['red', 'blue'], label=['female', 'male'])
plt.legend() 
plt.show()


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(12,6)) #counts the number of City_Category for each Stay_In_Current_City_Years
sns.countplot(df['City_Category'],hue=df['Stay_In_Current_City_Years']);


# In[ ]:


sns.barplot(x='City_Category',y='Purchase',data=df); #Generates a barplot with City_Category against Purchase


# In[ ]:


sns.boxplot(x="City_Category", y="Purchase", data=df); #Genetates a boxplot of purchase for each City_Category


# In[ ]:


sns.barplot(x='Stay_In_Current_City_Years',y='Purchase',data=df); #Generates a barplot with Stay_In_Current_City_Years against Purchase


# In[ ]:


sns.boxplot(x="Stay_In_Current_City_Years", y="Purchase", data=df); #Genetates a boxplot of purchase for each Stay_In_Current_City_Years


# In[ ]:


sns.barplot(x='Occupation',y='Purchase',data=df); #Genarates a barplot with Occupation against Purchase


# In[ ]:


sns.barplot(x='Product_Category_1',y='Purchase',data=df); #Generates a barplot with Product_Category_1 against Purchase


# In[ ]:


sns.barplot(x='Product_Category_2',y='Purchase',data=df);#Generates a barplot with Product_Category_2 against Purchase


# In[ ]:


sns.barplot(x='Product_Category_3',y='Purchase',data=df); #Genarates a barplot with Producu_3 against Purchase


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(12,6)) 
sns.countplot(df['Product_Category_1'],hue=df['Gender']);


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(12,6)) 
sns.countplot(df['Product_Category_1'],hue=df['Age']);


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(12,6)) 
sns.countplot(df['Product_Category_2'],hue=df['Gender']);


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(12,6)) 
sns.countplot(df['Product_Category_2'],hue=df['Age']);


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(12,6)) 
sns.countplot(df['Product_Category_3'],hue=df['Gender']);


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(12,6)) 
sns.countplot(df['Product_Category_3'],hue=df['Age']);


# In[ ]:


dfg=df.groupby('User_ID')['Purchase'].sum()
dfg.head()


# In[ ]:


df_temp=df.groupby('User_ID')['Purchase'].sum().reset_index()


# In[ ]:


df_temp


# In[ ]:


plt.hist(df_temp)


# In[ ]:


dfg.tail()


# In[ ]:


dfg.describe()


# In[ ]:





# In[ ]:


dfh = df.drop(['Product_ID','User_ID','Product_Category_1','Product_Category_2','Product_Category_3'], axis=1)
dfh = pd.get_dummies(dfh)
dfh.head()


# In[ ]:


train_X = dfh.drop('Purchase',axis=1)
train_X.head()


# In[ ]:


train_Y=dfh['Purchase']
train_Y.head()


# In[ ]:


from sklearn.linear_model import LinearRegression as LR
model = LR()
model.fit(train_X,train_Y)


# In[ ]:


from sklearn.model_selection import train_test_split
(train_X, test_X ,train_Y, test_Y) = train_test_split(train_X, train_Y, random_state = 100)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(train_X, train_Y)
pred = clf.predict(test_X)


# In[ ]:


from sklearn.metrics import (roc_curve, auc, accuracy_score)

pred = clf.predict(test_X)
fpr, tpr, thresholds = roc_curve(test_Y, pred, pos_label=1)
auc(fpr, tpr)
accuracy_score(pred, test_Y)


# In[ ]:


model.coef_


# In[ ]:


model.intercept_

