#!/usr/bin/env python
# coding: utf-8

# # Campus placements prediction

# Importing all the necessary modules

# In[ ]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error


# In[ ]:


df=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()


# Checking the number of null values in the data

# In[ ]:


df.isna().sum()


# Dropping serial no. column and the salary column due to the presence of many null values

# In[ ]:


df.drop('salary',axis=1,inplace=True)
df.drop('sl_no',axis=1,inplace=True)


# In[ ]:


df.head()


# # Visualizing the data

# Creating a pairplot with respect to placement status.

# In[ ]:


sns.pairplot(data=df,hue='status',palette='Set1')


# In[ ]:


sns.countplot(data=df,x='workex',hue='status',palette='Set2')


# In[ ]:


sns.countplot(data=df,x='gender',hue='status',palette='Set2')


# In[ ]:


sns.countplot(data=df,x='specialisation',hue='status',palette='Set2')


# In[ ]:


sns.countplot(data=df,x='degree_t',hue='status',palette='Set2')


# Plotting various exam percentages

# In[ ]:


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.distplot(df['ssc_p'],bins=50,)

plt.subplot(2,2,2)
sns.distplot(df['hsc_p'],bins=50,color='red')

plt.subplot(2,2,3)
sns.distplot(df['mba_p'],bins=50,color='green')

plt.subplot(2,2,4)
sns.distplot(df['etest_p'],bins=50,color='orange')


# A heatmap to check the correlation among various percentages.

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True,linewidth=0.2)


# Dropping 3 more columns that don't affect the classification

# In[ ]:


df.drop('hsc_b',axis=1,inplace=True)
df.drop('ssc_b',axis=1,inplace=True)
df.drop('hsc_s',axis=1,inplace=True)


# Changing values of different columns from strings or characters to numbers.

# In[ ]:


d1=pd.get_dummies(df['gender'],drop_first=True)
d2=pd.get_dummies(df['degree_t'],drop_first=True)
d3=pd.get_dummies(df['specialisation'],drop_first=True)
d4=pd.get_dummies(df['workex'],drop_first=True)
df=pd.concat([df,d1,d2,d3,d4],axis=1)
df.drop(['gender','workex','degree_t','specialisation'],axis=1,inplace=True)
labenc=LabelEncoder()
df['status']=labenc.fit_transform(df['status'])


# In[ ]:


df.head()


# # Splitting training and testing data

# In[ ]:


X=df.drop('status',axis=1)
y=df['status']


# In[ ]:


Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=1)


# # Scaling the data

# In[ ]:


sc=StandardScaler()
Xtrain=sc.fit_transform(Xtrain)
Xtest=sc.transform(Xtest)


# # Using logistic regression

# In[ ]:


logreg=LogisticRegression()
logreg.fit(Xtrain,ytrain)
ypred=logreg.predict(Xtest)
print('Accuracy is: {}%'.format(round(accuracy_score(ytest,ypred)*100,2)))


# Difference in predicted and actual y value-plot.

# In[ ]:


sns.distplot(ytest,bins=10,color='blue')
sns.distplot(ypred,bins=10,color='red')


# # Confusion matrix and classification report

# In[ ]:


print(confusion_matrix(ytest,ypred))


# In[ ]:


print(classification_report(ytest,ypred))


# In[ ]:


print(np.sqrt(mean_squared_error(ytest,ypred)))


# Thus our model has classified 12+27 values correctly and 3+1 values wrongly with an RMSE score of 0.304
# 
