#!/usr/bin/env python
# coding: utf-8

# **Please UPVOTE this kernel if it helps**

# Lets us load the dataset

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../input/creditcard.csv')
df.head()


# Lets check the datatypes of all the columns and see if there are any discrepancies

# In[ ]:


df.dtypes


# Simple Description of all the columns

# In[ ]:


df.describe()


# Now lets check for null values

# In[ ]:


df.isnull().sum()


# Finding correlation of different variables with the target variable

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corrwith(df['Class']).reset_index()
corr.columns = ['Index','Correlations']
corr = corr.set_index('Index')
corr = corr.sort_values(by=['Correlations'], ascending = False)
plt.figure(figsize=(4,15))
fig = sns.heatmap(corr, annot=True, fmt="g", cmap='YlGnBu')
plt.title("Correlation of Variables with Class")
plt.show()


# Distribution of the Time Variable

# In[ ]:


plt.figure(figsize=(15,4))
fig = sns.distplot(df['Time'], kde=False, color="green")
plt.show()


# Distribution of Amount

# In[ ]:


plt.figure(figsize=(8,4))
fig = sns.violinplot(x=df["Amount"], color="lightblue")
plt.show()


# Let us try plotting the amount and time and see if we can differentiate the fraud cases using just these two

# In[ ]:


plt.figure(figsize=(8,4))
fig = plt.scatter(x=df[df['Class'] == 1]['Time'], y=df[df['Class'] == 1]['Amount'], color="c")
plt.title("Time vs Transaction Amount in Fraud Cases")
plt.show()


# In[ ]:


plt.figure(figsize=(8,4))
fig = plt.scatter(x=df[df['Class'] == 0]['Time'], y=df[df['Class'] == 0]['Amount'], color="dodgerblue")
plt.title("Time vs Transaction Amount in Legit Cases")
plt.show()


# In[ ]:


df.hist(figsize=(20,20), color = "salmon")
plt.show()


# Distribution of Class

# In[ ]:


plt.figure(figsize=(7,5))
fig = sns.countplot(x="Class", data=df)
plt.show()


# In[ ]:


from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


# Now let us separate the legit and fraud cases so we can train the Isolation Forest using the legit cases

# In[ ]:


inliers = df[df.Class==0]
inliers = inliers.drop(['Class'], axis=1)
outliers = df[df.Class==1]
outliers = outliers.drop(['Class'], axis=1)
inliers_train, inliers_test = train_test_split(inliers, test_size=0.30, random_state=42)


# In[ ]:


inliers_train.head()


# In[ ]:


inliers_test.head()


# In[ ]:


model = IsolationForest()
model.fit(inliers_train)
inlier_pred_test = model.predict(inliers_test)
outlier_pred = model.predict(outliers)


# In[ ]:


print("Accuracy in Detecting Legit Cases:", list(inlier_pred_test).count(1)/inlier_pred_test.shape[0])
print("Accuracy in Detecting Fraud Cases:", list(outlier_pred).count(-1)/outlier_pred.shape[0])


# In[ ]:


from sklearn.neighbors import LocalOutlierFactor

model = LocalOutlierFactor(novelty=True)
model.fit(inliers_train)
inlier_pred_test = model.predict(inliers_test)
outlier_pred = model.predict(outliers)


# In[ ]:


print("Accuracy in Detecting Legit Cases:", list(inlier_pred_test).count(1)/inlier_pred_test.shape[0])
print("Accuracy in Detecting Fraud Cases:", list(outlier_pred).count(-1)/outlier_pred.shape[0])


# As we can see clearly Isolation Forest performs much better than LocalOutlierFactor

# **Please UPVOTE this kernel if you like it**
