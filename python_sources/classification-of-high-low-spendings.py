#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings


# In[ ]:


#Disabling warnings
warnings.simplefilter("ignore")


# In[ ]:


#Importing data
data=pd.read_csv('../input/Mall_Customers.csv')


# In[ ]:


#Peeking at data
data.head(10)


# In[ ]:


#Data description
data.describe()


# In[ ]:


#Renaming columns names
data.rename(columns={'Annual Income (k$)': 'AIncome', 'Spending Score (1-100)': 'SpendingScore'}, inplace=True)


# In[ ]:


#Plotting count of customers with similar Spending Scores
sns.set(context='notebook', style='whitegrid')
pl.figure(figsize =(20,20))
data.groupby(['SpendingScore']).CustomerID.count().plot(kind='barh')
pl.ylabel('Spending Score (1-100)', fontsize=15)
pl.xlabel('Total Count of Customers', fontsize=15)
pl.title('Distribution of Scores', fontsize=15)
plt.show()


# In[ ]:


#Plotting count of customers with similar Annual incomes
pl.figure(figsize =(20,20))
data.groupby(['AIncome']).CustomerID.count().plot(kind='barh')
pl.ylabel('Annual Income (k$)', fontsize=15)
pl.xlabel('Total Count of Customers', fontsize=15)
pl.title('Distribution of Annual Income', fontsize=15)
plt.show()


# In[ ]:


#Plotting Spending Scores distribution of females & males
pl.figure(figsize =(10,5))
sns.boxplot(x="SpendingScore", y="Gender", data=data, whis="range", palette="vlag")
sns.swarmplot(x="SpendingScore", y="Gender", data=data, linewidth=0)
plt.title('Spending Scores Distribution in Females & Males', fontsize=15)


# In[ ]:


#Plotting Annual income distribution of males & females
pl.figure(figsize =(10,5))
sns.boxplot(x="AIncome", y="Gender", data=data, whis="range", palette="vlag")
sns.swarmplot(x="AIncome", y="Gender", data=data, linewidth=0)
plt.title('Annual Income Distribution in Males & Females', fontsize=15)


# In[ ]:


#Plotting Annual income and Spending Score for males
mAI=data[data['Gender']=='Male'].groupby('Age').AIncome
mSS=data[data['Gender']=='Male'].groupby('Age').SpendingScore
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Annual Income & Spending Score of Males', fontsize=15)
plt.scatter(mAI.head().values, mSS.head().values)
plt.show()


# In[ ]:


#Plotting Annual income and Spending Score for females
fAI=data[data['Gender']=='Female'].groupby('Age').AIncome
fSS=data[data['Gender']=='Female'].groupby('Age').SpendingScore
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Annual Income & Spending Score of Females', fontsize=15)
plt.scatter(fAI.head().values, fSS.head().values)
plt.show()


# **Annual Income and Spending Score are not correlated!!**

# In[ ]:


#Adding labels to data where the Spending score is greater/Less than Spending Score mean value
#1: High Spendings
#0: Low Spendings
MeanSS=data['SpendingScore'].mean()
data['Labels'] = np.where(data['SpendingScore']>MeanSS, 1, 0)


# In[ ]:


#Pairplot of parameters
pl.figure(figsize =(20,20))
sns.pairplot(data, kind="reg")
plt.show()


# In[ ]:


#Correlation matrix & Heatmap - Finding correlation
pl.figure(figsize =(10,10))
corrmat = data.corr()
sns.heatmap(corrmat, annot=True, fmt='.1f', vmin=0, vmax=1, square=True);
plt.show()


# In[ ]:


#Labels and featureSet columns
columns = data.columns.tolist()
columns = [c for c in columns if c not in ['Labels','Gender']]
target = 'Labels'

X = data[columns]
y = data[target]


# In[ ]:


#Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

print("Training FeatureSet:", X_train.shape)
print("Training Labels:", y_train.shape)
print("Testing FeatureSet:", X_test.shape)
print("Testing Labels:", y_test.shape)


# In[ ]:


#Using random forrest Model
#Initializing the model with some parameters.
model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)
#Fitting the model to the data.
model.fit(X_train, y_train)
#Generating predictions for the test set.
predictions = model.predict(X_test)
#Computing the Model Accuracy
print("Random Forrest Accuracy:",round((metrics.accuracy_score(y_test, predictions))*100,2))
#Computing the error.
print("Mean Absoulte Error:",round((mean_absolute_error(predictions, y_test))*100,2))
#Computing classification Report
print("Classification Report:\n", classification_report(y_test, predictions))
#Plotting confusion matrix
print("Confusion Matrix:")
df = pd.DataFrame(
    confusion_matrix(y_test, predictions),
    index = [['actual', 'actual'], ['0','1']],
    columns = [['predicted', 'predicted'], ['0', '1']])
print(df)

