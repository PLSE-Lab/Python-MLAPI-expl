#!/usr/bin/env python
# coding: utf-8

# # Prediction of quality of red wine

# Import all the necessary modules for plotting, scaling and classification

# In[ ]:


from sklearn import svm
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error


# In[ ]:


df=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()


# Check for the presence of null values

# In[ ]:


df.isna().sum()


# # Data visualization

# Since there are no null values, next step will be data visualization and plotting.
# Plotting the various features of the dataset as distplots using seaborn shows the range of values that our features lie in and the distribution of our features over the range. This is just a basic visualization of the features in the dataset to asses their range.

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.distplot(df['alcohol'],bins=50)
plt.subplot(2,2,2)
sns.distplot(df['pH'],bins=50,color='red')
plt.subplot(2,2,3)
sns.distplot(df['sulphates'],bins=50,color='green')
plt.subplot(2,2,4)
sns.distplot(df['total sulfur dioxide'],bins=50,color='purple')


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.distplot(df['fixed acidity'],bins=50)
plt.subplot(2,2,2)
sns.distplot(df['volatile acidity'],bins=50,color='red')
plt.subplot(2,2,3)
sns.distplot(df['citric acid'],bins=50,color='green')
plt.subplot(2,2,4)
sns.distplot(df['residual sugar'],bins=50,color='purple')


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.distplot(df['chlorides'],bins=50)
plt.subplot(2,2,2)
sns.distplot(df['free sulfur dioxide'],bins=50,color='red')
plt.subplot(2,2,3)
sns.distplot(df['density'],bins=50,color='green')


# The heatmap from seaborn gives the relation between the different features. The negative values show that there is almost no correlation between the 2 respective variables whereas the opposite is true for positive values that appear in the grid.

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True,linewidth=0.2)


# Now, visualizing the values of the quality of red wine for different features' values as given in the dataset using seaborn's boxplot. It shows us for which values of various features, the quality is the highest.

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.boxplot(df['quality'],df['alcohol'])
plt.subplot(2,2,2)
sns.boxplot(df['quality'],df['sulphates'])
plt.subplot(2,2,3)
sns.boxplot(df['quality'],df['pH'])
plt.subplot(2,2,4)
sns.boxplot(df['quality'],df['density'])


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.boxplot(df['quality'],df['fixed acidity'])
plt.subplot(2,2,2)
sns.boxplot(df['quality'],df['volatile acidity'])
plt.subplot(2,2,3)
sns.boxplot(df['quality'],df['residual sugar'])
plt.subplot(2,2,4)
sns.boxplot(df['quality'],df['citric acid'])


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.boxplot(df['quality'],df['free sulfur dioxide'])
plt.subplot(2,2,2)
sns.boxplot(df['quality'],df['total sulfur dioxide'])


# Checking the minimum and maximum value of quality.

# In[ ]:


df['quality'].min()


# In[ ]:


df['quality'].max()


# In[ ]:


values = (2, 6, 9)
qual = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = values, labels = qual)
df.head()


# Now, we have 2 groups i.e. good quality and bad quality wine. This helps us with label encoding to classify data better.

# In[ ]:


df['quality'].value_counts()


# In[ ]:


le=LabelEncoder()
df['quality']=le.fit_transform(df['quality'])


# # Splitting training and testing data

# In[ ]:


X=df.drop('quality',axis=1)
y=df['quality']


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)


# Scaling the data and normalizing it to a particular range of values. A pre processing step for faster calculations.

# In[ ]:


sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.fit_transform(xtest)


# # Classification models:

# SVM  (Support Vector Machine)

# In[ ]:


model=svm.SVC()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
print(accuracy_score(ytest,ypred))


# Random forest

# In[ ]:


rf=RandomForestClassifier()
rf.fit(xtrain,ytrain)
y0pred=rf.predict(xtest)
print(accuracy_score(ytest,y0pred))


# XGBoost classifier-a boosting algorithm

# In[ ]:


xgb=XGBClassifier(max_depth=3,n_estimators=200,learning_rate=0.5)
xgb.fit(xtrain,ytrain)
y1pred=xgb.predict(xtest)
print(accuracy_score(ytest,y1pred))


# This model has a 90.3% accuracy score which is higher than the other 2 models.

# Printing the confusion matrix and checking the no. of false and true positives and negatives for the XGBoost classification model.

# In[ ]:


print(confusion_matrix(ytest,y1pred))


# We have 9+22 false predictions

# # Classification report

# In[ ]:


print(classification_report(ytest,y1pred))

