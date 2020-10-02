#!/usr/bin/env python
# coding: utf-8

# ## **GOALS**

# The goal of this project is to predict the Heart Disease presence within the patients in this datasets. 
# 
# This goal hopefully will be achieved through the simple Random Forest Classification Model & K - Nearest Neighbors Model. With maybe some EDA along the way.

# The variables are explained below : 
# 
# 1. age: The person's age 
# 2. sex: The person's gender (1 = Male, 0 = Female)
# 3. cp: The chest pain experienced (4 types)
# 4. trestbps: The person's resting blood pressure
# 5. chol: The person's cholesterol measurement in mg/dl
# 6. fbs: If the person's fasting blood sugar > 120 mg/dl (1 = True; 0 = False)
# 7. restecg: Resting electrocardiographic measurement
# 8. thalach: The person's maximum heart rate achieved
# 9. exang: Exercise induced angina (1 = Yes; 0 = No)
# 10. oldpeak: ST depression induced by exercise relative to rest
# 11. slope: the slope of the peak exercise ST segment
# 12. ca: The number of major vessels (0-3)
# 13. thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# 14. target: Heart disease diagnose (0 = No, 1 = Yes)

# ## **IMPORTING LIBRARIES**

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ## **LOAD DATASET - OVERVIEW**

# First let's load the dataset, and check the datatype for each variables.

# In[ ]:


df = pd.read_csv('../input/heart-disease-uci/heart.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# ### **SOME EDA**

# Let's do some EDA to see the pattern within this dataset. I'll be using only the continous variables.

# In[ ]:


for i in df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]:
    sns.distplot(df[i])
    plt.title('Distribution of ' + i)
    plt.show()


# In[ ]:


plt.figure(figsize = (10, 16))
sns.countplot(y = df[df['target']==1]['age'])
plt.title('Count Of Positive Cases by Age')

Here we can see that within this dataset, most the patients blood pressure on range of 100 to 140, with most of the cholesterol range in 200mg/dl to 300mg/dl. The person's maximum heart rate achieved from 150 to 170. Then we can see that the patients within range of 41 to 64 years are prone to Heart Disease as they showing more cases than the others.
# ## **GETTING DUMMIES**

# Now let's get to the Labelling to our categorical variable, I'll use the pd.get_dummies to achieve this.

# In[ ]:


a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")

frames = [df, a, b, c]
df_heart = pd.concat(frames, axis = 1)
df_heart = df_heart.drop(columns = ['cp', 'thal', 'slope'])


# In[ ]:


df_heart.head()


# In[ ]:


df_heart.info()


# ## **SPLITTING DATA**

# Now before fitting our models, I want to split it to 80% Train and 20% Test data.

# In[ ]:


x = df_heart.drop(['target'], axis = 1)
y = df_heart.target.values


# In[ ]:


# Split the data with 80% Train size

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,random_state=0)


# In[ ]:


x_train.head()


# In[ ]:


x_test.head()


# In[ ]:


y_train


# In[ ]:


y_test


# From this point below, we'll start building the Machine Learning Model and fitting it to our train data, then start to predict it to test data.

# ## **RANDOM FOREST CLASSIFIER**

# In[ ]:


# Model Fitting

RFC = RandomForestClassifier(n_estimators = 2000, min_samples_split= 2, min_samples_leaf = 1, max_depth = 25)
RFC.fit(x_train, y_train)


# In[ ]:


# Random Forest Classifier predict

yp_RFC = RFC.predict(x_test)


# In[ ]:


# Confusion Matrix

cm_RFC = confusion_matrix(y_test,yp_RFC)
cm_RFC


# In[ ]:


# Labels for Confusion Matrix

labels = ['No Disease', 'Have Disease']


# In[ ]:


# Printing Classification Report and Showing Confusion Matrix

print(classification_report(y_test, yp_RFC, target_names = labels))
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(cm_RFC, annot=True, fmt=".0f", ax=ax)

ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)

plt.title('Heart Prediction With Random Forest Classifier')
plt.xlabel("ACTUAL")
plt.ylabel("PREDICT")
plt.show()


# In[ ]:


# Printing Score

print(RFC.score(x_test,y_test))


# In[ ]:


# Classification Report for Summary

report_RFC = pd.DataFrame(classification_report(y_test, yp_RFC, target_names= labels, output_dict=True)).T


# # **K Nearest Neighbors**

# In[ ]:


# Determining the K-Value

k = round(len(x_train)**0.5)+1
k


# In[ ]:


# Fitting Model

KNN = KNeighborsClassifier(n_neighbors = k)
KNN.fit(x_train, y_train)


# In[ ]:


# KNN Predict

yp_KNN = KNN.predict(x_test)


# In[ ]:


# Confusion Matrix

cm_KNN = confusion_matrix(y_test,yp_KNN)
cm_KNN


# In[ ]:


# Printing Classification Report and Showing Confusion Matrix 

print(classification_report(y_test, yp_KNN, target_names = labels))
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(cm_KNN, annot=True, fmt=".0f", ax=ax)

ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)

plt.title('Heart Prediction With KNearest Neighbors')
plt.xlabel("ACTUAL")
plt.ylabel("PREDICT")
plt.show()


# In[ ]:


# Printing Score

print(KNN.score(x_test,y_test))


# In[ ]:


# Classification Report for Summary

report_KNN = pd.DataFrame(classification_report(y_test, yp_KNN, target_names= labels, output_dict=True)).T


# ## **SUMMARY**

# In[ ]:


# Showing the Confusion Matrix for both models

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,6))
sns.heatmap(cm_RFC, annot=True, fmt=".0f", ax=ax1)
sns.heatmap(cm_KNN, annot=True, fmt=".0f", ax=ax2)

ax1.xaxis.set_ticklabels(labels), ax1.yaxis.set_ticklabels(labels)
ax2.xaxis.set_ticklabels(labels), ax2.yaxis.set_ticklabels(labels)

ax1.set_title('RFC'), ax2.set_title('KNN')
ax1.set_xlabel('ACTUAL'), ax2.set_xlabel('ACTUAL')
ax1.set_ylabel('PREDICTED'), ax2.set_ylabel('PREDICTED')

plt.show()


# In[ ]:


print('RFC Model : ', RFC.score(x_test,y_test))
print('KNN Model : ', KNN.score(x_test,y_test))


# In[ ]:


# Printing Classification Report Summary
pd.concat([report_RFC, report_KNN], keys = ['RFC MODEL', 'KNN MODEL'])


# From the two models score comparison above, the Random Forest Classifier is having 0.88 score, while the K - Nearest Neighbors score is only 0.72. The F1-Score, Precision and Recall for Random Forest Classifier is also much higher than K - Nearest Neighbors. Therefore we can see that in predicting this case with the comparison between the two models above, Random Forest Classifier will be a better choice than the K-Nearest Neighbors.
