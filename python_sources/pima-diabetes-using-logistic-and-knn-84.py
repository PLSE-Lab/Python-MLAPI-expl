#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


import os
print(os.listdir("../input"))


# **Data Collection**

# Let's start by reading in the diabetes.csv file into a pandas dataframe.

# In[ ]:


df= pd.read_csv('../input/diabetes.csv')


# In[ ]:


df


# **Explorer Dataset**

# In[ ]:


# shape
print(df.shape)


# In[ ]:


#columns*rows
df.size


# **How many NA elements in every column**

# In[ ]:


df.isnull().sum()


# **For getting some information about the dataset you can use info() command**

# In[ ]:


print(df.info())


# **To check the first 5 rows of the data set, we can use head(5).**

# In[ ]:


df.head(5)


# **To check out last 5 row of the data set, we use tail() function**

# In[ ]:


df.tail()


# **To pop up 5 random rows from the data set, we can use sample(5) function**

# In[ ]:


df.sample(5)


# **To give a statistical summary about the dataset, we can use describe()****

# In[ ]:


df.describe()


# **To check out how many null info are on the dataset, we can use isnull().sum().**

# In[ ]:


df.isnull().sum()


# **To print dataset columns, we can use columns atribute**

# In[ ]:


df.columns


# **Visualization**

# **Histogram**

# **We can also create a histogram of each input variable to get an idea of the distribution.**

# In[ ]:


# histograms
df.hist(figsize=(16,48))
plt.figure()


# In[ ]:


df.hist(figsize=(8,8))
plt.show()


# **Pairplot**

# In[ ]:


# Using seaborn pairplot to see the bivariate relation between each pair of features
sns.pairplot(df)


# **Heatmap**

# **Plot rectangular data as a color-encoded matrix.**

# In[ ]:


sns.heatmap(df.corr())


# **Missing Data**

# We can use seaborn to create a simple heatmap to see where we are missing data!

# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# **Distplot**

# Flexibly plot a univariate distribution of observations.

# In[ ]:


sns.distplot(df['Pregnancies'])


# **Countplot**

# Show the counts of observations in each categorical bin using bars.

# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(x='Outcome',data=df)
plt.title('Positive Outcome to Diabetes in Dataset')
plt.ylabel('Number of People')
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(data=df,x='Outcome',y='Pregnancies')
plt.title('Pregnancies Among Diabetes Outcomes.')
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x='Pregnancies',data=df,hue='Outcome')
plt.title('Diabetes Outcome to Pregnancies')
plt.show()


# In[ ]:


plt.figure(figsize=(13,6))
sns.countplot(x='Age',data=df,hue='Outcome')
plt.title('Diabetes Outcome to Age')
plt.show()


# In[ ]:


plt.figure(figsize=(13,6))
sns.countplot(x='SkinThickness',data=df,hue='Outcome')
plt.title('Diabetes Outcome to SkinThickness')
plt.show()


# **Train Test Split**

# In[ ]:


X=df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
y=df['Outcome']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Outcome',axis=1), 
                                                    df['Outcome'], test_size=0.2, 
                                                    random_state=201)


# **Using Logistic Regression**

# Now its time to train our model on our training data!
# 
# --Import LinearRegression from sklearn.linear_model--

# In[ ]:


from sklearn.linear_model import LogisticRegression


# **Create an instance of a LogisticRegression() model named lm.**

# In[ ]:


logmodel = LogisticRegression()
#** Train/fit lm on the training data.**
logmodel.fit(X_train,y_train)


# **Predicting Test Data**

# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# 
# ** Use lm.predict() to predict off the X_test set of the data.**

# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


print(classification_report(y_test,predictions))


# **Standardize the Variables**

# Because the KNN classifier predicts the class of a given test observation by identifying the observations that are nearest to it, the scale of the variables matters. Any variables that are on a large scale will have a much larger effect on the distance between the observations, and hence on the KNN classifier, than variables that are on a small scale.

# In[ ]:


#Standardize the Variables
from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaler.fit(df.drop('Outcome',axis=1))


# In[ ]:


scaled_features = scaler.transform(df.drop('Outcome',axis=1))


# In[ ]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# **Now above data are Standardize**

# **Using KNN**
# 
# Remember that we are trying to come up with a model to predict whether someone will TARGET CLASS or not. We'll start with k=1.

# **Train Test Split**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['Outcome'],
                                                    test_size=0.10,random_state=200)


# In[ ]:


#Using KNN
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


pred = knn.predict(X_test)


# **Predictions and Evaluations**
# 
# 
# Let's evaluate our KNN model!

# In[ ]:


#Predictions and Evaluations
from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,pred))


# In[ ]:


print(classification_report(y_test,pred))


# **Choosing a K Value**

# Let's go ahead and use the elbow method to pick a good K Value:

# In[ ]:


#Choosing a K Value
error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# Here we can see that that after arouns K>16 the error rate just tends to hover around 0.0-0.16 Let's retrain the model with that and check the classification report!

# In[ ]:


# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:


# NOW WITH K=19
knn = KNeighborsClassifier(n_neighbors=16)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

#print('WITH K=19')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# **Conclusion**

# **In this kernel, I have tried to cover all the parts related to the process of Machine Learning algorithm logistic regression and Support vector machine with a variety of Python packages . I hope to get your feedback to improve it.**
