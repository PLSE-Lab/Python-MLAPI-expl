#!/usr/bin/env python
# coding: utf-8

# Welcome, and thank you for opening this Notebook.
# This Notebook will provide knowledge to novice Data Scientists with basic Machine Learning concepts like - 
# 1. Data Exploratory Ananlysis
# 2. Principle Component Analysis
# 3. Prediction and Model selection

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[1]:


import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("../input/winequality-red.csv")


# In[3]:


data.head()


# Check the correlation for each of the fields

# In[5]:


data.corr


# In[6]:


data.columns


# In[7]:


data.info()


# In[8]:


data['quality'].unique()


# In[10]:


#Check correleation between the variables using Seaborn's pairplot. 
sns.pairplot(data)


# No correlation between the fields as seen on the pairplot

# In[11]:


#count of each target variable
from collections import Counter
Counter(data['quality'])


# In[12]:


#count of the target variable
sns.countplot(x='quality', data=data)


# In[13]:


#Plot a boxplot to check for Outliers
#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'fixed acidity', data = data)


# In[14]:


sns.boxplot('quality', 'volatile acidity', data = data)


# In[15]:


sns.boxplot('quality', 'citric acid', data = data)


# In[16]:


sns.boxplot('quality', 'residual sugar', data = data)


# In[17]:


sns.boxplot('quality', 'chlorides', data = data)


# In[18]:


sns.boxplot('quality', 'free sulfur dioxide', data = data)


# In[19]:


sns.boxplot('quality', 'total sulfur dioxide', data = data)


# In[20]:


sns.boxplot('quality', 'density', data = data)


# In[21]:


sns.boxplot('quality', 'pH', data = data)


# In[22]:


sns.boxplot('quality', 'sulphates', data = data)


# In[23]:


sns.boxplot('quality', 'alcohol', data = data)


# In[24]:


#boxplots show many outliers for quite a few columns. Describe the dataset to get a better idea on what's happening
data.describe()
#fixed acidity - 25% - 7.1 and 50% - 7.9. Not much of a variance. Could explain the huge number of outliers
#volatile acididty - similar reasoning
#citric acid - seems to be somewhat uniformly distributed
#residual sugar - min - 0.9, max - 15!! Waaaaay too much difference. Could explain the outliers.
#chlorides - same as residual sugar. Min - 0.012, max - 0.611
#free sulfur dioxide, total suflur dioxide - same explanation as above


# In[25]:


#next we shall create a new column called Review. This column will contain the values of 1,2, and 3. 
#1 - Bad
#2 - Average
#3 - Excellent
#This will be split in the following way. 
#1,2,3 --> Bad
#4,5,6,7 --> Average
#8,9,10 --> Excellent
#Create an empty list called Reviews
reviews = []
for i in data['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
data['Reviews'] = reviews


# In[26]:


#view final data
data.columns


# In[27]:


data['Reviews'].unique()


# In[28]:


Counter(data['Reviews'])


# Split the x and y variables
# 

# In[29]:


x = data.iloc[:,:11]
y = data['Reviews']


# In[30]:


x.head(10)


# In[31]:


y.head(10)


# Now scale the data using StandardScalar for PCA

# In[32]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)


# In[33]:


#view the scaled features
print(x)


# Proceed to perform PCA
# 

# In[34]:


from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(x)


# In[35]:


#plot the graph to find the principal components
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()


# In[36]:


#AS per the graph, we can see that 8 principal components attribute for 90% of variation in the data. 
#we shall pick the first 8 components for our prediction.
pca_new = PCA(n_components=8)
x_new = pca_new.fit_transform(x)


# In[37]:


print(x_new)


# Split the data into train and test data

# In[38]:


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.25)


# In[39]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# Proceed with Modelling.
# We will use the following algorithms 
# 1. Logistic Regression
# 2. Decision Trees
# 3. Naive Bayes
# 4. Random Forests
# 5. SVM

# 1. Logistic Regression
# 

# In[40]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_predict = lr.predict(x_test)


# In[41]:


#print confusion matrix and accuracy score
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print(lr_conf_matrix)
print(lr_acc_score*100)


# 98% accuracy with Logistic Regression! Let's see of Decision Trees give us a better accuracy

# In[42]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_predict = dt.predict(x_test)


# In[43]:


#print confusion matrix and accuracy score
dt_conf_matrix = confusion_matrix(y_test, dt_predict)
dt_acc_score = accuracy_score(y_test, dt_predict)
print(dt_conf_matrix)
print(dt_acc_score*100)


# Lesser accuracy with Decision Tree! Let's Use NaiveBayes

# In[44]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
nb_predict=nb.predict(x_test)


# In[45]:


#print confusion matrix and accuracy score
nb_conf_matrix = confusion_matrix(y_test, nb_predict)
nb_acc_score = accuracy_score(y_test, nb_predict)
print(nb_conf_matrix)
print(nb_acc_score*100)


# Similar accuracy as Decision Tree. Let's use RandomForest classifier now.

# In[46]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_predict=rf.predict(x_test)


# In[47]:


#print confusion matrix and accuracy score
rf_conf_matrix = confusion_matrix(y_test, rf_predict)
rf_acc_score = accuracy_score(y_test, rf_predict)
print(rf_conf_matrix)
print(rf_acc_score*100)


# 98% accuracy! Improvement from Decision Tree and Naive Bayes but the same as Logistic Regression Classifier

# SVM Classifier

# In[48]:


from sklearn.svm import SVC


# In[49]:


#we shall use the rbf kernel first and check the accuracy
lin_svc = SVC()
lin_svc.fit(x_train, y_train)
lin_svc=rf.predict(x_test)


# In[50]:


#print confusion matrix and accuracy score
lin_svc_conf_matrix = confusion_matrix(y_test, rf_predict)
lin_svc_acc_score = accuracy_score(y_test, rf_predict)
print(lin_svc_conf_matrix)
print(lin_svc_acc_score*100)


# ![](http://)98.5% accuracy wit RBF Kernel! Same as Random Forest! Let's try the linear kernel now and see if it improves our accuracy in any way.

# In[51]:


rbf_svc = SVC(kernel='linear')
rbf_svc.fit(x_train, y_train)
rbf_svc=rf.predict(x_test)


# In[52]:


rbf_svc_conf_matrix = confusion_matrix(y_test, rf_predict)
rbf_svc_acc_score = accuracy_score(y_test, rf_predict)
print(rbf_svc_conf_matrix)
print(rbf_svc_acc_score*100)


# The same accuracy! So we can see that the SVC and the Random Forest give us good prediction accuracy for the Wine Classification problem.
# We can further improve accuracy by fine-tuning the parameters of each classifier.
# Hope you found this Kernel useful! Pleae leave in comments in case of any questions, concerns, and feedback! Thank you :) 
