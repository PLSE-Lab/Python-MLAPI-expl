#!/usr/bin/env python
# coding: utf-8

# # Let's check your red wine quality...
# We have given various features (like fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol) which will help us in predicting the quality of wine.
# 
# <img src = "https://www.calaiswine.co.uk/wp/wp-content/uploads/2015/12/wine-gif-2.gif" width=500px >

# **Problem Statement : ** Predicting red wine quality using various features of red wine.
# 
# **Solution to the problem : **
# 
# 1. [Import Libraries](#1)
# 2. [Load Data](#2)
#     * [Checking the information about each data column](#3)     
# 3. [Data Visualuzation](#4)
#     * [Barplot between `quality` and `fixed acidity`](#5)
#     * [Barplot between `quality` and `volatile acidity`](#6)
#     * [Barpolt between `quality` and `citric acid`](#7)
#     * [Barplot between `quality` and `residual sugar`](#8)
#     * [Barplot between `quality` and `chlorides`](#9)
#     * [Barplot between `quality` and `free sulfur dioxide`](#10)
#     * [Barplot between `quality` and `total sulfur dioxide`](#11)
#     * [Barplot between `quality` and `sulphates`](#12)
#     * [Barplot between `quality` and `alcohol`](#13)
#     * [Conclusion by visualization](#14)
# 4. [Data Preprocessing](#15)
#     * [Creating new column `review`](#16)
#     * [Checking unique values for column `review`](#17)
#     * [Scaling the data using StandardScaler for PCA](#18)
#     * [Viewing the data using StandardScaler](#19)
#     * [Proceed to perform PCA](#20)
#     * [Ploting the graph to find the principal components](#21)
# 5. [Splitting data into Train and Test](#22)
#     * [Checking for shape of splitted data](#23)
# 6. [Data Modelling](#24)
#     * [Logistic Regression](#25)
#     * [Decision Trees](#26)
#     * [Naive Bayes](#27)
#     * [Random Forests](#28)
#     * [SVM](#29) 
#     * [Accuracy for different algorithms](#30)

# ## 1. Import Libraries<a id="1"></a>

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2. Load Data <a id="2"></a>

# In[ ]:


wine = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")


# In[ ]:


#Let's check how the data is distributed
wine.head()


# ### Checking the information about each data column<a id="3"></a>

# In[ ]:


wine.info()


# ## 3. Data Visualization<a id="4"></a>
# Now, I am going to visualize this data to see how the data is distributed.

# In[ ]:


sns.countplot(x='quality',data=wine)


# ### Barplot between `quality` and `fixed acidity`<a id="5"></a>

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x='quality', y='fixed acidity', data=wine)


# Here, we see that `fixed acidity` does not give any specification to classify the `quality`.

# ### Barplot between `quality` and `volatile acidity`<a id="6"></a>

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x='quality', y='volatile acidity', data=wine)


# Here, we see that it's quite a downing trend in the `volatile acidity` as we go higher the `quality`.

# ### Barpolt between `quality` and `citric acid`<a id="7"></a>

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x='quality', y='citric acid', data=wine)


# Here, we see the increasing trend of `citric acid`. That is, as we go higher in `quality` of wine the composition of `citric acid` in wine also increases.

# ### Barplot between `quality` and `residual sugar`<a id="8"></a>

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x='quality', y='residual sugar', data=wine)


# Well, there is no significant effect of `residual sugar` on `quality` of wine.

# ### Barplot between `quality` and `chlorides`<a id="9"></a>

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x='quality', y='chlorides', data=wine)


# Here, we see the decreasing trend of `chlorides` with the increase in the `quality` of wine.

# ### Barplot between `quality` and `free sulfur dioxide`<a id="10"></a>

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x='quality', y='free sulfur dioxide', data=wine)


# ### Barplot between `quality` and `total sulfur dioxide`<a id="11"></a>

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x='quality', y='total sulfur dioxide', data=wine)


# Both the `free sulphur dioxide` and `total sulphur dioxide` are comparatively more in the 5th and 6th `quality` wine.

# ### Barplot between `quality` and `sulphates`<a id="12"></a>

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x='quality', y='sulphates', data=wine)


# Ohh yeah, here we the the increasing trend of `sulphates` as we go higher in `quality` of wine.

# ### Barplot between `quality` and `alcohol`<a id="13"></a>

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x='quality', y='alcohol', data=wine)


# Here, is also a increasing trend found between `quality` and `alcohol`.

# ### Overall conclusion by examining data.<a id="14"></a>
# Some features have great impact on `quality` of wine and some does not have any sigificant effect in the `quality`.
# 
# **Trends**
# 1. fixed acidity : No significant effect
# 2. volatile acidity : Decreasing
# 3. citric acid : Increasing
# 4. residual sugar : No significant effect
# 5. chlorides : Decreasing
# 6. free sulphur dioxide : No significant effect
# 7. total sulphur dioxide : No significant effect
# 8. sulphates : Increasing
# 9. alcohol : Increasing

# ## 4. Data Preprocessing<a id="15"></a>

# ### Creating new column `review`<a id="16"></a>

# >Now, we will create a new column called review. This column will contain the values of 1, 2 and 3 and will be split in the following way.
# * review ==> quality ==> meaning
# * 1 ==> 1, 2, 3 ==>Bad
# * 2 ==> 4, 5, 6, 7 ==> Average
# * 3 ==> 8, 9, 10 ==> Excellent

# In[ ]:


reviews = []
for i in wine['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
wine['Reviews'] = reviews


# In[ ]:


wine.columns


# ### Checking unique values for column `review`<a id="17"></a>

# In[ ]:


wine['Reviews'].unique()


# In[ ]:


Counter(wine['Reviews'])


# In[ ]:


X = wine.iloc[:,:11]
y = wine['Reviews']


# In[ ]:


X.head()


# In[ ]:


y.head()


# ### Scaling the data using StandardScaler for PCA<a id="18"></a>

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# ### Viewing the data using StandardScaler<a id="19"></a>

# In[ ]:


print(X)


# ### Proceed to perform PCA<a id="20"></a>

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA()
X_pca = pca.fit_transform(X)


# ### Ploting the graph to find the principal components<a id="21"></a>

# In[ ]:


plt.figure(figsize=(5,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()


# In[ ]:


#As per the graph, we can see that 8 principal components attribute for 90% of variation in the data. 
#we shall pick the first 8 components for our prediction.
pca_new = PCA(n_components=8)
X_new = pca_new.fit_transform(X)


# In[ ]:


print(X_new)


# ## 5. Splitting the dataset into train and test data.<a id="22"></a>

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25)


# ### Checking for shape of splitted data<a id="23"></a>

# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## 6. Data Modelling<a id="24"></a>
# We will use the following algorithms ==>
# 1. Logistic Regression
# 2. Decision Trees
# 3. Naive Bayes
# 4. Random Forests
# 5. SVM

# ### Logistic Regression<a id="25"></a>

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)


# In[ ]:


# print confusion matrix and accuracy score
lr_confusion_matrix = confusion_matrix(y_test, lr_predict)
lr_accuracy_score = accuracy_score(y_test, lr_predict)
print(lr_confusion_matrix)
print(lr_accuracy_score*100)


# 98.5% accuracy with Logistic Regression! Let's see of Decision Trees give us a better accuracy.

# ### Decision Tree<a id="26"></a>

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt_predict = dt.predict(X_test)


# In[ ]:


#print confusion matrix and accuracy score
dt_confusion_matrix = confusion_matrix(y_test, dt_predict)
dt_accuracy_score = accuracy_score(y_test, dt_predict)
print(dt_confusion_matrix)
print(dt_accuracy_score*100)


# 97% accuracy with Decision Tree! Let's use NaiveBayes

# ### Naive Bayes<a id="27"></a>

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_predict = nb.predict(X_test)


# In[ ]:


#print confusion matrix and accuracy score
nb_confusion_matrix = confusion_matrix(y_test, nb_predict)
nb_accuracy_score = accuracy_score(y_test, nb_predict)
print(nb_confusion_matrix)
print(nb_accuracy_score*100)


# 97.75% accuracy with Naive Bayes.

# ### Random Forest Classifier<a id="28"></a>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_predict = rf.predict(X_test)


# In[ ]:


# print confusion matrix and accuracy score
rf_confusion_matrix = confusion_matrix(y_test, rf_predict)
rf_accuracy_score = accuracy_score(y_test, rf_predict)
print(rf_confusion_matrix)
print(rf_accuracy_score*100)


# 98.25% accuracy with Random forest.

# ### Support Vector Machine (SVM)<a id="29"></a>

# In[ ]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
svc_predict = svc.predict(X_test)


# In[ ]:


#print confusion matrix and accuracy score
svc_confusion_matrix = confusion_matrix(y_test, rf_predict)
svc_accuracy_score = accuracy_score(y_test, rf_predict)
print(svc_confusion_matrix)
print(svc_accuracy_score*100)


# #### Accuracy for different algorithms:<a id="30"></a>
# 
# * Logistic Regression = 98.5% accuracy  
# * Decision Trees = 97% accuracy
# * Naive Bayes = 97.75% accuracy
# * Random Forest = 98.25% accuracy
# * SVM = 98.25% accuracy

# In[ ]:


wine1 = [[7.8, 0.760, 0.04, 2.3, 0.092, 15.0, 54.0, 0.99700]]
print("Decision Tree : ",dt.predict(wine1))
print("Logistic Regression : ",lr.predict(wine1))
print("Naive Bayes : ",nb.predict(wine1))
print("Random forest : ",rf.predict(wine1))
print("SVM : ",svc.predict(wine1))


# Well, Naive Bayes did wrong prediction! Therefore, in this way we can predict the quality of red wine using **Logistic regression** because it gives highest accuracy.
