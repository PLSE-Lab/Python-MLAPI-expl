#!/usr/bin/env python
# coding: utf-8

# # ML From Scratch With IRIS Dataset

# ### In this Notebook we will Learn:-
# * Basic EDA.
# * Plotly Visualisation.
# * Spliting the Dataset into training set and test set.
# * Dealing with Categorical Dataset.
# * K-Cross validation to check accuracy.
# * ML's Classification Models like:-
#              * Logistic Regression
#              * Support Vector Machine (SVM) with Linear kernel
#              * Support Vector Machine (SVM) with Gaussian kernel
#              * K-Nearest Neighbour (KNN)
#              * Naive Bayes
#              * Decision Tree
#              * Random Forest
# * Prediction on new Values.             

# In[123]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, download_plotlyjs
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))
print()
print("The files in the dataset are:-- ")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[124]:


# IMporting the dataset
df = pd.read_csv("../input/Iris.csv")


# In[125]:


# Checking the top 5 entries
df.head()


# # BASIC EDA

# In[126]:


print(f"The number of rows and columns in the dataset are \t {df.shape}")


# In[127]:


# Let's check the unique Species in the dataset, which we will predict in the end.
print(df['Species'].unique())
print("There are 3 species .")


# In[128]:


# Let us check whether we have null values in the dataset or not.
print(df.isnull().sum())
print()
print()
print("As one can see there is No Null Values in the dataset.")


# In[129]:


# Let us remove the unwanted columns/features which will not help us to predict the Species of the Flower
df.drop('Id', axis=1, inplace=True)


# #### Let us get Some Statistical Knowledge about the dataset

# In[130]:


# Let us see the distribution of the SepalLength,SepalWidth,PetalLength, PetalWidth
# And get the Statistical Knowledge of the dataset
temp_df = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm', 'PetalWidthCm']]
temp_df.iplot(kind='box', title='Distribution of Length and Width of Sepal and Petal in Cm', yTitle='Frequency')


# #### Let's see the correlation between the random Variable/ different features

# In[131]:


df.corr().iplot(kind='heatmap', )


# #### Observation:-
# * SepalLength and SepalWidth are less correlated.
# * SepalLength is higly correlated with PetalLength and PetalWidth.
# * SepalWidth is average correlated with PetalLength and PetalWidth
# * And finally PetalLength and PetalWidth are highly Correlated.
# * But for making prediction we will take all the features.

# #### =====================================================================================================

# # PREDICTION WITH ML MODELS

# #### Here we will use 7 Algoritms/Models of Classification of machine learning.
# #### They are as follow:-
# * Logistic Regresion
# * Support Vector Machine (SVM) with Linear kernel
# * Support Vector Machine (SVM) with Gaussian kernel
# * K-Nearest Neibhour (K-NN)
# * Naive Bayes
# * Decision Tree
# * Random Forest Model
# 
# #### We will use all the models one by one  and then we will check the accuracy in each model and compare them, then we will choose the best model for our dataset.

# In[132]:


# Let us Import the Important Libraries  to train our Model for Machine Learning 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # To deal with Categorical Data in Target Vector.
from sklearn.model_selection import train_test_split  # To Split the dataset into training data and testing data.
from sklearn.model_selection import cross_val_score   # To check the accuracy of the model.


# In[133]:


df.head()


# #### Let us do data preprocessing step 
# * Here we will deal with four concept 
# #### MCSS
# * 'M' for missing, means dealing with the missing data.
# * 'C' for Categorical, means dealing with the Categorical data.
# * 'S' for Spliting, means spliting the dataset into training set and test set.
# * 'S' for Scaling , means scaling the features so that we can compare many variable on the same scale.

# In[134]:


# Creating Feature Matric and Target Vector.
X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values


# In[135]:


# Let us check whether we have null values in the dataset or not.
print(df.isnull().sum())
print()
print()
print("As one can see there is No Null Values in the dataset.")


# In[136]:


# Now we have Categorical data in our Target vector and we need to convert 
# it into values, So that we can easyly perform Mathmethical operations.

label_y = LabelEncoder()
Y = label_y.fit_transform(Y)


# In[137]:


Y


# In[138]:


df['Species'].unique()


# * 0 means Iris-setosa, 1 means Iris-versicolor, 2 means Iris-virginica

# #### Let us split the dataset into training set and Test set so that we can check accuracy of model.

# In[139]:


x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2)


# In[140]:


# There is no need of Scaling the features.


# #### =================================================================================================
# ### Let us make Models one by one.

# ### 1). Logistic Regression

# In[141]:


# First step is to train our model .

classifier_logi = LogisticRegression()
classifier_logi.fit(x_train,y_train)


# In[142]:


# Let's Predict our model on test set.
y_pred = classifier_logi.predict(x_test)


# In[143]:


# Let us check the accuracy of the model
accuracy = cross_val_score(estimator=classifier_logi, X=x_train, y=y_train, cv=10)
print(f"The accuracy of the Logistic Regressor Model is \t {accuracy.mean()}")
print(f"The deviation in the accuracy is \t {accuracy.std()}")


# * Here we are getting the accuracy of 96% which is more than enough.
# * Let us check the accuracy of other models.

# #### ====================================================================================================

# ### 2). Support Vector Machine (SVM) with Linear kernel.

# In[144]:


# Let us tran model
classifier_svm1 = SVC(kernel='linear')
classifier_svm1.fit(x_train,y_train)


# In[145]:


# Let's predict on test dataset.
y_pred = classifier_svm1.predict(x_test)


# In[146]:


# Check the accuracy.
accuracy = cross_val_score(estimator=classifier_svm1, X=x_train, y=y_train, cv=10)
print(f"The accuracy of the SVM linear kernel Model is \t {accuracy.mean()}")
print(f"The deviation in the accuracy is \t {accuracy.std()}")


# * Here we get the accuracy of 97%.

# #### =====================================================================================================

# ### 3). SVM with Gaussian kernel

# In[147]:


# Train the model
classifier_svm2 = SVC(kernel='rbf', )
classifier_svm2.fit(x_train,y_train)


# In[148]:


# Predict on test set.
y_pred = classifier_svm2.predict(x_test)


# In[149]:


# Check the accuracy.
accuracy = cross_val_score(estimator=classifier_svm2, X=x_train, y=y_train, cv=10)
print(f"The accuracy of the SVM Gaussian kernel Model is \t {accuracy.mean()}")
print(f"The deviation in the accuracy is \t {accuracy.std()}")


# #### =================================================================================================
# 

# ### 4). K- Nearest Neighbour (KNN)

# In[150]:


# Train model
classifier_knn = KNeighborsClassifier()
classifier_knn.fit(x_train,y_train)


# In[151]:


# predict on test set
y_pred = classifier_knn.predict(x_test)


# In[152]:


# Check the accuracy.
accuracy = cross_val_score(estimator=classifier_knn, X=x_train, y=y_train, cv=10)
print(f"The accuracy of the KNN Model is \t {accuracy.mean()}") 
print(f"The deviation in the accuracy is \t {accuracy.std()}")


# #### ================================================================================================

# ### 5). Naive Bayes Model.

# In[153]:


# Train Model
classifier_bayes = GaussianNB()
classifier_bayes.fit(x_train,y_train)


# In[154]:


# Predict on test set.
y_pred = classifier_bayes.predict(x_test)


# In[155]:


# Check the accuracy and deviation in the accuracy
accuracy = cross_val_score(estimator=classifier_bayes, X=x_train, y=y_train, cv=10)
print(f"The accuracy of the Naive Bayes Model is \t {accuracy.mean()}") 
print(f"The deviation in the accuracy is \t {accuracy.std()}")


# #### ====================================================================================================

# ### 6). Decision Tree Model

# In[156]:


# Train model
classifier_deci = DecisionTreeClassifier()
classifier_deci.fit(x_train,y_train)


# In[157]:


# Predict on test set
y_pred = classifier_deci.predict(x_test)


# In[158]:


# Check the accuracy and deviation in the accuracy
accuracy = cross_val_score(estimator=classifier_deci, X=x_train, y=y_train, cv=10)
print(f"The accuracy of the Decision Tree Model is \t {accuracy.mean()}") 
print(f"The deviation in the accuracy is \t {accuracy.std()}")


# #### =====================================================================================================

# ### 7). Random Forest Model
# 

# In[159]:


# Train Model
classifier_ran = RandomForestClassifier()
classifier_ran.fit(x_train,y_train)


# In[160]:


# Predict on test set.
y_pred = classifier_ran.predict(x_test)


# In[161]:


# Check the accuracy and deviation in the accuracy
accuracy = cross_val_score(estimator=classifier_ran, X=x_train, y=y_train, cv=10)
print(f"The accuracy of the Random Forest Model is \t {accuracy.mean()}") 
print(f"The deviation in the accuracy is \t {accuracy.std()}")


# #### Observation:-
# * As we have completed all the models of classification.
# * Let us choose the best one among them on the basis of accuracy and their deviation.
# * Out all the models SVM with linear and SVM with gaussian kernel are best as both give the same accuracy and deviation of 97% and 5% respectively.
# * It means when we make prediction with SVM linear kernel and SVM gaussian kernel, then our accuracy will vary in range of 92% to 100%.

# ## Now Let us make Prediction on new values of SepalLength, SepalWidth, PetalLength, PetalWidth.

# In[162]:


# Let's make prediction on new values.
try:
    sepalLength = float(input("Enter Sepal Length:\t"))
    sepalWidth = float(input("Enter Sepal Width:\t"))
    petalLength = float(input("Enter Petal Length:\t"))
    petalWidth = float(input("Enter Petal Width:\t"))

    new_values = [[sepalLength,sepalWidth,petalLength,petalWidth],]  # Making 2-D array.

    species = classifier_svm2.predict(new_values) # Using SVM Gaussian kernel

    if species[0]==0:
        flag = 'Iris-setosa'
    elif species[0]==1:
        flag = 'Iris-versicolor'
    else:
        flag = 'Iris-virginica'

    print()
    print()
    print(f"*** The Species is: \t {flag} ****")    
    
except Exception as e:
    print("Run this code with Python")


# #### =====================================================================================================
# #### =====================================================================================================
# #### =====================================================================================================
# #### =====================================================================================================

# # IF THIS KERNEL IS HELPFUL, THEN PLEASE UPVOTE.
# <img src='https://drive.google.com/uc?id=1qihsaxx33SiVo5dIw-djeIa5SrU_oSML' width=400 >

# In[ ]:




