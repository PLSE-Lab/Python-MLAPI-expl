#!/usr/bin/env python
# coding: utf-8

# # CLASSIFICATION From Scratch-Wine Quality PREDICTION.

# <img src='https://drive.google.com/uc?id=1rDtG6-ZOCf_6lrLI5oLQMg_P1-XPkZih' width=1000 >

# ### In this Notebook we will Learn:-
# * Basic EDA.
# * Dimensionality reduction (PCA).
# * K-Cross validation to check accuracy.
# * ML's Classification Models like:-
#        * Logistic Regression
#        * Support Vector Machine (SVM) with Linear kernel
#        * Support Vector Machine (SVM) with Gaussian kernel
#        * K-Nearest Neighbour (KNN)
#        * Naive Bayes
#        * Decision Tree
#        * Random Forest
# * Prediction on new Values.

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))
print()
print("The files in the dataset are:-")
from subprocess import check_output
print(check_output(['ls','../input']).decode('utf'))

# Any results you write to the current directory are saved as output.



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# In[3]:


df = pd.read_csv('../input/winequality-red.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# * AS there is no Null Values in the dataset, good.
# * There is no Categorical values in the dataset.
# * Dataset is in proper format let us directly go to Classification portion.
# 

# In[10]:


# Let us see the correlation between the different variables
# We will reduce the number of features from Principle Componemt Analysis (PCA)
plt.figure(figsize=(12,6))
sns.heatmap(data=df.corr())
plt.show()


# ### ========================================================================

# # CLASSIFICATION:-
# 

# #### Here we will use 7 Algoritms/Models of Classification of machine learning.
# * They are as follow:-
# * Logistic Regresion
# * Support Vector Machine (SVM) with Linear kernel
# * Support Vector Machine (SVM) with Gaussian kernel
# * K-Nearest Neibhour (K-NN)
# * Naive Bayes
# * Decision Tree
# * Random Forest Model
# * We will use all the models one by one and then we will check the accuracy in each model and compare them, then we will choose the best model for our dataset.

# In[11]:


# Let us Import the Important Libraries  to train our Model for Machine Learning 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler      # For Scaling the dataset
from sklearn.model_selection import cross_val_score   # To check the accuracy of the model.


# * Here we do not have missing values, there is no categorical feature, and also we do not need to Split the dataset.
# * Here we will apply Scaling on our dataset.

# In[42]:


# Converting the DataFrame into Feature matrix and Target Vector.
x_train = df.iloc[:,:-1].values
y_train = df.iloc[:,-1].values


# #### 1). Scaling the dataset.

# In[43]:


# Let us use scaling on our dataset.

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)


# #### 2). Dimensionalty Reduction with Principle Component Analysis (PCA).

# In[44]:


from sklearn.decomposition import PCA
pca = PCA(n_components=None, )
x_train = pca.fit_transform(x_train)
explain_variance = pca.explained_variance_ratio_
explain_variance


# * We will take n_component = 5, as sum of initial 5 variance is 0.80, which is very good.

# In[45]:


# Let us apply PCA.
""" 
pca = PCA(n_components=5)
x_train = pca.fit_transform(x_train)
"""


# #### Observation:- 
# * I have check models without applying PCA and with applying PCA.
# * I am getting best result while not applying PCA.
# * So, we will not reduce the dimension of our matrix, as it is effecting our results by 5% which is very high.

# #### 3). Apply All Classification Models and compare their accuracies.

# In[68]:


def all_models():
    # Apply One model at a time , not all in a single function. If we run all in a single function then it will take too much memory(RAM) and time.

    # Apply Logistic regression
    # First step is to train our model .

    classifier_logi = LogisticRegression()
    classifier_logi.fit(x_train,y_train)

    # Let us check the accuracy of the model
    accuracy = cross_val_score(estimator=classifier_logi, X=x_train, y=y_train, cv=10)
    print(f"The accuracy of the Logistic Regressor Model is \t {accuracy.mean()}")
    print(f"The deviation in the accuracy is \t {accuracy.std()}")
    print()



    # Apply SVM with Gaussian kernel
    classifier_svm2 = SVC(kernel='rbf', )
    classifier_svm2.fit(x_train,y_train)
    accuracy = cross_val_score(estimator=classifier_svm2, X=x_train, y=y_train, cv=10)
    print(f"The accuracy of the SVM Gaussian kernel Model is \t {accuracy.mean()}")
    print(f"The deviation in the accuracy is \t {accuracy.std()}")
    print()



    # Apply K_NN Model
    # Train model
    classifier_knn = KNeighborsClassifier()
    classifier_knn.fit(x_train,y_train)
    # Check the accuracy.
    accuracy = cross_val_score(estimator=classifier_knn, X=x_train, y=y_train, cv=10)
    print(f"The accuracy of the KNN Model is \t {accuracy.mean()}") 
    print(f"The deviation in the accuracy is \t {accuracy.std()}")
    print()


    # Apply Naive Bayes Model.
    # Train Model
    classifier_bayes = GaussianNB()
    classifier_bayes.fit(x_train,y_train)
    # Check the accuracy and deviation in the accuracy
    accuracy = cross_val_score(estimator=classifier_bayes, X=x_train, y=y_train, cv=10)
    print(f"The accuracy of the Naive Bayes Model is \t {accuracy.mean()}") 
    print(f"The deviation in the accuracy is \t {accuracy.std()}")
    print()


    # Apply Random Forest Model.
    # Train Model
    classifier_ran = RandomForestClassifier(n_estimators=10, criterion='entropy')
    classifier_ran.fit(x_train,y_train)
    # Check the accuracy and deviation in the accuracy
    accuracy = cross_val_score(estimator=classifier_ran, X=x_train, y=y_train, cv=10)
    print(f"The accuracy of the Random Forest Model is \t {accuracy.mean()}") 
    print(f"The deviation in the accuracy is \t {accuracy.std()}")
    print()
    
    return classifier_svm2


# In[77]:


# Let us run the all_models funtion and see the accuracies of all model
classifier = all_models()


# #### Observation:- 
# * The best accuracy, we are getting is from SVM Gaussian kernel Model, which is 59% with deviation of 6%.
# * In future if we predict on new values then we will get the accuracy in range of 53% to 65%.
# * Now one can take all inputs from User and make it a numpy array and can predict from our model.

# In[78]:


# Our Target vector
y_train[:50]


# In[80]:


# Making prediction on our Feature matrix and compare it with our Target vector.
classifier.predict(x_train)[:50]


# * Maximum time we are getting the same result.

# ### =========================================================================
# ### =========================================================================
# ### =========================================================================
# ### =========================================================================

# # IF THIS KERNEL IS HELPFUL, THEN PLEASE UPVOTE.
# <img src='https://drive.google.com/uc?id=1N7u2YytozIoqjl_-wosgwj9Q0zzZu1mM' width=500 >

# In[ ]:




