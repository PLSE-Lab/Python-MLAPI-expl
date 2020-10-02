#!/usr/bin/env python
# coding: utf-8

# **INTRODUCTION**
# 
# This is my first kernel on Kaggle, and I will be attempting to predict the test data using a comparison of Supervised Learning ML models.
# 
# This is largely for beginners who are keen on learning ML, and are interested to learn a methodical workflow that covers importation of libraires and data cleaning to actual modelling. 
# 
# Sure, this is dataset done to depth on Kaggle, but if you are keen to see my approach, feel free to stay and give me an upvote.** I'll be explaining things in layman terms** so if that's what you are looking for, good for you :)

# **STEP 1**
# 
# Import the necessary libraries into this notebook/console
# 
# A notebook/console is just a place where you can type, execute and edit code. Think of it as an environment that everything else takes place in.
# 
# Libraries are exactly what they mean - pre made sets of code that are readily available for use. Without libraries, you'll have to write alot more code than you do now!
# 
# Notice that I also imported a dataset (ie the 'train') data to train our models. I labelled this 'df'.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('../input/train.csv')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df.head()


# In[ ]:


df.info()


# **STEP 2**
# 
# Check for missing data. Looks abit complex here, but nothing to worry about. I simply used seaborn, a python visualisation library, to run a heat map on missing values. The rest of the code is for aesthetic purposes.

# In[ ]:


sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# It is apparent that 'Age' and 'Cabin' are missing alot of values. Let's keep this in mind while we proceed to do some Exploratory Data Analysis.
# 
# Thereafter, we will proceed onto Data Cleaning.

# **STEP 3**
# 
# Exploratory Data Analysis (EDA)
# 
# For those of you unsure with EDA, it is just a way to visualise your training data, and look at the relationships between different variables. 
# 
# This is primarily done here once again using seaborn (ie. sns)

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=df,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=df,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=df,palette='rainbow')


# In[ ]:


df['Age'].hist(bins=30,color='darkred',alpha=0.7)


# **STEP 4**
# 
# Data Cleaning
# 
# We will attempt to impute values into the missing age values to run our model. We do this by first associating a relationship between age and passenger class.
# 
# Our boxplot shows the median age values for each passenger class.
# 
# Next, we create a function to impute these median age values into the missing slots as seen on our heatmap earlier.
# 
# Lastly, we apply this function to the age column of the training dataframe.

# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')


# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age) :
        
        if Pclass == 1 :
            return 37
        if Pclass == 2 :
            return 29
        else :
            return 24
        
    else : 
        return Age
    
df['Age'] = df[['Age','Pclass']].apply(impute_age, axis =1)


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# As the heatmap shows, we have successfully filled up the appropriate missing values under the Age column.
# 
# We will now drop the Cabin column as it has too many missing pieces of data.

# In[ ]:


df.drop('Cabin',axis =1, inplace = True)


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# **STEP 5**
# 
# Converting Categorical Variables into dummy variables
# 
# To put it simply, categorical variables like sex = male/ female are written in words, and very unfriendly to our ML model.Our model can't process words as training data.
# 
# We are going to convert these variables into numbers 1 & 0 for our model to process with ease.

# In[ ]:


df.info()


# In[ ]:


sex = pd.get_dummies(df['Sex'],drop_first=True)
embark = pd.get_dummies(df['Embarked'],drop_first=True)


# In[ ]:


df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


df = pd.concat([df,sex,embark],axis=1)


# In[ ]:


df.head()


# An explanation : 
# 
# As you can see, we have converted the variables 'sex' and 'embark' into dummy variables.
# 
# We have also dropped the initial columns (sex, embark, name, ticket).
# 
# This is because name and ticket are variables that intuitively do not affect survival, and they can't be converted to dummy variables as well.
# 
# We then join the newly made 'sex' and 'embark' columns (showing dummy variables) back into the dataframe. 
# 
# 

# **STEP 6**
# 
# Building and executing a Logistic Regression Model.
# 

# **STEP 6a. ** : Import Logistic Regression Model from Sci-Kit Learn
# 
# Sci-Kit Learn (aka sklearn) is a machine learning and data science library with pre-built prediction models. To code the regression formula as a full function using numpy is tedious and beyond the scope of this tutorial.
# 
# Remember, the goal of our model is one of classification - to predict which class of survival passengers in the test data will fall into.

# In[ ]:


from sklearn.linear_model import LogisticRegression


# **STEP 6b. ** : Train Test Split
# 
# What this step does is split the training data given by kaggle into a training data set and a test data set internally. 
# 
# We shall use this test data to gauge the predictive value of our model before we apply it to Kaggle's test data under the Titanic data set competition.
# 
# The model will then compare and find a relationship (based on logistic regression) between X_train (ie the training features) and y_train (ie the label).
# 
# We start by importing train_test_split from sklearn and then assigning examples in our current df dataframe to either 'train' or 'test' data sets for our model to run on. 

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived',axis=1), 
                                                    df['Survived'], test_size=0.30, 
                                                    random_state=42)


# Notice we has to remove the 'Survived' column from our 'X' as it is the label of the training examples, and not a feature/variable itself.

# **STEP 6c. ** : Fit data into Model
# 
# First we instantise the model as a function. This just means we can use other functions like '.fit' on it to fit our training data.

# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# **STEP 6d. ** : Import and run predictions on test data
# 
# We shall now use the trained model above to predict the y_test values from the X_test examples and features. 
# 
# We will then compare it to the true y_test labels and see how accurate our model is.
# 
# Remember, the goal of our model is one of classification - to predict which class of survival passengers in the test data will fall into.

# In[ ]:


predictions = logmodel.predict(X_test)

from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(predictions, y_test))


# In[ ]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test, predictions))


# **NOT TOO SHABBY! About an 80% prediction rate. ;)**

# **STEP 7**
# 
# **Support Vector Machines (SVM)**
# 
# We shall run similar steps to STEP 6 above, but this time, importing and using SVMs in the place of Logistic Regression, to see if it has a higher predictive value. 

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


SVCmodel = SVC()
SVCmodel.fit(X_train, y_train)


# In[ ]:


predictionsSVC = SVCmodel.predict(X_test)
print('The accuracy of the Support Vector Machine is',metrics.accuracy_score(predictionsSVC, y_test))


# In[ ]:


print(classification_report(y_test,predictionsSVC))
print(confusion_matrix(y_test, predictionsSVC))


# As you can see, SVM has a much lousier predictive value in this instance. This might be due to the nature of the data and its suitability to fit into SVMs.  However, before we give up, we have a few tricks up our sleeve to see if we can boost the effectivness of SVMs. 
# 
# What we are going to do is run something called a Grid Search. What this does is run through many possible combinations that make up the parameters of this SVM model, and find the most effective one. 
# 
# Parameters are basically settings in a model that affects the model's behaviour. They are not directly a source fo data/ training example, but rather, contro lthe way the model processes data, such as prioritizing certain features over others to make a prediction. 

# In[ ]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# In[ ]:


grid.fit(X_train,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid_predictions = grid.predict(X_test)
print('The accuracy of the Support Vector Machine with Grid Search is',metrics.accuracy_score(grid_predictions, y_test))


# In[ ]:


print(classification_report(y_test,grid_predictions))
print(confusion_matrix(y_test, grid_predictions))


# **As you can see, predictive value has certainly increased to 75 % ! **

# **STEP 8**
# 
# **K Nearest Neighbours (KNN)**
# 
# Finally, we shall attempt to a KNN model on this data set and see if it yields good results. Following the same procedures as before :

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train)


# In[ ]:


KNNpredictions = knn.predict(X_test)


# In[ ]:


print('The accuracy of KNN with 1 neighbor is',metrics.accuracy_score(KNNpredictions, y_test))


# Now, just like he SVM before, a poor result is not the end of the world! We can once again adjust the settings of a model (in this case the n_neighbors) to get an optimal result, 
# 
# **Before we proceed, do be reminded that KNN is not a parametric model! So we can't exactly call the n_neighbors a 'parameter'.**
# 
# Now, let's plot a graph showing the error rate of the model against the K value (ie the n_neighbors).
# 
# From here, we can find the K value with the lowest error rates! 
# 
# This also shows the importance of visualisation when it comes to data science. 

# In[ ]:


error_rate = []

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


# **Now we see that a K value of 7 yields a low error rate. Let's use 7 in our model. **

# In[ ]:


knn2 = KNeighborsClassifier(n_neighbors = 7)
knn2.fit(X_train,y_train)


# In[ ]:


KNN2predictions = knn2.predict(X_test)


# In[ ]:


print('The accuracy of KNN with 7 neighbors is',metrics.accuracy_score(KNN2predictions, y_test))


# **SLIGHTLY BETTER! But nowhere near Logistic Regression ;)**

# And now, it has come to the time to explain the reasons behind the results. 
# 
# First off, KNN did poorly in this instance as it is traditionally an unsupervised learning algorithm. However, because we are feeding our model labels for the training data (in the form of "Survived"), our problem is one suited for Supervised Learning, Simply classifying the passengers based on their features without a reference label can place emphasis on classification leaning towards other traits and group characteristics over survivability (our label).  While I have seen other kernels pull off >80% predictions with KNN, I can't seem to replicate the results.  Several sites have also shown the KNN classifier to be effective for the titanic dataset, so if anyone knows why, I would love to hear your answer.
# 
# SVMs didn't do much better as well. SVMs have their foundations in a technique called the "kernel trick", an act of using a kernel to preserve linearity between variables and labels. However, most SVMs require further engineering of the kernel, and various techniques to choose the appropriate hyperparamters to allow for sufficient generalisation performance. 
# 
# If you are interested, feel free to learn more about the various ML models here : 
# 
# http://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf [](http://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf)
# 
# https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7 [](https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7)
# 
# A great book to read on ML/DL from a mathematicakl perspective : 
# https://www.deeplearningbook.org/ [](https://www.deeplearningbook.org/)
# 

# **THANKS FOR TUNING IN AND SEE YOU IN MY NEXT KERNEL! **

# In[ ]:





# In[ ]:





# 

# 
