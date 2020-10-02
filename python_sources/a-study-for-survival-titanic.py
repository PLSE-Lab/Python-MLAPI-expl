#!/usr/bin/env python
# coding: utf-8

# # A Study for Survival. Titanic
# ### ** Ifeanyi N. **
# 30.04.2020
# 
# 1. ** 1. Introduction **
# 2. ** Load data **   
# 2.1 Load the training and test dataset.        
# 2.2 The data
# 
# 3. ** Exploratory data analysis **   
# 3.1 Missing data     
# 3.2 Data cleaning       
# 3.3 Categorical features        
# 4. ** Model **   
# 4.1 Split into training and test set   
# 4.2 Building the model
# 5. ** Prediction **
# 6. ** Conclusion **   
# #### ** Reference **
# 

# # ** Introduction **
# I am a newbie to machine learning and this is my first kaggle kernel. I chose the Titanic dataset to show some basic concept in data science and machine learning. First, we will visually expore the Titanic dataset and then some data cleaning with missing values imputation. Finally, we will train a model to predict survival on the Titanic.
# 
# 
# There are three parts to my notebook as follows:
# 
# * **Exploratory data analysis**
# * **Data cleaning and missing value imputation**
# * **Models and prediction**

# # ** 2. Load data **
# 
# Let's start by reading the training and test dataset.

# In[ ]:


# Import important libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
cf.go_offline()


# ### 2.1 Load trainig and test dataset

# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
pid = test["PassengerId"]


# ### 2.2 The Data 
# 
# Explore the structure and summary of our data set to try and understand the data and how to proceed.

# In[ ]:


train.info()
train.describe()


# # ** 3. Exploratory data analysis **
# 
# Exploratory analysis to help visualise the relationship or patterns in the data. Let's begin with missing values.

# ### 3.1 Missing data 
# 
# Using seaborn to create a heatmap to see where we are missing data.
# 

# In[ ]:


sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# We can see clearly that there are two columns with missing data, the 'Age' column has about 20% missing data which is small for some reasonable replacement through imputation. The 'Cabin' column has a lot of missing data and we might have to drop the column beacause it would be difficult to do something useful with it, 
# We continue with more exploratory analysis.

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x = 'Survived', data = train)


# Looks like we have more passengers that didn't survive than passengers that did survive in our dataset.

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x = 'Survived', hue = 'Sex', data = train)


# We can see a kind of trend here. Looks like passengers that did not survive are much more likely to be male and passengers that survived are likely to be female.

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x = 'Survived', hue = 'Pclass', data = train)


# In[ ]:


sns.barplot("Pclass", "Survived", data = train)


# In[ ]:


sns.barplot("Embarked", "Survived", data = train)


# Looking at the passengers that survived based on passenger class, it looks like passengers that did not survive tends to be from the third class or the cheapest class.
# 
# 

# In[ ]:


sns.distplot(train['Age'].dropna(), kde = False, bins = 30)

Looks like the 'Age' is skewed toward younger passengers of average of about 20 - 30.
# In[ ]:


sns.countplot(x = 'SibSp', data = train)


# This plot shows that most passengers in the dataset do not have children or spouse on board.

# In[ ]:


sns.barplot("SibSp", "Survived", data = train)


# In[ ]:


sns.barplot("Parch", "Survived", data = train)


# In[ ]:


train['Fare'].iplot(kind = 'hist', bins = 30, color = 'blue')


# We look also at the 'Fare' column. This plot indicates that most of the purchase range from 0 - 80. This makes sense because we have already seen that most passengers are in the third class of the 'Pclass' column. 

# ### 3.2 Data Cleaning 
# 
# Through our exploration, we found out we had some missing data. Cleaning the dataset and cleaning the data would help us for our machine learning algorithms.
# 

# In[ ]:


sns.boxplot(x = 'Pclass', y = 'Age', data = train)


# This plot shows when 'Age' is seperated by class ('Pclass'), the first class and second class tends to be older than the third class. We use this simple information to predict the missing values of 'Age' based on 'Pclass'.
# 

# In[ ]:


# We create a function to get the avg. age based on Pclass

def pred_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 38

        elif Pclass == 2:
            return 29

        else:
            return 25

    else:
        return Age
    
train['Age'] = train[['Age','Pclass']].apply(pred_age, axis = 1)


# In[ ]:


# We create a function to get the avg. age based on Pclass

def pred_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 40

        elif Pclass == 2:
            return 28

        else:
            return 24

    else:
        return Age
    
test['Age'] = test[['Age','Pclass']].apply(pred_age, axis = 1)


# In[ ]:


sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# In[ ]:


sns.heatmap(test.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# The missing values in 'Age' has been filled as shown in the plot. 

# The family column can be represented by one column instead of having two columns Parch & SibSp, we can have only one column to represent if the passenger was alone or had any family member aboard or not. Meaning, if having any family member will increase chances of Survival or not.

# In[ ]:



family = train["SibSp"] + train["Parch"]      # combining the siblings and family data
train["Alone"] = np.where(family >  0, 0, 1)  # If a passenger is traveling alone or with a family member

fam = test["SibSp"] + test["Parch"]
test["Alone"] = np.where(fam >  0, 0, 1)


# We drop some columns we do not need for our model.

# In[ ]:


train.drop(['PassengerId', 'Ticket', 'Cabin', 'Name', 'SibSp', 'Parch'], axis = 1, inplace = True)
test.drop(['PassengerId', 'Ticket', 'Cabin', 'Name', 'SibSp', 'Parch'], axis = 1, inplace = True)


# ### 3.3 Categorical Features 
# Using pandas, we can convert categorical features into dummy variables.

# In[ ]:


train.info()


# In[ ]:


sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# The 'Embarked' column has two missing values which is easy to fill in.
# 

# In[ ]:


sns.countplot(x = 'Embarked', data = train)


# The plot above shows that passengers are more likely to embark in Southampton (S). We use S to fill the missing data in the Embarked column

# In[ ]:


train.fillna({"Embarked": "S"}, inplace = True)


# In[ ]:


sns.heatmap(test.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# In[ ]:


test["Fare"].fillna(test["Fare"].median(), inplace = True)


# We create dummie variables for our categorical variables, however, we drop the first columns of our dummie variables to avoid the problem of multicollinearity.

# In[ ]:


train = pd.get_dummies(train, columns = ['Embarked', 'Sex', 'Pclass'], drop_first = True)
test = pd.get_dummies(test, columns = ['Embarked', 'Sex', 'Pclass'], drop_first = True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# # ** 4. Model **
# 
# Finally, we are ready to train our model and do some predictions. We will use four different classification techniques to approach this - 
# 
# * Logistic Regression
# * K Nearest Neighbors
# * Support Vector Machines
# * Random Forest

# ### 4.1 Split into training and test set

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


X = train.drop("Survived",axis = 1)
Y = train["Survived"]
#X_test = test


# The dataset is split into training dataset (80%) and test dataset (20%).
# 

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# ### 4.2 Building the Model

# In[ ]:


# Logistic Regression

logmod = LogisticRegression(max_iter = 200)

logmod.fit(X_train,Y_train)

pred = logmod.predict(X_test)

acc_logmod = round(accuracy_score(Y_test, pred) * 100, 2)
print(acc_logmod)


# In[ ]:


# K Nearest Neighbors

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, Y_train)

knn_pred = knn.predict(X_test)

acc_knn = round(accuracy_score(Y_test, knn_pred) * 100, 2)

print(acc_knn)
#knn.score(X_train, Y_train)


# In[ ]:


# Support Vector Machines


param_grid = {"C": [0.1, 1, 10, 100, 1000], "gamma": [1, 0.1, 0.01, 0.001, 0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose = 3)

grid.fit(X_train, Y_train)

grid_pred = grid.predict(X_test)

acc_svc = round(accuracy_score(Y_test, grid_pred) * 100, 2)
print(acc_svc)
#print(grid.score(X_train, Y_train))


# In[ ]:


# Random Forest


random_forest = RandomForestClassifier(n_estimators = 150, random_state = 5)

random_forest.fit(X_train, Y_train)

RF_pred = random_forest.predict(X_test)

acc_RF = round(accuracy_score(Y_test, RF_pred) * 100, 2)
print(acc_RF)
#random_forest.score(X_train, Y_train)


# ### 4.3 Model evaluation
# 
# We rank our four classifiers and choose the classifier with the better accuracy. For our prediction, we use the random forest model.

# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'Support Vector Machines', 'Random Forest'],
    'Score': [acc_logmod, acc_knn, acc_svc, acc_RF]})

models.sort_values(by='Score', ascending = False)


# We will use the Random Forest model for our predictions.

# # ** 5. Prediction **

# In[ ]:


predict = random_forest.predict(test)

submit = pd.DataFrame({
        "PassengerId": pid,
        "Survived": predict})

submit.to_csv('submit.csv', index=False)


# # ** Conclusion **
# 
# Thank you for reading. I look forward to doing more with more details. Please feel free to comment and give suggestions.
# 

# # ** Reference **
# 
# [Exploring Survival on the Titanic](http://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic)   
# [A Journey Through Titanic](https://www.kaggle.com/omarelgabry/a-journey-through-titanic)    
# [Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)   
