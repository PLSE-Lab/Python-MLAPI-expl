#!/usr/bin/env python
# coding: utf-8

# # Explaining Random Forest Model With LIME
# 
# ### Hello Kaggler!, <span style="color:PURPLE">Objective of this short kernal is to</span> <span style="color:red">Interprete a Random Forest Model with LIME method.</span>
# 
# To make this very easy to grasp I have used infamouse Titanic data set to train the ML model.
# 
# ### Additionally, after reading this Kernel I hope that you would 
# * Get an understanding How to use LIME library for a random forest classifier.
# * Get an understanding on how the model makes predictions using LIME method.
# 
# 
# The intent here is not to build the best possible model but rather the focus is on the aspect of interpretability.

# # Content
# 1. [Dataset](#-1)
# 1. [Data Preparation](#0)
# 1. [Model Training](#1)
# 1.  [Interpreting the Model With Shapely Values](#2)
#     1. [Import LIME package](#2_1)
#     1. [Create the Explainer](#2_2)
#     1. [Use the explainer to explain predictions](#2_3)
#         1. [Explaining Instance 1](#2_3_1)
#         1. [Explaining Instance 2](#2_3_2)
#         1. [Explaining Instance 3](#2_3_3)
#         1. [Explaining Instance 4](#2_3_4)
# 1. [Credits](#3)

# # Using LIME Package for understanding a Random Forest Classifier

# ## Dataset[^](#-1)<a id="-1" ></a><br>
# 
# For the demonstration I have used infamouse [**Titanic dataset**](https://www.kaggle.com/c/titanic) to train the Random Forest Classifier. Objective of this problem to predict the survival for passengers in Titanic.

# In[ ]:


# Importing Required Libraries
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# In[ ]:


data = pd.read_csv("../input/train.csv")
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])


# In[ ]:


train.head()


# ## Data Preparation [^](#0)<a id="0" ></a><br>
# let's get the datasets ready to put into training. Since our goal is not to make a better classiefier for the problem, let's train a simple model.
# 
# Following steps are carried out in the following code block.
# 1. Dropping unneeded Features
#     * Lets only use Pclass, Sex, Age, SibSp, Parch and Embarked features.
# 2. Convert categorical variables into dummy/indicator variables.
#     * Pclass, Sex and Embarked features needs to be converted.
# 3. Filling Null Values
# 4. Create X_train, Y_train, X_test, Y_test datasets

# In[ ]:


# Dropping Features
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)

train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)

train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)

train = train.drop(['PassengerId'], axis=1)
test = test.drop(['PassengerId'], axis=1)

# Convert categorical variables into dummy/indicator variables
train_processed = pd.get_dummies(train)
test_processed = pd.get_dummies(test)

# Filling Null Values
train_processed = train_processed.fillna(train_processed.mean())
test_processed = test_processed.fillna(test_processed.mean())

# Create X_train,Y_train,X_test
X_train = train_processed.drop(['Survived'], axis=1)
Y_train = train_processed['Survived']

X_test  = test_processed.drop(['Survived'], axis=1)
Y_test  = test_processed['Survived']

# Display
print("Processed DataFrame for Training : Survived is the Target, other columns are features.")
display(train_processed.head())


# ## Model Training[^](#0)<a id="1" ></a><br>
# 
# Let's train the RF model and get the accuracy measure.

# In[ ]:


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
random_forest_preds = random_forest.predict(X_test)
print('The accuracy of the Random Forests model is :\t',metrics.accuracy_score(random_forest_preds,Y_test))


# ## Interpreting the Model With Shapely Values[^](#2)<a id="2" ></a><br>
# 
# ### 1. Import LIME package[^](#2_1)<a id="2_1" ></a><br>

# In[ ]:


import lime
import lime.lime_tabular


# ### 2. Create the Explainer[^](#2_2)<a id="2_2" ></a><br>

# In[ ]:


predict_fn_rf = lambda x: random_forest.predict_proba(x).astype(float)
X = X_train.values
explainer = lime.lime_tabular.LimeTabularExplainer(X,feature_names = X_train.columns,class_names=['Will Die','Will Survive'],kernel_width=5)


# ### 3. Use the explainer to explain predictions[^](#2_3)<a id="2_3" ></a><br>

# #### Explaining Instance 1[^](#2_3_1)<a id="2_3_1" ></a><br>

# In[ ]:


test.loc[[421]]


# Choosen instance refers to an Unlucky (not survived) **Male passenger of age 21 travelling in passenger class 3, embarked from Q**. Let's see what and how our model predicts his survival.

# In[ ]:


choosen_instance = X_test.loc[[421]].values[0]
exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)
exp.show_in_notebook(show_all=False)


# ###### Interpretation :
# 
# Model predicted Will Die (Not survived). Biggest effect is person being a male; This has decreased his chances of survival significantly. Next, passenger class 3 also decreases his chances of survival while being 21 increases his chances of survival.

# ---
# ### Explaining Instance 2[^](#2_3_2)<a id="2_3_2" ></a><br>
# 

# In[ ]:


test.loc[[310]]


# Choosen instance refers to an **Survived female passenger of age 24 travelling in passenger class 1, embarked from C**. Let's see what and how our model predicts her survival.

# In[ ]:


choosen_instance = X_test.loc[[310]].values[0]
exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)
exp.show_in_notebook(show_all=False)


# ###### Interpretation :
# Model predicted 1 (Fully confident that passenger survives). Biggest effect is person being a female; This has increased her chances of survival significantly. Next, passenger class 1 and Fare>31 has also increases her chances of survival.

# ---
# ### Explaining Instance 3[^](#2_3_3)<a id="2_3_3" ></a><br>

# In[ ]:


test.loc[[736]]


# Choosen instance refers to an **Unlucky(Not Survived) female passenger of age 48 travelling in passenger class 3, embarked from S**. Let's see what and how our model predicts her survival.

# In[ ]:


choosen_instance = X_test.loc[[736]].values[0]
exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)
exp.show_in_notebook(show_all=False)


# ###### Interpretation :
# Model predicted Will Die. Biggest effect is person being a female; This has increased her chances of survival significantly. Fare value of 34.38 has also played a part incresing her chances. However, beign a passenger in class 3 and her age (48) has significantly decreased her chances of survival.

# ---
# ### Explaining Instance 4[^](#2_3_4)<a id="2_3_4" ></a><br>

# In[ ]:


test.loc[[788]]


# Choosen instance refers to an **Survived male passenger of age 1 travelling in passenger class 3, embarked from S**. Let's see what and how our model predicts his survival.

# In[ ]:


choosen_instance = X_test.loc[[788]].values[0]
exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)
exp.show_in_notebook(show_all=False)


# ###### Interpretation :
# Model predicted Will Survive. Although passenger class 3 and being a male passenger has decresed his chances of survival. Biggest effect has come from his age being 1 years old; This has increased his chances of survival significantly.

# ---
# ## Credits[^](#3)<a id="3" ></a><br>
# * https://www.analyticsvidhya.com/blog/2017/06/building-trust-in-machine-learning-models/
# * https://christophm.github.io/interpretable-ml-book/intro.html
# * https://blog.dominodatalab.com/shap-lime-python-libraries-part-2-using-shap-lime/

# ---
# # Thank you!
# ## If you like the notebook and think that it helped you..PLEASE UPVOTE. It will keep me motivated :) :)
