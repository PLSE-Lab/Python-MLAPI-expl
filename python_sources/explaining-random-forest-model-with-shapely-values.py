#!/usr/bin/env python
# coding: utf-8

# # Objectives
# ### Hello Kaggler!, <span style="color:PURPLE">Objective of this short kernal is to</span> <span style="color:red">Interprete a Random Forest Model with Shapely Values method.</span>
# 
# To make this very easy to grasp I have used infamouse Titanic data set to train the ML model.
# 
# ### Additionally, after reading this Kernel I hope that you would 
# * Get an understanding How to use SHAP library for calculating Shapley values for a random forest classifier.
# * Get an understanding on how the model makes predictions using shapely values method.
# 
# 
# The intent here is not to build the best possible model but rather the focus is on the aspect of interpretability.

# # Using SHAP library for calculating Shapley values for a Random Forest Classifier

# ## Dataset
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


# ## Data Preparation
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


# ## Model Training
# 
# Let's train the RF model and get the accuracy measure.

# In[ ]:


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
random_forest_preds = random_forest.predict(X_test)
print('The accuracy of the Random Forests model is :\t',metrics.accuracy_score(random_forest_preds,Y_test))


# ## Interpreting the Model With Shapely Values
# 
# ### 1. Import SHAP package

# In[ ]:


import shap 


# ### 2. Create the Explainer

# In[ ]:


# Create Tree Explainer object that can calculate shap values
explainer = shap.TreeExplainer(random_forest)


# ### 3. Use the explainer to explain predictions
# 

# #### Calculate Shap values example 1 

# In[ ]:


#Let's choose some instances from the test dataset to understand to the classifier makes predictions for them.
test.loc[[421]]


# Choosen instance refers to an **Unlucky (not survived) Male passenger of age 21 travelling in passenger class 3, embarked from Q.** Let's see what and how our model predicts his survival.

# In[ ]:


# Calculate Shap values
choosen_instance = X_test.loc[[421]]
shap_values = explainer.shap_values(choosen_instance)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)


# ##### Interpretation :
# What you see above is a force plot visualizing shapley values for the features. Feature values in pink cause to increase the prediction. Size of the bar shows the magnitude of the feature's effect. Feature values in blue cause to decrease the prediction. Sum of all feature SHAP values explain why model prediction was different from the baseline.
# 
# Model predicted 0.16 (Not survived), whereas the base_value is 0.3793. Biggest effect is person being a male; This has decreased his chances of survival significantly. Next, passenger class 3 also decreases his chances of survival while being 21 and port of embarkation beign S increases his chances of survival.

# ---
# #### Calculate Shap values example 2

# In[ ]:


#Let's choose some instances from the test dataset to understand to the classifier makes predictions for them.
test.loc[[310]]


# Choosen instance refers to an **Survived female passenger of age 24 travelling in passenger class 1, embarked from C.** Let's see what and how our model predicts her survival.

# In[ ]:


# Calculate Shap values
choosen_instance = X_test.loc[[310]]
shap_values = explainer.shap_values(choosen_instance)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)


# ##### Interpretation :
# 
# Model predicted 1 (Fully confident that passenger survives), whereas the base_value is 0.3793. Biggest effect is person being a female; This has increased her chances of survival significantly. Next, passenger class 1 also increases her chances of survival.

# ---
# #### Calculate Shap values example 3

# In[ ]:


#Let's choose some instances from the test dataset to understand to the classifier makes predictions for them.
test.loc[[736]]


# Choosen instance refers to an **Unlucky(Not Survived) female passenger of age 48 travelling in passenger class 3, embarked from S.** Let's see what and how our model predicts her survival.

# In[ ]:


# Calculate Shap values
choosen_instance = X_test.loc[[736]]
shap_values = explainer.shap_values(choosen_instance)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)


# ##### Interpretation :
# 
# Model predicted 0.42, whereas the base_value is 0.3793. Biggest effect is person being a female; This has increased her chances of survival significantly. Fare value of 34.38 has also played a part incresing her chances. However, beign a passenger in class 3 and her age (48) has significantly decreased her chances of survival.

# ---
# #### Calculate Shap values example 4

# In[ ]:


#Let's choose some instances from the test dataset to understand to the classifier makes predictions for them.
test.loc[[788]]


# Choosen instance refers to an **Survived male passenger of age 1 travelling in passenger class 3, embarked from S.** Let's see what and how our model predicts his survival.

# In[ ]:


# Calculate Shap values
choosen_instance = X_test.loc[[788]]
shap_values = explainer.shap_values(choosen_instance)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)


# ##### Interpretation :
# 
# Model predicted 0.66, whereas the base_value is 0.3793. Although passenger class 3 and being a male passenger has decresed his chances of survival. Biggest effect has come from his age being 1 years old; This has increased his chances of survival significantly.

# ---

# In[ ]:


shap.summary_plot(shap_values, X_train)


# ### Credits
# 
# * https://shap.readthedocs.io/en/latest/
# * https://christophm.github.io/interpretable-ml-book/intro.html
# * https://www.kaggle.com/dansbecker/shap-values#Code-to-Calculate-SHAP-Values

# # Thank you!
# 
# ### **If you like the notebook and think that it helped you..PLEASE UPVOTE. It will keep me motivated** :) :)
