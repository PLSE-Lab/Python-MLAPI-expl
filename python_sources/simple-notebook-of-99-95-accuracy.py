#!/usr/bin/env python
# coding: utf-8

# Here in this module I'm working on two classification models : 
# *    Decision Tree Classifier
# *    Ensemble Learning via Random Forest Classifier
#    
# to predict Anonymized credit card transactions as fraudulent or genuine. 
# 
# In order to do so I have done my data Processing : 
# *    Handled Missing values : Using mean
# *    Standarised the data : Using Standardisation
#         
# Then I applied my two models, which works really well - 
# *   Decision Tree Accuracy    :  99.9227549788
# *   Random Forest Accuracy    :  99.9550574422   -  A better approach to follow

# In[ ]:


# Loading libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# Loading dataset

dataset = pd.read_csv('../input/creditcard.csv')


# In[ ]:


dataset.head()


# In[ ]:


# Parameters and results

x = dataset.iloc[: , 1:30].values
y = dataset.iloc[:, 30].values


# In[ ]:


print("Parameters : \n ", x)


# In[ ]:


print("Results : \n ", y)


# In[ ]:


# Handling Missing Values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:, 1:30])
x[:, 1:30] = imputer.fit_transform(x[:, 1:30])


# In[ ]:


# Splitting the data-set

from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state = 0)


# In[ ]:


print("xtrain.shape : ", xtrain.shape)
print("xtest.shape  : ", xtest.shape)
print("ytrain.shape : ", ytrain.shape)
print("xtest.shape  : ", xtest.shape)

print("\nxtrain  : \n", xtrain)
print("\n\nxtest : \n", xtest)
print("\nytrain  : \n", xtrain)
print("\nytest   : \n", xtrain)


# In[ ]:


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)


# In[ ]:


print("Standardised Training Set : \n\n", xtrain)


# # Decision Tree Classification Model

# In[ ]:


# Fitting Decision Tree Classification to the Training set

from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(xtrain, ytrain)


# In[ ]:


# Predicting the Test set results

y_pred_decision_tree = dt_classifier.predict(xtest)


# In[ ]:


print("y_pred_decision_tree : \n", y_pred_decision_tree)


# In[ ]:


# Making the Confusion Matrix to validate the Decision Tree Model

from sklearn.metrics import confusion_matrix
cm_decision = confusion_matrix(ytest, y_pred_decision_tree)


# In[ ]:


cm_decision


# In[ ]:


# Validating the Prediction
Accuracy_Decison = ((cm_decision[0][0] + cm_decision[1][1]) / cm_decision.sum()) *100
print("Accuracy_Decison    : ", Accuracy_Decison)

Error_rate_Decison = ((cm_decision[0][1] + cm_decision[1][0]) / cm_decision.sum()) *100
print("Error_rate_Decison  : ", Error_rate_Decison)

# True Fake Recognition Rate
Specificity_Decison = (cm_decision[1][1] / (cm_decision[1][1] + cm_decision[0][1])) *100
print("Specificity_Decison : ", Specificity_Decison)

# True Genuine Recognition Rate
Sensitivity_Decison = (cm_decision[0][0] / (cm_decision[0][0] + cm_decision[1][0])) *100
print("Sensitivity_Decison : ", Sensitivity_Decison)


# # Ensemble Learning : Random Forest Classification Model

# In[ ]:


# Fitting Random Forest Model to the dataset

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, criterion = 'entropy', random_state = 0 )
rf_classifier.fit(xtrain, ytrain)


# In[ ]:


# Predicting the Test set results

y_pred_randomforest = rf_classifier.predict(xtest)


# In[ ]:


print("y_pred_randomforest : \n", y_pred_randomforest)


# In[ ]:


# Making the Confusion Matrix to validate the Random Forest Model

from sklearn.metrics import confusion_matrix
cm_rainforest = confusion_matrix(ytest, y_pred_randomforest)


# In[ ]:


# Validating the Prediction
Accuracy_rainforest = ((cm_rainforest[0][0] + cm_rainforest[1][1]) / cm_rainforest.sum()) *100
print("Accuracy_rainforest    : ", Accuracy_rainforest)

Error_rate_rainforest = ((cm_rainforest[0][1] + cm_rainforest[1][0]) / cm_rainforest.sum()) *100
print("Error_rate_rainforest  : ", Error_rate_rainforest)

# True Fake Recognition Rate
Specificity_rainforest = (cm_rainforest[1][1] / (cm_rainforest[1][1] + cm_rainforest[0][1])) *100
print("Specificity_rainforest : ", Specificity_rainforest)

# True Genuine Recognition Rate
Sensitivity_rainforest = (cm_rainforest[0][0] / (cm_rainforest[0][0] + cm_rainforest[1][0])) *100
print("Sensitivity_rainforest : ", Sensitivity_rainforest)


# Since I have already used the Standardisation of dataset - using StandardScaler. 
# That's why I used Confusion Matrix to check the Accuracy of my model :)
