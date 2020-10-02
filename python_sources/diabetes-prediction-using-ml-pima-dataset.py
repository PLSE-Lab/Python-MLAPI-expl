#!/usr/bin/env python
# coding: utf-8

# # Diabetes Onset Prediction using Pima Indians Diabetes Dataset

# **This dataset is an interesting one and aims at predicting the probability of the onset of diabetes. I, as an Indian was interested in this dataset and this is my take on this dataset.**

# *** (1). Importing the necessary libraries***

# In[47]:


#IMPORTING THE LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import binarize
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# ***(2). Loading and visualizing the dataset***

# In[48]:


#LOADING THE DATASET
dataset = pd.read_csv('../input/diabetes.csv')
dataset.head()


# In[49]:


#DESCRIPTION OF THE DATASET
dataset.describe()


# **Details about the dataset:**
# 1. Pregnancies decribes the number of times the person has been pregnant.
# 2. Gluose describes the blood glucose level on testing.
# 3. Blood pressure describes the diastolic blood pressure.
# 4. Skin Thickenss describes the skin fold thickness of the triceps.
# 5. Insulin describes the amount of insulin in a 2hour serum test.
# 6. BMI describes he body mass index.
# 7. DiabetesPedigreeFunction describes the family history of the person.
# 8. Age describes the age of the person
# 9. Outcome describes if the person is predicted to have diabetes or not.
# * It should also be noted that the dataset has no missing values and thus, filling up the dataset using algorithms will not be necessary.

# ***(3).Selection and Splitting of the Dataset***

# In[51]:


data = dataset.iloc[:,0:8]
outcome = dataset.iloc[:,8]
x,y = data,outcome


# In[52]:


#DISTRIBUTION OF DATASET INTO TRAINING AND TESTING SETS
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)


# In[53]:


#COUNTING THE POSITIVE AND NEGATIVE VALUES
print(y_test.value_counts())
#MEAN OF THE TESTING DISTRIBUTION
print(1- y_test.mean())


# ***(4). Machine Learning - Using GRID SEARCH CROSS VALIDATION for evaluating the best parameters***

# In[ ]:


#PARAMETER EVALUATION WITH GSC VALIDATION
gbe = GradientBoostingClassifier(random_state=0)
parameters={
    'learning_rate': [0.05, 0.1, 0.5],
    'max_features': [0.5, 1],
    'max_depth': [3, 4, 5]
}
gridsearch=GridSearchCV(gbe,parameters,cv=100,scoring='roc_auc')
gridsearch.fit(x,y)
print(gridsearch.best_params_)
print(gridsearch.best_score_)


# ***(5). Adjusting improvement threshold***

# In[9]:


#ADJUSTING DEVELOPMENT THRESHOLD
gbi = GradientBoostingClassifier(learning_rate=0.05,max_depth=3,max_features=0.5,random_state=0)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
gbi.fit(x_train,y_train)


# In[10]:


#STORING THE PREDICTION
yprediction = gbi.predict_proba(x_test)[:,1]


# In[11]:


#PLOTTING THE PREDICTIONS
plt.hist(yprediction,bins=10)
plt.xlim(0,1)
plt.xlabel("Predicted Proababilities")
plt.ylabel("Frequency")


# ***(6). Score of the Gradient Boosting Classifier***

# In[45]:


#CLASSIFIER SCORE
round(roc_auc_score(y_test,yprediction),5)


# ***(7). Score of Random Forest Classifier***

# In[44]:


#USING RANDOM FOREST CLASSIFICATION
from sklearn.ensemble import RandomForestClassifier
rmfr = RandomForestClassifier()
rmfr.fit(x_train, y_train)
y_pred = rmfr.predict(x_test)
accuracyrf = round(accuracy_score(y_pred, y_test), 5)
accuracyrf


# ***(8). Score of the XGBoost Classifier ***

# In[46]:


from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = round(accuracy_score(y_test, predictions),5)
accuracy


# # Thus, we can see that the Gradient Bossting Classifier gives the best performance with 85.24%, followed by XGBoost with 79.68% and Random Forest with 78.1%
# # I tried the same with other classifiers like Perceptron, KNN, Naive Bayes and Decision Trees but I have displayed the top 3 performers.
# # If anyone wants, I can display all the algorithms I used.
# # If there any suggestions please comment!

# 
