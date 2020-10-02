#!/usr/bin/env python
# coding: utf-8

# # Import Data Plotting Modules and Dataset
# 

# In[ ]:


# Import Data Plotting Modules

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Import Data Processing Modules

import pandas as pd
import numpy as np


# In[ ]:


# Import Dataset and Show Head

df = pd.read_csv("../input/african_crises.csv")
df.head()


# In[ ]:


# Get column names

df.columns


# In[ ]:


## Define X and Y Variables

# Define Features: Ignore the 'case', 'cc3', 'country', 'year' variables

X = df[['systemic_crisis', 'exch_usd', 'domestic_debt_in_default',
       'sovereign_external_debt_default', 'gdp_weighted_default',
       'inflation_annual_cpi', 'independence', 'currency_crises',
       'inflation_crises']]


# Define the Y variable 

Y = df['banking_crisis']


# # Split Data into Training and Test Sets

# In[ ]:


from sklearn.model_selection import train_test_split

train_features, val_features, train_targets, val_targets = train_test_split(X, Y, 
                                                                            test_size=0.30, 
                                                                            random_state=7)


# # Construct Simple Baseline Model Using Sci-Kit Learn Random Forest Package

# In[ ]:


# Load ML Accuracy Calibration Packages

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[ ]:


# Import the Random Forest Classifier from Sklearn

from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# The class_weight Parameter is Set to "balanced" to Account for the Class Imbalance

ensemble = RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='gini',
                                  max_depth=100, max_features='auto', max_leaf_nodes=10,
                                  min_impurity_decrease=0.0, min_impurity_split=None,
                                  min_samples_leaf=1, min_samples_split=2,
                                  min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
                                  oob_score=False, random_state=0, verbose=0, warm_start=False)


# In[ ]:


# Fit the Model to the Training Set and Test Set

ensemble.fit(train_features, train_targets)

predicted = ensemble.predict(val_features)


# In[ ]:


# Print Accuracy Score

print('Accuracy achieved is: ' + str(np.mean(predicted == val_targets)))
print(metrics.classification_report(val_targets, predicted, target_names=("crisis", "no_crisis"))),
metrics.confusion_matrix(val_targets, predicted)


# # Understand the Inner Workings of the Model Using LIME Framework

# In[ ]:


# Step 1: Import the Modules for Model Interpretability

import lime
import lime.lime_tabular


# In[ ]:


# Step 2: Define Feature Names

feature_names = ['systemic_crisis', 'exch_usd', 'domestic_debt_in_default',
                 'sovereign_external_debt_default', 'gdp_weighted_default',
                 'inflation_annual_cpi', 'independence', 'currency_crises', 
                 'inflation_crises']


# In[ ]:


# Step 3: Define Class Names

class_names = Y.unique()


# In[ ]:


# Step 4: Change the Features in the Training Set to a Numpy Array

train_features.values


# In[ ]:


# Step 5: Create Your Explainer

explainer = lime.lime_tabular.LimeTabularExplainer(train_features.values, 
                                                   feature_names=feature_names,
                                                   class_names=class_names,
                                                   discretize_continuous=True)


# In[ ]:


# Step 6: Pick a Set of Features and Target Name to Explain

val_features.iloc[136], val_targets.iloc[136]


# In[ ]:


# Step 7: Build The Explainer Instance

exp = explainer.explain_instance(val_features.iloc[136], ensemble.predict_proba, num_features=9, top_labels=1)


# In[ ]:


exp


# In[ ]:


# Step 9: Show Explanation of In [17] in Notebook

exp.show_in_notebook(show_table=True, show_all=False)


# # Interpreting the Explanation

# In[ ]:


# The sample "val_features.iloc[136]" was assigned a 40% probability of belonging to the "crisis" class and a 60% probability 
# of belonging to the "no_crisis" class. This means that the model predicts, with a 60% probability, that 
# the sample "val_features.iloc[136]" belongs to the "no_crisis" class.

# The rank of the features, in terms of influence on the prediction, is (see table):
# 1) "systemic_crisis", 2) "inflation_annual_cpi", 3) "inflation_crises", 4) "exch_usd",
# 5) "sovereign_external_debt_default", 6) "domestic_debt_in_default", 7) "gdp_weighted_default", 8) "currency_crises"
# and 9) "independence". 

# In the table, the color: Orange means that the variable is associated with the "no_crisis" probability; 
# Blue means that the variable is associated with the "crisis" probability; 
# White means that the variable is of no consequence.

