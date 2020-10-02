#!/usr/bin/env python
# coding: utf-8

# I briefly summarized [DanB](https://www.kaggle.com/dansbecker)'s great lessons about machine learning explainability for my study.  
# This is **[Course Home Page](https://www.kaggle.com/learn/machine-learning-explainability)**.

# # What is machine learning explainability?
# 
# - an ability to explain why machine learning model predict that results
# - an ability to show feature importance of machine learning model

# # Why is machine learning explainability needed?
# 
# - Machine learnig model tend to be a black box compared with statistical model.
# - If model can't explain why it output this result, users don't want to use ML model, especially for the case which is related to people's life.
#     - If it's not related to people's life, users want to know why ML model output this result in many cases.
# - Data scientists want to know how model predict in order to enhance and debug model.

# # Methods to explain ML model
# 
# I introduce three methods to explain machine learning model following DanB's lessons.
# 
# - permutation importance
# - partial plots
# - SHAP values

# # Permutation Importance
# [DanB's permutation importance lesson](https://www.kaggle.com/dansbecker/permutation-importance)
# 
# 
# ## Overview
# 
# - Permutation importance is a method to confirm which feature have the biggest impact on predictions.
# 
# ## How it works
# 
# 1. Prepare trained model
# 1. Shuffle one feature like below and make prediction. Calculate accuracy using prediction.
# 1. repeat 2. for every feature
#     - accuracy is decreased significantly -> shuffled feature is much important.
#     - accuracy is almost same before shuffling -> shuffled feature is not important.
# 
# ![Shuffle](https://i.imgur.com/h17tMUU.png)

# ## Code example
# 
# In this kernel, I used a model that predict whether a soccer team will have the "Man of the Game" winner based on the team's statistics.  
# I used code in this [DanB's kernel](https://www.kaggle.com/dansbecker/permutation-importance) 

# In[ ]:


# make a simple model to show ML explainability

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../input/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)


# In[ ]:


# show permutation importance

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())


# ## When do we use permutation importance?
# 
# - For shortening time to make model and prediction by removing less important features.
# - You can confirm that whether your feature has a big impact on predictions.

# # Partial Plots
# [DanB's Partial Plots lesson](https://www.kaggle.com/dansbecker/partial-plots)
# 
# ## Overview
# 
# - Partial plots is a method to confirm the change of predictions by altering one feature's value gradually
# 
# ## How it works
# 
# 1. Prepare a trained model
# 1. Alter one feature's value gradually, and plot prediction changes

# ## Code example
# I used code in this [DanB's kernel ](https://www.kaggle.com/dansbecker/partial-plots) (I changed some variable names)

# In[ ]:


from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=feature_names, feature='Goal Scored')

# plot it
pdp.pdp_plot(pdp_goals, 'Goal Scored')
plt.show()


# In[ ]:


feature_to_plot = 'Distance Covered (Kms)'
pdp_dist = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=feature_names, feature=feature_to_plot)

pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()


# You can check interactions between features from 2D partial plots.

# In[ ]:


# Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot
features_to_plot = ['Goal Scored', 'Distance Covered (Kms)']
inter1  =  pdp.pdp_interact(model=my_model, dataset=val_X, model_features=feature_names, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()


# ## When do we use partial plots?
# 
# - You can find out how much detail we should collect data from partial plots.(It costs collecting detailed data...)
#     - For example, we found out that we don't have to collect "Goal Scored" data in detail. "No Goal" and "More than one goal" may be enough, because prediction do not change whether a team get 1 goal  or 2 goals.
# - You can confirm whether your feature has a big impact on predictions.
#     - If partial plots doesn't change, you can find out this feature is not important.

# # SHAP values
# [DanB's SHAP values lesson](https://www.kaggle.com/dansbecker/shap-values)
# 
# ## Overview
# 
# - SHAP values shows that which feature has a big impact on predictions for individual predictions.

# ## Code Examples
# 
# I used code in this [DanB's kernel](https://www.kaggle.com/dansbecker/shap-values)

# In[ ]:


# extract single row of the dataset
row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)


my_model.predict_proba(data_for_prediction_array)


# In[ ]:


import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)

shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)


# ## When do we use SHAP values?
# 
# - You can use SHAP values for making decision, especially for the case which is related to people's life.
# - You can show why your model output this result to users.
# - You can use SHAP values to improve your model, because you can find out why your model made a wrong decision.

# Thanks for reading my kernel.  
# If you have any questions, please ask me.  
# If you feel my English is weird, please give me some comments.  
