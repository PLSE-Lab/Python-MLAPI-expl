#!/usr/bin/env python
# coding: utf-8

# This is my first public Notebook. I'm still learning all the in-and-outs of ML and creating/sharing notebooks.
# All constructive feedback would be greatly appreciated.

# # Data discovery

# Let's first load the data into a Pandas Dataframe

# In[ ]:


import numpy as np
import pandas as pd

random_seed = 297

import os
clinical_data_filepath = "../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv"
clinical_data = pd.read_csv(clinical_data_filepath)


# Using Pandas Profiling we analyze the data to check if all data is nicely behaving, and already check for possible correlations with the predictor

# In[ ]:


import pandas_profiling
pandas_profile = pandas_profiling.ProfileReport(clinical_data)


# In[ ]:


pandas_profile.to_widgets()


# # Data engineering

# Based on the pandas profiling, we have several boolean columns which are not yet recognized as such (for example anaemia). We will convert these to booleans.
# In addition, we can already see that "Time" seems to possive a significant negative correlation with "Death_event". We expect this will serve as a good predictor compared to the other features.

# In[ ]:


clean_data = clinical_data.astype({'anaemia': 'bool', 'diabetes': 'bool', 'high_blood_pressure':'bool', 'smoking':'bool'})
clean_data.head()


# Now we'll separate into a training/test set

# In[ ]:


from sklearn.model_selection import train_test_split

# Split into X and y dataset
y = clean_data.DEATH_EVENT
all_features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes','ejection_fraction', 'high_blood_pressure', 'platelets','serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
features = all_features #For this initial version, we'll just take all features for input. Due to the low amount of samples compared to features, we do fear for overfitting
X = clean_data[features]

# Create train/test set
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=random_seed)


# # Models
# 
# We'll use a RandomForestClassifier for the actual prediction model. We propose this model due to the low sample size and several boolean features (which mean "easy" yes/no decision leaves).
# For showing the importance of each feature we'll also train a XGBoost model and plot the importance of each of the features.

# # RandomForestClassifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
# Specify Model
hearth_failure_RFCmodel = RandomForestClassifier(random_state=random_seed)
# Fit Model
hearth_failure_RFCmodel.fit(train_X, train_y)


# In[ ]:


from sklearn.metrics import plot_roc_curve
plot_roc_curve(hearth_failure_RFCmodel, val_X, val_y)


# # XGBoost

# In[ ]:


import xgboost

# Specify Model
hearth_failure_XGBmodel = xgboost.XGBClassifier(random_state=random_seed)
# Fit Model
hearth_failure_XGBmodel.fit(train_X, train_y)


# In[ ]:


from sklearn.metrics import plot_roc_curve
plot_roc_curve(hearth_failure_XGBmodel, val_X, val_y)


# Although the XGBoost model performs slightly worse compared to the RandomForest (I assume due to overfitting considering the small amount of samples), let's use it to take a look at the importance of the different features.
# Although there are several downsides to this importance metric and it might be better to use SHAP values (see [TDS SHAP article](https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d)), it does give us an easy quick overview.

# In[ ]:


xgboost.plot_importance(hearth_failure_XGBmodel)


# As you can see, this is aligned with our initial thinking that "Time" has a big correlation with "Death_event" and is hence also used as the primary predictor in the model.
