#!/usr/bin/env python
# coding: utf-8

# There are many different method's to select the important features from a dataset. In this notebook I will show a quick way to select important features with the use of Boruta.
# 
# Boruta tries to find all relevant features that carry information to make an accurate classification. You can read more about Boruta [here](http://danielhomola.com/2015/05/08/borutapy-an-all-relevant-feature-selection-method/)
# 
# Let's start by doing all necessary imports.

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


# Next we load only the 'application_train' data as this is to demonstrate Boruta only. 

# In[ ]:


train = pd.read_csv("../input/application_train.csv")
train.shape


# All categorical values will be one-hot encoded.

# In[ ]:


train = pd.get_dummies(train, drop_first=True, dummy_na=True)
train.shape


# Get all feature names from the dataset

# In[ ]:


features = [f for f in train.columns if f not in ['TARGET','SK_ID_CURR']]
len(features)


# Replace all missing values with the Mean.

# In[ ]:


train[features] = train[features].fillna(train[features].mean()).clip(-1e9,1e9)


# Get the final dataset *X* and labels *Y*

# In[ ]:


X = train[features].values
Y = train['TARGET'].values.ravel()


# Next we setup the *RandomForrestClassifier* as the estimator to use for Boruta. The *max_depth* of the tree is advised on the Boruta Github page to be between 3 to 7.

# In[ ]:


rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)


# Next we setup Boruta. It uses the *scikit-learn* interface as much as possible so we can use *fit(X, y), transform(X), fit_transform(X, y)*. I'll let it run for a maximum of *max_iter = 50* iterations. With *perc = 90* a threshold is specified. The lower the threshold the more features will be selected. I usually use a percentage between 80 and 90. 

# In[ ]:


boruta_feature_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=4242, max_iter = 50, perc = 90)
boruta_feature_selector.fit(X, Y)


# After Boruta has run we can transform our dataset.

# In[ ]:


X_filtered = boruta_feature_selector.transform(X)
X_filtered.shape


# And we create a list of the feature names if we would like to use them at a later stage.

# In[ ]:


final_features = list()
indexes = np.where(boruta_feature_selector.support_ == True)
for x in np.nditer(indexes):
    final_features.append(features[x])
print(final_features)


# So I hope you enjoyed my very first Kaggle Kernel :-)
# Let me know if you have any feedback or suggestions.
