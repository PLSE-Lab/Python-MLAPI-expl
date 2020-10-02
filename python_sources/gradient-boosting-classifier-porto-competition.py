#!/usr/bin/env python
# coding: utf-8

# ## Initial Gradient boosting model to predict probabilities of an insurance claim
# 
# This kernel contains my initial sanity check of the data, a preliminary grid search of hyperparamaters and my initial submission from a gradient boost classification model.
# 
# First, we import the needed libraries:

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


# Next, the data is read in using pandas's read_csv() function

# In[ ]:


test_dat = pd.read_csv('../input/test.csv')
train_dat = pd.read_csv('../input/train.csv')
submission = pd.read_csv('../input/sample_submission.csv')


# An initial check of the data is made to identify any missing values.

# In[ ]:


train_dat.info()


# There appears to be no missing values across the dataset, so we can proceed without the need to impute. Most of the columns appear to be binary, with a few continious floating point predictors as well.

# In[ ]:


train_dat.describe()


# We split the training data into a y response variable, and the predictors, while dropping the ID column.

# In[ ]:



train_y = train_dat['target']
train_x = train_dat.drop(['target', 'id'], axis = 1)


# A grid search is performed to select the optimal hypterparamaters for the gradient boosting classifier.  You can change this section to your own set of values an likely improve the best estimator! Commented out to avoid the time cap.

# In[ ]:


"""
gb_params = {
    'n_estimators' : [100,200,300],
    'learning_rate' : [.1,.2,.3],
    'max_depth' : [3,5,7]
}

gb_class = GradientBoostingClassifier()

gb_grid = GridSearchCV(gb_class, gb_params, cv = 5, n_jobs=-1)
gb_grid.fit(train_x, train_y)

gb_grid.best_estimator_
"""


# An initial run produced the following optimal model paramaters, which I here use to train a new model using the complete training set.
# 
# First, model is initiated:

# In[ ]:


gb_opt = GradientBoostingClassifier(criterion='friedman_mse', init=None,
                            learning_rate=0.1, loss='deviance', max_depth=3,
                            max_features=None, max_leaf_nodes=None, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators=100,
                            presort='auto', random_state=None, subsample=1.0, verbose=0,
                            warm_start=False)
    


# Then it is fed the training data

# In[ ]:


gb_opt.fit(train_x, train_y)


# Since we are predicting probabilities of belonging to a given class, we use predict_proba() as opposed to predict(), which would give us the predicted classes instead of the probabilities.

# In[ ]:


test_y_gb = gb_opt.predict_proba(test_x)


# Below the predictions are placed into the sample submission dataframe, and the inverse of the predictions is taken. predict_proba() give the probability of the given instance being a '0' but we want the probability of an instance being a '1' (an insurance claim filed). Therefore, we take the inverse.

# In[ ]:



gb_out = submission
gb_out['target'] = test_y_gb

gb_out['target'] = 1-gb_out['target']


# Then we write the data to a csv, dropping the index and rounding the probabilities to 4 decimals (this was the number of decimals in the sample submission).

# In[ ]:


gb_out.to_csv('gb_predictions1.csv', index=False, float_format='%.4f')


# And thats it! If you play around with the hyperparameters then there are likely some large performance boosts to be found (accidental pun... but I'm leaving it). 
