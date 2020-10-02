#!/usr/bin/env python
# coding: utf-8

# # Notebooks to Explore
# * [Introduction to phyiological data](https://www.kaggle.com/stuartbman/introduction-to-physiological-data)
#   * Contains information about using signal processing to de-noise the data. 
#   * Provides some suggestions for engineering new features. 
# * [Reducing Commercial Aviation Fatalities (11th)](https://www.kaggle.com/shahaffind/reducing-commercial-aviation-fatalities-11th)
#   * Uses ideas from above notebook with gradient boosting to get really good results.
#   * I believe this notebook creates a different model for each pilot.
# * [Starter Code : EDA and LGBM Baseline](https://www.kaggle.com/theoviel/starter-code-eda-and-lgbm-baseline)
#   * Contains interesting data visualizations.

# # Things to Try
# 
# * Use the montage information in the "Introduction to physiological data" notebook to engineer new features. 
# * Use methods from the first two notebooks above to de-noise the data. 
# * Create a variable that identifies the pilot. Include this variable in the model. 
# * Create a separate model for each pilot. 
# * Attempt to train a model on the complete training set. 
# * Perform hyperparameter tuning on XGBCLassifier Model. Information about the parameters can be found [here](https://xgboost.readthedocs.io/en/latest/parameter.html) and [here](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/). 

# # Import Packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# # Load Data

# In[ ]:


train = pd.read_csv("../input/reducing-commercial-aviation-fatalities/train.csv")


# In[ ]:


train.sample(10)


# In[ ]:


print(train.shape)


# In[ ]:


test_iterator = pd.read_csv('../input/reducing-commercial-aviation-fatalities/test.csv', chunksize=5)
test_top = next(test_iterator)
test_top


# In[ ]:


sample_submission = pd.read_csv("../input/reducing-commercial-aviation-fatalities/sample_submission.csv")
sample_submission.sample(10)


# # Explore Training Data

# In[ ]:


pd.crosstab(train.experiment, train.event)


# In[ ]:


pd.crosstab(train.experiment, train.crew)


# In[ ]:


pd.crosstab(train.experiment, train.seat)


# In[ ]:


print(list(enumerate(train.columns)))


# In[ ]:


crew = 3
seat = 0
exp = 'DA'
ev = 'D'

sel = (train.crew == crew) & (train.experiment == exp) & (train.seat == seat)
pilot_info = train.loc[sel,:].sort_values(by='time')


plt.figure(figsize=[16,12])
for i in range(4, 27):
    plt.subplot(6,4,i-3)
    plt.plot(pilot_info.time, 
             pilot_info.iloc[:,i], zorder=1)
    plt.scatter(pilot_info.loc[pilot_info.event ==  ev,:].time, 
             pilot_info.loc[pilot_info.event == ev,:].iloc[:,i], c='red', zorder=2, s=1)
    plt.title(pilot_info.columns[i])

plt.tight_layout()
plt.show()


# # Create Feature and Label Arrays

# In[ ]:


y_train_full = train.event
X_train_full = train.iloc[:,4:27]
X_train_full.head()


# In[ ]:


pd.DataFrame({
    'min_val':X_train_full.min(axis=0).values,
    'max_val':X_train_full.max(axis=0).values
}, index = X_train_full.columns
)


# In[ ]:


y_train_full.value_counts() 


# In[ ]:


y_train_full.value_counts() / len(y_train_full)


# # Sample Training Set for Faster Training

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.98, stratify=y_train_full, random_state=1)

print(X_train.shape)


# # Logistic Regression

# In[ ]:


get_ipython().run_cell_magic('time', '', "lr_mod = LogisticRegression(solver='lbfgs', n_jobs=-1)\nlr_mod.fit(X_train, y_train)\n\nprint('Training Accuracy:  ', lr_mod.score(X_train, y_train))\nprint('Validation Accuracy:', lr_mod.score(X_valid, y_valid))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nlr_pipe = Pipeline(\n    steps = [\n        ('scaler', StandardScaler()),\n        ('classifier', LogisticRegression(solver='lbfgs', n_jobs=-1))\n    ]\n)\n\nlr_param_grid = {\n    'classifier__C': [0.0001, 0.001, 0.1, 1.0],\n}\n\n\nnp.random.seed(1)\ngrid_search = GridSearchCV(lr_pipe, lr_param_grid, cv=5, refit='True')\ngrid_search.fit(X_train, y_train)\n\nprint(grid_search.best_score_)\nprint(grid_search.best_params_)")


# # Random Forest Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nrf_mod = RandomForestClassifier(n_estimators=10, max_depth=32, n_jobs=-1)\nrf_mod.fit(X_train, y_train)\n\nprint('Training Accuracy:  ', rf_mod.score(X_train, y_train))\nprint('Validation Accuracy:', rf_mod.score(X_valid, y_valid))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "rf_pipe = Pipeline(\n    steps = [\n        ('scaler', StandardScaler()),\n        ('classifier', RandomForestClassifier(n_estimators=10, n_jobs=-1))\n    ]\n)\n\nlr_param_grid = {\n    'classifier__max_depth': [8, 16, 32, 64, 128]\n}\n\n\nnp.random.seed(1)\ngrid_search = GridSearchCV(rf_pipe, lr_param_grid, cv=5, refit='True')\ngrid_search.fit(X_train, y_train)\n\nprint(grid_search.best_score_)\nprint(grid_search.best_params_)")


# In[ ]:


grid_search.cv_results_['mean_test_score']


# In[ ]:


get_ipython().run_cell_magic('time', '', "rf_mod = RandomForestClassifier(n_estimators=100, max_depth=32, n_jobs=-1)\nrf_mod.fit(X_train, y_train)\n\nprint('Training Accuracy:  ', rf_mod.score(X_train, y_train))\nprint('Validation Accuracy:', rf_mod.score(X_valid, y_valid))")


# In[ ]:


rf_mod.predict_proba(X_train)


# In[ ]:


from sklearn.metrics import log_loss

log_loss(y_train, rf_mod.predict_proba(X_train))


# In[ ]:


log_loss(y_valid, rf_mod.predict_proba(X_valid))


# **Test Score: 0.61190**

# # Gradient Boosting Tree

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nxbg_mod = XGBClassifier()\nxbg_mod.fit(X_train, y_train)\n\nxbg_mod.score(X_train, y_train)')


# In[ ]:


xbg_mod.score(X_valid, y_valid)


# In[ ]:


log_loss(y_train, xbg_mod.predict_proba(X_train))


# In[ ]:


log_loss(y_valid, xbg_mod.predict_proba(X_valid))


# **Test Score: 0.88456**

# # Hyperparameter Tuning for Gradient Boosting

# In[ ]:


get_ipython().run_cell_magic('time', '', "xgd_pipe = Pipeline(\n    steps = [\n        ('classifier', XGBClassifier(learning_rate=0.3, max_depth=6, alpha=1, n_estimators=50, subsample=0.5))\n    ]\n)\n\nxgd_param_grid = {\n    'classifier__learning_rate' : [0.1, 0.3, 0.5, 0.7, 0.9],\n    'classifier__alpha' : [0, 1, 10, 100]\n    #'classifier__max_depth': [8, 16, 32, 64, 128]\n    \n}\n\n\n#np.random.seed(1)\n#xgd_grid_search = GridSearchCV(xgd_pipe, xgd_param_grid, cv=5, refit='True')\n#xgd_grid_search.fit(X_train, y_train)\n\n#print(xgd_grid_search.best_score_)\n#print(xgd_grid_search.best_params_)")


# # Generate Test Predictions

# In[ ]:


test_iterator = pd.read_csv('../input/reducing-commercial-aviation-fatalities/test.csv', chunksize=5)
test_top = next(test_iterator)
test_top


# In[ ]:


print(xbg_mod.predict_proba(test_top.iloc[:,5:]))


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ncs = 1000000\ni = 0\n\nfor test in pd.read_csv('../input/reducing-commercial-aviation-fatalities/test.csv', chunksize=cs):\n  \n    print('--Iteration',i, 'is started')\n    \n    test_pred = xbg_mod.predict_proba(test.iloc[:,5:])\n    \n    partial_submission = pd.DataFrame({\n        'id':test.id,\n        'A':test_pred[:,0],\n        'B':test_pred[:,1],\n        'C':test_pred[:,2],\n        'D':test_pred[:,3]\n    })\n        \n    if i == 0:\n        submission = partial_submission.copy()\n    else:\n        submission = submission.append(partial_submission, ignore_index=True)\n        \n\n    del test\n    print('++Iteration', i, 'is done!')\n    i +=1")


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv", index=False)


# In[ ]:




