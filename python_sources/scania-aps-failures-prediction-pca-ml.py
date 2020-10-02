#!/usr/bin/env python
# coding: utf-8

# <a id='Top'></a>
# <center>
# <h1><u>Scania Air Pressure System Failures Prediction</u></h1>
# </center>
# <br>
# 
# <!-- Start of Unsplash Embed Code - Centered (Embed code by @BirdyOz)-->
# <div style="width:60%; margin: 20px 20% !important;">
#     <img src="https://images.unsplash.com/photo-1540852360777-5f6fa7752aeb?ixlib=rb-1.2.1&amp;q=80&amp;fm=jpg&amp;crop=entropy&amp;cs=tinysrgb&amp;w=720&amp;fit=max&amp;ixid=eyJhcHBfaWQiOjEyMDd9" class="img-responsive img-fluid img-med" alt="trailer on road " title="trailer on road ">
#     <div class="text-muted" style="opacity: 0.5">
#         <small><a href="https://unsplash.com/photos/mS2ngGq6VO4" target="_blank">Photo</a> by <a href="https://unsplash.com/@vanveenjf" target="_blank">@vanveenjf</a> on <a href="https://unsplash.com" target="_blank">Unsplash</a>, accessed 21/02/2020</small>
#     </div>
# </div>
# <!-- End of Unsplash Embed code -->
#                 
# In this challange we are asked to predict if there is truck APS failure based on the sensor telemetry data. SVM will be used for classification.
# 
# This analysis is composed from the following steps:
# 1. reading data
# 2. dealing with missing values
# 3. data standarisation (normalisation)
# 4. creating custom scorer
# 5. creating pipeline of PCA, grid search and SVM
# 6. visualisation of results

# **READING DATA**
# 
# First I will load basic libraries and raw data. Additional libraries I will be loading as necessary to increase readibility.

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("../input/aps_failure_training_set.csv", na_values="na")
data.head()


# From a visual inspection of raw data it is obvious that some columns contain missing values. The first column named "class" is our target set (labels).

# In[ ]:


missing = data.isna().sum().div(data.shape[0]).mul(100).to_frame().sort_values(by=0, ascending = False)
missing.plot.bar(figsize=(50,10))
plt.show()


# Graph above shows that we have significant amount of missing data. I will drop columns containing more than 75% of missing values.

# In[ ]:


cols_missing = missing[missing[0]>75]
cols_missing


# In[ ]:


cols_to_drop = list(cols_missing.index) # list with columns to drop
cols_to_drop.append('class')
cols_to_drop


# **DEALING WITH MISSING DATA**

# In[ ]:


X = data.drop(cols_to_drop, axis=1)
y = data.loc[:,"class"]
y = pd.get_dummies(y).drop("neg",axis=1)


# Filling missing data with a mean.

# In[ ]:


X.fillna(X.mean(), inplace=True)


# **DATA STANDARISATION**
# 
# I am going to use the Support Vector Machine Classifier and it requires standarisation of data.

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


# **CREATING CUSTOM SCORER**
# 
# Here I will create a custom scorer accoring to the database guidelines.

# In[ ]:


from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import confusion_matrix

def my_scorer(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = 10*fp+500*fn
    return cost

my_func = make_scorer(my_scorer, greater_is_better=False)


# **PCA AND PARAMETERS OPTIMISATION PIPELINED**
# 
# I will chain PCA and classification model with a pipeline to perform a grid search optimisation. In the cell below I will use Support Vector Machine Classifier (SVC). 

# In[ ]:


from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

clf = SVC(probability = False, class_weight="balanced", gamma="auto") # initialising SVC classifier
pca = PCA() # initialising PCA component

pipe = Pipeline(steps=[("pca",pca), ("clf",clf)]) # creating pipeline

param_grid = {
    'pca__n_components': range(10,24),
    'clf__C': np.arange(0.2,0.5,0.05),
}

search = GridSearchCV(pipe, param_grid, iid=False, cv=3, return_train_score=False, scoring=my_func, n_jobs=-1, verbose=3) # Grid Search with 3-fold CV
search.fit(X_scaled, np.ravel(y))

# Plotting best classificator
print("Best parameters (CV score: {:0.3f}):".format(search.best_score_))
print(search.best_params_)


# In[ ]:


pca.fit(X_scaled)

fig, ax0 = plt.subplots(nrows=1, sharex=True, figsize=(12, 6))
ax0.plot(pca.explained_variance_ratio_, linewidth=2)
ax0.set_ylabel('PCA explained variance')
ax0.axvline(search.best_estimator_.named_steps['pca'].n_components, linestyle=':', label='n_components chosen')
ax0.legend(prop=dict(size=12))
plt.show()


# In[ ]:


search.best_estimator_


# In[ ]:


fig, ax1 = plt.subplots(nrows=1, sharex=True, figsize=(12, 6))

results = pd.DataFrame(search.cv_results_)
components_col = 'param_pca__n_components'
best_clfs = results.groupby(components_col).apply(lambda g: g.nlargest(1, 'mean_test_score'))

best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',legend=False, ax=ax1)

ax1.set_ylabel('Classification accuracy (val)')
ax1.set_xlabel('n_components')

plt.tight_layout()
plt.show()

