#!/usr/bin/env python
# coding: utf-8

# # The Unreasonable Effectiveness of Tree Boosting
# 
# This notebook illustrates how gradient boosting can learn even the most complex statistical relationships, at least if you feed enough data to the beast!

# ## Imports and settings

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import lightgbm as lgb

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=1.3)


# ## Generate data
# 
# We first sample 100'000 data points from the following two-dimensional function:
# $$
#     y = f(x, z) = \frac{\sin(x^2 + z^2)}{x^2 + z^2} + 0.5 \cos(x) - 0.5,
# $$
# $x, z \in [-2\pi, 2\pi]$.
# 
# Its profile looks as follows.

# In[ ]:


# Generate data
n = 100_000
t = np.linspace(-2 * np.pi, 2 * np.pi, int(np.sqrt(n)))
x, z = np.meshgrid(t, t)
y = np.sin(x**2 + z**2) / (x**2 + z**2) + np.cos(x) * 0.5 - 0.5

# Turn to DataFrame
data = pd.DataFrame({
    'x': x.flatten(), 
    'z': z.flatten(),
    'y': y.flatten()
})

fig, ax = plt.subplots(1, 2, figsize=(11, 5))

# Response distribution
sns.distplot(data.y, ax=ax[0])
ax[0].set(title="Distribution of y")

# Response profile
def nice_heatmap(data, v, ax):
    sns.heatmap(data.pivot('z', 'x', v), 
                xticklabels=False, yticklabels=False, 
                cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title("Heatmap of " + v)
    return None

nice_heatmap(data, v='y', ax=ax[1])
fig.tight_layout()


# ## Train/Test split
# 
# In order to not fall into the trap of overfitting, we set aside 33% of the data lines for testing only. The model is trained on the remaining 67%.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    data[["x", "z"]], data["y"], 
    test_size=0.33, random_state=63
)

print("All shapes:")
for dat in (X_train, X_test, y_train, y_test):
    print(dat.shape)


# ## Modelling
# 
# Now we fit a gradient boosting tree with [LightGBM](https://github.com/microsoft/LightGBM), besides [XGBoost](https://github.com/dmlc/xgboost) and [CatBoost](https://github.com/catboost/catboost) one of the tree major implementations of gradient boosting. 
# 
# The parameters have been manually chosen by five-fold cross-validation on the training data in order to minimize (root-)mean squared error.

# In[ ]:


# Parameters
params = {
    'objective': 'regression',
    'num_leaves': 63,
    'metric': 'l2_root',
    'learning_rate': 0.3,
    'bagging_fraction': 1,
    'min_sum_hessian_in_leaf': 0.01
}

# Data interface
lgb_train = lgb.Dataset(X_train, label=y_train)
                 
# Fitting the model
if False:
    # Find good parameter set by cross-validation
    gbm = lgb.cv(params,
                 lgb_train,
                 num_boost_round=20000,
                 early_stopping_rounds=1000,
                 stratified=False,
                 nfold=5,
                 verbose_eval=1000, 
                 show_stdv=False)
else: 
    # Fit with parameters
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=5000)


# ## Evaluation
# 
# After fitting the models, we are ready to apply the model on the factory fresh hold-out data set. Was gradient boosting able to well approximate our crazy function? To do so, we look at heatmaps of
# 
# - the response y (i.e. the ground truth)
# 
# - the predictions, as well as
# 
# - the out-of-sample residuals.

# In[ ]:


# Add predictions to test data
data_eval = pd.DataFrame(np.c_[X_test, y_test], columns=['x', 'z', 'y'])
data_eval["predictions"] = gbm.predict(X_test)
data_eval["residuals"] = data_eval["y"] - data_eval["predictions"]

# Plot the results
fig, ax = plt.subplots(1, 3, figsize=(21, 6))
for i, v in enumerate(['y', 'predictions', 'residuals']):
    nice_heatmap(data_eval, v=v, ax=ax[i])
fig.tight_layout()


# The white scatter in the image are due to plotting only the 33% hold-out sample.

# ## Wrap up
# 
# Congratulation to the gradient boosted tree! The approximation error is extremely small. Imagine how long you would have had to generate relevant features for a linear model...

# In[ ]:




