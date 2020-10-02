#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')

# Linear Algebra
import numpy as np

# Data Processing
import pandas as pd

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Algorithms
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

# Stop unnecessary Seaborn warnings
import warnings
warnings.filterwarnings('ignore')
sns.set()  # Stylises graphs


# ## Loading the Data

# In[ ]:


wine_df = pd.read_csv('../input/winequality-red.csv')


# In[ ]:


wine_df.info()


# In[ ]:


wine_df.head()


# ## Exploring the Dataset

# In[ ]:


# Plotting quality of wine

fig = plt.figure(figsize=(40, 8))
sns.countplot(x='quality', data=wine_df)
plt.title("Barplot of Quality of Wine")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.show()


# In[ ]:


# Heatmap of variables

sns.set(style="white")

# Computer correlation matrix
corr = wine_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colourmap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    square=True, linewidths=.5, cbar_kws={"shrink": .5}
)

ax.set_title('Correlation Heatmap of the Variables of Wine')

plt.show()


# Looking at the graph above, it seems that there are some correlations that may be worth exploring. There may be a correlation between alcohol and quality... Coincidence, I think not.

# In[ ]:


fig = plt.figure(figsize = (20,8))
sns.barplot(x='quality', y ='alcohol', data=wine_df)
plt.title("Quality of Wine with Alcohol")
plt.ylabel("Alcohol (% of wine)")
plt.show()


# ## Training a Model

# #### Cleaning and Prepping Data

# In[ ]:


# Determining if there are any columns with missing data

cols_with_missing = [col for col in wine_df.columns if wine_df[col].isnull().any()]
print(cols_with_missing)


# As we can see, there are no missing bits of data within the dataset - this makes life easiser!

# In[ ]:


# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)


# There are no columns containing categorical data either.

# In[ ]:


# Assigning training and validation data

X = wine_df.copy()
y = X.quality
X.drop(['quality'], axis=1, inplace=True)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=0)


# ### Modelling with Random Forests

# #### Determining the Number of Estimators

# In[ ]:


scores = {}

for n_estimators in range(10, 510, 10):
    RF_model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    RF_model.fit(X_train, y_train)
    RF_predictions = RF_model.predict(X_valid)
    RF_mae = mean_absolute_error(RF_predictions, y_valid)
    scores[n_estimators] = RF_mae


# In[ ]:


fig_RF, ax_RF = plt.subplots(figsize=(10, 4))
ax_RF.set_title("Mean Absolute Error with Number of Estimators of a Random Forest")
ax_RF.set_xlabel("Number of Estimators")
ax_RF.set_ylabel("Mean Absolute Error")
plt.plot(scores.keys(), scores.values())


# In[ ]:


best_n_estimators = 0

for n_estimators, score in scores.items():
    if score == min(scores.values()):
        best_n_estimators = n_estimators
        print(f"Best Number of Estimators: {n_estimators}")


# In[ ]:


RF_model = RandomForestClassifier(n_estimators=best_n_estimators, random_state=0)
RF_model.fit(X_train, y_train)
RF_predictions = RF_model.predict(X_valid)
RF_mae = mean_absolute_error(RF_predictions, y_valid)

print(f"Mean Absolute Error: {RF_mae}")
print(classification_report(y_valid, RF_predictions))


# ### Modelling using XGBoost

# #### Fix Learning Parameters

# In[ ]:


XGB_model = XGBRegressor(
    n_estimators=200, learning_rate=0.02,
    max_depth=10, min_child_weight=1,
    gamma=0, subsample=0.8,
    colsample_bytree=0.7,
    random_state=0, nthread=4
)

XGB_model.fit(X_train, y_train, early_stopping_rounds=10, verbose=False, eval_set=[(X_valid, y_valid)])
XGB_predictions = XGB_model.predict(X_valid)
XGB_mae = mean_absolute_error(XGB_predictions, y_valid)

print(f"Mean Absolute Error: {XGB_mae}")


# #### Tuning Gamma

# In[ ]:


base_XGB_model = XGBRegressor(
    n_estimators=200, learning_rate=0.02,
    max_depth=10, min_child_weight=1,
    subsample=0.8, colsample_bytree=0.7,
    random_state=0, nthread=4
)

param_test = {
 'gamma':[i/10 for i in range(20)]
}

grid_search = GridSearchCV(estimator=base_XGB_model, param_grid=param_test, n_jobs=1, iid=False, cv=5)
grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.cv_results_, grid_search.best_params_, grid_search.best_score_


# ### Applying Cross-Validation
# Since the dataset is smaller, we should use cross-validation 

# In[ ]:


# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(RF_model, X, y, cv=10, scoring='neg_mean_absolute_error')
print(f"MAE Scores: {scores}")
print(f"Average MAE Score: {scores.mean()}")

