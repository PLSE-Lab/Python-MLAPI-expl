#!/usr/bin/env python
# coding: utf-8

# # Santander Value Prediction Challenge

# In this competition, Santander Group is asking us to help them identify the value of transactions for each potential customer.
# We are provided with an anonymized dataset containing numeric feature variables, the numeric target column, and a string ID column. Our task is to predict the value of target column in the test set.
# 
# The evaluation metric for this competition is Root Mean Squared Logarithmic Error. The data set consists of train.csv and test.csv

# <img src="https://dynl.mktgcdn.com/p/OioPDkijUSBehXPo5nCC_CEd-0hZkZRv94-HHnJj-eA/2326x832.jpg" width=600/>

# ## Load Libraries

# In[ ]:


### Import required libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb
import xgboost as xgb

from IPython.display import display # Allows the use of display() for DataFrames

import warnings
warnings.filterwarnings('ignore')


# ### Load Data

# In[ ]:


# Read train and test files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# ## Data Summary

# In[ ]:


# training set
print ("Training set:")
n_data  = len(train_df)
n_features = train_df.shape[1]
print ("Number of Records: {}".format(n_data))
print ("Number of Features: {}".format(n_features))

# testing set
print ("\nTesting set:")
n_data  = len(test_df)
n_features = test_df.shape[1]
print ("Number of Records: {}".format(n_data))
print ("Number of Features: {}".format(n_features))


# We have more features than records in the Training set.
# 
# - We have a total of `4993` features in the `Train` set.
# - The number of records in `Train` set is just `4459`
# 
# 
# - We have a total of `4992` features in the `Test` set.
# - The number of records in `Test` set is just `49342`

# ### Train Data

# In[ ]:


train_df.head(n=10)


# ### Train Data Info

# In[ ]:


train_df.info()


# So there are a total of 4993 features out of which 1845 are of type float64, 3147 are int64 and 1 is object (ID is the object column).
# 
# The sad thing is that, as mentioned earier, we do not have meaningful names for the features. So, we have to rely on other ways to analyze and understand the data.

# ### Test Data

# In[ ]:


test_df.head(n=10)


# ### Test Data Info

# In[ ]:


test_df.info()


# So there are a total of 4992 features in the test set out of which 4991 are of type float64 and 1 is object (ID is the object column)

# ## Check for Missing Values

# In[ ]:


#### Check if there are any NULL values in Train Data
print("Total Train Features with NaN Values = " + str(train_df.columns[train_df.isnull().sum() != 0].size))
if (train_df.columns[train_df.isnull().sum() != 0].size):
    print("Features with NaN => {}".format(list(train_df.columns[train_df.isnull().sum() != 0])))
    train_df[train_df.columns[train_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)


# In[ ]:


#### Check if there are any NULL values in Test Data
print("Total Test Features with NaN Values = " + str(test_df.columns[test_df.isnull().sum() != 0].size))
if (test_df.columns[test_df.isnull().sum() != 0].size):
    print("Features with NaN => {}".format(list(test_df.columns[test_df.isnull().sum() != 0])))
    test_df[test_df.columns[test_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)


# We are lucky enough to get some data that has got no `NULLS`. So, we can go ahead and start anaylzing the data.

# ## Prepare the Data

# In[ ]:


X_train = train_df.drop(["ID", "target"], axis=1)
y_train = np.log1p(train_df["target"].values)

X_test = test_df.drop(["ID"], axis=1)


# ## Check and Remove Constant Features

# As we have lot of features, we will look for ways to trim down on the number of features. From the above sample records we can see that many of the records and features contains **`0`**. Also as there are many features, we will try to see if there are any features with constant values.

# In[ ]:


# check and remove constant columns
colsToRemove = []
for col in X_train.columns:
    if X_train[col].std() == 0: 
        colsToRemove.append(col)
        
# remove constant columns in the training set
X_train.drop(colsToRemove, axis=1, inplace=True)

# remove constant columns in the test set
X_test.drop(colsToRemove, axis=1, inplace=True) 

print("Removed `{}` Constant Columns\n".format(len(colsToRemove)))
print(colsToRemove)


# ## Check and Remove Duplicate Columns

# In[ ]:


# Check and remove duplicate columns
colsToRemove = []
colsScaned = []
dupList = {}

columns = X_train.columns

for i in range(len(columns)-1):
    v = X_train[columns[i]].values
    dupCols = []
    for j in range(i+1,len(columns)):
        if np.array_equal(v, X_train[columns[j]].values):
            colsToRemove.append(columns[j])
            if columns[j] not in colsScaned:
                dupCols.append(columns[j]) 
                colsScaned.append(columns[j])
                dupList[columns[i]] = dupCols
                
# remove duplicate columns in the training set
X_train.drop(colsToRemove, axis=1, inplace=True) 

# remove duplicate columns in the testing set
X_test.drop(colsToRemove, axis=1, inplace=True)

print("Removed `{}` Duplicate Columns\n".format(len(dupList)))
print(dupList)


# In[ ]:


print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))


# ## Exploratory Data Analysis

# As we have `4730 features` it would be very difficult to visualize and analyze all of them. So, we will try to analyze only some of the top features.

# ### Feature Importance from GradientBoostingRegressor

# Below, we will use the GradientBoostingRegressor to see the top features in our dataset.

# In[ ]:


# Find feature importance
clf_gb = GradientBoostingRegressor(random_state = 42)
clf_gb.fit(X_train, y_train)
print(clf_gb)


# In[ ]:


# GradientBoostingRegressor feature importance - top 100
feat_importances = pd.Series(clf_gb.feature_importances_, index=X_train.columns)
feat_importances = feat_importances.nlargest(100)
plt.figure(figsize=(16,15))
feat_importances.plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()


# The above plot looks very cluttered. Instead, we will take a look at the `top 25 features`.

# In[ ]:


# GradientBoostingRegressor feature importance - top 25
feat_importances_gb = pd.Series(clf_gb.feature_importances_, index=X_train.columns)
feat_importances_gb = feat_importances_gb.nlargest(25)
plt.figure(figsize=(16,8))
feat_importances_gb.plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()


# Below we will list the `top 10 features` with their feature importances.

# In[ ]:


print(pd.Series(clf_gb.feature_importances_, index=X_train.columns).nlargest(10))


# ### Feature Importance from RandomForestRegressor

# In[ ]:


# Find feature importance
#Don't forget to do feature importance usinge another regressor
clf_rf = RandomForestRegressor(random_state = 42)
clf_rf.fit(X_train, y_train)
print(clf_rf)


# We will plot the top 25 features from RandomForestRegressor to see how similar they are with the top features given by GradientBoostingRegressor.

# In[ ]:


# RandomForestRegressor feature importance - top 25
feat_importances_rf = pd.Series(clf_rf.feature_importances_, index=X_train.columns)
feat_importances_rf = feat_importances_rf.nlargest(25)
plt.figure(figsize=(16,8))
feat_importances_gb.plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()


# In[ ]:


print(pd.Series(clf_rf.feature_importances_, index=X_train.columns).nlargest(10))


# ### GradientBoostingRegressor vs RandomForestRegressor Top 25 Features

# In[ ]:


plt.figure()
fig, ax = plt.subplots(1, 2, figsize=(16,6))
feat_importances_gb.plot(kind='barh', ax=ax[0])
feat_importances_rf.plot(kind='barh', ax=ax[1])
ax[0].invert_yaxis()
ax[1].invert_yaxis()
plt.show()


# In[ ]:


s1 = pd.Series(clf_gb.feature_importances_, index=X_train.columns).nlargest(10).index
s2 = pd.Series(clf_rf.feature_importances_, index=X_train.columns).nlargest(10).index

common_features = pd.Series(list(set(s1).intersection(set(s2)))).values

print(common_features)


# So, we can see that there are a total of just 6 common features in top 25 features of RandomForestRegressor and GradientBoostingRegressor. So, we should be careful in choosing what feature sto pick for further analysis.

# ## Data Visualization

# Below we will see some visualizations related to the top features.

# In[ ]:


df_plot = X_train[['f190486d6', 'eeb9cd3aa', '58e2e02e6', '58232a6fb', '15ace8c9f', '9fd594eec']]
df_plot['target'] = y_train

g = sns.pairplot(df_plot, diag_kind="kde", palette="BuGn_r")
g.fig.suptitle('Pairplot of Top 6 Important Features',fontsize=26)


# ## Correlation HeatMap

# In[ ]:


# PLot Correlation HeatMap for top 20 features from GB and RF Models
s1 = pd.Series(clf_gb.feature_importances_, index=X_train.columns).nlargest(20).index
s2 = pd.Series(clf_rf.feature_importances_, index=X_train.columns).nlargest(20).index

common_features = pd.Series(list(set(s1).union(set(s2)))).values

print(common_features)


# In[ ]:


df_plot = pd.DataFrame(X_train, columns = common_features)
corr = df_plot.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 16))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Correlation HeatMap", fontsize=15)
plt.show()


# ## PCA Visualization

# In[ ]:


X_train_cpy = X_train.copy()
pca = PCA(n_components=3)
X_train_cpy = pca.fit_transform(X_train_cpy)


# In[ ]:


print(pca.components_)


# In[ ]:


print(pca.explained_variance_)


# In[ ]:


colors = np.random.random((4459, 3))

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(X_train_cpy[:, 0], X_train_cpy[:, 1], X_train_cpy[:, 2], c=colors,
           cmap=plt.cm.Set1, edgecolor=colors, alpha=0.5, s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()


# ## Modeling

# In[ ]:


dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


# ### LightGBM

# In[ ]:


def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 40,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result)
    
    pred_test_y = np.expm1(model.predict(test_X, num_iteration=model.best_iteration))
    return pred_test_y, model, evals_result


# In[ ]:


# Training LGB
pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)
print("LightGBM Training Completed...")


# In[ ]:


# feature importance
print("Features Importance...")
gain = model.feature_importance('gain')
featureimp = pd.DataFrame({'feature':model.feature_name(), 
                   'split':model.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(featureimp[:15])


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub["target"] = pred_test
print(sub.head())
sub.to_csv('sub_lgb.csv', index=False)

