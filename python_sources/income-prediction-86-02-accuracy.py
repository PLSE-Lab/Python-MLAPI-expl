#!/usr/bin/env python
# coding: utf-8

# # Adult Data Income Classification Notebook

# The Goal is to predict whether a person has an income of more than 50K a year or not. This is basically a binary classification problem where a person is classified into the >50K group or <=50K group. I have used Random Forests and Decision Tree to tackle this problem. 
# The dataset is taken from the UCI Machine Learning Repository. The link to the same is the following: https://archive.ics.uci.edu/ml/datasets/census+income
# ### This Notebook covers the following aspects:
#    #### 1. Data Preprocessing and Visualization
#    #### 2. Classification Task
#    #### 3. Hyperparameter Tuning
#    #### 4. Building the Final Model
#    #### Appendix - Additional Information and graphs about hyperparameter tuning of Random Forests

# ### 1. Data Preprocessing and Visualization

# In[ ]:


import numpy as np
import pandas as pd

df = pd.read_csv("../input/adult.csv")
df.head()


# In[ ]:


df.dtypes


# Checking for null and/or missing values

# In[ ]:


df.isnull().sum()


# In[ ]:


df.columns.isna()


# In[ ]:


df.isin(['?']).sum()


# In[ ]:


df = df.replace('?', np.NaN)
df.head()


# In[ ]:


df = df.dropna()
df.head()


# Mapping the income labels numerically

# In[ ]:


df['income'] = df['income'].map({'<=50K':0, '>50K':1})
df.income.head()


# In[ ]:


numerical_df = df.select_dtypes(exclude=['object'])
numerical_df.columns


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


plt.hist(df['age'], edgecolor='black')
plt.title('Age Histogram')
plt.axvline(np.mean(df['age']), color='yellow', label='average age')
plt.legend()


# In[ ]:


age50k = df[df['income']==1].age
agel50k = df[df['income']==0].age

fig, axs = plt.subplots(2, 1)

axs[0].hist(age50k, edgecolor='black')
axs[0].set_title('Distribution of Age for Income > 50K')

axs[1].hist(agel50k, edgecolor='black')
axs[1].set_title('Distribution of Age for Income <= 50K')
plt.tight_layout()


# #### Inferences:
# 
# For Income > 50K, Age is almost normally distributed
# 
# For Income <=50K, Age is positively skewed. More people in the 20s and 30s have income <= 50K.

# In[ ]:


df['marital.status'].unique()


# In[ ]:


ax = sns.countplot(df['marital.status'], hue=df['income'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()


# #### Converting marital.status to 2 categories
# 
# It seems better to reduce the number of categories for marital status to better visualize the effect of marital status on income. 
# We need to convert the following into 2 distinct categories namely, "married" and "single"

# In[ ]:


df['marital.status'] = df['marital.status'].replace(['Widowed', 'Divorced', 'Separated', 'Never-married'], 'single')

df['marital.status'] = df['marital.status'].replace(['Married-spouse-absent', 'Married-civ-spouse', 'Married-AF-spouse'], 'married')


# In[ ]:


categorical_df = df.select_dtypes(include=['object'])
categorical_df.columns


# In[ ]:


sns.countplot(df['marital.status'], hue=df['income'])


# #### Inference:
# 
# Married people are more likely to earn more than 50K as income

# #### Encoding categorical variables numerically for classification 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()


# In[ ]:


ax = sns.countplot(df['income'], hue=df['race'])
ax.set_title('')


# In[ ]:


categorical_df = categorical_df.apply(enc.fit_transform)
categorical_df.head()


# In[ ]:


df = df.drop(categorical_df.columns, axis=1)
df = pd.concat([df, categorical_df], axis=1)
df.head()


# In[ ]:


sns.factorplot(data=df, x='education', y='hours.per.week', hue='income', kind='point')


# In[ ]:


sns.FacetGrid(data=df, hue='income', size=6).map(plt.scatter, 'age', 'hours.per.week').add_legend()


# #### Inferences: 
#     1. Maximum people between the age of 25 to 80 earn more than 50K as income
#     2. Most people which work atleast 36 to 70 hours a week earn more than 50K
#     3. Most people under the age of 20 earn less than as m50K income

# In[ ]:


plt.figure(figsize=(15,12))
cor_map = df.corr()
sns.heatmap(cor_map, annot=True, fmt='.3f', cmap='YlGnBu')


# ### 2. Classification Task

# In[ ]:


from sklearn.model_selection import train_test_split

X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=24)
clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score

print("Random Forests accuracy", accuracy_score(y_test, y_pred))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='gini', random_state=21, max_depth=10)

dtree.fit(X_train, y_train)
tree_pred = dtree.predict(X_test)

print("Decision Tree accuracy: ", accuracy_score(y_test, tree_pred))


# Both the Random Forest and Decision Tree return similar prediction accuracy scores. 
# However, Random Forest is marginally better and thus, it is the selected model. 
# 
# We will now optimize the Random Forest Classifier by tuning the Hyperparameters.

# ### 3. Hyperparameter Tuning
# 
# The random forest hyperparameters we will tune are the following: 
# 
# 1. n_estimators: represents the number of trees in the forest. More trees translates to better learning from the data, however at the cost of performance. Thus, a careful consideration must be placed on what is the optimal value.
# 
# 2. max_features: the number of features to consider before making a split. A high value causes overfitting. Thus, an optimized value must be found.
# 
# 3. min_samples_leaf: the minimum number of samples needed for a node to be considered a leaf node. Increasing this value can cause underfitting. 
# 
# More about this in the Appendix.

# #### Methodology
# First we do a Randomized Search to narrow down the possibilites and then perform a Grid Search to further optimize the model. This approach is more suited since directly running a Grid Search is computationally intensive.
# 
# I found this article about Random and Grid Search particularly useful: https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search
# 
# #### 1. Randomized Search  

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold

n_estimators = np.arange(100, 1000, 100)
max_features = np.arange(1, 10, 1)
min_samples_leaf = np.arange(2, 10, 1)
kfold = KFold(n_splits = 3)
start_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'min_samples_leaf': min_samples_leaf,
    }

rf = RandomForestClassifier()

test_rf = RandomizedSearchCV(estimator=rf, param_distributions=start_grid, cv=kfold)
print(start_grid)


# In[ ]:


'''
Commented out since takes a long time to run. 

------------------------------
OPTIMIZED PARAMETERS:
max_features = 3
min_samples_leaf = 5
n_estimators = 100
------------------------------


test_rf.fit(X_train, y_train)
test_rf.best_params_
'''


# #### 2. Grid Search

# In[ ]:


'''
Commented out since takes about 25 minutes to run
----------------------------------
OPTIMIZED HYPERPARAMETERS:

max_features = 3
min_samples_leaf = 3
n_estimators = 450
-----------------------------------

kfold_gs = KFold(n_splits=3)
n_estimators = np.arange(100, 500, 50)
max_features = np.arange(1, 5, 1)
min_samples_leaf = np.arange(2, 5, 1)

gs_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'min_samples_leaf': min_samples_leaf
}

test_grid = GridSearchCV(estimator = rf, param_grid=gs_grid, cv=kfold_gs)
res = test_grid.fit(X_train, y_train)
print(res.best_params_)
print(res.best_score_)
'''


# ### 4. Building the Model

# In[ ]:


final_model = RandomForestClassifier(n_estimators=450, min_samples_leaf=3, max_features=3, random_state=24)
final_model.fit(X_train, y_train)


# In[ ]:


predictions = final_model.predict(X_test)
print(accuracy_score(y_test, predictions))


# The previous Random Forest Classifier without tuning gave an accuracy score of 0.851
# The tuned model gives an accuracy score of 0.86
# 
# By tuning the model, we are able to get an improvement of 0.01 or 1%.

# ### Appendix
# 
# The appendix has graphs for when the hyperparameters are under-fitting or over-fitting. 
# This can be used when determining the range of values for the hyperparameters. 
# 
# AUC (Area Under Curve) is used as the evaluation metric. For binary classification problems, AUC is a good evaluation metric.
# 
# 
# This article explains how hyperparameters should be tuned for Random Forest:
# https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d 
# 
# (I have used the code from this article to visualize overfitting and underfitting in training and testing case.)

# #### n_estimators

# In[ ]:


from sklearn.metrics import roc_curve, auc
n_estimators = np.arange(100, 1000, 100)

train_results = []
test_results = []
for n_est in n_estimators:
   rf = RandomForestClassifier(n_estimators = n_est)
   rf.fit(X_train, y_train)

   train_pred = rf.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)

   y_pred = rf.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')
line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('n_estimators')


# #### max_features
# 
# A case of overfitting. It is quite unexpected that the model is over-fitting for all values of max_features. However, the scikit-learn documentation states that until a valid parition node is not found, the splitting does not stop even if it exceeds the value of max_features features.

# In[ ]:


from sklearn.metrics import roc_curve, auc
max_features = np.arange(1, 10, 1)

train_results = []
test_results = []
for max_f in max_features:
   rf = RandomForestClassifier(max_features=max_f)
   rf.fit(X_train, y_train)

   train_pred = rf.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)

   y_pred = rf.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('max_features')


# #### min_samples_leaf
# 
# This is a case of underfitting. Increasing this value can cause underfitting.

# In[ ]:


from sklearn.metrics import roc_curve, auc
min_samples_leafs = np.arange(2, 10, 1)

train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
   rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf)
   rf.fit(X_train, y_train)

   train_pred = rf.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)

   y_pred = rf.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('min samples leaf')

