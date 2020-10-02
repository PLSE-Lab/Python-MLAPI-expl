#!/usr/bin/env python
# coding: utf-8

# **Goal**: Build a model that can predict customer's response towards a marketing campaign
# 
# **Data Description**: 
# 
# * There is one campaign, having 5k customers
# * Customers are divided into treatment and control groups
#     * If customer is in treatment (Treatment=1) => Offer the product
#     * If customer is in control (Treatment=0) => No offer
# * Response shows whether customer purchased the product or not
#     * Response=1 : Purchased
#     * Response=0 : Not Purchased
# * It is clear that customer may purchase the product even if he is not targeted, or he may not purchase the product even if he is targeted
# 
# ---
# 

# In[2]:


import warnings

warnings.filterwarnings('ignore')


# In[3]:


import pandas as pd
import numpy as np

df = pd.read_excel(
    'https://github.com/bukanyusufandroid/dataset/'
    'blob/master/XL%20Case%20Study/XL%20AA%20Interview'
    '%20-%20Data.xlsx?raw=true',
    sheet_name=0
)

df.head(3)


# In[4]:


print(df.describe())
print(df.columns)


# In[5]:


feature = ['Treatment', 'VAR_1', 'VAR_2',
       'VAR_3', 'VAR_4', 'VAR_5', 'VAR_6', 'VAR_7', 'VAR_8', 'VAR_9', 'VAR_10',
       'VAR_11', 'VAR_12', 'VAR_13', 'VAR_14', 'VAR_15', 'VAR_16', 'VAR_17',
       'VAR_18', 'VAR_19', 'VAR_20']

categorical = [ 'Response', 'Treatment', 'VAR_7', 'VAR_10', 'VAR_20' ]


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None, 'display.max_columns', None)
sns.set(style="darkgrid")


# In[7]:


plt.figure(figsize=(10, 6))
plt.title('Distribution of Response')
sns.countplot(df['Treatment'])


# In[8]:


plt.figure(figsize=(10, 6))
plt.title('Distribution of Response over Treatment')
sns.countplot(df['Treatment'], hue=df['Response'])


# As we see from the above graph, there is an imbalance problem here as the class is not a 50/50 or 60/40 distribution. Let's keep this information from now, might be useful later.
# 
# Let's now move on to see whether there exists any relationship between features. Note that I use the treshold of 0.75 to define strong relationship.

# In[9]:


_tmp = sorted([i for i in df.columns if 'VAR' in i])


# In[10]:


_tmp_ = _tmp[:10]

for i in ['Response', 'Treatment']:
    _tmp_.append(i)

data_ = pd.melt(df[_tmp_],
                id_vars = "Response",
                var_name="features",
                value_name="value"
              )

plt.subplots(figsize=(10,5))
sns.violinplot(x="features", y="value", hue="Response", 
              data=data_, split=True, inner="quart",
                palette='pastel')


# In[11]:


_tmp_ = _tmp[10:]

for i in ['Response', 'Treatment']:
    _tmp_.append(i)

data_ = pd.melt(df[_tmp_],
                id_vars = "Response",
                var_name="features",
                value_name="value"
              )

plt.subplots(figsize=(10,5))
sns.violinplot(x="features", y="value", hue="Response", 
              data=data_, split=True, inner="quart",
                palette='pastel')


# In[12]:


corr = df.corr()
unstack_corr = corr.unstack()


# In[13]:


data_ = pd.melt(
    df[[i for i in df.columns if i not in ['Customer_No', 'Campaign_ID']]],
    id_vars = "Response",
    var_name="features",
    value_name="value"
)

plt.figure(figsize=(20,8))
sns.violinplot(x="features", y="value", hue="Response", 
              data=data_, split=True, inner="quart")
plt.xticks(rotation=90)
plt.show()


# In[14]:


feature_ = ['VAR_8', 'VAR_3', 'VAR_14', 'VAR_13', 'VAR_7', 'Response']

data_ = pd.melt(
    df[feature_], id_vars = "Response",
    var_name="features",
    value_name="value"
)

plt.figure(figsize=(20,8))
sns.violinplot(x="features", y="value", hue="Response", 
              data=data_, split=True, inner="quart")
plt.xticks(rotation=90)
plt.show()


# From the violin plot above, we can see how data is distributed among groups. We spot also features that can help distuingishing group of Response 1 and 0 really well. So, now we have information of the feature.
# 
# ```[ VAR_3, VAR_7, VAR_8, VAR_13, VAR_14 ]```
# 
# Let's see whether there are any correlation between them.

# In[15]:


sns.countplot(unstack_corr[abs(unstack_corr) >= 0.75])


# In[16]:


unstack_corr[abs(unstack_corr) >= 0.75]


# Not exists any strong relationship between any feature. So we don't need to eliminate anything if we're going to use that 5 features later.

# ## Parameter Tuning

# In[17]:


from sklearn.model_selection import train_test_split
from random import randint
from imblearn.over_sampling import SMOTENC

random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    df[feature],
    df['Response'], 
    test_size=0.2,
    random_state=random_state
)

X_train.head(3)


# In[18]:


smote_nc = SMOTENC(categorical_features=[0, 7, 10, 20], random_state=42)
X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)


# In[19]:


X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    df[[i for i in feature_ if i != 'Response']],
    df['Response'], 
    test_size=0.2,
    random_state=42
)

X_train_red.head(3)


# In[20]:


smote_nc = SMOTENC(categorical_features=[4], random_state=42)
X_resampled_red, y_resampled_red = smote_nc.fit_resample(X_train_red, y_train_red)


# In[21]:


from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

params_to_test = {
    'n_estimators':[i for i in range(1, 100, 25)],
    'max_depth':[i for i in range(1, 100, 25)],
    'min_samples_split':[i for i in np.arange(0.1, 1, 0.25)],
    'min_samples_leaf':[i for i in np.arange(0.1, 0.5, 0.25)],
    'max_features':[float(i) for i in np.arange(0.1, 1, 0.25)]    
}


# In[22]:


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_jobs=-1, random_state=42)


# Let's try to find the best parameter using GridSearchCV with the combination of SMOTE and feature reduction

# ### A. With SMOTE

# #### A. 1. Without feature reduction

# In[23]:


import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

grid_search = GridSearchCV(
    rf_model,
    param_grid=params_to_test,
    cv=5,
    scoring=scoring,
    refit='AUC',
    n_jobs=-1
)

grid_search.fit(X_resampled, y_resampled)

best_params = grid_search.best_params_ 
best_model = RandomForestClassifier(**best_params)


# In[24]:


best_model.get_params()


# In[25]:


best_model.fit(X_resampled, y_resampled)
mean_accuracy = best_model.score(X_test, y_test)

print(
  "Mean accuracy of RF model auto-parameter with SMOTE and"
  "without feature reduction:",
  mean_accuracy
)


# #### A. 1. With feature reduction

# In[26]:


grid_search = GridSearchCV(
    rf_model,
    param_grid=params_to_test,
    cv=5,
    scoring=scoring,
    refit='AUC',
    n_jobs=-1
)

grid_search.fit(X_resampled_red, y_resampled_red)

best_params = grid_search.best_params_ 
best_model = RandomForestClassifier(**best_params)


# In[27]:


best_model.get_params()


# In[28]:


best_model.fit(X_resampled_red, y_resampled_red)
mean_accuracy = best_model.score(X_test_red, y_test_red)

print(
  "Mean accuracy of RF model auto-parameter with SMOTE and feature reduction:",
  mean_accuracy
)


# ### B. Without SMOTE

# #### B. 1. Without feature reduction

# In[29]:


grid_search = GridSearchCV(
    rf_model,
    param_grid=params_to_test,
    cv=5,
    scoring=scoring,
    refit='AUC',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_ 
best_model = RandomForestClassifier(**best_params)


# In[30]:


best_model.get_params()


# In[31]:


best_model.fit(X_train, y_train)
mean_accuracy = best_model.score(X_test, y_test)

print(
  "Mean accuracy of RF model auto-parameter without SMOTE "
  "and feature reduction:",
  mean_accuracy
)


# #### B. 2. With feature reduction

# In[32]:


grid_search = GridSearchCV(
    rf_model,
    param_grid=params_to_test,
    cv=5,
    scoring=scoring,
    refit='AUC',
    n_jobs=-1
)

grid_search.fit(X_train_red, y_train_red)

best_params = grid_search.best_params_ 
best_model = RandomForestClassifier(**best_params)


# In[33]:


best_model.get_params()


# In[34]:


best_model.fit(X_train_red, y_train_red)
mean_accuracy = best_model.score(X_test_red, y_test_red)

print(
  "Mean accuracy of RF model auto-parameter without SMOTE"
  " but with reduction:",
  mean_accuracy
)


# In[35]:


ws_wred = {
    'bootstrap': True,
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': 1,
    'max_features': 0.1,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'min_impurity_split': None,
    'min_samples_leaf': 0.1,
    'min_samples_split': 0.1,
    'min_weight_fraction_leaf': 0.0,
    'n_estimators': 76,
    'n_jobs': None,
    'oob_score': False,
    'random_state': None,
    'verbose': 0,
    'warm_start': False
}

ws_wtred = {
    'bootstrap': True,
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': 26,
    'max_features': 0.1,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'min_impurity_split': None,
    'min_samples_leaf': 0.1,
    'min_samples_split': 0.1,
    'min_weight_fraction_leaf': 0.0,
    'n_estimators': 76,
    'n_jobs': None,
    'oob_score': False,
    'random_state': None,
    'verbose': 0,
    'warm_start': False
}


# In[36]:


ws_wred == ws_wtred


# | Method | Score |
# | - | - |
# | With SMOTE, without feature reduction | 0.656 |
# | With SMOTE and feature reduction | 0.783 |
# | Without SMOTE and feature reduction | 0.864 |
# | Without SMOTE, with feature reduction | 0.864 |

# Although there is an imbalance in the data, adding SMOTE-NC only make it worst to the score on test dataset. Probably the SMOTE isn't balanced enough. While without SMOTE, for both with and without feature reduction, they are giving the same score. The only differences between their recommended-parameter is the `max_depth`. I will go with the **Without SMOTE, with feature reduction** method.
# 
# Because that means 2 things:
# 
# 1. We can bypass the SMOTE process
# 2. Fewer feature
# 
# In conclusion we can save up computational resources with that.

# ### Manual Inspection

# In[37]:


best_param = {
    'bootstrap': True,
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': 1,
    'max_features': 0.1,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'min_impurity_split': None,
    'min_samples_leaf': 0.1,
    'min_samples_split': 0.1,
    'min_weight_fraction_leaf': 0.0,
    'n_estimators': 76,
    'n_jobs': None,
    'oob_score': False,
    'random_state': None,
    'verbose': 0,
    'warm_start': False
}


# #### `max_depths`

# In[38]:


from sklearn.metrics import roc_curve, auc

max_depths = list(range(1, 20, 1))

train_results = []
test_results = []

for max_depth in max_depths:
  
   rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1, random_state=42)
   rf.fit(X_resampled_red, y_resampled_red)
   train_pred = rf.predict(X_resampled_red)
    
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_resampled_red, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test_red)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_red, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)


# In[39]:


from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()


# #### `max_features`

# In[40]:


max_features = list(range(1,df[feature_].shape[1]))

train_results = []
test_results = []

for max_feature in max_features:
  
   rf = RandomForestClassifier(max_features=max_feature)
   rf.fit(X_resampled_red, y_resampled_red)
   train_pred = rf.predict(X_resampled_red)
   
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_resampled_red, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test_red)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_red, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)


# In[41]:


line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('max features')
plt.show()


# Our manual inspection is showing a little bit different value for the parameter. But to keep in mind, while trying to see it manually, we leave the rest parameter to its default.
# 
# * max_depth = None
# * max_leaf_nodes = None
# * min_samples_leaf = 1
# * min_samples_split = 2
# 
# The default parameters do not prevent overfitting. This allows the model to go all the way to an mse (mean squared error) of 0. Like in the `max_features`, they gave 1 easily to the train data.

# In[96]:


get_ipython().system('pip install forestci duecredit')


# In[75]:


import forestci as fci

responding = np.where(y_test_red == 1)[0]
not_responding = np.where(y_test_red == 0)[0]


# In[76]:


rf_model = RandomForestClassifier(**best_param)


# In[77]:


rf_model.fit(X_train_red, y_train_red)


# In[87]:


prediction = rf_model.predict_proba(X_test_red)
probability = prediction[:,1]


# In[94]:


fig, ax = plt.subplots(1)
ax.hist(prediction[responding, 1], histtype='step', label='Responding')
ax.hist(prediction[not_responding, 0], histtype='step', label='Not Responding')
ax.set_xlabel('Prediction (responding probability)')
ax.set_ylabel('Number of observations')
plt.legend()


# The best parameter not good enough in predicting `Responding` data. But good enough in predicting `Not Responding` data.

# ### Predict the new data

# In[42]:


rf_model = RandomForestClassifier(**best_param)
rf_model.get_params()


# In[43]:


rf_model.fit(X_train_red, y_test_red)


# In[44]:


test_data = pd.read_excel(
    'https://github.com/bukanyusufandroid/dataset/'
    'blob/master/XL%20Case%20Study/XL%20AA%20Interview'
    '%20-%20Data.xlsx?raw=true',
    sheet_name=1
)

test_data = test_data[[i for i in feature_ if i != 'Response']]


# In[45]:


test_data.head(3)


# In[46]:


pred = rf_model.predict(test_data)
pred_proba = rf_model.predict_proba(test_data)


# In[47]:


test_data['Response'] = pred
test_data['Probability 0'] = pred_proba[:,0]
test_data['Probability 1'] = pred_proba[:,1]


# In[48]:


test_data.head(3)


# _To do: Evaluate the model on the new data_

# ---
# 
# ### Conclusion From This Playground (WIP)
# 
# 1. Random Forest can handle binary features, categorical features, and numerical features. There is very little pre-processing that needs to be done. The data does not need to be rescaled or transformed.
# 2. As we see before that with or without feature reduction it's having the same score. It means that Random forests is great with high dimensional data.
# 3. Random Forest works well without SMOTE, means that it can handle imbalance problem within the dataset
# 

# In[ ]:




