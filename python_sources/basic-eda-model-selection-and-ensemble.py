#!/usr/bin/env python
# coding: utf-8

# # Import basics

# In[ ]:


# load basic neccesities
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

pd.options.display.max_columns = 200


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# # Loading everything

# In[ ]:


train = pd.read_csv("../input/learn-together/train.csv")
test = pd.read_csv("../input/learn-together/test.csv")


# In[ ]:


print(train.shape)
print(test.shape)


# # First impression

# In[ ]:


train.head(5)


# Lets have a look at the training data stats. 

# In[ ]:


train.describe()


# Comparing with the test data - everything looks pretty close, similar. Looks like we shall not have any problems with predictions. 

# In[ ]:


test.describe()


# In[ ]:


classes = set(train.Cover_Type)
n_classes = len(classes)
print(classes, ' - ', n_classes)


# Lets check if the dataset is balanced

# In[ ]:


sns.countplot(train.Cover_Type)


# Perfect, we have absolutely evenly split dataset in terms of classes.

# In[ ]:


# check if there are NaN cells
train.count()


# In[ ]:


test.head(5)


# In[ ]:


test.count()


# # Features analysis

# We can relax a bit - categorical features already have been encoded for us: Soil type and Wilderness Area. All other features are integers and nothing is missing. We would only need to scale them if using Neural Networks sensitive to the scale of numbers. Don't need to do that for most of tree based algorithms.

# In[ ]:


df_train = train.drop('Id', axis=1).copy()
train_cols = df_train.columns.tolist()
train_cols = train_cols[-1:] + train_cols[:-1]
df_train = df_train[train_cols]
corr = df_train.corr()


# In[ ]:


# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Looks like we don't have significant correlation between any features and the target. There are just a few features with more or less correlation between them, but nothing significant enough that would prompt to drop some of them.

# ## Soil types distribution
# Which soil types are more common in train dataset? 

# In[ ]:


df_train = train.drop('Id', axis=1).copy()
train_cols = df_train.columns.tolist()
train_cols = train_cols[-1:] + train_cols[:-1]
df_train[train_cols[15:]].sum()/df_train.shape[0]*100


# A few soil types are quite abundant, while about half of them are less than 1% of cases. Need to check correlation with the target. It could be that rare soil types will help with certain class prediction. Or otherwise will just mess the calculations.
# In any case Soil type 7 and 15 dont have any samples in train dataset. 
# 
# **Lets check test dataset now.**

# In[ ]:


test_cols = test.columns.tolist()
test[test_cols[14:]].sum()/test.shape[0]*100


# And here we have a significantly different distribution of soil types. Should probably avoid to rely on soil types as major features for predictions? 
# Also we definitely shall drop Soil types 7 and 15 since there are just a few samples in test set with them (and we won't be able to relate to prediction since there is no data in train for these features).

# # Base model

# ## Light GBM

# In[ ]:


# lets start with Light GBM
import lightgbm as lgb
from lightgbm import LGBMClassifier


# In[ ]:


# almost basic parameters. some adjustments to prevent overfitting
LGB_PARAMS_1 = {
    'objective': 'multiclass',
    "num_class" : n_classes,
    'metric': 'multiclass',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 64,
#    'num_leaves': 128,
    'max_depth': -1,
#    'subsample_freq': 10,
    'subsample_freq': 1,
    'subsample': 0.5,
    'bagging_seed': 1970,
    'reg_alpha': 0.3,
#    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'colsample_bytree': 0.90
}


# In[ ]:


# basic parameters
LGB_PARAMS = {
    'objective': 'multiclass',
    "num_class" : n_classes,
    'metric': 'multiclass',
    'reg_alpha': 0.9,
    'reg_lambda': 0.9,
    'verbosity': -1
}


# In[ ]:


X_data = train.drop(['Id','Cover_Type','Soil_Type7','Soil_Type15'], axis = 1).copy()
y_data = train.Cover_Type
cols = X_data.columns
print(X_data.shape)
print(y_data.shape)


# In[ ]:


# split train set into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=128)


# In[ ]:


# number of samples in Validation dataset
N_valid = y_val.shape[0]


# In[ ]:


LB_model = LGBMClassifier(**LGB_PARAMS, n_estimators=1500, n_jobs = -1)


# In[ ]:


LB_model.fit(X_train, y_train, 
        eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='multiclass',
        verbose=10, early_stopping_rounds=50)


# In[ ]:


LG_pred = LB_model.predict(X_val)
print(accuracy_score(LG_pred, y_val))


# In[ ]:


confusion_matrix(y_val, LG_pred)


# ## Features
# 
# Lets see quick feature importance analysis , LGBM version.

# In[ ]:


df_importance = pd.DataFrame({'feature': cols, 'importance': LB_model.feature_importances_})


# In[ ]:


# Lets plot it
plt.figure(figsize=(6, 10))
sns.barplot(x="importance", y="feature", data=df_importance.sort_values('importance', ascending=False));


# Clearly there are some feautures of outmost importance while some of the features have 0 weight in decision making.
# We have to reduce number of features, and try to engineer some new ones as well.

# In[ ]:


new_features = df_importance.loc[df_importance.importance>100].sort_values('importance', ascending=False)
new_columns = new_features.feature


# ## KNN
# Lets see how K Nearest Neighbors will do

# In[ ]:


KNN_clf = KNeighborsClassifier(n_neighbors = 5, weights ='distance', metric = 'minkowski', p = 2)
KNN_clf.fit(X_train, y_train)
yk_pred = KNN_clf.predict(X_val)
cm = confusion_matrix(y_val, yk_pred)
print(accuracy_score(yk_pred, y_val))


# In[ ]:


cm


# ## Random Forest
# 
# Now lets see how RandomForest will work.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# just default settings
#RF_clf = RandomForestClassifier(n_estimators=1000, max_depth=10,
RF_clf = RandomForestClassifier(n_estimators=1000, 
                              random_state=1970)
RF_clf.fit(X_train, y_train) 


# In[ ]:


RF_pred = RF_clf.predict(X_val)
accuracy_score(y_val, RF_pred)


# In[ ]:


confusion_matrix(y_val, RF_pred)


# LightGBM seems preforming slightly better, than RandomForest. And the KNN is the worst so far.

# In[ ]:


df_importance = pd.DataFrame({'feature': cols, 'importance': RF_clf.feature_importances_})


# In[ ]:


# Lets plot it
plt.figure(figsize=(6, 10))
sns.barplot(x="importance", y="feature", data=df_importance.sort_values('importance', ascending=False));


# ## Conclusion about the features
# Random Forest gave slightly different evalution of feature importances, however location features and Wilderness type are still most important.
# And it seems that most of Soil types were not really useful for RF as well as for LGBM.

# # Things to try
# 
# * shall try to engeneer new spatial features
# * drop unimportant features
# * try different models including Neural Nets

# # Submission 
# Need to create a dataframe for submission with 2 columns, make a prediction on test data and write it into the submission csv file. Will use LightGBM for now.

# In[ ]:


submission_id = test.Id
test.drop(['Id','Soil_Type7','Soil_Type15'], axis = 1, inplace = True)
prediction = LB_model.predict(test)
submission = pd.DataFrame({'Id': submission_id, 'Cover_Type': prediction})
submission.to_csv('submission_LGBM.csv', index = False)


# # Ensemble

# In[ ]:


#lets check Ensemble prediction on validation set
RF_val = RF_clf.predict_proba(X_val)
KNN_val = KNN_clf.predict_proba(X_val)
LG_val = LB_model.predict_proba(X_val)

val_probs = (RF_val + LG_val + KNN_val)/3
val_preds = np.argmax(val_probs, axis=1)+1

print(accuracy_score(y_val, val_preds))
# lets check the distribution of predicted classes
# as we know all classes shall be evenly split. at least they shall be close to it
sns.countplot(val_preds)


# Accuracy seems better than for separate models as expected. 
# However the distribution is slightly skewed. I'm afraid we will have much lower LeaderBoard score

# In[ ]:


# final Test set prediction
RF_probs = RF_clf.predict_proba(test)
KNN_probs = KNN_clf.predict_proba(test)
LG_probs = LB_model.predict_proba(test)


# In[ ]:


# averaging the prediction
final_probs = (RF_probs + LG_probs + KNN_probs)/3
final_preds = np.argmax(final_probs, axis=1)+1


# In[ ]:


# lets see the distribution of predicted classes
# hint - as we know class 1 shall be around 37%, and as I've checked it class 2 shall be around 50% actually
sns.countplot(final_preds)


# Looks like a bit smaller prediction for class 2. Shall be around 283000. Class 1 looks more or less good. The rest of the classes shall be smaller.

# In[ ]:


submission = pd.DataFrame({'Id': submission_id, 'Cover_Type': final_preds})
submission.to_csv('submission_ensemble.csv', index = False)

