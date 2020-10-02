#!/usr/bin/env python
# coding: utf-8

# # Feedback, please
# Please give me feedback.
# 
# I'm especially wondering about my neural network classifier which scores 50% accuracy :**(
# 
# Thank you very much,
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, KFold, cross_validate

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score as auc


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


test_data = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")
train_data = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")


print("Shape of test_data "+str(test_data.shape))

print("Shape of train_data "+str(train_data.shape))


# Going to try this person's data engineering: https://www.kaggle.com/asimandia/let-s-try-some-feature-engineering

# In[ ]:


y_train = train_data['target']
train_data_id = train_data['id']

test_data_id = test_data['id']

train_data.drop(['target','id'], axis=1,inplace=True)
test_data.drop('id', axis=1,inplace=True)

print("New shape of test_data "+str(test_data.shape))

print("New shape of train_data "+str(train_data.shape))


# In[ ]:


print("Contents of train_data\n")


train_data.head()


# In[ ]:


print("Contents of test_data\n")


test_data.head()


# In[ ]:


train_data.columns


# In[ ]:


test_data.columns


# Checking for null items in the data set

# In[ ]:


missing_val_count_by_col = train_data.isnull().sum()

print("Columns in train_data with missing values, and the number of missing values")
print(missing_val_count_by_col[missing_val_count_by_col > 0])


# Fortunately there are none

# performing train/validation set split here

# In[ ]:


X_train = train_data

print("Shape of y_train before train/validation split is"+str(y_train.shape))
print("Shape of X_train before train/validation split is"+str(X_train.shape))
print("\n")
#Split training set into a training set and validation set

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9, test_size=0.1, random_state=0)

print("Shape of y_train is"+str(y_train.shape))
print("Shape of X_train is"+str(X_train.shape))

print("Shape of y_valid is"+str(y_valid.shape))
print("Shape of X_valid is"+str(X_valid.shape))


# # Number of unique values for each feature

# In[ ]:


#Printing out the number of unique values for each column in the training data
for col_name in X_train.keys():
    print("Column " + col_name + " has " + str( len(X_train[col_name].unique()) ) + " unique values")


# Looking at nominal variables with high unique value counts

# In[ ]:


print(X_train["nom_6"].value_counts().sort_values(ascending=False))
print(X_train["nom_7"].value_counts().sort_values(ascending=False))
print(X_train["nom_8"].value_counts().sort_values(ascending=False))
print(X_train["nom_9"].value_counts().sort_values(ascending=False))


# ## Transform ordinal features to numeric labels
# Just going to transform all ordinal features to numeric labels

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nlabel_X_train = X_train.copy()\nlabel_X_valid = X_valid.copy()\nlabel_test_data = test_data.copy()\n\nsk_label_encoder = LabelEncoder()\n\nfor mycol in ["ord_0","ord_1","ord_2","ord_2","ord_3","ord_4","ord_5"]:\n    label_X_train[mycol] = sk_label_encoder.fit_transform(label_X_train[mycol])\n    label_X_valid[mycol] = sk_label_encoder.transform(label_X_valid[mycol])\n    label_test_data[mycol] = sk_label_encoder.transform(label_test_data[mycol])\n    ')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'label_X_train.head()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'label_X_valid.head()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'label_test_data.head()')


# ## Handling nominal valued features
# 
# For features that have high cardinality (>=10) will do hashing/frequency encoding
# For features that have low cardinality (<10) will do one-hot encoding
# 

# In[ ]:


low_cardinality_nom_cols = []
high_cardinality_nom_cols = []


for nom_col in range(10):
    nom_col_name = "nom_"+str(nom_col)
    if label_X_train[nom_col_name].nunique() < 10:
        low_cardinality_nom_cols.append(nom_col_name)
    else:
        high_cardinality_nom_cols.append(nom_col_name)

print("Nominal columns low cardinality (<=10):", low_cardinality_nom_cols)
print("Nominal columns with high cardinality (>10):", high_cardinality_nom_cols)


# ### Convert low cardinality nominal variables to one-hot encoded variables
# 

# In[ ]:


#combining everything into a single data frame so as to apply a uniform encoding across train, validation, test data sets
#If this is not OK please provide your feedback, with references as to why (tyvm)
label_X_train["kind"] = "train"
label_X_valid["kind"] = "valid"
label_test_data["kind"] = "test"

big_df = pd.concat([label_X_train, label_X_valid, label_test_data], sort=False ).reset_index(drop=True)

print("big_df shape is "+str(big_df.shape))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for col in low_cardinality_nom_cols:\n    temp_df_to_concat = pd.get_dummies(big_df[col], prefix=col)\n    big_df = pd.concat([big_df, temp_df_to_concat], axis=1)\n    big_df.drop([col],axis=1, inplace=True)\n\n\nfor col in high_cardinality_nom_cols:\n        big_df[f"hash_{col}"] = big_df[col].apply( lambda x: hash(str(x)) % 5000)\n        \n\n#Not sure if I can run this over all of big_df. In the example the coder runs it over df_train only\n\n#Just modify training or validation data set\n\nbig_df_train_valid = big_df.loc[ (big_df["kind"] == "train") | (big_df["kind"]=="valid") ]\nbig_df_test = big_df.loc[big_df["kind"] == "test"]\n\nfor col in high_cardinality_nom_cols:\n    enc_nom_1 =  (big_df_train_valid.groupby(col).size() ) / len(big_df_train_valid)\n    big_df_train_valid[f"freq_{col}"] = big_df_train_valid[col].apply( lambda x : enc_nom_1[x])\n\nfor col in high_cardinality_nom_cols:\n    enc_nom_1 =  (big_df_test.groupby(col).size() ) / len(big_df_test)\n    big_df_test[f"freq_{col}"] = big_df_test[col].apply( lambda x : enc_nom_1[x])\n    \nlabel_X_train = big_df_train_valid.loc[ big_df["kind"]=="train" ]\nlabel_X_valid = big_df_train_valid.loc[ big_df["kind"]=="valid" ]\nlabel_test_data = big_df_test.loc[ big_df["kind"]=="test" ]\n\nlabel_X_train.drop("kind", axis=1, inplace=True)\nlabel_X_valid.drop("kind", axis=1, inplace=True)\nlabel_test_data.drop("kind", axis=1, inplace=True)')


# In[ ]:


label_X_train.head()


# In[ ]:


label_X_valid.head()


# In[ ]:


label_test_data.head()


# In[ ]:


print("shape of label_test_data "+str(label_test_data.shape))
print("shape of label_X_train "+str(label_X_train.shape))
print("shape of label_X_valid "+str(label_X_valid.shape))

del big_df
del big_df_test
del big_df_train_valid


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n#More encoding. Borrowed idea from another notebook. Trying other things were too slow\n\nbinary_dict = {"T":1, "F":0, "Y":1, "N":0}\n\n\nlabel_X_train["bin_3"] = label_X_train["bin_3"].map(binary_dict)\nlabel_X_train["bin_4"] = label_X_train["bin_4"].map(binary_dict)\n\nlabel_X_valid["bin_3"] = label_X_valid["bin_3"].map(binary_dict)\nlabel_X_valid["bin_4"] = label_X_valid["bin_4"].map(binary_dict)\n\nlabel_test_data["bin_3"] = label_test_data["bin_3"].map(binary_dict)\nlabel_test_data["bin_4"] = label_test_data["bin_4"].map(binary_dict)\n')


# So now I'm going to drop old features
# 
# in label_X_train I will drop these:
# 
# * `high_cardinality_nom_cols`
# 
# in label_X_valid I will drop these:
# * `high_cardinality_nom_cols`
# 
# in label_test_data I will drop these:
# * `high_cardinality_nom_cols`
# 
# *Note*: In the cell above I dropped the `low_cardinality_nom_cols` as I one-hot encoded them.

# In[ ]:


label_X_train.drop(high_cardinality_nom_cols, axis=1, inplace=True)
label_X_valid.drop(high_cardinality_nom_cols, axis=1, inplace=True)
label_test_data.drop(high_cardinality_nom_cols, axis=1, inplace=True)


# In[ ]:


label_X_train.head()


# So I've massaged the training a validation data
# There should be an equal number of rows in the y_train and label_X_train pairs and the y_val, label_X_val pairs

# In[ ]:


print("Rows of label_X_train "+str(label_X_train.shape[0]))
print("Rows of y_train "+str(y_train.shape[0]))
print("Rows of label_X_valid "+str(label_X_valid.shape[0]))
print("Rows of y_valid "+str(y_valid.shape[0]))


# ## Build a Model
# [SKLearn AdaBoostClassifier FTW!](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)

# In[ ]:


ada_boost_model = AdaBoostClassifier(n_estimators=100, random_state=0, learning_rate=0.05, base_estimator=DecisionTreeClassifier(max_depth=10))


# Trying a [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

# In[ ]:


#using a StandardScaler as the sklearn documents suggest scaling the inputs
neural_model = MLPClassifier(hidden_layer_sizes=(96,96,48,48,24,12,6,3,1), 
                             solver="adam", 
                             batch_size="auto", 
                             #learning_rate="adaptive",
                             learning_rate_init=0.002,
                             max_iter=200,
                             n_iter_no_change=10,
                             random_state=1,
                             verbose=True
                            )

NNPipeline = Pipeline([("scaler",StandardScaler()), ("NN",neural_model)])


# Trying a [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

# In[ ]:


gradient_boost_model = GradientBoostingClassifier(n_estimators=50)


# So going to test each one out by doing the following:
# 
# Going to do k-fold cross validation (TODO: add reference) with `n_folds` folds
# 
# Per the competition will do ROC AUC scoring.
# 
# 

# In[ ]:


#I was thinking of doing my own train/valid split
#but realized with the cross_val_score() function, I need to
#recombine my train/validation sets and let the internal functionality of cross_val_score()
#do this splitting for me. So that's why I'm recombining them below :\
new_X_train = pd.concat([label_X_train, label_X_valid], axis=0)
new_y_train = pd.concat([y_train, y_valid], axis=0)
new_X_train_scaled = pd.DataFrame()
my_columns= new_X_train.columns


# In[ ]:


get_ipython().run_cell_magic('time', '', 'n_folds = 7\n\nkfold = KFold(n_splits=n_folds, shuffle=False, random_state=42)\n\ncv_results = cross_val_score(gradient_boost_model, new_X_train.values, new_y_train,\n                            cv=kfold, scoring=\'roc_auc\', n_jobs=-1)\n\nprint("gradient_boost_model average results",cv_results.mean())\n\ncv_results = cross_val_score(ada_boost_model, new_X_train.values, new_y_train,\n                            cv=kfold, scoring=\'roc_auc\', n_jobs=-1)\n\nprint("ada_boost_model average results",cv_results.mean())\n\n#cv_results = cross_val_score(NNPipeline, new_X_train.values, new_y_train,\n#                            cv=kfold, scoring=\'roc_auc\', n_jobs=-1)\n\n#print("NNPipeline average results",cv_results.mean())')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'gradient_boost_model.fit(new_X_train, new_y_train)')


# In[ ]:


y_test_pred = gradient_boost_model.predict(label_test_data)

myscore = gradient_boost_model.score(label_test_data)


# In[ ]:


label_test_data.head()


# In[ ]:


y_test_pred


# In[ ]:


y_test_pred.shape


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.hist(y_test_pred, density=True, bins=2)
#plt.xticks(x+0.5,['0','1'])
plt.ylabel("number of predictions")
plt.xlabel("values")


# In[ ]:


submission = pd.DataFrame({'id':test_data_id, 'target':y_test_pred})


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# 
