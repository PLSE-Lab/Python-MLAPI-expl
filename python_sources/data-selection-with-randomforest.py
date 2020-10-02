#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv("../input/train.csv")


# # 1. Target are per household

# In[ ]:


target_per_household = df.groupby(['idhogar'])['Target'].nunique()

no_target = len(target_per_household.loc[target_per_household == 0])
unique_target = len(target_per_household.loc[target_per_household == 1])
more_targets = len(target_per_household.loc[target_per_household > 1])
more_targets_perc = more_targets / (no_target + unique_target + more_targets)

print("No per household: {}".format(no_target))
print("1 target per household: {}".format(unique_target))
print("More targets per household: {} or {:.1f}%" .format(more_targets, more_targets_perc * 100))


# As in the competition title, "Household Poverty Level Prediction", we will consider the Target per household, and define the other as **outliers** that we will in a first time **delete**.
# 
# Kaggle discussion, mention to clean the data using the household value in caseof discrepency: https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403

# In[ ]:


df.loc[(df.idhogar == '1b31fd159'), 'meaneduc'] = 10
df.loc[(df.idhogar == 'a874b7ce7'), 'meaneduc'] = 5
df.loc[(df.idhogar == 'faaebf71a'), 'meaneduc'] = 12
df.edjefe.replace({'no': 0}, inplace=True)
df.edjefa.replace({'no': 0}, inplace=True)
df.edjefe.replace({'yes': 1}, inplace=True)
df.edjefa.replace({'yes': 1}, inplace=True)


# # 2. Categorical features

# In[ ]:


categorical_features = df.columns.tolist()
for feature in df.describe().columns:
    categorical_features.remove(feature)

# Just for saving them
numerical_features = df.columns.tolist()
for categorical_feature in categorical_features:
    numerical_features.remove(categorical_feature)
    
categorical_features


# In[ ]:


df[categorical_features].head()


# Features ID, those will obviously not beeing predictive (or will overfit), so we can ignore:
# - Id
# - idhogar
# 
# Other categorical features:
# - **dependency**': Dependency rate. We can use its squraed feature **SQBdependency**.
# - **edjefe**, years of education of male head of household. We can use its squared feature **SQBedjefe**
# - **edjefa**, years of education of female head of household.

# In[ ]:


df[['edjefe', 'SQBedjefe']].head()


# In[ ]:


df[['edjefe', 'SQBedjefe']].head()


# # 3. Empty values in numerical features

# In[ ]:


print("Number of observations {}".format(len(df)))


# In[ ]:


features_with_null = df.isna().sum().sort_values(ascending=False)
features_with_null = features_with_null.loc[features_with_null > 0]
feature_names_with_null = features_with_null.index.tolist()

features_with_null


# * rez_esc      7928 null values for Years behind in school. Too much null values: unusable
# * v2a1         6860 null values for Monthly rent payment. Unusable.
# 
# * v18q1        7342 null values for number of tablets household owns. Unusable but summing **v18q** by household may help.
# 
# * meaneduc        5 null values for average years of education for adults. We may fullfill those values.
# * SQBmeaned       5 null values for square of the mean years of education of adults. We may fullfill those values.
# 
# # 4. Feature selection with RandomForest

# In[ ]:


selectable_features = numerical_features.copy()
selectable_features.remove('Target')
for feature in feature_names_with_null:
    selectable_features.remove(feature)

X = df[selectable_features]
y = df.Target


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=112, test_size=0.2)


# In[ ]:


# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier()
clf.fit(X_train, y_train)

sorted(zip(X.columns, clf.feature_importances_ * 100), key=lambda x: -x[1])


# # 5. Let's take a step backward
# Out of the feature selection education is an important feature but we ignored **edjefa** and **meaneduc**

# In[ ]:


id_with_null = df[df.meaneduc.isna()].idhogar
df[df.idhogar.isin(id_with_null)][['idhogar', 'meaneduc', 'escolari', 'age']]


# We will fullfill meaneduc with the average value of the household

# In[ ]:


df.loc[(df.idhogar == '1b31fd159'), 'meaneduc'] = 10
df.loc[(df.idhogar == 'a874b7ce7'), 'meaneduc'] = 5
df.loc[(df.idhogar == 'faaebf71a'), 'meaneduc'] = 12


# In[ ]:


df.edjefe.replace({'no': 0}, inplace=True)
df.edjefa.replace({'no': 0}, inplace=True)
df.edjefe.replace({'yes': 1}, inplace=True)
df.edjefa.replace({'yes': 1}, inplace=True)


# # 6. Second evaluation
# How to resist?

# In[ ]:


selectable_features = numerical_features.copy()
selectable_features.append('edjefe')
selectable_features.append('edjefa')
selectable_features.remove('Target')
for feature in feature_names_with_null:
    selectable_features.remove(feature)

X = df[selectable_features]
y = df.Target


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=112, test_size=0.2)


# In[ ]:


selected_features = ['SQBedjefe', 'SQBdependency', 'overcrowding', 'qmobilephone', 'SQBage', 'rooms', 'SQBhogar_nin', 'edjefe', 'edjefa' ]

X_train_4predict = X_train[selected_features]
predictor = RandomForestClassifier()
predictor.fit(X_train_4predict, y_train)


# In[ ]:


X_test_4predict = X_test[selected_features]
y_predict = predictor.predict(X_test_4predict)


# In[ ]:


from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(y_test, y_predict)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_test, y_predict, average='macro')


# # 5. Predicting and sending

# In[ ]:


df_eval = pd.read_csv("../input/test.csv")


# In[ ]:


df_eval.edjefe.replace({'no': 0}, inplace=True)
df_eval.edjefa.replace({'no': 0}, inplace=True)
df_eval.edjefe.replace({'yes': 1}, inplace=True)
df_eval.edjefa.replace({'yes': 1}, inplace=True)


# In[ ]:


X_eval = df_eval[selected_features]
df_eval['Target'] = predictor.predict(X_eval)


# In[ ]:


df_eval[['Id', 'Target']].to_csv("sample_submission.csv", index=False)


# Kaggle gave me a result of 0.349  on this first try, which show a huge overfitt, which is also normal with a model like random forest.
# 
# I also ranked 82 / 106.
