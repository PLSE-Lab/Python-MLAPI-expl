#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# We have done an EDA (Exploratory Data Analysis) and applied different machine learning models. Some of them, like Classification Trees, are  explanatories and help to understand the influence of the most important features. Others like XGBoost can help to achieve a better accuraty although they are black boxes. 

# ## EDA

# In[99]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[100]:


data_demographic = pd.read_csv('../input/demographic_info.csv')


# In[101]:


data_demographic


# In[102]:


data_eeg = pd.read_csv('../input/EEG_data.csv')


# In[103]:


data_eeg.head()


# In[104]:


data_eeg.info()


# In[105]:


data_eeg['SubjectID'] = data_eeg['SubjectID'].astype(int)
data_eeg['VideoID'] = data_eeg['VideoID'].astype(int)
data_eeg['predefinedlabel'] = data_eeg['predefinedlabel'].astype(int)
data_eeg['user-definedlabeln'] = data_eeg['user-definedlabeln'].astype(int)


# In[106]:


data_eeg.iloc[:, 2:].describe()


# In[107]:


data_eeg['user-definedlabeln'].value_counts()


# In[108]:


data_resume = data_eeg.groupby(['SubjectID', 'VideoID'])['user-definedlabeln'].agg(lambda x: sum(x) > 0).unstack("VideoID")
data_resume


# In[109]:


fig = plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
data_resume.apply(sum).plot(kind='bar', title='Number of subjets surprised by a video')
plt.subplot(1, 2, 2)
data_resume.apply(sum, axis=1).plot(kind='bar', title="Number of videos that surprised a subject")
plt.show()


# It looks like that movie #8 is the one that most confusion generates. Users #4, #5, #7 are the most confused ones.

# In[110]:


data_user1_video1 = data_eeg.query('SubjectID==0 & VideoID==0')


# In[111]:


len(data_user1_video1)


# Let's see the time serie for a user and a video:

# In[112]:


features = ['Attention', 'Mediation', 'Raw', 'Delta',
            'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']


# In[113]:


data_user1_video1[features].plot(figsize=(18,6))
plt.show()


# This person doesn't feel confused at any time of the video:

# In[114]:


(data_user1_video1['user-definedlabeln']!=0).sum()


# ### Data Cleaning
# As only 1 minute of each video was shown to each person, and sampling rate was 0.5 records per second, each person/video should have 60*2=120 rows. The reality is that there are slightly differences. Even for the same video, the number of records change from person to person:**

# In[115]:


data_eeg.groupby(['SubjectID', 'VideoID']).size().loc[(slice(None), 2)].plot(kind='bar', figsize=(12,6))
plt.title("Number of rows per user for video #3")
plt.ylabel("Number of rows")
plt.show()


# There are not missing values:

# In[116]:


data_eeg.isnull().any().sum()


# But there is something strange:

# In[117]:


data_eeg['Attention'].plot(figsize=(18,6))
plt.show()


# In[118]:


data_eeg['Mediation'].plot(figsize=(18,6))
plt.show()


# In[119]:


data_eeg['Raw'].plot(figsize=(18,6))
plt.show()


# There are some row with bad data. As they are together they could come from the same user. Let's check:

# In[120]:


data_eeg.groupby(['SubjectID', 'VideoID']).filter(lambda x: x['Attention'].sum()==0).groupby(['SubjectID', 'VideoID']).size()


# In[121]:


data_eeg.groupby(['SubjectID', 'VideoID']).filter(lambda x: x['Mediation'].sum()==0).groupby(['SubjectID', 'VideoID']).size()


# It looks like that all records from user #6 have problems and it also the same case for records from user #3 for the specific movie #3
# Looking at the rest of the cases it looks like these are the only ones which have errors and we should remove them.

# In[122]:


data = data_eeg.query('(SubjectID != 6) & (SubjectID != 3 | VideoID !=3)')
len(data), len(data_eeg)


# In[123]:


data.reset_index()['Attention'].plot(figsize=(18,6))
plt.show()


# Let'c check correlation:

# In[124]:


corr = data[features].corr()
corr


# In[125]:


import seaborn as sns
plt.figure(figsize = (12, 12))
sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap="RdBu_r")
plt.show()


# As it is clear that there are high correlation between some variables, we could try later to do some feature selection and dimension reduction to see if some models achieve better results this way.

# ## Models

# ###  Baseline model (Logistic Regression)
# Let's try an easy model only to test. This would be our reference model.

# In[126]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
seed = 123


# In[127]:


X = data[features]
y = data['user-definedlabeln']


# As recommended for Logistic Regression models we will normalize data:

# In[163]:


X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2, random_state=42)


# In[164]:


model_lr = LogisticRegression(random_state=seed)


# In[165]:


model_lr.fit(X_train, y_train)


# In[166]:


pred_lr = model_lr.predict(X_test)


# In[250]:


print("Test Accuracy: {:.5f}".format(accuracy_score(y_test, pred_lr)))


# ###  Interpretable model (Classification Trees)
# Let's try a model based on classification trees to understand better what are the more influential features. Trees don't required normalized data

# In[172]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[173]:


from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier(max_depth=2)
model_tree.fit(X_train, y_train)


# In[248]:


pred_tree = model_tree.predict(X_test)

print("Test Accuracy: {:.5f}".format(accuracy_score(y_test, pred_tree)))


# In[175]:


import graphviz 
from sklearn.tree import export_graphviz
tree_view = export_graphviz(model_tree, 
                            out_file=None, 
                            feature_names = features,
                            class_names = ['No confused', 'Confused'])  
tree1viz = graphviz.Source(tree_view)
tree1viz


# It looks like that Delta and Attention are  good indicators to know if a user is confused.

# ###   Improved model (XGBoost)
# Let's try to build a model more accurate. XGBoost can be a good option to try.**

# In[176]:


import xgboost as xgb


# In[177]:


model_xgb = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=seed)
model_xgb.fit(X_train, y_train)


# In[254]:


data_dmatrix = xgb.DMatrix(data=X.values, label=y.values)

# Create the parameter dictionary: params
#params = {"objective":"reg:logistic", "max_depth":5, "eta":0.1, "n_estimators":1000, "colsample_bytree": 0.7, "learning_rate": 0.1}
params = {}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=5, metrics="error", as_pandas=True, seed=seed)

print("Test Accuracy: {:.5f}".format(((1-cv_results["test-error-mean"]).iloc[-1])))


# Let's see if gender, ethnicity and age can help to get better accuracy:

# In[236]:


data_demographic.columns = ['subject ID', 'age', 'ethnicity', 'gender']


# In[237]:


data.head()


# In[238]:


data_extended = data.merge(data_demographic, left_on="SubjectID", right_on="subject ID")
data_extended.head()


# In[239]:


data_extended['ethnicity'] = data_extended['ethnicity'].astype("category").cat.codes
data_extended['gender'] = data_extended['gender'].astype("category").cat.codes
features_extra = features + ['age', 'ethnicity', 'gender']


# In[258]:


data_dmatrix = xgb.DMatrix(data=data_extended[features_extra].values, label=y.values)

# Create the parameter dictionary: params
#params = {"objective":"reg:logistic", "max_depth":5, "eta":0.1, "n_estimators":1000, "colsample_bytree": 0.7, params = {"learning_rate": 0.1}}
params = {}
# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=5, metrics="error", as_pandas=True, seed=seed)

# Print the accuracy
print("Test Accuracy: {:.5f}".format(((1-cv_results["test-error-mean"]).iloc[-1])))


# ### TODO
# 
# - feature engeneering: dimension reduction,...
# - try different hyperparameters
# - try different models
# - try stacking
