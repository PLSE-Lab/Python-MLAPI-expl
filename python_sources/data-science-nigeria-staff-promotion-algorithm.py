#!/usr/bin/env python
# coding: utf-8

# # __Data Science Nigeria Staff Promotion Algorithm__

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
from sklearn import preprocessing


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head(2)


# In[ ]:


train.shape


# In[ ]:


train.describe()


# In[ ]:


for i in train.columns:
    print (i + ": "+str(sum(train[i].isnull()))+" missing values")


# In[ ]:


for i in test.columns:
    print (i + ": "+str(sum(test[i].isnull()))+" missing values")


# In[ ]:


# Looking at categorical values
def cat_exploration_train(column):
    return train[column].value_counts()
def cat_exploration_test(column):
    return test[column].value_counts()


# In[ ]:


# Imputing the missing values
def cat_imputation_train(column, value):
    train.loc[train[column].isnull(),column] = value
def cat_imputation_test(column, value):
    test.loc[test[column].isnull(),column] = value


# In[ ]:


# first degree or hnd occurs the most
cat_exploration_train('Qualification')


# In[ ]:


# first degree or hnd occurs the most
cat_exploration_test('Qualification')


# In[ ]:


# since first degree or hnd occurs highest, null value = this
cat_imputation_train('Qualification', 'First Degree or HND')


# In[ ]:


# since first degree or hnd occurs highest, null value = this
cat_imputation_test('Qualification', 'First Degree or HND')


# In[ ]:


for i in train.columns:
    print (i + ": "+str(sum(train[i].isnull()))+" missing values")


# In[ ]:


for i in test.columns:
    print (i + ": "+str(sum(test[i].isnull()))+" missing values")


# In[ ]:


cat_exploration_train('Last_performance_score')


# In[ ]:


cat_exploration_test('Last_performance_score')


# In[ ]:


train['age'] = 2019 - train['Year_of_birth']
train.drop('Year_of_birth', axis=1)


# In[ ]:


test['age'] = 2019 - test['Year_of_birth']
test.drop('Year_of_birth', axis=1)


# In[ ]:


train.columns


# In[ ]:


# employee has won both awards and met target
train["Target&Award"] = np.where(((train["Targets_met"]==1) & (train["Previous_Award"]==1)),1,0)
test["Target&Award"] = np.where(((test["Targets_met"]==1) & (test["Previous_Award"]==1)),1,0)


# In[ ]:


train.columns


# In[ ]:


test.columns


# In[ ]:


train_features_eng = train
train_features_eng = train_features_eng.drop(['EmployeeNo','Channel_of_Recruitment','State_Of_Origin','Year_of_birth','No_of_previous_employers'],axis=1) #features that are not needed


# In[ ]:


test_features_eng = test
test_features_eng = test_features_eng.drop(['EmployeeNo','Channel_of_Recruitment','State_Of_Origin','Year_of_birth','No_of_previous_employers'],axis=1) #features that are not needed


# In[ ]:


train_features_eng.head(2)


# In[ ]:


test_features_eng.head(2)


# In[ ]:


train[train['Trainings_Attended'] > 5] = 6
test[test['Trainings_Attended'] > 5] = 6


# In[ ]:


# Encode all categorical features
train_features_eng=pd.get_dummies(train_features_eng, columns=["Division","Qualification", "Foreign_schooled", "Marital_Status", "Past_Disciplinary_Action", "Previous_IntraDepartmental_Movement", "Gender"], 
                                  prefix=["Div", "Qua", "ForSc", "MS", "PsDA", "PrvID", "Gd"])
train_features_eng.head()


# In[ ]:


# Encode all categorical features
test_features_eng=pd.get_dummies(test_features_eng, columns=["Division","Qualification", "Foreign_schooled", "Marital_Status", "Past_Disciplinary_Action", "Previous_IntraDepartmental_Movement", "Gender"], 
                                  prefix=["Div", "Qua", "ForSc", "MS", "PsDA", "PrvID", "Gd"])
test_features_eng.head()


# In[ ]:


train_features_eng.info()


# In[ ]:


x_train = train_features_eng.drop(["Promoted_or_Not"],axis=1)
y_train = train_features_eng['Promoted_or_Not']


# In[ ]:


from sklearn import preprocessing
# Get column names first
# names = x_train.columns
# Create the Scaler object
std_scale = preprocessing.StandardScaler().fit(x_train)
x_train_norm = std_scale.transform(x_train)


training_norm_col = pd.DataFrame(x_train_norm, index=x_train.index, columns=x_train.columns) 
x_train.update(training_norm_col)
print (x_train.head())


# In[ ]:


x_test = test_features_eng


# In[ ]:


# Normalize Testing Data by using mean and SD of training set
x_test_norm = std_scale.transform(x_test)
testing_norm_col = pd.DataFrame(x_test_norm, index=x_test.index, columns=x_test.columns) 
x_test.update(testing_norm_col)
print (x_test.head())


# In[ ]:


x_train.head(2) 


# In[ ]:


x_test.head(2) 


# In[ ]:


xgclass1 = xgb.XGBClassifier(max_depth=9, n_estimators=455, learning_rate=0.015)
xgclass2 = xgb.XGBClassifier(max_depth=9, base_estimator=xgclass1, n_estimators=455, learning_rate=0.015)
# xgclass3 = xgb.XGBClassifier(max_depth=9, base_estimator=xgclass2, n_estimators=455, learning_rate=0.015)
xgclass = xgb.XGBClassifier(max_depth=9, base_estimator=xgclass2, n_estimators=455, learning_rate=0.015)
xgclass.fit(x_train,y_train)


# In[ ]:


predictions = xgclass.predict(x_test)


# In[ ]:


predictions = predictions.astype(int)
submission = pd.DataFrame({
"EmployeeNo": test["EmployeeNo"],
"Promoted_or_Not": predictions
})

submission.to_csv("submission.csv", index=False)


# In[ ]:




