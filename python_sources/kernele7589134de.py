#!/usr/bin/env python
# coding: utf-8

# In[22]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ** Loading training and testing data **

# In[23]:


def load_data(file):
    file = '../input/'+file+'.csv'
    return pd.read_csv(file)

df_train_set = load_data("train")
df_test_set = load_data("test")


# ** Getting information on the data attributes **

# In[24]:


df_train_set.info()


# 1. The training set has 891 rows with 12 columns
# 2. Data has 5 object datatypes, 2 float datatypes and 5 integer datatypes
# 3. Of 891 rows only 204 have cabin data filled (this is a data quality issue that has to be addressed)
# 4. Embarked data is present for all the rows except 2 rows. We will handle this issue in below code
# 5. Similary Age attribute is only present for 714 rows

# In[25]:


df_train_set.head(5)


# In[26]:


pd.pivot_table(df_train_set, values="PassengerId", index="Pclass", columns="Survived",aggfunc='count',
               margins=True)


# In[27]:


func = lambda x: 100*x.count()/df_train_set.shape[0]

pd.pivot_table(df_train_set, values="PassengerId", index=["Pclass"], columns="Survived", aggfunc=func,
               margins=True, fill_value=0)


# In[28]:


pd.pivot_table(df_train_set, values="PassengerId", index="Embarked", columns="Survived",aggfunc='count',
               margins=True)


# Make a copy of the training dataset

# In[29]:


df_train_set_final = df_train_set


# Handling Data quality issues

# In[30]:


# Remove the two rows that are missing embarkation information
df_train_set_final = df_train_set_final.dropna(subset=["Embarked"])

# Remove the columns Cabin and Ticket information for this analysis
df_train_set_final.drop(columns=['Cabin','Ticket'], inplace=True)


# Checking for correlations among the variables with Survival

# In[31]:


corr_matrix_train = df_train_set_final.corr()


# In[32]:


corr_matrix_train["Survived"].sort_values(ascending=False)


# In[33]:


df_train_set_final.boxplot(by='Survived', column=['Fare'], grid = False)


# In[34]:


df_train_set_final.info()


# In[35]:


df_train_set_final.describe()


# ** Lets create a pipelines to handle both the numeric and categorical data attributes **

# * One way to handle the missing "Age" rows is to replace the age with median age *

# In[36]:


# Creating a DataFrame Selector that will pull either categorical or numerical columns
# Data Frame selector

from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		return X[self.attribute_names].values


# In[37]:


# Copying the labels into a dataset

y_train = df_train_set_final["Survived"]


# In[38]:


from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# One hot encoder for converting the categorical values into binary values
cat_encoder = OneHotEncoder(sparse=False)

# Using median strategy to replace the missing values for Age column
Imputer = SimpleImputer(strategy="median")

# List of numerical and categorical attributes
num_attribs = ["Age","SibSp","Parch","Fare"]
cat_attribs = ["Sex","Embarked","Pclass"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
cat_pipeline = Pipeline([
('selector', DataFrameSelector(cat_attribs)),
('cat_encode', OneHotEncoder(sparse=False)),
])
full_pipeline = FeatureUnion(transformer_list=[
("num_pipeline", num_pipeline),
("cat_pipeline", cat_pipeline),
])


# In[39]:


# Applying the full pipeline on the training set

X_train = full_pipeline.fit_transform(df_train_set_final)


# In[40]:


X_train


# In[41]:


X_train.shape


# ** For the first model lets use SVM model **

# In[42]:


from sklearn.svm import SVC

#svm_clf = SVC(gamma="auto)
svm_clf = SVC(gamma="auto", C=1, degree=1, kernel='rbf')

svm_clf.fit(X_train, y_train)


# In[43]:


X_test = full_pipeline.fit_transform(df_test_set)
y_pred = svm_clf.predict(X_test)


# In[44]:


from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()


# ** From the cross validation exercise we see a 82% accuracy **

# In[45]:


X_test


# In[46]:


submission = pd.DataFrame({
    "PassengerId": df_test_set["PassengerId"],
    "Survived": y_pred
})


submission.head(5)


# In[47]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




