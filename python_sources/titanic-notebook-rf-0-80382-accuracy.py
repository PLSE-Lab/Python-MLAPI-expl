#!/usr/bin/env python
# coding: utf-8

# Define the file paths of training and testing data sets

# In[ ]:


train_file_path = '/kaggle/input/titanic/train.csv'
test_file_path = '/kaggle/input/titanic/test.csv'


# In[ ]:


import numpy as np 
import pandas as pd


# Read the training and testing dataset

# In[ ]:


train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)


# In[ ]:


train_df.info()


# After reading the discussion forums, extracting the **Title** from the name made sense. 
# 
# I use the following method to get a list of titles from the training set. I have seen some notebooks where the names from test set where also used to get the list of titles. But I do not use any information from test set because it would result in information leak.

# In[ ]:


import re
#get the list of titles which are atleast more than 2 from the combined training and testing set of names
def getValidTitles(train_df):    
    all_names = list(train_df.Name.values)
    all_names = " ".join(n for n in all_names)
    all_titles = re.findall(r'([A-Za-z]+)\.\W', all_names)
    title_counts = [(x,1) for x in all_titles]
    title_df = pd.DataFrame(title_counts, columns = ['Title', 'Count'])
    title_df = title_df.groupby('Title').sum()
    common_titles = title_df[title_df.Count > 2].index.values
    return common_titles

VALID_TITLES = getValidTitles(train_df)


# Out of the available features in the raw data, not all features make sense to be used. 
# 
# Lets us define some functions to add additional features like **Title** and whether a person boarded as an **Individual**.
# Further let us define a class to add these features. This is useful for creating a pipeline with scikit-learn pipeline and column transformers.

# In[ ]:


def getTitle(name, title_scope):
    matches = re.findall(r'([A-Za-z]+)\.\W', name)
    if len(matches) == 0:
        title = 'NA'
    else:
        if matches[0] in title_scope:
            title = matches[0] #first matching pattern in the list
        else:
            title = 'NA'
    return title

def getIsIndividual(row):
    if pd.isna(row.SibSp) or pd.isna(row.Parch):
        individual = 1
    elif (row.SibSp + row.Parch) > 0:
        individual = 0
    else:
        individual = 1
    return individual

from sklearn.base import BaseEstimator, TransformerMixin

class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, common_titles):
       self.feature_list =  ['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'Individual']
       self.title_scope = common_titles
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X['Title'] = X['Name'].apply(lambda name: getTitle(name, self.title_scope))
        X['Individual'] = X.apply(lambda row: getIsIndividual(row), axis=1)
        return X[self.feature_list]


# Now use the defined class and get the processed train and test set.  
# 
# I create a pipeline as follows: 
#  - First step is to use the Feature transformer class which adds additional features and keeps only the required features.  
#  - Next the numerical pipeline is defined which imputes the missing values with mean, followed by minmax scaling (scaling is not required for RF)
#  - Next the categorical features are encoded
#  - A column transformer defines which columns are to be passed to numerical pipeline and categorical pipeline
#  - Final pipeline sequentially joins these Feature transformer and column transformer
# 
# The list of features that I get after passing through the pipeline are:
# 1. Pclass: Encoded
# 2. Sex: Encoded
# 3. Age: Missing values are imputed with mean and then minmax scaled (actually scaling is not required for RF, but i kept it because I wanted a common pipeline to try other models like Logistic Regression)
# 4. Fare: Missing values are imputed with mean and then minmax scaled
# 5. Title: Encoded
# 6. Individual: 1 (if SibSp + Parch = 0) else 0
# 
# **Note: These features gave the best result on the submission set. Initially I had also used **Embarked** feature, but the accuracy was low. So I removed this feature.** 

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

feature_transformer = FeatureTransformer(VALID_TITLES)
num_pipeline = Pipeline(steps = [('imputer', SimpleImputer(strategy='mean')), ('minmax_scale', MinMaxScaler())])
cat_pipeline = Pipeline(steps = [('encode_PClass', OrdinalEncoder())])
column_transformer = ColumnTransformer(transformers=[('num_transformer', num_pipeline, ['Age', 'Fare']),
                                                    ('cat_transformer', cat_pipeline, ['Pclass', 'Sex', 'Title', 'Individual'])])

data_pipeline = Pipeline(steps = [('add_select_features', feature_transformer), 
                                  ('column_transformations', column_transformer)], verbose=True)


# Now let us transform the raw data using the pipeline

# In[ ]:


X = data_pipeline.fit_transform(train_df) #use fit transform on the training data
X_test = data_pipeline.transform(test_df) #use transform on the test data
y = train_df['Survived'].values


# Split the train set further into a hold-out set (20%)
# 
# 
# (Initially I used 5% and 10%, but the submisison accuracy was low because of overfitting. Finally using 20% hold-out set from the training data helped in getting a generalized model) 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a RF classifier using grid search with 3 fold cross validation

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score

param_grid = [{'n_estimators': [50, 100, 200, 300], 'max_depth': [10,20,30], 'bootstrap': [True, False]}]
grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, scoring='accuracy', cv=3, verbose=True)
grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_


# In[ ]:


model = grid_search.best_estimator_


# Evalute the model

# In[ ]:


y_train_hat = model.predict(X_train)
y_train_hat_prob = model.predict_proba(X_train)[:,1]
accuracy = accuracy_score(y_train, y_train_hat,)
auc = roc_auc_score(y_train, y_train_hat_prob)
print("Model performance on the training set:\nAccuracy: {:4f}\nAUC: {:4f}".format(accuracy, auc))
y_val_hat = model.predict(X_val)
y_val_hat_prob = model.predict_proba(X_val)[:,1]
accuracy = accuracy_score(y_val, y_val_hat,)
auc = roc_auc_score(y_val, y_val_hat_prob)
print("Model performance on the hold out set:\nAccuracy: {:4f}\nAUC: {:4f}".format(accuracy, auc))


# predictions on the test set

# In[ ]:


y_test_hat = model.predict(X_test)


# In[ ]:


sub_df = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': y_test_hat})


# In[ ]:


sub_df.to_csv('submit_final.csv', index=False)

