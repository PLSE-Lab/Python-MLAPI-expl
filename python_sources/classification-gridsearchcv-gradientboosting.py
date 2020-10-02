#!/usr/bin/env python
# coding: utf-8

# # Classification with GradientBoostingClassifier, GridSearchCV and Pipeline + Preprocessing.
# 
# Classification on 'Campus Recruitment Data'.  Predicting if Placed or Not Placed.

# # Step 1: Read and Visualize, small cleanings

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


#reading dataset
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


data.head()


# In[ ]:


#Deleting 'sl_no' column
data.drop(['sl_no'], axis=1, inplace=True)
data.head()


# # Step 2: Binary encoding on target 'status' and small cleans

# In[ ]:


#data_placed = data[data['status'] == 'Placed']
dataset = data.copy()


# In[ ]:


#drop 'salary' column
dataset.drop(['salary'], axis=1, inplace=True)
dataset.head()


# In[ ]:


#binary enconding 'status' label
for i in range(len(dataset)):
    if(dataset.loc[i, 'status'] == 'Placed'):
        dataset.loc[i, 'status'] = 1
    else: dataset.loc[i, 'status'] = 0
        
dataset['status'] = dataset['status'].astype('int')


# In[ ]:


#preparing x(features) and y(target) data
X = dataset.drop(['status'], axis=1)
y = dataset['status']


# # Step 3: Split Data and Build Pipeline

# In[ ]:


#Preparing Train and Test dataset
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


# In[ ]:


#separating numerical and categorical col
numerical_col = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
categorical_col = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']


# In[ ]:


#Creating Pipeline to Missing Data 

#inpute numerical missing data with median
numerical_transformer = make_pipeline(SimpleImputer(strategy='median'),
                                      StandardScaler())

#inpute categorical data with the most frequent value of the feature and make one hot encoding
categorical_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                        OneHotEncoder(handle_unknown='ignore'))

preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_col),
                                               ('cat', categorical_transformer, categorical_col)])


# # Step 4: Build Model with GradientBoostingClassifier and GridSearchCV

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# In[ ]:


clf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingClassifier())])


# In[ ]:


#Using GradientBoostingClassifier with GridSearchCV to get better parameters

param_grid = {'model__learning_rate':[0.001, 0.01, 0.1], 
              'model__n_estimators':[100, 150, 200, 300, 350, 400]}

#param_grid = {'model__learning_rate':[0.1], 
#              'model__n_estimators':[150]}

#use recall score
grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy', n_jobs=-1)


# In[ ]:


grid.fit(X_train, y_train)


# In[ ]:


grid.best_params_


# # Step 5: Evaluating Model

# In[ ]:


#CV F1 Score
scores = cross_val_score(grid, X_test, y_test, cv=5, scoring='f1', n_jobs=-1)


# In[ ]:


print(scores, '\nAverage F1 Score: ',scores.mean())


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


predictions = grid.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

