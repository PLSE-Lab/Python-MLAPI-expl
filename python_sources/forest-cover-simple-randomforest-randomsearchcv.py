#!/usr/bin/env python
# coding: utf-8

# This is a very basic solution of forestCover problem using RandomForest and followed by hyperparameter tuning  using RandomSerachCV 

# ### Fetch Data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd 
train = pd.read_csv('../input/learn-together/train.csv')
test = pd.read_csv('../input/learn-together/test.csv')


# ### Explore data

# In[ ]:


train.isnull().sum()


# Data set has no missing data

# In[ ]:


train.head()


# In[ ]:


train.info()


# Data set  features ('Wilderness_Area' & 'Soil_Type') are already one hot encoded 

# In[ ]:


#no of unique values in each feature
for column in list(train.columns):
    print ("{0:25} {1}".format(column, train[column].nunique()))


# Features 'Soil_Type7' and 'Soil_Type15' has only single value hence will be dropped before modeling.
# 

# ### Data Modeling

# #### Train-test split 

# In[ ]:


#  droping not so useful training columns 
dropable_attributes = ['Id','Soil_Type7','Soil_Type15','Cover_Type']
X = train.drop((dropable_attributes), axis =1)
y = train['Cover_Type']

# creating test-train set  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=42, test_size=.20)
 


# #### Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
# define
rf = RandomForestClassifier()
# train
rf.fit(X_train,y_train)


# #### cross validation accuracy measure

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
acc = accuracy.mean()
acc


# Lets do hyperparameter tuning using RandomSearchCV

# ### Randomized SearchCV

# In[ ]:


#### random search cv 
# Note: this code block will take time  to execute
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

 
params = {
        "n_estimators": [200, 400, 600   ],
        "criterion" : ["entropy", "gini"],
        "max_depth" : [  20, 40, 60],
        "max_features" : [ .30, .50 , .70 ],    
        "bootstrap" : [True, False]
           }

rs = RandomizedSearchCV(rf, params, cv=3, scoring='accuracy',verbose=10)
rs.fit(X_train, y_train) 


# In[ ]:


rs.best_params_


# Lets do cross validation accuracy measure for rs model

# In[ ]:


rf_rs = RandomForestClassifier(n_estimators= 400,max_features =0.3,
                               max_depth =40,criterion ='entropy', bootstrap= False )
rf_rs.fit(X_train,y_train)
accuracy = cross_val_score(rf_rs , X, y, cv=5, scoring='accuracy')
acc = accuracy.mean()
acc


# After RandomizedSearch, model accuracy has improved from .75 to .801
# Lets create submission file for leaderboard score

# #### Prepare Submission file 

# In[ ]:


def create_submission_file( predictions, name):
    submission = pd.DataFrame()
    submission['ID'] = test['Id']     
    submission['Cover_Type'] = predictions
    submission.to_csv( name+'.csv',index=False, header= True)


# In[ ]:


testcopy = test.drop((['Id','Soil_Type7','Soil_Type15']), axis =1)
predictions = rf_rs.predict(testcopy) 
create_submission_file( predictions, 'out')

