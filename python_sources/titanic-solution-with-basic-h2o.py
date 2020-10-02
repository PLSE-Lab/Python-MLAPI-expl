#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


#Basic Packages
import pandas as pd
import numpy as numpy

#H2O
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator

#Evaluation Packages
from sklearn import metrics
from sklearn.metrics import roc_auc_score


# In[ ]:


#Initialize H2O
h2o.init()


# # Import train and test Datasets

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# ## Check for Missing Values

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# # Treat Missing Values

# In[ ]:


all = pd.concat([train, test], sort = False)
all.info()


# In[ ]:


#Fill Missing numbers with median for Age and Fare
all['Age'] = all['Age'].fillna(value=all['Age'].median())
all['Fare'] = all['Fare'].fillna(value=all['Fare'].median())

#Treat Embarked
all['Embarked'] = all['Embarked'].fillna(value=all['Embarked'].mode()[0])

#Bin Age
all.loc[ all['Age'] <= 16, 'Age'] = 0
all.loc[(all['Age'] > 16) & (all['Age'] <= 32), 'Age'] = 1
all.loc[(all['Age'] > 32) & (all['Age'] <= 48), 'Age'] = 2
all.loc[(all['Age'] > 48) & (all['Age'] <= 64), 'Age'] = 3
all.loc[ all['Age'] > 64, 'Age'] = 4 

#Cabin
all['Cabin'] = all['Cabin'].fillna('Missing')
all['Cabin'] = all['Cabin'].str[0]

#Family Size & Alone 
all['Family_Size'] = all['SibSp'] + all['Parch'] + 1
all['IsAlone'] = 0
all.loc[all['Family_Size']==1, 'IsAlone'] = 1


# In[ ]:


all.isnull().sum()


# ## Extra Features: Title

# In[ ]:


#Extract Title from Name
all['Title'] = all['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


all['Title'].value_counts()


# In[ ]:


#We will combine a few categories, since few of them are unique 
all['Title'] = all['Title'].replace(['Capt', 'Dr', 'Major', 'Rev'], 'Officer')
all['Title'] = all['Title'].replace(['Lady', 'Countess', 'Don', 'Sir', 'Jonkheer', 'Dona'], 'Royal')
all['Title'] = all['Title'].replace(['Mlle', 'Ms'], 'Miss')
all['Title'] = all['Title'].replace(['Mme'], 'Mrs')
all['Title'].value_counts()


# In[ ]:


#Drop unwanted variables
all = all.drop(['Name', 'Ticket'], axis = 1)
all.head()


# ## Create Dummy Values
# We will drop one of them using drop_first = True

# In[ ]:


all_dummies = pd.get_dummies(all, drop_first = True)
all_dummies.head()


# ## Covert Pandas Dataframe to H2O Frame

# In[ ]:


all_train = h2o.H2OFrame(all_dummies[all_dummies['Survived'].notna()])
all_test = h2o.H2OFrame(all_dummies[all_dummies['Survived'].isna()])


# # Train/Test Split

# In[ ]:


#Get columns names for Building H2O Models
target = 'Survived'
predictors = [f for f in all_train.columns if f not in ['Survived','PassengerId']]


# ### Diving the dataset into Train, Validation and Test
# - **Train:** will be used to build model
# - **Validation** is used to help improve the evaluation metric (We will not use this in this kernel)
# - **Test** is used to help us evaluate the model we built

# In[ ]:


train_df, valid_df, test_df = all_train.split_frame(ratios=[0.7, 0.15], seed=2018)


# In[ ]:


#Covert dtype to factor as per H2O implementation
train_df[target] = train_df[target].asfactor()
valid_df[target] = valid_df[target].asfactor()
test_df[target] = test_df[target].asfactor()


# # Build Model

# In[ ]:


#Check X Variables
predictors


# In[ ]:


# initialize the H2O GBM 
gbm = H2OGradientBoostingEstimator()

# train with the initialized model
gbm.train(x=predictors, y=target, training_frame=train_df)


# In[ ]:


#Predict on Test Frame to evaluate how well our model performed
#as_data_frame() converts the data to Pandas DataFrame
pred_val = gbm.predict(test_df[predictors])[0].as_data_frame()
pred_val


# # Check Accuracy

# In[ ]:


true_val = (test_df[target]).as_data_frame()
prediction_auc = roc_auc_score(pred_val, true_val)
prediction_auc


# # Final Predictions for Competition

# In[ ]:


#Get X Variables from Competition Test Dataset
TestForPred = all_test.drop(['PassengerId', 'Survived'], axis = 1)


# In[ ]:


#Predict
fin_pred = gbm.predict(TestForPred[predictors])[0].as_data_frame()


# In[ ]:


#Get Competition Test Ids
PassengerId = all_test['PassengerId'].as_data_frame()


# In[ ]:


#Make Submission File
h2o_Sub = pd.DataFrame({'PassengerId': PassengerId['PassengerId'].tolist(), 'Survived':fin_pred['predict'].tolist() })
h2o_Sub.head()


# In[ ]:


#Export Submission File
h2o_Sub.to_csv("1_h2o_Submission.csv", index = False)

