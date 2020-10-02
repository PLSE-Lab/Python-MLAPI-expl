#!/usr/bin/env python
# coding: utf-8

# It is necessary to predict whether the client will leave the Bank in the near future or not. You are presented with historical data on customer behavior and termination of contracts with the Bank.

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
rndd = 12345

df = pd.read_csv('/kaggle/input/bank-customer-churn-modeling/Churn_Modelling.csv')
df.info()


# ## **1. DataSet preparation**

# As we can see the data set is full without any NaN values.
# Now let's briefly see the data from the top and from the end.

# In[ ]:


df.head()


# In[ ]:


df.tail()


# The 'RowNumber' columns looks like index duplicate. Let's dropp it.

# In[ ]:


df.drop('RowNumber', axis = 1, inplace = True)
df.head()


# For a make prediction via scikit-learn, we should prepare our dataset, the should consist only numeric values. We starting from 'Gender' column.

# In[ ]:


df.Gender.unique()


# The consist only from two values, for this purpose we change values to int

# In[ ]:


df.Gender = df.Gender.map({'Female': 0, 'Male':1})
df.head()


# For columns 'Surname' and 'Geography' we will used OrdinalEncoder, for coding every string value to the int value. In general cloumns like 'Surname' should not be presented in real dataset, and they cannot affect the final result, but looking a little ahead in our case have a positive impact on the metrics.

# In[ ]:


encoder = OrdinalEncoder()
data = encoder.fit_transform(df)
df_trans = pd.DataFrame(data, columns = df.columns)
df_trans.head()


# As you can see dataset consist only from numeric values.
# <br> Now checking value types.

# In[ ]:


df_trans.info()


# We can optimize value types.

# In[ ]:


df_trans = df_trans.astype({
    'CustomerId'    : 'int32',
    'Surname'       : 'int32',
    'Geography'     : 'int32',
    'Gender'        : 'int32',
    'Age'           : 'int32',
    'Tenure'        : 'int32',
    'NumOfProducts' : 'int32',
    'HasCrCard'     : 'int32',
    'IsActiveMember': 'int32',
    'Exited'        : 'int32'})
df_trans.info()


# ### **2. Split data for trainig**

# Now let's create datasets for training. First cut the 'Exited' it is our target value.

# In[ ]:


target = df_trans['Exited']
train = df_trans.drop('Exited', axis = 1)


# For having ability to reproduce experimental values , we define constant for future using in random state generators

# rndd=12345

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.25, random_state=rndd)


# ![](http://)now we have training a model RandomForestClassifier, let's start training data and predict target data.

# In[ ]:


#create a model for prediction
rand_Forest = RandomForestClassifier(random_state = rndd)

# define model parameters and values for tuning
parameters = {
    'n_estimators':np.arange(1,300, 50),
    'max_depth' : np.arange(2, 30, 2),
    'min_samples_split': np.arange(2, 30, 2),
    'min_samples_leaf': np.arange(2, 30, 2)    
}
#create a searchCV to cycle through the possible values
rand_Forest_grid = RandomizedSearchCV(
    estimator = rand_Forest,
    param_distributions  = parameters,
    scoring='f1',
    n_jobs=2,
    cv = 5,
    n_iter = 150,
    verbose=True, refit=True, return_train_score = True, random_state = rndd)
    
#fit the model    
rand_Forest_grid.fit(X_train, y_train)
#check scores result
f1_train = rand_Forest_grid.best_score_
print('Best Estimator: ', rand_Forest_grid.best_estimator_)
print('Best Params: ', rand_Forest_grid.best_params_)
print('f1 =', f1_train)
predicted_train = rand_Forest_grid.predict(X_train)
accuracy_train = accuracy_score(y_train, predicted_train)
print('accuracy =', accuracy_train)
roc_auc_score_train =  roc_auc_score(y_train, predicted_train)
print('roc_auc_score',  roc_auc_score_train)


# now checking our model on test parts of data

# In[ ]:


#predict values on previously trained model
y_predicted = rand_Forest_grid.predict(X_test)

f1_test = f1_score(y_test, y_predicted)
accuracy_test = accuracy_score(y_test, y_predicted)
roc_auc_score_test =  roc_auc_score(y_test, y_predicted)
print('TEST       f1      =', f1_test)
print('TEST accuracy      =', accuracy_test)
print('TEST roc_auc_score =', roc_auc_score_test)


# For future comparing results we will save all result in dataset result.

# In[ ]:


#Create empty dataframe with columns
results = pd.DataFrame(columns=['expirement', 'f1_train', 'f1_test', 'accuracy_train', 'accuracy_test', 'roc_auc_train', 'roc_auc_test'])
#add values to columns accordingly
results = results.append([{'expirement':'simple model',
                           'f1_train':f1_train, 'f1_test': f1_test,
                           'accuracy_train': accuracy_train, 'accuracy_test':accuracy_test,
                           'roc_auc_train':roc_auc_score_train, 'roc_auc_test':roc_auc_score_test}])
results


# For now we got not bad results. Lets check our data deeper.

# In[ ]:


X_train.describe()


# As you can see EstimatedSalary mean = 5008.469733 and CreditScore mean=260.16760  - the order of values is 5008/260 = 16+ times different. It is not good for model. Let's bring to one order via StandartScaler

# In[ ]:


#create scaler
scaler = StandardScaler()
#fit and transform data
X_train_scaled = scaler.fit_transform(X_train)
#transform data based on previous fit process
X_test_scaled = scaler.transform(X_test)

#put transformed data for pretty print
d = pd.DataFrame(columns=X_train.columns, data=X_train_scaled).describe()
print('order of values', abs(d.loc['mean','EstimatedSalary']/ d.loc['mean','CreditScore']))


# Now the scale order of values is same, lets check result on our model. For now we will used hyper parmeters from previous grid trainig.
# <br> Best Params:  {'n_estimators': 101, 'min_samples_split': 20, 'min_samples_leaf': 2, 'max_depth': 18}

# In[ ]:


#create model with parameters vased on previous training result
rand_Forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=18, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=20,
                       min_weight_fraction_leaf=0.0, n_estimators=101,
                       n_jobs=None, oob_score=False, random_state=12345,
                       verbose=0, warm_start=False)

#define function for reducing code duplication
def checkModel(X_train, y_train, X_test, y_test, model = rand_Forest):
    
    model.fit(X_train, y_train)
    y_train_predicted = rand_Forest.predict(X_train)
    f1_train = f1_score(y_train, y_train_predicted)
    accuracy_train = accuracy_score(y_train, y_train_predicted)
    roc_auc_score_train =  roc_auc_score(y_train, y_train_predicted)
    
    print('roc_auc_score',  roc_auc_score_train)
    print('f1 =', f1_train)
    print('accuracy =', accuracy_train)
    
    y_test_predicted = rand_Forest.predict(X_test)
    f1_test = f1_score(y_test, y_test_predicted)
    accuracy_test = accuracy_score(y_test, y_test_predicted)
    roc_auc_score_test =  roc_auc_score(y_test, y_test_predicted)
    
    print('TEST       f1 =', f1_test)
    print('TEST accuracy =', accuracy_test)
    print('TEST roc_auc_score =', roc_auc_score_test)
    
    return f1_train, accuracy_train, roc_auc_score_train, f1_test, accuracy_test, roc_auc_score_test

#call function
f1_train, accuracy_train, roc_auc_score_train, f1_test, accuracy_test, roc_auc_score_test = checkModel(X_train_scaled, y_train, X_test_scaled, y_test)


# Put result scores to dataframe

# In[ ]:


results = results.append([{'expirement':'scaled data model',
                           'f1_train':f1_train, 'f1_test': f1_test,
                           'accuracy_train': accuracy_train, 'accuracy_test':accuracy_test,
                           'roc_auc_train':roc_auc_score_train, 'roc_auc_test':roc_auc_score_test}])
results


# As we can see there are improvements, but they showed themselves only in the training sample. Now lets check our target value

# In[ ]:


y_train.value_counts()


# We observe that 0 is a value 4 times greater than 1. Let's try to equalize their number by applying the technique upsampling/downsampling. To do this, randomly mix the existing data with the target feature 1. For this porprouse define a function upsample_1

# In[ ]:


def upsample_1(features, target, repeat):
    #array only with 0 values from features
    features_zeros = features[target == 0]
    #array only with 1 values from features
    features_ones = features[target == 1]
    
    #array only with 0 values from target
    target_zeros = target[target == 0]
    #array only with 1 values from target
    target_ones = target[target == 1]
    
    #create new data frame with features 0 values and features 1 value repeated Repeat(incoming parameters in functions) times
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    
    #create new data frame with target 0 values and target 1 value repeated Repeat(incoming parameters in functions) times
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    
    #just shuffle values in dataframe
    features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=rndd)
    
    return features_upsampled, target_upsampled


# Check their upsmpling result

# In[ ]:


X_train_u, y_train_u = upsample_1(X_train, y_train, 4)
X_test_u, y_test_u = upsample_1(X_test, y_test, 4)
y_train_u.value_counts()


# Now you can see taht 0 and 1 meet about the same time, lets check result on our model.

# In[ ]:


f1_train, accuracy_train, roc_auc_score_train, f1_test, accuracy_test, roc_auc_score_test = checkModel(X_train_u, y_train_u, X_test_u, y_test_u)


# put result to our dataset

# In[ ]:


results = results.append([{'expirement':'upsmpled data model',
                           'f1_train':f1_train, 'f1_test': f1_test,
                           'accuracy_train': accuracy_train, 'accuracy_test':accuracy_test,
                           'roc_auc_train':roc_auc_score_train, 'roc_auc_test':roc_auc_score_test}])
results


# As we can see the result is also positive. Particularly for the main metric for classification F1
# <br> Now lets apply scaller also, and check result.

# In[ ]:


scaler = StandardScaler()
X_train_u_scaled = scaler.fit_transform(X_train_u)
X_test_u_scaled = scaler.transform(X_test_u)

f1_train, accuracy_train, roc_auc_score_train, f1_test, accuracy_test, roc_auc_score_test = checkModel(X_train_u_scaled, y_train_u, X_test_u_scaled, y_test_u)


# In[ ]:


results = results.append([{'expirement':'upsmpled scaled data model',
                           'f1_train':f1_train, 'f1_test': f1_test,
                           'accuracy_train': accuracy_train, 'accuracy_test':accuracy_test,
                           'roc_auc_train':roc_auc_score_train, 'roc_auc_test':roc_auc_score_test}])
results


# **In this paper, we have considered two techniques for dealing with data imbalances in data classification. This is a dimensionality reduction of values and upsampling by target value.**
