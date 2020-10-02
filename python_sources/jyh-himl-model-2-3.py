#!/usr/bin/env python
# coding: utf-8

# # JON'S HiML Competition RFG_Log Model (v 2.3)
# ## Make Date: 04/10/18
# #### Updates included in this version (V 2.3):
# 1. Version 2.2's use of log transform on quantity and the predictor combination scanning dropped my competetion score to 0.59859.
# 1. From Monday (4/9/17) meetup, we know Customer 7 skews the data. So what if we make 2 different models: 1 for Customer 7 and another for everyone else?
# 
# #### Concepts from Version V 2.2 used in this version:
# 1. This ML model was built using scikit's RandomForestRegressor model.  
# 1. Can we automatically find the best combination of input parameters?  
#     For now, just do each combination of predictors. Later (if things get crazy), we can do random searcy - perhaps get fancy and implement an optimization scheme (e.g. gradient descent or sim anneal)  
# 1. Let's implement log transform on the quantities to improve score.

# In[ ]:


#Some initialization procedures:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split

# load in data files
FILE_DIR = '../input/hawaiiml-data'
for f in os.listdir(FILE_DIR):
    print('{0:<30}{1:0.2f}MB'.format(f, 1e-6*os.path.getsize(f'{FILE_DIR}/{f}')))
df_train = pd.read_csv(f'{FILE_DIR}/train.csv', encoding='ISO-8859-1') #write training data to dataframe
df_test = pd.read_csv(f'{FILE_DIR}/test.csv', encoding='ISO-8859-1') # Read the test data

#define the error function:
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))


# # Split Data Between Customer 7 and Everyone Else  

# In[ ]:


#split training data between customer 7 and everyone else:
df_train_7 = df_train.loc[df_train['customer_id'] == 7] #customer 7 only
df_train_no7 = df_train.loc[df_train['customer_id'] != 7] #everyone else

#split test data:
df_test_7 = df_test.loc[df_test['customer_id'] == 7] #customer 7 only
df_test_no7 = df_test.loc[df_test['customer_id'] != 7] #everyone else


# # Define Training Target Data:
# 1. We want to predict the quantity data field.    
# 1. By convention, we define this target as 'y'.  
# 1. From model 2.02, we know that doing a log transform on the tartget reduces skewness and improves score.  

# In[ ]:


y = df_train.quantity
logy_7 = np.log1p(df_train_7.quantity) #take log of quantities
logy_no7 = np.log1p(df_train_no7.quantity) #take log of quantities


# # Define ML Predictors
# build a function to go through each combination of predictors

# ### Here is the list of columns we can choose predictors from. To keep it simple, just select from numeric data types.

# In[ ]:


print('Column Names & Data Types: \n', df_train.dtypes)


# ## Define feature combination picker

# In[ ]:


ls_AllPredictors = ['invoice_id', 'stock_id', 'customer_id', 'unit_price']

# https://stackoverflow.com/questions/464864/how-to-get-all-possible-combinations-of-a-list-s-elements
from itertools import chain, combinations
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
#build combos:
ls_PredictorCombos = [list(combo[1]) for combo in enumerate(powerset(ls_AllPredictors), 1)]
#Define combo getting function:
def GetX(comboID, adf_train):
    #get X values in adf_train for the given list of predictors 
    return adf_train[ls_PredictorCombos[comboID]]


# # Implement Random Forest Regressor Model on Each Predictor Combo
# ## Define Some Helper Functions

# In[ ]:


def Run_RFG(SeedVal, X, logy):
    #fit, predict, and then evaluate a model using passed in values (e.g. training set)
    #return back RMSLE value and trained model
    from sklearn.ensemble import RandomForestRegressor

    myModel = RandomForestRegressor()
    train_X, val_X, train_y, val_y = train_test_split(X, logy,random_state = SeedVal) #split training data into a test and train part
    myModel.fit(train_X, train_y)
    predicted_vals = np.expm1(myModel.predict(val_X)) #transform predicted values from log to "normal" Y value
    return rmsle(np.expm1(val_y), predicted_vals), myModel #include transform of val_Y values from log to "normal" Y value


# In[ ]:


def Train_Model(adf_train, logy):
    #iteratively search for best parameter combinations to match X values in adf_train with logy
    #return best combo, error results of each model, training error, and model
    MinErr = 100000000
    df_Track = pd.DataFrame()
    for comboID in range(1,len(ls_PredictorCombos)):
        SeedVal = 6
        X = GetX(comboID, adf_train)
        TrainErr,myModel = Run_RFG(SeedVal,X,logy)
        df2 = pd.DataFrame([[comboID, TrainErr, ','.join(ls_PredictorCombos[comboID])]],columns=['ComboID','err','Preds'])
        if df_Track.shape[0] >0:
            df_Track = pd.concat([df2, df_Track])
        else:
            df_Track = df2.copy(deep = True)
        if TrainErr < MinErr:
            MinErr = TrainErr
    #         BestSeedVal = SeedVal
            bestComboID = comboID
            print ('Best Combo: ', comboID, ' Params: ', ls_PredictorCombos[comboID], ' Err: ', TrainErr)
    #train with best combo:
    X = GetX(bestComboID, adf_train)
    TrainErr,myModel = Run_RFG(0,X,logy)
    print ('fin')
    return ls_PredictorCombos[bestComboID], df_Track, TrainErr, myModel


# ### Build Customer 7 and Everyone Else Models:
# 1. Find Best Predictors
# 1. Define Separate Models

# In[ ]:


ls_mypredictors_7, df_Track_7, TrainErr_7, myModel_7 = Train_Model(df_train_7, logy_7) #train customer 7 model
ls_mypredictors_no7, df_Track_no7,  TrainErr_no7, myModel_no7 = Train_Model(df_train_no7, logy_no7) #train everyone else model


# ## Observations

# In[ ]:


display(df_Track_7.sort_values(by=['err']))
display(df_Track_no7.sort_values(by=['err']))


# In[ ]:


train_X_7, val_X_7, train_y_7, val_y_7 = train_test_split(df_train_7, logy_7, random_state = 0) #split training data into a test and train part
train_X_no7, val_X_no7, train_y_no7, val_y_no7 = train_test_split(df_train_no7, logy_no7, random_state = 0) #split training data into a test and train part

# df_train_7_pred = df_train_7.copy(deep = True) #copy test dataframe
val_X_7 = val_X_7.assign(pred_quantity = np.expm1(myModel_7.predict(val_X_7[ls_mypredictors_7])) )#predict and transform from log to "normal" Y value
# df_train_no7_pred = df_train_no7.copy(deep = True) #copy test dataframe
val_X_no7 = val_X_no7.assign(pred_quantity = np.expm1(myModel_no7.predict(val_X_no7[ls_mypredictors_no7]))) #predict and transform from log to "normal" Y value

df_preds = pd.concat([val_X_7, val_X_no7]) #concatenate predictions into 1 dataframe
df_preds = pd.DataFrame (df_preds['pred_quantity'])
df_preds.sort_index(inplace=True)

df_y = np.expm1(pd.concat([val_y_7,val_y_no7])) #concat y vals into 1 dataframe and then transform from log to "normal" y
df_y.sort_index(inplace=True)

rmsle(df_y, df_preds['pred_quantity'])


# # Submit Model's Predictions

# ## First, output model's predictions for test data set:

# In[ ]:


# Use the 2 models to make predictions
df_test_7_pred = df_test_7.copy(deep = True) #copy test dataframe
df_test_7_pred['pred_quantity'] = np.expm1(myModel_7.predict(df_test_7[ls_mypredictors_7])) #predict and transform from log to "normal" Y value
df_test_no7_pred = df_test_no7.copy(deep = True) #copy test dataframe
df_test_no7_pred['pred_quantity'] = np.expm1(myModel_no7.predict(df_test_no7[ls_mypredictors_no7])) #predict and transform from log to "normal" Y value

df_preds = pd.concat([df_test_7_pred, df_test_no7_pred]) #concatenate predictions into 1 dataframe
# # We will look at the predicted prices to ensure we have something sensible.
display(df_preds)


# ## Next, submit predicted values

# In[ ]:


my_submission = pd.DataFrame({'Id': df_preds.id, 'quantity': df_preds.pred_quantity})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[ ]:




