#!/usr/bin/env python
# coding: utf-8

# # Making our model
# I have included 2 models, the simplist possible and also one with hyperparameters which when tuned give much better results. 
# Try playing with them and becareful not to overfit. 

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

#Opening our file with the training data in
train = pd.read_csv('../input/train.csv')

#We are trying to predict the sale price column
target = train.SalePrice

#Get rid of the answer and anything thats not an object
train = train.drop(['SalePrice'],axis=1).select_dtypes(exclude=['object'])

#Split the data into test and validation
train_X, test_X, train_y, test_y = train_test_split(train,target,test_size=0.25)

#Impute all the NaNs
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.fit_transform(test_X)

#Simplist XGBRegressor
#my_model = XGBRegressor()
#my_model.fit(train_X, train_y)

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=8, 
             eval_set=[(test_X, test_y)], verbose=False)


#Make predictions
predictions = my_model.predict(test_X)

print("Mean absolute error = " + str(mean_absolute_error(predictions,test_y)))


# # Making a submission

# In[ ]:


#Getting our test data
test = pd.read_csv('../input/test.csv')

#Getting it to the right format that we used with our model
test = test.select_dtypes(exclude=['object'])

#Fill in all the NaN values with ints
test_X = my_imputer.fit_transform(test)

#Make predictions
predictions = my_model.predict(test_X)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)




