#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score


main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

test_file = '../input/house-prices-advanced-regression-techniques/test.csv'
# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
test = pd.read_csv(test_file)
#drop house where saleprice is missing
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
y = data.SalePrice

train_X , test_X , train_y , test_y = train_test_split(X, y, random_state=0)

newtest = test.select_dtypes(exclude=['object'])


my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
newtest = my_imputer.transform(newtest)


my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
                 eval_set=[(test_X, test_y)], verbose=False)

prd = my_model.predict(newtest)

#print("Mean Absolute Error : " + str(mean_absolute_error(prd, test_y)))
print (prd)
# prepare file
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': prd})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
print("File created")


# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 
