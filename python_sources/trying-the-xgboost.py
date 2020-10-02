#!/usr/bin/env python
# coding: utf-8

# Hello All ,
# This kernel is realted to the iowa dataset where we use cardinality to hot encode the string colums and use imputation to fit to the NAN values .Please have a look at the code below : 
# 
# **Your submission scored 17266.76666, which is an improvement of your previous score of 18282.02181. Great job!**
# We hope to improve the model further as the course progresses :-).
# 
# 

# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
#foudn the minimum value of n_estimator 
n_estimator=126
#load data
iowa_train_path='../input/train.csv'
iowa_train_data=pd.read_csv(iowa_train_path)
iowa_test_path='../input/test.csv'
iowa_test_data=pd.read_csv(iowa_test_path)
x = iowa_train_data.drop(['SalePrice'], axis=1)
y=iowa_train_data.SalePrice
#now split the train and test set between the two 
train_x, test_x, train_y, test_y=train_test_split(x,y,train_size=0.99,test_size=0.01, random_state = 0)
#use the one hot encoding for the training set 
train_low_cardinality_cols = [cname for cname in train_x.columns if 
                                train_x[cname].nunique() < 10 and
                                train_x[cname].dtype == "object"]
train_numeric_cols = [cname for cname in train_x.columns if 
                                train_x[cname].dtype in ['int64', 'float64']]
train_total_cols=train_low_cardinality_cols+train_numeric_cols
print(len(train_total_cols))
train_x_cardinal = train_x[train_total_cols]
test_x_cardinal = test_x[train_total_cols]
pred_test_x_cardinal=iowa_test_data[train_total_cols]
#print(train_x_cardinal.shape)
#print(test_x_cardinal.shape)
#test set 
train_x_one_hot_encoded=pd.get_dummies(train_x_cardinal)
test_x_one_hot_encoded=pd.get_dummies(test_x_cardinal)
pred_test_x_one_hot_encoded=pd.get_dummies(pred_test_x_cardinal)
print(train_x_one_hot_encoded.shape)
print(test_x_one_hot_encoded.shape)
#now aligning the two 
train_x_final, test_x_final = train_x_one_hot_encoded.align(test_x_one_hot_encoded,
                                                                    join='left', 
                                                             axis=1)
print(train_x_final.shape)
print(test_x_final.shape)
#trying to align teh predic_train 
train_x_final_copy=train_x_final.copy()
pred_train_x_final,pred_test_x_final=train_x_final_copy.align(pred_test_x_one_hot_encoded,join='left',axis=1)
print(train_x_final_copy.shape)
print(pred_test_x_one_hot_encoded.shape)
print(pred_train_x_final.shape)
print(pred_test_x_final.shape)
final_test_set=pred_test_x_final
#call the imputer on the train and test data 
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()

imputed_train_x_plus = train_x_final.copy()
imputed_test_x_plus=test_x_final.copy()
imputed_final_test_x_plus=final_test_set.copy()
cols_with_missing = {col for col in train_x_final.columns 
                                 if train_x_final[col].isnull().any()}
for col in cols_with_missing:
    imputed_train_x_plus[col + '_was_missing'] = imputed_train_x_plus[col].isnull()
    imputed_test_x_plus[col + '_was_missing'] = imputed_test_x_plus[col].isnull()
    imputed_final_test_x_plus[col + '_was_missing'] = imputed_final_test_x_plus[col].isnull()
imputed_train_x_plus = my_imputer.fit_transform(imputed_train_x_plus)
imputed_test_x_plus=my_imputer.transform(imputed_test_x_plus)
imputed_final_test_x_plus=my_imputer.transform(imputed_final_test_x_plus)
#print(imputed_train_x_plus.shape)
#print(imputed_test_x_plus.shape)
#print(train_y.shape)
#print(test_y.shape)
#obtained a numpy array of the test and train data 
#function for computing the mae for xgboost given a particular learning rate, number of iterations ,train and test dataset
#importing the mean square scikit metric 
from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=127, learning_rate=0.7,random_state=1)
my_model.fit(imputed_train_x_plus, train_y, early_stopping_rounds=5, 
             eval_set=[(imputed_test_x_plus, test_y)], verbose=False)

pred=my_model.predict(imputed_final_test_x_plus)
output = pd.DataFrame({'Id': iowa_test_data.Id,
                       'SalePrice': pred})
output.to_csv('submission.csv', index=False)


# In[ ]:




