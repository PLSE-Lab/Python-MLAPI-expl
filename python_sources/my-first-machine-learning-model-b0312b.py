#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This will be your workspace for the [Machine Learning course](https://www.kaggle.com/learn/machine-learning).**
# 
# You will need to translate the concepts to work with the data in this notebook, the Iowa data. Each page in the Machine Learning course includes instructions for what code to write at that step in the course.
# 
# # Write Your Code Below

# In[ ]:


import pandas as pd

main_file_path = '../input/test-data/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print('Some output from running this cell')


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

data.columns


# In[ ]:


from sklearn.model_selection import train_test_split
y = data.SalePrice
interesting = ["LotArea",'YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
x =data[interesting]
#applying the split funtion to the dataset
train_x,test_x, train_y,test_y = train_test_split(x,y,random_state =0)
#defining the model
model = DecisionTreeRegressor()
#fitting the model
model.fit(train_x,train_y)
#predicting the prices on test data
model.predict(test_x)
print(mean_absolute_error(test_y,model.predict(test_x)))


# In[ ]:


def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

for max_leaf_nodes in [5,40,45, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_x, test_x, train_y, test_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

forest_model =RandomForestRegressor()
forest_model.fit(train_x, train_y)
forest_model.predict(test_x)
print(mean_absolute_error(test_y,forest_model.predict(test_x)))


# In[ ]:


#applying the max_leaf_nodes function to random forest to bring down the mean absolute error
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

for max_leaf_nodes in [5,45, 50,55, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_x, test_x, train_y, test_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[ ]:


#submitting my first dataset on Kaggle!!!

test = pd.read_csv('../input/test-data/test.csv')
test_x = test[interesting]
predicted_prices = forest_model.predict(test_x)
print(predicted_prices)

my_submission = pd.DataFrame({'Id': test.Id,'SalePrice':predicted_prices})
my_submission.to_csv('submission.csv', index=False)


# In[ ]:


#getting columns with missing data
print(data.isnull().sum())


# In[ ]:


target = data.SalePrice
predictors = data.drop(['SalePrice'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
numeric_predictors = predictors.select_dtypes(exclude=['object'])


# In[ ]:


numeric_predictors.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(numeric_predictors, 
                                                    target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)


# In[ ]:


def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


# In[ ]:


#checking if predictions get better when we drop columns
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))


# Yes it does gets better!!
#  Now let's check the same after imputing

# In[ ]:


from sklearn.preprocessing import Imputer

imputer= Imputer()
imputed_X_train = imputer.fit_transform(X_train)
imputed_X_test = imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))


# MAE goes quite down. Lastly, let's check the same for name the columns that were imputed

# In[ ]:


imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns if X_train[col].isnull().any())

for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
    
#imputation
imputed_X_train_plus = imputer.fit_transform(imputed_X_train_plus )
imputed_X_test_plus = imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))


# Now, let's perform one hot encoding 

# In[ ]:


one_hot_encoded_training_predictors = pd.get_dummies(X_train)


# In[ ]:


from sklearn.model_selection import cross_val_score

def get_mae(X_train,y_train):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50),x,y, scoring = 'neg_mean_absolute_error').mean()

predictors_without_categoricals = X_train.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals, target)
mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)


# In[ ]:


print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))


# In[ ]:


one_hot_encoded_training_predictors = pd.get_dummies(X_train)
one_hot_encoded_test_predictors = pd.get_dummies(X_test)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)
final_train.drop(['GarageYrBlt','LotFrontage'],axis=1,inplace=True)

imputer.fit_transform(final_train)
final_train.drop('MasVnrArea',axis=1,inplace= True)

model.fit(final_train,y_train)
final_test.drop(['GarageYrBlt','MasVnrArea','LotFrontage'],axis=1,inplace=True)
model.predict(final_test)


# In[ ]:


print(get_mae(final_train,y_train))


# In[ ]:


my_submission1 = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission1.to_csv('submission1.csv', index=False)

data=pd.read_csv('../input/test-data/train.csv')
testdata = pd.read_csv('../input/test-data/test.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop('SalePrice',axis = 1).select_dtypes(exclude = ['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

testdata = testdata.select_dtypes(exclude = ['object'])
testdata = my_imputer.fit_transform(testdata)


# Improving the predictions by applying XGBoost model

# In[ ]:



y = data.SalePrice
X = data.drop('SalePrice',axis = 1).select_dtypes(exclude = ['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)


# In[ ]:


from xgboost import XGBRegressor

xg = XGBRegressor()


# In[ ]:


xg.fit(train_X, train_y, verbose=False)


# In[ ]:


preds = xg.predict(test_X)


# In[ ]:


print("Mean absolute error : "+ str(mean_absolute_error(preds,test_y)))


# In[ ]:


xg1 =XGBRegressor(n_estimators=1000)
xg1.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X,test_y)],verbose=False)


# In[ ]:


pred = xg1.predict(test_X)
print("Mean absolute error : "+ str(mean_absolute_error(pred,test_y)))


# In[ ]:


xg = XGBRegressor(n_estimators=5, learning_rate=0.05)
xg.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X,test_y)],verbose=False)


# In[ ]:


preds = xg.predict(test_X)
print("Mean absolute error : "+ str(mean_absolute_error(preds,test_y)))


# In[ ]:





# 

# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 
