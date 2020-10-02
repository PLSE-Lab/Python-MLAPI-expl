#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This will be your workspace for Kaggle's Machine Learning education track.**
# 
# You will build and continually improve a model to predict housing prices as you work through each tutorial.  Fork this notebook and write your code in it.
# 
# The data from the tutorial, the Melbourne data, is not available in this workspace.  You will need to translate the concepts to work with the data in this notebook, the Iowa data.
# 
# Come to the [Learn Discussion](https://www.kaggle.com/learn-forum) forum for any questions or comments. 
# 
# # Write Your Code Below
# 
# 

# In[ ]:


##importing panda library
import pandas as pd


# In[ ]:


house_data = pd.read_csv('../input/train.csv')
#print(type(house_data))
#house_data_1 = pd.read_csv('../input/test.csv')
#print(house_data.describe())
#print(house_data_1.describe())


# In[ ]:


##selecting and filtering in columns
#print(house_data.columns)


# In[ ]:


##selecting one column

one_column_sale_condition = house_data.SalePrice
#print(one_column_sale_condition.describe())


# In[ ]:


## selecting multiple column
temp_list=['Fence','SaleCondition']
multiple_column_together=house_data[temp_list]
print(multiple_column_together.describe())


# In[ ]:


## predicton column
y = house_data.SalePrice
print(y.describe())


# In[ ]:


## from which columns we can predict results
iowa_predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = house_data[iowa_predictors]
print(X.describe())


# In[ ]:


## as per exercise we will use decision tree regressor model right now don't know why, but as per instruction we will use it

#Define Model
from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor()

#Fit Model
iowa_model.fit(X,y)


# In[ ]:


## make some prediction for 5 houses

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(iowa_model.predict(X.head()))


# In[ ]:


## mean absolute error

from sklearn.metrics import mean_absolute_error
predicted_home_prices = iowa_model.predict(X)
mean_absolute_error(y,predicted_home_prices)


# In[ ]:


## Now we will deal with In-Sample scores
## So we will split the data into train and test data set and validate the model

from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=0)

#Define Model
iowa_model = DecisionTreeRegressor()
#Fit the model into training data set and buil the model
iowa_model.fit(train_X,train_y)
#Now builded model will use to predict the result on the validation data set
val_predictions = iowa_model.predict(val_X)
print(mean_absolute_error(val_y,val_predictions))


# In[ ]:


## we find mae for different value of max_leaf_nodes to overcome the problem of overfitting and underfitting of dataset

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes_value,predictors_X,predictors_y,validation_X,validation_y):
    model = DecisionTreeRegressor(max_leaf_nodes= max_leaf_nodes_value,random_state=0)
    model.fit(predictors_X,predictors_y)
    new_predictions = model.predict(validation_X)
    mae = mean_absolute_error(validation_y,new_predictions)
    return mae

for max_leaf_nodes in [5,50,45,500,5000]:
    mae_value = get_mae(max_leaf_nodes,train_X,train_y,val_X,val_y)
    print("Value of mae : " + str(mae_value) + " for max nodes : " + str(max_leaf_nodes))

## for here ideal value of leaf node is 45


# In[ ]:


## Random Forests example

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

random_forest_model = RandomForestRegressor()
random_forest_model.fit(train_X,train_y)
predict_forest_val = random_forest_model.predict(val_X)
mae = mean_absolute_error(val_y,predict_forest_val)

print("Value of mae using Random_Forest is : " + str(mae))


# In[ ]:


## Now it's time to submit the solution (first for me on Kaggle)

test = pd.read_csv('../input/test.csv')
## now we will treat test data as same as the training data with same columns which we have use to define model
test_X = test[iowa_predictors]
#print(iowa_predictors)
#print(test_X)

## use above Random forests model to predict the result
#predicted_prices = random_forest_model.predict(test_X)
#print("Predicted Prices : " + str(predicted_prices))


# In[ ]:


## now we will make submission of above results

#my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# we can use any filename. We choose submission_iowa here
#my_submission.to_csv('submission_iowa.csv', index=False)


# In[ ]:


## missing values 
## we can see howmany column have missing value using isnull function
#print(house_data.isnull().sum())
#print(house_data_1.isnull().sum())


# In[ ]:


## missing values in IOWA dataset
## we will use all columns in IOWA data set and let's see whether we predict good result or not

import pandas as pd
iowa_data = pd.read_csv('../input/train.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_data_target = iowa_data['SalePrice']
iowa_data_predictors = iowa_data.drop(['SalePrice'],axis=1)

#iowa_data_target.describe()
#iowa_data_predictors.describe()

## drop all non numeric columns from predictors column

iowa_data_numeric_predictors = iowa_data_predictors.select_dtypes(exclude=['object'])
#iowa_data_numeric_predictors.describe()

## now we will split the data set

X_train,X_test,y_train,y_test = train_test_split(iowa_data_numeric_predictors,iowa_data_target,train_size=0.7,
                                                 test_size=0.3,random_state=0)

## now we will create function which will calculate the mae score

def mae_data_score(X_train,X_test,y_train,y_test):
    model = RandomForestRegressor()
    model.fit(X_train,y_train)
    val_predict_y = model.predict(X_test)
    mae_score = mean_absolute_error(y_test,val_predict_y)
    return mae_score

## first calculate the mae score for by 1st approach
## Drop a column with missing value

cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(mae_data_score(reduced_X_train,reduced_X_test,y_train,y_test))
#print("Before not dropping missing value column Mean Absolute Error is : ")
#print(mae_data_score(X_train,X_test,y_train,y_test))

## Now finding missing value by second approach
## Impute the missing values in dataset

from sklearn.preprocessing import Imputer
imputer_model = Imputer()
imputed_X_train = imputer_model.fit_transform(X_train)
imputed_X_test = imputer_model.transform(X_test)
print("Mean Absolute Error from Imputation : ")
print(mae_data_score(imputed_X_train,imputed_X_test,y_train,y_test))

## Now finding missing value by third approach
## Extension of Imputation , Imputation with extra columns showing what was imputed

imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
## Imputation inside

my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(mae_data_score(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))

## so from the result we can see that the best solution for us is first approach
## Because in our data set we have less missing value in columns



# In[ ]:


## Now it's time to submit the solution 

test_v2 = pd.read_csv('../input/test.csv')
new_X_v2 = test_v2[list(iowa_data_numeric_predictors)]

imputed_X_test_v2 = imputer_model.transform(new_X_v2)

forest_model_v2 = RandomForestRegressor()
forest_model_v2.fit(imputed_X_train,y_train)
predicted_prices = forest_model_v2.predict(imputed_X_test_v2)


#my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# we can use any filename. We choose submission_iowa here
#my_submission.to_csv('submission_iowa_v2.csv', index=False)




# In[ ]:


## we will tune our model by using the XGBoost model
## we will define our model with n_estimators , early_stopping_rounnds , learning_rate

import pandas as pd
iowa_data_v3 = pd.read_csv('../input/train.csv')

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_data_v3.dropna(axis=0, subset=['SalePrice'], inplace=True)
iowa_data_v3_target = iowa_data_v3.SalePrice
iowa_data_v3_predictors = iowa_data_v3.drop(['SalePrice'],axis=1)
## drop all non numeric columns from predictors column
iowa_data_numeric_predictors_v3 = iowa_data_v3_predictors.select_dtypes(exclude=['object'])

print(len(iowa_data_numeric_predictors_v3.columns))

train_X_v3,test_X_v3,train_y_v3,test_y_v3 = train_test_split(iowa_data_numeric_predictors_v3.as_matrix()
                                                             ,iowa_data_v3_target.as_matrix(),
                                                            test_size=0.25)

from sklearn.preprocessing import Imputer
myimputer_v3 = Imputer()
imputed_train_X_v3 = myimputer_v3.fit_transform(train_X_v3)
imputed_test_X_v3 = myimputer_v3.transform(test_X_v3)

## we will import XGBoost model ans build model using imputed columns

from xgboost import XGBRegressor
xgboost_model = XGBRegressor()
xgboost_model.fit(imputed_train_X_v3,train_y_v3,verbose=False)
print("Means absolute error using simple XGBoost model : " + 
      str(mean_absolute_error(test_y_v3,xgboost_model.predict(imputed_test_X_v3))))

## we will use other parameters and check what is the mae for XGBoost model
## we will tune our model
xgboost_model_tune = XGBRegressor()
xgboost_model_tune = XGBRegressor(n_estimators = 1000, learning_rate = 0.05)
xgboost_model_tune.fit(imputed_train_X_v3,train_y_v3,
                  early_stopping_rounds=5,
                  eval_set=[(imputed_test_X_v3,test_y_v3)],verbose=False)
predictions_v3 = xgboost_model_tune.predict(imputed_test_X_v3)
print("Mean Absolute Error after model tuning : " + str(mean_absolute_error(test_y_v3,predictions_v3)))


# In[ ]:


### now we will submit the latest solution after model tuning usinf XGBooster model

test_v3 = pd.read_csv('../input/test.csv')
new_X_v3 = test_v3[list(iowa_data_numeric_predictors_v3)]



imputed_X_test_v3 = imputer_model.transform(new_X_v3)
predicted_prices = xgboost_model_tune.predict(imputed_X_test_v3)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# we can use any filename. We choose submission_iowa here
my_submission.to_csv('submission_iowa_v3.csv', index=False)


# In[ ]:





# In[ ]:


import pandas as pd
import pandas as pd
iowa_data_v5 = pd.read_csv('../input/train.csv')

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_data_v5.dropna(axis=0, subset=['SalePrice'], inplace=True)
iowa_data_v5_target = iowa_data_v3.SalePrice
iowa_data_v5_predictors = iowa_data_v3.drop(['SalePrice'],axis=1)

#print(iowa_data_v5_predictors.columns)

dummy_data = pd.get_dummies(iowa_data_v5_predictors,columns=['SaleCondition','LotShape'])
print(len(dummy_data.columns))

iowa_data_numeric_predictors_v5 = dummy_data.select_dtypes(exclude=['object'])

print(len(iowa_data_numeric_predictors_v5.columns))

X_train_v5,X_test_v5,y_train_v5,y_test_v5 = train_test_split(iowa_data_numeric_predictors_v5
                                                             ,iowa_data_v5_target,
                                                             random_state=0)

from sklearn.preprocessing import Imputer
imputer_model_v5 = Imputer()
imputed_X_train_v5 = imputer_model_v5.fit_transform(X_train_v5)
imputed_X_test_v5 = imputer_model_v5.transform(X_test_v5)

from xgboost import XGBRegressor
xgboost_model_v5 = XGBRegressor()

xgboost_model_v5 = XGBRegressor(n_estimators = 1500, learning_rate = 0.03)
xgboost_model_v5.fit(imputed_X_train_v5,y_train_v5,
                  early_stopping_rounds=7,
                  eval_set=[(imputed_X_test_v5,y_test_v5)],verbose=False)
predictions_v5 = xgboost_model_v5.predict(imputed_X_test_v5)
print("Mean Absolute Error after model tuning with one hot Encoding : " 
      + str(mean_absolute_error(y_test_v5,predictions_v5)))



# In[ ]:


test_v5 = pd.read_csv('../input/test.csv')
#print(len(iowa_data_v5_predictors.columns))
#print(len(test_v5.columns))
#dummy_data_pred = pd.get_dummies(test_v5,columns=['MSZoning'])
#print(len(dummy_data_pred.columns))

categoricals = test_v5.select_dtypes(include=[object])
print(categoricals.describe())

part_1 = iowa_data_v5_predictors.select_dtypes(include=[object])
print(part_1.describe())





# In[ ]:


import pandas as pd

iowa_data_v7 = pd.read_csv('../input/train.csv')

target_v7 = iowa_data_v7['SalePrice']
predictors_v7 = iowa_data_v7.drop(['SalePrice'],axis=1)

numeric_predictors_v7 = predictors_v7.select_dtypes(exclude=['object'])

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
my_pipeline = make_pipeline(Imputer(), XGBRegressor())

from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_pipeline, numeric_predictors_v7, target_v7, scoring='neg_mean_absolute_error')
print(scores)

print('Mean Absolute Error %2f' %(-1 * scores.mean()))


# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np
data_set = pd.read_csv('../input/train.csv')
print("Dataset" , data_set.shape)

target = data_set['SalePrice']

#cormat = data_set.corr()
#top_corr_features = cormat.index[abs(cormat["SalePrice"])>0.5]
#print(top_corr_features)

#print("Find most important features relative to target")
corr = data_set.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
#print(corr.SalePrice)
features = ['Id','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces','BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath','LotArea','BsmtFullBath','BsmtUnfSF','BedroomAbvGr','ScreenPorch','PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath','MiscVal','LowQualFinSF','YrSold','OverallCond','MSSubClass','EnclosedPorch','KitchenAbvGr']
predictors = data_set[features]
print("target" , target.shape)
print("Features " , predictors.shape)

# Handle remaining missing values for numerical features by using median as replacement
print("NAs for numerical features in train : " + str(predictors.isnull().values.sum()))
predictors = predictors.fillna(predictors.median())
print("Remaining NAs for numerical features in train : " + str(predictors.isnull().values.sum()))

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(predictors,target,test_size=0.3,random_state=0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor()
forest_model.fit(X_train,y_train)
pred_y = forest_model.predict(X_test)
print("Mean Error " + str(mean_absolute_error(y_test,pred_y)))

from xgboost import XGBRegressor
xgboost_model = XGBRegressor()

xgboost_model = XGBRegressor(n_estimators = 1500 ,learning_rate = 0.03)
xgboost_model.fit(X_train,y_train,
                  early_stopping_rounds=7,
                  eval_set=[(X_test,y_test)],verbose=False)
pred_xg = xgboost_model.predict(X_test)
print("Mean Absolute Error after model tuning with one hot Encoding : " 
      + str(mean_absolute_error(y_test,pred_xg)))


# In[ ]:


import pandas as pd
import numpy as np
data_set = pd.read_csv('../input/train.csv')
print("Dataset" , data_set.shape)

target = data_set['SalePrice']

num_features = data_set.select_dtypes(exclude=[object])
print(num_features.shape)
num_features = num_features.drop(['SalePrice'],axis=1)
print("Numeric : " , num_features.shape)

cat_features = data_set.select_dtypes(include=[object])
print("Categorical : " , cat_features.shape)

# Handle remaining missing values for numerical features by using median as replacement
print("NAs for numerical features in train : " + str(num_features.isnull().values.sum()))
num_features = num_features.fillna(num_features.median())
print("Remaining NAs for numerical features in train : " + str(num_features.isnull().values.sum()))

cat_features = pd.get_dummies(cat_features)
print("one hot encoding : " , cat_features.shape)
#cat_features.isnull().values.sum()

predictors = pd.concat([num_features,cat_features],axis=1)
print("Final Shape : " , predictors.shape)

X_train,X_test,y_train,y_test = train_test_split(predictors,target,test_size=0.3,random_state=0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

xgboost_model.fit(X_train,y_train,
                  early_stopping_rounds=7,
                  eval_set=[(X_test,y_test)],verbose=False)
pred_xg = xgboost_model.predict(X_test)
print("Mean Absolute Error after model tuning with one hot Encoding : " 
      + str(mean_absolute_error(y_test,pred_xg)))


# In[ ]:


test_data = pd.read_csv('../input/test.csv')
test_data.shape

num_test = test_data.select_dtypes(exclude=['object'])
cat_test = test_data.select_dtypes(include=['object'])
num_test.shape
# Handle remaining missing values for numerical features by using median as replacement
print("NAs for numerical features in train : " + str(num_test.isnull().values.sum()))
num_test = num_test.fillna(num_test.median())
print("Remaining NAs for numerical features in train : " + str(num_test.isnull().values.sum()))

cat_test = pd.get_dummies(cat_test)
print("one hot encoding : " , cat_test.shape)


# In[ ]:




