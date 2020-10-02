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


import pandas as pd

melbourne_file_path = '../input/train.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.columns)


# In[ ]:


melbourne_price_data = melbourne_data.SalePrice
print (melbourne_price_data.describe())


# In[ ]:


melbourne_data


# In[ ]:


columns_of_interest = ['LotArea','SalePrice']
melbourne_price_data = melbourne_data[columns_of_interest]
print (melbourne_price_data.describe())


# In[ ]:


Y = melbourne_data.SalePrice
melbourne_predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X =  melbourne_data[melbourne_predictors]


# In[ ]:


print (Y.head(3))
print( X.head(3))


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor (random_state=33)
model.fit(X,Y)


# In[ ]:


print ("Making predictions of the following 5 houses")
print (X.head(5))


# In[ ]:


print ("The Predictions are :")
print (model.predict (X.head(5)))


# In[ ]:


print ("Coefficient of Determination R^2 is")
print (model.score (X, Y))
print (Y.head(5))


# In[ ]:


from sklearn.metrics import mean_absolute_error
predicted_home_price = model.predict(X)
print("mean_absolute_error is")
print (mean_absolute_error(Y,predicted_home_price))


# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.8, random_state=33 )
model.fit(X_train,y_train)
predicted_house_price = model.predict(X_test)
print(mean_absolute_error(y_test, predicted_house_price))
print(model.score(X_test,y_test))


# In[ ]:


#from sklearn.metrics import mean_absolute_error
#from sklearn.tree import DecisionTreeRegressor

def get_mae(regressor,max_leaf_nodes, train_X,test_X,train_y,test_y):
    model_mae =  regressor(max_leaf_nodes=max_leaf_nodes, random_state=33)
    model_mae.fit(train_X,train_y)
    predict_sales_price = model_mae.predict(test_X)
    mae = mean_absolute_error(test_y,predict_sales_price)
    return(mae)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model_list =  [DecisionTreeRegressor, RandomForestClassifier]
for leaf_count in [5,10,50,500,5000]:
    for model_name in model_list:
        my_mae = get_mae(model_name, leaf_count, X_train,X_test,y_train, y_test)
        print ("model_name = {} \t max_leaf_nodes=  {} \t\t mean_absolute_error=  {}". format(model_name,leaf_count, my_mae))


# In[ ]:


#Load Test Data 
test = pd.read_csv("../input/test.csv")
test_X = test[melbourne_predictors]
print (test_X.head(3))


# In[ ]:


#prediction on Test data
DTmodel= DecisionTreeRegressor (max_leaf_nodes = 50,random_state=33)
DTmodel.fit(X_train,y_train)
sale_predict = DTmodel.predict(X_test)
mean_ae = mean_absolute_error ( y_test,sale_predict)
print ( "mean_ae for X_test ={}".format(mean_ae))


# In[ ]:


sale_predict_x_test = DTmodel.predict(test_X)
print(sale_predict_x_test)


# In[ ]:


print(test.columns)


# In[ ]:


from catboost import Pool, CatBoostRegressor, cv
model_cat = CatBoostRegressor(eval_metric='MAE',use_best_model=True,verbose=False,random_seed=42)


# In[ ]:


model_cat.fit(X_train,y_train,eval_set=(X_test,y_test))


# In[ ]:


sale_predict_cat = model_cat.predict(X_test)
mean_ae_cat = mean_absolute_error ( y_test,sale_predict_cat)
print ( "mean_ae for X_test ={}".format(mean_ae_cat))


# In[ ]:


sale_predict_x_test_cat = model_cat.predict(test_X)
print(sale_predict_x_test_cat)
#house_sale_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': sale_predict_x_test_cat})
#house_sale_submission.to_csv('submission.csv',index=False)


# In[ ]:


#house_sale_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': sale_predict_x_test})
#house_sale_submission.to_csv('submission.csv',index=False)


# In[ ]:


melbourne_data.head(5)


# In[ ]:


#print(melbourne_data.isnull().sum())
melbourne_data.dtypes


# In[ ]:



raw_data_len = len(melbourne_data)
dataset = pd.concat(objs=[melbourne_data,test_X],axis=0)
dataset_preprocessed =  pd.get_dummies(dataset)
melbourne_data_OHE= dataset_preprocessed[:raw_data_len]
test_X_OHE =  dataset_preprocessed[raw_data_len:]
encoded_test = list(test_X_OHE.columns)
print ("{} total features after one-hot encoding.".format(len(encoded_test)))
encoded_features = list(melbourne_data_OHE.columns)
print ("{} total features after one-hot encoding.".format(len(encoded_features)))


# In[ ]:


from sklearn.preprocessing import Imputer
my_imputer = Imputer()
imputed_melbourne_data = my_imputer.fit_transform(melbourne_data_OHE)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(imputed_melbourne_data,Y, test_size=0.8,random_state=33)
model_cat = CatBoostRegressor(eval_metric='MAE', use_best_model=True, verbose =False, random_seed =42)
model_cat.fit(X_train,y_train,eval_set=(X_test,y_test))
sale_predict_cat=model_cat.predict(X_test)
mean_ae_cat = mean_absolute_error (y_test,sale_predict_cat)
print ("mean_ae for X_test ={}".format(mean_ae_cat))


# In[ ]:


sale_predict_x_test_cat = model_cat.predict(test_X_OHE)
print(sale_predict_x_test_cat)
house_sale_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': sale_predict_x_test_cat})
house_sale_submission.to_csv('submission.csv',index=False)


# In[ ]:




