#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
learn_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'
learn_data =pd.read_csv(learn_file_path)
print(learn_data.describe())


# In[ ]:


import pandas as pd
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
print(melbourne_data.columns)


# In[ ]:


import pandas as pd
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_price =melbourne_data.Price
print(melbourne_price.head())


# In[ ]:


import pandas as pd
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
columns_of_interest = ['Price','Landsize']
columns_data =melbourne_data[columns_of_interest]
print(columns_data.describe())


# In[ ]:


import pandas as pt
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data= pt.read_csv(melbourne_file_path)
columns_of_interest =['Price','Landsize']
columns_interest_data =melbourne_data[columns_of_interest]
print(columns_interest_data.describe())


# In[ ]:


import pandas as PD
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error 
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data=PD.read_csv(melbourne_file_path)
melbourne_data.fillna(melbourne_data.mean(), inplace = True)
model_builder = ['Rooms','Bathroom','Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude']
X=melbourne_data[model_builder]
y=melbourne_data.Price
melbourne_model =DecisionTreeRegressor()
melbourne_model.fit(X, y)
predicted_house_price = melbourne_model.predict(X)
print("The deviation from the error")
print(mean_absolute_error(y,predicted_house_price))
print("The predicitions are")
print(melbourne_model.predict(X.head()))


# 
# 

# In[ ]:


import pandas as pd pd.read_csv(melbourne_file_path) 
melbourne_data.fillna(melbourne_data.mean(), inplace = True)
melbourne_predictors = ["Price","Rooms","Bathroom","Lattitude","Longtitude"]
X = melbourne_data[melbourne_predictors]
y = melbourne_data.Landsize
melbourne_model.fit(X, y)
print(melbourne_model.predict(X.head(10)))


# In[ ]:


import pandas as PD
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error 
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data=PD.read_csv(melbourne_file_path)
melbourne_data.fillna(melbourne_data.mean(), inplace = True)
model_builder = ['Rooms','Bathroom','Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude']
X=melbourne_data[model_builder]
y=melbourne_data.Price
melbourne_model =DecisionTreeRegressor()
melbourne_model.fit(X, y)
predicted_house_price = melbourne_model.predict(X)
print("The deviation from the error")
print(mean_absolute_error(y,predicted_house_price))


# In[ ]:


#To learn about the model optimization 
import pandas as PD
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split
def mae(max_leaf_nodes,predictors_train,predictors_val,targ_train,targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(Predictors_train,targ_train)
    predict_val = model.predict(predictor_val)
    mae=mean_absolute_error(predict_val,targ_val)
    return mae
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data=PD.read_csv(melbourne_file_path)
melbourne_data.fillna(melbourne_data.mean(), inplace = True)
model_builder = ['Rooms','Bathroom','Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude']
X=melbourne_data[model_builder]
y=melbourne_data.Price
train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=3)
for max_leaf_value in [5,10,50,55,1000]:
    mae(max_le)

