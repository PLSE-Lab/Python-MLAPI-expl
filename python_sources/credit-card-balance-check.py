#!/usr/bin/env python
# coding: utf-8

#  **Credit card makes use of a variety of alternative data--including telco and transactional information--to predict their clients' credit balance spending and limit balance. While Credit card is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are at time and provide more limitt balance.**

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[ ]:


base_Dataset = pd.read_csv("../input/Credit_Card.csv")


# In[ ]:


base_Dataset.describe().columns


# In[ ]:


base_Dataset=base_Dataset.sample(1000)


# In[ ]:


base_Dataset.reset_index(inplace=True)


# In[ ]:


base_Dataset.drop(['index','ID'],axis=1,inplace=True)


# In[ ]:


base_Dataset.shape


# ****EDA Analysis

# In[ ]:


def nullvalue_function(base_dataset,percentage):
    
    # Checking the null value occurance
    
    print(base_dataset.isna().sum())

    # Printing the shape of the data 
    
    print(base_dataset.shape)
    
    # Converting  into percentage table
    
    null_value_table=pd.DataFrame((base_dataset.isna().sum()/base_dataset.shape[0])*100).sort_values(0,ascending=False )
    
    null_value_table.columns=['null percentage']
    
    # Defining the threashold values 
    
    null_value_table[null_value_table['null percentage']>percentage].index
    
    # Drop the columns that has null values more than threashold 
    base_dataset.drop(null_value_table[null_value_table['null percentage']>percentage].index,axis=1,inplace=True)
    
    # Replace the null values with median() # continous variables 
    for i in base_dataset.describe().columns:
        base_dataset[i].fillna(base_dataset[i].median(),inplace=True)
    # Replace the null values with mode() #categorical variables
    #for i in base_dataset.describe(include='object').columns:
     #   base_dataset[i].fillna(base_dataset[i].value_counts().index[0],inplace=True)
  
    print(base_dataset.shape)
    
    return base_dataset


# In[ ]:


base_dataset_null_value = nullvalue_function(base_Dataset, 30)


# ********Model building

# In[ ]:


y=base_dataset_null_value['LIMIT_BAL']
x=base_dataset_null_value.drop(['LIMIT_BAL'],axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
lm=RandomForestRegressor()
lm.fit(x_train,y_train)
lm.predict(x_test)
pred_values_1=lm.predict(x_test)


# In[ ]:


import numpy as np
def regression_model(predicted_values,y_test):
    
    from sklearn.metrics import mean_absolute_error,r2_score
    from sklearn.metrics import mean_squared_error
    total_error=sum(abs(predicted_values-y_test.values))
    MAE=mean_absolute_error(y_test.values,predicted_values)
    MSE=mean_squared_error(y_test.values,predicted_values)
    RMSE=np.sqrt(mean_squared_error(y_test.values,predicted_values))
    MAPE=sum(abs((y_test.values-predicted_values)/(y_test.values)))/x_test.shape[0]
    r2=r2_score(predicted_values,y_test)
    print("total error",total_error)
    print("MSE",MSE)
    print("MAE",MAE)
    print("RMSE",RMSE)
    print("MAPE",MAPE)
    print("R2",r2)
    return [MSE, MAE,RMSE,MAPE,r2]


# In[ ]:


regression_model(pred_values_1,y_test)


# In[ ]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)
lm.predict(x_test)
pred_values_2=lm.predict(x_test)


# In[ ]:


import numpy as np
def regression_model(predicted_values,y_test):
    
    from sklearn.metrics import mean_absolute_error,r2_score
    from sklearn.metrics import mean_squared_error
    total_error=sum(abs(predicted_values-y_test.values))
    MAE=mean_absolute_error(y_test.values,predicted_values)
    MSE=mean_squared_error(y_test.values,predicted_values)
    RMSE=np.sqrt(mean_squared_error(y_test.values,predicted_values))
    MAPE=sum(abs((y_test.values-predicted_values)/(y_test.values)))/x_test.shape[0]
    r2=r2_score(predicted_values,y_test)
    print("total error",total_error)
    print("MSE",MSE)
    print("MAE",MAE)
    print("RMSE",RMSE)
    print("MAPE",MAPE)
    print("R2",r2)
    return [MSE, MAE,RMSE,MAPE,r2]


# In[ ]:


regression_model(pred_values_2,y_test)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
lm=KNeighborsRegressor(n_neighbors=3)
lm.fit(x_train,y_train)
lm.predict(x_test)
pred_values_3=lm.predict(x_test)


# In[ ]:


import numpy as np
def regression_model(predicted_values,y_test):
    from sklearn.metrics import mean_absolute_error,r2_score
    from sklearn.metrics import mean_squared_error
    total_error=sum(abs(predicted_values-y_test.values))
    MAE=mean_absolute_error(y_test.values,predicted_values)
    MSE=mean_squared_error(y_test.values,predicted_values)
    RMSE=np.sqrt(mean_squared_error(y_test.values,predicted_values))
    MAPE=sum(abs((y_test.values-predicted_values)/(y_test.values)))/x_test.shape[0]
    r2=r2_score(predicted_values,y_test)
    print("total error",total_error)
    print("MSE",MSE)
    print("MAE",MAE)
    print("RMSE",RMSE)
    print("MAPE",MAPE)
    print("R2",r2)
    return [MSE, MAE,RMSE,MAPE,r2]


# In[ ]:


regression_model(pred_values_3,y_test)


# In[ ]:


from sklearn.ensemble import BaggingRegressor
lm=BaggingRegressor(n_estimators=10)
lm.fit(x_train,y_train)
lm.predict(x_test)
pred_values_4=lm.predict(x_test)

import numpy as np
def regression_model(predicted_values,y_test):
    
    from sklearn.metrics import mean_absolute_error,r2_score
    from sklearn.metrics import mean_squared_error
    total_error=sum(abs(predicted_values-y_test.values))
    MAE=mean_absolute_error(y_test.values,predicted_values)
    MSE=mean_squared_error(y_test.values,predicted_values)
    RMSE=np.sqrt(mean_squared_error(y_test.values,predicted_values))
    MAPE=sum(abs((y_test.values-predicted_values)/(y_test.values)))/x_test.shape[0]
    r2=r2_score(predicted_values,y_test)
    print("total error",total_error)
    print("MSE",MSE)
    print("MAE",MAE)
    print("RMSE",RMSE)
    print("MAPE",MAPE)
    print("R2",r2)
    return [MSE, MAE,RMSE,MAPE,r2]


# In[ ]:


regression_model(pred_values_4,y_test)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
lm=DecisionTreeRegressor()
lm.fit(x_train,y_train)
lm.predict(x_test)
pred_values_5=lm.predict(x_test)

import numpy as np
def regression_model(predicted_values,y_test):
    
    from sklearn.metrics import mean_absolute_error,r2_score
    from sklearn.metrics import mean_squared_error
    total_error=sum(abs(predicted_values-y_test.values))
    MAE=mean_absolute_error(y_test.values,predicted_values)
    MSE=mean_squared_error(y_test.values,predicted_values)
    RMSE=np.sqrt(mean_squared_error(y_test.values,predicted_values))
    MAPE=sum(abs((y_test.values-predicted_values)/(y_test.values)))/x_test.shape[0]
    r2=r2_score(predicted_values,y_test)
    print("total error",total_error)
    print("MSE",MSE)
    print("MAE",MAE)
    print("RMSE",RMSE)
    print("MAPE",MAPE)
    print("R2",r2)
    return [MSE, MAE,RMSE,MAPE,r2]


# In[ ]:


regression_model(pred_values_5,y_test)


# In[ ]:


comparision=pd.DataFrame(pred_values_1,pred_values_4)


# In[ ]:


comparision.reset_index(inplace=True)


# In[ ]:


comparision.columns=['Linear','Bagging']


# In[ ]:


comparision


# In[ ]:


comparision['actual']=y_test.values


# With the above all algorithms, Bagging regressor is the more efficient aalgorithm to make prediction of the credit card limit balance.
