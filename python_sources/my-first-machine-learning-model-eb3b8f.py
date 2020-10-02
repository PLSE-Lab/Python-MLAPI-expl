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
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

'''
# Prediccion Melbourne
melbourne_model = DecisionTreeRegressor()
main_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(main_file_path)
melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
melbourne_data_real = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', 'Price']
data_csv = melbourne_data[melbourne_data_real]
X = melbourne_data[melbourne_predictors]
y = melbourne_data.Price
# Fit model
melbourne_model.fit(X, y)
print("Data real")
print(data_csv.head())
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

# Prediccion Iowa

LotArea
YearBuilt
1stFlrSF
2ndFlrSF
FullBath
BedroomAbvGr
TotRmsAbvGrd
'''
iowa_model = DecisionTreeRegressor()
main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'
iowa_data = pd.read_csv(main_file_path)

columnas_csv = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd', 'SalePrice']
predictores = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = iowa_data[predictores]
y = iowa_data.SalePrice
iowa_model.fit(X, y)

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
iowa_model_split = DecisionTreeRegressor()
iowa_model_split.fit(train_X, train_y)

val_predictions = iowa_model_split.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


'''print('Data Real CSV')
print(iowa_data[columnas_csv].head())
print('Impresion de Predictores')
print(X.head())
print('Prediccion SalePrice')
print(iowa_model.predict(X.head()))
'''





# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 
