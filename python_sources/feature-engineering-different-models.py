''' Carmen Su  - Kaggle House Price - 
If you find it useful in any ways, please upvote this kernel, it'll encourage me to do more. Many thanks!
Data preprocessing: Feature Selection, Label Encoding and filling NaNs 
Models used: Linear Regression, LASSO Regression, Random Forest Regression
Output: one .csv file for each model prediction 
'''
print ("start")

# data analysis and wrangling tools
import pandas as pd
import numpy as np
import csv

# machine learning tools and validation tools
import sklearn
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from math import sqrt

# visualization tools
import matplotlib.pyplot as plt

# Defined functions

# check Nans in data 
def checkNans (dataset, str):
	if dataset.isnull().values.any() == True :
		print (str,":",  dataset.isnull().values.sum())
	else :
		print (str,":", "0")

# check percentage of missing data in each feature
def miss (dataset, str):
	total = dataset.isnull().sum().sort_values(ascending=False)
	percent = (dataset.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
	missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
	print (str)
	print (missing_data)

# examine dataset
def examine (dataset, str1, str2):
	print ()
	print ("examine", str1, str2, ":")
	print (dataset.shape)
	print (list(dataset.columns.values))	
	print (dataset.head())
	print ()

def ml (model, alpha, train, label, test, str1): 
	print ("Training Model: ", model) # Linear Regression, LASSO regression
	if model == "Linear Regression" :
		reg = linear_model.LinearRegression()
		reg.fit(train, label)
		pred = reg.predict(test)

	elif model == "LASSO Regression" :
		lassoreg = Lasso(alpha = alpha, max_iter=30000)
		lassoreg.fit(train,label)	
		pred = lassoreg.predict(test)

	elif model == "Random Forest Regression" :
		rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
		rf.fit(train, label)
		pred = rf.predict(test)

	else :
		print ("Model Input Error")

	pred = pred.reshape((-1,1))
	prediction = pd.DataFrame(pred)
	prediction.columns = ["SalePrice"]
	prediction.index = prediction.index + 1461
	file_name = str1 + ".csv"
	prediction.to_csv(file_name)

# input data files
train= pd.read_csv('../input/train.csv') # original dataset with lable
test = pd.read_csv('../input/test.csv')

label_df = train.iloc[:, -1] # extracing lables from original data 
train_df =train.iloc[:, 1:-1]
test_df = test.iloc[:, 1:]

examine (train_df, "train_df", " ")
examine (test_df, "test_df", " ")

df = pd.concat([train_df, test_df])
examine (df, "df", "df combines train_df and test_df for cleansing")

print ("Feature Engerineering")
dropped_features = ["Street",
					"Alley",
					"LotShape",
					"YearBuilt",
					"YearRemodAdd",
					"RoofStyle",
					"RoofMatl",
					"Exterior1st",
					"Exterior2nd",
					"BsmtFinSF1",
					"BsmtFinSF2",
					"1stFlrSF",
					"2ndFlrSF",
					"Fireplaces",
					"FireplaceQu",
					"GarageYrBlt",
					"GarageFinish",
					"GarageCars",
					"GarageQual",
					"PavedDrive",
					"MiscFeature",
					"MiscVal",
					"MoSold",
					"YrSold", 
					"PoolQC", # PoolQC with 0.99 missing data
					"Fence"
					]

df = df.drop(dropped_features, 1)
examine (df, "df", "after dropping unnecessary features")

print ("Label Encoding")
# Convert object columns to categories, then implement label encoding
obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()
examine (obj_df, "obj_df", "ater extract objects from df")

print ("column header of obj_df: ")
print (list(obj_df.columns.values))
col_obj = ['BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
								'BsmtQual', 'CentralAir', 'Condition1', 'Condition2', 'Electrical', 
								'ExterCond', 'ExterQual', 'Foundation', 'Functional', 
								'GarageCond', 'GarageType', 'Heating', 'HeatingQC', 'HouseStyle', 
								'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig', 'MSZoning', 
								'MasVnrType', 'Neighborhood', 'SaleCondition', 'SaleType', 'Utilities']

for col in col_obj:
     df[col] = df[col].astype('category')
     df[col] = df[col].cat.codes

examine (df, "df", "after label encoding")
print (df.dtypes)

checkNans (df, "df before modeling")
miss (df, "missin data in df")
df = df.fillna(0)
checkNans (df, "df after fill NaNs with 0")

print ()
print ("Selecting dataset for model: ") 
print ("Row ID in train_df: ", train_df.index)

train_80 = df.iloc[:1167, :]
lable_80 = label_df[:1167]
test_20 = df.iloc[1167:1459, :]
valid_20 = label_df[1168:]

train_full = df.iloc[:1460, :]
examine (train_full, "train_full", "")
label_full = label_df
print ("examine label_full", label_full)
test_full = df.iloc[1460:, :]
examine (test_full, "test_full", "")

print ("Predict with real test kaggle data with Linear Regression")
ml ("Linear Regression", 0.0, train_full, label_full, test_full, "prediction_full_reg")
ml ("LASSO Regression", 5000.0, train_full, label_full, test_full, "prediction_full_lasso")
ml ("Random Forest Regression", 0.0, train_full, label_full, test_full, "prediction_full_rf")

print ('end')