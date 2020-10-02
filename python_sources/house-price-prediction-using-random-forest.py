#!/usr/bin/env python
# coding: utf-8

# # **HOUSE PRICE PREDICTION USING RANDOM FOREST WITH HYPER PARAMETER OPTIMIZATION**

# ## TOPICS:

# #### **            1: DATA PREPROCESSING
# ####    **         2: HYPER PARAMETER OPTIMIZATION
# ####      **       3: USING RANDOM FOREST

# # DATA PREPROCESSING

# In[ ]:


#first import the module pandas and numpy
import pandas as pd#this module provide data frame 
import numpy as np


# In[ ]:


#import the data using pandas data frame
test ='../input/test.csv'
train = '../input/train.csv'
df = pd.read_csv(test)
df1 = pd.read_csv(train)
#to visualize all number of rows and columns to be displayed to preprocess the data
pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 1500)


# ## HANDLING MISSING DATA

# In[ ]:


#some missing data have continous values of numerical data so to fill the value with median values
median = df['LotFrontage'].median()
median1 = df['BsmtUnfSF'].median()
median2 = df['BsmtUnfSF'].median()
median3 = df['GarageArea'].median()


# In[ ]:


#check through every rows and columns for missing values and fill according, some are categorical data and some are numeric data
N = 'None'
df['LotFrontage'].fillna(median, inplace=True)
print (df['MasVnrType'].fillna(N ,inplace=True))
print (df['Alley'].fillna(N ,inplace=True))
print (df['Utilities'].fillna('AllPub' ,inplace=True))
print (df['MSZoning'].fillna('RH' ,inplace=True))
print (df['Exterior1st'].fillna('VinylSd' ,inplace=True))
print (df['Exterior2nd'].fillna('HdBoard' ,inplace=True))
print (df['MasVnrType'].fillna(N ,inplace=True))
print (df['MasVnrArea'].fillna(0 ,inplace=True))
print (df['FireplaceQu'].fillna(N ,inplace=True))
print (df['GarageType'].fillna(N, inplace=True))
print (df['GarageYrBlt'].fillna(N, inplace=True))
print (df['GarageFinish'].fillna(N, inplace=True))
print (df['GarageQual'].fillna(N, inplace=True))
print (df['GarageCond'].fillna(N, inplace=True))
print (df['PoolQC'].fillna(N, inplace=True))
print (df['Fence'].fillna(N, inplace=True))
print (df['MiscFeature'].fillna(N, inplace=True))
print (df['BsmtQual'].fillna(N, inplace=True))
print (df['BsmtCond'].fillna(N, inplace=True))
print (df['BsmtExposure'].fillna(N, inplace=True))
print (df['BsmtFinType1'].fillna(N, inplace=True))
print (df['BsmtFinType2'].fillna(N, inplace=True))
print (df['BsmtFinSF1'].fillna(0, inplace=True))
print (df['BsmtFinSF2'].fillna(0, inplace=True))
print (df['BsmtUnfSF'].fillna(median1, inplace=True))
print (df['TotalBsmtSF'].fillna(median2, inplace=True))
print (df['BsmtFullBath'].fillna(0, inplace=True))
print (df['BsmtHalfBath'].fillna(0, inplace=True))
print (df['KitchenQual'].fillna('TA', inplace=True))
print (df['Functional'].fillna('Typ', inplace=True))
print (df['GarageCars'].fillna(1, inplace=True))
print (df['GarageArea'].fillna(median3, inplace=True))
print (df['SaleType'].fillna('WD', inplace=True))


# In[ ]:


print (df.isnull().sum().sum())
#find duplicate
print(df.duplicated().sum())


# In[ ]:


print (df1 ['LotFrontage'])
print (df1 ['LotFrontage'].isnull())

print(df1.isnull().sum())

median = df1['LotFrontage'].median()
print (median)
df1['LotFrontage'].fillna(median, inplace=True)
N = 'None'
print (df1['MasVnrType'].fillna(N ,inplace=True))
print (df1['Alley'].fillna(N ,inplace=True))
print (df1['MasVnrArea'].fillna(0 ,inplace=True))
print (df1['FireplaceQu'].fillna(N ,inplace=True))
print (df1['GarageType'].fillna(N, inplace=True))
print (df1['GarageYrBlt'].fillna(N, inplace=True))
print (df1['GarageFinish'].fillna(N, inplace=True))
print (df1['GarageQual'].fillna(N, inplace=True))
print (df1['GarageCond'].fillna(N, inplace=True))
print (df1['PoolQC'].fillna(N, inplace=True))
print (df1['Fence'].fillna(N, inplace=True))
print (df1['MiscFeature'].fillna(N, inplace=True))
print (df1['BsmtQual'].fillna(N, inplace=True))
print (df1['BsmtCond'].fillna(N, inplace=True))
print (df1['BsmtExposure'].fillna(N, inplace=True))
print (df1['BsmtFinType1'].fillna(N, inplace=True))
print (df1['BsmtFinType2'].fillna(N, inplace=True))
print (df1['Electrical'].fillna('SBrkr', inplace=True))
#find missing total of missing values
print (df1.isnull().sum())
print (df1.isnull().values.any())
print (df1.isnull().sum().sum())
#find duplicate
print(df1.duplicated().sum())


# ## CONVERT TO NUMERICAL DATA

# #### '''Data will not always comes in numeric form, you should need to convert categorical data into numerical data before feeding them to an algorithm '''

# In[ ]:


def convert_non_numerical_data(df):
      columns = df.columns.values
      
      for column in columns:
            text_to_val = {}
            def convert_val(val):
                  return text_to_val[val]
            if df[column].dtype != np.int64 and df[column].dtype !=np.float64:
                  unique_element = set(df[column].values.tolist())
                  x=0
                  for unique in unique_element:
                        if unique not in text_to_val:
                              text_to_val[unique]=x
                              x+=1
                  df[column]=list(map(convert_val, df[column]))
      return df 
df = convert_non_numerical_data(df)   
print(df) 


# In[ ]:


def convert_non_numerical_data(df1):
      columns = df1.columns.values
      
      for column in columns:
            text_to_val = {}
            def convert_val(val):
                  return text_to_val[val]
            if df1[column].dtype != np.int64 and df1[column].dtype !=np.float64:
                  unique_element = set(df1[column].values.tolist())
                  x=0
                  for unique in unique_element:
                        if unique not in text_to_val:
                              text_to_val[unique]=x
                              x+=1
                  df1[column]=list(map(convert_val, df1[column]))
      return df1 
df1 = convert_non_numerical_data(df1)   
print(df1) 


# In[ ]:


#split the data into explonatory variable and response variable
x =df1.iloc[:,:-1].values
y =df1.iloc[:,80].values
y = y.reshape(-1,1)


# # HYPER PARAMETER OPTIMIZATION

# In[ ]:


#import sklearn modules
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
#from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error


# In[ ]:


#implementing algorithm
regr=RandomForestRegressor().fit(x,y)
#feature selection
feature_selection = SelectFromModel(regr,threshold=0.019)
#fitting algorithm with new selected features
reg = RandomForestRegressor(criterion='mae', max_depth=18,
           max_features='sqrt',
           n_estimators=270, n_jobs=-1,
           random_state=42)

#building pipeline
model = Pipeline([('fs',feature_selection),('reg',reg)])


# In[ ]:


#these are the list of hyper parameter to be optimized
crit_option = ['mse','mae']
max_feature_option = ['sqrt','log2','auto']
max_depth_option = list(range(12,32))

param_grid = dict(reg__criterion =crit_option, reg__max_depth= max_depth_option,
                  reg__max_features = max_feature_option)

print(param_grid)


# ### Note: This may take long time to run depending on your cpu

# In[ ]:


#grid = GridSearchCV(model, param_grid,cv=5)
#grid.fit(x,y)

#grid.grid_scores_

#print(grid.best_params_)
#print(grid.best_score_)


# ## FEATURE SELECTION

# ### Feature selection is essential because you cannot feed all the given data to algorithm, which may lead to overfitting the model result in poor performance

# In[ ]:


regr = RandomForestRegressor( criterion='mae', max_depth=26,
           max_features='sqrt', n_estimators=270, n_jobs=-1,
           random_state=42).fit(x,y)


# In[ ]:


feature_selection = SelectFromModel(regr,threshold=0.019,prefit = True)
X_new = feature_selection.transform(x)
mask = feature_selection.get_support(indices = True)
print(X_new.shape)
mask = feature_selection.get_support(indices=True)


# ### For cross_validation our result we must split our data to training and test set

# In[ ]:


from sklearn.cross_validation import train_test_split as ts

x_train, x_test , y_train , y_test = ts(X_new,y,test_size= 0.3,random_state = 42)


# ### Feed the data to the Random Forest algorithm

# In[ ]:


reg = RandomForestRegressor(bootstrap=True, criterion='mae', max_depth=26,
           max_features='sqrt', n_estimators=270, n_jobs=-1,random_state=42)
reg.fit(x_train,y_train)
predict = reg.predict(x_test)


# In[ ]:


print(regr.feature_importances_)
print('mse', mean_squared_error(y_test,predict))
print('accuracy : %.5f' % r2_score(y_test, predict))


# ###  Accuracy is good, it still can be improved better, Hope its usefull

# ### In same way find the prediction for the originally given test data

# ### thank you, Have a nice day
# 
# ##### BY: Jackson paul.T
