#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



# In[ ]:


house_data_train = pd.read_csv('E:\data set\House_prices\\train.csv') 
house_data_test  = pd.read_csv('E:\data set\House_prices\\test.csv')
#list(house_data_train)
house_data_train1 = house_data_train.copy()
house_data_test1 = house_data_train.copy()
house_data_train2 = house_data_train1.drop(['Id'], axis = 1)
house_data_test2 = house_data_test1.drop(['Id'], axis = 1)


# In[ ]:


house_data_train3 = house_data_train2.replace(np.nan, 0)
house_data_test3 = house_data_test2.replace(np.nan, 0)

house_data_train4 = house_data_train3.copy()
house_data_test4 = house_data_test3.copy()
house_data_train4.get_dtype_counts()
#house_data_train4.dtypes


# In[ ]:



#List = list(house_data_train4)

#List[0]
#house_data_train[List[5]]
#Z = []
#for i in range(len(List)):
#        y =  house_data_train4[List[i]].unique()
#        if len(y) == 1:
#            Z = Z.append(List[i])

#house_data_train4 = house_data_train4.drop(Z, inplace=True, axis=1)


# In[ ]:


# defining words to number for computation
def ConvertNum(data_name, column_name):
    new_data = data_name[column_name].copy()
    y = data_name[column_name].unique()
    x = np.zeros((1, len(y)))
    for i in range( len(y)  -1):
        x[0,i] = (data_name[column_name].eq(y[i]).sum())/1460
        #x[i] = count/1460
        
   
    z =  x[0,:]
    new_data1 = new_data.replace(y,z)
    
    return new_data1
        
    


# In[ ]:


#c  = ConvertNum(house_data_train4, 'LandContour')
#c
#list(house_data_train)
#count  =house_data_train['Alley'].count(t)
#count


# In[ ]:


#List = list(house_data_train)
#ist


# In[ ]:


#house_data_train.info()


# In[ ]:



col = [ 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd','MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'YrSold', 'SaleType', 'SaleCondition']
#len(col)
#house_data_train4
for i in range(len(col)):   
    
    y1 = ConvertNum(house_data_train4, col[i])
    y2 = ConvertNum(house_data_test4, col[i])

    house_data_train4[col[i]] = y1
    house_data_test4[col[i]] = y2
    
house_data_train5 = house_data_train4.copy()
house_data_test5 = house_data_test4.copy()

house_data_train5.to_csv('E:\data set\House_prices\\newfile_1_train.csv')
house_data_test5.to_csv('E:\data set\House_prices\\newfile_1_test.csv')
#list(house_data_test5)


# In[ ]:


#list(house_data_train3)
#col = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual','OverallCond','MasVnrType','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF', '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath','HalfBath', 'BedroomAbvGr', 'KitchenAbvGr','TotRmsAbvGrd','Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea','WoodDeckSF','OpenPorchSF', 'EnclosedPorch','3SsnPorch', 'ScreenPorch','PoolArea','MiscVal','MoSold','SalePrice' ]
#col = ['MSSubClass']
#List3 = List2.remove('MSSubclass')
#del List2(['MSSubClass'])


# In[ ]:


#normalizing data for the rest
col_norm = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual','OverallCond','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF', '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath','HalfBath', 'BedroomAbvGr', 'KitchenAbvGr','TotRmsAbvGrd','Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea','WoodDeckSF','OpenPorchSF', 'EnclosedPorch','3SsnPorch', 'ScreenPorch','PoolArea','MiscVal','MoSold' ]

for i in range(len(col_norm)):
    y1 = (house_data_train5[col_norm[i]] - house_data_train5[col_norm[i]].min())/(house_data_train5[col_norm[i]].max() - house_data_train5[col_norm[i]].min())
    y2 = (house_data_test5[col_norm[i]] - house_data_test5[col_norm[i]].min())/(house_data_test5[col_norm[i]].max() - house_data_test5[col_norm[i]].min())

    house_data_train5[col_norm[i]] = y1
    house_data_test5[col_norm[i]] = y2

house_data_train6 = house_data_train5.copy()
house_data_test6 = house_data_test5.copy()

house_data_train6.to_csv('E:\data set\House_prices\\newfile_2_train.csv')
house_data_test6.to_csv('E:\data set\House_prices\\newfile_2_test.csv')
    




# In[ ]:


#xc =  (house_data_test5['MSSubClass'] - house_data_test5['MSSubClass'].min())/(house_data_test5['MSSubClass'].max() - house_data_test5['MSSubClass'].min())
#xc


# In[ ]:


output = house_data_train6['SalePrice']
house_data_train7 = house_data_train6.copy()
house_data_train7 = house_data_train7.drop('SalePrice', axis = 1)

house_data_test7 = house_data_test6.copy()
output_test = house_data_test7['SalePrice']
house_data_test7 = house_data_test7.drop('SalePrice', axis = 1)
#house_data_train7.shape


# In[ ]:


classifier = LinearRegression()
classifier.fit(house_data_train7, output)
prediction = classifier.predict(house_data_test7)
classifier.score(house_data_test7,output_test)
submit_file = pd.read_csv('E:\data set\House_prices\\sample_submission.csv')
#submit_file.replace(submit_file, prediction)
prediction = np.delete(prediction, (1459), axis = 0)
#prediction.shape
submit_file['SalePrice'] = prediction
submit_file.to_csv('E:\data set\House_prices\\final_submission.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




