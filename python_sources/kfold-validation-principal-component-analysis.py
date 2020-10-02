#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np 
import pandas as pd

import os
print(os.listdir("../input"))


# In[71]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[72]:


import types


# In[73]:


housetrain=pd.read_csv("../input/housetrain/train.csv")


# In[74]:


housetest=pd.read_csv("../input/housetrain/test.csv")


# In[75]:


housetrain.shape


# In[76]:


housetrain.dtypes


# In[77]:


housetrain.head()


# In[78]:


house_train_num=housetrain.select_dtypes(include=[np.number])


# In[79]:



house_train_cat=housetrain.select_dtypes(include=['object'])


# In[80]:


house_train_num.describe().transpose()


# In[81]:


house_train_num_corr=house_train_num.corr()


# In[82]:



house_train_num_corr["SalePrice"]


# In[83]:


house_train_num_cols = []
house_train_num_cols.extend(house_train_num_corr[(house_train_num_corr["SalePrice"]>0.3) ].index.values)
house_train_num_cols.extend(house_train_num_corr[(house_train_num_corr["SalePrice"]<-0.3) ].index.values)


# In[84]:


house_train_num_cols


# In[85]:


h_train_num_col_filtered=house_train_num[house_train_num_cols]


# In[86]:



h_train_num_col_filtered.head()


# In[87]:


(house_train_num.isnull().sum().sort_values(ascending=False))


# In[88]:


for hc in ["LotFrontage","GarageYrBlt","MasVnrArea"]:
    print (hc)
    print(house_train_num[hc].mean())
    print(house_train_num[hc].median())
    


# In[89]:


for col in ["LotFrontage","GarageYrBlt","MasVnrArea"]:
    h_train_num_col_filtered[col].fillna(h_train_num_col_filtered[col].median(),inplace=True)


# In[90]:


from sklearn.preprocessing import LabelEncoder


# In[91]:


le=LabelEncoder()


# In[92]:


house_train_cat.head()


# In[93]:


(house_train_cat.isnull().sum().sort_values(ascending=False))


# In[94]:


for col in ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu"]:
    house_train_cat[col].fillna('No Value',inplace=True)


# In[95]:


for col in ["GarageCond","GarageQual","GarageFinish","GarageType","BsmtFinType2","BsmtExposure","BsmtFinType1","BsmtQual","BsmtCond","MasVnrType","Electrical"]:
    house_train_cat[col].fillna(house_train_cat[col].value_counts().idxmax(),inplace=True)


# In[96]:


house_train_cat1=house_train_cat.apply(le.fit_transform)


# In[97]:


housetraindf1=pd.concat([h_train_num_col_filtered,house_train_cat1],axis=1)


# In[98]:


X1=housetraindf1.drop(["SalePrice"],axis=1)


# In[99]:



y=housetraindf1["SalePrice"]


# In[100]:


from sklearn.model_selection import train_test_split


# In[101]:


X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3)


# In[102]:


house_test_num=housetest.select_dtypes(include=[np.number])
house_test_cat=housetest.select_dtypes(include=['object'])


# In[103]:



house_train_num_cols.remove('SalePrice')


# In[104]:


h_test_num_col_filtered=house_test_num[house_train_num_cols]


# In[105]:


(h_test_num_col_filtered.isnull().sum().sort_values(ascending=False))


# In[106]:



for hc in ["LotFrontage","GarageYrBlt","MasVnrArea","TotalBsmtSF","GarageArea","GarageCars","BsmtFinSF1"]:
    print (hc)
    print(h_test_num_col_filtered[hc].mean())
    print(h_test_num_col_filtered[hc].median())
    


# In[107]:


for col in ["LotFrontage","GarageYrBlt","MasVnrArea","TotalBsmtSF","GarageArea","GarageCars","BsmtFinSF1"]:
    h_test_num_col_filtered[col].fillna(h_test_num_col_filtered[col].median(),inplace=True)


# In[108]:


(house_test_cat.isnull().sum().sort_values(ascending=False))


# In[109]:


for col in ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu"]:
    house_test_cat[col].fillna('No Value',inplace=True)


# In[110]:


for col in ["GarageCond","GarageQual","GarageFinish","GarageType","BsmtQual","BsmtCond","BsmtFinType2","BsmtExposure","BsmtFinType1","MasVnrType","MSZoning","Utilities","Functional","KitchenQual","SaleType","Exterior2nd","Exterior1st"]:
    house_test_cat[col].fillna(house_test_cat[col].value_counts().idxmax(),inplace=True)


# In[111]:


(house_test_cat.isnull().sum().sort_values(ascending=False))


# In[112]:


house_test_cat1=house_test_cat.apply(le.fit_transform)


# In[113]:


housetestdf1=pd.concat([h_test_num_col_filtered,house_test_cat1],axis=1)


# In[114]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[115]:


DecTree=DecisionTreeRegressor()
RandFor=RandomForestRegressor(n_estimators=5000)
GBM=GradientBoostingRegressor(n_estimators=3000)


# In[116]:


dt_M=DecTree.fit(X_train,y_train)

dt_M.score(X_train,y_train)


# In[117]:


rf_M=RandFor.fit(X_train,y_train)
rf_M.score(X_train,y_train)


# In[118]:


gb_M=GBM.fit(X_train,y_train)
gb_M.score(X_train,y_train)
             


# In[119]:


from sklearn.metrics import accuracy_score, r2_score


# In[120]:


gb_y_pred = gb_M.predict(X_test)
rf_y_pred = rf_M.predict(X_test)
dt_y_pred = dt_M.predict(X_test)


# In[121]:



r2_score(y_test,gb_y_pred)


# In[122]:


r2_score(y_test,rf_y_pred)


# In[123]:


r2_score(y_test,dt_y_pred)


# In[124]:



y_eval=gb_M.predict(housetestdf1)


# In[125]:


from sklearn.model_selection import KFold
from sklearn.decomposition import PCA


# In[126]:


def train_test_evaluate_with_kFold(train_data,kmax,algo):
  test_scores={}
  train_scores={}
  train_cols=train_data.shape[1]
  train_cols_25 = int(train_cols/4)
  train_cols_50 = int(train_cols_25*2)
  train_cols_75 = int(train_cols_25*3)
  for n_comp in [train_cols_25,train_cols_50,train_cols_75,train_cols-1]:
    kf = KFold(n_splits=kmax)
    sum_train = 0
    sum_test = 0
    #data = housetraindf1
    data = train_data
    for train, test in kf.split(data):
        pca = PCA(n_components=n_comp)
        #train_data = np.array(data)[train]
        #test_data = np.array(data)[test]
        train_data = data.iloc[train,:]
        test_data = data.iloc[test,:]
        x_train = train_data.drop(["SalePrice"],axis=1)
        principalComponents_train = pca.fit_transform(x_train)
        y_train = train_data["SalePrice"]
        x_test = test_data.drop(["SalePrice"],axis=1)
        principalComponents_test = pca.fit_transform(x_test)
        y_test = test_data["SalePrice"]
        algo_model = algo.fit(principalComponents_train,y_train)
        sum_train += algo_model.score(principalComponents_train,y_train)
        y_pred = algo_model.predict(principalComponents_test)
        sum_test += r2_score(y_test,y_pred)
    average_test = sum_test/kmax
    average_train = sum_train/kmax
    test_scores[n_comp] = average_test
    train_scores[n_comp] = average_train
    print("kvalue: ",n_comp)
  return (train_scores,test_scores)


# In[127]:


from sklearn.linear_model import LinearRegression
RandFor_0=RandomForestRegressor(n_estimators=5000)
DecTree_0=DecisionTreeRegressor()
GBM_0=GradientBoostingRegressor(n_estimators=3000)
LinReg_0=LinearRegression()


# In[128]:


algo_dict = {"LinReg":LinReg_0,"DecTree":DecTree_0,"GBM":GBM_0,"RF":RandFor_0}
algo_train_scores={}
algo_test_scores={}


# In[129]:


max_kfold = 8
for algo_name in algo_dict.keys():
    print(algo_name)
    train_score, test_score = train_test_evaluate_with_kFold(housetraindf1,max_kfold+1,algo_dict[algo_name])
    algo_train_scores[algo_name]=train_score
    algo_test_scores[algo_name]=test_score
    
print(algo_train_scores)
print(algo_test_scores)


# In[130]:


test_scores_df=pd.DataFrame(algo_test_scores)
train_scores_df=pd.DataFrame(algo_train_scores)


# In[131]:


from matplotlib import style


# In[132]:


train_scores_df.plot(figsize=(6,6))


# In[133]:


test_scores_df.plot(figsize=(6,6))


# In[134]:


train_scores_df


# In[135]:



test_scores_df

