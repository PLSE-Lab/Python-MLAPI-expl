#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
#import mca
from scipy import sparse
import os,time
import lightgbm as lgb


# In[5]:


os.chdir("../input")
HousTrain = pd.read_csv("train.csv")
HousTrain


# In[6]:


HousTrain.info()


# In[7]:


HousTrain.describe()


# In[8]:


HousTest = pd.read_csv("test.csv")
HousTest


# In[9]:


HousTest.info()


# In[10]:


HousTrain.columns.difference(HousTest.columns)


# In[11]:


Target = HousTrain['SalePrice']
HousTrain1= HousTrain[HousTest.columns]
HousTrain1.columns.difference(HousTest.columns)


# In[12]:


PreProcessHousData = pd.concat([HousTrain1,HousTest], axis = 'index')
PreProcessHousData


# In[13]:


def ReplaceNA(DataCol, ColValue):
    return(DataCol.fillna(value=ColValue))


# In[14]:


PreProcessHousData['PoolQC'].unique()


# In[15]:


PreProcessHousData.isnull().any()

#PreProcessHousData['PoolQC'] = ReplaceNA(PreProcessHousData['PoolQC'],'Np')
#PreProcessHousData['Fence'] = ReplaceNA(PreProcessHousData['Fence'],'Nf')
#PreProcessHousData['MiscFeature'] = ReplaceNA(PreProcessHousData['MiscFeature'],'Nn')
#PreProcessHousData['GarageCond'] = ReplaceNA(PreProcessHousData['GarageCond'],'Ng')
#PreProcessHousData['GarageQual'] = ReplaceNA(PreProcessHousData['GarageQual'],'Ng')
#PreProcessHousData['GarageFinish'] = ReplaceNA(PreProcessHousData['GarageFinish'],'Ng')
#PreProcessHousData['GarageType'] = ReplaceNA(PreProcessHousData['GarageType'],'Ng')
#PreProcessHousData['FireplaceQu'] = ReplaceNA(PreProcessHousData['FireplaceQu'],'Nf')
#PreProcessHousData['BsmtFinType2'] = ReplaceNA(PreProcessHousData['BsmtFinType2'],'NB')
#PreProcessHousData['BsmtFinType1'] = ReplaceNA(PreProcessHousData['BsmtFinType1'],'NB')


# In[16]:


#NullColumns = PreProcessHousData.columns[PreProcessHousData.isnull().any()]
HousTrain1.describe().columns
#CharNulCol = NullColumns.difference[HousTrain1.describe().columns]
#CharNulCol
#CharPreProcHousDataCol = 
#PreProcessHousData[NullColumns]
#NullColumns
#HousTrain1.describe().columns.tolist()
#NumColData=PreProcessHousData[HousTrain1.describe().columns]
#NumNullCol = NumColData.columns[NumColData.isnull().any()]


# In[17]:


#NumColData[NumNullCol].fillna(value=NumColData.mean())
PreProcessHousData=PreProcessHousData.fillna(PreProcessHousData.mean())#Missing Numeric Columns Mean


# In[18]:


#Missing Categorical Data
NullColumns = PreProcessHousData.columns[PreProcessHousData.isnull().any()]
PreProcessHousData[NullColumns] = ReplaceNA(PreProcessHousData[NullColumns],"Other")
NullColumns


# In[19]:


PreProcessHousData.info()


# In[20]:


PreProcessPCAData = PreProcessHousData[HousTrain1.describe().columns]
PreProcessPCAData_Id = PreProcessPCAData["Id"]
PreProcessPCAData.drop(["Id"],axis=1,inplace=True)
PreProcessPCAData.info()


# In[21]:


model_pca = PCA()#n_components=18)


# In[22]:


model_pca.fit(PreProcessPCAData)


# In[23]:


model_pca.explained_variance_ratio_


# In[24]:


#model_pca.get_precision()
var1=np.cumsum(np.round(model_pca.explained_variance_ratio_, decimals=9)*100)
var1


# In[25]:


#So, we take 8 samples:
model_pca = PCA(n_components=8)
model_pca.fit(PreProcessPCAData)
ProcessedPCAData = model_pca.fit_transform(PreProcessPCAData)#It has no predict only transform as we are transforming the dimensions


# In[26]:


ProcessedPCADataDF = pd.DataFrame(ProcessedPCAData, columns = ['P1','P2','P3','P4','P5','P6','P7','P8'])
ProcessedPCADataDF


# In[27]:


NumTransData = pd.concat([PreProcessPCAData_Id.reset_index(drop = True),ProcessedPCADataDF],axis = 1)
NumTransData


# In[28]:


NumTransData.values.T


# In[29]:


type(ProcessedPCAData)


# In[30]:


spPCAdata=sparse.csr_matrix(ProcessedPCAData)
print(spPCAdata)# To see the details of sparse matrix


# In[31]:


PreProcessHousData[HousTrain1.describe().columns]
PreProcessMCAData = PreProcessHousData.drop(HousTrain1.describe().columns,axis=1)
PreProcessMCAData1= PreProcessMCAData.drop(['HouseStyle'],axis=1)
#type(PreProcessMCAData)


# In[32]:


#model_mca= mca.MCA(PreProcessMCAData1)


# In[33]:


le = LabelEncoder()


# In[34]:


y = PreProcessMCAData.apply(le.fit_transform)


# In[35]:


enc = OneHotEncoder(categorical_features = "all")


# In[36]:


enc.fit(y)
PreProcessMCAData_dummy = enc.transform(y)
print(PreProcessMCAData_dummy)


# In[37]:


df_sparse = hstack([spPCAdata,PreProcessMCAData_dummy], format = 'csr')


# In[38]:


print(df_sparse)


# In[39]:


trainData = df_sparse[:HousTrain.shape[0],:]
testData = df_sparse[HousTrain.shape[0]:,:]
OutTestId = PreProcessPCAData_Id.iloc[HousTrain.shape[0]:]
OutTestId


# In[40]:


trainData.shape[0]


# In[41]:


testData.shape[0]


# In[42]:


Target_train = Target
Target_train
Target


# In[43]:


X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse = train_test_split(
                                     trainData, Target_train,
                                     test_size=0.33,
                                     random_state=42
                                     )


# In[44]:


regr = RandomForestRegressor(n_estimators= 1000,       # No of trees in forest
                             criterion = "mse",       # Can also be mae
                             max_features = "sqrt",  # no of features to consider for the best split
                             max_depth= 30,    #  maximum depth of the tree
                             min_samples_split= 2,   # minimum number of samples required to split an internal node
                             min_impurity_decrease=0, # Split node if impurity decreases greater than this value.
                             oob_score = True,       # whether to use out-of-bag samples to estimate error on unseen data.
                             n_jobs = -1,            #  No of jobs to run in parallel
                             random_state=0,
                             verbose = 10            # Controls verbosity of process
                             )


# In[45]:


# 14.1 Do regression
start = time.time()
regr.fit(X_train_sparse,y_train_sparse)
end = time.time()
rf_model_time=(end-start)/60.0
print("Time taken to model: ", rf_model_time , " minutes" )


# In[46]:


regr.oob_score_


# In[47]:


rf_sparse=regr.predict(X_test_sparse)
squared = np.square(rf_sparse - y_test_sparse)
rf_error = np.sqrt(np.sum(squared)/len(y_test_sparse))
rf_error


# In[48]:


SalePrice = regr.predict(testData)
#SalePrice = regr.predict(trainData)
SalePrice


# In[49]:


Output1 = pd.DataFrame(data=SalePrice,columns = ['SalePrice'])
Output1


# In[50]:


Output1['Id'] = OutTestId
Output1
col = ['Id','SalePrice']
Output1 = Output1[col]
Output1


# In[51]:


#Output1.to_csv("Actual_Submit_123_trail.csv",index=False)


# In[52]:


params = {
    'learning_rate': 0.25,
    'application': 'regression',
    'is_enable_sparse' : 'true',
    'max_depth': 3,
    'num_leaves': 60,
    'verbosity': -1,
    'bagging_fraction': 0.5,
    'nthread': 4,
    'metric': 'RMSE'
}


# In[53]:


d_train = lgb.Dataset(X_train_sparse, label=y_train_sparse, max_bin=8000)
d_test = lgb.Dataset(X_test_sparse, label = y_test_sparse, max_bin = 8000)
watchlist = [d_train, d_test]


start = time.time()
model = lgb.train(params,
                  train_set=d_train,
                  num_boost_round=240,
                  valid_sets=watchlist,
                  early_stopping_rounds=20,
                  verbose_eval=10)
end = time.time()
end - start

                  
lgb_pred = model.predict(X_test_sparse)
squared = np.square(lgb_pred - y_test_sparse)
lgb_error = np.sqrt(np.sum(squared)/len(y_test_sparse))
lgb_error


# In[54]:


SalePrice_lgb = model.predict(testData)


# In[55]:


Output2 = pd.DataFrame(data=SalePrice_lgb,columns = ['SalePrice'])
Output2['Id'] = OutTestId
Output2
col = ['Id','SalePrice']
Output2 = Output2[col]
Output2


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




