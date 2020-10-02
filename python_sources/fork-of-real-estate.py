#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import scipy as sc
import sklearn as sk
from sklearn import decomposition
from sklearn import svm
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics 
import seaborn as sn
from scipy import stats
import lightgbm as lgbm
import catboost as catgbm
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')   


# ## Loading Data and preprocessing

# In[ ]:


df_train= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_test_index = df_test.Id
df_out = df_train.SalePrice  #seperating output
df_train = df_train.drop(['Id','SalePrice','MiscVal','MiscFeature','Alley'],axis=1) #dropping ID, output and few not so useful features
df_test  = df_test.drop(['Id','MiscVal','MiscFeature','Alley'],axis=1)
df = pd.concat([df_train,df_test], ignore_index = True)  #joining train and test data

dataType = pd.read_csv('../input/datatype/categoricalDataType.csv',header = None,index_col = 0)


# In[ ]:


#handling missing data for lot frontage based on building type and lot area
#Town houses
Twnhs1 = np.mean(df.LotFrontage.loc[df.BldgType.
            isin(['TwnhsE','Twnhs'])&(df.LotFrontage.notnull())
             &(df['LotArea']<=6000)])
Twnhs2  = np.mean(df.LotFrontage.loc[df.BldgType.
            isin(['TwnhsE','Twnhs'])&(df.LotArea>=6000)
            &(df['LotFrontage'].notnull())])    
df.LotFrontage.loc[df.BldgType.
            isin(['TwnhsE','Twnhs'])&df.LotFrontage.isnull()&
            (df['LotArea']<=6000)] = Twnhs1
df.LotFrontage.loc[df.BldgType.
            isin(['TwnhsE','Twnhs'])&df.LotFrontage.isnull()&
            (df['LotArea']>=6000)] = Twnhs2 
# rest of the houses
rhse1 = np.mean(df.LotFrontage.loc[~(df.BldgType.
            isin(['TwnhsE','Twnhs']))&(df.LotFrontage.notnull())
             &(df['LotArea']<=6000)])
rhse2  = np.mean(df.LotFrontage.loc[~(df['BldgType'].
            isin(['TwnhsE','Twnhs']))&(df['LotArea']>=6000)
            &(df['LotFrontage'].notnull())])    
df.LotFrontage.loc[df.LotFrontage.isnull()&
            (df.LotArea<=6000)] = rhse1
df.LotFrontage.loc[df.LotFrontage.isnull()&
            (df.LotArea>=6000)] = rhse2


# In[ ]:


#handling categorical variables
df_cat = df.fillna(0)
df = df.fillna(0)
dataType = dataType.fillna(0)
for col in df.columns: 
    if (dataType.loc[col,1] == 'ordinal'): 
        df_cat = pd.get_dummies(df_cat,columns=[col])
    if (dataType.loc[col,1] == 'nominal'):
        if (dataType.loc[col,2] == 0):
           df_cat.loc[:,col] = df_cat.loc[:,col].astype('category')
           df_cat.loc[:,col] = df_cat.loc[:,col].cat.codes
        else:
          labels = dataType.loc[col,2].split(",")
          replace_map_comp = {col: {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
          print(replace_map_comp)
          df_cat.replace(replace_map_comp, inplace=True)


# In[ ]:


#Seperating the training data and the test data
df_cat_train = df_cat.iloc[0:1460,:]
df_cat_test = df_cat.iloc[1460:,:]


# In[ ]:


#Identify the continuous variables for outlier detection.
cont_variables = df_train.columns[(dataType.loc[df_train.columns,1]=='continuous')]
print(cont_variables)


# In[ ]:


outlier_filter = (np.abs(df_cat_train[cont_variables].apply(sc.stats.zscore)) < 5).all(axis=1) 
train_inp_no_outlier = df_cat_train[outlier_filter]
train_out_no_outlier = df_out[(outlier_filter)]    
print(train_inp_no_outlier.shape)
sn.scatterplot(x = train_inp_no_outlier['GrLivArea'],y = train_out_no_outlier)


# In[ ]:


#scaling the data
scaler =  preprocessing.StandardScaler()
train_scale = scaler.fit(train_inp_no_outlier)
test_inp  = scaler.transform(df_cat_test)
train_inp_scale= scaler.transform(train_inp_no_outlier)
train_out_scale = preprocessing.minmax_scale(train_out_no_outlier,feature_range=(0,1))


# In[ ]:


train_inp,valid_inp,train_out,valid_out = sk.model_selection.train_test_split(train_inp_scale,train_out_scale,test_size=0.1,random_state=50)


# ## preprocessing with pca

# In[ ]:


#pca_fsel = decomposition.PCA(n_components = 30).fit(df_train_inp)
#train_inp = pca_fsel.fit_transform(df_train_inp)
#valid_inp = pca_fsel.fit_transform(df_valid_inp)
#test_inp  = pca_fsel.fit_transform(df_cat_test)
#train_inp =  preprocessing.minmax_scale(train_inp,feature_range=(0, 1))
#valid_inp =  preprocessing.minmax_scale(valid_inp,feature_range=(0, 1))
#test_inp  =  preprocessing.minmax_scale(test_inp,feature_range=(0, 1))
#train_out_pre = preprocessing.minmax_scale(df_train_out,feature_range=(0,1))


# ## Defining the models

# In[ ]:


#defining the model a.k.a setting the hyperparameters
clf_lgbm = lgbm.LGBMRegressor(boosting_type='gbdt', max_depth=4, learning_rate=0.1, n_estimators=200)
clf_gbt = sk.ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=120,
                                           min_samples_split=2, max_depth=4,validation_fraction=0.1)

# Fitting regression model a.k.a training the model 
clf_lgbm.fit(train_inp, train_out)
clf_gbt.fit(train_inp,train_out)

#predicting the validation output and rescaling it to original price
valid_pred1 = clf_lgbm.predict(valid_inp)
valid_pred2 = clf_gbt.predict(valid_inp)
valid_pred1 = preprocessing.minmax_scale(valid_pred1,feature_range=(min(train_out_no_outlier),max(train_out_no_outlier)))
valid_pred2 = preprocessing.minmax_scale(valid_pred2,feature_range=(min(train_out_no_outlier),max(train_out_no_outlier)))
valid_out_rescaled = preprocessing.minmax_scale(valid_out,feature_range=(min(train_out_no_outlier),max(train_out_no_outlier)))
valid_out_rescaled = preprocessing.minmax_scale(valid_out,feature_range=(min(train_out_no_outlier),max(train_out_no_outlier)))
#calculating the error
print(metrics.mean_squared_log_error(valid_pred1,valid_out_rescaled))
print(metrics.mean_squared_log_error(valid_pred2,valid_out_rescaled))

#calculating the training error.
train_pred1 = clf_lgbm.predict(train_inp)
train_pred2 = clf_gbt.predict(train_inp)
train_pred1 = preprocessing.minmax_scale(train_pred1,feature_range=(min(train_out_no_outlier),max(train_out_no_outlier)))
train_pred2 = preprocessing.minmax_scale(train_pred2,feature_range=(min(train_out_no_outlier),max(train_out_no_outlier)))
train_out_rescaled = preprocessing.minmax_scale(train_out,feature_range=(min(train_out_no_outlier),max(train_out_no_outlier)))
train_out_rescaled = preprocessing.minmax_scale(train_out,feature_range=(min(train_out_no_outlier),max(train_out_no_outlier)))
print(metrics.mean_squared_log_error(train_pred1,train_out_rescaled))
print(metrics.mean_squared_log_error(train_pred2,train_out_rescaled))


# ## Model Fit with feature selection

# In[ ]:


featSel = SelectFromModel(clf_lgbm,prefit = True)
train_inp_imp = featSel.transform(train_inp)
valid_inp_imp = featSel.transform(valid_inp)

clf_lgbm_imp = lgbm.LGBMRegressor(boosting_type='gbdt', max_depth=3, learning_rate=0.1, n_estimators=400)
clf_gbt_imp = sk.ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=400,
                                           min_samples_split=10, max_depth=3,validation_fraction=0.1)
# Fit regression model
clf_lgbm_imp.fit(train_inp_imp, train_out)
clf_gbt_imp.fit(train_inp_imp,train_out)

#calculating the validation error
valid_pred1 = clf_lgbm_imp.predict(valid_inp_imp)
valid_pred2 = clf_gbt_imp.predict(valid_inp_imp)
valid_pred1 = preprocessing.minmax_scale(valid_pred1,feature_range=(min(train_out_no_outlier),max(train_out_no_outlier)))
valid_pred2 = preprocessing.minmax_scale(valid_pred2,feature_range=(min(train_out_no_outlier),max(train_out_no_outlier)))
print(metrics.mean_squared_log_error(valid_pred1,valid_out_rescaled))
print(metrics.mean_squared_log_error(valid_pred2,valid_out_rescaled))

#calculating the training error
train_pred1 = clf_lgbm_imp.predict(train_inp_imp)
train_pred2 = clf_gbt_imp.predict(train_inp_imp)
train_pred1 = preprocessing.minmax_scale(train_pred1,feature_range=(min(train_out_no_outlier),max(train_out_no_outlier)))
train_pred2 = preprocessing.minmax_scale(train_pred2,feature_range=(min(train_out_no_outlier),max(train_out_no_outlier)))
print(metrics.mean_squared_log_error(train_pred1,train_out_rescaled))
print(metrics.mean_squared_log_error(train_pred2,train_out_rescaled))


# In[ ]:


print(np.shape(train_inp_imp))


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


#test predict
test_inp_imp = featSel.transform(test_inp)
df_test_index.reset_index()
test_pred = clf_gbt_imp.predict(test_inp_imp)
test_pred =preprocessing.minmax_scale(test_pred,feature_range=(min(train_out_no_outlier),max(train_out_no_outlier)))
pred_values = pd.DataFrame(test_pred, columns=['SalePrice'])
pred_val = pd.concat([df_test_index,pred_values],axis = 1)
pred_val.to_csv('/kaggle/working/submission.csv',index = False)


# In[ ]:


from IPython.display import FileLink
FileLink(r'submission.csv')

