#!/usr/bin/env python
# coding: utf-8

# 
# #  **Introduction**
# French Motor Claims:
# 
# A dataset including 
# 
# * Policy No: ID
# * 9 Explanatory Variables: Driver Age, Vehicle Age, Vehicle Power, Density, Bonus Malus, Area, Vehicle Brand, Vehicle Gas, Region
# * Weight: Exposure
# * Response Variable: Claims Number
# 
# Objectives:
# 1. Data Exploration / Pre-Process
#     Want to understand our explanatory variables to make suitable grouped and banded variables
# 2. Testing Models
#     Using different techniques such as xgBoost, GLM etc
# 3. Finding the best model
#     

# 
# ### Importing packages and dataset. 
# 
# Also Creating Frequency variable (Claims/Exposure)

# In[ ]:




import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn as sc
import os
from bokeh import __version__ as bk_version


data = pd.read_csv('/kaggle/input/french-motor-claims-datasets-fremtpl2freq/freMTPL2freq.csv')


data['freq'] = data['ClaimNb']/data['Exposure']
data['LogDensity'] = np.log(data['Density'])

print(data.describe)


# ### Creating a subset

# In[ ]:


random_subset = data.sample(n=20000)
subset_summary = random_subset.describe()
print(subset_summary)


# ### Creating a Function to create visualisation of Exposure and Claim Frequency over factors

# In[ ]:


def plotBarChart(data,x,ylimitFREQ):
    EVY= data.groupby(x,as_index=False).agg({'Exposure': 'sum'})
    plt.subplot(1, 2, 1)
    plt.bar(EVY[x],EVY['Exposure'],align='center')
    plt.xlabel(x)
    plt.ylabel('Exposure')
    plt.title("Exposure over " + str(x))
    plt.rcParams['figure.figsize'] = (10,6)
    plt.xticks(rotation=50)
    
    Freq= data.groupby(x,as_index=False).agg({'freq': 'mean'})
    plt.subplot(1, 2, 2)
    plt.bar(Freq[x],Freq['freq'],align='center')
    plt.xlabel(x)
    plt.ylabel('Freq')
    plt.title("Freq over " + str(x))
    plt.rcParams['figure.figsize'] = (10,6)
    plt.ylim(top=ylimitFREQ)
    plt.xticks(rotation=50)   
    
    plt.tight_layout()
plt.show()



# # Data Exploration
# Trying to understand each of the explanatory variables (based on the subset that we have).
# 
# These graphs will give us a good visualisation of how Frequency and Exposure behave on these variables. Whilst we have a rough idea how it might look like, we are looking at France and not the UK hence we might find something else.
# 
# Frequency capped at 2.
# 

# ### Driver Age

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nplotBarChart(data,'DrivAge',2)")


# Exposure
# * Median of 44.
# * Exposure largely concentrated from ~30 to ~60.
# 
# Frequency 
# * As expected, we see a rise in Frequency in young ages and old ages

# ### Vehicle Power

# In[ ]:


plotBarChart(data,'VehPower',2)


# Exposure 
# * largely concentrated around 4-7. With median of 6.
# 
# Frequency
# * Does not give us a good idea of any trends or correlation

# ### Vehicle Age

# In[ ]:


plotBarChart(data,'VehAge',2)


# Exposure
# * median of 6
# * Exposure is concentrated from 1-15
# 
# Frequency
# * Frequency high at Vehicle age 0
# * Also high for some very high Vehicle Ages, though EVY is minimal

# ### Bonus Malus

# In[ ]:


plotBarChart(data,'BonusMalus',2)


# Exposure
# * large amount of our data has a default Bonus Malus of 50, and the rest has minimal exposure.
# 
# Frequency
# * Frequency low for Bonus Malus of 50, where most EVY is concentrated
# * Very high at some high values, but with small EVY

# ### Density
# 

# In[ ]:


plotBarChart(data,'Density',2)


# The standard deviation is very high for density, it seems that there are a large amount of different density. So we want to create a new variable which takes log(density):

# In[ ]:


plotBarChart(data,'LogDensity',2)


# We have logDensity as a continous numeric variable. This is good to be an input for our model.

# ### Area

# In[ ]:


plotBarChart(data,'Area',2)


# Area looks clean and already grouped up as well.
# 
# Exposure
# * Good amount of EVY for every Area except for F
# 
# Frequency
# * Trend of higher frequency from A to F. F hinted to be a risky area

# ### Vehicle Brand

# In[ ]:


plotBarChart(data,'VehBrand',2)


# ### Vehicle Gas

# In[ ]:


plotBarChart(data,'VehGas',2)


# ### Region

# In[ ]:


plotBarChart(data,'Region',2)


# In[ ]:


import geopandas as gpd



sf = gpd.read_file('/kaggle/input/france/departements-version-simplifiee.geojson')


# Looking for a possible chance to merge Map data with the French Motor Claims Project, however it seems that the data is split into 'Department' areas,different to the French Claims data, where it is split into the 22 Regions (old Region mapping).

# # Models

# 1. Check the data
# 2. Split Train/Validation/Holdout Data
# 3. Model 
# 4. Fitting
# 5. Predict

# ### Checking if there are any missing values

# In[ ]:


data.isnull().sum()


# ### One hot encoding variables to make them suitable for XGBoost

# In[ ]:


data_new=pd.get_dummies(data,prefix=['Area','VehBrand','VehGas','Region'])

print(data_new.describe)


# ### Splitting Data and construction DMatrix

# In[ ]:


# Import modules specific for this section
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Proportions we want to split in (must sum to 1)
split_props = pd.Series({
    'train': 0.7,
    'validation': 0.15,
    'holdout': 0.15
})


# Split out training data
df_train,df_not_train = train_test_split(
    data_new, test_size=(1 - split_props['train']), random_state=51, shuffle=True
)
# Split remaining data between validation and holdout
df_validation, df_holdout= train_test_split(
    df_not_train, test_size=(split_props['holdout'] / (1 - split_props['train'])), random_state=13, shuffle=True
)


y_train = df_train.filter(['ClaimNb'])


x_train = df_train.drop(columns=['ClaimNb','IDpol','freq','Exposure'])
x_train_weight = df_train.filter(['Exposure'])

train_dmatrix = xgb.DMatrix(data=x_train,label=y_train,weight=x_train_weight)

y_valid = df_validation.filter(['ClaimNb'])

x_valid = df_validation.drop(columns=['ClaimNb','IDpol','freq','Exposure'])
x_valid_weight = df_validation.filter(['Exposure'])

valid_dmatrix = xgb.DMatrix(data=x_valid,label=y_valid,weight=x_valid_weight)


print(df_validation.describe)
print(x_valid.describe)


# ### XgBoost
# 

# Default Model

# In[ ]:


import xgboost as xgb

params={'eval_metric':'poisson-nloglik',"objective": "count:poisson",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5,'min_child_weight':1, 'reg_alpha': 10}

log_exp= np.log(df_train.filter(['Exposure']))

train_dmatrix.set_base_margin(log_exp)


xgmodel = xgb.train(
    params=params,
    dtrain=train_dmatrix,
    num_boost_round=999,
    early_stopping_rounds=10,
    evals=[(train_dmatrix, "train"),(valid_dmatrix,"valid")],
    verbose_eval=False
)


    
xgb.plot_tree(xgmodel,num_trees=0)
plt.rcParams['figure.figsize'] = [10,10]
plt.show()
    
xgb.plot_importance(xgmodel)
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()


# In[ ]:


from sklearn.metrics import mean_absolute_error

y_train= df_train.iloc[:,1]
y_valid = df_validation.iloc[:,1]
# "Learn" the mean from the training data
mean_train = np.mean(y_train)

# Get predictions on the test set
baseline_predictions = np.ones(y_valid.shape)

# Compute MAE
mae_baseline = mean_absolute_error(y_valid, baseline_predictions)

print("Baseline MAE is {:.2f}".format(mae_baseline))

print("Best Poisson Likelihood: {:.2f} with {} rounds".format(
                 xgmodel.best_score,
                 xgmodel.best_iteration+1))


# ### Final Model

# In[ ]:


params={'eval_metric':'poisson-nloglik',"objective": "count:poisson",'colsample_bytree':0.9 ,'learning_rate':0.1 ,
                'max_depth':3 ,'min_child_weight':1, 'reg_alpha':5, 'subsample':0.9 }

log_exp= np.log(df_train.filter(['Exposure']))

train_dmatrix.set_base_margin(log_exp)


xgmodel = xgb.train(
    params=params,
    dtrain=train_dmatrix,
    num_boost_round=65,
    early_stopping_rounds=10,
    evals=[(train_dmatrix, "train"),(valid_dmatrix,"valid")],
    verbose_eval=False
)


    
xgb.plot_tree(xgmodel,num_trees=0)
plt.rcParams['figure.figsize'] = [10,10]
plt.show()
    
xgb.plot_importance(xgmodel)
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()


# In[ ]:


print(xgmodel)
predictions_valid=xgmodel.predict(valid_dmatrix)


# XGB Regressor

# In[ ]:


xg_reg = xgb.XGBRegressor(eval_metric='poisson-nloglik',objective= "count:poisson",colsample_bytree=0.9 ,learning_rate=0.1 ,
                max_depth=3 ,min_child_weight=1, reg_alpha=5, subsample=0.9)

xg_reg.fit(x_train,y_train)

reg_pred = xg_reg.predict(x_valid)


# ### Lift and Residual

# In[ ]:



import matplotlib.pyplot as plt
import scikitplot as skplt

# Reg_Pred
regpredictions=pd.Series(reg_pred)
regpredictions.rename(columns={"0":"ClaimNb"})
regpredictions = regpredictions.reset_index()
del regpredictions['index']

# Prediction
predictions=pd.Series(predictions_valid)
predictions.rename(columns={"0":"ClaimNb"})
predictions = predictions.reset_index()
del predictions['index']


#Actual

actual_valid = df_validation.iloc[:,8]
actual=pd.Series(actual_valid)
actual = actual.reset_index()
del actual['index']

#Exposure 
exposure = df_validation.iloc[:,2]
exposure=pd.Series(exposure)
exposure = exposure.reset_index()
del exposure['index']

print(regpredictions)
print(predictions)
print(actual)


# ### Residual

# In[ ]:


nppred=np.array(predictions)
npactual=np.array(actual)
npexposure=np.array(exposure)

npregpred=np.array(regpredictions)

residual=npactual-nppred
reg_residual = npactual-npregpred


overall_res=np.average(residual, weights=npexposure)
overall_reg_res=np.average(reg_residual, weights=npexposure)

print(overall_res)
print(overall_reg_res)


# ### Saving the model and predicted validation dataset

# In[ ]:


print(df_validation.describe())


# In[ ]:



filtered_pred_valid_set=df_validation.filter(['IDpol','ClaimNb','Exposure','freq'])

filtered_pred_valid_set = filtered_pred_valid_set.reset_index()
del filtered_pred_valid_set['index']

df_validation = df_validation.reset_index()
del df_validation['index']

predicted_validation_set = df_validation.merge(predictions, left_index=True, right_index=True)
predicted_validation_set.columns.values[-1] = 'pred_ClaimNb'

filt_pred_valid_set = filtered_pred_valid_set.merge(predictions, left_index=True, right_index=True)
filt_pred_valid_set.columns.values[-1] = 'pred_ClaimNb'

filt_regpred_valid_set = filtered_pred_valid_set.merge(regpredictions, left_index=True, right_index=True)
filt_regpred_valid_set.columns.values[-1] = 'pred_ClaimNb'

predicted_reg_validation_set = df_validation.merge(regpredictions, left_index=True, right_index=True)
predicted_reg_validation_set.columns.values[-1] = 'pred_ClaimNb'

predicted_validation_set.to_pickle('xgb_pred_valid_set_new.gzip')
filt_pred_valid_set.to_pickle('xgb_filtered_pred_valid_set_new.gzip')
filt_regpred_valid_set.to_pickle('xgb_filt_reg_pred_valid_set_new.gzip')
predicted_reg_validation_set.to_pickle('xgb_pred_reg_valid_set_new.gzip')

x_train.to_pickle('xtrain.gzip')

print(filt_pred_valid_set.describe)
print(predicted_validation_set.describe)

import pickle
file_name = "regressorxgbmodel.pkl"
file_name2 = "xgbmodel.pkl"


# save
pickle.dump(xg_reg, open(file_name, "wb"))
pickle.dump(xgmodel, open(file_name2, "wb"))


# In[ ]:


print(filt_pred_valid_set.describe())

