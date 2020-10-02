#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Import Data

# In[ ]:


AQI = pd.read_csv("../input/Air Quality Index Prediction.csv")
aq=AQI.iloc[1::2]
aq


# # Data Preprocessing

# In[ ]:


aq=aq.reset_index(drop=True)
aq


# In[ ]:


aq.isna().sum().sum()


# In[ ]:


aq.isna().sum()


# In[ ]:


aq['PM 2.5'][aq['PM 2.5'].isna()]


# In[ ]:


aq['PM 2.5'].mode()


# In[ ]:


aq.iloc[184:185,8:9] = aq.iloc[184:185,8:9].fillna('0')


# In[ ]:


aq.isna().sum().sum()


# # Data Visualization

# In[ ]:


plt.rcParams["figure.figsize"] = 15,18
aq.hist()


# In[ ]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,7))
sns.distplot(aq.SLP, bins = 25)
plt.xlabel('Sea Level Pressure')
plt.ylabel('Count')
plt.title('Sea Level Pressure Distribution')


# In[ ]:


fig_dims = (15, 5)
fig, ax = plt.subplots(figsize=fig_dims)
sns.countplot(aq.TM,ax=ax)


# In[ ]:


fig_dims = (15, 5)
fig, ax = plt.subplots(figsize=fig_dims)
sns.countplot(aq['PM 2.5'],ax=ax)


# In[ ]:


aq.groupby('TM').SLP.mean().plot(kind='line',figsize=(15,4))


# In[ ]:


aq.groupby('PM 2.5').agg(['mean', 'std']).mean().plot(kind='line',figsize=(15,4))


# In[ ]:


aq.groupby(by=['PM 2.5','VM']).agg(['median','max']).mean().plot(kind='line',figsize=(15,4))


# In[ ]:


aq['PM 2.5'] =aq['PM 2.5'].astype("float")


# In[ ]:


from pylab import rcParams
rcParams["figure.figsize"] = 12,8
import matplotlib.pyplot as plt
fig,axes = plt.subplots(3,3)


axes[0,0].set_title("TM")
axes[0,0].boxplot(aq['TM'])
axes[0,1].set_title("T")
axes[0,1].boxplot(aq['T'])
axes[0,2].set_title("SLP")
axes[0,2].boxplot(aq['SLP'])



axes[1,0].set_title("VV")
axes[1,0].boxplot(aq['VV'])
axes[1,1].set_title("VM")
axes[1,1].boxplot(aq['VM'])
axes[1,2].set_title("PM 2.5")
axes[1,2].boxplot(aq['PM 2.5'])


axes[2,0].set_title("Tm")
axes[2,0].boxplot(aq['Tm'])
axes[2,1].set_title("H")
axes[2,1].boxplot(aq['H'])
axes[2,2].set_title("V")
axes[2,2].boxplot(aq['V'])


# In[ ]:


fig_dims = (10, 3)
fig, ax = plt.subplots(figsize=fig_dims)
sns.boxplot(x=aq['SLP'],data=aq,ax=ax)


# In[ ]:


sns.relplot(x="TM",y="SLP",data=aq)


# In[ ]:


sns.catplot(x="TM",y="SLP",data=aq)


# In[ ]:


fig_dims = (15, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(aq.corr(), vmax=.8,annot_kws={'size': 20}, annot=True,ax=ax)


# In[ ]:


sns.pairplot(aq)


# In[ ]:


aq.corr()


# # Train Test Split

# In[ ]:


x=aq.iloc[:,:-1] 
y=aq.iloc[:,-1]


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# # Decision Tree Model

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


dtree=DecisionTreeRegressor(criterion="mse")
dtree.fit(x_train,y_train)


# In[ ]:


print("Coefficient of determination :: train set: {}".format(dtree.score(x_train, y_train)))


# In[ ]:


print("Coefficient of determination :: test set: {}".format(dtree.score(x_test, y_test)))


# In[ ]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(dtree,x,y,cv=5)


# In[ ]:


score.mean()


# In[ ]:


dtreepred = dtree.predict(x_test)
dtreepred


# In[ ]:


sns.distplot(y_test-dtreepred)


# In[ ]:


plt.scatter(y_test,dtreepred)


# # Grid Search Cross Validation

# In[ ]:


params={
 "splitter"    : ["best","random"] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15,20],
 "min_samples_leaf" : [ 1,2,3,4,5,6,7 ],
"min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
 "max_features" : ["auto","log2","sqrt",None ],
    "max_leaf_nodes":[None,10,20,30,40,50,60,70]
    
}


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


random_search=GridSearchCV(dtree,param_grid=params,scoring='neg_mean_squared_error',n_jobs=-1,cv=10,verbose=3)


# In[ ]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[ ]:


from datetime import datetime
start_time = timer(None) 
random_search.fit(x,y)
timer(start_time) 


# In[ ]:


random_search.best_params_


# In[ ]:


random_search.best_score_


# In[ ]:


dtreepredCV=random_search.predict(x_test)
dtreepredCV


# In[ ]:


sns.distplot(y_test-dtreepredCV)


# In[ ]:


plt.scatter(y_test,dtreepredCV)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import r2_score
dtreeMAE=(metrics.mean_absolute_error(y_test, dtreepred))
dtreeMSE=(metrics.mean_squared_error(y_test, dtreepred))
dtreeRMSE=np.sqrt(metrics.mean_squared_error(y_test, dtreepred))
dtree=r2_score(y_test, dtreepred)


# In[ ]:


dtreeCVMAE=(metrics.mean_absolute_error(y_test, dtreepredCV))
dtreeCVMSE=(metrics.mean_squared_error(y_test, dtreepredCV))
dtreeCVRMSE=np.sqrt(metrics.mean_squared_error(y_test, dtreepredCV))
dtreecv=r2_score(y_test, dtreepredCV)


# # Random Forest Model

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


regressor=RandomForestRegressor()
regressor.fit(x_train,y_train)


# In[ ]:


print("Coefficient of determination :: train set: {}".format(regressor.score(x_train, y_train)))


# In[ ]:


print("Coefficient of determination :: train set: {}".format(regressor.score(x_test, y_test)))


# In[ ]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,x,y,cv=5)


# In[ ]:


score.mean()


# In[ ]:


prediction=regressor.predict(x_test)
prediction


# In[ ]:


sns.distplot(y_test-prediction)


# In[ ]:


plt.scatter(y_test,prediction)


# # Randomized Search Cross Validation

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
RandomForestRegressor()


# In[ ]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[ ]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


# In[ ]:


yy = pd.DataFrame(y_test)
yz = pd.DataFrame(prediction)
yy = pd.concat([yy,yz],axis=1)
yy = yy.rename({0:'Prediction'},axis=1)
yy = yy.rename({'PM 2.5':'y_test'},axis=1)


# In[ ]:


sns.scatterplot(x=yy["y_test"], y=yy['Prediction'], data=yy)


# In[ ]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[ ]:


rf = RandomForestRegressor()


# In[ ]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[ ]:


rf_random.fit(x_train,y_train)


# In[ ]:


predictionscv=rf_random.predict(x_test)
predictionscv


# In[ ]:


sns.distplot(y_test-predictionscv)


# In[ ]:


plt.scatter(y_test,predictionscv)


# In[ ]:


from sklearn  import metrics
from sklearn.metrics import r2_score


# In[ ]:


rfMAE=(metrics.mean_absolute_error(y_test, prediction))
rfMSE=(metrics.mean_squared_error(y_test, prediction))
rfRMSE=(np.sqrt(metrics.mean_squared_error(y_test, prediction)))
RF=r2_score(y_test, prediction)
rfRMSE


# In[ ]:


rfCV_MAE=(metrics.mean_absolute_error(y_test, predictionscv))
rfCV_MSE=(metrics.mean_squared_error(y_test, predictionscv))
rfCV_RMSE=(np.sqrt(metrics.mean_squared_error(y_test, predictionscv)))
RF_CV=r2_score(y_test, predictionscv)


# # XGboost Model

# In[ ]:


import xgboost as xgb


# In[ ]:


regressor=xgb.XGBRegressor()
regressor.fit(x_train,y_train)


# In[ ]:


print("Coefficient of determination :: train set: {}".format(regressor.score(x_train, y_train)))


# In[ ]:


print("Coefficient of determination :: train set: {}".format(regressor.score(x_test, y_test)))


# In[ ]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,x,y,cv=5)


# In[ ]:


score.mean()


# In[ ]:


predictionXgb=regressor.predict(x_test)
predictionXgb


# In[ ]:


sns.distplot(y_test-predictionXgb)


# In[ ]:


plt.scatter(y_test,predictionXgb)


# # Randomized Search Cross Validation

# In[ ]:


RandomForestRegressor()


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[ ]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


# In[ ]:


yy = pd.DataFrame(y_test)
yz = pd.DataFrame(prediction)
yy = pd.concat([yy,yz],axis=1)
yy = yy.rename({0:'Prediction'},axis=1)
yy = yy.rename({'PM 2.5':'y_test'},axis=1)


# In[ ]:


sns.scatterplot(x=yy["y_test"], y=yy['Prediction'], data=yy)


# In[ ]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[ ]:


rf = RandomForestRegressor()


# In[ ]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[ ]:


rf_random.fit(x_train,y_train)


# In[ ]:


XbgpredictionCV1=rf_random.predict(x_test)
XbgpredictionCV1


# In[ ]:


sns.distplot(y_test-XbgpredictionCV1)


# In[ ]:


plt.scatter(y_test,XbgpredictionCV1)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import r2_score
XGb_MAE=(metrics.mean_absolute_error(y_test, predictionXgb))
XGb_MSE=(metrics.mean_squared_error(y_test, predictionXgb))
XGb_RMSE=(np.sqrt(metrics.mean_squared_error(y_test, predictionXgb)))
Xgb=r2_score(y_test,predictionXgb)


# In[ ]:


xgbCV_MAE=(metrics.mean_absolute_error(y_test, XbgpredictionCV1))
xgbCV_MSE=(metrics.mean_squared_error(y_test, XbgpredictionCV1))
xgbCV_RMSE=(np.sqrt(metrics.mean_squared_error(y_test, XbgpredictionCV1)))
XGb_CV=r2_score(y_test,XbgpredictionCV1)


# # Best Modal

# In[ ]:


Best_Model ={'Model':['Desicion Tree','Cross_Validated_DTree', 'Random Forest','Cross_Validated_RF', 'XGboost','Cross_Validated_Xgb'],
        'R2_Score':[dtree,dtreecv,RF,RF_CV,Xgb,XGb_CV],'RMSE':[dtreeRMSE,dtreeCVRMSE,rfRMSE,rfCV_RMSE,XGb_RMSE,xgbCV_RMSE],'MSE':[dtreeMSE,dtreeCVMSE,rfMSE,rfCV_MSE,XGb_MSE,XGb_MSE],'MAE':[dtreeMAE,dtreeCVMAE,rfMAE,rfCV_MAE,XGb_MAE,XGb_MAE]}
Best_Model =pd.DataFrame(Best_Model)
Best_Model=Best_Model.set_index('Model')
Best_Model


# In[ ]:





# In[ ]:




