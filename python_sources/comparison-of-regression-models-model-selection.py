#!/usr/bin/env python
# coding: utf-8

# Data description
# The Boston data frame has 506 rows and 14 columns.
# 
# This data frame contains the following columns:
# 
# #### crim
# per capita crime rate by town.
# 
# #### zn
# proportion of residential land zoned for lots over 25,000 sq.ft.
# 
# #### indus
# proportion of non-retail business acres per town.
# 
# #### chas
# Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# 
# #### nox
# nitrogen oxides concentration (parts per 10 million).
# 
# #### rm
# average number of rooms per dwelling.
# 
# #### age
# proportion of owner-occupied units built prior to 1940.
# 
# #### dis
# weighted mean of distances to five Boston employment centres.
# 
# #### rad
# index of accessibility to radial highways.
# 
# #### tax
# full-value property-tax rate per $10,000.
# 
# #### ptratio
# pupil-teacher ratio by town.
# 
# #### black
# 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
# 
# #### lstat
# lower status of the population (percent).
# 
# #### medv
# median value of owner-occupied homes in $1000s.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


with open("../input/housing.csv","r") as f:
    data=f.readlines()

housing_data=[]
for line in data:
    samples=[np.float32(x) for x in line.split()]
    housing_data.append(samples)

housing_data=np.asarray(housing_data)
boston=pd.DataFrame(housing_data,columns=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LTSTAT","MEDV"])
print(boston.head())


# ### Scatter Plot 

# In[ ]:


from pandas.plotting import scatter_matrix

scatter_matrix(boston,figsize=(16,16))
plt.show()


# ### Correlation Matrix

# In[ ]:


import seaborn as sns
cor=boston.corr()
fig=plt.figure(figsize=(12,12))
fig=sns.heatmap(cor,annot=True)
plt.show()


# ### When we take a threshold of |0.4| we get 6 important features
# 
# - seperating dataset to features and target sets

# In[ ]:


X1=boston.loc[:,["RM","PTRATIO","LTSTAT","INDUS","NOX","TAX"]]

y=boston["MEDV"]


# In[ ]:


print("X\n",X1.head(),"\n\nY\n",y.head())


# In[ ]:


print(X1.describe())


# now we take some regressors and fit and test the  model.
# 
# we are going to use LR,lasso,elasticnet,svr,knr,gaussian,decisiontree
# 
# We will cover ensemble methods seperately.
# 
# we are not removing outliers for now. we will check the efficiency by removing outliers later.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV


# setting the seed so that we will get same splits for every case.
# 
# Perfomance of each models can be easily compared then.

# In[ ]:


seed=35
kfold=KFold(n_splits=10,random_state=seed)
scoring="r2"


# splitting the dataset to train and validation sets

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X1,y,test_size=.3,random_state=seed)


# Now we will use gridsearch to get the best parameters

# In[ ]:


models=[]
models.append(["LR",LinearRegression()])
models.append(["ENet",ElasticNet()])

models.append(["SVR",SVR(gamma="scale")])
models.append(["KNR",KNeighborsRegressor()])
models.append(["GPR",GaussianProcessRegressor(normalize_y=True)])
models.append(["CART",DecisionTreeRegressor()])


# In[ ]:


param_grids=[]
LR_param_grid={}
param_grids.append(LR_param_grid)
ENet_param_grid={}
ENet_param_grid["alpha"]=[.001,.01,.1,.3,.5]
ENet_param_grid["l1_ratio"]=[0,.2,.4,.5,.7,1]
param_grids.append(ENet_param_grid)
svr_param_grid={}
svr_param_grid["kernel"]=["poly","linear","rbf"]
svr_param_grid["degree"]=[1,2,3,4]
svr_param_grid["C"]=[.001,0.1,.3,.5,1,2,3]
param_grids.append(svr_param_grid)
knr_param_grid={}
knr_param_grid["n_neighbors"]=[3,5,7,11]
knr_param_grid["weights"]=["uniform","distance"]
param_grids.append(knr_param_grid)
gpr_param_grid={}
param_grids.append(gpr_param_grid)
cart_param_grid={}
cart_param_grid["max_depth"]=[1,2,3,4]
param_grids.append(cart_param_grid)


# ### Running the below cell takes some time and kaggle kernel, while commiting usually stops. 
# So i have stored the results i got seperately and have used in the next cell
# 
# ### for all the cells running GridSearch, I have done the same. Stored the results i got seperately and commented out the cells running GridSearch
# 
# ### you can try running these seperately, if you want

# '''
# results=[]
# for model,params in zip(models,param_grids):
#     gcv=GridSearchCV(estimator=model[1],param_grid=params,cv=kfold,scoring=scoring,iid=False)
#     gcv.fit(x_train,y_train)
#     results.append([model[0],gcv.best_params_,gcv.best_score_])
# ''' 

# In[ ]:


results=[['LR', {}, 0.6743397010174477], ['ENet', {'alpha': 0.1, 'l1_ratio': 0.7}, 0.6758633980611956], ['SVR', {'C': 0.3, 'degree': 1, 'kernel': 'linear'}, 0.6646899021487152], ['KNR', {'n_neighbors': 3, 'weights': 'distance'}, 0.7612886361653691], ['GPR', {}, -13.56991654534516], ['CART', {'max_depth': 3}, 0.7346860566317643]]


# In[ ]:


print(results)


# 
# We can see how these estimators performs on the test set.
# 
# 

# In[ ]:


test_scores=[]
for model,result in zip(models,results):
    clf=model[1]
    clf.set_params(**result[1])
    clf.fit(x_train,y_train)
    score=clf.score(x_test,y_test)
    test_scores.append([model[0],result[1],score])


# In[ ]:


for model,param,score in test_scores:
    print("%s : %0.4f"%(model,score))


# ### KNearestRegressor has performed the best on test set.
# #### Now we can explore Ensemble Methods

# ## Ensemble Methods
# 
# We will now use the family of ensemble methods
# 
# There exists mainly two classes of ensemble methods
# #### Averaging  and Boosting 
# 
# - I am only covering Averaging class of ensemble methods for now. Boosting methods may be added later.
# 

# In averaging methods, we will be trying Bagging Regressor with KNR, RandomForest and ExtraTrees

# In[ ]:


from sklearn.ensemble import BaggingRegressor
bgr_param_dict={"n_estimators":list(np.arange(1,100,5)),"max_samples":list(np.linspace(.1,1,10)),"random_state":[seed]}
knn=KNeighborsRegressor(n_neighbors=3,weights="distance")
bg=BaggingRegressor(base_estimator=knn)


# '''
# gcv=GridSearchCV(estimator=bg,param_grid=bgr_param_dict,iid=False,cv=kfold,scoring=scoring)
# 
# gcv.fit(x_train,y_train)
# '''

# In[ ]:


bagging_best_params = {'max_samples': 1.0, 'n_estimators': 76, 'random_state': 35}
bagging_best_score=0.764816


# In[ ]:


print("best training score : %0.6f\nbest params : %r"%(bagging_best_score,bagging_best_params))


# In[ ]:


bg.set_params(**bagging_best_params)
bg.fit(x_train,y_train)
print("Bagging Test score : %0.6f"%bg.score(x_test,y_test))


# ### KNR Ensemble vs single estimator

# In[ ]:


knn.fit(x_train,y_train)
print(knn.score(x_test,y_test))


# #### Here BaggingRegressor performed worse than base estimator.
# #### We can explore the reasons later.
# #### Feel free to comment the answers

# ### Random Forests

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# We will use GridSearchCV to find the best parametets

# In[ ]:


params_dict={}
params_dict["n_estimators"]=list(np.arange(1,100,5))
params_dict["max_depth"]=[None,2,3,4,5,6,7,8,9,10]
params_dict["max_features"]=[.2,.6,.8,1.0]
params_dict["bootstrap"]=[True]
params_dict["random_state"]=[seed]


# '''
# gcv=GridSearchCV(estimator=RandomForestRegressor(bootstrap=True,random_state=seed),param_grid=params_dict,cv=kfold,scoring=scoring,iid=False)
# gcv.fit(x_train,y_train)
# '''

# In[ ]:


#print(gcv.best_params_)
rfc_best_params={'bootstrap': True,
 'max_depth': 9,
 'max_features': 0.2,
 'n_estimators': 96,
 'random_state': 35}
print(rfc_best_params)


# In[ ]:


#print(gcv.best_score_)
rfc_best_score=0.8607639890718424
print(rfc_best_score)


# In[ ]:


rfc=RandomForestRegressor()
rfc.set_params(**rfc_best_params)
rfc.fit(x_train,y_train)


# ### Extra Trees

# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor


# '''
# gcv=GridSearchCV(estimator=ExtraTreesRegressor(),param_grid=params_dict,cv=kfold,scoring=scoring,iid=False)
# gcv.fit(x_train,y_train)
# '''

# In[ ]:


#print(gcv.best_params_)
et_best_params={'bootstrap': True,
 'max_depth': None,
 'max_features': 0.6,
 'n_estimators': 71,
 'random_state': 35}
print(et_best_params)


# In[ ]:


#print(gcv.best_score_)
et_best_score=0.870845214813861
print(et_best_score)


# In[ ]:


et=ExtraTreesRegressor()
et.set_params(**et_best_params)
et.fit(x_train,y_train)


# In[ ]:


print(et.score(x_test,y_test))


# #### This is the best score we got from the models we used.

# In[ ]:


evaluated_models=['LR', 'ENet', 'SVR', 'KNR', 'GPR', 'CART', 'BAGGING', 'RForest','ETrees']
evaluated_test_scores= [0.6077, 0.6012,0.587,0.6843,-0.0222,0.6223,0.6673,0.8172,0.8490]   


# In[ ]:


plt.figure(figsize=(8,6))
plt.plot(evaluated_models,evaluated_test_scores,marker="o",linestyle="--",color="r")
plt.ylim(-0.5,1)
plt.title("Model Evaluation Plot")
plt.xlabel("Models")
plt.ylabel("R-Score")
plt.show()


# ### We haven't done data cleaning for this dataset. 
# ### Data cleaning and scaling will improve the model performance.
# ### I will cover the impacts of that in another kernel

# In[ ]:




