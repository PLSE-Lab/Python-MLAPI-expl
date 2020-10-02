#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import train_test_split
import xgboost as xgb
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

ss = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# Purpose of this function is to one-hot-encode categorical features after NaN values, low frequency categories, and categories that only exist in the testing set have been converted to an 'other' category

# In[ ]:


def checkcategories(train, test, fthreshold):
    #1 Check to see if all unique test values exist in train
    #If not, convert to NaN
    
    trt = pd.concat((train,test),axis=0)
    trtshape = trt.shape
    alt = np.zeros(trtshape[0])
    
    tru = train.unique()
    teu = test.unique()
    
    for values in teu:
        if (values in tru) == False:
            trt[(trt==values)==True] = 'zzztempzzz'
    
    #2 Convert all NaN to its own category
    
    trt[(trt.isnull())==True] = 'zzztempzzz'
    
    #3 Do frequency count, provide threshold that converts dummy variables less than threshold to the NaN category in step 2, remove columns of dummy variables that are less than threshold
    
    tdict = trt.value_counts().to_dict()
    
    for values in tdict:
        if tdict[values] < fthreshold:
            trt[(trt==values)==True] = 'zzztempzzz'
    
    #3 Get Dummies
    
    dummies = pd.get_dummies(trt)
    
    #4 Split back to train,test
    
    train = dummies.iloc[0:train.shape[0],:]
    test = dummies.iloc[train.shape[0]:train.shape[0]+test.shape[0]+1,:]
    
    return train,test

traintemp,testtemp = checkcategories(train.loc[:,'Fence'],test.loc[:,'Fence'],100)


# Features have been classified as categorical and numerical

# In[ ]:


catcolumns = [1,2,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23,24,25,26,27,28,29,30,31,32,33,35,39,40,41,42,47,48,49,50,52,53,55,56,57,58,59,60,61,63,64,65,72,73,74,77,78,79]
numcolumns = [3,4,17,18,19,20,34,36,37,38,43,44,45,46,51,54,62,66,67,68,69,70,71,75,76]


# In[ ]:


for count,values in enumerate(catcolumns):
    traintemp,testtemp = checkcategories(train.iloc[:,values],test.iloc[:,values],100)
    if count == 0:
        traincat = traintemp.to_numpy()
        testcat = testtemp.to_numpy()
    else:
        traincat = np.concatenate((traincat,traintemp.to_numpy()),axis=1)
        testcat = np.concatenate((testcat,testtemp.to_numpy()),axis=1)


# In[ ]:


for count,values in enumerate(numcolumns):
    traintemp = train.iloc[:,values]
    testtemp = test.iloc[:,values]
    
    mx = max(traintemp.max(),testtemp.max())
    mi = min(traintemp.min(),testtemp.min())
    
    traintemp = (traintemp - mi)/(mx-mi)
    testtemp = (testtemp - mi)/(mx-mi)
    
    a = traintemp.to_numpy()
    b = testtemp.to_numpy()
    
    temp = np.concatenate((a.reshape(-1,1),b.reshape(-1,1)))
    s = skew(temp)
    
    if s > .75:
        temp = np.log1p(temp)
    
    traintemp = temp[0:traintemp.shape[0]]
    testtemp = temp[traintemp.shape[0]:traintemp.shape[0]+testtemp.shape[0]]
    
    if count == 0:
        trainnum = traintemp
        testnum = testtemp
    else:
        trainnum = np.concatenate([trainnum,traintemp],axis=1)
        testnum = np.concatenate([testnum,testtemp],axis=1)


# In[ ]:


trainall = np.concatenate((traincat,trainnum),axis=1)
testall = np.concatenate((testcat,testnum),axis=1)
trainall[np.isnan(trainall)==True]=.5
testall[np.isnan(testall)==True]=.5


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(trainall,train.iloc[:,80].to_numpy(), test_size=0.2, random_state=42)


# **Below is a series of models that we have tested**

# XGB Regressor

# In[ ]:


xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .7, learning_rate = .1,
         max_depth = 3, reg_lambda = .8, n_estimators = 500, evaluation_metric = 'rmsle')
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds)-np.log(y_test)))))


# **Ridge Regression**

# In[ ]:


from sklearn import linear_model
reg = linear_model.Ridge(alpha=0.01, fit_intercept=True, normalize=True, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
preds2[preds2<0]=np.mean(preds2)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Ordinary Least Squares**

# In[ ]:


from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Lasso Regression**

# In[ ]:


from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.001, max_iter = 50000)
reg.fit(X_train, y_train)
preds3 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Elastic Net**

# In[ ]:


from sklearn import linear_model
reg = linear_model.ElasticNet(alpha=0.4, l1_ratio = .5, max_iter = 50000)
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Lars Lasso**

# In[ ]:


from sklearn import linear_model
from warnings import filterwarnings
filterwarnings('ignore')
reg = linear_model.LassoLars(alpha=0.5)
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Orthogonal Matching Pursuit**

# In[ ]:


from sklearn import linear_model
from warnings import filterwarnings
filterwarnings('ignore')
reg = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize=True, precompute='auto')
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Bayesian Ridge**

# In[ ]:


from sklearn import linear_model
from warnings import filterwarnings
filterwarnings('ignore')
reg = linear_model.BayesianRidge()
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Stochastic Gradient Descent**

# In[ ]:


from warnings import filterwarnings
filterwarnings('ignore')
reg = linear_model.SGDRegressor()
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Kernel Ridge**

# In[ ]:


from warnings import filterwarnings
from sklearn import kernel_ridge
filterwarnings('ignore')
reg = kernel_ridge.KernelRidge(alpha=.001, kernel='poly', gamma=None, degree=3, coef0=1, kernel_params=None)
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Support Vector Regression**

# In[ ]:


from sklearn import svm
filterwarnings('ignore')
#C=1000000
reg = svm.SVR(kernel='poly', C=100000, gamma='auto', degree=7, epsilon=.001,
               coef0=1,shrinking=True,)
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **NuSVR**

# In[ ]:


from sklearn import svm
filterwarnings('ignore')
#C=1000000
reg = svm.NuSVR(kernel='linear', C=10000, gamma='auto', degree=7,
               coef0=1,shrinking=True,)
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **K-Nearest Neighbors**

# In[ ]:


from sklearn import neighbors
filterwarnings('ignore')
reg = neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Radius Neighbors**

# In[ ]:


from sklearn import neighbors
filterwarnings('ignore')
reg = neighbors.RadiusNeighborsRegressor()
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# Gaussian Process Regression

# In[ ]:


from sklearn import gaussian_process
filterwarnings('ignore')
reg = gaussian_process.GaussianProcessRegressor(kernel=None, alpha=.0001, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# PLSRegression

# In[ ]:


from sklearn import cross_decomposition
filterwarnings('ignore')
reg = cross_decomposition.PLSRegression(n_components=2, scale=False, max_iter=5000, tol=.01, copy=True)
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Cross Decomposition**

# In[ ]:


from sklearn import cross_decomposition
filterwarnings('ignore')
reg = cross_decomposition.PLSCanonical(n_components=2, scale=False, algorithm='svd', max_iter=5000, tol=.1, copy=True)
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Gaussian NB**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
filterwarnings('ignore')
reg = GaussianNB(priors=None, var_smoothing=.1)
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Multinomial NB**

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
filterwarnings('ignore')
reg = MultinomialNB(alpha=.01, fit_prior=True, class_prior=None)
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Complement NB**

# In[ ]:


from sklearn.naive_bayes import ComplementNB
filterwarnings('ignore')
reg = ComplementNB()
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Bernoulli NB**

# In[ ]:


from sklearn.naive_bayes import BernoulliNB
filterwarnings('ignore')
reg = BernoulliNB()
reg.fit(X_train, y_train)
preds2 = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Decision Tree Regressor**

# In[ ]:


from sklearn import tree
clf = tree.DecisionTreeRegressor(max_depth=8)
clf = clf.fit(X_train, y_train)
preds2 = clf.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# **Gradient Boosting Regressor**

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor

reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
reg2 = RandomForestRegressor(random_state=1, n_estimators=100)
ereg = VotingRegressor(estimators=[('gb', xg_reg), ('rf', reg2)])
ereg = ereg.fit(X_train, y_train)
preds2 = ereg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# In[ ]:


trainall.shape


# **Below we have tested three feature selection techniques**

# In[ ]:


from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(.01)
temp = sel.fit_transform(trainall)
temp.shape


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
temp = SelectKBest(chi2, k=100).fit_transform(trainall, train.iloc[:,80])
temp.shape


# **Best**

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(trainall, train.iloc[:,80])
clf.feature_importances_  
model = SelectFromModel(clf, prefit=True)
temp = model.transform(trainall)
temp.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(trainall,train.iloc[:,80].to_numpy(), test_size=0.2, random_state=42)


# **MLP Regressor**

# In[ ]:


from sklearn import neural_network
reg = neural_network.MLPRegressor(hidden_layer_sizes=(100,100,100), activation='relu', solver='lbfgs', alpha=.001, 
                                          batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
                                          power_t=0.5, max_iter=600, shuffle=True)
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds)-np.log(y_test)))))


# **We implemented a gridsearch but decided to keep it commented out for right now**

# In[ ]:


# from sklearn.model_selection import GridSearchCV
# parameters = {'solver': ['lbfgs','adam'], 'random_state':[0,1]}
# clf_grid = GridSearchCV(neural_network.MLPRegressor(hidden_layer_sizes=(100,100,100), activation='relu', solver='lbfgs', alpha=.001, 
#                                           batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
#                                           power_t=0.5, max_iter=600, shuffle=True), parameters, n_jobs=-1)
# clf_grid.fit(X_train,y_train)
# print("-----------------Original Features--------------------")
# print("Best score: %0.4f" % clf_grid.best_score_)
# print("Using the following parameters:")
# print(clf_grid.best_params_)


# In[ ]:


# from sklearn.model_selection import GridSearchCV
# parameters = {'colsample_bytree': [.4,.5,.6,.7], 'learning_rate':[.1,.2,.3], 'n_estimators':[200,400,600],'reg_lambda':[.4,.5,.7]}
# clf_grid = GridSearchCV(xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .7, learning_rate = .1,
#          max_depth = 3, reg_lambda = .8, n_estimators = 500, evaluation_metric = 'rmsle'), parameters, n_jobs=-1)
# clf_grid.fit(X_train,y_train)
# print("-----------------Original Features--------------------")
# print("Best score: %0.4f" % clf_grid.best_score_)
# print("Using the following parameters:")
# print(clf_grid.best_params_)


# **I wanted to plot the predicted values to actual values for visualization purposes for three different models to get a sense of what is going on**
# **First two models are trained directly below (reg1, reg2), preds 3 is taken from the ridge regression model trained above**

# In[ ]:


xg_reg1 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .5, learning_rate = .1,
         max_depth = 3, reg_lambda = .8, n_estimators = 500, evaluation_metric = 'rmsle')
xg_reg1.fit(X_train, y_train)
preds1 = xg_reg1.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds1)-np.log(y_test)))))


# In[ ]:


xg_reg2 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .5, learning_rate = .1,
         max_depth = 3, reg_lambda = .6, n_estimators = 700, evaluation_metric = 'rmsle')
xg_reg2.fit(X_train, y_train)
preds2 = xg_reg1.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# In[ ]:


x = np.linspace(0,800000,100)


# In[ ]:


import matplotlib.pyplot as plt

plt.scatter(y_test,preds1)
plt.plot(x,x)


# In[ ]:


plt.scatter(y_test,preds2)
plt.plot(x,x)


# In[ ]:


plt.scatter(y_test,preds3)
plt.plot(x,x)


# A quick observation can be made that all three plots are very similar. The largest deviances come from data points from the tail end of the dependent variable. Hmmm, I wonder if we can do anything about that...

# In[ ]:


salesprice = train.iloc[:,80]


# In[ ]:


np.median(salesprice)


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=600)
clf = clf.fit(trainall, train.iloc[:,80])
clf.feature_importances_  
model = SelectFromModel(clf, prefit=True,threshold='median')
temp = model.transform(trainall)
temp.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(trainall,train.iloc[:,80].to_numpy(), test_size=0.2, random_state=42)


# In[ ]:


xg_reg1 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .7, learning_rate = .1,
         max_depth = 3, reg_lambda = .8, n_estimators = 500, evaluation_metric = 'rmsle')
xg_reg1.fit(X_train, y_train)
preds_test = xg_reg1.predict(X_test)
preds_train = xg_reg1.predict(X_train)
print(math.sqrt(np.mean(np.square(np.log(preds_test)-np.log(y_test)))))

a = np.zeros(80)
thresh = 210000
for i in range(0,80):
    print(i)
    thresh = 100000 + 2500*i
    trainall1 = X_train[preds_train < thresh]
    trainall2 = X_train[preds_train >= thresh]
    testall1 = X_test[preds_test < thresh]
    testall2 = X_test[preds_test >= thresh]
    y_train1 = y_train[preds_train < thresh]
    y_train2 = y_train[preds_train >= thresh]
    y_test1 = y_test[preds_test < thresh]
    y_test2 = y_test[preds_test >= thresh]

    xg_reg2 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .7, learning_rate = .1,
             max_depth = 3, reg_lambda = .8, n_estimators = 500, evaluation_metric = 'rmsle')
    xg_reg2.fit(trainall1, y_train1)
    preds1 = xg_reg2.predict(testall1)
    #print(math.sqrt(np.mean(np.square(np.log(preds1)-np.log(y_test1)))))

    xg_reg3 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .7, learning_rate = .1,
             max_depth = 3, reg_lambda = .8, n_estimators = 500, evaluation_metric = 'rmsle')
    xg_reg3.fit(trainall2, y_train2)
    preds2 = xg_reg3.predict(testall2)
    #print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test2)))))

    predsall = np.concatenate((preds1, preds2),axis=0)
    y_testall = np.concatenate((y_test1,y_test2),axis=0)
    print(math.sqrt(np.mean(np.square(np.log(predsall)-np.log(y_testall)))))
    a[i] = math.sqrt(np.mean(np.square(np.log(predsall)-np.log(y_testall))))


# In[ ]:


b = np.arange(1,81)
b
plt.plot(b,a)


# In[ ]:


b = np.arange(1,81)
b
plt.plot(b,a)


# In[ ]:


plt.scatter(predsall,y_test)
plt.plot(x,x)


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
import mlxtend
from sklearn import ensemble

xg_reg1 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .7, learning_rate = .1,
         max_depth = 3, reg_lambda = .8, n_estimators = 500, evaluation_metric = 'rmsle')
xg_reg1.fit(trainall, train.iloc[:,80].to_numpy())

xg_reg2 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .7, learning_rate = .1,
         max_depth = 3, reg_lambda = 1, n_estimators = 600, evaluation_metric = 'rmsle')
xg_reg2.fit(trainall, train.iloc[:,80].to_numpy())

xg_reg3 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .7, learning_rate = .1,
         max_depth = 3, reg_lambda = .9, n_estimators = 400, evaluation_metric = 'rmsle')
xg_reg3.fit(trainall, train.iloc[:,80].to_numpy())

xg_reg4 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .3, learning_rate = .1,
         max_depth = 3, reg_lambda = .5, n_estimators = 500, evaluation_metric = 'rmsle')
xg_reg4.fit(trainall, train.iloc[:,80].to_numpy())

xg_reg5 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .5, learning_rate = .1,
         max_depth = 3, reg_lambda = .3, n_estimators = 800, evaluation_metric = 'rmsle')
xg_reg5.fit(trainall, train.iloc[:,80].to_numpy())

ereg = ensemble.VotingRegressor(estimators=[('gb', xg_reg1), ('rf', xg_reg2), ('rf1', xg_reg3), ('rf2', xg_reg4), ('rf3', xg_reg5)])
ereg = ereg.fit(X_train, y_train)
preds2 = ereg.predict(X_test)
print(math.sqrt(np.mean(np.square(np.log(preds2)-np.log(y_test)))))


# In[ ]:


xg_reg1 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .7, learning_rate = .1,
         max_depth = 3, reg_lambda = .8, n_estimators = 500, evaluation_metric = 'rmsle')
xg_reg1.fit(trainall, train.iloc[:,80].to_numpy())

xg_reg2 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .7, learning_rate = .1,
         max_depth = 3, reg_lambda = 1, n_estimators = 600, evaluation_metric = 'rmsle')
xg_reg2.fit(trainall, train.iloc[:,80].to_numpy())

xg_reg3 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .7, learning_rate = .1,
         max_depth = 3, reg_lambda = .9, n_estimators = 800, evaluation_metric = 'rmsle')
xg_reg3.fit(trainall, train.iloc[:,80].to_numpy())

xg_reg4 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .3, learning_rate = .1,
         max_depth = 3, reg_lambda = .5, n_estimators = 500, evaluation_metric = 'rmsle')
xg_reg4.fit(trainall, train.iloc[:,80].to_numpy())

xg_reg5 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .5, learning_rate = .1,
         max_depth = 3, reg_lambda = .3, n_estimators = 800, evaluation_metric = 'rmsle')
xg_reg5.fit(trainall, train.iloc[:,80].to_numpy())

ereg = VotingRegressor(estimators=[('gb', xg_reg1), ('rf', xg_reg2), ('rf1', xg_reg3), ('rf2', xg_reg4), ('rf3', xg_reg5)])

ereg = ereg.fit(trainall, train.iloc[:,80].to_numpy())
preds = ereg.predict(testall)


# In[ ]:


# from sklearn.inspection import plot_partial_dependence
# ##plot_partial_dependence(xg_reg1, X_train[:,1].reshape(1,-1), y_train.reshape(1,-1)) 

# from sklearn.datasets import make_hastie_10_2
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.inspection import plot_partial_dependence

# X, y = make_hastie_10_2(random_state=0)
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,random_state=0).fit(trainall, train.iloc[:,80].to_numpy())
# features = [0, 1, (0, 1)]
# plot_partial_dependence(clf, trainall, features) 


# In[ ]:


y.shape


# In[ ]:


ss.iloc[:,1] = preds


# In[ ]:


ss.to_csv('ss.csv', index=False)

