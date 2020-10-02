#!/bin/python3

import numpy as np
import math
import sys
import pandas as pd
import xgboost as xgb
import random
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.decomposition import NMF
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from sklearn.svm import SVR

from sklearn.decomposition import NMF
from sklearn.feature_selection import SelectKBest,f_regression

from sklearn.preprocessing import PolynomialFeatures
from copy import copy
import lightgbm as lgb



#########
outputName = 'submission'
#########

#### HIGH LEVEL HYPER-PARAMETERS ####
#####################################
LOG_SKEWED = True
NORMALIZE = True
ORDINALIZE_DATA = False
DEORDINALIZE_DATA = True
FEATURE_CREATION = True
REMOVE_OUTLIERS = True
FEATURE_SELECTION = True
UNNI_FEATURE_CREATION = False
BIN_ATTRS = False

USE_XGB = False
USE_LASSO = True
USE_RIDGE = True
USE_RF = False
USE_MLP = False
#####################################



lassoParams = {'alpha': np.arange(.0001, .001, .0001), 'normalize':[True,False], 'fit_intercept':[False,True]}#, 'copy_X':True, 'fit_intercept':False, 'max_iter':1000,
   #'normalize':True, 'positive':False, 'precompute':False, 'random_state':None,
   #'selection':'cyclic', 'tol':0.0001, 'warm_start':False}

# bestLasso = {'alpha': 0.0001, 'normalize': False, 'fit_intercept': True}
lassoAcc =  0.0148827145999
bestLasso = {'alpha': 0.00050000000000000001, 'normalize': False, 'fit_intercept': True}


ridgeParams = {'alpha':[0.0001, .001, .01, .1, 10], 'normalize':[True,False], 'fit_intercept':[False,True]}#, 'copy_X':True, 'fit_intercept':False, 'max_iter':1000,
ridgeAcc = 0.0206608045187
# bestRidge = {'alpha': 10, 'fit_intercept': True, 'normalize': False}
bestRidge = {'alpha': 10 } #, 'fit_intercept': True, 'normalize': True}

#mae gives seg fault :(
randomForestParams = {'n_estimators':[200], 'criterion':['mse'],
	'max_depth':[None,5,10,20,40], 'max_features':[.2,.4,.6,.8,'auto']}
# min_samples_split=2, min_samples_leaf=1, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False
# min_weight_fraction_leaf=0.0,  max_leaf_nodes=None, min_impurity_split=1e-07,
# bestRandomForest = {'criterion': 'mse', 'max_depth': None, 'max_features': 0.2, 'n_estimators': 200}
bestRandomForest = {'max_depth': 40, 'criterion': 'mse', 'n_estimators': 200, 'max_features': 0.4}
randomForestAcc = 0.0196056881813


extraTreesParams = {'n_estimators':[200], 'criterion':['mse'],
	'max_depth':[None,5,10,20,40], 'max_features':[.2,.4,.6,.8,'auto']}
# bestExtraTrees = {'criterion': 'mse', 'n_estimators': 200, 'max_features': 0.2, 'max_depth': None}
bestExtraTrees  ={'max_depth': None, 'criterion': 'mse', 'n_estimators': 200, 'max_features': 0.6}
extraTreesAcc = 0.0210039305211


adaBoostParams = { 'n_estimators':[100], 'learning_rate':[.0001,.001,.01,.1,1.0,10.0], 'loss':['linear','square'] }
adaBoostAcc = 0.0384596265692
bestAdaBoost = {'learning_rate': 1.0, 'loss': 'linear', 'n_estimators': 100}


xgBoostParams = {'learning_rate': [.0001, .0005, .001, .005, .025, .1, .5], 'silent': [1], 'max_depth':[3,5,7,10],
		'min_child_weight':[2,4,6,8], 'gamma':[0.0,.1,.2,.3,.4,.5],
	'subsample':[.1, .2, .3, .4, .5, .6,.7,.8,.9], 'colsample_bytree':[.1, .2, .3, .4, .5, .6,.7,.8,.9]}

xgBoostAcc = 0.0157843952784
# bestXGBoost = {'max_depth': 3, 'min_child_weight': 2, 'gamma': 0.1, 'subsample': 0.9, 'learning_rate': 0.1, 'colsample_bytree': 0.7, 'silent': 1}
bestXGBoost = {'colsample_bytree': 0.3, 'max_depth': 7, 'gamma': 0.0, 'silent': 1, 'min_child_weight': 2, 'subsample': 0.5, 'learning_rate': 0.1}

supportVectorParams = {'kernel':['poly'], 'degree':[2,3,5], 'gamma':[.1, .3, .5, .7,'auto'], 'coef0':[0.0, .3, .5 , .7,1.0], 'C':[.0001, .01, .1, 1.0, 10], 'max_iter':[1000]}

supportVectorAcc = 0.169503583615
bestSupportVector = {'coef0': 0.3, 'C': 10, 'kernel': 'rbf', 'max_iter': 5000, 'gamma': 'auto', 'degree': 5}


neuralNetworkParams = {'hidden_layer_sizes' : [(100 ), (50), (10), (1)], 'activation':['relu', 'logistic'],
	'solver':['adam', 'lbfgs', 'sgd'], 'alpha':[0.0001, .001, .01, .1, 1, 10], 'batch_size':[50, 100,'auto'],
	'learning_rate':['constant', 'inv_scaling'], 'learning_rate_init':[0.001,10,1],
	'momentum':[.95,0.9,.8,.7]}

lgbParams = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l1', 'auc'},
    # 'metric': 'l1',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': 0
}



xgBoost2Acc = 0.0184334515823
best2XGBoost = {'silent': 1, 'learning_rate': 0.1, 'subsample': 0.6, 'max_depth': 10, 'min_child_weight': 2, 'colsample_bytree': 0.7, 'gamma': 0.1}



allModels = [
#			{'name' : 'SVR', 'model': SVR, 'acc' : supportVectorAcc, 'params' , bestSupportVector } ,
			# {'name' : 'ADABoost', 'model': AdaBoostRegressor, 'acc' : adaBoostAcc, 'params' : bestAdaBoost } ,
			# {'name' : 'ExtraTrees', 'model': ExtraTreesRegressor, 'acc' : extraTreesAcc, 'params' : bestExtraTrees } ,
				]


if USE_XGB:
	allModels.append(			{'name' : 'XGBoost', 'model': xgb, 'acc' : xgBoostAcc, 'params' : bestXGBoost })
if USE_LASSO:
	allModels.append(			{'name' : 'Lasso', 'model': Lasso, 'acc' : lassoAcc, 'params' : bestLasso } )
if USE_RIDGE:
	allModels.append(			{'name' : 'Ridge', 'model': Ridge, 'acc' : ridgeAcc, 'params' : bestRidge } )
if USE_RF:
	allModels.append( 			{'name' : 'RandomForest', 'model': RandomForestRegressor, 'acc' : randomForestAcc, 'params' : bestRandomForest } )

# prepocessing steps:
# log(+1) the results and all skewed attributes
# normalize all numerical attributes
# get dummies
# fill data w/ mean and mode
# create kfold cross val

# training steps:
# train on multiple different classifiers:
# LinReg, LASSO, RIDGE, RandFor, ExRandFor, Adaboost, BayesReg, ElasticNet


# use training results on stacked classifier
# (xgboost,elasticnet, NeuralNetwork)


# trainfilename = sys.argv[1]
# testfilename = sys.argv[2]

NO_FS = False
NUM_TREES = 1000


def preprocess(data):


	data = handmadeTweaks(data)




	#Fill NA values with means and modes
	# print(data['MiscFeature'])

	#log transform skewed attrs (already done for price)
	if LOG_SKEWED:
		data = logskewed(data)

	data = fillNas(data, data)


	if BIN_ATTRS:
		data = binAttrs(data)


	if NORMALIZE:
		data = normalize(data)

	data = pd.get_dummies(data)
	# data = completeMatrix(data)


	return data

def binAttrs(data):
	#bin the year attributes - In theory perhaps very old homes are more expensive, or there is a nonlinear relationship
	#on second thought this isnt working so well
	binned = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
	for yrcol in binned:
		print(data[yrcol])
		data[yrcol] = pd.cut(np.array(data[yrcol]) , [1000, 1930, 1960, 1980, 1992, 1998, 2001, 2003, 2005, 2007, 2008, 2009, 2010])
		data[yrcol] = data[yrcol].astype(object)
		print(data[yrcol])

	binned = ['YrSold']
	for yrcol in binned:
		print(data[yrcol])
		data[yrcol] = pd.cut(np.array(data[yrcol]) , [1000, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010])
		data[yrcol] = data[yrcol].astype(object)
		print(data[yrcol])


	return data

def handmadeTweaks(data):
	#If we dont do this nas will be filled with mode which is potentially undesirable

	data = data.fillna({
	'MiscFeature' : 'None',
	'Fence' : 'None',
	'PoolQC' : 'None',
	'FireplaceQu': 'None',
	'GarageCond': 'None',
	'GarageQual' : 'None',
	'GarageFinish' : 'None',
	'GarageType' : 'None',
	'BsmtQual' : 'None',
	'BsmtCond' : 'None',
	'BsmtExposure' : 'None',
	'BsmtFinType1' : 'None',
	'BsmtFinType2' : 'None',
	'Alley' : 'None',
	'KitchenQual' : 'None' ,
	'Functional' : 'None'
	})

	#hot garbage
	data = data.drop('Utilities')

	#https://www.trulia.com/blog/trends/springtime-for-housing/
	data['PeakMonths'] = data['MoSold'].map({1 : 0, 2: 0, 3: 2, 4 : 2, 5 : 4, 6 : 5,
									7 : 4, 8 : 4, 9 : 3, 10 : 3, 11 : 2, 12 : 2})



	#This should only make sense--in no ways are these interval/ordinal quantities
	data['MoSold'] = data['MoSold'].apply(str)
	data['MSSubClass'] = data['MSSubClass'].apply(str)

	if DEORDINALIZE_DATA:
		data = deordinalize(data)

	if ORDINALIZE_DATA:
		data = ordinalize(data)


	#combine sq footage
	if FEATURE_CREATION:
		intr = ['TotalBsmtSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF' , 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
		extr = ['LotArea']
		data['TotalSqFT'] = data[intr].sum(axis=1)

		data['hasRemodel'] = 1.0 * (data['YearRemodAdd'] != data['YearBuilt'])
		data['soldEqBuilt'] = 1.0 * (data['YearBuilt'] == data['YrSold'] )



		data['timeSinceRemodel'] = data['YrSold'] - data['YearRemodAdd']
		data['timeRemodelAfterBuilt'] = data['YearBuilt'] - data['YearRemodAdd']
		data['Age'] = 2010 - data['YearBuilt']
		data['SoldAgo'] = 2010 - data['YrSold']
		data['RemAgo'] = data['YrSold'] - data['YearRemodAdd']
	return data

def deordinalize(data):
	#Can exploit nonlinearities in ordinal attrs
	data['FullBath'] = data['FullBath'].apply(str)
	data['HalfBath'] = data['HalfBath'].apply(str)
	data['BedroomAbvGr'] = data['BedroomAbvGr'].apply(str)
	data['BsmtFullBath'] = data['BsmtFullBath'].apply(str)
	data['BsmtHalfBath'] = data['BsmtHalfBath'].apply(str)
	data['KitchenAbvGr'] = data['KitchenAbvGr'].apply(str)
	data['Fireplaces'] = data['Fireplaces'].apply(str)
	data['GarageCars'] = data['GarageCars'].apply(str)
	# data['OverallQual'] = data['OverallQual'].apply(str)
	# data['OverallCond'] = data['OverallCond'].apply(str)
	return data


def ordinalize(data):
	#Map features to ordinals to reduce dimensionality:
	data["LotShape"] = data["LotShape"].map({'Reg' : 0, 'IR1' : 1, 'IR2' : 2, 'IR3' : 3}).astype(int)

	# data["MSZoning"] = data["MSZoning"].map({'RH' : 0, 'RM' : 1, 'RL' : 2}).astype(int)

	data['LandSlope'] = data['LandSlope'].map({'Gtl' : 0, 'Mod' : 1, 'Sev' : 2}).astype(int)


	std = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
	for col in std:
		data[col] = data[col].map({'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5}).astype(int)

	data['BsmtExposure'] = data['BsmtExposure'].map({'None' : 0, 'No' : 1 , 'Mn' : 3, 'Av' : 3, 'Gd' : 4}).astype(int)

	data['BsmtFinType1'] = data['BsmtFinType1'].map({'None' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6}).astype(int)

	data['BsmtFinType2'] = data['BsmtFinType2'].map({'None' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6}).astype(int)


	data['Functional'] = data['Functional'].map({'None' : -1, 'Sal' : 0, 'Sev' : 1, 'Maj2' : 2, 'Maj1' : 3, 'Mod' : 4, 'Min2' : 5, 'Min1' : 6, 'Typ' : 7 } ).astype(int)



	data['CentralAir'] = data['CentralAir'].map({'N' : 0, 'Y' : 1}).astype(int)


	data['GarageFinish'] = data['GarageFinish'].map({'None' : 0, 'Unf' : 1, 'RFn' : 2, 'Fin' : 3}).astype(int)


	data['Fence'] = data['Fence'].map({'None' : 0, 'MnWw' : 1, 'GdWo' : 2, 'MnPrv' : 3 , 'GdPrv' : 4}).astype(int)
	return data

def completeMatrix(data):
	data = data.fillna(value=0)
	nmf = NMF()
	W = nmf.fit_transform(data);
	H = nmf.components_;
	comp = np.dot(W,H)
	return comp


def logskewed(data):
	skew = data.skew(numeric_only=True)

	#these 2 are just Binary variables
	dontSkew = ['BsmtHalfBath','KitchenAbvGr']

	for colName in skew.index:
		if skew[colName] > .8 and colName not in dontSkew:
			data[colName] = np.log1p(data[colName])

	return data

def normalize(data):
	toScale = data.dtypes[data.dtypes != "object"].index
	scaler = StandardScaler()
	scaler.fit(data[toScale])
	normalized = scaler.transform(data[toScale])
	for index, col in enumerate(toScale):
		data[col] = normalized[:, index]
	return data


def fillNas(data, meanData):
	for colName in data.columns.values:
		if data[colName].dtype == 'object': # or data[colName].dtype == 'category':
			mode = meanData[colName].mode().iloc[0]
			if mode == 'nan' or mode == 'NaN':
				mode = mean[colName].mode().iloc[1]
			data[colName] = data[colName].fillna(value = mode)
		else:
			# print(data[colName])
			data[colName] = data[colName].fillna(value = meanData[colName].mean())
	return data

def selectFeatures(data, testdata, target):
	kf = KFold(n_splits = 5, shuffle=True)
	fc = PolynomialFeatures()
	data = fc.fit_transform(data)
	print(type(data))
	names = fc.get_feature_names()
	KBest = SelectKBest(f_regression,k=5000)
	KBest.fit(data,target)
	indices = KBest.get_support()
	data = data[:,indices]
	testdata = fc.transform(testdata)
	testdata = testdata[:,indices]
	return data, testdata



def getAccuracy(pred, actual):
	return mean_squared_error((actual), (pred)) #already logged

def runModel(Model,  kf, data, target, parameters, stochastic=-1, silent=True):

	paramsets = list(ParameterGrid(parameters))


	lowestErr = 1
	bestPset = {}

	count = 0
	for pset in paramsets:
		if stochastic > 0:
			if count >= stochastic:
				break
			pset = random.choice(paramsets)
		count+=1

		avg = 0

		print(pset)
		model = Model(**pset)

		for train, test in kf.split(data):
			traindata = np.take(data, train, axis=0)
			traintarget = np.take(target, train, axis=0)
			testdata = np.take(data, test, axis=0)
			testtarget = np.take(target, test, axis=0)

			model.fit(traindata, traintarget)
			pred = model.predict(testdata)
			accuracy = getAccuracy(pred, testtarget)
			avg += accuracy
			if not silent:
				print('MSE:', accuracy)


		avg /= kf.get_n_splits()

		if not silent:
			print('avg', avg)

		if avg < lowestErr:
			lowestErr = avg
			bestPset = pset

		del model

	return lowestErr, bestPset


def runXGBoost(kf, data, target, parameters, numtrials):
	paramsets = list(ParameterGrid(parameters))
	avg = 0

	lowestErr = 1
	bestPset = {}

	count = 0
	NUM_TRIALS = numtrials
	silent = True

	NUM_XGB = 500

	for i in range(0,NUM_TRIALS):
		pset = random.choice(paramsets)
		count += 1
		if not silent:
			print('Testing Model ', count , '/', len(paramsets))
			print(pset)
		avg = 0

		splitCount = 0
		for train, test in kf.split(data):
			traindata = np.take(data, train, axis=0)
			traintarget = np.take(target, train, axis=0)
			train = xgb.DMatrix(traindata, label=traintarget)
			model = xgb.train(pset, train, NUM_XGB)


			testdata = np.take(data, test, axis=0)
			testtarget = np.take(target, test, axis=0)
			test = xgb.DMatrix(testdata)

			pred = model.predict(test)
			# print(pred, testtarget)

			accuracy = getAccuracy(pred, testtarget)
			avg += accuracy
			if not silent:
				print('MSE:', accuracy)
			splitCount += 1

			del model

		avg /= splitCount

		if not silent:
			print('avg', avg)

		if avg < lowestErr:
			lowestErr = avg
			bestPset = pset

	return lowestErr, bestPset




def predModel(Model, data, target, test, parameters):

	print(Model , parameters)
	model = Model(**parameters)
	model.fit(data, target)


	ret = {}
	for key, value in test.items():

		pred = model.predict(value)
		ret[key] = pred


	return ret

def predXGBoost(data, target, test, parameters):


	train = xgb.DMatrix(data, label=target)
	model = xgb.train(parameters, train, 200)

	ret = {}

	for key, value in test.items():
		value = xgb.DMatrix(value)

		pred = model.predict(value)
		ret[key] = pred

	return ret



def outputPredictions(name, predictions, ids):

	with open(name + '.csv', 'w') as f:
		f.write('Id,SalePrice\n')
		for i in range(0, len(predictions)):
			f.write(str(ids[i]) + ',' + str(np.expm1(predictions[i])) + '\n')

def parameterList(parameters):
	p = parameters.copy()
	for key, value in p.items():
		p[key] = [value]
	return p


def arithMean(preds):
	# print([preds[key]['testdata'] for key in preds])
	s = np.sum([preds[key]['testdata'] for key in preds], axis=0)
	s = np.divide(s, len(preds))
	return s


def selectInliers(data, target):
	from sklearn.ensemble import IsolationForest
	from sklearn.feature_selection import SelectKBest,f_regression

	params = {'contamination' : .11}
	isoforest = IsolationForest(**params)
	isoforest.fit(data)
	is_inlier = isoforest.predict(data)
	inliers = []
	for i in range(len(is_inlier)):
	    if is_inlier[i] == 1:
	        inliers.append(i)

	data=data[inliers,:]
	target=target[inliers]
	return data, target

data = pd.read_csv("../input/train.csv")
data.drop(data[data["GrLivArea"] > 4000].index, inplace=True)
trainLength = data.shape[0]
# data = fillNas(data, data);

testdata = pd.read_csv("../input/test.csv")
ids = testdata.iloc[:,0]
testLength = data.shape[0]
# testdata = fillNas(testdata, data)

target = np.log1p(data.iloc[:,-1]) # Log transform the house prices since they are skewed
data = data.iloc[:,1:-1] #drop ids and target

ccdata = pd.concat([data,testdata])


#prepocess data as a whole for means and stuff
ccdata = preprocess(ccdata)

# ccdata2 = ccdata.copy(deep =True)

#split it back up
data = ccdata[0:trainLength]

testdata = ccdata[trainLength:]

data=np.array(data)



testdata=np.array(testdata)
target=np.array(target)





# KBest = SelectKBest(f_regression,k=240)
# KBest.fit(data,target)
# indices = KBest.get_support() #the best features
# data = data[:,indices]
# testdata = testdata[:,indices]

if REMOVE_OUTLIERS:
	data, target = selectInliers(data, target)

if UNNI_FEATURE_CREATION:
	data,testdata = selectFeatures(data, testdata, target)


# ''''
# outliers
# '''
# maxi = -1000;
# summ = 0
# count = 0
# train = xgb.DMatrix(data, label=target)
# model = xgb.train(best2XGBoost, train, 200)
#
# value = xgb.DMatrix(data)
# pred = model.predict(value)
# print("size:", str(len(pred)))
# for i in reversed(range(len(pred))):
#     dif = pred[i] - target[i]
#     summ = summ + abs(dif)
#     maxi = max(maxi, abs(dif))
#     if(abs(dif) > 0.11):
#         count = count + 1
#         np.delete(data, i)
#         #target.remove(i)
#         np.delete(target, i)
# print("max: ", str(maxi))
# print("avg: ", str(summ/len(pred)))
# print("count: ", str(count))
# ''''
# outliers
# '''
#
#
# KBest = SelectKBest(f_regression,k=240)
# KBest.fit(data,target)
# indices = KBest.get_support()
# data = copy(data[:,indices])
# testdata = copy(testdata[:,indices])


#blending holdout set

# data, level2, target, level2Target = train_test_split(data, target, test_size = .2)

# -> randomize--does the data have some bias in terms of order???
# mark = int(.7*data.shape[0])
# level2 = data[mark:,:]
# data = data[:mark,:]
# level2Target = target[mark:]
# target = target[:mark]



#this will allow us to combine sklearn with non-sklearn models like xgboost
kf = KFold(n_splits = 5, shuffle=True)


# err, params1 = runModel(SVR ,kf, data, target, supportVectorParams, 100)

# err, params2 = runModel(Lasso ,kf, data, target, lassoParams, 100)
# err, params3 = runModel(Ridge ,kf, data, target, ridgeParams, 100, silent=False)
# print(err, params3)

# err, params4 = runModel(RandomForestRegressor ,kf, data, target, randomForestParams, 100)
#
# err, params5 = runModel(ExtraTreesRegressor ,kf, data, target, extraTreesParams, 100)
# err, params = runXGBoost(kf, data, target, xgBoostParams, 400)
# print( params)


# preds = predModel(SVR , data, target, testdata, bestSupportVector)
# preds = predXGBoost(data, target, testdata, bestXGBoost)


# print(preds)
# print(level2.shape, level2Target.shape, data.shape, target.shape)


# outputPredictions('XGBoost', preds, ids)
preds = {}

for m in allModels:
	if m['model'] == xgb:
		err, p = runXGBoost(kf, data, target, parameterList(m['params']), 1)
		print(m['name'], err)
		preds[m['name']] = predXGBoost(data, target, {'data':data, 'testdata':testdata}, m['params'])
	else:
		err, p = runModel(m['model'], kf, data, target, parameterList( m['params']), 1)
		print(m['name'], err)
		preds[m['name']] = predModel(m['model'] , data, target, {'data':data, 'testdata':testdata}, m['params'])

# for key, value in preds.items():
# 	value['data'] = np.transpose(np.array([value['data']]))
# 	value['testdata'] = np.transpose(np.array([value['testdata']]))
# 	value['level2'] = np.transpose(np.array([value['level2']]))
#
# 	data = np.append(data, value['data'], axis=1)
# 	testdata = np.append(testdata, value['testdata'], axis=1)
# 	level2 = np.append(level2, value['level2'], axis=1)

preds = arithMean(preds)

# print(preds)

lgb_train = lgb.Dataset(data, target)

gbm = lgb.train(lgbParams,
                lgb_train,
                num_boost_round=1000)
                #eval_set=[(X_test, y_test)])
                # early_stopping_rounds=6000)
                
y_pred2 = gbm.predict(testdata, num_iteration=gbm.best_iteration)
x = np.sum(preds - y_pred2)
print(x)
# err, pset = runXGBoost(kf,level2,level2Target,xgBoostParams)


# err, pset = runModel(MLPRegressor, kf, data, target, neuralNetworkParams)

# print(err, pset)

# preds = predXGBoost(level2, level2Target, {'testdata':testdata}, best2XGBoost)

outputPredictions(outputName, preds, ids)


# bestLinErr, alpha = test_alphas(data, target)

# bestErr, selected_feats, bad_feats = feat_select(data,target)


# if NO_FS:
# 	selected_feats = []
# 	for feat in data.columns.values:
# 		selected_feats.append(feat)

# ids = testdata.iloc[:,0]

# testdata = testdata.iloc[:,1:]

# # rf = Lasso(normalize = True)
# # rf.fit(lassoData,target2)
# # lassoPred = rf.predict(lassoTest)

# rftestdata = testdata[selected_feats]
# rfdata = data[selected_feats]

# cc = pd.concat([rfdata,rftestdata])
# cc = pd.get_dummies(cc)
# rfdata = cc.iloc[:1460,:]
# rftestdata = cc.iloc[1460:,:]

# rfdata = rfdata.as_matrix()
# ids = ids.as_matrix()

# rftestdata = rftestdata.as_matrix()

# dd = pd.concat([data,testdata])
# dd = pd.get_dummies(dd)
# lassodata = dd.iloc[:1460,:]
# lassotestdata = dd.iloc[1460:,:]

# lassodata = lassodata.as_matrix()
# lassotestdata = lassotestdata.as_matrix()

# rf = Lasso(alpha=.005,normalize = True, max_iter = 100000, warm_start=True)
# rf.fit(lassodata,target)
# lassoPred = np.expm1(rf.predict(lassotestdata))


# rf2 = RandomForestRegressor(n_estimators = 100)
# rf2.fit(rfdata, target)

# predictTest = np.expm1(rf2.predict(rftestdata))

# print(bestErr, bestLinErr)
# lassRat = 1-bestLinErr/(bestErr+bestLinErr)
# rfRat = 1 - lassRat
# print(rfRat, lassRat)

# lassRat = 0
# rfRat = 1

# with open('verybigrf.txt', 'w') as f:
# 	f.write('Id,SalePrice\n')
# 	count = 0
# 	for i in ids:
# 		pp = predictTest[count]*rfRat + lassoPred[count]*lassRat
# 		f.write(str(i) + ',' + str(predictTest[count]) + '\n')
# 		count+=1







# def createDevSet(data, target):
# 	tuneMark = int(data.shape[0]*.8)
# 	tuneData = data.iloc[tuneMark:,:]
# 	data = data.iloc[:tuneMark,:]

# 	tuneTarget = target.iloc[tuneMark:]
# 	target = target.iloc[:tuneMark]

# 	return data, target, tuneData, tuneTarget

# def test_tree(data, target):
# 	data = pd.get_dummies(data)

# 	data, target, tuneData, tuneTarget = createDevSet(data, target)

# 	data = data.as_matrix()
# 	target = target.as_matrix()
# 	tuneTarget = tuneTarget.as_matrix()
# 	tuneData = tuneData.as_matrix()

# 	rf = RandomForestRegressor(n_estimators=NUM_TREES)
# 	rf.fit(data,target)
# 	predict = np.expm1(rf.predict(tuneData))

# 	return mean_squared_error(np.log(np.expm1(tuneTarget)), np.log(predict))

# def test_linear(data, target, a):
# 	data = pd.get_dummies(data)

# 	data, target, tuneData, tuneTarget = createDevSet(data, target)

# 	data = data.as_matrix()
# 	target = target.as_matrix()
# 	tuneTarget = tuneTarget.as_matrix()
# 	tuneData = tuneData.as_matrix()

# 	rf = Lasso(normalize = True, max_iter = 10000, warm_start = True, alpha=a)
# 	rf.fit(data,target)
# 	predict = np.expm1(rf.predict(tuneData))

# 	return mean_squared_error(np.log(np.expm1(tuneTarget)), np.log(predict))


# def feat_select(data, target):
# 	#initialize err to max
# 	bestErr = 1.0
# 	selected_columns = []
# 	bad_feats = []
# 	for featName in data.columns.values:
# 		selected_columns.append(featName)
# 		data_select = data[selected_columns]

# 		print(selected_columns)
# 		err = test_tree(data_select, target)
# 		print(err)

# 		if err > bestErr:
# 			#shitty feature--dont use it
# 			selected_columns.remove(featName)
# 			bad_feats.append(featName)
# 		else:
# 			bestErr = err

# 	return bestErr, selected_columns, bad_feats

# def test_alphas(data, target):
# 	bestErr = 1.0
# 	alpha = .001
# 	for i in range(0, 10):

# 		print(alpha)
# 		err = test_linear(data, target, alpha)
# 		print(err)
# 		alpha*=5
# 		if err < bestErr:
# 			bestErr = err

# 	return bestErr, alpha
