'''
1) feature engineering:
1.1 PCA or not to compare
1.2 create more data points based on the original traing dataset
1.3 scaling method

2)algorithms:
2.1 NN
2.2 boosting model
2.3 svm model on multiple labels
2.4 random forest
2.5 logistic regression
2.6 ensembling
'''
import time
#track running time
start_time = time.time()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
pylab.ion()
import ggplot as glt
import seaborn as sbn
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
pd.set_option('display.max_rows',2000)




class ScalingByRange(BaseEstimator, TransformerMixin):
	'''
	#all columns are not normally distributed, but skewed, with most of 
	#columns are 0; so in terms of scaling, if we use mean0 and std1, it 
	#doesnot work well, since after this scaling, it is still not normal, 
	#and more importantly, we will still have lots of the 
	#effective columns that are in a very large range, this will bias our
	#further analysis; or, maybe we can take the log first, try to make it 
	#as the normal shape, and then normalize-the issue is first the logged 
	#shape is not normal shape either, and the 0 values donot have a log value,
	#we need to +1 for all values; use method2 of scaling, just divide each
	#value by its range, so the shape is not normal, but all values are between
	#0 and 1, which would reduce the key effect of large values on algorithms
	'''
	def __init__(self):
		pass
	def fit(self, X0, y=None, **fit_params):
		return self
	def transform(self, X0, y=None, **transform_params):
		return X0/255.0

#create new data points based on the training data
class CreateNewDataByTwist(BaseEstimator,TransformerMixin):
	'''
	skip by this version
	'''
	pass

# #PCA: this is self written pca, no additional information
# class PCATransform(BaseEstimator, TransformerMixin):
# 	def __init__(self, variance_explained):
# 		self.pca = PCA(n_components = variance_explained)
# 	def fit(self, X, y=None, **fit_params):
# 		self.pca.fit(X)
# 		return self
# 	def transform(self, X, y=None, **transform_params):
# 		return self.pca.transform(X)



#fix random number
np.random.seed(7)
##########################load the data######################
print('loading the data.......')
data = pd.read_csv('../input/train.csv') #reading data is so easy
y_data = data.label
select = pd.Series(np.random.random_sample(data.shape[0]), index = data.index)<=0.8
train = data[select] #this will generates a copy, not a view
y_train = train.label
valid = data[select.apply(lambda x: not x)] #this will generates a copy, not a view
y_valid = valid.label
data.drop('label',axis=1, inplace =True)
train.drop('label',axis=1, inplace =True)
valid.drop('label',axis=1, inplace =True)
test = pd.read_csv('../input/test.csv')



#*************** check the data's mean and std and distribution
# print data.describe()
# plt.hist(data.pixel334)


#*************** random forest model
# rfc_pip = Pipeline([('scaler',ScalingByRange()), 
# 					('pca',PCA()), 
# 					('rfc',RandomForestClassifier(oob_score=True))])

#***** for random forest, no need to do pca to get higher performance
rfc_pip = Pipeline([('scaler',ScalingByRange()), 
					# ('pca',PCA()), 
					('rfc',RandomForestClassifier(oob_score=True))])

# rfc_pip = Pipeline([#('scaler',ScalingByRange()), 
# 					('pca',PCA()), 
# 					('rfc',RandomForestClassifier(oob_score=True))])

# rfc_cv = True
# rfc_cv = False
rfc_cv = None

if rfc_cv is None:
	print('we skip the random forest model training at this time.......')

elif rfc_cv:
	#************ parameter grid setting: too long to run simply
	# pca__n_components = [0.8, 0.9, 0.95]
	# rfc__n_estimators = [25, 50, 100, 200]
	# # rfc__max_features = since given pca, the # of features will vary
	# rfc__max_depth = [5, 7, 10, 20]
	# rfc__min_samples_splits = [2, 5, 10, 25, 50]

	#************ test the parameter setting manually here
	# pca__n_components = [30, 37]
	rfc__n_estimators = [100, 150]
	# rfc__max_features = since given pca, the # of features will vary
	rfc__max_depth = [None]
	rfc__min_samples_split = [2]

	# param_grids = dict(pca__n_components = pca__n_components, rfc__n_estimators=rfc__n_estimators,
	# 	rfc__max_depth=rfc__max_depth,rfc__min_samples_split=rfc__min_samples_split)

	param_grids = dict(rfc__n_estimators=rfc__n_estimators,
		rfc__max_depth=rfc__max_depth,rfc__min_samples_split=rfc__min_samples_split)

	print('training the random forest model with the grid search and cross validation......')
	estimator = GridSearchCV(rfc_pip, param_grids, cv = 3, refit=True)
	# estimator = rfc_pip.set_params(pca__n_components =0.8,rfc__n_estimators=50 )
	estimator.fit(train, y_train)
	predict_train = estimator.predict(train)
	predict_valid = estimator.predict(valid)

	# ###checking results
	print('the combinations scores of each parameter setting:')
	for item in sorted(estimator.grid_scores_, key=lambda x: x[1]): print(item)
	print('the best parameter setting is:', estimator.best_estimator_)
	# print 'the number of components kept from the original is:', estimator.best_estimator_.named_steps['pca'].n_components_
	# print 'the variance kept is:', estimator.best_estimator_.named_steps['pca'].explained_variance_ratio_.sum()
	print('the best CV score of the GridSearchCV is:', estimator.best_score_)
	print('the best oob score of the best estimator of grid search results are:', estimator.best_estimator_.named_steps['rfc'].oob_score_)
	print('the in-bag prediction accuracy rate is:', (y_train == predict_train).sum()/float(y_train.shape[0]))
	print('the validation prediction accuracy rate is:', (y_valid == predict_valid).sum()/float(y_valid.shape[0]))
	# print 'the feature importances are:', sorted(list(zip(train.columns, estimator.best_estimator_.feature_importances_)), key = lambda x: x[1], reverse=True)
	print('the running time is:', time.time() -start_time)

else:
	#************ combine train and valid dataset to train the model again and then predict the test dataset
	print('training the model use all the data, not only train dataset.........')
	# estimator = rfc_pip.set_params(pca__n_components = 35, rfc__n_estimators= 150, rfc__max_depth = None, rfc__min_samples_split = 2)
	#*** if no pca upfront
	estimator = rfc_pip.set_params(rfc__n_estimators= 100, rfc__max_depth = None, rfc__min_samples_split = 2)
	estimator.fit(data, y_data)
	predict_data= estimator.predict(data)
	predict_test = estimator.predict(test)
	#*** checking results
	print('the in-bag prediction accuracy rate is:', (y_data == predict_data).sum()/float(y_data.shape[0]))
	print('the oob score of the estimator is:', estimator.named_steps['rfc'].oob_score_)
	#*** output the result
	print('save the result in test_result.csv file.......')
	output = pd.read_csv('sample_submission.csv')
	output.loc[:,'Label'] = predict_test
	output.to_csv('test_result.csv', index=False)


#*************** svm model
#different types of pipline, with pca or not
svm_pip = Pipeline([('scaler',ScalingByRange()), 
					('pca',PCA()), 
					('svm',SVC())])

# svm_pip = Pipeline([('scaler',ScalingByRange()), 
# 					# ('pca',PCA()), 
# 					('svm',SVC())])

svm_cv = True
# svm_cv = False
# svm_cv = None

if svm_cv is None:
	print('we skip the SVM model training at this time.......')

elif svm_cv:
	#************ parameter grid setting: too long to run simply
	# pca__n_components = [0.8, 0.9, 0.95]
	# svm__C = [0.001, 0.003, 0.1, 0.3, 0.6, 1.0, 3.0, 10, 20]
	# svm__gamma = [0.001, 0.003, 0.1, 0.3, 0.6, 1.0, 3.0, 10, 20]
	# svm__kernel = ['sigmoid', 'rbf','linear']	

	#************ test the parameter setting manually here
	pca__n_components = [0.9]
	svm__C = [0.1]
	svm__gamma = [0.9]
	svm__kernel = ['linear']

	param_grids = dict(pca__n_components=pca__n_components, svm__C=svm__C, 
		svm__gamma=svm__gamma,svm__kernel=svm__kernel)
	# no pca parameters grid:
	# param_grids = dict(svm__C=svm__C, 
	# 	svm__gamma=svm__gamma,svm__kernel=svm__kernel)

	print('training the SVM model with the grid search and cross validation......')
	estimator = GridSearchCV(svm_pip, param_grids, cv = 3, refit=True)
	estimator.fit(train, y_train)
	predict_train = estimator.predict(train)
	predict_valid = estimator.predict(valid)

	# ###checking results
	print('the combinations scores of each parameter setting:')
	for item in sorted(estimator.grid_scores_, key=lambda x: x[1]): print(item)
	print('the best parameter setting is:', estimator.best_estimator_)
	print('the number of components kept from the original is:', estimator.best_estimator_.named_steps['pca'].n_components_)
	print('the variance kept is:', estimator.best_estimator_.named_steps['pca'].explained_variance_ratio_.sum())
	print('the best CV score of the GridSearchCV is:', estimator.best_score_)
	print('the in-bag prediction accuracy rate is:', (y_train == predict_train).sum()/float(y_train.shape[0]))
	print('the validation prediction accuracy rate is:', (y_valid == predict_valid).sum()/float(y_valid.shape[0]))
	print('the running time is:', time.time() -start_time)

else:
	#************ combine train and valid dataset to train the model again and then predict the test dataset
	print('training the model use all the data, not only train dataset.........')
	#*** if there is PCA in the pipeline:
	estimator = svm_pip.set_params(pca__n_components = 35, svm__C=1.0, svm__kernel = 'rbf', svm__gamma = 0.9)
	#*** if no pca in the pipeline:
	# estimator = svm_pip.set_params(svm__C=1.0, svm__kernel = 'rbf', svm__gamma = 0.9)
	estimator.fit(data, y_data)
	predict_data= estimator.predict(data)
	predict_test = estimator.predict(test)
	#*** checking results
	print('the in-bag prediction accuracy rate is:', (y_data == predict_data).sum()/float(y_data.shape[0]))
	#*** output the result
	print('save the result in test_result.csv file.......')
# 	output = pd.read_csv('sample_submission.csv')
# 	output.loc[:,'Label'] = predict_test
# 	output.to_csv('test_result.csv', index=False)









