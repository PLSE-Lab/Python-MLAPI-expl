#!/usr/bin/env python
# coding: utf-8

# # Optimizing the prediction of NHL player salary:
# ## Blending machine learning algorithms to reduce predction error.
#    In previous explorations of this dataset, I have employed extreme gradient boosting and random forest regression to try to make accurate predictions of NHL player salary. These models have proven interesting, showing that player's draft year along with numerous advanced stats are some of the best predictors of the amount of money they make. The default XGBoost model had a fairly high root mean square error though: \$1,574,073. This is a definite improvement over just guessing the median salary, which provides an rmse of \$2,878,624. However, here I am looking to improve this error and create a more accurate predictor of player salary. To do this, I will be attempting to train several optimized models (via cross validation of hyperparamaters) and then blend the results to create a more accurate predition. I have chosen to proceed with the two previously employed models, Random Forest and XGBoost, and also add in a support vector machine to see if I can reach a blended and optimized ensemble model.
# 
# As a tangental goal, I am also looking to streamline the data munging process using some of the built in functions of Scikit Learn to clean the data and make dummy variables where needed.

# First up, a big old pile of imports! This is definitely the downside to using the SciKit built-ins, you have to remember the imports for all the functions you need!

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion
from datetime import datetime
import gc
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


# ## Load the data

# In[ ]:


train = pd.read_csv('../input/train.csv', encoding = "ISO-8859-1")
test_x = pd.read_csv('../input/test.csv', encoding = "ISO-8859-1")
test_y = pd.read_csv('../input/test_salaries.csv') 


# set up the train and test variables

# In[ ]:



test_y=list(test_y['Salary'].values)

train_x = train.drop('Salary',axis=1)
train_y = list(train['Salary'])

train=[]
gc.collect()


# ## clean the data
# 
# Based on my previous experience with the dataset, I've seen that the team of the player, along with the country of origin are poor predictors of salary, so I am electing to remove these outright. Below I use the python datetime functions to take the date of birth column, and turn it into the age (in days) at season's start.

# In[ ]:



#Born - datetime needs to be changed to days since a set date
#days form birth to season start

def elapsed_days(start, end=datetime(2016,10,12)):
	""" calcualte the number of days start and end dates"""
	x = (end - start)
	return x.days

#
train_x['age_season_start'] = train_x.apply(lambda x: 
	elapsed_days(datetime.strptime(x['Born'], '%y-%m-%d')) ,axis=1)

test_x['age_season_start'] = test_x.apply(lambda x: 
	elapsed_days(datetime.strptime(x['Born'], '%y-%m-%d')) ,axis=1)


# With the age columns altered, we can drop the unneeded information

# In[ ]:



# Drop the city, province and Cntry cols, will include nationality but all these
# seemed redundant on the initial rf and XGBoost models

drop_cols = ['City', 'Pr/St', 'Cntry', 'Last Name', 'First Name', 'Team', 'Born']

test_x.drop(drop_cols, axis = 1, inplace = True)

train_x.drop(drop_cols, axis = 1, inplace = True)


# ## Use of sckikit pipelines for data processing
# 
# With the data cleaned, I can now use parallel data processing pipelines to:
#  - impute the median for missing numerical values
#  - binarize (one-hot encode) each of the categorical columns
#  - merge the numerical and categorical arrays into a single input
#  
# This process is shown below. I first identify the numerical and categorical columns, then write a DataFrameSelector class to pull these columns out of the input and use them in the processsing pipelines.

# In[ ]:




#check the data types of the remaining columns
train_x.dtypes
for i in train_x.dtypes:
	print(i)


#Categoricals:
cat_attribs = ['Nat', 'Hand', 'Position']

num_attribs = list(train_x.drop(cat_attribs,axis=1).columns)


# In[ ]:



class DataFrameSelector(BaseEstimator, TransformerMixin):
	""" this class will select a subset of columns,
		pass in the numerical or categorical columns as 
		attribute names to get just those columns for processing"""
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		return X[self.attribute_names]


# Below is a class to employ the LabelBinarizer() function for multiple categorical columns at once. It returns a single binary array, and also has the self.classes_ variable that keeps track of which variables are stored in which columns.

# In[ ]:




class MultiColBinarize(BaseEstimator, TransformerMixin):
	""" take a df with multiple categoricals
		one hot encode them all and return the numpy array"""
	def __init__(self, alter_df= True):
		self.alter_df = alter_df
	def fit(self, X, y=None):
		"""load the data in, initiate the binarizer for each column"""
		self.X = X
		self.cols_list = list(self.X.columns)
		self.binarizers = []
		for i in self.cols_list:
			encoder = LabelBinarizer()
			encoder.fit(self.X[i])
			self.binarizers.append(encoder)
		return self
	def transform(self, X):
		""" for each of the columns, use the existing binarizer to make new cols """
		self.X = X
		self.binarized_cols = self.binarizers[0].transform(self.X[self.cols_list[0]])
		self.classes_ = list(self.binarizers[0].classes_)
		for i in range(1,len(self.cols_list)):
			binarized_col = self.binarizers[i].transform(self.X[self.cols_list[i]])
			self.binarized_cols = np.concatenate((self.binarized_cols , binarized_col), axis = 1)
			self.classes_.extend(list(self.binarizers[i].classes_))
		return self.binarized_cols


# The numerical processing and categorical processing functions are then deployed on the respective data subsets using the following pipelines

# In[ ]:



num_pipeline = Pipeline([
		('selector', DataFrameSelector(num_attribs)),
		('imputer', Imputer(strategy="median")),
		('std_scaler', StandardScaler()),
	])


# In[ ]:



# select the categorical columns, binarize them 
cat_pipeline = Pipeline([
		('selector', DataFrameSelector(cat_attribs)),
		('label_binarizer', MultiColBinarize()),
	])


# The two pipelines are called on the train data, and the output is concatenated into a single array

# In[ ]:



train_num_processed = num_pipeline.fit_transform(train_x)
train_cat_processed = cat_pipeline.fit_transform(train_x)

train_x_clean =  np.concatenate((train_num_processed,train_cat_processed),axis=1)


# The test data is just transformed (not fit!), this is so we impute based on the training data, and so the binarized columns match across the datasets.

# In[ ]:



test_num_processed = num_pipeline.transform(test_x)
test_cat_processed = cat_pipeline.transform(test_x)

test_x_clean =  np.concatenate((test_num_processed,test_cat_processed),axis=1)


# Sanity check that the number of columns are the same for both

# In[ ]:



train_x_clean.shape


# In[ ]:



test_x_clean.shape


# ## Training the predictors
# 
# A simple grid search of hyperparamaters is performed below to optimize the three models we are deploying
# 
# These are commented out for kaggle time restrictions. try them yourself!
# ### support vector machine

# In[ ]:


"""
svm_reg = SVR(kernel="linear")


svr_param_grid = [
		{'kernel': ['rbf','linear'], 'C': [1.0, 10., 100., 1000.0],
		'gamma': [0.01, 0.1,1.0]}
	]


svm_grid_search = GridSearchCV(svm_reg, svr_param_grid, cv=5,
						scoring='neg_mean_squared_error')

svm_grid_search.fit(train_x_clean, train_y)

svm_grid_search.best_params_

svm_grid_search.best_estimator_

cvres = svm_grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(np.sqrt(-mean_score), params)

"""


# ### Random forest regression

# In[ ]:


"""
forest_reg = RandomForestRegressor(random_state=42)

rf_param_grid = [
	{'n_estimators': [3, 10, 30,100,300,1000], 'max_features': [2, 4, 6, 8]},
	{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
rf_grid_search = GridSearchCV(forest_reg, rf_param_grid, cv=5,
						   scoring='neg_mean_squared_error')
rf_grid_search.fit(train_x_clean, train_y)


rf_grid_search.best_params_
rf_grid_search.best_estimator_

cvres = rf_grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(np.sqrt(-mean_score), params)

"""


# ### XG Boost

# In[ ]:


"""
XGBoost_reg = xgb.XGBRegressor()

#note all the params below must be wrapped in lists
xgb_param_grid  = [{'min_child_weight': [20,25,30], 
					'learning_rate': [0.1, 0.2, 0.3], 
					'colsample_bytree': [0.9], 
					'max_depth': [5,6,7,8], 
					'reg_lambda': [1.], 
					'nthread': [-1], 
					'n_estimators': [100,1000,2000],
					'early_stopping_rounds':50,
					'objective': ['reg:linear']}]


xgb_grid_search = GridSearchCV(XGBoost_reg, xgb_param_grid, cv=5,
					scoring='neg_mean_squared_error', n_jobs=1)

xgb_grid_search.fit(train_x_clean, train_y)



xgb_grid_search.best_params_

xgb_grid_search.best_estimator_

cvres = xgb_grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(np.sqrt(-mean_score), params)"""


# ### test the above 3 models to see their predictive abilities

# In[ ]:



#SVM
opt_svm_params = {'C': 1000.0, 
				'gamma': 0.01, 
				'kernel': 'linear'}

#need the ** to unpack the dictonary so all the params don't get assigned to one
opt_svm_reg = SVR(**opt_svm_params)

opt_svm_reg.fit(train_x_clean, train_y)


y1 = opt_svm_reg.predict(test_x_clean)

y1_mse = mean_squared_error(test_y, y1)
y1_rmse = np.sqrt(y1_mse)
y1_rmse


# In[ ]:



#RF
opt_rf_params= {'max_features': 8, 'n_estimators': 100}

opt_forest_reg = RandomForestRegressor(**opt_rf_params, random_state=42)

opt_forest_reg.fit(train_x_clean, train_y)

y2 = opt_forest_reg.predict(test_x_clean)

y2_mse = mean_squared_error(test_y,y2 )
y2_rmse = np.sqrt(y2_mse)
y2_rmse


# In[ ]:


#XGB
opt_xgb_params = {'colsample_bytree': 0.9,
				'learning_rate': 0.1,
				'max_depth': 7,
				'min_child_weight': 30,
				'n_estimators': 1000,
				'nthread': -1,
				'objective': 'reg:linear',
				'reg_lambda': 1.0}


opt_XGBoost_reg = xgb.XGBRegressor(**opt_xgb_params)

opt_XGBoost_reg.fit(train_x_clean, train_y)

y3 = opt_XGBoost_reg.predict(test_x_clean)

y3_mse = mean_squared_error(test_y, y3)
y3_rmse = np.sqrt(y3_mse)
y3_rmse


# # combine optimal models into a single model
# 
# I developed a skeleton class that can be altered to combine multiple predictors into a single sklearn class that spits
# out predicted values. Below this is used on the three models we have developed here.
# The class has a tuning param that changes the weights of the models and uses cross validation function to get the rmse scores. Running this class for multiple weights can provide an approximation of the best model weights to use in the combined predictions.
# 

# In[ ]:



class ensemble_predictor(BaseEstimator, TransformerMixin):
	""" take in a dataset and train it with three models,
		combining the outputs to make predictions"""
	def __init__(self, weights= { 'xgb': 0.33, 'rf': 0.33, 'svm' : 0.34}):
		self.weights = weights
		self.opt_xgb_params = {'colsample_bytree': 0.9,
					'learning_rate': 0.1,
					'max_depth': 7,
					'min_child_weight': 30,
					'nthread': -1,
					'objective': 'reg:linear',
					'reg_lambda': 1.0}
		self.opt_svm_params = {'C': 1000.0, 
				'gamma': 0.01, 
				'kernel': 'linear'}
		self.opt_rf_params= {'max_features': 8, 'n_estimators': 100}

	def fit(self, X, y):
		"""load the data in, initiate the models"""
		self.X = X
		self.y = y
		self.opt_XGBoost_reg = xgb.XGBRegressor(**self.opt_xgb_params)
		self.opt_forest_reg = RandomForestRegressor(**self.opt_rf_params)
		self.opt_svm_reg = SVR(**self.opt_svm_params)
		""" fit the models """
		self.opt_XGBoost_reg.fit(self.X ,self.y)
		self.opt_forest_reg.fit(self.X ,self.y)
		self.opt_svm_reg.fit(self.X ,self.y)
	def predict(self, X2):
		""" make the predictions for the models, combine based on weights """
		self.y_xgb = self.opt_XGBoost_reg.predict(X2)
		self.y_rf = self.opt_forest_reg.predict(X2)
		self.y_svm = self.opt_svm_reg.predict(X2)
		""" multiply the predictions by their weights, return optimal """
		self.prediction = self.y_xgb * self.weights['xgb'] 						+ self.y_rf * self.weights['rf'] 						+ self.y_svm * self.weights['svm']
		return self.prediction


# In[ ]:


weight_variants = [
{ 'xgb': 0.33, 'rf': 0.33, 'svm' : 0.34},
{ 'xgb': 0.9, 'rf': 0.05, 'svm' : 0.05},
{ 'xgb': 0.8, 'rf': 0.1, 'svm' : 0.1},
{ 'xgb': 0.5, 'rf': 0.3, 'svm' : 0.2},
{ 'xgb': 0.3, 'rf': 0.2, 'svm' : 0.5},
{ 'xgb': 0.3, 'rf': 0.5, 'svm' : 0.2}
]


# In[ ]:



#determine the optimal weights for the different models via cross validation
w_results = []
for params in weight_variants:
    model = ensemble_predictor(weights = params)
    ensemble_score = cross_val_score(model, train_x_clean, train_y,
                                    scoring="neg_mean_squared_error", cv=5)
    ensemble_rmse = np.sqrt(-ensemble_score)
    print('%s\t %s'% (params, ensemble_rmse.mean()))
    w_results.append(ensemble_rmse.mean())


# In[ ]:



import matplotlib.pyplot as plt

y_pos = np.arange(len(w_results))

weight_variant_names = ["{ 'xgb': 0.33, 'rf': 0.33, 'svm' : 0.34}",
                         "{ 'xgb': 0.9, 'rf': 0.05, 'svm' : 0.05}",
                         "{ 'xgb': 0.8, 'rf': 0.1, 'svm' : 0.1}",
                         "{ 'xgb': 0.5, 'rf': 0.3, 'svm' : 0.2}",
                         "{ 'xgb': 0.3, 'rf': 0.2, 'svm' : 0.5}",
                         "{ 'xgb': 0.3, 'rf': 0.5, 'svm' : 0.2}"]
plt.bar(y_pos, w_results, align='center', alpha=0.5)
plt.xticks(y_pos, weight_variant_names, rotation=90)
plt.ylabel('rmse')
plt.ylim(1300000,1450000)
plt.title('RMSE of different ensemble model weights')
 
plt.show()


# winner: {'xgb': 0.8, 'rf': 0.1, 'svm': 0.1} 1322950.1668
# Try again with the new weight variants, tuned in towards the optimal numbers
# 

# In[ ]:


weights ={'xgb': 0.8, 'rf': 0.15, 'svm': 0.05} #SUB IN OPTIMAL WEIGHTS


# In[ ]:


opt_model = ensemble_predictor(weights)
opt_model.fit(train_x_clean, train_y)
final_predictions = opt_model.predict(test_x_clean)


# In[ ]:


opt_mean_squared_error = mean_squared_error(test_y,final_predictions)

opt_rmse = np.sqrt(opt_mean_squared_error)
opt_rmse


# In[ ]:


meadian_guess = [np.median(test_y) for x in test_y]
meadian_guess


median_mse= mean_squared_error(test_y,meadian_guess)

median_rmse = np.sqrt(median_mse)
median_rmse


# When I ran this locall the ensemble model was about 1.3 million dollars closer on average than guessing by just the median alone.
# 
# This mixed model was better then previous iterations, when we ran Random Forest regression the model was off by an average of \$1,578,497 and with XGBoost alone the model was only slightly improved at \$1,574,073 When we combined models here, we see that we are about \$25,000 closer on average, which is a slight improvement, but an improvement nonetheless! Any suggestions on ways to further tweak the predictive model are welcomed in the comments.

# 
