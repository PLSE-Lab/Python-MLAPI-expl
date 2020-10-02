# M6 - WEEK 6 | PROJECT: Kaggle Competition - Predict Traffic Congestion#!/usr/bin/env python
# coding: utf-8

# # M6 - WEEK 6 | PROJECT: Kaggle Competition - Predict Traffic Congestion

# >"Our task here is to predict traffic congestion based on aggregate measures of  stopping distance and waiting times at intersections in 4 major US cities."

# ## Goal

#  -   process data
#  -  run model
#  - submit predictions

# ### Process data

# We start by importing all useful modules:
import pandas as pd # import pandas
import numpy as np # import numpy

# encoding and splitting
from sklearn import preprocessing # preprocessing for Label and OneHotEncode
from sklearn.model_selection import train_test_split #for train test split

# models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor


## multioutput necessary models
from sklearn.multioutput import MultiOutputRegressor # multioutput wrapper
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor

## import Gridsearch for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# used metrics
from sklearn.metrics import mean_squared_error # import mean squared error as we are suppose to use rmse

#silence warning
#####import warnings
#####warnings.filterwarnings("ignore") 


# We will start by importing our datasets:
train = pd.read_csv('../input/bigquery-geotab-intersection-congestion/train.csv') # load train
test = pd.read_csv('../input/bigquery-geotab-intersection-congestion/test.csv') #load test
subdf  = pd.read_csv('../input/bigquery-geotab-intersection-congestion/sample_submission.csv') # load submission file


# Then we define a function to process data:

# In[8]:


def dataproc(df):
    _df = df
    
    # 'IntersectionId', 'EntryStreetName', 'ExitStreetName', 'Path',   --- removed as latitutde and longitude gives location and heading gives direction
    feat_list = ['Latitude', 'Longitude', 'EntryHeading', 'ExitHeading', 'Hour', 'Weekend', 'Month', 'City']
    num_feat_list = ['Latitude', 'Longitude', 'Hour', 'Weekend', 'Month']
    cat_feat_list = ['EntryHeading', 'ExitHeading', 'City']
    
    
    X = _df[feat_list]
        
    #select cat
    X1 = X.select_dtypes(include=[object])

    #LabelEncode
    # instantiate
    le = preprocessing.LabelEncoder()

    # fit and transform
    X2 = X1.apply(le.fit_transform)

    #OneHotEncode
    # instantiate
    enc = preprocessing.OneHotEncoder()

    # fit and transform
    enc.fit(X2)
    X3 = enc.transform(X2).toarray()

    X3p = pd.DataFrame(X3, columns = ["Cat_"+str(int(i)) for i in range(X3.shape[1])])

    Xx = pd.concat([X, X3p], axis=1) # merge new features with old
    X = Xx.drop(columns=cat_feat_list, axis=1) # drop unecessary features
    
    return X

# Processing suite
X = dataproc(train) # train processed dataset

# list of targets
targ_list = ['TotalTimeStopped_p20', 'TotalTimeStopped_p40', 'TotalTimeStopped_p50', 'TotalTimeStopped_p60', 'TotalTimeStopped_p80', 'DistanceToFirstStop_p20', 'DistanceToFirstStop_p40', 'DistanceToFirstStop_p50', 'DistanceToFirstStop_p60', 'DistanceToFirstStop_p80']
y = train[targ_list] #target dataset
stest = dataproc(test) # processed test submission data


# these '#####' can be removed to get operational code if necessary
# To train our model let's split our train dataset
#####X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1) #, train_size=100000, test_size=5000

# dictionary of models
#####models = {'knn': KNeighborsRegressor(), 'linear regression': LinearRegression(),'lasso': Lasso(),  'ridge':  RidgeCV(), 'elasticNet': ElasticNet(), 'random forest': RandomForestRegressor(), 'decision tree':DecisionTreeRegressor(),'extra-trees': ExtraTreesRegressor(),'sdg regressor': MultiOutputRegressor(SGDRegressor()),'ada boost regressor': MultiOutputRegressor(AdaBoostRegressor()),'gradient boosting regressor': MultiOutputRegressor(GradientBoostingRegressor()),'extreme boosting regressor': MultiOutputRegressor(XGBRegressor(**{'objective':'reg:squarederror'}))}

# # Build an empty dictionary to collect prediction values
#####y_pred = dict()
#####mse = dict()
 
#####for key, model in models.items():     
#####     model.fit(X_train, y_train)                    
#####     y_pred[key] = model.predict(X_test)   
#####     mse[key] = mean_squared_error(y_test, model.predict(X_test))

#####print(mse)


# Gridsearch
model = MultiOutputRegressor(XGBRegressor(**{'objective':'reg:squarederror'}))
#gridparams = {'estimator__learning_rate': [0.0001, 0.001, 0.05, 0.1, 0.2, 0.3], 'estimator__n_jobs' : [-1], 'estimator__max_depth': [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]}
gridparams = {'estimator__learning_rate': [0.0001, 0.001, 0.05, 0.1, 0.2, 0.3], 'estimator__max_depth': [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]}

#g = GridSearchCV(model, gridparams, verbose=1, cv=6, n_jobs=-1, scoring='neg_mean_squared_error')
g = GridSearchCV(model, gridparams, verbose=1, cv=6, scoring='neg_mean_squared_error')

r = g.fit(X, y)
print("Best: %f using %s" % (r.best_score_, r.best_params_))
print(np.sqrt(mean_squared_error(y_test, r.predict(X_test))))


# I will use our model to predict based on submission:
y_pred = r.predict(stest)

# Now let's submit:

def subm(y_pred):
    tpre = pd.DataFrame(y_pred, columns=targ_list)
    dpred = pd.DataFrame({"0": tpre["TotalTimeStopped_p20"], "1": tpre["TotalTimeStopped_p50"], "2": tpre["TotalTimeStopped_p80"], "3": tpre["DistanceToFirstStop_p20"], "4": tpre["DistanceToFirstStop_p50"], "5": tpre["DistanceToFirstStop_p80"]})
    subdf['Target'] = dpred.stack().values
    subdf.to_csv('meosub.csv', index=False)


#... and run our sub function
subdf  = pd.read_csv('data/sample_submission.csv') # load submission file
subm(y_pred)

