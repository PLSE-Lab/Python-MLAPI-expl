#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn import cross_validation, grid_search, linear_model, metrics, pipeline, preprocessing


# **Submissions are evaluated one the Root Mean Squared Logarithmic Error (RMSLE) so lets define it. Also we are using common mean absolute error**

# In[ ]:


def rmsle(y, y_):
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


# In[ ]:


data = pd.read_csv("../input/train.csv")


# In[ ]:


data.head(3)


# In[ ]:


data.isnull().values.any()


# In[ ]:


data.datetime = data.datetime.apply(pd.to_datetime)
data['month'] = data.datetime.apply(lambda x : x.month)
data['hour'] = data.datetime.apply(lambda x : x.hour)
data.head()


# In[ ]:


train_data = data.iloc[:-1000, :]
test_data = data.iloc[-1000:, :]
print(data.shape, train_data.shape, test_data.shape)


# In[ ]:


train_labels = train_data['count'].values
train_data = train_data.drop(['datetime', 'count', 'casual', 'registered'], axis = 1)
test_labels = test_data['count'].values
test_data = test_data.drop(['datetime', 'count', 'casual', 'registered'], axis = 1)


# In[ ]:


binary_data_columns = ['holiday', 'workingday']
binary_data_indices = np.array([(column in binary_data_columns) for column in train_data.columns], dtype = bool)

categorical_data_columns = ['season', 'weather', 'month'] 
categorical_data_indices = np.array([(column in categorical_data_columns) for column in train_data.columns], dtype = bool)

numeric_data_columns = ['temp', 'atemp', 'humidity', 'windspeed', 'hour']
numeric_data_indices = np.array([(column in numeric_data_columns) for column in train_data.columns], dtype = bool)


# In[ ]:


transformer_list = [        
            #binary
            ('binary_variables_processing', preprocessing.FunctionTransformer(lambda data: data[:, binary_data_indices])), 
                    
            #numeric
            ('numeric_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, numeric_data_indices])),
                ('scaling', preprocessing.StandardScaler(with_mean = 0))            
                        ])),
        
            #categorical
            ('categorical_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, categorical_data_indices])),
                ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown = 'ignore'))            
                        ])),
        ]


# **SGDRegressor**

# In[ ]:


regressor = linear_model.Lasso(max_iter = 2000)


# In[ ]:


estimator = pipeline.Pipeline(steps = [       
    ('feature_processing', pipeline.FeatureUnion(transformer_list=transformer_list)),
    ('model_fitting', regressor)
    ]
)

estimator.fit(train_data, train_labels)
predicted = estimator.predict(test_data)

print("RMSLE: ", rmsle(test_labels, predicted))
print("MAE: ",  metrics.mean_absolute_error(test_labels, predicted))


# **Lets try out to filter best model parameters**

# In[ ]:


parameters_grid = {
    'model_fitting__alpha' : [0.1, 1, 2, 3, 4, 10, 30]
}


# In[ ]:


rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)
grid_cv = grid_search.GridSearchCV(estimator, parameters_grid, scoring = rmsle_scorer, cv = 4)
grid_cv.fit(train_data, train_labels)

predicted = grid_cv.best_estimator_.predict(test_data)

print("RMSLE: ", rmsle(test_labels, predicted))
#print("MAE: ",  metrics.mean_absolute_error(test_labels, predicted))
print("Best params: ", grid_cv.best_params_)


# In[ ]:


estimator.get_params().keys()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(random_state = 0, max_depth = 20, n_estimators = 150)
estimator = pipeline.Pipeline(steps = [       
    ('feature_processing', pipeline.FeatureUnion(transformer_list = transformer_list)),
    ('model_fitting', regressor)
    ]
)
estimator.fit(train_data, train_labels)
#metrics.mean_absolute_error(test_labels, estimator.predict(test_data))
print("RMSLE: ", rmsle(test_labels, estimator.predict(test_data)))


# In[ ]:


#estimator.get_params().keys()


# In[ ]:


##parameters_grid = {
##    'model_fitting__n_estimators' : [70, 100, 130],
##    'model_fitting__max_features' : [3, 4, 5, 6],
##}
##
##grid_cv = grid_search.GridSearchCV(estimator, parameters_grid, scoring = 'neg_mean_absolute_error', cv = 3)
##grid_cv.fit(train_data, train_labels)
##
##print(-grid_cv.best_score_)
##print(grid_cv.best_params_)


# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
pylab.figure(figsize=(8, 3))

pylab.subplot(1,2,1)
pylab.grid(True)
pylab.xlim(-100,1100)
pylab.ylim(-100,1100)
pylab.scatter(train_labels, grid_cv.best_estimator_.predict(train_data), alpha=0.5, color = 'red')
pylab.scatter(test_labels, grid_cv.best_estimator_.predict(test_data), alpha=0.5, color = 'blue')
pylab.title('linear model')

pylab.subplot(1,2,2)
pylab.grid(True)
pylab.xlim(-100,1100)
pylab.ylim(-100,1100)
pylab.scatter(train_labels, estimator.predict(train_data), alpha=0.5, color = 'red')
pylab.scatter(test_labels, estimator.predict(test_data), alpha=0.5, color = 'blue')
pylab.title('random forest model')


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.9, max_depth = 4)

estimator = pipeline.Pipeline(steps = [       
    ('feature_processing', pipeline.FeatureUnion(transformer_list = transformer_list)),
    ('model_fitting', gbr)
    ]
)
estimator.fit(train_data, train_labels)
#metrics.mean_absolute_error(test_labels, estimator.predict(test_data))
print("RMSLE: ", rmsle(test_labels, estimator.predict(test_data)))


# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
pylab.figure(figsize=(8, 3))

pylab.subplot(1,2,1)
pylab.grid(True)
pylab.xlim(-100,1100)
pylab.ylim(-100,1100)
pylab.scatter(train_labels, grid_cv.best_estimator_.predict(train_data), alpha=0.5, color = 'red')
pylab.scatter(test_labels, grid_cv.best_estimator_.predict(test_data), alpha=0.5, color = 'blue')
pylab.title('linear model')

pylab.subplot(1,2,2)
pylab.grid(True)
pylab.xlim(-100,1100)
pylab.ylim(-100,1100)
pylab.scatter(train_labels, estimator.predict(train_data), alpha=0.5, color = 'red')
pylab.scatter(test_labels, estimator.predict(test_data), alpha=0.5, color = 'blue')
pylab.title('gbr model')


# In[ ]:


real_test_data = pd.read_csv("../input/test.csv")
real_test_data_ids = real_test_data["datetime"]
real_test_data.head()


# In[ ]:


real_test_data.datetime = real_test_data.datetime.apply(pd.to_datetime)
real_test_data['month'] = real_test_data.datetime.apply(lambda x : x.month)
real_test_data['hour'] = real_test_data.datetime.apply(lambda x : x.hour)
real_test_data.head()


# In[ ]:


real_test_data = real_test_data.drop(['datetime'], axis = 1)


# In[ ]:


real_test_predictions = estimator.predict(real_test_data)


# In[ ]:


real_test_predictions.min()


# In[ ]:


submission = pd.DataFrame({
        "datetime": real_test_data_ids,
        "count": [max(0, x) for x in real_test_predictions]
    })
submission.head()


# In[ ]:


submission.to_csv('bike_predictions.csv', index=False)


# In[ ]:




