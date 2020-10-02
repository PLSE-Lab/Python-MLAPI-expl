#!/usr/bin/env python
# coding: utf-8

# # <center> IRIS PROJECT WITH H2O </center>

# In[ ]:


import h2o


# In[ ]:


h2o.init()


# ## importing data with h20 we can import data from pandas too

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


## import data 
get_ipython().system('ls ')


# In[ ]:


df = h2o.import_file('../input/irisdataset/Iris.csv')


# In[ ]:


df.head()


# ## importing with different parameter

# In[ ]:



df = h2o.import_file('../input/irisdataset/Iris.csv',col_names=['id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'],
                    col_types=['numeric',"numeric", "numeric", "numeric", "numeric", "enum"])


# In[ ]:


df.head()


# ##  they are are some other option for importing data
# #df = h2o.import_file("hdfs://namenode/user/path/to/my.csv")
# #df = h2o.import_file("s3://<AWS_ACCESS_KEY>:<AWS_SECRET_KEY>@mybucket/my.csv")
# #df = h2o.import_file("https://s3.amazonaws.com/mybucket/my.csv")
# #df = h2o.import_file("/path/to/my.csv")
# 

# ## we can use pandas dataframe too

# In[ ]:


dfp = pd.read_csv('../input/irisdataset/Iris.csv')


# In[ ]:


df = h2o.H2OFrame(dfp)


# In[ ]:


df.head()


# ## names of the column

# In[ ]:


df.names


# ## df types

# In[ ]:


df.types


# In[ ]:


df.frame_id


# In[ ]:


df[['SepalLengthCm']].max()


# In[ ]:


df[['SepalLengthCm']].min()


# In[ ]:


df[['SepalLengthCm']].mean()


# # histogram plot

# In[ ]:


df[['SepalLengthCm']].hist()


# In[ ]:


df[['SepalWidthCm']].hist()


# In[ ]:


df[['PetalWidthCm']].hist()


# In[ ]:


df[['PetalLengthCm']].hist()


# In[ ]:


df.head()


# In[ ]:


## dropping column


# In[ ]:


df.drop('Id',axis=1)


# In[ ]:


###or


# In[ ]:


df=df[:,1:]   ## take the datafrmame but lose the first column


# In[ ]:


df.head()


# In[ ]:


## applying seeding
df.frame_id


# In[ ]:


## describe data


# In[ ]:


df.describe()


# ## we can see there is no misssing value

# ## Featuer Engineering

# In[ ]:


df['sepal_ratio'] = df['SepalLengthCm']/df['SepalWidthCm']


# In[ ]:


df.head()


# In[ ]:


df['petal_ratio'] = df['PetalLengthCm']/df['PetalWidthCm']


# In[ ]:


df.head()


# ## difference between pandas correlation and H2o correlation

# In[ ]:


df.cor() 


# In[ ]:


dfp.corr()


# In[ ]:


sns.heatmap(dfp.corr())


# In[ ]:


dfp.corr(method='spearman')


# In[ ]:


sns.heatmap(dfp.corr(method='spearman'))


# # splitting data in h2o and in sklearn

# In[ ]:


## in H20
train_h,test_h,valid_h = df.split_frame([0.6,0.2])


# In[ ]:


train_h


# In[ ]:


test_h


# In[ ]:


valid_h


# ## in sklearn

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


xtrain,ytrain,xtest,ytest = train_test_split(dfp.drop('Species',axis=1),dfp[['Species']])


# In[ ]:


xtrain.head()


# In[ ]:


ytrain.head()


# In[ ]:


xtest.head()


# In[ ]:


ytest.head()


# ## exporting data

# In[ ]:


#h2o.exportFile(d, "/path/to/d.csv")
#h2o.exportFile(d, "s3://mybucket/d.csv")
#h2o.exportFile(d, "s3://<AWS_ACCESS_KEY>:<AWS_SECRET_KEY>@mybucket/d.csv")
#h2o.exportFile(d, "hdfs://namenode/path/to/d.csv")


# In[ ]:


h2o.export_file(df,'export.csv')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('cat export.csv')


# ## For saving to a location on the H2O server, use h2o.exportFile() (h2o.export_file() in
# # Python), where the first parameter is the frame to save, and the second is the disk path and
# # filename.

# ## POJO
# 
# ## pojo is  plain old java object
# ## we can save our machine learning model or deep learning model in pojo format 

# ## you can apply pandas function 
# ## but in order to do that you have to change it to a data frame in pandas

# In[ ]:


df.as_data_frame().plot()


# In[ ]:


sns.heatmap(df.as_data_frame().corr())


# In[ ]:


sns.heatmap(df.as_data_frame().corr(method='spearman'))


# In[ ]:


df.as_data_frame().plot.barh()


# In[ ]:


d = df.as_data_frame()


# In[ ]:


pd.plotting.scatter_matrix(d)


# In[ ]:


## you can import the whole csv in  a folder
train_h


# In[ ]:


y = 'Species'


# In[ ]:


x = df.names


# In[ ]:


x.remove(y)


# In[ ]:


## we will do a grid search for finding the best parameter


# In[ ]:


import math
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch

#We only provide the required parameters, everything else is default
gbm = H2OGradientBoostingEstimator()
gbm.train(x=x, y=y, training_frame=train_h)

## Show a detailed model summary
print (gbm)


# In[ ]:


perf = gbm.model_performance(valid_h)


# In[ ]:


print(perf)


# In[ ]:


## with valid dataframne
cv_gbm = H2OGradientBoostingEstimator(nfolds = 4, seed = 0xDECAF)
cv_gbm.train(x = x, y = y, training_frame = train_h.rbind(valid_h))


# In[ ]:


cv_gbm


# In[ ]:


## now with grid search (this may take time depending on the spec of your computer .altough its run on jvm dont take too much time)


# In[ ]:


gbm_params1 = {'learn_rate': [0.01, 0.1],
                'max_depth': [3, 5, 9],
                'sample_rate': [0.8, 1.0],
                'col_sample_rate': [0.2, 0.5, 1.0]}


# In[ ]:


gbm_grid1 = H2OGridSearch(model=H2OGradientBoostingEstimator,
                          grid_id='gbm_grid1',
                          hyper_params=gbm_params1)


# In[ ]:


gbm_grid1.train(x=x, y=y,
                training_frame=train_h,
                validation_frame=valid_h,
                ntrees=100,stopping_metric = "AUC",
                seed=1)


# In[ ]:


gbm_grid1


# In[ ]:


## RANDOm GRID search 


# In[ ]:


gbm_params2 = {'learn_rate': [i * 0.01 for i in range(1, 11)],
                'max_depth': list(range(2, 11)),
                'sample_rate': [i * 0.1 for i in range(5, 11)],
                'col_sample_rate': [i * 0.1 for i in range(1, 11)]}

# Search criteria
search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 36, 'seed': 1}

# Train and validate a random grid of GBMs
gbm_grid2 = H2OGridSearch(model=H2OGradientBoostingEstimator,
                          grid_id='gbm_grid2',
                          hyper_params=gbm_params2,
                          search_criteria=search_criteria)
gbm_grid2.train(x=x, y=y,
                training_frame=train_h,
                validation_frame=valid_h,
                ntrees=100,
                seed=1)



# In[ ]:


gbm_grid2


# In[ ]:


gbm_gridperf2 = gbm_grid2.get_grid(sort_by='mse', decreasing=True)


# In[ ]:


gbm_gridperf2

# Grab the top GBM model, chosen by validation AUC
best_gbm2 = gbm_gridperf2.models[0]

# Now let's evaluate the model performance on a test set
# so we get an honest estimate of top model performance
#best_gbm_perf2 = best_gbm2.model_performance(test)


# In[ ]:


best_gbm2


# In[ ]:



best_gbm_perf2 = best_gbm2.model_performance(test_h)

best_gbm_perf2.mse()


# In[ ]:


best_gbm_perf2.rmse()


# In[ ]:


## doing this thing again witha  differene


# In[ ]:


predict = best_gbm2.predict(test_h)


# In[ ]:


predict


# In[ ]:


# Predict the contributions using the GBM model and test data.
staged_predict_proba = best_gbm2.staged_predict_proba(test_h)


# In[ ]:


staged_predict_proba


# In[ ]:


conf = best_gbm2.confusion_matrix(test_h)


# In[ ]:


conf


# # Random forest classification 
# 
# 

# In[ ]:


from h2o.estimators.random_forest import H2ORandomForestEstimator
m = H2ORandomForestEstimator(
ntrees=100,
stopping_metric="misclassification",
stopping_rounds=3,
stopping_tolerance=0.02, #2%
max_runtime_secs=60,
model_id="RF:stop_test"
)
m.train(x, y, train_h, validation_frame=valid_h)


# In[ ]:


m


# In[ ]:


pref = m.model_performance(valid_h)


# In[ ]:


pref


# In[ ]:


cv_m = H2OGradientBoostingEstimator()
cv_m.train(x = x, y = y, training_frame = train_h.rbind(valid_h))


# In[ ]:


cv_m


# In[ ]:


import h2o.grid
g = h2o.grid.H2OGridSearch(
h2o.estimators.H2ORandomForestEstimator(
nfolds=10
),
hyper_params={
"ntrees": [50, 100, 120],
"max_depth": [40, 60],
"min_rows": [1, 2]
}
)
g.train(x, y, train_h)


# In[ ]:


g_gridperf2 = g.get_grid(sort_by='mse', decreasing=True)


# In[ ]:


g_gridperf2


# In[ ]:


g.confusion_matrix(test_h)


# # Naive bayes classifier

# In[ ]:


from h2o.estimators import naive_bayes


# In[ ]:


nv = naive_bayes.H2ONaiveBayesEstimator()


# In[ ]:


nv.train(x,y,train_h)


# In[ ]:


nv


# In[ ]:


pred = nv.predict(test_h)


# In[ ]:


pred


# In[ ]:


nv.confusion_matrix(test_h)


# In[ ]:


nv.mse()


# In[ ]:


nv.rmse()


# In[ ]:


nv.cross_validation_metrics_summary()


# In[ ]:


## this is the pojo model for the naive bias classifier


# In[ ]:


nv.download_pojo()


# In[ ]:


cv_m = naive_bayes.H2ONaiveBayesEstimator()
cv_m.train(x = x, y = y, training_frame = train_h.rbind(valid_h))


# In[ ]:


cv_m


# ## all model possible (crazy right???!!!)

# In[ ]:


from h2o.automl import H2OAutoML


# In[ ]:


aml = H2OAutoML(max_models=25, seed=1)
aml.train(x=x, y=y, training_frame=train_h)


# In[ ]:


lb = aml.leaderboard
lb


# In[ ]:


preds = aml.leader.predict(test_h)


# In[ ]:


preds


# In[ ]:


preds = aml.predict(test_h)


# In[ ]:


preds


# In[ ]:


aml.sort_metric


# In[ ]:


aml.leaderboard


# ## saving the naive bias model

# In[ ]:



# save the model
model_path = h2o.save_model(model=nv, path="/tmp/mymodel", force=True)

print (model_path)



# ### loading model

# In[ ]:


# load the model
saved_model = h2o.load_model(model_path)


# In[ ]:


saved_model


# In[ ]:


from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator


# In[ ]:


nfolds = 5

# There are a few ways to assemble a list of models to stack together:
# 1. Train individual models and put them in a list
# 2. Train a grid of models
# 3. Train several grids of models
# Note: All base models must have the same cross-validation folds and
# the cross-validated predicted values must be kept.


# 1. Generate a 2-model ensemble (GBM + RF)

# Train and cross-validate a GBM
my_gbm = H2OGradientBoostingEstimator(
                                      ntrees=10,
                                      max_depth=3,
                                      min_rows=2,
                                      learn_rate=0.2,
                                      nfolds=nfolds,
                                      fold_assignment="Modulo",
                                      keep_cross_validation_predictions=True,
                                      seed=1)
my_gbm.train(x=x, y=y, training_frame=train_h)


# Train and cross-validate a RF
my_rf = H2ORandomForestEstimator(ntrees=50,
                                 nfolds=nfolds,
                                 fold_assignment="Modulo",
                                 keep_cross_validation_predictions=True,
                                 seed=1)
my_rf.train(x=x, y=y, training_frame=train_h)


# Train a stacked ensemble using the GBM and GLM above
ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_binomiale",
                                       base_models=[my_gbm, my_rf])
ensemble.train(x=x, y=y, training_frame=train_h)

# Eval ensemble performance on the test data
perf_stack_test = ensemble.model_performance(test_h)


# In[ ]:


# Compare to base learner performance on the test set
perf_gbm_test = my_gbm.model_performance(test_h)
perf_rf_test = my_rf.model_performance(test_h)
baselearner_best_auc_test = max(perf_gbm_test.mse(), perf_rf_test.mse())
stack_auc_test = perf_stack_test.mse()
print("Best Base-learner Test MSE:  {0}".format(baselearner_best_auc_test))
print("Ensemble Test MSE:  {0}".format(stack_auc_test))

# Generate predictions on a test set (if neccessary)
pred = ensemble.predict(test_h)


# In[ ]:


pred


# ## DEEP LEARNING

# <ul>
# <li>        If the distribution is bernoulli, the the response column must be 2-class categorical
# <li>        If the distribution is multinomial, the response column must be categorical.
# <li>        If the distribution is poisson, the response column must be numeric.
# <li>        If the distribution is laplace, the response column must be numeric.
# <li>        If the distribution is tweedie, the response column must be numeric.
# <li>        If the distribution is gaussian, the response column must be numeric.
# <li>        If the distribution is huber, the response column must be numeric.
# <li>        If the distribution is gamma, the response column must be numeric.
# <li>        If the distribution is quantile, the response column must be numeric.
# </ul>

# In[ ]:


from h2o.estimators import deeplearning
m = h2o.estimators.deeplearning.H2ODeepLearningEstimator()
m.train(x, y, train_h)
p = m.predict(test_h)


# In[ ]:


p


# In[ ]:


r2 = m.r2()
mse = m.mse()
rmse = m.rmse()


# In[ ]:


r2


# In[ ]:


from h2o.estimators.deeplearning import H2ODeepLearningEstimator


# In[ ]:


model = H2ODeepLearningEstimator(
distribution="multinomial",
activation="RectifierWithDropout",
hidden=[128,128,128,128],
input_dropout_ratio=0.2,
sparse=True,
l1=1e-5,
epochs=10)


# In[ ]:


model.train(
x=x,
y=y,
training_frame=train_h,
validation_frame=test_h)


# In[ ]:


model


# In[ ]:


model.predict(test_h)


# In[ ]:


model_cv = H2ODeepLearningEstimator(
distribution="multinomial",
activation="RectifierWithDropout",
hidden=[32,32,32],
input_dropout_ratio=0.2,
sparse=True,
l1=1e-5,
epochs=100,
nfolds=5)


# In[ ]:


model_cv.train(
x=x,
y=y,
training_frame=train_h)


# In[ ]:


# View specified parameters of the Deep Learning model
model.params


# In[ ]:


model_cv


# In[ ]:


model_cv.model_performance(train=True)


# In[ ]:


model_cv.model_performance(valid=True)


# In[ ]:


model.mse(valid=True)

# Cross-validated MSE
model_cv.mse(xval=True)


# In[ ]:


model_cv.predict(test_h).head()


# # Train Deep Learning model and validate on test set
# # and save the variable importances
# 
# 

# In[ ]:


model_vi = H2ODeepLearningEstimator(
distribution="multinomial",
activation="RectifierWithDropout",
hidden=[32,32,32],
input_dropout_ratio=0.2,
sparse=True,
l1=1e-5,
epochs=10,
variable_importances=True)


# In[ ]:


model_vi.train(
x=x,
y=y,
training_frame=train_h,
validation_frame=test_h)


# In[ ]:


model_vi.varimp()


# In[ ]:


## grid Search in python deep learning book Deep Learning booket page 35


# In[ ]:


hidden_opt = [[32,32],[32,16,8],[100]]
#l1_opt = [1e-4,1e-3]
hyper_parameters = {"hidden":hidden_opt}
from h2o.grid.grid_search import H2OGridSearch
model_grid = H2OGridSearch(H2ODeepLearningEstimator,
hyper_params=hyper_parameters)
model_grid.train(x=x, y=y,
distribution="multinomial", epochs=1000,
training_frame=train_h, validation_frame=test_h,
score_interval=2, stopping_rounds=3,
stopping_tolerance=0.05,
stopping_metric="misclassification")


# In[ ]:


model_grid


# In[ ]:


model_grid.confusion_matrix(test_h)


# In[ ]:


model_grid.r2


# In[ ]:


model_grid.mse()


#  # Random grid Search DEEP Learning

# In[ ]:


hidden_opt =[[17,32],[8,19],[32,16,8],[100],[10,10,10,10]]
l1_opt = [s/1e6 for s in range(1,1001)]
hyper_parameters = {"hidden":hidden_opt, "l1":l1_opt}
search_criteria = {"strategy":"RandomDiscrete",
"max_models":10, "max_runtime_secs":100,
"seed":123456}
from h2o.grid.grid_search import H2OGridSearch
model_grid = H2OGridSearch(H2ODeepLearningEstimator,
hyper_params=hyper_parameters,
search_criteria=search_criteria)
model_grid.train(x=x, y=y,
distribution="multinomial", epochs=1000,
training_frame=train_h, validation_frame=test_h,
score_interval=2, stopping_rounds=3,
stopping_tolerance=0.05,
stopping_metric="misclassification")


# In[ ]:


model_grid


# In[ ]:


model_grid.confusion_matrix


# In[ ]:


model_grid.mse()


# In[ ]:




