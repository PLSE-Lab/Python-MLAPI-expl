#!/usr/bin/env python
# coding: utf-8

# ![h2o.ai](https://avatars0.githubusercontent.com/u/1402695?s=200&v=4)

# **Automated machine learning (AutoML)** is the process of automating the end-to-end process of applying machine learning to real-world problems. In a typical machine learning application, the typical stages (and sub-stages) of work are the following:
# 1. Data preparation
#   * data pre-processing
#   * feature engineering
#   * feature extraction
#   * feature selection
# 2. Model selection
# 3. Hyperparameter optimization (to maximize the performance of the final model)
# 
# Many of these steps are often beyond the abilities of non-experts. **AutoML** was proposed as an artificial intelligence-based solution to the ever-growing challenge of applying machine learning. 
# 
# Some of the notable platforms tackling various stages of AutoML are the following:
# * [auto-sklearn](https://automl.github.io/auto-sklearn/stable/) is a Bayesian hyperparameter optimization layer on top of [scikit-learn](https://scikit-learn.org/).
# * [TPOT](https://github.com/EpistasisLab/tpot) (TeaPOT) is a Python library that automatically creates and optimizes full machine learning pipelines using genetic programming.
# * [TransmogrifAI](https://github.com/salesforce/TransmogrifAI) is a Scala/SparkML library created by [Salesforce](http://salesforce.com/) for automated data cleansing, feature engineering, model selection, and hyperparameter optimization.
# * [H2O AutoML](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) performs (simple) data preprocessing, automates the process of training a large selection of candidate models, tunes hyperparameters of the models and creates stacked ensembles.
# * [H2O Driverless AI](https://www.h2o.ai/products/h2o-driverless-ai/) is a commercial software package that automates lots of aspects of machine learning applications. It has a strong focus on automatic feature engineering. 
# 
# An overview of AutoML capabilities of H2O library is presented in this tutorial. The library can be installed simply by

# In[ ]:


get_ipython().system('pip install h2o')


# Let's import the required packages and call `h2o.init()`. The specified arguments (`nthreads` and `max_mem_size`) are optional.

# In[ ]:


import sys, os, os.path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle

import h2o
from h2o.automl import H2OAutoML

h2o.init(
    nthreads=-1,     # number of threads when launching a new H2O server
    max_mem_size=12  # in gigabytes
)


# **Example 1: a classification task**

# Let's apply the power of H2O AutoML to the ["Flight delays" competition](https://www.kaggle.com/c/flight-delays-fall-2018) (it's a binary classification task) from [mlcourse.ai](https://mlcourse.ai/).

# In[ ]:


train_df = pd.read_csv('../input/mlcourse/flight_delays_train.csv')
test_df = pd.read_csv('../input/mlcourse/flight_delays_test.csv')


# In[ ]:


print('train_df cols:', list(train_df.columns))
print('test_df cols: ', list(test_df.columns))
train_df.head()


# In[ ]:


train_df.dtypes


# The features `Month`, `DayofMonth`, `DayOfWeek`, `DepTime`, `Distance` can be represented as numbers. Let's convert those features to numerical type (a new feature `HourFloat` is added):

# In[ ]:


for df in [train_df, test_df]:
    df['Month'] = df['Month'].apply(lambda s: s.split('-')[1]).astype('int')
    df['DayofMonth'] = df['DayofMonth'].apply(lambda s: s.split('-')[1]).astype('int')
    df['DayOfWeek'] = df['DayOfWeek'].apply(lambda s: s.split('-')[1]).astype('int')
    
    df['HourFloat'] = df['DepTime'].apply(
        lambda t: (t // 100) % 24 + ((t % 100) % 60) / 60
    ).astype('float')


# Let's also introduce a new feature `Route` that is the concatenation of `Origin` and `Dest`:

# In[ ]:


for df in [train_df, test_df]:
    df['Route'] = df[['Origin', 'Dest']].apply(
        lambda pair: ''.join([str(a) for a in pair]),
        axis='columns'
    ).astype('str')


# We will not use the column `DepTime` anymore. Split the target column from the features columns in `train_df`:

# In[ ]:


target = train_df['dep_delayed_15min'].map({'Y': 1, 'N': 0})

feature_cols = [
    'Month', 'DayofMonth', 'DayOfWeek', 'HourFloat', 
    'UniqueCarrier', 'Origin', 'Dest', 'Route', 'Distance',]
train_df_modif = train_df[feature_cols]
test_df_modif = test_df[feature_cols]


# The features `UniqueCarrier`, `Origin`, `Dest`, `Route` should be categorical:

# In[ ]:


N_train = train_df_modif.shape[0]
train_test_X = pd.concat([train_df_modif, test_df_modif], axis='index')

for feat in ['UniqueCarrier', 'Origin', 'Dest', 'Route']:
    train_test_X[feat] = train_test_X[feat].astype('category')


# In[ ]:


X_train = train_test_X[:N_train]
X_test = train_test_X[N_train:]
y_train = target


# Pandas DataFrames should be converted to H2O dataframes before calling `H2OAutoML()`.
# 
# Note: if you don't have to preprocess the data, you can get H2O dataframes directly from the data files by a call like `df = h2o.import_file(datafile_path)` (where `datafile_path` is a filesystem path or a URL).

# In[ ]:


X_y_train_h = h2o.H2OFrame(pd.concat([X_train, y_train], axis='columns'))
X_y_train_h['dep_delayed_15min'] = X_y_train_h['dep_delayed_15min'].asfactor()
# ^ the target column should have categorical type for classification tasks
#   (numerical type for regression tasks)

X_test_h = h2o.H2OFrame(X_test)

X_y_train_h.describe()


# In[ ]:


aml = H2OAutoML(
    max_runtime_secs=(3600 * 8),  # 8 hours
    max_models=None,  # no limit
    seed=17
)


# Among the most important arguments (with their default values) of `H2OAutoML()` are the following:
# * `nfolds=5` -- number of folds for k-fold cross-validation (`nfolds=0` disables cross-validation)
# * `balance_classes=False` -- balance training data class counts via over/under-sampling
# * `max_runtime_secs=3600` -- how long the AutoML run will execute (in seconds)
# * `max_models=None` -- the maximum number of models to build in an AutoML run (`None` means no limitation)
# * `include_algos=None` -- list of algorithms to restrict to during the model-building phase (cannot be used in combination with `exclude_algos` parameter; `None` means that all appropriate H2O algorithms will be used)
# * `exclude_algos=None` -- list of algorithms to skip during the model-building phase (`None` means that all appropriate H2O algorithms will be used)
# * `seed=None` -- a random seed for reproducibility (AutoML can only guarantee reproducibility if `max_models` or
#   early stopping is used because `max_runtime_secs` is resource limited, meaning that if the resources are
#   not the same between runs, AutoML may be able to train more models on one run vs another)
# 
# H2O AutoML trains and cross-validates:
# * a default Random Forest (DRF), 
# * an Extremely-Randomized Forest (XRT),
# * a random grid of Generalized Linear Models (GLM),
# * a random grid of XGBoost (XGBoost),
# * a random grid of Gradient Boosting Machines (GBM), 
# * a random grid of Deep Neural Nets (DeepLearning), 
# * and 2 Stacked Ensembles, one of all the models, and one of only the best models of each kind.
# 

# In the cell below, I call `aml.train()`, save the leaderboard and all individual models. The running time is about 8 hours, so after running it once I saved the output files as a new dataset, connected the dataset to this kernel and commented out the code in the cell.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# aml.train(\n#     x=feature_cols,\n#     y=\'dep_delayed_15min\',\n#     training_frame=X_y_train_h\n# )\n\n# lb = aml.leaderboard\n# model_ids = list(lb[\'model_id\'].as_data_frame().iloc[:,0])\n# out_path = "."\n\n# for m_id in model_ids:\n#     mdl = h2o.get_model(m_id)\n#     h2o.save_model(model=mdl, path=out_path, force=True)\n\n# h2o.export_file(lb, os.path.join(out_path, \'aml_leaderboard.h2o\'), force=True)')


# Some of the arguments for `H2OAutoML.train()` are the following:
# * `training_frame` -- the H2OFrame having the columns indicated by `x` and `y`
# * `x` -- list of feature column names in `training_frame`
# * `y` -- a column name indicating the target
# * `validation_frame` -- the H2OFrame with validation data (by default and when `nfolds` > 1, `validation_frame` will be ignored)
# * `leaderboard_frame` -- the H2OFrame with test data for scoring the leaderboard (optinal; by default (`leaderboard_frame=None`) the cross-validation metric on `training_frame` will be used to generate the leaderboard rankings)
# 
# Let's take a look at the leaderboard:

# In[ ]:


models_path = "../input/h2o-automl-saved-models-classif/"

lb = h2o.import_file(path=os.path.join(models_path, "aml_leaderboard.h2o"))

lb.head(rows=10)
#lb.head(rows=lb.nrows)
# ^ to see the entire leaderboard


# Among the individual models, XGBoost is the leader (auc = 0.749523) for this task. Best individual GBM has auc = 0.741785, best XRT has auc = 0.731317, best DRF has auc = 0.725166, best DNN has auc = 0.706676.
# 
# `StackedEnsemble_AllModels` is usually the leader, `StackedEnsemble_BestOfFamily` is usually at the 2nd place. Let's look inside the `StackedEnsemble_AllModels`. It is an ensemble of all of the individual models in the AutoML run. 

# In[ ]:


se_all = h2o.load_model(os.path.join(models_path, "StackedEnsemble_AllModels_AutoML_20190414_112210"))
# Get the Stacked Ensemble metalearner model
metalearner = h2o.get_model(se_all.metalearner()['name'])


# The AutoML Stacked Ensembles use the GLM with non-negative weights as the default metalearner (combiner) algorithm. Let's examine the variable importance of the metalearner algorithm in the ensemble. This shows us how much each base learner is contributing to the ensemble. `Intercept` represents the constant term in a linear model.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
metalearner.std_coef_plot(num_of_features=20)
# ^ all importance values starting from the 16th are zero

#metalearner.coef_norm()
# ^ to see the table in the text form


# `StackedEnsemble_BestOfFamily` shows the following:

# In[ ]:


se_best_of_family = h2o.load_model(os.path.join(models_path, "StackedEnsemble_BestOfFamily_AutoML_20190414_112210"))
# Get the Stacked Ensemble metalearner model
metalearner = h2o.get_model(se_best_of_family.metalearner()['name'])

get_ipython().run_line_magic('matplotlib', 'inline')
metalearner.std_coef_plot(num_of_features=10)
#metalearner.coef_norm()


# Let's reproduce the result (auc) of a few best individual models.

# In[ ]:


from h2o.estimators.xgboost import H2OXGBoostEstimator

model_01 = h2o.load_model(os.path.join(models_path, "XGBoost_grid_1_AutoML_20190414_112210_model_19"))

excluded_params = ['model_id', 'response_column', 'ignored_columns']
model_01_actual_params = {k: v['actual'] for k, v in model_01.params.items() if k not in excluded_params}

reprod_model_01 = H2OXGBoostEstimator(**model_01_actual_params)
reprod_model_01.train(
    x=feature_cols,
    y='dep_delayed_15min',
    training_frame=X_y_train_h
)
reprod_model_01.auc(xval=True)
# ^ 0.749453, slightly worse compared to the leaderboard value


# In[ ]:


from h2o.estimators.gbm import H2OGradientBoostingEstimator

model_12 = h2o.load_model(os.path.join(models_path, "GBM_grid_1_AutoML_20190414_112210_model_85"))

excluded_params = ['model_id', 'response_column', 'ignored_columns']
model_12_actual_params = {k: v['actual'] for k, v in model_12.params.items() if k not in excluded_params}

reprod_model_12 = H2OGradientBoostingEstimator(**model_12_actual_params)
reprod_model_12.train(
    x=feature_cols,
    y='dep_delayed_15min',
    training_frame=X_y_train_h
)
reprod_model_12.auc(xval=True)
# ^ 0.741785, the same as at the leaderboard


# In[ ]:


from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch

model_93 = h2o.load_model(os.path.join(models_path, "GLM_grid_1_AutoML_20190414_112210_model_1"))

excluded_params = ['model_id', 'response_column', 'ignored_columns', 'lambda']
model_93_actual_params = {k: v['actual'] for k, v in model_93.params.items() if k not in excluded_params}

reprod_model_93 = H2OGeneralizedLinearEstimator(**model_93_actual_params)
reprod_model_93.train(
    x=feature_cols,
    y='dep_delayed_15min',
    training_frame=X_y_train_h
)
reprod_model_93.auc(xval=True)
# ^ 0.699418, the same as at the leaderboard


# Let's train the CatBoostClassifier with the default parameters and compare its results with AutoML run results.

# In[ ]:


from catboost import Pool, CatBoostClassifier, cv

cb_model = CatBoostClassifier(
    eval_metric='AUC',
    use_best_model=True,
    random_seed=17
)

cv_data = cv(
    Pool(X_train, y_train, cat_features=[4,5,6,7]),
    cb_model.get_params(),
    fold_count=5,
    verbose=False
)

print("CatBoostClassifier: the best cv auc is", np.max(cv_data['test-AUC-mean']))


# The CatBoostClassifier cross-validation auc result is 0.749009. This value falls between the 2nd (auc = 0.749523) and 3rd (auc = 0.749192) places among the individual models at the leaderboard.

# **Example 2: a regression task**

# Let's consider a regression task from the ["New York City Taxi Trip Duration" competition](https://www.kaggle.com/c/nyc-taxi-trip-duration). The challenge is to build a model that predicts the total ride duration of taxi trips in New York City. The features include pickup time, geo-coordinates, number of passengers, and a few other variables.

# In[ ]:


df_train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv', index_col=0)
df_test  = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv',  index_col=0)


# We will use only `df_train` (perform 5-fold cross-validation on it). Convert the date- and time-related features to the `datetime` format; take the logarithm (`log(1 + x)`) of the target value (trip duration). After the logarithm transform, the distribution of the target variable is close to normal (see this [kernel](https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-367)).

# In[ ]:


df_train['pickup_datetime'] = pd.to_datetime(df_train.pickup_datetime)
df_train.loc[:, 'pickup_date'] = df_train['pickup_datetime'].dt.date
df_train['dropoff_datetime'] = pd.to_datetime(df_train.dropoff_datetime)
df_train['store_and_fwd_flag'] = 1 * (df_train.store_and_fwd_flag.values == 'Y')
df_train['check_trip_duration'] = (df_train['dropoff_datetime'] - df_train['pickup_datetime']).map(
    lambda x: x.total_seconds()
)
df_train['log_trip_duration'] = np.log1p(df_train['trip_duration'].values)

cnd = np.abs(df_train['check_trip_duration'].values  - df_train['trip_duration'].values) > 1
duration_difference = df_train[cnd]

if len(duration_difference[['pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration']]) == 0:
    print('Trip_duration and datetimes are ok.')
else:
    print('Ooops.')


# Select the columns common to the train set and test set; convert `pd.DataFrame` to `H2OFrame`:

# In[ ]:


common_cols = [
    'vendor_id', 
    'pickup_datetime', 
    'passenger_count', 
    'pickup_longitude', 'pickup_latitude', 
    'dropoff_longitude', 'dropoff_latitude',
    'store_and_fwd_flag',
]

X_y_train_h = h2o.H2OFrame(
    pd.concat(
        [df_train[common_cols], df_train['log_trip_duration']],
        axis='columns'
    )
)

for ft in ['vendor_id', 'store_and_fwd_flag']:
    X_y_train_h[ft] = X_y_train_h[ft].asfactor()
    
X_y_train_h.describe()


# I have run the cell below (~8 hours), saved all models and the leaderboard, then commented out the code:

# In[ ]:


# aml = H2OAutoML(
#     max_runtime_secs=(3600 * 8),  # 8 hours
#     max_models=None,  # no limit
#     seed=SEED,
# )

# aml.train(
#     x=common_cols,
#     y='log_trip_duration',
#     training_frame=X_y_train_h
# )

# lb = aml.leaderboard
# model_ids = list(lb['model_id'].as_data_frame().iloc[:,0])
# out_path = "."

# for m_id in model_ids:
#     mdl = h2o.get_model(m_id)
#     h2o.save_model(model=mdl, path=out_path, force=True)

# h2o.export_file(lb, os.path.join(out_path, 'aml_leaderboard.h2o'), force=True)


# Interestingly, there is only one model at the leaderboard:

# In[ ]:


models_path = "../input/h2o-automl-saved-models-regress/"

lb = h2o.import_file(path=os.path.join(models_path, "aml_leaderboard.h2o"))
lb.head(rows=10)


# Let's compare the result of the model `XGBoost_1_AutoML_20190417_212831` with that of the CatBoostRegressor with the default parameters.

# In[ ]:


from catboost import Pool, CatBoostRegressor, cv

cb_model = CatBoostRegressor(
    eval_metric='RMSE',
    use_best_model=True,
    random_seed=17
)

cv_data = cv(
    Pool(df_train[common_cols], df_train['log_trip_duration'], cat_features=[0,7]),
    cb_model.get_params(),
    fold_count=5,
    verbose=False
)


# In[ ]:


print("CatBoostRegressor: the best cv rmse is", np.min(cv_data['test-RMSE-mean']))


# Default CatBoost's RMSE is slightly worse than that of the XGBoost model from the H2O AutoML run.

# **Conclusion**

# I think that H2O AutoML is worth a try. And I hope you have found this tutorial useful.
# 
# There are extremely useful "H2O AutoML Pro Tips" in the presentation "Scalable Automatic Machine Learning in H2O" mentioned in the References below.

# **References**
# 
# * [H2O.ai](https://www.h2o.ai/)
# * [H2O AutoML documentation](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
# * [AutoML Tutorial](https://github.com/h2oai/h2o-tutorials/tree/master/h2o-world-2017/automl): R and Python notebooks
# * Intro to AutoML + Hands-on Lab: [1 hour video](https://www.youtube.com/watch?v=42Oo8TOl85I), [slides](https://www.slideshare.net/0xdata/intro-to-automl-handson-lab-erin-ledell-machine-learning-scientist-h2oai)
# * Scalable Automatic Machine Learning in H2O: [1 hour video](https://www.youtube.com/watch?v=j6rqrEYQNdo), [slides](https://www.slideshare.net/0xdata/scalable-automatic-machine-learning-in-h2o-89130971)
# * [H2O for GPU](https://www.h2o.ai/products/h2o4gpu/) (H2O4GPU)
