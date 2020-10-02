#!/usr/bin/env python
# coding: utf-8

# # Data manipulation

# In[ ]:


# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings("ignore")
import time

# Listing files
DIR = '/kaggle/input/covid19-global-forecasting-week-5/'
import os
for dirname, _, filenames in os.walk(DIR):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Loading data
df_train   = pd.read_csv(DIR + 'train.csv')
df_test    = pd.read_csv(DIR + 'test.csv')
submission = pd.read_csv(DIR + 'submission.csv')
df_train.head()


# ## Making Date become a number

# In[ ]:


df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)
df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)
df_train['Date'] = df_train['Date'].apply(lambda s: time.mktime(s.timetuple()))
df_test['Date'] = df_test['Date'].apply(lambda s: time.mktime(s.timetuple()))
min_timestamp = np.min(df_train['Date'])
df_train['Date'] = df_train['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)
df_test['Date'] = df_test['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)
df_train.tail()


# ## Merging Location Info

# In[ ]:


combine_columns = ['Country_Region', 'Province_State', 'County']
df_train.loc[:, combine_columns] = df_train.loc[:, combine_columns].fillna("")
df_train['Location'] = df_train[combine_columns].apply(lambda x: '_'.join(x), axis=1)
df_test.loc[:, combine_columns] = df_test.loc[:, combine_columns].fillna("")
df_test['Location'] = df_train[combine_columns].apply(lambda x: '_'.join(x), axis=1)
df_train.drop(columns=combine_columns, inplace=True)
df_test.drop(columns=combine_columns, inplace=True)


# ## Splitting into ConfirmedCases and Fatalities sets

# In[ ]:


features_cols = ['Population', 'Date', 'Location', 'TargetValue'] # Keeping Location and TargetValue for values handling

X_train_cc = df_train.loc[df_train['Target']=='ConfirmedCases', features_cols]
X_test_cc  = df_test.loc[df_test['Target']=='ConfirmedCases', ['Population', 'Date', 'Location']]
X_train_ft = df_train.loc[df_train['Target']=='Fatalities', features_cols]
X_test_ft  = df_test.loc[df_test['Target']=='Fatalities', ['Population', 'Date', 'Location']]


# ## Fixing some weird values
# 
# In a certain location, confirmed cases and fatalities should not decrease nor be negative

# In[ ]:


locs = list(set(df_train['Location']))

def handle_weird_stuff (v):
    v[v<0]=0
    for i in range(1, len(v)):
        if v[i] < v[i-1]:
            v[i] = v[i-1]
    return v

for location in tqdm(locs):
    X_train_cc.loc[X_train_cc['Location'] == location, 'TargetValue'] = handle_weird_stuff(X_train_cc.loc[X_train_cc['Location'] == location, 'TargetValue'].values)
    X_train_ft.loc[X_train_ft['Location'] == location, 'TargetValue'] = handle_weird_stuff(X_train_ft.loc[X_train_ft['Location'] == location, 'TargetValue'].values)


# ## Finally getting target

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

y_train_cc = X_train_cc['TargetValue'].values
y_train_ft = X_train_ft['TargetValue'].values

X_train_cc.drop(columns=['Location', 'TargetValue'], inplace=True)
X_test_cc.drop(columns=['Location'], inplace=True)
X_train_ft.drop(columns=['Location', 'TargetValue'], inplace=True)
X_test_ft.drop(columns=['Location'], inplace=True)

X_train_cc = X_train_cc.values
X_train_ft = X_train_ft.values
X_test_cc  = X_test_cc.values
X_test_ft  = X_test_ft.values

cc_scale_factor = 1.5 * np.max(y_train_cc)
ft_scale_factor = 1.5 * np.max(y_train_ft)
cc_scaler = MinMaxScaler()
ft_scaler = MinMaxScaler()

X_train_cc = cc_scaler.fit_transform(X_train_cc)
X_test_cc  = cc_scaler.transform(X_test_cc)
X_train_ft = ft_scaler.fit_transform(X_train_ft)
X_test_ft  = ft_scaler.transform(X_test_ft)
y_train_cc /= cc_scale_factor
y_train_ft /= ft_scale_factor

X_cc_train, X_cc_valid, y_cc_train, y_cc_valid = train_test_split(X_train_cc, y_train_cc, test_size=0.2, random_state=42)
X_ft_train, X_ft_valid, y_ft_train, y_ft_valid = train_test_split(X_train_ft, y_train_ft, test_size=0.2, random_state=42)


# # Modeling

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from functools import partial
import tensorflow.keras.backend as K
from sklearn.metrics import mean_absolute_error

class CustomNN ():

    def __init__ (self, layers_list, loss='mse', optimizer='adam', epochs=5000, patience=25, batch_size=256, verbose=True, callbacks=[]):
        self.__verbose = verbose
        self.__model = Sequential()
        for layer in layers_list:
            self.__model.add(layer)
        self.__model.compile(optimizer, loss=loss)
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__es = EarlyStopping(monitor='loss', mode='min', verbose=self.__verbose)
        self.__callbacks = []
        self.__callbacks.append(self.__es)
        self.__loss = loss
        self.__optimizer = optimizer

    def fit (self, X_train, y_train):
        self.__history = self.__model.fit (
            X_train,
            y_train,
            epochs = self.__epochs,
            batch_size = self.__batch_size,
            verbose = self.__verbose,
            callbacks = self.__callbacks,
        )
        return self

    def predict (self, X_test):
        return self.__model.predict(X_test)

    def save_weights (self, fname):
        self.__model.save_weights(fname)

    def load_weights (self, fname):
        self.__model.load_weights(fname)
        self.__model.compile(self.__optimizer, loss=self.__loss)

    def __str__ (self):
        return self.__model.summary()

    def __repr__ (self):
        return self.__str__()

    @property
    def model (self):
        return self.__model

def pinball_loss(y_true, y_pred, tau=0.1, *args, **kwargs):
    err = y_true - y_pred
    return K.mean(K.maximum(tau * err, (tau - 1) * err), axis=-1)

def plot_learning_curves_nn (model, X_train, y_train, X_valid, y_valid):
    scores_train = []
    scores_val = []
    model.save_weights('model.h5')
    this_range = np.arange(1, len(X_train), len(X_train)/10)
    for i in tqdm(this_range):
        i = int(i)
        model.load_weights('model.h5')
        model.fit(X_train[:i], y_train[:i])
        p_train = model.predict(X_train[:i])
        p_val = model.predict(X_valid)
        scores_train.append(mean_absolute_error(y_train[:i], p_train))
        scores_val.append(mean_absolute_error(y_valid, p_val))
    plt.figure(figsize=(16,10))
    plt.title("Learning Curves")
    plt.scatter(this_range, scores_train)
    plt.scatter(this_range, scores_val)
    plt.legend(["Train", "Validation"])
    plt.xlabel("# Samples")
    plt.ylabel("SP")
    plt.show()

initial_cc_model_config = [
    Input(shape=(X_train_cc.shape[1],)),
    Dense(3, activation='linear'),
    Activation('sigmoid'),
    Dense(1, activation='linear'),
    Activation('sigmoid')
]

cc_model = CustomNN(initial_cc_model_config, loss=pinball_loss, verbose=False)
plot_learning_curves_nn(cc_model, X_cc_train, y_cc_train, X_cc_valid, y_cc_valid)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)
df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)
#df_intersection = df_test[df_test['Date'] <= np.max(df_train['Date'])]
#df_intersection


# In[ ]:


# Following the idea at
# https://www.kaggle.com/ranjithks/25-lines-of-code-results-better-score#Fill-NaN-from-State-feature
# Filling NaN states with the Country

EMPTY_VAL = "EMPTY_VAL"

def fillState(state, country):
    if state == EMPTY_VAL: return country
    return state

def replaceGeorgiaState (state, country):
    if (state == 'Georgia') and (country == 'US'):
        return 'Georgia_State'
    else:
        return state

df_train['Province_State'].fillna(EMPTY_VAL, inplace=True)
df_train['Province_State'] = df_train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
df_train['Province_State'] = df_train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : replaceGeorgiaState(x['Province_State'], x['Country_Region']), axis=1)

df_test['Province_State'].fillna(EMPTY_VAL, inplace=True)
df_test['Province_State'] = df_test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
df_test['Province_State'] = df_test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : replaceGeorgiaState(x['Province_State'], x['Country_Region']), axis=1)

# df_intersection['Province_State'].fillna(EMPTY_VAL, inplace=True)
# df_intersection['Province_State'] = df_intersection.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
# df_intersection['Province_State'] = df_intersection.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : replaceGeorgiaState(x['Province_State'], x['Country_Region']), axis=1)

df_train[df_train['Province_State'] == 'Georgia_State']


# In[ ]:


# Making Date become timestamp
df_train['Date'] = df_train['Date'].apply(lambda s: time.mktime(s.timetuple()))
df_test['Date'] = df_test['Date'].apply(lambda s: time.mktime(s.timetuple()))
# df_intersection['Date'] = df_intersection['Date'].apply(lambda s: time.mktime(s.timetuple()))

min_timestamp = np.min(df_train['Date'])
df_train['Date'] = df_train['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)
df_test['Date'] = df_test['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)
# df_intersection['Date'] = df_intersection['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)


# In[ ]:


df_train[df_train['Country_Region']=='Brazil']


# In[ ]:


# Adding validation data into the Intersection DF
states = sorted(set(df_intersection['Province_State']))
df_intersection['ConfirmedCases'] = float('NaN')
df_intersection['Fatalities'] = float('NaN')

for state in states:
    dates = sorted(set(df_intersection[df_intersection['Province_State'] == state]['Date']))
    min_date = np.min(dates)
    max_date = np.max(dates)
    idx = df_intersection[df_intersection['Province_State'] == state].index
    values = df_train[(df_train['Province_State'] == state) & (df_train['Date'] >= min_date) & (df_train['Date'] <= max_date)][['ConfirmedCases', 'Fatalities']].values
    values = pd.DataFrame(values, index = list(idx), columns=['ConfirmedCases', 'Fatalities'])
    df_intersection['ConfirmedCases'].loc[idx] = values['ConfirmedCases']
    df_intersection['Fatalities'].loc[idx] = values['Fatalities']
df_intersection


# In[ ]:


# Filtering data for public leaderboard
df_train = df_train[df_train['Date'] < np.min(df_test['Date'])]
# Check if any Province_State value on test dataset isn't on train dataset
# If nothing prints, everything is okay
for a in set(df_test['Province_State']):
    if a not in set(df_train['Province_State']):
        print (a)


# In[ ]:


# Generating features based on evolution of COVID-19
# Idea from https://www.kaggle.com/binhlc/sars-cov-2-exponential-model-week-2
print ("Generating features on evolution of COVID-19")
from tqdm import tqdm
evolution = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
def generateFeatures (state):
    should_filter = False
    train = df_train[df_train['Province_State'] == state].drop(columns=['Id'])
    test  = df_test[df_test['Province_State'] == state].drop(columns=['ForecastId'])
    y_cases = train['ConfirmedCases']
    y_fatal = train['Fatalities']
    for evo_type in ['ConfirmedCases', 'Fatalities']:
        for value in evolution:
            min_day = train[train[evo_type] >= value]['Date']
            if min_day.count() > 0:
                min_day = np.min(min_day)
                should_filter = True
            else:
                #print ("{} -> Not found min_day for {} {}".format(state, evo_type, value))
                continue
            train['{}_{}'.format(evo_type, value)] = train['Date'].apply(lambda x: x - min_day)
            test ['{}_{}'.format(evo_type, value)] = test ['Date'].apply(lambda x: x - min_day)
    train.drop(columns=['ConfirmedCases', 'Fatalities', 'Province_State', 'Country_Region'], inplace=True)
    test.drop(columns=['Province_State', 'Country_Region'], inplace=True)
    if should_filter:
        idx     = train[train['ConfirmedCases_1'] >= 0].index
        train   = train.loc[idx]
        y_cases = y_cases.loc[idx]
        y_fatal = y_fatal.loc[idx]
    return train, test, y_cases, y_fatal

dataframes = {}
states = sorted(set(df_train['Province_State']))
for state in tqdm(states):
    dataframes[state] = {}
    train, test, y_cases, y_fatal = generateFeatures(state)
    dataframes[state]['train']   = train
    dataframes[state]['test']    = test
    dataframes[state]['y_cases'] = y_cases
    dataframes[state]['y_fatal'] = y_fatal


# In[ ]:


# # Checking shapes
# state = 'Georgia'
# print (dataframes[state]['train'].shape, dataframes[state]['test'].shape, dataframes[state]['y_cases'].shape, dataframes[state]['y_fatal'].shape)
# dataframes[state]['test'].head()


# # Modeling and predicting for competition

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, BayesianRidge, Lasso
from sklearn.metrics import mean_squared_log_error
from sklearn.base import TransformerMixin
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.passive_aggressive import PassiveAggressiveRegressor
from sklearn.linear_model.theil_sen import TheilSenRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

def handle_predictions (predictions, lowest = 0):
    #predictions = np.round(predictions, 0)
    # Predictions can't be negative
    predictions[predictions < 0] = 0
    # Predictions can't decrease from greatest value on train dataset
    # predictions[predictions < lowest] = lowest
    # Predictions can't decrease over time
    for i in range(1, len(predictions)):
        if predictions[i] < predictions[i - 1]:
            predictions[i] = predictions[i - 1]
    #return predictions.astype(int)
    return predictions

def fillSubmission (state, column, values,):
    idx = df_test[df_test['Province_State'] == state].index
    values = pd.DataFrame(np.array(values), index = list(idx), columns=[column])
    submission[column].loc[idx] = values[column]
    return submission

def avg_rmsle():
    idx = df_intersection.index
    my_sub = submission.loc[idx][['ConfirmedCases', 'Fatalities']]
    cases_pred = my_sub['ConfirmedCases'].values
    fatal_pred = my_sub['Fatalities'].values
    pred = np.append(cases_pred, fatal_pred)
    cases_targ = df_intersection.loc[idx]['ConfirmedCases'].values
    fatal_targ = df_intersection.loc[idx]['Fatalities'].values
    targ = np.append(cases_targ, fatal_targ)
    score = np.sqrt(mean_squared_log_error( targ, pred ))
    return score

def make_combinations (iterable):
    from itertools import combinations
    my_combs = []
    for item in iterable.copy():
        iterable.remove(item)
        for i in range(len(iterable)):
            for comb in combinations(iterable, i+1):
                my_combs.append((item, comb))
        iterable.append(item)
    return my_combs

test_models = [
#     make_pipeline(MinMaxScaler(), PolynomialFeatures(2), LinearRegression()),                  # 0.43248400978264234
#     make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), LinearRegression()),                  # --> 0.41772679149716213
#     make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression()),                # 0.4334087200925956
#     make_pipeline(RobustScaler(), PolynomialFeatures(2), LinearRegression()),                  # 0.4401684177017516
#     make_pipeline(Normalizer(), PolynomialFeatures(2), LinearRegression()),                    # 0.5073105363889515
#     make_pipeline(QuantileTransformer(), PolynomialFeatures(2), LinearRegression()),           # 0.5436308167750011
#     make_pipeline(PowerTransformer(), PolynomialFeatures(2), LinearRegression()),              # 3.723951842476838

#     make_pipeline(MinMaxScaler(), PolynomialFeatures(2), TheilSenRegressor()),                  # 0.429714197718972
#     make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), TheilSenRegressor()),                  # --> 0.416881016597718
#     make_pipeline(StandardScaler(), PolynomialFeatures(2), TheilSenRegressor()),                # 0.46545608570380087
#     make_pipeline(RobustScaler(), PolynomialFeatures(2), TheilSenRegressor()),                  # 0.46325130998855407
#     make_pipeline(Normalizer(), PolynomialFeatures(2), TheilSenRegressor()),                    # 0.5066829122736244
#     make_pipeline(QuantileTransformer(), PolynomialFeatures(2), TheilSenRegressor()),           # 0.5436262279295473
#     make_pipeline(PowerTransformer(), PolynomialFeatures(2), TheilSenRegressor()),              # 3.7357046570144115

#     make_pipeline(MinMaxScaler(), PolynomialFeatures(2), BayesianRidge()),                  # 0.41814599178288325
#     make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), BayesianRidge()),                  # 0.41277237195607513
#     make_pipeline(StandardScaler(), PolynomialFeatures(2), BayesianRidge()),                # 0.41564703233710143
#     make_pipeline(RobustScaler(), PolynomialFeatures(2), BayesianRidge()),                  # 0.4152229761235273
#     make_pipeline(Normalizer(), PolynomialFeatures(2), BayesianRidge()),                    # --> 0.3906797240096884
#     make_pipeline(QuantileTransformer(), PolynomialFeatures(2), BayesianRidge()),           # 0.5453994559085396
#     make_pipeline(PowerTransformer(), PolynomialFeatures(2), BayesianRidge()),              # 0.4807433364820885

#     make_pipeline(MinMaxScaler(), PolynomialFeatures(2), Lasso()),                  # 0.4863083627388429
#     make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), Lasso()),                  # 0.47408909074033034
#     make_pipeline(StandardScaler(), PolynomialFeatures(2), Lasso()),                # 0.4509150256440627
#     make_pipeline(RobustScaler(), PolynomialFeatures(2), Lasso()),                  # 0.4732912500189848
#     make_pipeline(Normalizer(), PolynomialFeatures(2), Lasso()),                    # 0.53992733050457
#     make_pipeline(QuantileTransformer(), PolynomialFeatures(2), Lasso()),           # 0.5509175786196774
#     make_pipeline(PowerTransformer(), PolynomialFeatures(2), Lasso()),              # --> 0.3916916463210968

#     make_pipeline(MinMaxScaler(), PolynomialFeatures(2), LGBMRegressor()),                  # 0.5512175074492168
#     make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), LGBMRegressor()),                  # --> 0.5512154967563805
#     make_pipeline(StandardScaler(), PolynomialFeatures(2), LGBMRegressor()),                # 0.5512174701464707
#     make_pipeline(RobustScaler(), PolynomialFeatures(2), LGBMRegressor()),                  # 0.5512174701464707
#     make_pipeline(Normalizer(), PolynomialFeatures(2), LGBMRegressor()),                    # 0.5512174651618788
#     make_pipeline(QuantileTransformer(), PolynomialFeatures(2), LGBMRegressor()),           # 0.5512175074492168
#     make_pipeline(PowerTransformer(), PolynomialFeatures(2), LGBMRegressor()),              # 0.5512174729007725

#     make_pipeline(MinMaxScaler(), PolynomialFeatures(2), XGBRegressor()),                  # 0.5512174419704444
#     make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), XGBRegressor()),                  # --> 0.5512173836794546
#     make_pipeline(StandardScaler(), PolynomialFeatures(2), XGBRegressor()),                # 0.5512174424297206
#     make_pipeline(RobustScaler(), PolynomialFeatures(2), XGBRegressor()),                  # 0.5512174424297206
#     make_pipeline(Normalizer(), PolynomialFeatures(2), XGBRegressor()),                    # 0.5512174325463026
#     make_pipeline(QuantileTransformer(), PolynomialFeatures(2), XGBRegressor()),           # 0.5512174419704444
#     make_pipeline(PowerTransformer(), PolynomialFeatures(2), XGBRegressor()),              # 0.5512174424209609

]

for model in test_models:
    print (' * Model: {}'.format(model))
    for state in tqdm(states):
        try:
            train   = dataframes[state]['train']
            test    = dataframes[state]['test']
            y_cases = dataframes[state]['y_cases']
            y_fatal = dataframes[state]['y_fatal']
            model.fit(train, y_cases)
            cases = model.predict(test)
            lowest_pred = np.max(df_train[df_train['Province_State'] == state]['ConfirmedCases'].values)
            cases = handle_predictions(cases, lowest_pred)
            submission = fillSubmission (state, 'ConfirmedCases', cases)
            model.fit(train, y_fatal)
            fatal = model.predict(test)
            lowest_pred = np.max(df_train[df_train['Province_State'] == state]['Fatalities'].values)
            fatal = handle_predictions(fatal, lowest_pred)
            submission = fillSubmission (state, 'Fatalities', fatal)
        except:
            print ("Model {}\n failed to predict country {}. Will continue...".format(model, state))
    print ('   - Score: {}'.format(avg_rmsle()))


# In[ ]:


class CustomEnsemble (BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, meta_model, scaler=MaxAbsScaler(), feature_generator=None):
        self.models = models
        if scaler:
            if feature_generator:
                self.meta_model = make_pipeline(scaler, feature_generator, meta_model)
            else:
                self.meta_model = make_pipeline(scaler, meta_model)
        else:
            if feature_generator:
                self.meta_model = make_pipeline(feature_generator, meta_model)
            else:
                self.meta_model = meta_model
    def fit(self,X,y):
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            model.fit (X, y)
            predictions[:,i] = model.predict(X)
        self.meta_model.fit(predictions, y)
    def predict(self,X):
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:,i] = model.predict(X)
        return self.meta_model.predict(predictions)
    def __str__ (self):
        return "<CustomEnsemble (meta={}, models={})>".format(self.meta_model, self.models)
    def __repr__ (self):
        return self.__str__()
    
test_models = [
    make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), LinearRegression()),       # --> 0.41772679149716213
#     make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), TheilSenRegressor()),      # --> 0.416881016597718
    make_pipeline(Normalizer(), PolynomialFeatures(2), BayesianRidge()),            # --> 0.3906797240096884
    make_pipeline(PowerTransformer(), PolynomialFeatures(2), Lasso()),              # --> 0.3916916463210968
    make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), LGBMRegressor()),          # --> 0.5512154967563805
    make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), XGBRegressor()),           # --> 0.5512173836794546
]

# Version 6 best model. Data will be used to test few more models
# <CustomEnsemble (meta=Pipeline(memory=None,
#          steps=[('maxabsscaler', MaxAbsScaler(copy=True)),
#                 ('pipeline',
#                  Pipeline(memory=None,
#                           steps=[('maxabsscaler', MaxAbsScaler(copy=True)),
#                                  ('polynomialfeatures',
#                                   PolynomialFeatures(degree=2,
#                                                      include_bias=True,
#                                                      interaction_only=False,
#                                                      order='C')),
#                                  ('linearregression',
#                                   LinearRegression(copy_X=True,
#                                                    fit_intercept=True,
#                                                    n_jobs=None,
#                                                    normalize=False))],
#                           verbose=False))],
#          verbose=False), models=[Pipeline(memory=None,
#          steps=[('normalizer', Normalizer(copy=True, norm='l2')),
#                 ('polynomialfeatures',
#                  PolynomialFeatures(degree=2, include_bias=True,
#                                     interaction_only=False, order='C')),
#                 ('bayesianridge',
#                  BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, alpha_init=None,
#                                compute_score=False, copy_X=True,
#                                fit_intercept=True, lambda_1=1e-06,
#                                lambda_2=1e-06, lambda_init=None, n_iter=300,
#                                normalize=False, tol=0.001, verbose=False))],
#          verbose=False)])>

# Version 7 best model. It's the same as version 6.
# <CustomEnsemble (meta=Pipeline(memory=None,
#          steps=[('maxabsscaler', MaxAbsScaler(copy=True)),
#                 ('pipeline',
#                  Pipeline(memory=None,
#                           steps=[('maxabsscaler', MaxAbsScaler(copy=True)),
#                                  ('polynomialfeatures',
#                                   PolynomialFeatures(degree=2,
#                                                      include_bias=True,
#                                                      interaction_only=False,
#                                                      order='C')),
#                                  ('linearregression',
#                                   LinearRegression(copy_X=True,
#                                                    fit_intercept=True,
#                                                    n_jobs=None,
#                                                    normalize=False))],
#                           verbose=False))],
#          verbose=False), models=[Pipeline(memory=None,
#          steps=[('normalizer', Normalizer(copy=True, norm='l2')),
#                 ('polynomialfeatures',
#                  PolynomialFeatures(degree=2, include_bias=True,
#                                     interaction_only=False, order='C')),
#                 ('bayesianridge',
#                  BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, alpha_init=None,
#                                compute_score=False, copy_X=True,
#                                fit_intercept=True, lambda_1=1e-06,
#                                lambda_2=1e-06, lambda_init=None, n_iter=300,
#                                normalize=False, tol=0.001, verbose=False))],
#          verbose=False)])>

my_combs = make_combinations(test_models)
print ("I tested {} models =)".format(len(my_combs)))
best = 10000
results = []
with tqdm(total = len(my_combs) * len(states)) as pbar:
    for comb in my_combs:
        try:
            for state in states:
                train   = dataframes[state]['train']
                test    = dataframes[state]['test']
                y_cases = dataframes[state]['y_cases']
                y_fatal = dataframes[state]['y_fatal']
                model = CustomEnsemble(list(comb[1]), comb[0])
                model.fit(train, y_cases)
                cases = model.predict(test)
                lowest_pred = np.max(df_train[df_train['Province_State'] == state]['ConfirmedCases'].values)
                cases = handle_predictions(cases, lowest_pred)
                submission = fillSubmission (state, 'ConfirmedCases', cases)
                model.fit(train, y_fatal)
                fatal = model.predict(test)
                lowest_pred = np.max(df_train[df_train['Province_State'] == state]['Fatalities'].values)
                fatal = handle_predictions(fatal, lowest_pred)
                submission = fillSubmission (state, 'Fatalities', fatal)
                pbar.update(1)
            score = avg_rmsle()
            results.append(score)
            if (score < best):
                print ("Score {:.4f} is better than previous best. Saving...".format(score))
                best = score
                best_model = model
        except:
            print("Model {}\nfailed. Will continue now...".format(model))
# best_model = CustomEnsemble(
#     meta_model = make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), LinearRegression()),
#     models = [
#         make_pipeline(Normalizer(), PolynomialFeatures(2), BayesianRidge())
#     ]
# )
# best = 0.4330928705901224
            
print ("And the best model goes to...")
print (best_model)
print ("with score {}".format(best))


# In[ ]:


# Making predicitons using the best model for the private leaderboard
print ("Reloading data and making predictions...")
# Load raw train
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
# Handle it
df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)
EMPTY_VAL = "EMPTY_VAL"
df_train['Province_State'].fillna(EMPTY_VAL, inplace=True)
df_train['Province_State'] = df_train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
df_train['Date'] = df_train['Date'].apply(lambda s: time.mktime(s.timetuple()))
df_train['Date'] = df_train['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)
# Re-generate features and predict
dataframes = {}
states = sorted(set(df_train['Province_State']))
for state in tqdm(states):
    dataframes[state] = {}
    train, test, y_cases, y_fatal = generateFeatures(state)
    model = best_model
    model.fit(train, y_cases)
    cases = model.predict(test)
    lowest_pred = np.max(df_train[df_train['Province_State'] == state]['ConfirmedCases'].values)
    cases = handle_predictions(cases, lowest_pred)
    submission = fillSubmission (state, 'ConfirmedCases', cases)
    model.fit(train, y_fatal)
    fatal = model.predict(test)
    lowest_pred = np.max(df_train[df_train['Province_State'] == state]['Fatalities'].values)
    fatal = handle_predictions(fatal, lowest_pred)
    submission = fillSubmission (state, 'Fatalities', fatal)


# # Sanity check with random samples

# In[ ]:


from datetime import datetime, timedelta

def checkState (state):
    idx = df_test[df_test['Province_State'] == state].index
    return submission.loc[idx]

def plotStatus (states):
    if type(states) == list:
        for state in states:
            initial_date = datetime (2020, 1, 22)
            df = df_train[df_train['Province_State'] == state]
            dates_train = sorted(list(set(df['Date'])))
            dates_test  = sorted(list(set(df_test['Date'])))
            for i in range(len(dates_train)):
                dates_train[i] = initial_date + timedelta(days=dates_train[i])
            for i in range(len(dates_test)):
                dates_test[i] = initial_date + timedelta(days=dates_test[i])
            idx = df_test[df_test['Province_State'] == state].index
            plt.figure(figsize=(14,8))
            plt.title('COVID-19 cases on {}'.format(state))
            plt.xlabel('Date')
            plt.ylabel('Number')
            plt.plot(dates_train, df['ConfirmedCases'], linewidth=2, color='#ff9933')
            plt.plot(dates_test , submission['ConfirmedCases'].loc[idx], linewidth=2, color='#e67300', linestyle='dashed')
            legend = []
            legend.append('{} confirmed cases'.format(state))
            legend.append('{} predicted cases'.format(state))
            plt.legend(legend)
            plt.show()
            plt.figure(figsize=(14,8))
            plt.title('COVID-19 fatalities on {}'.format(state))
            plt.xlabel('Date')
            plt.ylabel('Number')
            plt.plot(dates_train, df['Fatalities'], linewidth=2, color='#ff9933')
            plt.plot(dates_test , submission['Fatalities'].loc[idx], linewidth=2, color='#e67300', linestyle='dashed')
            legend = []
            legend.append('{} fatalities'.format(state))
            legend.append('{} predicted fatalities'.format(state))
            plt.legend(legend)
            plt.show()
    else:
        print ("Please send me a list")

raw_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
raw_test  = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
raw_train['Date'] = pd.to_datetime(raw_train['Date'], infer_datetime_format=True)
raw_test['Date']  = pd.to_datetime(raw_test['Date'], infer_datetime_format=True)

def rmsle (state):
    idx = df_intersection[df_intersection['Province_State'] == state].index
    my_sub = submission.loc[idx][['ConfirmedCases', 'Fatalities']]
    cases_pred = my_sub['ConfirmedCases'].values
    fatal_pred = my_sub['Fatalities'].values
    cases_targ = df_intersection.loc[idx]['ConfirmedCases'].values
    fatal_targ = df_intersection.loc[idx]['Fatalities'].values
    cases = np.sqrt(mean_squared_log_error( cases_targ, cases_pred ))
    fatal = np.sqrt(mean_squared_log_error( fatal_targ, fatal_pred ))
    return cases, fatal

samples = list(df_train['Province_State'].sample(n=1))
samples.append('Brazil')
plotStatus(samples)


# # Sanity check with global data

# In[ ]:


def plotGlobalStatus ():
    legend = []
    initial_date = datetime (2020, 1, 22)
    df = df_train.groupby('Date').sum()
    df['Date'] = df.index
    test = df_test
    test['ConfirmedCases'] = submission['ConfirmedCases']
    test['Fatalities'] = submission['Fatalities']
    test = test.groupby('Date').sum()
    test['Date'] = test.index
    dates_train = sorted(list(set(df['Date'])))
    dates_test  = sorted(list(set(test['Date'])))
    for i in range(len(dates_train)):
        dates_train[i] = initial_date + timedelta(days=dates_train[i])
    for i in range(len(dates_test)):
        dates_test[i] = initial_date + timedelta(days=dates_test[i])
    plt.figure(figsize=(14,8))
    plt.title('Global COVID-19 cases')
    plt.xlabel('Date')
    plt.ylabel('Number')
    plt.plot(dates_train, df['ConfirmedCases'], linewidth=2, color='#ff9933')
    plt.plot(dates_test , test['ConfirmedCases'], linewidth=2, color='#e67300', linestyle='dashed')
    legend.append('{} confirmed cases'.format('World'))
    legend.append('{} predicted cases'.format('World'))
    plt.legend(legend)
    plt.show()
    plt.figure(figsize=(14,8))
    plt.title('Global COVID-19 fatalities')
    plt.xlabel('Date')
    plt.ylabel('Number')
    plt.plot(dates_train, df['Fatalities'], linewidth=2, color='#ff9933')
    plt.plot(dates_test , test['Fatalities'], linewidth=2, color='#e67300', linestyle='dashed')
    legend = []
    legend.append('{} fatalities'.format('World'))
    legend.append('{} predicted fatalities'.format('World'))
    plt.legend(legend)
    plt.show()

plotGlobalStatus()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# # Predicting for the future - when will this end?
# 
# I know this certainly is not accurate, but just wanted to have an idea of when and how this ends, based on the model I chose.

# In[ ]:


df_train   = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
df_test    = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)
df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)
df_test['Province_State'].fillna(EMPTY_VAL, inplace=True)
df_test['Province_State'] = df_test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
df_test['Province_State'] = df_test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : replaceGeorgiaState(x['Province_State'], x['Country_Region']), axis=1)
df_train['Date'] = df_train['Date'].apply(lambda s: time.mktime(s.timetuple()))
df_test['Date'] = df_test['Date'].apply(lambda s: time.mktime(s.timetuple()))
min_timestamp = np.min(df_train['Date'])
df_train['Date'] = df_train['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)
df_test['Date'] = df_test['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)
max_date = 113
days_to_add = 200
with tqdm(total = days_to_add * len(states)) as pbar:
    for state in states:
        for i in range(1, days_to_add + 1):
            row_df = pd.DataFrame([[-1, state, state, max_date + i]], columns = ['ForecastId', 'Country_Region', 'Province_State', 'Date'])
            df_test = pd.concat([df_test, row_df], ignore_index=True)
            pbar.update(1)


# In[ ]:


df_test['ConfirmedCases'] = -1
df_test['Fatalities'] = -1
df_test


# In[ ]:


# Making predicitons using the best model for the private leaderboard
print ("Reloading data and making predictions...")
# Load raw train
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
# Handle it
df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)
EMPTY_VAL = "EMPTY_VAL"
df_train['Province_State'].fillna(EMPTY_VAL, inplace=True)
df_train['Province_State'] = df_train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
df_train['Date'] = df_train['Date'].apply(lambda s: time.mktime(s.timetuple()))
df_train['Date'] = df_train['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)
# Re-generate features and predict
dataframes = {}
states = sorted(set(df_train['Province_State']))
for state in tqdm(states):
    idx = df_test[df_test['Province_State'] == state].index
    dataframes[state] = {}
    train, test, y_cases, y_fatal = generateFeatures(state)
    model = best_model
    model.fit(train, y_cases)
    test.drop(columns=['ConfirmedCases', 'Fatalities'], inplace=True)
    cases = model.predict(test)
    lowest_pred = np.max(df_train[df_train['Province_State'] == state]['ConfirmedCases'].values)
    cases = handle_predictions(cases, lowest_pred)
    df_test.loc[idx, 'ConfirmedCases'] = cases
    model.fit(train, y_fatal)
    fatal = model.predict(test)
    lowest_pred = np.max(df_train[df_train['Province_State'] == state]['Fatalities'].values)
    fatal = handle_predictions(fatal, lowest_pred)
    df_test.loc[idx, 'Fatalities'] = fatal


# In[ ]:


def plotStatus (states):
    if type(states) == list:
        for state in states:
            initial_date = datetime (2020, 1, 22)
            df = df_train[df_train['Province_State'] == state]
            dates_train = sorted(list(set(df['Date'])))
            dates_test  = sorted(list(set(df_test['Date'])))
            for i in range(len(dates_train)):
                dates_train[i] = initial_date + timedelta(days=dates_train[i])
            for i in range(len(dates_test)):
                dates_test[i] = initial_date + timedelta(days=dates_test[i])
            idx = df_test[df_test['Province_State'] == state].index
            plt.figure(figsize=(14,8))
            plt.title('COVID-19 cases on {}'.format(state))
            plt.xlabel('Date')
            plt.ylabel('Number')
            plt.plot(dates_train, df['ConfirmedCases'], linewidth=2, color='#ff9933')
            plt.plot(dates_test , df_test['ConfirmedCases'].loc[idx], linewidth=2, color='#e67300', linestyle='dashed')
            legend = []
            legend.append('{} confirmed cases'.format(state))
            legend.append('{} predicted cases'.format(state))
            plt.legend(legend)
            plt.show()
            plt.figure(figsize=(14,8))
            plt.title('COVID-19 fatalities on {}'.format(state))
            plt.xlabel('Date')
            plt.ylabel('Number')
            plt.plot(dates_train, df['Fatalities'], linewidth=2, color='#ff9933')
            plt.plot(dates_test , df_test['Fatalities'].loc[idx], linewidth=2, color='#e67300', linestyle='dashed')
            legend = []
            legend.append('{} fatalities'.format(state))
            legend.append('{} predicted fatalities'.format(state))
            plt.legend(legend)
            plt.show()
    else:
        print ("Please send me a list")

plotStatus(['Brazil'])


# In[ ]:


from datetime import datetime, timedelta
def plotGlobalStatus ():
    legend = []
    initial_date = datetime (2020, 1, 22)
    df = df_train.groupby('Date').sum()
    df['Date'] = df.index
    test = df_test
    test['ConfirmedCases'] = df_test['ConfirmedCases']
    test['Fatalities'] = df_test['Fatalities']
    test = test.groupby('Date').sum()
    test['Date'] = test.index
    dates_train = sorted(list(set(df['Date'])))
    dates_test  = sorted(list(set(test['Date'])))
    for i in range(len(dates_train)):
        dates_train[i] = initial_date + timedelta(days=dates_train[i])
    for i in range(len(dates_test)):
        dates_test[i] = initial_date + timedelta(days=dates_test[i])
    plt.figure(figsize=(14,8))
    plt.title('Global COVID-19 cases')
    plt.xlabel('Date')
    plt.ylabel('Number')
    plt.plot(dates_train, df['ConfirmedCases'], linewidth=2, color='#ff9933')
    plt.plot(dates_test , test['ConfirmedCases'], linewidth=2, color='#e67300', linestyle='dashed')
    legend.append('{} confirmed cases'.format('World'))
    legend.append('{} predicted cases'.format('World'))
    plt.legend(legend)
    plt.show()
    plt.figure(figsize=(14,8))
    plt.title('Global COVID-19 fatalities')
    plt.xlabel('Date')
    plt.ylabel('Number')
    plt.plot(dates_train, df['Fatalities'], linewidth=2, color='#ff9933')
    plt.plot(dates_test , test['Fatalities'], linewidth=2, color='#e67300', linestyle='dashed')
    legend = []
    legend.append('{} fatalities'.format('World'))
    legend.append('{} predicted fatalities'.format('World'))
    plt.legend(legend)
    plt.show()

plotGlobalStatus()


# In[ ]:


initial_date = datetime (2020, 1, 22)
df_test['Date'] = df_test['Date'].apply(lambda x: initial_date + timedelta(days=x))
df_test.drop(columns=['ForecastId'], inplace=True)
df_test = df_test[['Date', 'Country_Region', 'Province_State', 'ConfirmedCases', 'Fatalities']]


# In[ ]:


df_test.to_csv('future.csv', index=False)


# In[ ]:




