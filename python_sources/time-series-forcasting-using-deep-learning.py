#!/usr/bin/env python
# coding: utf-8

# # Time Series Forecasting using LSTMs
# ![image.png](attachment:image.png)
# 
# LTFS receives a lot of requests for its various finance offerings that include housing loan, two-wheeler loan, real estate financing and micro loans. The number of applications received is something that varies a lot with season. Going through these applications is a manual process and is tedious. Accurately forecasting the number of cases received can help with resource and manpower management resulting into quick response on applications and more efficient processing.
# 
# Our task is to forecast daily cases for next 3 months for 2 different business segments aggregated at the country level keeping in consideration the following major Indian festivals (inclusive but not exhaustive list): Diwali, Dussehra, Ganesh Chaturthi, Navratri, Holi etc.

# In[ ]:


#Libraries to be imported
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import warnings
from sklearn.utils import check_array 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)


# In[ ]:


df_sample_sub = pd.read_csv("../input/ltfs-2020/sample_submission_IIzFVsf.csv")
df_test = pd.read_csv("../input/ltfs-2020/test_1eLl9Yf.csv")
df_train = pd.read_csv("../input/ltfs-2020/train_fwYjLYX.csv")


# In[ ]:


display(df_train.info())
display(df_train.head())
display(df_train.describe())


# In[ ]:


print('Minimum date from training set: {}'.format(pd.to_datetime(df_train.application_date.min()).date()))
print('Maximum date from training set: {}'.format(pd.to_datetime(df_train.application_date.max()).date()))


# In[ ]:


max_date_train = pd.to_datetime(df_train.application_date.max()).date()
max_date_test = pd.to_datetime(df_test.application_date.max()).date()
lag_size = (max_date_test - max_date_train).days
print('Maximum date from training set: {}'.format(max_date_train))
print('Maximum date from test set: {}'.format(max_date_test))
print('Forecast Lag: {}'.format(lag_size))


# # EDA
# In this section I have performed some basic exploratory data analysis on the training dataset using plotly library. The train data has been provided in the following way:
# - For business segment 1, historical data has been made available at branch ID level
# - For business segment 2, historical data has been made available at State level.

# In[ ]:


daily_cases_1 = df_train[df_train['segment'] == 1].groupby(['branch_id','state','zone','application_date'], as_index = False)['case_count'].sum()
daily_cases_2 = df_train[df_train['segment'] == 2].groupby(['state','application_date'], as_index = False)['case_count'].sum()


# In[ ]:


daily_cases_1_sc = []
for state in daily_cases_1['state'].unique():
    current_daily_cases_1 = daily_cases_1[daily_cases_1['state'] == state]
    daily_cases_1_sc.append(go.Scatter(x=current_daily_cases_1['application_date'], y=current_daily_cases_1['case_count'], name=('%s' % state)))

layout = go.Layout(title='Daily Case Count - Segment 1', xaxis=dict(title='Date'), yaxis=dict(title='Case Count'))
fig = go.Figure(data=daily_cases_1_sc, layout=layout)
iplot(fig)


# In[ ]:


daily_cases_2_sc = []
for state in daily_cases_2['state'].unique():
    current_daily_cases_2 = daily_cases_2[daily_cases_2['state'] == state]
    daily_cases_2_sc.append(go.Scatter(x=current_daily_cases_2['application_date'], y=current_daily_cases_2['case_count'], name=('%s' % state)))

layout = go.Layout(title='Daily Case Count - Segment 2', xaxis=dict(title='Date'), yaxis=dict(title='Case Count'))
fig = go.Figure(data=daily_cases_2_sc, layout=layout)
iplot(fig)


# > Plots for both the business segments: 1 & 2, have a seasonal component but no trend component.

# # Data Preprocessing
# Rearrange the dataset so that we can apply shift methods. Since, the prediction is to be done on country level, we just need to group by the data on date and segment and get rid of all other columns(state, branch_id, e.t.c). Also, the aggregation applied is sum so that we get the total number of cases throughout the country for that particular day and segment.

# In[ ]:


df_train['application_date'] = pd.to_datetime(df_train['application_date'])
df_train = df_train.sort_values('application_date').groupby(['application_date','segment'], as_index=False)
df_train = df_train.agg({'case_count':['sum']})
df_train.columns = ['application_date','segment', 'case_count']
df_train.head()


# ## Convert to a Supervised Learning problem
# This seems to be a *Univariate Time series forecasting* problem. Next, we need to be able to frame the univariate series of observations as a supervised learning problem so that we can train neural network models. A supervised learning framing of a series means that the data needs to be split into multiple examples that the model learn from and generalize across.Each sample must have both an input component and an output component.
# 
# The input component(window) will be some number of prior observations, such as 30 days or time steps. The output component will be the total case counts in the next day because we are interested in developing a model to make one-step forecasts.

# In[ ]:


def series_to_supervised(data, window=1, lag=1, dropnan = True):
    cols, names = list(), list()
    #Input Sequence (t-n, ... t-1)
    for i in range(window-1,0,-1):
        cols.append(data.shift(i))
        names+=[('%s(t-%d)' % (col,i)) for col in data.columns]
    #Current Timestamp (t=0)
    cols.append(data)
    names+=[('%s(t)' % (col)) for col in data.columns]
    #Target Timestamp (t=lag)
    cols.append(data.shift(-lag))
    names+=[('%s(t+%d)' %  (col,lag)) for col in data.columns]
    #Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[ ]:


df_train_1 = df_train[df_train['segment'] == 1] #Segment 1 records
df_train_2 = df_train[df_train['segment'] == 2] #Segment 2 records


# Here's a working example of the series_to_supervised() method. It will take in the counts of last 30 days as input <t-29, t-28, ..., t> and give the output for (t+1)th day.

# In[ ]:


window = 30 #use the last 30 days
lag = 1 #predict the next day
series = series_to_supervised(df_train_1.drop(['application_date','segment'], axis=1), window=window, lag=lag)
series.head()


# Below are the first 31 values of case count column. In the above table it can be seen that the first row values are exactly similar. So, in a way we have rearranged the data so that the model learns to take in the input of the first 30 days and returns the output i.e for the 31st day. It can be taken as the label for the supervised learning problem.

# In[ ]:


df_train_1.case_count[:31]


# ## Train-test split
# The train_test_split() function below will split the series taking the raw observations and the number of observations to use in the test set as arguments

# In[ ]:


def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


# ## Evaluation Metric
# I will be using Mean Absolute Percentage Error(MAPE) as my metric for evaluation.
# ![image.png](attachment:image.png)
# Where At is the actual value and Ft is the forecast value.

# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ## Training the Model
# The model_fit() function for fitting an LSTM model is provided below.
# 
# The model expects a list of four model hyperparameters; they are:
# 
# - **n_input**: The number of lag observations to use as input to the model.
# - **n_nodes**: The number of LSTM units to use in the hidden layer.
# - **n_epochs**: The number of times to expose the model to the whole training dataset.
# - **n_batch**: The number of samples within an epoch after which the weights are updated.
# A single input must have the three-dimensional structure of samples, timesteps, and features, which in this case we only have 1 sample and 1 feature: [1, n_input, 1].

# In[ ]:


def model_fit(train, config):
    # unpack config
    n_input, n_nodes, n_epochs, n_batch = config
    df = series_to_supervised(train, window=n_input)
    data = df.to_numpy()
    train_x, train_y = data[:, :-1], data[:, -1]
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# ## Predicting the next day case count
# The model_predict() function takes in the trained model, the history(last 30 days case counts in our case) and the configuration as arguments and returns the next day case count.

# In[ ]:


def model_predict(model, history, config):
    # unpack config
    window, _, _, _ = config
    x_input = np.array(history[-window:]).reshape((1, window, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


# ## Walk Forward Validation
# Time series forecasting models can be evaluated on a test set using walk-forward validation.
# 
# Walk-forward validation is an approach where the model makes a forecast for each observation in the test dataset one at a time. After each forecast is made for a time step in the test dataset, the true observation for the forecast is added to the test dataset and made available to the model.
# 
# First, the dataset is split into train and test sets. We will call the train_test_split() function to perform this split and pass in the pre-specified number of observations to use as the test data.
# 
# A model will be fit once on the training dataset for a given configuration. Each time step of the test dataset is enumerated. A prediction is made using the fit model.
# 
# The prediction is added to a list of predictions and the true observation from the test set is added to a list of observations that was seeded with all observations from the training dataset. This list is built up during each step in the walk-forward validation, allowing the model to make a one-step prediction using the most recent history.
# 
# All of the predictions can then be compared to the true values in the test set and an error measure calculated.
# 
# We will calculate the mean absolute percentage error, or MAPE, between predictions and the true values.

# In[ ]:


#walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # fit model
    model = model_fit(train, cfg)
    # seed history with training dataset
    history = [x for x in train.to_numpy()]
    test = test.to_numpy()
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = model_predict(model, history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = mean_absolute_percentage_error(test, predictions)
    print(' > %.3f' % error)
    return error


# ## Repeat Evaluate
# Neural network models are stochastic.
# 
# This means that, given the same model configuration and the same training dataset, a different internal set of weights will result each time the model is trained that will in turn have a different performance.
# 
# This is a benefit, allowing the model to be adaptive and find high performing configurations to complex problems.
# 
# It is also a problem when evaluating the performance of a model and in choosing a final model to use to make predictions.
# 
# To address model evaluation, we will evaluate a model configuration multiple times via walk-forward validation and report the error as the average error across each evaluation.
# 
# This is not always possible for large neural networks and may only make sense for small networks that can be fit in minutes or hours.
# 
# The repeat_evaluate() function below implements this and allows the number of repeats to be specified as an optional parameter that defaults to 30 and returns a list of model performance scores: in this case, MAPE values.

# In[ ]:


# repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats=30):
    # fit and evaluate the model n times
    scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
    return scores


# ## Summarize Scores
# Finally, we need to summarize the performance of a model from the multiple repeats.
# 
# We will summarize the performance first using summary statistics, specifically the mean and the standard deviation.
# 
# We will also plot the distribution of model performance scores using a box and whisker plot to help get an idea of the spread of performance.
# 
# The summarize_scores() function below implements this, taking the name of the model that was evaluated and the list of scores from each repeated evaluation, printing the summary and showing a plot.

# In[ ]:


# summarize model performance
def summarize_scores(name, scores):
    # print a summary
    scores_m, score_std = np.mean(scores), np.std(scores)
    print('%s: %.3f MAPE (+/- %.3f)' % (name, scores_m, score_std))
    # box and whisker plot
    plt.boxplot(scores)
    plt.show()


# In[ ]:


print('Date Range for Segment 1: {} days'.format((df_train_1.application_date.max() - df_train_1.application_date.min()).days))
print('Date Range for Segment 2: {} days'.format((df_train_2.application_date.max() - df_train_2.application_date.min()).days))


# In[ ]:


config = [30, 50, 100, 100]


# In[ ]:


#Training segment 1 data
n_test = 225
df_1 = df_train_1.drop(['segment','application_date'], axis=1)
scores = repeat_evaluate(df_1, config, n_test)
summarize_scores('LSTM', scores)


# In[ ]:


#Training segment 2 data
n_test = 243
df_2 = df_train_2.drop(['segment','application_date'], axis=1)
scores = repeat_evaluate(df_2, config, n_test)
summarize_scores('LSTM', scores)


# In[ ]:


def train_model(data, config):
    n_input, n_nodes, n_epochs, n_batch = config
    df = series_to_supervised(data, window=n_input)
    data = df.to_numpy()
    train_x, train_y = data[:, :-1], data[:, -1]
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(n_nodes, activation='relu', input_shape=(window, 1)))
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# In[ ]:


def time_series_forecasting(train, test, segment, config):
    #Drop the unwanted columns
    df = train.drop(['segment','application_date'], axis=1)
    #Get the window
    window = config[0]
    #Train the model
    model = train_model(df, config)
    #Define the history to be taken into consideration for prediction
    history = [x for x in df.to_numpy()]
    #Get the records of the specified segment
    test_seg = test[test['segment'] == segment]
    #Define an empty case_count column to be inserted later on
    cases = pd.Series([])
    #One by one do prediction and append it to history and the series.
    for i in range(test.shape[0]):
        x_input = np.array(history[-window:]).reshape((1, window, 1))
        y = model.predict(x_input, verbose=0)
        history.append(y[0])
        cases[i] = round((y[0][0]),0) #Since number of cases are supposed to be integer
    #Add the calculated column to the dataset
    test_seg.insert(loc=3, column='case_count', value=cases)
    return test_seg


# In[ ]:


test1 = time_series_forecasting(df_train_1, df_test, 1, config)
test2 = time_series_forecasting(df_train_2, df_test, 2, config)


# In[ ]:


submit = pd.concat([test1, test2], ignore_index=True)


# In[ ]:


submit.to_csv('csv_to_submit.csv', index = False)


# ## This submission was done for a competition. I will keep on additing more models and even using Hyperparameter tuning to get better predictions on the test set. Please upvote if you liked the notebook.
