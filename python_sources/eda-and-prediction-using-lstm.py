#!/usr/bin/env python
# coding: utf-8

# # Stock Analysis

# ## Table of Contents
# 
# 1. [Introduction](#intro)
# 2. [Load libraries and global variables](#libraries)
# 3. [Read Data](#read_data)
# 4. [Data Overview](#data_overview)
# 5. [Data Cleaning](#data_cleaning)
# 6. [Feature Engineering](#feature_engineering)
# 7. [Time Series Analysis](#time_series_analysis)
# 8. [Corelation Analysis](#corelation)
# 9. [Data Modelling](#data_modelling)
# 10. [Model Selection](#model_selection)
# 11. [Fetch Data](#fetch_data)
# 12. [Feature Scaling](#feature_scaling)
# 13. [Train-test timestep data creation](#train_test_split)
# 14. [Forecasting](#forecasting)
# 15. [Deep Neural Nets](#dl)
# 16. [Prediction Analysis](#prediction_analysis)
# 17. [Conclusion](#conclusion)

# <a id='intro'></a>
# ## Introduction

# This notebook analyzes content present in the `DIJA 30 Stock Time Series` dataset. It provides detailed insight into the stock trends and also includes a guide to train and fit a LSTM model for stock price prediction.

# <a id='libraries'></a>
# ## Load libraries and set global options

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pd.options.display.float_format = '{:.2f}'.format
sns.set(rc={'figure.figsize':(20, 20)})


# In[3]:


import sys
print("Python version: {}". format(sys.version))

import pandas as pd 
print("pandas version: {}". format(pd.__version__))

import matplotlib 
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np 
print("NumPy version: {}". format(np.__version__))

import scipy as sp 
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display 
print("IPython version: {}". format(IPython.__version__)) 

import sklearn 
print("scikit-learn version: {}". format(sklearn.__version__))

import keras
print("keras version: {}".format(keras.__version__))

import tensorflow as tf
print("tensorflow version: {}".format(tf.__version__))


# <a id='read_data'></a>
# ## Read Data

# In[5]:


df = pd.read_csv('../input/all_stocks_2006-01-01_to_2018-01-01.csv')


# <a id='data_overview'></a>
# ## Data Overview

# In[6]:


df.head()


# From the website, we find the following information about the columns:
# 
# - `Date` - Date for which the price is given
# - `Open` - Price of the stock at market open (In USD)
# - `High` - Highest price reached in the day
# - `Low` - Lowest price reached in the day
# - `Close` - Closing price for the day
# - `Volume` - Number of shares traded
# `- `Name` - the stock's ticker name

# Further, the author has mentioned that the data has been collected using the `pandas_datareader` package which fetches data from Google Finance API. This could be a cause for concern as the API has long been deprecated.

# In[7]:


df = pd.read_csv('../input/all_stocks_2006-01-01_to_2018-01-01.csv', parse_dates=['Date'])


# In[8]:


df.info()


# In[9]:


df.Date = pd.to_datetime(df.Date)


# In[10]:


df.describe()


# The dataset has some missing values. We will analyze this and see how to fix it.

# In[11]:


df.isnull().sum()


# The `Open` column has the maximum number of null values. Let's find the rows for which the values are missing.

# In[12]:


df[df.Open.isnull()]


# Interesting! The data is missing only for 31 July, 2017. This could be because:
# - The API had an unexpected error while fetching the data.
# - The data for this day does not exist in the source.

# Let's check the number of `business days` for which the records as missing.

# In[13]:


rng = pd.date_range(start='2006-01-01', end='2018-01-01', freq='B')
rng[~rng.isin(df.Date.unique())]


# There are about 111 days for which the stock price data is missing. This could lead to potential problems with the analysis.

# In[14]:


df.groupby('Name').count().sort_values('Date', ascending=False)['Date']


# In[15]:


gdf = df[df.Name == 'AABA']
cdf = df[df.Name == 'CAT']


# In[17]:


cdf[~cdf.Date.isin(gdf.Date)]


# Some of the companies(Google, Microsoft, etc.) don't have an entry for the date 2010-04-01.

# Let's check if all the listed companies have an entry on each date.

# In[18]:


# Total number of companies
df.Name.unique().size


# In[19]:


df.groupby('Date').Name.unique().apply(len)


# This confirms that each company had a stock price entry on each day.

# <a id='data_cleaning'></a>
# ## Data Cleaning

# Let us first fill in the null values on date 31 july, 2017 with the values from the previous day(i.e 28th July, 2017)

# In[20]:


df.set_index('Date', inplace=True)

#Backfill `Open` column
values = np.where(df['2017-07-31']['Open'].isnull(), df['2017-07-28']['Open'], df['2017-07-31']['Open'])
df['2017-07-31']= df['2017-07-31'].assign(Open=values.tolist())

values = np.where(df['2017-07-31']['Close'].isnull(), df['2017-07-28']['Close'], df['2017-07-31']['Close'])
df['2017-07-31']= df['2017-07-31'].assign(Close=values.tolist())

values = np.where(df['2017-07-31']['High'].isnull(), df['2017-07-28']['High'], df['2017-07-31']['High'])
df['2017-07-31']= df['2017-07-31'].assign(High=values.tolist())

values = np.where(df['2017-07-31']['Low'].isnull(), df['2017-07-28']['Low'], df['2017-07-31']['Low'])
df['2017-07-31']= df['2017-07-31'].assign(Low=values.tolist())

df.reset_index(inplace=True)


# In[21]:


df[df.Date == '2017-07-31']


# We can confirm that the backfill has worked as expected.

# Simlarly, we noticed that 8 of the 31 stocks have missing data on 1st April, 2014. As done before, we will use the stock prices of the previous day to fill the data.

# In[22]:


missing_data_stocks = ['CSCO','AMZN','INTC','AAPL','MSFT','MRK','GOOGL', 'AABA']


# In[23]:


columns = df.columns.values


# In[24]:


for stock in missing_data_stocks:
    tdf = df[(df.Name == stock) & (df.Date == '2014-03-28')].copy()
    tdf.Date = '2014-04-01'
    pd.concat([df, tdf])
print("Complete")


# Let's check if the backfill worked as expected.

# In[25]:


df[(df.Name == 'CSCO') & (df.Date == '2014-04-01')]


# Awesome! The backfill has worked for that particular day

# Finally, there is just one more null record. We will drop that record.

# In[26]:


df[df.Open.isnull()]


# In[27]:


df = df[~((df.Date == '2012-08-01') & (df.Name == 'DIS'))]


# Let's perform a quick sanity check for null values again.

# In[28]:


df.isnull().sum()


# We have dealt with all the null values in the dataset.

# <a id='feature_engineering'></a>
# ## Feature Engineering 

# Since we have four values of stock price for each day, let's create a feature called `Price` which is the average of all these values.

# In[29]:


values = (df['High'] + df['Low'] + df['Open'] + df['Close'])/4
df = df.assign(Price=values)


# In[30]:


df.head()


# In[31]:


df.Price.describe()


# We can see that 75% of the stocks have a price of under 94$, indicating that the stock market is mostly dominated by the bigger companies.

# Let's go one step further and compute the daily growth of the stock prices compared to day 1 of the prices(i.e compute cumalative compound growth)

# In[32]:


stock_names = df.Name.unique()


# In[33]:


day_prices = df[df.Date == df.Date.min()].Price


# In[34]:


price_mapping = {n : c for n, c in zip(stock_names, day_prices)}


# In[35]:


base_mapping = np.array(list(map(lambda x : price_mapping[x], df['Name'].values)))


# In[36]:


df['Growth'] = df['Price'] / base_mapping - 1


# In[37]:


df.Growth.describe()


# **Inferences**
# 
# Wow! The worst performing company had a decline of 81% in their shares compared to their first ever opening price and the best company had a whopping 2439% increase in their share price. (_Hint_: EC2 instances)

# <a id='time_series_analysis'></a>
# ## Time Series Analysis

# Let's find out the top 5 best and worst performing stocks!

# In[38]:


sample_dates = pd.date_range(start='2006-01-01', end='2018-01-01', freq='B')


# In[39]:


year_end_dates = sample_dates[sample_dates.is_year_end]


# In[40]:


year_end_dates


# In[41]:


worst_stocks = df[df.Date == df.Date.max()].sort_values('Growth').head(5)


# In[42]:


best_stocks = df[df.Date == df.Date.max()].sort_values('Growth', ascending=False).head(5)


# In[43]:


ws = worst_stocks.Name.values


# In[44]:


bs = best_stocks.Name.values


# In[45]:


tdf = df.copy()


# In[46]:


tdf = df.set_index('Date')


# In[47]:


tdf[tdf.Name.isin(ws)].groupby('Name').Growth.plot(title='Historical trend of worst 5 stocks of 2017', legend=True)


# In[48]:


tdf[tdf.Name.isin(bs)].groupby('Name').Growth.plot(title='Historical trend of best 5 stocks of 2017', legend=True)


# In[49]:


worst_stocks


# In[50]:


best_stocks


# ** Question: How much would an investment of 1USD in Google in 2006 increase by in 2017?**

# According to the above information, an investment of 1 USD in Google in 2006 would have increased the amount to 393USD in 2017. However, the same investment in General Electric Co would have decreased the amount of 0.49USD.
# This information could help us create an ideal portfolio of stocks to maximize profit.

# <a id='corelation'></a>
# ## Corelation Analysis

# Let us now try to find some corelation between the growth vs time of each stock in the dataset

# In[51]:


corr = df.pivot('Date', 'Name', 'Growth').corr()
sns.heatmap(corr)


# Although we can see some positive and negative corelations, the graph above is very dense. Let's us just focus on high positive and high negative corelations.

# In[52]:


def unique_corelations(indices):
    mapping = {}
    for record in indices:
        (stock_a, stock_b) = record
        value_list = mapping.get(stock_a)
        if value_list:
            if stock_b not in value_list:
                value_list.append(stock_b)
                mapping.update({stock_a: value_list})
        else:
            mapping.update({stock_a: [stock_b]})

    return mapping

def filter_corelations_positive(corr, threshold=0.9):
    indices = np.where(corr > threshold)
    indices = [(corr.index[x], corr.columns[y]) for x, y in zip(*indices)
                                        if x != y and x < y]
    mapping = unique_corelations(indices)
    return mapping
    
def filter_corelations_negative(corr, threshold=-0.8):
    indices = np.where(corr < threshold)
    indices = [(corr.index[x], corr.columns[y]) for x, y in zip(*indices)
                                        if x != y and x < y]
    mapping = unique_corelations(indices)
    return mapping


# In[53]:


filter_corelations_positive(corr, threshold=0.95)


# In[54]:


filter_corelations_negative(corr, -0.1)


# From the above results, we can note the following:-
# - There is a **Strong Positive** corelation in the stock growth of GOOGL with MSFT, NKE, etc.
# - There is a **Weak Negative** corelation in the stock growth of GE with IBM and MCD.

# Let us try to forecast the prices of the Google Stock(GOOGL) and measure how close we came to the actual value.

# <a id='data_modelling'></a>
# ## Data Modelling

# In[55]:


google_df = df[df.Name == 'GOOGL']


# In[56]:


gdf = google_df[['Date', 'Price']].sort_values('Date')


# <a id='model_selection'></a>
# ## Model Selection

# As this is a time-series problem, we can use one of the following models to solve it:
# - ARIMA/ARMA: Auto-Regressive Moving Average models are a class of model that captures a suite of different standard temporal structures in time series data.
# - LSTM: Long-Short-Term_memory networks are a form of Recurrent Neural Networks. Few advantages of neural nets are:
#     - Neural networks can model any non-linear function
#     - Neural networks give good results without much parameter tuning
# 

# Hence, We will choose LSTM's as our model for forecasting stock prices.

# We will try to predict the `Price` of the Google based on the previous 60 values(i.e stock prices on the previous 30 days) 

# <a id='fetch_data'></a>
# ## Fetch Data

# We will only use the `Price` column to forecast the future stock price for simplicity. We will ignore all of the other columns.

# In[57]:


training_set = gdf[gdf.Date.dt.year != 2017].Price.values


# In[58]:


test_set =  gdf[gdf.Date.dt.year == 2017].Price.values


# In[59]:


print("Training set size: ",training_set.size)
print("Test set size: ", test_set.size)


# <a id='feature_scaling'></a>
# ## Feature Scaling

# As the amount of stock prices vary by a huge margin, we will scale the prices to be in the 0-1 range

# In[60]:


from sklearn.preprocessing import MinMaxScaler


# In[61]:


scaler = MinMaxScaler()


# In[62]:


training_set_scaled = scaler.fit_transform(training_set.reshape(-1, 1))


# <a id='train_test_split'></a>
# ## Train and Test timestep data creation

# For training, we will use previous 30 stock values to predict the stock price at time t. For this, we have to create our training and test set.

# In[63]:


def create_train_data(training_set_scaled):
    X_train, y_train = [], []
    for i in range(30, training_set_scaled.size):
        X_train.append(training_set_scaled[i-30: i])
        y_train.append(training_set_scaled[i])
    # Converting list to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train


# In[64]:


X_train, y_train = create_train_data(training_set_scaled)


# Similarly, we'll create our test data set. 

# In[65]:


def create_test_data():
    X_test = []
    inputs = gdf[len(gdf) - len(test_set) - 30:].Price.values
    inputs = scaler.transform(inputs.reshape(-1, 1))
    for i in range(30, test_set.size+30): # Range of the number of values in the training dataset
        X_test.append(inputs[i - 30: i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test


# In[66]:


X_test = create_test_data()


# In[67]:


X_test.shape


# <a id='forecasting'></a>
# ## Forecasting

# Let us first start with a very simple model. We will use a single LSTM layer.

# In[68]:


from keras.models import Sequential
from keras.layers import Dense, LSTM


# In[69]:


def create_simple_model():
    model = Sequential()
    model.add(LSTM(units = 10, return_sequences = False, input_shape = (X_train.shape[1], 1)))
    model.add(Dense(units = 1))
    return model


# We now need to pick the optimizer for our model and a function to measure how well the model is doing i.e loss. We will pick RMSE as the loss function and Sigmoid Gradient Descent(SGD) as our optimizer with default learning rate(i.e 0.01)

# In[70]:


def compile_and_run(model, epochs=50, batch_size=64):
    model.compile(metrics=['accuracy'], optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=3)
    return history


# In[71]:


def plot_metrics(history):
    metrics_df = pd.DataFrame(data={"loss": history.history['loss']})
    metrics_df.plot()


# In[72]:


simple_model = create_simple_model()
history = compile_and_run(simple_model, epochs=20)


# In[74]:


plot_metrics(history)


# Let us now use the model on the test set.

# In[75]:


def make_predictions(X_test, model):
    y_pred = model.predict(X_test)
    final_predictions = scaler.inverse_transform(y_pred)
    fp = np.ndarray.flatten(final_predictions)
    ap = np.ndarray.flatten(test_set)
    pdf = pd.DataFrame(data={'Actual': ap, 'Predicted': fp})
    ax = pdf.plot()


# In[76]:


make_predictions(X_test, simple_model)


# As we can see from the image above, although our model seems to have found some trend in the prices, it it _very_ far away from the actual stock value. Hence, this model cannot be used in a production environment.
# 
# **Question**: Can we improve this model?
# 
# Lets find out.

# <a id='dl'></a>
# ## Deep Neural Nets

# As always, here is the solution to our problem

# ![DL](https://s14-eu5.ixquick.com/cgi-bin/serveimage?url=https%3A%2F%2Fmemegenerator.net%2Fimg%2Finstances%2F49099937%2Fwe-need-to-go-deeper.jpg&sp=820b593abca83c69cfd8fe7944c66fce)

# Deep neural nets can capture trends over a largely spread dataset and could improve our model.
# For this problem of forecasting, we will use a stacked LSTM(i.e multiple LSTM layers instead of 1). 
# Further, we will also increase the number of units per LSTM cell to 50.

# In[87]:


def create_dl_model():
    model = Sequential()

    # Adding the first LSTM layer
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

    # Adding a second LSTM layer
    model.add(LSTM(units = 50, return_sequences = True))
    
    # Adding a third LSTM layer
    model.add(LSTM(units = 50, return_sequences = True))

    # Adding a fourth LSTM layer
    model.add(LSTM(units = 50))

    # Adding the output layer
    model.add(Dense(units = 1))
    return model


# In[81]:


dl_model = create_dl_model()


# In[82]:


dl_model.summary()


# In[83]:


history = compile_and_run(dl_model, epochs=20)


# In[85]:


plot_metrics(history)


# As expected, we see that the overall loss of the model reducing until a minima is reached. After that, the loss just stays constant.

# <a id='prediction_analysis'></a>
# ## Prediction Analysis

# Let us try our deep neural net on the test set.

# In[86]:


make_predictions(X_test, dl_model)


# As we can see, the deep neural net is a significant boost from our simple model. It has determined that the stock price is indeed related to the previous stock prices, and surprisngly, it is very accurate, so much so that at various points it is actually equal to the exact value of the stock on that day!

# We could further improve this model by:
# 
# - **Deeper network** - Deep Neural Networks can learn relationships between data points seperated by a large time frame.
# - **Dropout** - Adding a dropout layer would help prevent overfitting and stabalize the loss curves, which could give a better generalized model.
# - **Training for longer** - Deep neural nets work better when they are trained for longer tend to work better. However, we must be careful not to overfit.
# - **Adaptive Optimizers** - We could also use the Adam optimizer instead of SGD, which incorporates adaptive learning rates based on momentum. Adaptive optimizers should work better than SGD for this problem.
# - **Hyper-parameter tuning** - Tuning the hyper-parameters is one of the simplest ways in which we can improve the network. We could use SGD with Restarts(part of the Fast.ai library) to help find the optimal learning rate, experiment with the number of units in each LSTM layer, etc.

# <a id='conclusion'></a>
# ## Conclusion

# In this notebook, we have acheived the following:
# - Successfully analyzed the trends present in stock market prices of various companies and concluded that there is indeed a temporal relationship between these prices. 
# - Devised a model capable of reliably forecasting the stock prices of a company, thereby providing a tool to maximize profits.
# - Discussed possible implementation details to deploy such a model at scale and make it available to a larger number of people.

# Please let me know about the areas where the solution could be improved. As always, please upvote the kernel if you find it useful. Cheers! :)
