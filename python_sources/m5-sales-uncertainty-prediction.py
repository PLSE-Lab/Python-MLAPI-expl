#!/usr/bin/env python
# coding: utf-8

# # M5 - Sales Uncertainty Prediction
# 
# 1. [Sources and guidelines](#sources)
# 2. [Preparing to start](#prepare)
#     * [Loading packages](#packages)
#     * [Loading data](#data)
# 3. ["What is meant by probabilistic forecasting?"...](#prob_forecasting)
#     * [What is a grouped time series?](#grouped_ts)
#     * [How does the hierarchy look like?](#hierarchy_ts)
#     * [How can we generate forecasts for grouped timeseries?](#forecasts_ts)
# 4. [The submission format](#submission)
#     * [Intro](#intro)
#     * [Prediction intervals and quartiles](#PIs)
#     * [Aggregation levels](#sub_aggregation_levels)
#     * [Submission EDA](#submission_eda)
# 5. [The Weighted Scaled Pinball loss](#loss)
#     * [The formula](#formula)
#     * [Playing with the loss implementation](#loss_implementation)
# 6. [The Naive method](#naive)
#     * [Prediction intervals for the Naive method](#prediction_intervals_naive)
#     * [Computing the loss for one timeseries of level 12](#loss_example)
# 7. [Facebook's Prophet](#prophet)
# 8. [LSTM and bootstrapped residuals](#lstm_bootstrapped_res)
#     * [Basic idea](#basic_idea)
#     * [Setting up LSTM](#lstm_setup)
#     * [Fitting the model to the top-level series](#fitting_lstm)
#     * [Check residuals for autocorrelation ](#residuals_checkup)
#     * [Computing PIs using bootstrapped residuals](#bootstrapped_PIs)
# 9. [Bayesian LSTM](#bayesian_lstm)
# 10. [Where to go next?](#next)

# # Sources and guidelines <a class="anchor" id="sources"></a>
# 
# * [M5 Competition guideline](https://mofc.unic.ac.cy/m5-competition/)
# * [M5 github repository](https://github.com/Mcompetitions/M5-methods/tree/master/validation)
# * [Forecasting - Principles and Practice (by Rob J Hyndman and George Athanasopoulos)](https://otexts.com/fpp2/)

# # Preparing to start <a class="anchor" id="prepare"></a>
# 
# ## Loading packages <a class="anchor" id="packages"></a>

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()


import time
from tqdm import tqdm_notebook as tqdm

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Loading data <a class="anchor" id="data"></a>

# ### Sales Training Data

# In[ ]:


train = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/sales_train_validation.csv")
train.head()


# In[ ]:


train.shape


# ### Calendar information

# In[ ]:


calendar = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/calendar.csv")
calendar.head()


# ### Sell prices information

# In[ ]:


sell_prices = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/sell_prices.csv")
sell_prices.head()


# # "What is meant by probabilistic forecasting?"... <a class="anchor" id="prob_forecasting"></a>
# 
# When I read about this competition this was one of the first questions that came into my mind. I know a few probabilistic models and methods and I have done some time series analysis before but I haven't directly got in touch with probabilistic timeseries analysis so far. The M5 competition is a good way to close this gap and to learn something new. To start I like to follow a question driven approach...
# 
# ## What is a grouped time series? <a class="anchor" id="grouped_ts"></a>
# 
# * Reading the competiton guideline we can find out that we have to deal with grouped time series of unit sales data. 
# * They show a hierarchy of different aggregation levels that are weighted equally in the loss functions. 
# * When working with grouped time series it's common to compute forecasts only for disaggregated time series and to add them up the same way the aggregation is performed for all remaining time series. 
# * In Chapter 10 of [Forecasting - Principles and Practice](https://otexts.com/fpp2/hierarchical.html) we can find even more information about how to do this "forecasting aggregation".

# ## How does the hierarchy look like? <a class="anchor" id="hierarchy_ts"></a>
# 
# In the competition guideline we can find that the hierarchy consits of 12 levels. Let's try to reconstruct some of them:
# 
# 1. The top is given by the unit sales of all products, aggregated for all stores/states. 
# 2. Unit sales of all products, aggregated for each state.
# 3. Unit sales of all products, aggregated for each store.
# 4. Unit sales of all products, aggregated for each category.
# 5. Unit sales of all products, aggregated for each department.
# 
# ...
# 
# Ok, time for a vis: ;-)

# In[ ]:


series_cols = train.columns[train.columns.str.contains("d_")].values
level_cols = train.columns[train.columns.str.contains("d_")==False].values


# In[ ]:


train.head(1)


# In[ ]:


sns.set_palette("colorblind")

fig, ax = plt.subplots(5,1,figsize=(20,28))
train[series_cols].sum().plot(ax=ax[0])
ax[0].set_title("Top-Level-1: Summed product sales of all stores and states")
ax[0].set_ylabel("Unit sales of all products");
train.groupby("state_id")[series_cols].sum().transpose().plot(ax=ax[1])
ax[1].set_title("Level-2: Summed product sales of all stores per state");
ax[1].set_ylabel("Unit sales of all products");
train.groupby("store_id")[series_cols].sum().transpose().plot(ax=ax[2])
ax[2].set_title("Level-3: Summed product sales per store")
ax[2].set_ylabel("Unit sales of all products");
train.groupby("cat_id")[series_cols].sum().transpose().plot(ax=ax[3])
ax[3].set_title("Level-4: Summed product sales per category")
ax[3].set_ylabel("Unit sales of all products");
train.groupby("dept_id")[series_cols].sum().transpose().plot(ax=ax[4])
ax[4].set_title("Level-4: Summed product sales per product department")
ax[4].set_ylabel("Unit sales of all products");


# ### Insights
# 
# * It has become much clearer how these levels are aggregated by performing groupby- and summing up the sales.
# * We can already observe nice periodic patterns. 

# ## How can we generate forecasts for grouped timeseries? <a class="anchor" id="forecasts_ts"></a>
# 
# * Our training data consists of 30490 timeseries. They belong to the bottom-level 12: Unit sales of product x, aggregated for each store.
# * A simple method to generate forecasts for all levels is to focus only on the bottom level. All of its predictions are then summed up to create the forecasts of all levels up to the top. This is called the bottom-up approach. 
# * As you can see [here](https://otexts.com/fpp2/bottom-up.html), there are many more approaches one could use, for example top-down or middle-out. 

# # The submission format <a class="anchor" id="submission"></a>
# 
# ## Intro <a class="anchor" id="intro"></a>
# 
# * We have 28 F-columns as we are predicting daily sales for the next 28 days. 
# * We are asked to make uncertainty estimates for these days.

# In[ ]:


submission = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/sample_submission.csv")
submission.head(10)


# In[ ]:


submission.shape


# * In the first submission row we are asked to make precitions for the top level 1 (unit sales of all products, aggregated for all stores/states)
# * The next 3 rows represent level 2.
# * Followed by level 3. 
# * This may goes on and on until the bottom level 12 is reached? Probably not as there seem to be only 3 combinations of ids.
# * Some rows contain aggregations at different levels. An X indicates the absence of an second aggregration level.
# * The prediction interval can be validation (related to the public leaderboard) or evaluation (related to the private leaderboard).

# ## Prediction intervals and quartiles <a class="anchor" id="PIs"></a>
# 
# Reading in the competition guideline, we can find that we are asked to make predictions for the median and four prediction intervals (PI): 50%, 67%, 95% and 99%. They belong to the following quartiles:
# 
# * 99% PI - $u_{1} = 0.005$ and $u_{9} = 0.995$
# * 95% PI - $u_{2} = 0.025$ and $u_{8} = 0.975$
# * 67% PI - $u_{3} = 0.165$ and $u_{7} = 0.835$
# * 50% PI - $u_{4} = 0.25$ and $u_{6} = 0.75$
# * median - $u_{5} = 0.5$

# ## Aggregation levels <a class="anchor" id="sub_aggregation_levels"></a>

# In[ ]:


np.random.choice(submission.id.values, replace=False, size=15)


# Browsing through the submission ids, we can see that we are given values of $u_{i}$ and information about the aggregation type like:
# 
# * the state id
# * the department id
# * the item id
# * the store id
# 
# It's a bit confusing that missing states are not represented by X. This makes splitting the id for EDA a bit more complicated. :-( Furthermore there is no clear separator. The $ \_ $ sign is also present in the department id. **It seems that one asked aggregation always consists of 3 ids. In cases of counts smaller than 3, we can observe X as placeholder.**  

# ## Submission EDA <a class="anchor" id="submission_eda"></a> 

# In[ ]:


def find_quartil(l):
    
    if "0.005" in l:
        return 0.005
    elif "0.025" in l:
        return 0.025
    elif "0.165" in l:
        return 0.165
    elif "0.25" in l:
        return 0.25
    elif "0.5" in l:
        return 0.5
    elif "0.75" in l:
        return 0.75
    elif "0.835" in l:
        return 0.835
    elif "0.975" in l:
        return 0.975
    elif "0.995" in l:
        return 0.995
    else:
        return 0
    
def find_state(l):
    if "CA" in l:
        return "CA"
    elif "TX" in l:
        return "TX"
    elif "WI" in l:
        return "WI"
    else:
        return "Unknown"
    
def find_category(l):
    if "FOODS" in l:
        return "foods"
    elif "HOBBIES" in l:
        return "hobbies"
    elif "HOUSEHOLD" in l:
        return "household"
    else:
        return "Unknown"


# In[ ]:


submission_eda = pd.DataFrame(submission.id, columns=["id"])
submission_eda.loc[:, "lb_type"] = np.where(submission.id.str.contains("validation"), "validation", "evaluation")
submission_eda.loc[:, "u"] = submission.id.apply(lambda l: find_quartil(l))
submission_eda.loc[:, "state"] = submission.id.apply(lambda l: find_state(l))
submission_eda.loc[:, "category"] = submission.id.apply(lambda l: find_category(l))


# In[ ]:


sns.set_palette("husl")

fig, ax = plt.subplots(3,3,figsize=(20,20))
sns.countplot(submission_eda.u, ax=ax[0,0]);
sns.countplot(submission_eda.lb_type, ax=ax[0,1]);
sns.countplot(submission_eda.state, ax=ax[1,0]);
sns.countplot(submission_eda.loc[submission_eda.lb_type=="validation"].state, ax=ax[1,1]);
sns.countplot(submission_eda.loc[submission_eda.lb_type=="evaluation"].state, ax=ax[1,2]);
sns.countplot(submission_eda.category, ax=ax[2,0]);
sns.countplot(submission_eda.loc[submission_eda.lb_type=="validation"].category, ax=ax[2,1]);
sns.countplot(submission_eda.loc[submission_eda.lb_type=="evaluation"].category, ax=ax[2,2]);
for n in range(1,3):
    ax[n,2].set_title("in evaluation")
    ax[n,1].set_title("in validation")


# ### Insights
# 
# * Each quartile u has exactily $2*42840 = 85680$ requests. The total number of all 12 level timeseries is 42840.
# * We have the same number of validation and evaluation requests and this explains the factor 2.
# * It seems that really all 12 aggregation levels are represented in the submission id. This is not clear yet and can be shown with further EDA (work in progress).

# # The Weighted Scaled Pinball loss <a class="anchor" id="loss"></a>
# 
# ## The formula <a class="anchor" id="formula"></a>
# 
# For each time series and for each quantile the **Scaled Pinball loss** can be computed by:
# 
# $$ SPL(u) = \frac{1}{h} \cdot \frac{1}{\frac{1}{n-1} \cdot \sum_{t=2}^{n} |Y_{t} - Y_{t-1}|} \cdot \sum_{t=n+1}^{n+h}
# \begin{cases} 
#     (Y_{t} - Q_{t}(u))\cdot u & \text{if } Y_{t} \geq Q_{t}(u) \\
#     (Q_{t}(u) - Y_{t})\cdot (1-u)       & \text{if } Y_{t} < Q_{t}(u)
# \end{cases} $$
# 
# whereas:
# 
# * $Y_{t}$ is the actual true future value of the time series at point $t$
# * $u$ is the considered quantile
# * $Q_{t}$ is the generated forecast for quantile $u$
# * $h$ is the forecasting horizon (28 days)
# * $n$ is the length of the training sample (number of historical observations)
# 
# After computing this loss for all 42840 time series and for all requested quantiles, the **Weighted Scaled Pinball** loss is computed as follows:
# 
# $$ WSPL = \sum_{i=1}^{42840} \cdot w_{i} \cdot \frac{1}{9} \sum_{j=1}^{9}SPL(u_{j})$$
# 
# In the M5 competiton we have 12 aggregation levels and as all hierarchical levels are equally weighted the weights should be $w_{i}=\frac{1}{12}$. The total number of time series is higher than what is given in train as all levels up to the top aggregation are included.
# 
# 
# 
# ## Playing with the loss implementation <a class="anchor" id="loss_implementation"></a>
# 
# Let's pick a single timeseries to get started with the loss and its implementation. As I just like to get started it's a bit quick and dirty and surely not the best way to write it down. ;-) 
# 

# In[ ]:


def spl_denominator(train_series):
    N = len(train_series)
    sumup = 0
    for n in range(1, N):
        sumup += np.abs(train_series[n]-train_series[n-1])
    return sumup/(N-1)


# In[ ]:


def spl_numerator(dev_series, Q, u):
    sumup = 0
    for m in range(len(dev_series)):
        if Q[m] <= dev_series[m]:
            sumup += (dev_series[m] - Q[m])*u
        else:
            sumup += (Q[m] - dev_series[m])*(1-u)
    return sumup


# In[ ]:


def spl(train_series, dev_series, Q, u):
    h = len(dev_series)
    spl_denomina = spl_denominator(train_series)
    spl_numera = spl_numerator(dev_series, Q, u)
    
    return spl_numera/(h*spl_denomina)


# I'm going to rework this section soon to make sure and check that the implementation is valid.

# # The Naive method <a class="anchor" id="naive"></a>
# 
# To really compute the loss for our example we need to make quantile forecasts $Q_{t}(u)$ for a given quantile $u$. So far we haven't setup a model, but we can start easily using the naive method described in the competition guideline:
# 
# $Y_{n+i} = Y_{n}$
# 
# for $i=1,2,...h$.
# 
# It's used for predicting series of the lowest level of the hierarchy. You can see that this approach just assumes the last known daily sale value for all requested predictions of the series. 
# 
# 
# ## Prediction intervals for the Naive method <a class="anchor" id="prediction_intervals_naive"></a>
# 
# 
# ### Assuming normally distributed forcasting errors
# 
# But how can we compute prediction intervals for this method? If we always assume the last known value for the next points of the time period with horizont h, there would be no distribution per time point that could tell us something about uncertainty. I really start to like [Forecasting - Principles and Practice](https://otexts.com/fpp2/). Take a look at the chapter ["Prediction intervals"](https://otexts.com/fpp2/prediction-intervals.html#prediction-intervals). Here we can read that one way to go is to assume normally distributed forcasting errors:
# 
# $$\epsilon(y) \sim N(\sigma_{h})$$
# 
# whereas $\sigma_{h}$ stands for the estimated standard deviation of future time step h. A prediction interval is then computed as a multiple of this standard deviation:
# 
# $$y_{lower, h} = y - c \cdot \sigma_{h}$$
# 
# $$y_{upper, h} = y + c \cdot \sigma_{h}$$
# 
# The factor $c$ is called **multiplier** and often such values are used for the requested prediction intervals:
# 
# * c = 2.58 for 99% PI
# * c = 1.96 for 95% PI
# * c ~ 0.95 for 67% PI
# * c = 0.67 for 50% PI
# 
# If you like to read more about prediction intervals, I found this [Wikipedia article](https://en.wikipedia.org/wiki/Prediction_interval) useful as well.
# 
# ### Prediction intervals for multi-step time horizonts
# 
# In our case we are asked to compute uncertainty estimates for a time period of 28 days. It's intuitive that our predictions become more uncertain the greater the time step $h$ of our time horizont. Therefore we can assume that $\sigma_{h}$ increases with h. 
# 
# 
# In contrast to one-step predictions we can't just use the standard deviation $\sigma$ of residuals $\epsilon_{t} = y_{true, t} - y_{fitted, t}$ of all time points $t$ between our observations and fitted values of the training data. Instead we could use some common benchmark methods:
# 
# * **Mean forcasts: $\sigma_{h} = \sigma \cdot \sqrt{1 + \frac{1}{T}}$ **
# * **Naive forcasts:** $\sigma_{h} = \sigma \cdot \sqrt{h}$
# * **Seasonal naive forcasts:** $\sigma_{h} = \sigma \cdot \sqrt{k+1}$ with k as the integer part of $\frac{(h-1)}{m}$ and $m$ as the seasonal period
# * **Drift forcasts:** $\sigma_{h} = \sigma \cdot \sqrt{h \cdot \left(\frac{1+h}{T}\right) }$
# 
# In these cases $T$ stands for the total time span in the training data and $h$ for our prediction time horizont (in our case 28 days). All methods assume that we have given uncorrelated residuals. Consequently we need to perform a residual analysis to check if this is true! :-) 
# 
# 
# But before doing so, we need to compute an example of the Naive method! 

# ## Computing the loss for one timeseries of level 12 <a class="anchor" id="loss_example"></a>

# In[ ]:


idx = 1000


# In[ ]:


train[level_cols].iloc[idx]


# Ok, we can see that this series represents all daily sales of item 445 in the CA_1 store. This item is part of the household_1 department and is also present as a row in the other shops:

# In[ ]:


train.loc[train.item_id=="HOUSEHOLD_1_445"].store_id.unique()


# Let's take a look at the daily sales of this series:

# In[ ]:


plt.figure(figsize=(20,5))
plt.plot(train[series_cols].iloc[idx].values, 'o')
plt.title("Item 445 daily sales in shop CA_1");
plt.xlabel("observed days")
plt.ylabel("Unit sales");


# Now, we need to split our single row data into a training and validation (dev) part. I decided to use the same period of time (28 days) for validation:

# In[ ]:


timeseries = train[series_cols].iloc[idx].values
h = 28

train_timeseries = timeseries[0:len(timeseries)-h]
dev_timeseries = timeseries[(len(timeseries)-h)::]

print(len(train_timeseries), len(dev_timeseries))


# Then we choose the last known value of the train timeseries as predictions for all asked 28 time points: 

# In[ ]:


naive_val = train_timeseries[-1]
naive_Q = np.ones(dev_timeseries.shape) * naive_val
naive_Q


# In this case all predictions are zero. Let's compute the loss for the median:

# In[ ]:


spl(train_timeseries, dev_timeseries, naive_Q, 0.5)


# ## Residual analysis 
# 
# Ok, let's start with the computation of residuals for our example time series. As we assumed that the last known value is valid for all future values, I would choose this one also as fitted value for all training data points in the past:

# In[ ]:


naive_val


# In[ ]:


residuals = train_timeseries - naive_val


# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.distplot(residuals, ax=ax[0], kde=False)
ax[0].set_xlabel("residuals")
ax[0].set_ylabel("frequency");
ax[0].set_title("Distribution of residuals");


# In[ ]:


np.mean(residuals)


# ### Insights
# 
# * As the last known value was 0 and this value is also the only possible minimum value, we do not observe a normal distribution of residuals!
# * Furthermore the mean of our residuals is not zero. In this case they are called biased and it's also an indicator that our model does not suite well.
# * I have decided to compute this example for the bottom level 12. In contrast to top-level time series the corresponding series do not show nice and clear periodic patterns that would likely yield more normally distributed residuals. 
# * Consequently we may conclude that some methods and models could be better suited for top or low-level series in the hierarchy but not for both. This is something to keep in mind.

# ### Computing prediction intervals
# 
# Now we need to choose a benchmark to compute how the predictions become more unsecure when time moves on. As I'm on the "naive" way, I like to use $\sigma_{h} = \sigma \cdot \sqrt{h}$. Let's do it for this single time series example:

# In[ ]:


std_dev = np.std(residuals)
std_h = np.ones(dev_timeseries.shape)

for h in range(1, 29):
    std_h[h-1] = std_dev * np.sqrt(h)


# In[ ]:


std_h


# Ok, the rest now is simple. We have seen that we can compute PIs using our multipliers:
# 
# $$y_{lower, h} = y - c \cdot \sigma_{h}$$
# 
# $$y_{upper, h} = y + c \cdot \sigma_{h}$$
# 
# * c = 2.58 for 99% PI
# * c = 1.96 for 95% PI
# * c ~ 0.95 for 67% PI
# * c = 0.67 for 50% PI

# I will only use one as an example: c=2.58 for 99% PI:

# In[ ]:


y_lower = np.ones(len(std_h))
y_upper = np.ones(len(std_h))
for h in range(len(std_h)):
    low_val = naive_Q[h] - 2.58 * std_h[h]
    if low_val < 0:
        y_lower[h] = 0
    else:
        y_lower[h] = low_val
    y_upper[h] = naive_Q[h] + 2.58 * std_h[h]


# In[ ]:


plt.figure(figsize=(20,5))
plt.plot(y_lower, c="r", label="0.005 boundary")
plt.plot(y_upper, c="g", label="0.995 boundary")
plt.plot(naive_Q, 'o', c="b", label="predicted value")
plt.title("Computing 99% PI for one timeseries of level 12");
plt.xlabel("time horizont h=28 days")
plt.ylabel("Unit sales");
plt.legend();


# # Facebook's Prophet <a class="anchor" id="prophet"></a>
# 
# 
# ## Model description
# 
# [Prophet](https://facebook.github.io/prophet/) is a decomposable time series model with 3 main model components and one error term:
# 
# $$y(t) = g(t) + s(t) + h(t) + \epsilon_{t}$$
# 
# * trend g(t) - non-periodic changes of the value
# * seasonality s(t) - periodic changes (e.g. weekly and yearly) 
# * holidays h(t) - effect of holidays (e.g. irregular patterns over one or more days)
# * $\epsilon_{t}$ - error term that describes any idiosyncratic changes (assumed to be normally distributed)
# 
# 
# ## Uncertainty estimates
# 
# By default it returns uncertainty intervals of the predicted value $y_{hat}$ consisting of three different sources:
# * uncertainty in the trend,
# * uncertainty in the seasonality estimates,
# * additional observation noise
# 
# To compute the uncertainty in the trend it is assumed that the average frequency and magnitude of trend changes will be the same in the future as observed in the history. This trend changes are projected forward into the future and by computing their distribution uncertainty intervals are obtained. **By default Prophet only returns uncertainty in the trend and observation noise!**

# ## Example - Total unit sales prediction
# 
# Let's sum up all unit sales given in our training data to obtain the top level time series of total unit sales of all stores and states:

# In[ ]:


timeseries = train[series_cols].sum().values
len(timeseries)


# As we are asked to predict a time window of 28 days, the easiest way to go now is to use the last 28 days for validation: 

# In[ ]:


train_timeseries = timeseries[0:-28]
eval_timeseries = timeseries[-28::]
print(len(train_timeseries), len(eval_timeseries))
days = np.arange(1, len(series_cols)+1)


# In[ ]:


plt.figure(figsize=(20,5))
plt.plot(days[0:-28], train_timeseries, label="train")
plt.plot(days[-28::], eval_timeseries, label="validation")
plt.title("Top-Level-1: Summed product sales of all stores and states");
plt.legend()
plt.xlabel("Day")
plt.ylabel("Unit sales");


# As far as I currently know Prophet likes to have the dates that we can find in our calendar dataframe:

# In[ ]:


dates = calendar.iloc[0:len(timeseries)].date.values
df = pd.DataFrame(dates, columns=["ds"])
df.loc[:, "y"] = timeseries
df.head()


# In[ ]:


train_df = df.iloc[0:-28]
train_df.shape


# In[ ]:


eval_df = df.iloc[-28::]
eval_df.shape


# In[ ]:


uncertainty_interval_width = 0.25


# In[ ]:


m = Prophet(interval_width=uncertainty_interval_width)
m.fit(train_df)
future = m.make_future_dataframe(periods=28)
forecast = m.predict(future)
forecast.head()


# In[ ]:


col_int = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
forecast[col_int].head()


# In[ ]:


plt.plot(forecast.iloc[-28::].yhat.values, 'o', label="predicted yhat")
plt.plot(eval_df.y.values, 'o-', label="target")
plt.legend();


# In[ ]:


fig = plot_plotly(m, forecast)  
py.iplot(fig)


# ## Submission for validation
# 
# Remember that we are asked to predict the following intervals PI:
# 
# * 99% PI - $u_{1} = 0.005$ and $u_{9} = 0.995$
# * 95% PI - $u_{2} = 0.025$ and $u_{8} = 0.975$
# * 67% PI - $u_{3} = 0.165$ and $u_{7} = 0.835$
# * 50% PI - $u_{4} = 0.25$ and $u_{6} = 0.75$
# * median - $u_{5} = 0.5$
# 
# Now let's fit the whole training data and predict for the validation timeperiod of the submission file. We have to set the interval in advance and personally it feels a bit overcomplicated to do so for each requested interval. But as I still need to understand Prophet in its details I'm going the following way:  

# In[ ]:


uncertainty_interval_width = 0.25


# In[ ]:


f_cols = [col for col in submission.columns if "F" in col]


# In[ ]:


submission_val = submission[submission.id.str.contains("validation")].copy()


# In[ ]:


def plugin_total_predictions():
    
    for uncertainty_interval_width in [0.005, 0.025, 0.165, 0.25]:
        upper = 1-uncertainty_interval_width
        lower = uncertainty_interval_width
    
        m = Prophet(interval_width=uncertainty_interval_width)
        m.fit(df)
        future = m.make_future_dataframe(periods=28)
        forecast = m.predict(future)
    
        submission_val.loc[
            (submission_val.id.str.contains("Total")) & (submission_val.id.str.contains(str(lower))),f_cols
        ] = np.round(forecast.yhat_lower.values[-28::])
    
        submission_val.loc[
            (submission_val.id.str.contains("Total")) & (submission_val.id.str.contains(str(upper))),f_cols
        ] = np.round(forecast.yhat_upper.values[-28::])
    
    submission_val.loc[
        (submission_val.id.str.contains("Total")) & (submission_val.id.str.contains(str(0.5))),f_cols
    ] = forecast.yhat.values[-28::]
    
    return submission_val


# In[ ]:


submission_val = plugin_total_predictions()
submission_val.loc[submission_val.id.str.contains("Total")]


# I'm not sure if this whole stuff makes sense. Personally I feel a strong need for Bayesian ML oand [credible intervals](https://en.wikipedia.org/wiki/Credible_interval). What I miss most is a much more detailed mathematical description of Prophet in the documentation. Only using the model without a deeper understanding of what is going on feels very sloppy and dangerous. :-(

# # LSTM and bootstrapped residuals <a class="anchor" id="lstm_bootstrapped_res"></a>
# 
# ## Basic idea <a class="anchor" id="basic_idea"></a>
# 
# * Here we are using the Frequentist perspective of probability:
#     * The model parameters $w$ are assumed to be fixed and we estimate it using our estimator. 
#     * The estimation depends on the dataset D we observe and consequently we can obtain error bars for our estimated parameters $w_{est}$ by considering multiple datasets.
#     * One way to do this is by creating new datasets, for example with bootstrapping.
# * We are creating new datasets by using bootstrapped residuals:
#     * We only assume uncorrelated residuals that need not be normally distributed.

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ## Setting up LSTM <a class="anchor" id="lstm_setup"></a>

# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[ ]:


class MyLSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, batch_size, num_layers=1, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout = 0.25)
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        
    def init_hidden(self):
        self.h_zero = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device)
        self.c_zero = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device)
    
    def forward(self, x):
        lstm_output, (h_n, c_n) = self.lstm(x.view(len(x), self.batch_size, -1),
                                           (self.h_zero, self.c_zero))
        last_time_step = lstm_output.view(self.batch_size, len(x), self.hidden_dim)[-1]
        pred = self.linear(last_time_step)
        return pred
    

def train_model(model, data_dict, lr=1e-4, num_epochs=500):
    
    loss_fun = torch.nn.MSELoss(reduction="mean")
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = np.zeros(num_epochs)
    phases = ["train", "eval"]
    losses_dict = {"train": [], "eval": []}
    predictions_dict = {"train": [], "eval": [] }
    
    for n in range(num_epochs):
        
        for phase in phases:
            
            x = data_dict[phase]["input"].to(device, dtype=torch.float)
            y = data_dict[phase]["target"].to(device, dtype=torch.float)
            
            if phase == "train":
                model.train()
            else:
                model.eval()
        
            optimiser.zero_grad()
            
            model.init_hidden()
            y_pred = model(x)
            
            if n == (num_epochs-1):
                predictions_dict[phase] = y_pred.float().cpu().detach().numpy()
            
            loss = loss_fun(y_pred.float(), y)
            losses_dict[phase].append(loss.item())
            
            if n % 50 == 0:
                print("{} loss: {}".format(phase, loss.item()))
            
            if phase == 'train':
                loss.backward()
                optimiser.step()
        
    return losses_dict, predictions_dict

def create_sequences(timeseries, seq_len):
    inputs = []
    targets = []
    
    max_steps = len(timeseries) - (seq_len+1)
    
    for t in range(max_steps):
        x = timeseries[t:(t+seq_len)]
        y = timeseries[t+seq_len]
        inputs.append(x)
        targets.append(y)
    
    return np.array(inputs), np.array(targets)


# ## The top timeseries - preprocessing <a class="anchor" id="preprocessing"></a>

# Let's use the timeseries of total unit sales as an example again. For preprocessing we should remove the trend and scale the values.

# In[ ]:


diff_series = np.diff(timeseries)
train_size = np.int(0.7 * len(diff_series))
train_diff_series = diff_series[0:train_size]
eval_diff_series = diff_series[train_size::]
scaler = MinMaxScaler(feature_range=(-1,1))
scaled_train = scaler.fit_transform(train_diff_series.reshape(-1, 1))
scaled_eval = scaler.transform(eval_diff_series.reshape(-1,1))


# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
ax[0].plot(scaled_train, '-o', c="b")
ax[1].plot(scaled_eval, '-o', c="g")
ax[0].set_title("Single preprocessed top timeseries in train")
ax[1].set_title("Single preprocessed top timeseries in eval");
ax[0].set_xlabel("Days in dataset")
ax[1].set_xlabel("Days in dataset")
ax[0].set_ylabel("$\Delta y$ scaled")
ax[1].set_ylabel("$\Delta y$ scaled");


# ## Fitting the model to the top-level series <a class="anchor" id="fitting_lstm"></a>

# In[ ]:


seq_len = 400
input_dim = 1
hidden_dim = 128
num_epochs = 600
lr=0.0005


x_train, y_train = create_sequences(scaled_train, seq_len)
x_eval, y_eval = create_sequences(scaled_eval, seq_len)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_eval = torch.from_numpy(x_eval).float()
y_eval = torch.from_numpy(y_eval).float()

data_dict = {"train": {"input": x_train, "target": y_train},
             "eval": {"input": x_eval, "target": y_eval}}


# In[ ]:


model = MyLSTM(input_dim=input_dim,
               hidden_dim=hidden_dim,
               batch_size=seq_len)
model = model.to(device)


# In[ ]:


run_training = True
if run_training:
    losses_dict, predictions_dict = train_model(model, data_dict, num_epochs=num_epochs, lr=lr)


# In[ ]:


if run_training:
    
    fig, ax = plt.subplots(3,1,figsize=(20,20))
    ax[0].plot(losses_dict["train"], '.-', label="train", c="red")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("MSE")
    ax[0].plot(losses_dict["eval"], '.-', label="eval", c="blue");
    ax[0].legend();

    ax[1].plot(predictions_dict["train"], '-o', c="red")
    ax[1].plot(y_train, '-o', c="green")
    ax[1].set_title("Fitted and true values of y in train");
    ax[1].set_ylabel("Unit sales y");
    ax[1].set_xlabel("Number of days in train");

    ax[2].plot(predictions_dict["eval"], '-o', c="red")
    ax[2].plot(y_eval, '-o', c="green")
    ax[2].set_title("Predicted and true values of y in eval");
    ax[2].set_xlabel("Number of days in eval");
    ax[2].set_ylabel("Unit sales y");


# ## Check residuals for autocorrelation <a class="anchor" id="residuals_checkup"></a>

# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf

if run_training:
    
    train_residuals = y_train-predictions_dict["train"]
    eval_residuals = y_eval-predictions_dict["eval"]
    
    fig, ax = plt.subplots(2,2,figsize=(20,10))
    sns.distplot(train_residuals, ax=ax[0,0], color="red")
    sns.distplot(eval_residuals, ax=ax[0,1], color="green")
    ax[0,0].set_title("Train residuals")
    ax[0,1].set_title("Eval residuals")
    ax[0,0].set_xlabel("$y_{true} - y_{pred}$")
    ax[0,1].set_xlabel("$y_{true} - y_{pred}$")
    ax[0,0].set_ylabel("density")
    ax[0,1].set_ylabel("density")
    
    plot_acf(train_residuals, ax=ax[1,0])
    plot_acf(eval_residuals, ax=ax[1,1])


# ### Insights
# 
# * The residuals in train only show a significant autocorrelation with their previous, 1-lag timepoint. 
# * That's great as we are close to uncorrelated residuals that were assumed when using bootstrapped residuals.

# ## Computing PIs using bootstrapped residuals <a class="anchor" id="bootstrapped_PIs"></a>
# 
# The idea of computing PIs using bootstrapped residuals is as follows:
# 
# 1. Fit the model to your data to obtain the fitted values $\hat{y}_{i}$ and the forcasting errors $\epsilon_{i} = y_{true, i} - \hat{y}_{i}$.
# 2. Randomly sample a residual $\epsilon_{i}$ of the distribution of all $\epsilon_{j}$ to generate a new response variable $y^{*}$ using the fitted value: $y^{*} = \hat{y}_{i} + \epsilon_{i}$. 
# 3. Doing this repeatively we obtain many different, synthetic values for future predictions that we can use to compute prediction intervals.
# 
# Let's take a look at a single example first:

# In[ ]:


sampled_residuals = np.random.choice(train_residuals[:, 0], size=len(y_train), replace=True)
new_response = predictions_dict["train"] + sampled_residuals


# In[ ]:


fig, ax = plt.subplots(2,2,figsize=(20,10))
ax[0,0].plot(predictions_dict["train"][0:200], 'o-', color="purple")
ax[0,0].set_title("Original fitted values $y_{pred}$ in train")
ax[0,0].set_xlabel("200 example days")
ax[0,0].set_ylim(-0.4, 0.4)
ax[0,0].set_ylabel("$y_{fitted}$")

ax[0,1].plot(new_response[0:200,0], 'o-', color="orange")
ax[0,1].set_title("Response values $y^{*}$ using sampled residuals");
ax[0,1].set_xlabel("200 example days")
ax[0,1].set_ylabel("$y^{*}$");
ax[0,1].set_ylim(-0.4, 0.4)

ax[1,0].plot(sampled_residuals[0:200], 'o-', color="cornflowerblue")
ax[1,0].set_title("Sampled residuals")
ax[1,0].set_xlabel("200 example days")
ax[1,0].set_ylabel("$\epsilon$")

ax[1,1].plot(y_train[0:200], 'o-', color="firebrick")
ax[1,1].set_title("True values $y_{train}$")
ax[1,1].set_xlabel("200 example days")
ax[1,1].set_ylabel("$y_{train}$");


# Now we need to do this multiple times to obtain multiple response series. Then we need to reverse our preprocessing to obtain the related values for PIs.

# In[ ]:


responses = []
for n in range(50):
    sampled_residuals = np.random.choice(train_residuals[:, 0], size=len(y_train), replace=True)
    new_response = predictions_dict["train"] + sampled_residuals
    responses.append(new_response[:,0])
responses = np.array(responses)
responses.shape


# Reverse the preprocessing:

# ### ToDo: 
# 
# * Increase the hidden size. Play more with the sequence length.
# * Implement LSTM with more than 1 layer to further increase the number of parameters. Play with it.
# * Add cyclical learning rate.
# * Compute PIs using the bootstrapped residuals method.

# ## Bayesian neural network <a class="anchor" id="bayesian_lstm"></a>
# 
# * In this case we are using the Bayesian perspective of probability:
#     * We only observe one dataset. (the one we actually have)
#     * Our uncertainty in the model parameters $w$ is now explicity expressed by a probability distribution over $w$.
# 
# I found this [article](https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd) very helpful and I'm still reading it. Perhaps I can code something like this as well... at least in a few months. Unfortunately the competition is over then but for learning it would be great. :-) 

# # Where to go next? <a class="anchor" id="next"></a>
# 
# I'm going to continue with the following topics:
# 
# * More about Prophet and personal adjustments - computing PIs
# * Perhaps (not sure yet if I will make it in the next weeks) - Computing PIs even with neural networks
# * Conclusion of what I have learnt by writing this kernel
