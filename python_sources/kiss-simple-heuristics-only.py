#!/usr/bin/env python
# coding: utf-8

# # KISS - Simple Heuristics Only 
# 
# _Keep is Simple, Stupid - by Nick Brooks, March 2020_
# 
# 
# **KISS Series:** <br>
# [Predicting Future Demand Competition Notebook](https://www.kaggle.com/nicapotato/cutting-edge-kiss-method)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics 

import math
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import time
import random

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
notebookstart = time.time()


# #### Load

# In[ ]:


train_sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
submission_file = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

print("Train Sales Shape: {} Rows, {} Columns".format(*train_sales.shape))
display(train_sales.head())
print("Submission File Shape: {} Rows, {} Columns".format(*submission_file.shape))
display(submission_file.head(2))
print("Calendar Shape: {} Rows, {} Columns".format(*calendar.shape))
calendar['date'] = pd.to_datetime(calendar['date'])
display(calendar.head(5))


# In[ ]:


days = range(1, 1913 + 1)
time_series_columns = [f'd_{i}' for i in days]
time_series_data = train_sales[time_series_columns]


# #### Quick Exploration

# In[ ]:


calendar_merge_cols = ['d','date', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI']
plot_pd = pd.merge(calendar.loc[:,calendar_merge_cols], train_sales.iloc[:,6:].T, left_on='d', right_index = True, how = 'left')    .set_index('date')


# In[ ]:


plot_pd.head()


# In[ ]:


f, ax = plt.subplots(5,1, figsize = [15,10])
# Demand Sum
i = 0
time_series_data.sum(axis = 0).plot(ax=ax[i], c = 'b')
ax[i].set_title("Product Sum Over Time")
ax[i].set_xlabel("Date")
ax[i].set_ylabel("Product Quantity Demanded")

# Demand Mean
i += 1
time_series_data.mean(axis = 0).plot(ax=ax[i], c = 'k')
ax[i].set_title("Product Mean Over Time")
ax[i].set_xlabel("Date")
ax[i].set_ylabel("Product Quantity Demanded")

# Demand Standard Deviation
i += 1
time_series_data.std(axis = 0).plot(ax=ax[i], c = 'r')
ax[i].set_title("Product Stdev Over Time")
ax[i].set_xlabel("Date")
ax[i].set_ylabel("Product Quantity Demanded")

# Demand Median
i += 1
time_series_data.median(axis = 0).plot(ax=ax[i], c = 'b')
ax[i].set_title("Product Median Over Time")
ax[i].set_xlabel("Date")
ax[i].set_ylabel("Q Demanded")

# Zero to None-Zero Proportion
i += 1
pd.Series(np.count_nonzero(time_series_data.values, axis = 0) / time_series_data.shape[0]).plot(ax=ax[i], c = 'm')
ax[i].set_title("Non-Zero Proportion")
ax[i].set_xlabel("Date")
ax[i].set_ylabel("Percent Non-Zero")

plt.tight_layout(pad=1)
plt.show()


# **Observations:** <br>
# - Increasing product sold over time
# - Standard Deviation has a bump in the middle
# - Median Product Demand is Zero
# - Increasing Non-Zero Proportion suggests that new products are introduced over time.
# 
# ***
# 
# #### Simple Heuristics ONLY

# In[ ]:


def flat_function(train, prediction_size, func, kwargs):
    train = func(train, **kwargs)
    train = np.tile(train.transpose(), (prediction_size, 1)).transpose()
    return train

def step_function(train, step_count, prediction_size, func, kwargs):
    for step_i in range(0, step_count):
        step_pred = func(train[:,-prediction_size:], **kwargs)
        train = np.concatenate((train, step_pred.reshape(-1,1)), axis = 1)
    return train 

# Custom Heuristics


# In[ ]:


####################################################################################################
simple_models = [
    # Method, Rolling, Func, Kwargs
    ('Flat Mean', False, np.mean, dict(axis = 1)),
    ('Rolling Mean', True, np.mean, dict(axis = 1)),
    ('Flat Median', False, np.median, dict(axis = 1)),
    ('Rolling Median', True, np.median, dict(axis = 1)),
]
test_windows = [
    7,
    14,
    28,
    56
]

sub_size = 28*2
pred_step_size = 1
ts_metrics = ['rmse','mse','mae']
plot_metric = 'rmse'
full_eval = {}
full_oof = {}

columns = 1
rows = math.ceil(len(simple_models)/columns)
n_plots = rows*columns
f,ax = plt.subplots(rows, columns, figsize = [16,rows*4])

for method_i, (method, rolling, func, kwargs) in enumerate(simple_models):
    palette = itertools.cycle(sns.color_palette("Dark2", 15))
    ax = plt.subplot(rows, columns, method_i+1)
    
    for plot_i, prediction_size in enumerate(test_windows):
        time_splits = time_series_data.shape[1] // prediction_size
        rows_to_consider = time_splits * prediction_size
        train_subset = time_series_data.iloc[:, -rows_to_consider:].values
        time_split_results_list = []
        train_oof = np.zeros(train_subset.shape)
        
        # Time-Split Backtesting..
        for i in range(0, time_splits - 1):
            train = train_subset[:,(i)*prediction_size:(i+1)*prediction_size]
            validation = train_subset[:,(i+1)*prediction_size:(i+2)*prediction_size]

            # Step predicitons or flat values
            if not rolling:
                train = flat_function(train=train, prediction_size=prediction_size,
                                      func=func, kwargs=kwargs)
            elif rolling:
                train = step_function(train=train, step_count=prediction_size,
                            prediction_size=prediction_size, func=func, kwargs=kwargs)
            else:
                raise("rolling Variables Not Correctly Defined")
                
            rmse = metrics.mean_squared_error(validation, train[:,-prediction_size:],
                                              squared = False)
            mse = metrics.mean_squared_error(validation, train[:,-prediction_size:],
                                             squared = True)
            mae = metrics.mean_absolute_error(validation, train[:,-prediction_size:])
            time_split_results_list.append([i, [mse, rmse, mae]])
            train_oof[:,(i+1)*prediction_size:(i+2)*prediction_size] = train[:,-prediction_size:]

        if not rolling:
            tmp_matrix = flat_function(
                train=time_series_data.iloc[:, -prediction_size:].values,
                prediction_size=sub_size, func=func, kwargs=kwargs)
        elif rolling:
            tmp_matrix = step_function(
                train=time_series_data.iloc[:, -prediction_size:].values,
                step_count=sub_size,
                prediction_size=prediction_size, func=func, kwargs=kwargs)
            tmp_matrix = tmp_matrix[:,-sub_size:]
        
        forecast = pd.DataFrame(np.concatenate((tmp_matrix[:,:28], tmp_matrix[:,28:])),
                                columns = [f'F{i}' for i in range(1, 28 + 1)])

        validation_ids = train_sales['id'].values
        evaluation_ids = [i.replace('validation', 'evaluation') for i in validation_ids]
        ids = np.concatenate([validation_ids, evaluation_ids])

        predictions = pd.DataFrame(ids, columns=['id'])
        
        predictions = pd.concat([predictions, forecast], axis=1)
        predictions.to_csv(f"{method}-{prediction_size}_sub.csv", index=False)

        # Evaluation
        time_split_results = pd.DataFrame(
            time_split_results_list, columns = ['time_slice', 'metrics'])
        time_split_results[ts_metrics] = pd.DataFrame(
            time_split_results['metrics'].values.tolist(),
            index=time_split_results.index)
        time_split_results.drop(["metrics"],axis =1, inplace=True)

        # Overall Eval
        overall_eval = time_split_results[ts_metrics].mean(axis = 0).to_dict()
        for k, v in overall_eval.items():
            overall_eval[k] = round(v, 2)
            
        full_oof[f"{method}-{str(prediction_size)}"] = np.concatenate([train_oof, tmp_matrix], axis = 1)
        full_eval[f"{method}-{str(prediction_size)}"] = overall_eval

        # Plot
        line_info = (f"{method}-{prediction_size} "
                     f"{overall_eval}")
        ax.plot(np.linspace(0, rows_to_consider,
                    num=int(rows_to_consider / prediction_size))[1:],
                time_split_results[plot_metric].values,
                label=line_info, alpha = .8)
        
    ax.set_title(f"{method}")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{plot_metric} Error")
    ax.legend(fontsize='medium', loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout(pad=1)
plt.show()


# In[ ]:


for k in full_oof.keys():
    print(k, full_oof[k].shape)


# In[ ]:


time_series_data.shape


# In[ ]:


# Plot Parameters
sample_n = 20
rows = sample_n
columns = 1
quantile = .99
ground_truth_rolling_average = 2
plot_method_subset = "Rolling Mean"
plot_last_n_points = 1800


# Plot Loop
f,ax = plt.subplots(columns, rows, figsize = [40,5*sample_n])
samples = [random.randint(0, time_series_data.shape[0]) for _ in range(sample_n)]
for plot_i, index in enumerate(samples):
    ax = plt.subplot(rows, columns, plot_i+1)
    
    # Plot Predictions
    for method in [x for x in full_oof.keys() if plot_method_subset in x]:
        ax.plot(full_oof[method][index,-plot_last_n_points:], label = f"{method}", alpha = .6)
    
    # Plot Ground Truth
    ground_truth = pd.Series(time_series_data.values[index,-(plot_last_n_points-sub_size):])        .rolling(window = ground_truth_rolling_average).mean()
    ax.plot(ground_truth,
            alpha = .3, label="Ground Truth", c='k')
    ax.axvline(x = plot_last_n_points-sub_size, linewidth=5, color='r')
    
    # Annotage Plot
    ax.set_title(f"{method} - Row:{index}")
    ax.set_ylim(0, np.quantile(ground_truth.dropna(), quantile))
    ax.set_ylabel("Product Demand")
    ax.set_xlabel("Time")
    ax.legend(fontsize='medium', loc='center left', bbox_to_anchor=(1, 0.5))
    
plt.tight_layout(pad=1)
plt.show()


# In[ ]:


results = pd.DataFrame(full_eval).T
results.sort_values(by="rmse", ascending=True)


# In[ ]:


print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


# In[ ]:




