#!/usr/bin/env python
# coding: utf-8

# # IDAO: expected time of orders in airports
# 
# Airports are special points for taxi service. Every day a lot of people use a taxi to get to the city centre from the airport.
# 
# One of important task is to predict how long a driver need to wait an order. It helps to understand what to do. Maybe the driver have to wait near doors, or can drink a tea, or even should drive to city center without an order.
# 
# We request you to solve a simple version of this prediction task.
# 
# **Task:** predict time of $k$ orders in airport (time since now when you get an order if you are $k$-th in queue), $k$ is one of 5 values (different for every airports).
# 
# **Data**
# - train: number of order for every minutes for 6 months
# - test: every test sample has datetime info + numer of order for every minutes for last 2 weeks
# 
# **Submission:** for every airport you should prepare a model which will be evaluated in submission system (code + model files). You can make different models for different airports.
# 
# **Evaluation:** for every airport for every $k$ sMAPE will be calculated and averaged. General leaderboard will be calculated via Borda count. 
# 
# ## Baseline

# In[14]:


get_ipython().run_line_magic('pylab', 'inline')

import catboost
import pandas as pd
import pickle
import tqdm


# Let's prepare a model for set1.

# # Load train dataset

# In[15]:


set_name = 'set1'
path_train_set = '../input/{}.csv'.format(set_name)

data = pd.read_csv(path_train_set)
data.datetime = data.datetime.apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
data = data.sort_values('datetime')
data.head()


# Predict position for set2.

# In[16]:


target_positions = {
    'set1': [10, 30, 45, 60, 75],
    'set2': [5, 10, 15, 20, 25],
    'set3': [5, 7, 9, 11, 13]
}[set_name]


# Some useful constant.

# In[17]:


HOUR_IN_MINUTES = 60
DAY_IN_MINUTES = 24 * HOUR_IN_MINUTES
WEEK_IN_MINUTES = 7 * DAY_IN_MINUTES

MAX_TIME = DAY_IN_MINUTES


# ## Generate train samples with targets

# We have only history of orders (count of orders in every minutes) but we need to predict time of k orders since current minutes. So we should calculate target for train set. Also we will make a lot of samples from all set (we can only use two weeks of history while prediction so we can use only two weeks in every train sample).

# In[18]:


samples = {
    'datetime': [],
    'history': []}

for position in target_positions:
    samples['target_{}'.format(position)] = []
    
num_orders = data.num_orders.values


# To calculate target (minutes before k orders) we are going to use cumulative sum of orders. 

# In[19]:


# start after 2 weeks because of history
# finish earlier because of target calculation
for i in range(2 * WEEK_IN_MINUTES,
               len(num_orders) - 2 * DAY_IN_MINUTES):
    
    samples['datetime'].append(data.datetime[i])
    samples['history'].append(num_orders[i-2*WEEK_IN_MINUTES:i])
    
    # cumsum not for all array because of time economy
    cumsum_num_orders = num_orders[i+1:i+1+2*DAY_IN_MINUTES].cumsum()
    for position in target_positions:
        orders_by_positions = np.where(cumsum_num_orders >= position)[0]
        if len(orders_by_positions):
            time = orders_by_positions[0] + 1
        else:
            # if no orders in last days
            time = MAX_TIME
        samples['target_{}'.format(position)].append(time)


# Convert to pandas.dataframe. Now we have targets to train and predict.

# In[20]:


df = pd.DataFrame.from_dict(samples)
df.head()


# # Train model

# Let's generate simple features.
# 
# By time:

# In[21]:


df['weekday'] = df.datetime.apply(lambda x: x.weekday())
df['hour'] = df.datetime.apply(lambda x: x.hour)
df['minute'] = df.datetime.apply(lambda x: x.minute)
df['day'] = df.datetime.apply(lambda x: x.day)


# Aggregators by order history with different shift and window size:

# In[22]:


SHIFTS = [
    HOUR_IN_MINUTES // 4,
    HOUR_IN_MINUTES // 2,
    HOUR_IN_MINUTES,
    DAY_IN_MINUTES,
    DAY_IN_MINUTES * 2,
    WEEK_IN_MINUTES,
    WEEK_IN_MINUTES * 2
]
WINDOWS = [
    HOUR_IN_MINUTES // 4,
    HOUR_IN_MINUTES // 2,
    HOUR_IN_MINUTES,
    DAY_IN_MINUTES,
    DAY_IN_MINUTES * 2,
    WEEK_IN_MINUTES,
    WEEK_IN_MINUTES * 2
]


# In[23]:


df.shape


# In[24]:


for shift in SHIFTS:
    for window in WINDOWS:
        if window > shift:
            continue
        df['num_orders_{}_{}'.format(shift, window)] =             df.history.apply(lambda x: x[-shift : -shift + window].sum())
        


# In[ ]:





# In[25]:


df.shape


# In[26]:


for shift in SHIFTS:
    for window in WINDOWS:
        if ((window > shift) or (shift + window > WEEK_IN_MINUTES * 2)):
            continue
        if shift==window:
            df['diff_num_orders_{}-{}_{}'.format(shift * 2, shift, window)] =                 df['num_orders_{}_{}'.format(shift, window)]                 - df.history.apply(lambda x: x[-shift - window: -shift].sum())


# In[27]:


SHIFTS = [
    DAY_IN_MINUTES * 1,
    DAY_IN_MINUTES * 2,
    DAY_IN_MINUTES * 3,
    DAY_IN_MINUTES * 4,
    DAY_IN_MINUTES * 5,
    DAY_IN_MINUTES * 6,
    WEEK_IN_MINUTES,
    WEEK_IN_MINUTES * 2]


# In[28]:


for shift in SHIFTS:
    cumsum_num_orders = df.history.apply(lambda x : x[-shift:-shift+1*DAY_IN_MINUTES-1].cumsum())
    for position in target_positions:
        orders_by_positions = cumsum_num_orders.apply(lambda x : np.where(x >= position)[0])
        time = orders_by_positions.apply(lambda x : x[0] if len(x) else MAX_TIME)
        df['wait_time_{}_k_{}'.format(shift, position)] = time


# In[29]:


df.shape


# In[30]:


df['diff_2880-1440_60'] =  df['num_orders_2880_60'] - df['num_orders_1440_60']
df['diff_20160-10080_60'] =  df['num_orders_20160_60'] - df['num_orders_10080_60']
df['diff_10020-60_60'] =  df.history.apply(lambda x: x[-10080 - 60: -10080].sum()) - df['num_orders_60_60']
df['diff_10050-30_30'] =  df.history.apply(lambda x: x[-10080 - 30: -10080].sum()) - df['num_orders_30_30']
df['diff_10065-15_15'] =  df.history.apply(lambda x: x[-10080 - 15: -10080].sum()) - df['num_orders_15_15']


# In[31]:


for position in target_positions:    
    df['lag_target_avg_10_min_{}'.format(position)] = None
    lag_ser = (
        df['target_{}'.format(position)][:-WEEK_IN_MINUTES] + 
        df['target_{}'.format(position)][1:-WEEK_IN_MINUTES+1] + 
        df['target_{}'.format(position)][2:-WEEK_IN_MINUTES+2] + 
        df['target_{}'.format(position)][3:-WEEK_IN_MINUTES+3] +
        df['target_{}'.format(position)][4:-WEEK_IN_MINUTES+4] + 
        df['target_{}'.format(position)][5:-WEEK_IN_MINUTES+5]
    )
    df['lag_target_avg_10_min_{}'.format(position)][df.shape[0] - lag_ser.shape[0]:] = lag_ser    
    
df.dropna(inplace=True)


# In[ ]:





# In[32]:


get_ipython().run_cell_magic('time', '', "SHIFTS = [\n    0,\n    WEEK_IN_MINUTES,\n]\n\ndef make_get_idx(pos, start):\n    def get_closest_index(a):\n        indexes = np.where(np.flip(a)[start:].cumsum() >= pos)[0]\n        return indexes[0] if len(indexes) else MAX_TIME\n    return get_closest_index\n\nfor shift in SHIFTS:\n    for position in target_positions:\n        df['time_waited_for_{}_{}_ago'.format(position, shift)] = df['history'].apply(\n            make_get_idx(position, shift))")


# In[33]:


df_train = df.loc[df.datetime <= df.datetime.max() - datetime.timedelta(days=28)]
df_test = df.loc[df.datetime > df.datetime.max() - datetime.timedelta(days=28)]


# In[34]:


target_cols = ['target_{}'.format(position) for position in target_positions]

y_train = df_train[target_cols]
y_test = df_test[target_cols]

y_full = df[target_cols]
df_full = df.drop(['datetime', 'history'] + target_cols, axis=1)

df_train = df_train.drop(['datetime', 'history'] + target_cols, axis=1)
df_test = df_test.drop(['datetime', 'history'] + target_cols, axis=1)


# In[35]:


def sMAPE(y_true, y_predict, shift=0):
    return 2 * np.mean(
        np.abs(y_true - y_predict) /
        (np.abs(y_true) + np.abs(y_predict) + shift))


# Also we will save models for prediction stage.

# In[36]:


model_to_save = {
    'models': {}
}


# In[37]:


df_train.shape


# What is good or bad model? We can compare our model with constant solution. For instance median (optimal solution for MAE).

# In[38]:


for position in target_positions:
    model = catboost.CatBoostRegressor(
        iterations=600, max_depth=7, l2_leaf_reg=20, 
        learning_rate=1, loss_function='MAE', #has_time=True, 
        task_type='GPU', random_seed=42)
    model.fit(
        X=df_full,
        y=y_full['target_{}'.format(position)],
        verbose=False)
    
    model_to_save['models'][position] = model


# target_10
# model:	0.3348871993829415
# 
# target_30
# model:	0.26760281067011377
# 
# target_45
# model:	0.24277043007404484
# 
# target_60
# model:	0.2235240653846019
# 
# target_75
# model:	0.2090631840662612

# target_10
# stupid:	0.523681447755815
# model:	0.3357004371082772
# 
# target_30
# stupid:	0.5162922494395047
# model:	0.2673064847922403
# 
# target_45
# stupid:	0.5119119032355639
# model:	0.24067873802193088
# 
# target_60
# stupid:	0.5090360366765123
# model:	0.2215424577756922
# 
# target_75
# stupid:	0.496690263839422
# model:	0.20585880384897284

# target_10
# stupid:	0.523681447755815
# model:	0.33657616907074867
# 
# target_30
# stupid:	0.5162922494395047
# model:	0.2684801884797687
# 
# target_45
# stupid:	0.4960557520117922
# model:	0.24076796228859576
# 
# target_60
# stupid:	0.48420937414842236
# model:	0.2232904456702252
# 
# target_75
# stupid:	0.47617614691570065
# model:	0.2073645487124787
# 

# In[ ]:





# In[ ]:





# target_10
# model:	0.30444717323311676
# 
# target_30
# model:	0.19020018428259872
# 
# target_45
# model:	0.12716225012718482
# 
# target_60
# model:	0.06894638047975396
# 
# target_75
# model:	0.013732056002829182

# target_5_ 0.45198970734849553
# 
# target_7_ 0.40044636098431524
# 
# target_9_ 0.37325054456402385
# 
# target_11 0.3441435253815883
# 
# target_13 0.30141112540996073
# 
# 
# 
# 

# Our model is better than constant solution. Saving model.

# In[ ]:


pickle.dump(model_to_save, open('models.pkl', 'wb'))


# target_10 0.33839824987623884
# 
# target_30 0.27497175468398594
# 
# target_45 0.2454199119262242
# 
# target_60 0.22526960456425713
# 
# target_75 0.210372415480981
# 
# 
# 
