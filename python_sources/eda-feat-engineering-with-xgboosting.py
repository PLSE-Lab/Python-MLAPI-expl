#!/usr/bin/env python
# coding: utf-8

# Welcome to my Kernel to try understand the TalkingData AdTrack. 
# 
# <i>*English is not my first language, so sorry for any error</i>
# 
# 

# I will try understand and reply some questions that I formulate. <br>
# 
# For this competition,  our objective is to predict whether a user will download an app after clicking a mobile app advertisement. 
# 
# <h2>Introducing to dataset:</h2>
# Each row of the training data contains a click record, with the following features.
# 
# This is our features on train dataset:<br>
# <b>ip:</b> ip address of click.<br>
# <b>app:</b> app id for marketing.<br>
# <b>device: </b>device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)<br>
# <b>os:</b> os version id of user mobile phone<br>
# <b>channel:</b> channel id of mobile ad publisher<br>
# <b>click_time:</b> timestamp of click (UTC)<br>
# <b>attributed_time: </b>if user download the app for after clicking an ad, this is the time of the app download<br>
# <b>is_attributed: </b>the target that is to be predicted, indicating the app was downloaded<br>
# <b>Note</b> that <i>ip, app, device, os, and channel</i> are encoded.<br>
# 
# 
# <h2>So, I will try answer this questions: </h2>
# - Are the downloads and clicks balanced? 
# - Are the IP's with the same distribuition? 
# - Are all devices on our dataset with the same distribuition?
# - What is the most commom app ?
# - What is the most commom channel ?
# - Have any hour or minute that the download rate iw? 
# - What's the distribuition of time? 
# - We have some patterns that might can explain the downloads?

# # Couting the lines of full dataset
# 

# In[1]:


import subprocess
print('# Line count:')
for file in ['train.csv', 'test.csv', 'train_sample.csv']:
    lines = subprocess.run(['wc', '-l', '../input/{}'.format(file)], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(lines, end='', flush=True)


# That makes 185 million rows in the training set and 19 million in the test set. 

# # **Importing the librarys and datasets **

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# I will import 10 millions of rows to do this analysis

# In[3]:


df_talk = pd.read_csv("../input/train.csv", nrows=2500000,parse_dates=['click_time'])


# <h2>Looking data types and if we have any null values</h2>

# In[4]:


df_talk.info()


# The feature attributed_time have a high number of null values

# <h2>Unique values of this sample of 1000000</h2>

# In[5]:


df_talk.nunique()


# We can see that we have 39611 different ip's  

#  <h2>Looking the data </h2>

# In[6]:


df_talk.head()


# <h2>Starting Feature Engineering in datetime column</h2>

# I will do some feature engineering in datetime column to we have more feature that might can help explain the download
# 

# In[7]:


def datetime_to_deltas(series, delta=np.timedelta64(1, 's')):
    t0 = series.min()
    return ((series-t0)/delta).astype(np.int32)

df_talk['sec'] = datetime_to_deltas(df_talk.click_time)


# <h3>Extracting datetime values </h3>

# In[8]:


df_talk['day'] = df_talk['click_time'].dt.day.astype('uint8')
df_talk['hour'] = df_talk['click_time'].dt.hour.astype('uint8')
df_talk['minute'] = df_talk['click_time'].dt.minute.astype('uint8')
df_talk['second'] = df_talk['click_time'].dt.second.astype('uint8')
df_talk['week'] = df_talk['click_time'].dt.dayofweek.astype('uint8')

df_talk.head()


# Did it, let's start ploting some graphs to try understand the distribuitions. '

# <h2>Starting by % of downloaded over the rest of data </h2>

# In[9]:


print("The proportion of downloaded over just click: ")
print(round((df_talk.is_attributed.value_counts() / len(df_talk.is_attributed) * 100),2))
print(" ")
print("Downloaded over just clicks description: ")
print(df_talk.is_attributed.value_counts())

plt.figure(figsize=(8, 5))
sns.set(font_scale=1.2)
mean = (df_talk.is_attributed.values == 1).mean()

ax = sns.barplot(['Fraudulent (1)', 'Not Fradulent (0)'], [mean, 1-mean])
ax.set_xlabel('Target Value', fontsize=15) 
ax.set_ylabel('Probability', fontsize=15)
ax.set_title('Target value distribution', fontsize=20)

for p, uniq in zip(ax.patches, [mean, 1-mean]):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height+0.01,
            '{}%'.format(round(uniq * 100, 2)),
            ha="center") 


# We can see a very unbalanced dataset. Sample of 1 million and we have just 1693 or .17% of target to train the model. But, it's very normal when we are working with fraud datasets. <br>
# 
# Let's try take a best understand of this distribuition using another features. 

# <h2>Most Frequent IPs on dataset</h2>

# In[10]:


ip_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['ip'].value_counts()[:20]
ip_frequency_click = df_talk[df_talk['is_attributed'] == 0]['ip'].value_counts()[:20]

plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
g = sns.barplot(ip_frequency_downloaded.index, ip_frequency_downloaded.values, color='blue')
g.set_title("TOP 20 IP's where the click come from was downloaded",fontsize=20)
g.set_xlabel('Most frequents IPs',fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(ip_frequency_click.index, ip_frequency_click.values, color='blue')
g1.set_title("TOP 20 IP's where the click come from was NOT downloaded",fontsize=20)
g1.set_xlabel('Most frequents IPs',fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()


# We can see that 2 ip's have a almost 2 times the others ip's clicks, but it isn't very significant in the download rate, with just 9 downloads in total

# <h2>Taking a look on App feature</h2>

# In[11]:


app_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['app'].value_counts()[:20]
app_frequency_click = df_talk[df_talk['is_attributed'] == 0]['app'].value_counts()[:20]

plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
g = sns.barplot(app_frequency_downloaded.index, app_frequency_downloaded.values,
                palette='husl')
g.set_title("TOP 20 APP where the click come from and downloaded",fontsize=20)
g.set_xlabel('Most frequents APP ID',fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(app_frequency_click.index, app_frequency_click.values,
                palette='husl')
g1.set_title("TOP 20 APP where the click come from NOT downloaded",fontsize=20)
g1.set_xlabel('Most frequents APP ID',fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()


# It's very intereresting note that the app 19, 9 and 35 have the 3 highest numbers of downloads but no one appear's on the most frequent APP's.

# <h3>Percentual Distribuition of  App's</h3>

# In[12]:


print("App percentual distribuition description: ")
print(round(df_talk[df_talk['is_attributed'] == 1]['app'].value_counts()[:5]             / len(df_talk[df_talk['is_attributed'] == 1]) * 100),2)


# 
# 
# With first 5 highest values we have 64% of downloads total. 

# <h2>Channel feature </h2>
# - <i> channel is of  id of mobile ad publisher

# In[13]:


channel_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['channel'].value_counts()[:20]
channel_frequency_click = df_talk[df_talk['is_attributed'] == 0]['channel'].value_counts()[:20]

plt.figure(figsize=(16,10))

plt.subplot(2,1,1)
g = sns.barplot(channel_frequency_downloaded.index, channel_frequency_downloaded.values,                 palette='husl')
g.set_title("TOP 20 channels with download Count",fontsize=20)
g.set_xlabel('Most frequents Channels ID',fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(channel_frequency_click.index, channel_frequency_click.values,                 palette='husl')
g1.set_title("TOP 20 channels clicks Count",fontsize=20)
g1.set_xlabel('Most frequents Channels ID',fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()


# Let's take a look at channel proportion distribuition

# In[14]:


print("Channel percentual distribuition description: ")
print(round(df_talk[df_talk['is_attributed'] == 1]['channel'].value_counts()[:5]             / len(df_talk[df_talk['is_attributed'] == 1]) * 100),2)


# The top five highest channels corresponds to 59% of total downloads registereds in this sample. 

# <h2>Device Feature  </h2>

# In[15]:


device_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['device'].value_counts()[:20]
device_frequency_click = df_talk[df_talk['is_attributed'] == 0]['device'].value_counts()[:20]

plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
g = sns.barplot(device_frequency_downloaded.index, device_frequency_downloaded.values,
                palette='husl')
g.set_title("TOP 20 devices with download - Count",fontsize=20)
g.set_xlabel('Most frequents Devices ID',fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(device_frequency_click.index, device_frequency_click.values,
                palette='husl')
g1.set_title("TOP 20 devices with download - Count",fontsize=20)
g1.set_xlabel('Most frequents Devices ID',fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()


# We can see a clear difference in the data. Almost all data is from the same device type. 

# In[16]:


print("Device percentual distribuition: ")
print(round(df_talk[df_talk['is_attributed'] == 1]['device'].value_counts()[:5]             / len(df_talk[df_talk['is_attributed'] == 1]) * 100),2)


# The top 5 corresponds to 89% of our sample, but with significant values in just two variable... What corresponds to this values? I'm am very interested to understand. Why just two? 

# <h2>Operational System version (os) Feature</h2>

# In[17]:


os_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['os'].value_counts()[:20]
os_frequency_click = df_talk[df_talk['is_attributed'] == 0]['os'].value_counts()[:20]

plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
g = sns.barplot(os_frequency_downloaded.index, os_frequency_downloaded.values,
                palette='husl')
g.set_title("TOP 20 OS with download - Count",fontsize=20)
g.set_xlabel("Most frequents OS's ID",fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(os_frequency_downloaded.index, os_frequency_downloaded.values,
                palette='husl')
g1.set_title("TOP 20 OS with download - Count",fontsize=20)
g1.set_xlabel("Most frequents OS's ID",fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()


# In[18]:


print("Device percentual distribuition: ")
print(round(df_talk[df_talk['is_attributed'] == 1]['os'].value_counts()[:5]             / len(df_talk[df_talk['is_attributed'] == 1]) * 100),2)


# The first 5 highest values in this sample represents 55% of total downloads

# <h2>Let's take a look at our new features extracteds by time</h2>

# Visualizing the value's in hour column

# In[19]:


hour_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['hour'].value_counts()
hour_frequency_click = df_talk[df_talk['is_attributed'] == 0]['hour'].value_counts()

plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
g = sns.barplot(hour_frequency_downloaded.index, hour_frequency_downloaded.values,
                palette='husl')
g.set_title("Downloads Count by Hour",fontsize=20)
g.set_xlabel("Hour Download distribuition",fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(hour_frequency_click.index, hour_frequency_click.values,
                palette='husl')
g1.set_title("Clicks Count by Hour",fontsize=20)
g1.set_xlabel("Hour Click distribuition",fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()


# We can see a clear difference in distribuition of hours, but we need see with the full dataset to a betters understand. 

# ## Calculating new features using click_time
# 
# - First I will transform the click_time in nanosecs

# In[20]:


df_talk['click_nanosecs'] = (df_talk['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)


# In[21]:


df_talk['next_click'] = (df_talk.groupby(['ip', 'app', 'device', 'os']).click_nanosecs.shift(-1) - df_talk.click_nanosecs).astype(np.float32)


# In[22]:


df_talk['next_click'].fillna((df_talk['next_click'].mean()), inplace=True)


# In[ ]:





# <h2> Visualizing the minute column</h2> 

# In[23]:


minute_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['minute'].value_counts()
minute_frequency_click = df_talk[df_talk['is_attributed'] == 0]['minute'].value_counts()

plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
g = sns.barplot(minute_frequency_downloaded.index, minute_frequency_downloaded.values,
                palette='husl')
g.set_title("Downloads Count by Minute",fontsize=20)
g.set_xlabel("Minute Download distribuition",fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(minute_frequency_click.index, minute_frequency_click.values,
                palette='husl')
g1.set_title("Clicks Count by Minute",fontsize=20)
g1.set_xlabel("Minute Click distribuition",fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()


# Interesting that in the first minutes of an hour the rate of downloads is higher. We can see this in the both filters
# 

# <h2>Visualizing the Second distribuition.</h2>

# In[24]:


second_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['second'].value_counts()
second_frequency_click = df_talk[df_talk['is_attributed'] == 0]['second'].value_counts()

plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
g = sns.barplot(second_frequency_downloaded.index, second_frequency_downloaded.values,
                palette='husl')
g.set_title("Downloads Count by Hour",fontsize=20)
g.set_xlabel("Second Download distribuition",fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(second_frequency_click.index, second_frequency_click.values,
                palette='husl')
g1.set_title("Clicks Count by Hour",fontsize=20)
g1.set_xlabel("Second Click distribuition",fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()


# In seconds we see a little difference to a secnnd to a nother

# <h2>Feature Engineering in the categorical's features and IP</h2>
# - This is a copy of the brilliannnt kernel of user NanoMathias that you can see the kernel <a href="https://www.kaggle.com/nanomathias/feature-engineering-importance-testing"> here</a>, that is a lecture of feature engineering

# In[ ]:


import gc
#Define all the groupby transformations
GROUPBY_AGGREGATIONS = [
    
    # V1 - GroupBy Features #
    #########################    
    # Variance in day, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'day', 'agg': 'var'},
    # Variance in hour, for ip-app-os
    {'groupby': ['ip','app','os'], 'select': 'hour', 'agg': 'var'},
    # Variance in hour, for ip-day-channel
    {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'},
    # Count, for ip-day-hour
    {'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app
    {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},        
    # Count, for ip-app-os
    {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app-day-hour
    {'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Mean hour, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean'}, 
    
    # V2 - GroupBy Features #
    #########################
    # Average clicks on app by distinct users; is it an app they return to?
    {'groupby': ['app'], 
     'select': 'ip', 
     'agg': lambda x: float(len(x)) / len(x.unique()), 
     'agg_name': 'AvgViewPerDistinct'
    },
    # How popular is the app or channel?
    {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
    {'groupby': ['channel'], 'select': 'app', 'agg': 'count'},
    
    # V3 - GroupBy Features                                              #
    # https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977 #
    ###################################################################### 
    {'groupby': ['ip'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','day'], 'select': 'hour', 'agg': 'nunique'}, 
    {'groupby': ['ip','app'], 'select': 'os', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'device', 'agg': 'nunique'}, 
    {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','device','os'], 'select': 'app', 'agg': 'cumcount'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'}, 
    {'groupby': ['ip'], 'select': 'os', 'agg': 'cumcount'}, 
    {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'}    
]

# Apply all the groupby transformations
for spec in GROUPBY_AGGREGATIONS:
    
    # Name of the aggregation we're applying
    agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
    
    # Name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])
    
    # Info
    print("Grouping by {}, and aggregating {} with {}".format(
        spec['groupby'], spec['select'], agg_name
    ))
    
    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))
    
    # Perform the groupby
    gp = df_talk[all_features].         groupby(spec['groupby'])[spec['select']].         agg(spec['agg']).         reset_index().         rename(index=str, columns={spec['select']: new_feature})
        
    # Merge back to X_total
    if 'cumcount' == spec['agg']:
        df_talk[new_feature] = gp[0].values
    else:
        df_talk = df_talk.merge(gp, on=spec['groupby'], how='left')
        
     # Clear memory
    del gp
    gc.collect()


# Did it, let's see if the new features affects our results

# # Evaluating Feature Importance
#  - I'll fit xgBoost to the data, and evaluate the feature importances. First split into X and y

# In[ ]:


import xgboost as xgb

# Split into X and y
y = df_talk['is_attributed']
X = df_talk.drop('is_attributed', axis=1).select_dtypes(include=[np.number])

# Create a model
# Params from: https://www.kaggle.com/aharless/swetha-s-xgboost-revised
clf_xgBoost = xgb.XGBClassifier(
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.7,
    colsample_bylevel = 0.7,
    scale_pos_weight = 9,
    min_child_weight = 0,
    reg_alpha = 4,
    n_jobs = 4, 
    objective = 'binary:logistic'
)
# Fit the models
clf_xgBoost.fit(X, y)


# ## Verifying feature importances

# In[ ]:


from sklearn import preprocessing

# Get xgBoost importances
feature_importance = {}
for import_type in ['weight', 'gain', 'cover']:
    feature_importance['xgBoost-'+import_type] = clf_xgBoost.get_booster().get_score(importance_type=import_type)
    
# MinMax scale all importances
features = pd.DataFrame(feature_importance).fillna(0)
features = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(features),
    columns=features.columns,
    index=features.index
)

# Create mean column
features['mean'] = features.mean(axis=1)

# Plot the feature importances
features.sort_values('mean').plot(kind='bar', figsize=(16, 6))
plt.show()


# 

# I will continue with the conclusion on this kernel

# <h2>Fonts:</h2>
# I have implemented some some techniques from another kernels. Some of this:<br>
# https://www.kaggle.com/nanomathias/feature-engineering-importance-testing<br>
# https://www.kaggle.com/anokas/talkingdata-adtracking-edahttps://www.kaggle.com/anokas/talkingdata-adtracking-eda<br>
# https://www.kaggle.com/jtrotman/eda-talkingdata-temporal-click-count-plots<br>
# https://www.kaggle.com/chubing/feature-engineering-and-xgboost <br>
# and much others that I will also tag. 
#     
#     
# 

# In[ ]:





# In[ ]:





# In[ ]:




