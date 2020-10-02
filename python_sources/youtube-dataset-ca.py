#!/usr/bin/env python
# coding: utf-8

# ## Package load

# In[ ]:


import os
import json
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wordcloud

DATA_DIR = "/kaggle/input/youtube-new/"
COLOR = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
         'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


# ## Data load
# ### load function
#     On load_data function, I extract category title and category_id generate new dict. 
#     And scale function is the $$ X_{scale} = \frac{X - min(X)}{max(X)} $$

# In[ ]:


def load_data(region):
      videos_path = os.path.join(DATA_DIR, "%svideos.csv" %(region, ))
      category_path = os.path.join(DATA_DIR, "%s_category_id.json" %(region, ))
      df = pd.read_csv(videos_path, sep=',', header=0)
      with open(category_path, 'r') as f:
          items = json.load(f)['items']
          items_id = list(map(lambda x: x['id'], items))
          items_title = list(map(lambda x: x['snippet']['title'], items))
          ids_titles = {int(k):v for k, v in zip(items_id, items_title)}
      return df, ids_titles

def scale(df):
    return (df - np.min(df, axis=0, keepdims=True)) / np.max(df, axis=0, keepdims=True)


# ### from file load data

# In[ ]:


origin_CA, CA_category = load_data('CA')


# ## Data analysis
# ### preprocess
# 1. delete the category id on CA data set but not present on the CA_category info list.

# In[ ]:


CA = origin_CA.copy()
for i in origin_CA.category_id.unique():
    if i not in CA_category.keys():
        CA.drop(CA.loc[CA.category_id == i].index, inplace=True)
CA.set_index(np.arange(0, CA.shape[0]), inplace=True)


# 2. convert publish_time to datetime type, and modifyed trending_date to the correct datetime format.

# In[ ]:


CA.publish_time = pd.to_datetime(CA.publish_time, infer_datetime_format=True)
CA.trending_date = pd.to_datetime('20' + CA.trending_date, format='%Y.%d.%m',
                                  infer_datetime_format=True, utc=True)


# 3. generate new variable trending_period, which represent the video stay days on Youtube trending list. I also records datetime which video first time found stay on trending list.

# In[ ]:


trending_period = {}
CA_trending_date_begin = {}
I = pd.Index(CA.video_id)
for vid in I.unique():
    t = I.get_value(CA.trending_date, vid)
    if isinstance(t, pd.datetime):
        trending_period[vid] = 1
        CA_trending_date_begin[vid] = t
    else:
        trending_period[vid] = t.shape[0]
        CA_trending_date_begin[vid] = t.min()

CA['trending_period'] = CA.video_id.map(trending_period)


# 4. split data set.

# In[ ]:


def sample_data(df, train=0.9, val=0.1, test=0.1):
    df_height = df.shape[0]
    train_num = int(df_height * 0.9)
    train_set = df.iloc[:int(train_num*0.9)]
    val_set = df.iloc[int(train_num*0.9) : train_num]
    test_set = df.iloc[train_num:]
    return train_set, val_set, test_set

def select_label(df, y):
    Y = df[y]
    X = df.drop(y, axis=1)
    return X, Y

CA_train, CA_vali, CA_test = sample_data(CA)
CA_xtrain, CA_ytrain = select_label(CA_train, 'category_id')
CA_xvali, CA_yvali = select_label(CA_vali, 'category_id')
CA_xtest, CA_ytest = select_label(CA_test, 'category_id')


# ### Describe analysis
# 1. explore the count for every category.

# In[ ]:


video_id_unique_idx = CA.video_id.drop_duplicates().index
CA_category_count = CA.loc[video_id_unique_idx].groupby('category_id').video_id.count()

plt.figure(figsize=(9.0, 6.0), facecolor='white')
x = list(range(CA_category_count.shape[0]))
plt.bar(x, CA_category_count, width=0.8)
xlabel = CA_category_count.index.map(CA_category)
plt.xticks(x, labels=xlabel, rotation=90)
plt.xlabel('Category')
plt.ylabel('Unique count')
plt.title('Count each category display on YouTube trend')
plt.show()


# 2. explore the average days on trending list for every category.

# In[ ]:


category_trending_period_mean = CA.groupby(['category_id']).trending_period.mean()

plt.figure(figsize=(9.0, 6.0), facecolor='white')
x = list(range(category_trending_period_mean.shape[0]))
plt.plot(x, category_trending_period_mean, 'o-')
xlabel = category_trending_period_mean.index.map(CA_category)
plt.xticks(x, labels=xlabel, rotation=90)
plt.ylim(bottom=1)
plt.grid(True)
plt.show()


# 3. the time between publish time to first time present on trending list.

# In[ ]:


CA_between_time = CA.video_id.map(CA_trending_date_begin) - CA.publish_time
days = pd.Series([i.days for i in CA_between_time])
CA['between_time'] = days
days_count = days.value_counts()[:10]

plt.figure(figsize=(9.0, 6.0), facecolor='white')
plt.bar(days_count.index, days_count)
plt.xticks(days_count.index)
plt.grid(True)


# 4. explore every category average time from publish to become trending.

# In[ ]:


CA_category_between_time_mean = CA.loc[:, ['category_id', 'between_time']].groupby('category_id').mean()
CA_category_between_time_mean.sort_values(by='between_time', ascending=False, inplace=True)

plt.figure(figsize=(9.0, 6.0), facecolor='white')
x = list(range(CA_category_between_time_mean.shape[0]))
plt.plot(x, CA_category_between_time_mean, 'o-')
plt.xticks(x, labels=list(map(lambda x: CA_category[x], CA_category_between_time_mean.index)), rotation=90)
plt.grid(True)


# 5. explore the correlation on every numerical feature.

# In[ ]:


corr = CA.corr()
x = list(range(corr.shape[0]))

plt.style.use('ggplot')
plt.figure(figsize=(9.0, 6.0))
plt.imshow(CA.corr(), cmap=plt.cm.Reds, interpolation='nearest')
plt.colorbar()
plt.xticks(x, labels=corr.columns, rotation=90)
plt.yticks(x, labels=corr.columns)
plt.show()


# 6. Top 10 category on CA.

# In[ ]:


top_ten_category_id = CA_category_count.sort_values(ascending=False)[:10]
idx = CA_ytrain.apply(lambda x: x in top_ten_category_id.keys())
CAv_xtrain = CA_xtrain.loc[idx]
CAv_ytrain = CA_ytrain.loc[idx]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(9.0, 6.0))
for i, vid in enumerate(top_ten_category_id.keys()):
    idx = CAv_ytrain == vid
    x = CAv_xtrain.views[idx]
    y = CAv_xtrain.likes[idx]
    ax.scatter(x, y, c=COLOR[i % 10], alpha=0.5, label=CA_category[vid])
ax.legend()
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('views')
ax.set_ylabel('likes')
fig.tight_layout(pad=0.0)
plt.show()


# 7. wordcloud for tags.

# In[ ]:


tags = CA.loc[CA.tags.str.find('none') == -1].tags
CA_tags = {}
for x in tags:
    words = x.split('|')
    for w in words:
        w = w.lower().replace('\"', '')
        CA_tags[w] = CA_tags.get(w, 0) + 1

CA_wordcloud = wordcloud.WordCloud(background_color='white')
CA_wordcloud.generate_from_frequencies(CA_tags)

plt.figure(figsize=(9.0, 6.0))
plt.imshow(CA_wordcloud)
plt.axis('off')
plt.tight_layout(pad=0.0)
plt.show()

