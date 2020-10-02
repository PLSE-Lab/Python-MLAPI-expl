#!/usr/bin/env python
# coding: utf-8

# ## First we load the important libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools # use chain method to efficiently flatten list of lists 
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Reading the data naming the columns and dropping the columns without data 

# In[ ]:


# header=None --> there are no headers in data
colnames=['user_id','game_title', 'behavior_name','value','temp'] 
users = pd.read_csv("/kaggle/input/steam-video-games/steam-200k.csv",header=None,names=colnames,usecols=['user_id','game_title', 'behavior_name','value'])


# In[ ]:


pd.set_option('display.max_rows', 500)
users[users.user_id==151603712]


# ### let's check if there are Null values
# 
# 

# In[ ]:


users.isnull().values.any()


# ### No Nulls

# In[ ]:


users.info()


# ### Lets check how many users we have in the DF

# In[ ]:


users.user_id.nunique()


# ### Lets check how many games we have in the DF

# In[ ]:


users.game_title.nunique()


# ## Handling Duplicated values since we asume we should have only two rows max per user per game one for Purchase and one for Play
# ### there might be duplicates purchase so in this case we will keep only one and duplicated playtimes and in this case we have decieded also to keep only the first
# > Play is the SUM of all times this user playes this game

# 

# In[ ]:


users.drop_duplicates(subset=['user_id','game_title','behavior_name'], keep='first', inplace=True)


# In[ ]:


users.info()


# In[ ]:


print ("number of rows we have droped",200000-199281)
print ("number of games",users.game_title.nunique(),"number of users",users.user_id.nunique())


# ### we see that we have deleted 719 lines no users or games are missing

# In[ ]:


# user statistics:
user_games_num = users.game_title.nunique()
user_play_hrs = users[users.behavior_name=='play'].value.sum()
user_play_avg = users[users.behavior_name=='play'].value.mean()
user_play_median = users[users.behavior_name=='play'].value.median()
user_purchase_num = users[users.behavior_name=='purchase'].value.sum()
user_purchase_avg = user_purchase_num / user_games_num

print('number of games played:',user_games_num)
print('number of total hours played: %.2f'%(user_play_hrs))
print('number of avg hours played: %.2f'%(user_play_avg))
print('number of median hours played: %.2f'%(user_play_median))
print('number of purchases: %.0f'%(user_purchase_num))
print('number of purchases per game: %.0f'%(user_purchase_avg))


# In[ ]:


# every game involves a purchase event

x = users.groupby(['user_id','behavior_name']).count()


# In[ ]:


# aggregate games by total play hrs / mean play hrs / median play hrs

game_stats = users[users.behavior_name == 'play'].groupby(['game_title']).agg({'value':[np.sum,np.mean,np.median]})

game_stats.columns = ['_'.join(col).strip() for col in game_stats.columns.values] # flatten hierarchy in column multi-index


# In[ ]:


# top 15 games in total play hrs
game_stats.value_sum.sort_values(ascending=False).head(15)


# In[ ]:


# top 15 games in avg play time
game_stats.value_mean.sort_values(ascending=False).head(15)


# In[ ]:


# top 15 games in median play time
game_stats.value_median.sort_values(ascending=False).head(15)


# In[ ]:


# consider top 10 games from each statistic

a = [game_stats[col].sort_values(ascending=False).head(10).index.tolist() for col in game_stats.columns.values]

top_tens = list(set(itertools.chain.from_iterable(a))) # join top 10 games by each statistic 

top_tens.sort(key=str.casefold) # sort alphabetically


# In[ ]:


total_playtime = game_stats.loc[top_tens,'value_sum']
mean_playtime = game_stats.loc[top_tens,'value_mean']
median_playtime = game_stats.loc[top_tens,'value_median']


# In[ ]:


# plot top 10 games for each statistic

plt.figure(figsize=(15,15),facecolor="w")
plt.xticks(rotation='vertical')
ax1 = plt.subplot(3,1,1)
ax2 = plt.subplot(3,1,2,sharex=ax1)
ax3 = plt.subplot(3,1,3,sharex=ax1)
ax1.plot(total_playtime,color='g')
ax2.plot(mean_playtime,color='r')
ax3.plot(median_playtime,color='b')
ax1.set_ylabel('total playtime')
ax2.set_ylabel('avg playtime')
ax3.set_ylabel('median playtime')
# ax2.set_xlabel("x Axis")
plt.setp(ax1.get_xticklabels(),visible=False)
plt.setp(ax2.get_xticklabels(),visible=False)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=90 )

plt.tight_layout()
plt.show()

# NOTE: we have two clear outliers we might want to treat seperately


# In[ ]:


ax = users[users.behavior_name=="purchase"].groupby(['user_id']).count().sort_values(by='value',  axis=0, ascending=False).value.plot.hist(bins=100,figsize=(18,9))
# plt.xlim(2, 200)
ax.set(yscale ='log')
# users.groupby(['user_id']).count().sort_values(self, by, axis=0, ascending=True, inpla)


# In[ ]:


ax=users[users.behavior_name=="play"].groupby(['user_id']).sum().sort_values(by='value',  axis=0, ascending=False).value.plot.hist(bins=150,figsize=(18,9))
# plt.xlim(2, 2000)
ax.set(yscale ='log')


# ### Extract top  games in terms of users.

# In[ ]:


top_games = users.groupby(['game_title'])['user_id'].count().sort_values(ascending=False)
top_games = top_games/2


# In[ ]:


# need to devied by 2 since we have 2 rows for each game we should filter only purchase

ax = top_games[0:15].plot.bar(figsize=(14,7))
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.001, p.get_height() * 1.02))


# > ### Filter data for rows with top 10 games only

# In[ ]:


users = users[users.game_title.isin(top_games[0:10].index)]


# In[ ]:


users.shape


# In[ ]:


users.user_id.nunique()


# ### using this technique will allow us to cluster only 8810 out of 12393, There are ~4k users we will not be able to cluster we will deal with it later

# In[ ]:


users.game_title.nunique()


# ### as expected we have 10 games 
# 
# ### Now we need to remove outliers
# I will create a small program that will handle only the numbers and will remove only high values

# In[ ]:


from pandas.api.types import is_numeric_dtype
def remove_outlier(df):
    low = .05
    high = .95
    quant_df = df.quantile([low, high])
    for name in list(df.columns):
        if is_numeric_dtype(df[name]):
            df = df[ (df[name] < quant_df.loc[high, name])]
    return df


# In[ ]:


#now we need to check the types of the variables in the users
users.info()


# In[ ]:


# we need to change the type of the user_id to be string
users.user_id=users.user_id.astype('str')


# In[ ]:


#now we need to check the types of the variables in the users
users.info()


# In[ ]:


users = remove_outlier(users)   


# In[ ]:


users.shape


# > ### Building the DF (Users) for clustering we need to have one row per user per game that will include the purchase and the play time in one row

# In[ ]:


from functools import reduce
df = {}

# top 10 games in numbers of users
games = top_games[0:10].index

for i, game in enumerate(games):
    df_purchase = users[(users.game_title == game) & (users.behavior_name == 'purchase')][['user_id','value']]
    df_play = users[(users.game_title == game) & (users.behavior_name == 'play')][['user_id','value']]
    df_game = pd.merge(left = df_purchase, right = df_play, how = 'left', on = ['user_id'], suffixes= ('_purchase','_playtime'))
    df_game.rename(columns={'value_purchase':'purchase_'+str(i),'value_playtime':'playtime_'+str(i)},inplace=True)
    df[game] = df_game

df_final = reduce(lambda left,right: pd.merge(left,right,how='outer',on=['user_id'],suffixes=('',''),sort=False), df.values())

df_final.fillna(value=0,inplace=True)


# In[ ]:


df_final.head()


# In[ ]:





# ## we need to scale since the values of the value might be very high

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


df_final_scale = scaler.fit_transform(df_final)

df_final_scale = pd.DataFrame(df_final_scale,columns = df_final.columns)


# In[ ]:


df_final_scale


# ## Clustering set up

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
from sklearn.cluster import KMeans
import time
from pandas.plotting import scatter_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context('talk')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}


# In[ ]:


k = 5
model = KMeans(n_clusters=k, 
               max_iter=10, random_state=1, 
               init='k-means++', n_init=10)
df_final_scale['cluster'] = pd.Series(model.fit_predict(df_final_scale))


# In[ ]:


model.labels_


# In[ ]:


model.cluster_centers_


# In[ ]:


def calc_inertia(k):
    model = KMeans(n_clusters=k).fit(df_final_scale)
    return model.inertia_

inertias = [(k, calc_inertia(k)) for k in range(1, 10)]


# In[ ]:


plt.figure(figsize=(12, 8))
plt.plot(*zip(*inertias))
plt.title('Inertia vs. k')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.grid(True)
plt.text(0.5, 0.120, '*K=5 looks promissing', horizontalalignment='center',verticalalignment='center',color='blue', transform=ax.transAxes)
plt.text(0.25, 0.4, '*K=2 is too small', horizontalalignment='center',verticalalignment='center',color='red', transform=ax.transAxes)


# In[ ]:


# data[['user_id','value_y','cluster']]
# scatter = scatter_matrix(data[['user_id','value_y','cluster']] ,figsize=(15, 10), s=22 ,alpha=1)
if 'cluster' in df_final_scale.columns:
    df_final_scale.drop(['cluster'], axis=1,inplace=True)
k = 5
model = KMeans(n_clusters=k, 
               max_iter=10, random_state=1, 
               init='k-means++', n_init=10)
df_final_scale['cluster'] = pd.Series(model.fit_predict(df_final_scale))



# In[ ]:


df_final_scale.cluster.value_counts()


# In[ ]:


sns.set_context('paper')
ax = df_final_scale.cluster.value_counts().plot.bar(figsize=(14,7))
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.001, p.get_height() * 1.02))


# In[ ]:


sns.set_context('paper')
data = df_final_scale[['cluster','purchase_0','purchase_1','playtime_0','playtime_1']]
scatter = scatter_matrix(data ,figsize=(10, 8), s=22 ,alpha=.7)
sns.set_context('talk')


# In[ ]:


df_final_scale


# In[ ]:


for i in df_final_scale.cluster.unique():
    print("this is the statistics for cluster ",i)
    print(df_final_scale[df_final_scale.cluster==i].describe())


# ## Hdbscan Clustering

# In[ ]:


pip install hdbscan


# In[ ]:


import hdbscan


# In[ ]:


if 'cluster' in df_final_scale.columns:
    df_final_scale.drop(['cluster'], axis=1,inplace=True)
clusterer = hdbscan.HDBSCAN(min_cluster_size=200)
cluster_labels = clusterer.fit_predict(df_final_scale)


# In[ ]:


cluster_labels 


# In[ ]:


df_final_scale['cluster'] = cluster_labels


# In[ ]:


df_final_scale


# In[ ]:


df_final_scale.cluster.value_counts()


# In[ ]:


sns.set_context('paper')
ax = df_final_scale.cluster.value_counts().plot.bar(figsize=(14,7))
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.001, p.get_height() * 1.02))


# In[ ]:


df_final_scale[df_final_scale.cluster==5]


# In[ ]:


df_final_scale[df_final_scale.cluster==5].describe()


# In[ ]:


for i in df_final_scale.cluster.unique():
    print("this is the statistics for cluster ",i)
    print(df_final_scale[df_final_scale.cluster==i].describe())


# ## Agglomerative Clustering

# In[ ]:


from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


# In[ ]:


if 'cluster' in df_final_scale.columns:
    df_final_scale.drop(['cluster'], axis=1,inplace=True)
Z = linkage(df_final_scale, method='ward', metric='euclidean')


# In[ ]:


dn = dendrogram(Z)


# In[ ]:


df_final_scale['cluster'] = fcluster(Z, 5, criterion='maxclust')
df_final_scale.plot('playtime_0', 'playtime_1', kind='scatter', c=df_final_scale['cluster'], s=100)


# In[ ]:


for i in df_final_scale.cluster.unique():
    print("this is the statistics for cluster ",i)
    print(df_final_scale[df_final_scale.cluster==i].describe())


# In[ ]:





# In[ ]:




