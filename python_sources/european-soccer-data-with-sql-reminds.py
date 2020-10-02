#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


path = '../input/'
database = path + 'database.sqlite'
conn = sqlite3.connect(database)

tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", conn)
tables


# In[ ]:


player = pd.read_sql("SELECT * FROM Player",conn)


# In[ ]:


player_att = pd.read_sql("SELECT * FROM Player_Attributes",conn)


# In[ ]:


player_att.dropna(inplace=True)


# In[ ]:


player_att = player_att.drop_duplicates(subset='player_api_id')


# In[ ]:


player_att.overall_rating.describe()


# In[ ]:


plt.figure(figsize=(12,8))
plt.title("Overall Rating Distributions",fontdict={'fontsize':15})
plt.xlabel("Overall")
plt.ylabel("Ratio")
sns.distplot(list(player_att.overall_rating),bins=100)


# In[ ]:


player_att.columns


# In[ ]:


player_att.dropna(inplace=True)


# In[ ]:


player_att.head()


# In[ ]:


model = KMeans(n_clusters=5)
model.fit(player_att[['overall_rating','potential']])
predict = pd.DataFrame(model.predict(player_att[['overall_rating','potential']]))
predict.columns=['predict']
plt.figure(figsize=(8,7))
plt.scatter(x=player_att.overall_rating,y=player_att.potential,c=predict.predict)
plt.title("Clustering Using Overall Ratings and Potential Scores",fontdict={'fontsize':15})
plt.ylabel("Potential")
plt.xlabel("Overall_Rating")


# In[ ]:


player_att.columns


# In[ ]:


plt.figure(figsize=(10,8))
corr = player_att.corr()
sns.heatmap(corr)


# In[ ]:


forward_features = ['finishing','volleys']
mid_features = ['short_passing','vision']
defender_features = ['standing_tackle','sliding_tackle']
gk_features = ['gk_diving','gk_handling','gk_kicking','gk_positioning']

"""
def position_column(position):
    feature = str(position)+'_features'
    player_att[str(position)] = player_att[x for x in feature]
    """


# In[ ]:


print(map(sum,player_att[forward_features]))


# In[ ]:


player_att['forward'] = (player_att[forward_features].iloc[:,0] + player_att[forward_features].iloc[:,1])/2
player_att['mid'] = (player_att[mid_features].iloc[:,0] + player_att[mid_features].iloc[:,1])/2
player_att['defender'] = (player_att[defender_features].iloc[:,0] + player_att[defender_features].iloc[:,1])/2
player_att['gk'] = (player_att[gk_features].iloc[:,0] + player_att[gk_features].iloc[:,1] +
                   player_att[gk_features].iloc[:,2]+player_att[gk_features].iloc[:,3])/4


# In[ ]:


player_att


# ## Clustering After Feature engineering:
# ### In my opinion, below's features are well representing the each position's characteristics
# ### Forward : Finishing, Volleys, Shot_Power
# ### Mid-fielder : Short_Passing, Vision
# ### Defender : Standing_Tackle, Sliding_Tackle
# ### Goalkeeper : GK_handling, GK_positioning, GK_reflexes
# 
# #### So, I will average this sub-features into position columnsb

# In[ ]:


def cluster_position(pos1,pos2):
    model = KMeans(n_clusters=4)
    model.fit(player_att[['forward','mid','defender','gk']])
    predict = pd.DataFrame(model.predict(player_att[['forward','mid','defender','gk']]))
    predict.columns=['predict']
    plt.figure(figsize=(8,7))
    plt.scatter(x=player_att[str(pos1)],y=player_att[str(pos2)],c=predict.predict,marker='o')
    plt.title("Clustering Using "+pos1+" Stats "+pos2+" Stats",fontdict={'fontsize':15})
    plt.ylabel(pos2)
    plt.xlabel(pos1)


# In[ ]:


cluster_position('forward','mid')
cluster_position('mid','defender')
cluster_position('forward','defender')
cluster_position('gk','forward')


# In[ ]:


def find_player(name):
    id = player[player.player_name==name]['player_api_id']
    return(player_att[player_att.player_api_id==int(id)])


# In[ ]:


find_player("Lionel Messi")


# In[ ]:


find_player("Aaron Doran")


# In[ ]:


country = pd.read_sql("""SELECT * FROM Country""",conn)
country


# In[ ]:


country

