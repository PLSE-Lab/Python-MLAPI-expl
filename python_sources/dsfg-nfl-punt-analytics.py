#!/usr/bin/env python
# coding: utf-8

# ## 1. Import Data
# Refer to https://www.kaggle.com/c/NFL-Punt-Analytics-Competition for details of the data resource

# In[ ]:


import pandas as pd
import numpy as np
import dask.dataframe as dd


# In[ ]:


# Load raw data
game = pd.read_csv('../input/NFL-Punt-Analytics-Competition/game_data.csv')
play_info = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv',index_col=['GameKey','PlayID'])
play_role = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')
player_punt = pd.read_csv('../input/NFL-Punt-Analytics-Competition/player_punt_data.csv')
video = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv',index_col=['GameKey','PlayID'])

# Combine all NGS files to parq
def ngs():
    ndtypes = {'GameKey': 'int16',         
           'PlayID': 'int16',         
           'GSISID': 'float32',                
           'x': 'float32',         
           'y': 'float32',         
           'dis': 'float32',
           'o': 'float32'}
    nddf = dd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS*', 
                    usecols=[n for n in ndtypes.keys()], dtype=ndtypes)
    nddf['GSISID'] = nddf.GSISID.fillna(-1).astype('int32')
    df = nddf.groupby(['GameKey','PlayID','GSISID']).mean()
    df = df.compute()
    df.to_parquet('../input/NFL-Punt-Analytics-Competition/NGS.parq')


# ## 2. Process Data
# ### 2.1 Classify play by punt types

# In[ ]:


type_reg_mapping = [("not_punted", "(no play)|(delay of game)|(false start)|blocked|incomplete"),
                     ("out_of_bounds", "out of bounds"), 
                     ("downed","downed"),        
                     ("touchback","touchback"),
                     ("fair_catch","fair catch"),
                     ("returned","(no gain)|(for.*yard)")
                    ]

for t,r in type_reg_mapping:
    play_info[t]=play_info["PlayDescription"].str.contains(r,case=False)


# ## 3. EDA
# 
# Show concussion stats in terms of different aspects.

# In[ ]:


import matplotlib.pyplot as plt


# ### 3.1 Counts per year
# 

# In[ ]:


video["Season_Year"].plot.hist()
plt.show()


# ### 3.2 Turf, Weather, Temperature

# In[ ]:


temp = video.merge(game[['GameKey','Turf','GameWeather','Temperature']],how='left',left_on='GameKey',right_on='GameKey')
fig=plt.figure(figsize=[20,5])
fig.add_subplot(131)
temp['Temperature'].value_counts().sort_index().plot()
fig.add_subplot(132)
temp['Turf'].str.strip().str.lower().value_counts().plot.pie()
fig.add_subplot(133)
temp['GameWeather'].str.strip().str.lower().value_counts().plot.pie()
plt.show()


# ### 3.3 Player Position and Role
# 

# In[ ]:


fig=plt.figure(figsize=[10,5])
fig.add_subplot(121)
video.merge(player_punt,left_on='GSISID',right_on='GSISID',how='left')["Position"].value_counts().plot.pie()
fig.add_subplot(122)
video.merge(play_role,left_on='GSISID',right_on='GSISID',how='left')["Role"].value_counts().plot.pie()
plt.show()


# ### 3.4 Player and Partner Activity
# 

# In[ ]:


playerA = video['Player_Activity_Derived'].value_counts().sort_index()
partnerA = video['Primary_Partner_Activity_Derived'].value_counts().sort_index()
pd.DataFrame({'Player':playerA,'Partner':partnerA},index=partnerA.index).plot.bar()
plt.show()


# ### 3.5 Turnover and Friendly Fire

# In[ ]:


fig=plt.figure(figsize=[10,5])
fig.add_subplot(121)
video["Turnover_Related"].value_counts().plot.pie()
fig.add_subplot(122)
video["Friendly_Fire"].value_counts().plot.pie()
plt.show()


# ### 3.6 Primary Impact Type
# 

# In[ ]:


video["Primary_Impact_Type"].value_counts().plot.barh()
plt.show()


# ### 3.7 Punt Type

# In[ ]:


xTick=["not_punted","out_of_bounds","downed","touchback","fair_catch","returned"]
xidx=range(len(xTick))
t=[]
f=[]

for s in xTick:
    v = video.merge(play_info,on=['GameKey','PlayID'],how='left')[s].value_counts().values
    if len(v)>1:
        t.append(v[1])
    else:
        t.append(0)
    f.append(v[0])

p1=plt.bar(xidx,t)
p2=plt.bar(xidx,f,bottom=t)
plt.xticks(xidx, xTick, rotation='vertical')
plt.legend((p1[0],p2[0]),('True','False'),bbox_to_anchor=(1,.5))
plt.show()


# ### 3.8 Return Yard

# In[ ]:


play_info['yardage'] = play_info['PlayDescription'].str.extract(r'for (\d.) yard')
play_info.loc[play_info['yardage'].isnull(),'yardage']=0
play_info.loc[play_info['returned'].isnull(),'yardage']=0

video.merge(play_info,on=['GameKey','PlayID'],how='left').loc[~play_info['returned'].isnull(),'yardage'].value_counts().plot.barh()
plt.show()


# ### 3.9 Punt Type Mapping

# In[ ]:


import holoviews as hv
hv.extension('bokeh')


# In[ ]:


def getEdges():
    nodes1=['returned','downed','fair_catch']
    nodes2=['no_returned','no_downed','no_fair_catch']
    edges = [(x,y) for x in nodes1 for y in nodes2]
    result=[]
    d=video.merge(play_info,on=['GameKey','PlayID'],how='left')[['returned','downed','fair_catch']]
    for (n1,n2) in edges:
        n2_col=[x for x in nodes1 if n2.find(x)>-1][0]
        if n1!=n2_col:
            result.append([n1,n2,len(d.loc[(d[n1]==True)&(d[n2_col]==True)])])
        else:
            result.append([n1,n2,0])
    return result


edges=getEdges()
sankey=hv.Sankey(edges)
display(sankey)


# ## 4. Clustering of NGS

# In[ ]:


# Load parq data
ngs = pd.read_parquet('../input/dsfg-ngs/NGS.parq')
ngs2 = ngs.groupby(['GameKey','PlayID','GSISID']).mean()


# ### 4.1 PCA

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer


# In[ ]:


pca = PCA(n_components=4)
train = StandardScaler().fit_transform(ngs2.values)
train= Normalizer().fit_transform(train)
pca_result = pca.fit_transform(train)
print(pca.explained_variance_ratio_)
plt.scatter(pca_result[:,0],pca_result[:,1])
plt.show()


# ### 4.2 TSNE

# In[ ]:


from sklearn.manifold import TSNE
np.random.seed(42)


# In[ ]:


N=1000
rndperm = np.random.permutation(ngs2.shape[0])
ngs2_sub = ngs2.loc[rndperm[:N],:].copy()
train = StandardScaler().fit_transform(ngs2_sub.values)
train= Normalizer().fit_transform(train)
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(train)
plt.scatter(tsne_results[:,0],tsne_results[:,1])
plt.show()


# ### 4.3 TSNE of ngs with consussion

# In[ ]:


df = video.merge(ngs2,on=['GameKey','PlayID','GSISID'],how='left')[['x','y','dis','o']]
rndperm = np.random.permutation(ngs2.shape[0])
ngs2_sub = ngs2.iloc[rndperm[:N]][['x','y','dis','o']]
train = pd.concat([ngs2_sub,df])

train = StandardScaler().fit_transform(train.values)
train= Normalizer().fit_transform(train)
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(train)
plt.scatter(tsne_results[:N,0],tsne_results[:N,1],facecolor='blue')
plt.scatter(tsne_results[N:,0],tsne_results[N:,1],facecolor='red')
plt.show()


# In[ ]:




