#!/usr/bin/env python
# coding: utf-8

# [](http://)

# <img src='https://blogdovladimir.files.wordpress.com/2017/03/terrorismo.jpg' style='height:400px'>

# <div class="list-group" id="list-tab" role="tablist">
#   <h1 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">&nbsp;Summary Table:</h1>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#1" role="tab" aria-controls="profile">1. Introduction<span class="badge badge-primary badge-pill">1</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#2" role="tab" aria-controls="profile">2. Outlier Detection and Pattern Recognition<span class="badge badge-primary badge-pill">2</span></a>
#    <a class="list-group-item list-group-item-action" data-toggle="list" href="#3" role="tab" aria-controls="profile">3. Clustering<span class="badge badge-primary badge-pill">3</span></a>
#    <a class="list-group-item list-group-item-action" data-toggle="list" href="#4" role="tab" aria-controls="profile">4. Conclusion<span class="badge badge-primary badge-pill">4</span></a>
# </div>

# <font size="+3" color="black"><b>1 - Introduction</b></font><br><a id="1"></a>
# 
# 
# * This kernel has propose to identify nonstandard attacks of 10 principal terrorist groups around the world and compare similarity over each group attack
# * Analyze target, country and country used in attack logs
# * To identify nonstandard attacks I will use OneClassSVM to outlier detection in the 10 terrorist groups
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
get_ipython().system('pip3 install pyod  ')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df_terrorism = pd.read_csv('/kaggle/input/gtd/globalterrorismdb_0718dist.csv', encoding = "ISO-8859-1")
df_terrorism = df_terrorism[['iyear','imonth','iday','extended','country','country_txt','region', 'region_txt','success',  'attacktype1_txt', 'targtype1', 'targtype1_txt', 'gname', 'weaptype1_txt']]
df_terrorism.head()


# In[ ]:


df_terrorism.head()


# In[ ]:


count = df_terrorism.pivot_table(columns='gname', aggfunc='size', fill_value=0)
terror_gname = dict(zip(count.index, count[:]))
terror_gname = sorted(terror_gname.items(), key=lambda kv: kv[1], reverse=True)
terror_gname = dict(terror_gname)
terror_gname_11_keys = list(terror_gname.keys())
terror_gname_10_values = list(terror_gname.values())
terror_gname_10_values = terror_gname_10_values[:10]
names =terror_gname_11_keys[0:11]


# In[ ]:


countries = pd.get_dummies(df_terrorism['country_txt'])
countries.reset_index(drop=True, inplace=True)
regions = pd.get_dummies(df_terrorism['region_txt'])
regions.reset_index(drop=True, inplace=True)
attack = pd.get_dummies(df_terrorism['attacktype1_txt'])
attack.reset_index(drop=True, inplace=True)
target = pd.get_dummies(df_terrorism['targtype1_txt'])
target.reset_index(drop=True, inplace=True)
weapon = pd.get_dummies(df_terrorism['weaptype1_txt'])
weapon.reset_index(drop=True, inplace=True)
df_terrorism.reset_index(drop=True, inplace=True)
df_terrorism_new = pd.concat([df_terrorism, countries, regions, attack, target, weapon], axis=1)
df_terrorism_new = df_terrorism_new.drop(['iyear','imonth','iday','country_txt', 'region_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'] ,axis=1)


# <font size="+3" color="black"><b>2 - Outlier Detection and Pattern Recognition</b></font><br><a id="2"></a>
# 
# * Use the OCSVM to detect nonstandard attack in each of 10 most dangerous groups
# 
# * To more details of patterns groups attacks click over cicles that will expand the region to see better how forms of attacks are composed

# In[ ]:


from pyod.models.ocsvm import OCSVM
from tqdm import tqdm
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

def get_indexs(name, df):
    tmp = df[df['gname']==name]
    X_tmp = tmp.iloc[:,~tmp.columns.isin(['gname'])]
    clf = OCSVM(gamma='auto').fit(X_tmp)
    tmp['outlier'] = clf.labels_
    indexs_out  = tmp[tmp['outlier']==1].index
    indexs_in = tmp[tmp['outlier']==0].index
    return indexs_out, indexs_in

indexs_out, indexs_in = get_indexs(names[1], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()


# # Taliban Attacks
# 
# * It's not normal Taliban actions out of Afghanistan.
# 
# * In Afghanistan generally dont carry out attacks on transports.
# 
# * The Taliban when using armed attack is generally not used in structures such as business, transport and non-state militia. 
# 
# * telecommunications systems are not targets of the Taliban.
# 
# * Taliban do not take hostages in business buildings, the targets are usually state buildings.
#  

# In[ ]:


indexs_out, indexs_in = get_indexs(names[2], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()


# # Islamic State of Iraq and the Levant (ISIL)
# 
# * Operation of ISIL attacks happen mainly in Iraq and Syria.
# 
# * Main weapon used is bombing explosion.
# 
# * Syria attacks pattern are focus on private structures against civilians, rarely attack focus government military bodies.

# In[ ]:


indexs_out, indexs_in = get_indexs(names[3], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()


# # Shining Path (SL)
# 
# * Actions are in Peru.
# 
# * In Peru Shining Path dont attack with explosives utilities buildings.

# In[ ]:


indexs_out, indexs_in = get_indexs(names[4], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()


# # Farabundo Marti National Liberation Front (FMLN)
# 
# * El Salvador is the principal country of operation.
# 
# * Explode business building is not normal operation of FMLN in El Salvador.
#  
# * FMLN use Bombing/Explosion in Utilities.
#  
# * FMLN uses firearm attacks mainly against military structures, perhaps causing fire exchanges between group members and military defense organizations.
# 
# * In general FMLN does not target business. 

# In[ ]:


indexs_out, indexs_in = get_indexs(names[5], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()


# # Al-Shabaab
# 
# * Somalia and Kenya are the principal countries. 
# 
# * As the main weapon used is the use of bombs in Kenya use armed assault is out of standard.
# 
# * Al-Shabaab attack mainly government military forces, their attacks appear to be carried out as if they were rebel or separatist government forces.

# In[ ]:


indexs_out, indexs_in = get_indexs(names[6], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()


# # New People's Army (NPA)
# 
# * Act exclusively in the philippines.
# 
# * Firearms are the main weapon.

# In[ ]:


indexs_out, indexs_in = get_indexs(names[7], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()


# # Irish Republican Army (IRA)
# 
# * United Kingdom is the principal country target
# * Explosives and firearms are the main weapons used
# 

# In[ ]:


indexs_out, indexs_in = get_indexs(names[8], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()


# # Revolutionary Armed Forces of Colombia (FARC)
# 
# * Presents a well-defined pattern.
# 
# * Operation is basically in Colombia.
# 
# * The most used weapon in FARC attacks are explosive and fire arms.

# In[ ]:


indexs_out, indexs_in = get_indexs(names[9], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()


# # Boko Haram
# 
# * Operation main in Nigeria, Cameroon is the secound country where Boko Haram operates.
# 
# * The pattern of attacks at Cameroon is with explosives focus in Private Citizens and Properties, any other form of action other than in this way evades the attacks carried out until then.
# 
# * Firearms,explosives and Incediary are the major weapons used by Boku Haram.

# In[ ]:


indexs_out, indexs_in = get_indexs(names[10], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()


#  

#  # Kurdistan Workers' Party (PKK)
#  
#  * Main Country action is Turkey.
#  
#  * Nonstandard attacks in Turkey are composed of armed assaults and assasination with main use of firearms and explosives.
#  
#  * Explosion, Incediary and firearms are the standard weapons used in PKK attacks.
# 

# In[ ]:


unknown = df_terrorism[df_terrorism['gname'] =='Unknown']
unknown.head()


# In[ ]:


in_ = []
out_ = []

for i in range(1, 11): 
    indexs_out, indexs_in = get_indexs(names[i], df_terrorism_new)
    for index in indexs_in:
        in_.append(index)
    for index in indexs_out:
        out_.append(index)


# In[ ]:


pattern = df_terrorism_new[df_terrorism_new.index.isin(in_)]


# <font size="+3" color="black"><b>3 - Clustering</b></font><br><a id="3"></a>
# 
# * To evaluate the pattern of Attacks intra terrorist groups, we analyse the composition of each cluster to undestand better the composition this I remove from the data samples that are non standard from each group to keep real and more frequent operation attacks.

# In[ ]:


from sklearn.cluster import KMeans
X = pattern.iloc[:,~pattern.columns.isin(['gname'])]
kmeans = KMeans(n_clusters=10, random_state=42).fit(X)
kmeans.labels_


# In[ ]:


import seaborn as sns
pattern['cluster'] = kmeans.labels_
pattern['cluster'].value_counts()
ax = sns.countplot(pattern['cluster'])
ax


# In[ ]:


ax = sns.countplot(pattern['gname'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax


# * Is similar the composition of real groups terrorism and amount of created clusters

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
pattern['x'] =  X_pca[:,0]
pattern['y'] = X_pca[:,1]
pattern['z'] = X_pca[:,2]


# In[ ]:


import plotly.express as px
fig = px.scatter_3d(pattern, x='x', y='y', z='z',
              color='cluster')
fig.show()


# In[ ]:


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=2, cols=5, specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}, {"type": "pie"}, {"type": "pie"}],
                                          [{"type": "pie"}, {"type": "pie"}, {"type": "pie"}, {"type": "pie"}, {"type": "pie"}]])


j=1
k=1
for i in range(1, 11): 
    if ((i-1) %5 ==0) and (i !=1): 
        k = 1
        j+=1
    tmp = pattern[pattern['gname']==names[i]]
    fig.append_trace(go.Pie(values=list(tmp['cluster'].value_counts()), 
                            labels=tmp['cluster'].value_counts().index, title_text='Clustering Distribuition '+str(names[i])), row=j, col=k)
    k+=1
fig.show()


# <font size="+3" color="black"><b>4 - Conclusion</b></font><br><a id="4"></a>
# 
# * The 10 largest terrorist groups in the world have their countries well defined, among them no one operate in same country so only by country is possible to recognize the group. The ways in which attacks are carried out in general are also well defined, the forms, weapons and targets are standardized, thus demonstrating that groups consciously or unconsciously specialize in certain forms of attack. 
# 
# * Contrary to what is generally common sense, most terrorist groups do not focus on public bodies or mititarians in general are private properties using explosions, in actions in general they are the civilians who suffer most
# 
# * General as compositions of forms of attack, taking into account targets, parents and weapons used is very well registered, the groups that most resemble and divide clusters are due to their ways of practicing terrorism and the targets where they perform
# 
# * IRA, FARC, PKK and Taliban are the group with best pattern recognition
# 
# * SL and NPA, are the most similar operation groups, and divides the sames clusters

# In[ ]:




