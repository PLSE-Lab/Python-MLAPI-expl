#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[ ]:


plt.rcParams['figure.figsize']=(13.5,8)
plt.rcParams['axes.titlesize']=15


# In[ ]:


Extra=pd.read_csv('../input/extrastate.csv')


# ## Extra-State Conflicts:

# #### Cleaning the data:

# In[ ]:


Extra.head()


# In[ ]:


Extra.info()


# [](http://)Let's see null values ( -8 or -9)

# In[ ]:


Extra.replace(-8,0,inplace=True)
Extra.replace(-9,0,inplace=True)


# In[ ]:


sns.heatmap(Extra==0,yticklabels=False)


# let's drop these columns (for now!), seems like we won't be able to do much with them .

# In[ ]:


Extra.drop(['side2_code','start_year2','end_year2','start_month2','end_month2','end_day2','start_day2'],axis=1,inplace=True)


# Let's add some New Features :

# In[ ]:


Extra['Century']=Extra['start_year1'].apply(lambda x:round(x,-2))


# In[ ]:


Extra['decade']=Extra['start_year1'].apply(lambda x:round(x,-1))


# In[ ]:


sns.heatmap(Extra.corr(),cmap='YlOrRd')


# ## fatalities :

# #### by State:

# In[ ]:


sns.boxplot(data=Extra[['side1_name','state_fatalities']].sort_values('state_fatalities'),x='side1_name',y='state_fatalities')
plt.xticks(rotation=90)
plt.title(' state_fatalities by State involved ')


# In[ ]:


State_fat=Extra.groupby('side1_name').sum()['state_fatalities']
State_fat.sort_values().plot.bar()
for x in range(0,43):
    plt.text(x-0.2,y=State_fat.sort_values().values[x]+17000
             ,s=State_fat.sort_values().values[x],rotation=90,fontsize=10,
            color='k')

plt.title('Sum of state_fatalities by State involved ')
sns.set_style('whitegrid')


# In[ ]:


Extra.groupby('side1_name').sum()['nonstate_fatalities'].sort_values().plot.bar()
plt.title('Sum of Nonstate_fatalities by State involved ')


# In[ ]:


Extra['duration']=abs(Extra['end_year1']-Extra['start_year1'])

Extra['death_per_year']=(Extra['state_fatalities']+Extra['nonstate_fatalities'])/Extra[Extra['duration']!=0]['duration']


# In[ ]:


Extra.groupby('side1_name').mean()['death_per_year'].dropna().sort_values().plot.bar()
plt.title('Mean  Death_per year for every state')


# In[ ]:


Extra[['state_fatalities','nonstate_fatalities']].sum().plot.bar()
plt.title('Count of Fatalities:\nState vs Non-State')


# In[ ]:


Extra.groupby('war_name').sum()['nonstate_fatalities'].nlargest(20).sort_values().plot.bar()
plt.title('Bloodiest wars: \n non-state fatalities')


# In[ ]:


Extra.groupby('war_name').sum()['state_fatalities'].nlargest(20).sort_values().plot.bar()
plt.title('Bloodiest wars: \n state fatalities')


# #### by Duration/Date:

# In[ ]:


Extra.groupby('decade').sum()[['nonstate_fatalities','state_fatalities']].plot.bar()
plt.title('Deadliest decades :\nAll_fatalities')


# In[ ]:


sns.heatmap(Extra.groupby(['side1_name','decade']).sum()['state_fatalities'].unstack().fillna(0),cmap='viridis')
plt.title('heatmap of State_Fatalities by Decade & by Country')


# In[ ]:


sns.heatmap(Extra.groupby(['side1_name','decade']).sum()['nonstate_fatalities'].unstack().fillna(0),cmap='viridis')
plt.title('heatmap of NonState_Fatalities by Decabe & by Country')


# In[ ]:


Extra.head()


# Let's Filter the wars that have not ended yet

# In[ ]:


Extra=Extra[Extra['duration']<50]


# #### Longest wars:

# In[ ]:


sns.distplot(Extra[Extra['duration']<50]['duration'] ,bins=15,kde=False)
plt.title('Distibution plot of war durations')


# In[ ]:


plt.figure(figsize=(15,30))
sns.barplot(data=Extra.sort_values('duration',ascending=False) ,y='war_name',x='duration')
plt.title('The Longest Wars ')


# In[ ]:


Extra.head()


# In[ ]:


plt.figure(figsize=(9,16))
sns.boxplot(data=Extra.sort_values('duration'),y='side1_name',x='duration')
plt.title('Longest wars by State')


# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,10))
plt.figure(figsize=(10,10))
ax1.scatter(data=Extra,x='decade',y='state_fatalities',s=Extra['nonstate_fatalities']/100,label='NonState_Fatalities')
blue_patch = mpatches.Patch(color='blue', label='NonState_fatalities')
ax1.legend(handles=[blue_patch],fontsize=12)
ax1.set_ylabel('State_fatalities')
ax1.set_title('State & nonstate fatalities by decade')
("")

plt.figure(figsize=(10,10))
ax2.scatter(data=Extra,x='duration',y='state_fatalities',s=Extra['nonstate_fatalities']/100)
ax2.set_title('State & nonstate fatalities by duration')
ax2.set_ylabel('State_Fatalities',size=15)
ax2.legend(handles=[blue_patch],fontsize=12)

("")


# In[ ]:


sns.countplot(data=Extra,y='decade')
plt.title('Count of Wars by Decade')


# In[ ]:


sns.kdeplot(data=Extra['decade'],shade=True,)
plt.title(' Distribution of non_state fatalities by decade')


# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2)
sns.barplot(data=Extra.sort_values('state_fatalities'),x='decade',y='state_fatalities',estimator=np.sum,ax=ax1,ci=None)
ax1.set_title(' Sum of State fatalities by decade')


sns.barplot(data=Extra.sort_values('nonstate_fatalities'),x='decade',y='nonstate_fatalities',estimator=np.sum,ax=ax2,ci=None)
ax2.set_title(' Sum of non_state fatalities by decade')


# ####  Centuries:

# In[ ]:


sns.countplot(data=Extra,x='Century')
plt.title('Number of Wars by Century')


# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2)
sns.barplot(data=Extra,x='Century',y='state_fatalities',estimator=np.sum,ax=ax1,ci=None)
ax1.set_title(' Sum of State fatalities by Century')


sns.barplot(data=Extra,x='Century',y='nonstate_fatalities',estimator=np.sum,ax=ax2,ci=None)
ax2.set_title(' Sum of non_state fatalities by Century')


# 
# #### let's look at recurring wars:

# In[ ]:


Extra=pd.read_csv('../input/extrastate.csv')


# In[ ]:


Recursive=Extra.replace([-8,-7],np.nan).dropna(subset=['start_year2'])


# Only 3 Wars started a second time

# In[ ]:


fig,axes=plt.subplots(1,2,sharey=True)

Recursive[['war_name','state_fatalities']] .plot.bar(x='war_name',ax=axes[0])
axes[0] .set_title('Number of State Fatalities for wars that reoccured twice')

Recursive[['war_name','nonstate_fatalities']] .plot.bar(x='war_name',ax=axes[1])
plt.title('Number os NonState fatalities for wars that reoccured twice')


# ### Please upvote if you like it !! :D
