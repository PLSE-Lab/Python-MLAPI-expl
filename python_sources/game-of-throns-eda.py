#!/usr/bin/env python
# coding: utf-8

# # Context
# Game of Thrones is a hit fantasy tv show based on the equally famous book series "A Song of Fire and Ice" by George RR Martin. The show is well known for its vastly complicated political landscape, large number of characters, and its frequent character deaths.

# # Content
# This dataset combines three sources of data, all of which are based on information from the book series.
# 
# Firstly, there is battles.csv which contains Chris Albon's "The
# War of the Five Kings" Dataset. Its a
# great collection of all of the battles in the series.
# 
# Secondly we have character-deaths.csv from Erin Pierce and Ben
# Kahle. This dataset was created as a part of their Bayesian Survival
# Analysis.
# 
# Finally we have a more comprehensive character dataset with
# character-predictions.csv. It
# includes their predictions on which character will die

# # Load Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.preprocessing import scale
from scipy import stats
sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")
import warnings
warnings.filterwarnings('ignore')


# # Load Data

# In[ ]:


bdf=pd.read_csv('../input/game-of-thrones/battles.csv')
cddf=pd.read_csv('../input/game-of-thrones/character-deaths.csv')


# # Read the Data And Sneeking Variables

# In[ ]:


print('The no of columns and rows in battle csv :',bdf.shape)
print('The no of columns and rows in character csv :',cddf.shape)


# In[ ]:


print('Name of the columns in battle dataset: ', bdf.columns)
print('Name of the columns in charecter data set :', cddf.columns)


# # now doing EDA on Battels

# In[ ]:


print('sneeking the 1st five rows of battels data :')
bdf.head()


# In[ ]:


print('sneeking the last five rows of battels data :')
bdf.tail()


# In[ ]:


print('finding the datatype and non- null count of each row: ')
bdf.info()


# In[ ]:


#here we can see that total no of numerical columns are 9 and object columns 16


# In[ ]:


print('lets observed the no of battels along with name :')
pd.melt(frame=bdf, id_vars="year",value_vars="name")


# #Here we can observed that there are around 38 battels fought and time line is varied from 298 to 300.

# # visualion with step by step analysis

# Correlations
# Let's take a look at how everything is correlated within their datasets, and how they are correlated to each other. A positive number indicates that as x increases, so does y. A negative number indicates that as x increases, y decreases.

# In[ ]:


bdf.corr().style.background_gradient(cmap='Reds')


# In[ ]:


fig,ax=plt.subplots(figsize=(20,8))
sns.heatmap(bdf.corr(),annot=True);


# count of battels per year

# In[ ]:


bpy=bdf.groupby('year',as_index=False).sum()
plt.barh(bpy['year'],bpy['battle_number'])
plt.xticks(rotation=90);


# In[ ]:


#attacker size count..
plt.figure(figsize=(20,8))
sns.countplot(bdf['attacker_size']);


# In[ ]:


plt.figure(figsize=(20,8))
sns.countplot(bdf['defender_size']);


# In[ ]:


bdf['battle_type'].value_counts().plot(kind = 'barh');


# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
sns.countplot(bdf['attacker_king'])
plt.xticks(rotation=70)
plt.subplot(1,2,2)
sns.countplot(bdf['battle_type']);
plt.xticks(rotation=70);


# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
sns.countplot(bdf['defender_king'])
plt.xticks(rotation=70)
plt.subplot(1,2,2)
sns.countplot(bdf['battle_type']);
plt.xticks(rotation=70);


# In[ ]:


pd.crosstab(bdf['attacker_king'],bdf['attacker_outcome']).plot(kind='bar',figsize=(15,5));
plt.xticks(rotation='horizontal');


# In[ ]:


#find attacjer king and battles type
plt.figure(figsize=(20,8))
sns.countplot(bdf['attacker_king'],hue=bdf['battle_type']);


# In[ ]:


plt.figure(figsize=(20,8))
sns.countplot(bdf['attacker_king'],hue=bdf['attacker_outcome']);


# In[ ]:


plt.figure(figsize=(20,8))
sns.countplot(bdf['attacker_commander'],hue=bdf['attacker_outcome']);
plt.xticks(rotation=90);


# In[ ]:


plt.figure(figsize=(20,8))
sns.countplot(bdf['attacker_king'],hue=bdf['defender_king']);
plt.xticks(rotation=90);


# In[ ]:


bdf.loc[:, "totaldefender"] = (4 - bdf[["defender_1", "defender_2", "defender_3", "defender_4"]].isnull().sum(axis = 1))
bdf.loc[:, "totalattacker"] = (4 - bdf[["attacker_1", "attacker_2", "attacker_3", "attacker_4"]].isnull().sum(axis = 1))
bdf.loc[:, "totalcommon"] = [len(x) if type(x) == list else np.nan for x in bdf.attacker_commander.str.split(",")]


# In[ ]:


p = sns.boxenplot("totalcommon", "attacker_king", data = bdf, saturation = .6,palette = ["lightgray", sns.color_palette()[1], "grey", "darkblue"])
_ = p.set(xlabel = "No. of Attacker Commanders", ylabel = "Attacker King", xticks = range(8))


# In[ ]:


p = sns.boxenplot("totaldefender", "defender_king", data = bdf, saturation = .6,palette = ["lightgray", sns.color_palette()[1], "grey", "darkblue"])
_ = p.set(xlabel = "No. of defender commander", ylabel = "defender king", xticks = range(8))


# In[ ]:


bdf['attacker_size'].mean()


# In[ ]:


bdf['defender_size'].mean()


# In[ ]:


nbdf = bdf[['defender_size','attacker_size','attacker_outcome']].dropna()


# In[ ]:


nbdf.reset_index(inplace=True)


# In[ ]:


nbdf = nbdf.iloc[:,1:]


# In[ ]:


sns.pairplot(nbdf, hue='attacker_outcome');


# In[ ]:


bdf.groupby('battle_type')['attacker_outcome'].value_counts().plot(kind = 'bar');


# In[ ]:


bdf['region'].value_counts().plot(kind = 'pie');


# In[ ]:


sns.countplot(x=bdf['location'])
plt.xticks(rotation=90);


# In[ ]:


plt.figure(figsize=(20,8))
sns.countplot(bdf['attacker_king'],hue=bdf['region']);


# In[ ]:


plt.figure(figsize=(20,8))
sns.countplot(bdf['defender_king'],hue=bdf['region']);


# In[ ]:


data = bdf.groupby("region").sum()[["major_death", "major_capture"]]
p = pd.concat([data, bdf.region.value_counts().to_frame()], axis = 1).sort_values("major_death", ascending = False).copy(deep = True).plot.bar(color = [sns.color_palette()[1], 
"grey", "darkblue"], rot = 0)
_ = p.set(xlabel = "Region", ylabel = "No. of Events"), p.legend(["No. of Battles", "Major Deaths", "Major Captures"], fontsize = 12.)
plt.xticks(rotation=90);


# charecter deth analysis.

# In[ ]:


cddf['Allegiances'] = cddf['Allegiances'].apply(lambda x : 'House Martell' if(x == 'Martell') else 'House Stark' if(x=='Stark') else 'House Targaryen' if(x=='Targaryen') else 'House Tully' if(x=='Tully') else 'House Tyrell' if(x=='Tyrell') else x)


# In[ ]:


cddf['Gender'].value_counts().plot(kind = 'pie');


# In[ ]:


cddf['Allegiances'].value_counts().plot(kind = 'bar', stacked='True');


# In[ ]:


cddf[cddf['Death Year'].notnull()]['Allegiances'].value_counts().plot(kind = 'barh',color='y');


# In[ ]:





# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 12))
sns.violinplot(x="year", y="battle_number", data=bdf,color = 'pink',ax=axes[0][0]).set_title('battle number')
sns.swarmplot(x="year", y="battle_number", data=bdf,ax = axes[0][0])

sns.violinplot(x="year", y="major_death", data=bdf,color = 'pink',ax=axes[0][1]).set_title('major_death')
sns.swarmplot(x="year", y="major_death", data=bdf,ax = axes[0][1])

sns.violinplot(x="year", y="major_capture", data=bdf,color = 'pink',ax=axes[1][0]).set_title('major_capture')
sns.swarmplot(x="year", y="major_capture", data=bdf,ax = axes[1][0])

sns.violinplot(x="year", y="attacker_size", data=bdf,color = 'pink',ax=axes[1][1]).set_title('attacker_size')
sns.swarmplot(x="year", y="attacker_size", data=bdf,ax = axes[1][1])

sns.violinplot(x="year", y="defender_size", data=bdf,color = 'pink',ax=axes[2][0]).set_title('defender_size')
sns.swarmplot(x="year", y="defender_size", data=bdf,ax = axes[2][0])

sns.violinplot(x="totaldefender", y="totalattacker", data=bdf,color = 'gray',ax=axes[2][1]).set_title('totalattacker vs totaldefender')
sns.swarmplot(x="totaldefender", y="totalattacker", data=bdf,ax = axes[2][1])

plt.grid()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()


# # conclusion
# if this visualization is helpfull kindly upvote!!! :) 

# In[ ]:




