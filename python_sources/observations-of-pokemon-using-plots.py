#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/Pokemon.csv")


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()
#We have nulls in 'Type 2', which is possible as a Pokemon can be completely of one type


# In[ ]:


data.Legendary.value_counts()
#we have 65 Legendary Pokemon


# In[ ]:


data.Generation.value_counts()
#We have 6 Generation Pokemon


# In[ ]:


#Function to label in plots
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                '%d' % int(height),
                ha='center', va='bottom')


# In[ ]:


sns.countplot(x='Type 1', data = data)
plt.xticks(rotation='vertical')


# In[ ]:


sns.countplot(x='Type 1',hue='Type 2',data = data)
plt.xticks(rotation='vertical')


# In[ ]:


sns.countplot(x='Type 1', data= data, hue = 'Legendary')
plt.xticks(rotation = 'vertical')


# In[ ]:


#Plot below shows the most effective attack Gropued by Pokemon Type, which is 'Water' type Pokemon
temp_df = data.groupby('Type 1')['Total'].agg('sum').reset_index().sort_values(by='Total',ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]
labels = np.array(temp_df['Type 1'])
width = 0.9
ind = np.arange(len(labels))
fig,ax = plt.subplots()
rects = ax.bar(ind,np.array(temp_df['Total']),color='green')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels,rotation='vertical')
ax.set_ylabel("Count")
autolabel(rects)
plt.show()


# In[ ]:


#Plot below shows the fastest Pokemon Types, which is again'Water' type Pokemon
#This also shows that 'Rock' type Pokemon is second from last, which is obvious
temp_df = data.groupby('Type 1')['Speed'].agg('sum').reset_index().sort_values(by='Speed',ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]
labels = np.array(temp_df['Type 1'])
ind = np.arange(len(labels))
fig,ax = plt.subplots()
rects = ax.bar(ind,np.array(temp_df['Speed']),color='green')
width = 0.9
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels,rotation='vertical')
ax.set_ylabel("Count")
autolabel(rects)
plt.show()


# In[ ]:


#As per the below polt it shows that again Water type Pokemon are best in defense at a overall scale, I wonder How ?
temp_df = data.groupby('Type 1')['Defense'].agg('sum').reset_index().sort_values(by='Defense',ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]
labels = np.array(temp_df['Type 1'])
ind = np.arange(len(labels))
fig,ax = plt.subplots()
rects = ax.bar(ind,np.array(temp_df['Defense']),width=0.9,color='orange')
width = 0.9
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels,rotation='vertical')
ax.set_ylabel("Count")
autolabel(rects)
plt.show()


# In[ ]:


#Plot below shows the slowest Pokemon as 'Flying', which is not true and this shows that we have data problems whe comes to 
#All the Pokemons we actually have
temp_df = data.groupby('Type 1')['Speed'].agg('sum').reset_index().sort_values(by='Speed',ascending=True).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]
labels = np.array(temp_df['Type 1'])
ind = np.arange(len(labels))
fig,ax = plt.subplots()
rects = ax.bar(ind,np.array(temp_df['Speed']),width=0.9,color='green')
width = 0.9
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels,rotation='vertical')
ax.set_ylabel("Count")
autolabel(rects)
plt.show()


# In[ ]:


sns.countplot(x='Type 1', data = data, hue= 'Generation')
plt.xticks(rotation = 'vertical')
plt.tight_layout()


# In[ ]:


#We have a temp data frame where we have only Legendary Pokemon
legend = data[data['Legendary']==True]


# In[ ]:


legend.shape
#We have Legendary Pokemon


# In[ ]:


#Plot below shows the most effective attack Gropued by Pokemon Type, which is 'Water' type Pokemon
temp_df = legend.groupby('Type 1')['Total'].agg('sum').reset_index().sort_values(by='Total',ascending=False).reset_index(drop=True)
labels = np.array(temp_df['Type 1'])
ind = np.arange(len(labels))
fig,ax = plt.subplots()
rects = ax.bar(ind,np.array(temp_df['Total']),color='green')
width = 0.9
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels,rotation='vertical')
ax.set_ylabel("Count")
autolabel(rects)
plt.show()


# In[ ]:


sns.countplot(x='Type 1', data = legend, hue= 'Type 2')
plt.legend(loc='upper right')
plt.xticks(rotation = 'vertical')
plt.tight_layout()


# In[ ]:


sns.barplot(x='Type 1', y='Total',data = legend)
plt.xticks(rotation='vertical')


# In[ ]:


legend.loc[legend['Total'].idxmax()]
#This is the Pokemon with most TotalPower in the Legendary Type


# In[ ]:


legend.loc[legend['Defense'].idxmax()]
#This is the Pokemon with most Defense Power in the Legendary Type


# In[ ]:


legend.loc[legend['Speed'].idxmax()]
#This is the fastest Pokemon in the Legendary Type


# In[ ]:


sns.countplot(x='Type 1', data= data)
plt.xticks(rotation = 'vertical')


# In[ ]:


data.loc[data['Defense'].idxmax()]
#This is the most defensive Pokemon, and not a legendary too


# In[ ]:


#Showing the Perncentage of Pokemon in Type1
data['Type 1'].value_counts().plot(kind='pie',autopct='%1.1f%%')


# In[ ]:


#Showing the Count of Pokemon in Type 2 
total = sum(data['Type 2'].value_counts())
data['Type 2'].value_counts().plot(kind='pie',autopct=(lambda p: '{:.0f}'.format(p * total / 100)))


# Plese upvote if you have found this useful, give me a comment if there are any suggestions or corrections
# Happy to Learn by sharing !!!
