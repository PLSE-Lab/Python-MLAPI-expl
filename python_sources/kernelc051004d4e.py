#!/usr/bin/env python
# coding: utf-8

# # Analysis of Police Shoot dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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


df = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')


# # I am new but Love PieCharts

# # Here i am taking threat level attacked

# In[ ]:


df[df['threat_level'] == 'undetermined']['id'].count()
attacked = df[df['threat_level'] == 'attack']


# # This function is to adjust Pct according to the values

# In[ ]:


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


# # Storing the count of attacked group according to state

# In[ ]:


statecount = attacked.groupby('state')['id'].count()


# ## This plotting the pie chart statewise attacked 

# In[ ]:


statecount.plot.pie(y ='values',figsize =(15,15),legend = True, pctdistance=0.8,  autopct=make_autopct(statecount.values), labeldistance=1.2)


# ### Look at the head to help me go through rest of the code

# In[ ]:


df.head()


# # now looking at the ones who fleed and thier manner of death

# In[ ]:


df.groupby(['manner_of_death','flee'])['flee'].describe()


# # We find that the most were not trying to flee but where shot dead

# In[ ]:


vals = make_autopct(df.groupby(['signs_of_mental_illness','threat_level'])['id'].count())
df.groupby(['signs_of_mental_illness','threat_level'])['id'].count().plot.pie(legend = True, figsize = (10,10), autopct=vals)


# ## Another thought crossed my mind whether bieng mentally ill effects the shooting
# ### But yea it does not

# In[ ]:


vals = make_autopct(df.groupby(['body_camera','threat_level'])['id'].count())
df.groupby(['body_camera','threat_level'])['threat_level'].count().plot.pie(legend = True, autopct = vals, figsize = (10, 10))


# # The ones who attacked where not having a body camera on to witness the attack
# ## Possibly a case of no fear

# In[ ]:


vals = make_autopct(df.groupby(['body_camera','manner_of_death'])['id'].count())
df.groupby(['body_camera','manner_of_death'])['threat_level'].count().plot.pie(legend = True, autopct = vals, figsize = (10, 10))


# # Clearly see that more than 84% of police with no camera shot the victim

# In[ ]:


df['Age-group'] = pd.cut(df['age'],bins = [0,20,40,60,80,100])


# # Simple binning to see effect of age group on different happenings

# In[ ]:


df.groupby(['Age-group','manner_of_death'])['manner_of_death'].count().plot.pie(legend = True, figsize = (10,10), 
                                                            autopct = make_autopct(df.groupby(['Age-group','manner_of_death'])['manner_of_death'].count()))


# ## Looking at the above pie chart the most shot age group was below 20
# ### More clearly they were shot dead with 55% of all the ages
# ## Notice how the second ones are at 40 to 60

# In[ ]:


vals = df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[0*len(statecount.index):0*len(statecount.index) + 10]
df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[0*len(statecount.index):0*len(statecount.index)+10].plot.pie(figsize = (10,10),
                                                                                                                                        autopct = make_autopct(vals))


# ## California shootings with below 20 are still the highest shooting victims
# ## followed by Colorado

# In[ ]:


vals = df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[2*len(statecount.index):2*len(statecount.index) + 10]
df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[2*len(statecount.index):2*len(statecount.index)+10].plot.pie(figsize = (10,10),
                                                                                                                                        autopct = make_autopct(vals))


# # In age group 20 to 40 still california is at the top, corncerned with 433 people shot among that age group

# In[ ]:


vals = df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[4*len(statecount.index):4*len(statecount.index) + 10]
df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[4*len(statecount.index):4*len(statecount.index)+10].plot.pie(figsize = (10,10),
                                                                                                                                        autopct = make_autopct(vals))


# # 40 to 60 age group still California is leading followed by Florida

# In[ ]:


vals = df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[3*2*len(statecount.index):3*2*len(statecount.index) + 10]
df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[3*2*len(statecount.index):3*2*len(statecount.index)+10].plot.pie(figsize = (10,10),
                                                                                                                                        autopct = make_autopct(vals))


# # 60 to 80 years of age are shot down by the California police

# In[ ]:


vals = df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[8*len(statecount.index):8*len(statecount.index) + 10]
df.groupby(['Age-group','manner_of_death','state'])['id'].count().iloc[8*len(statecount.index):8*len(statecount.index)+10].plot.pie(figsize = (10,10),
                                                                                                                                        autopct = make_autopct(vals))


# # Looking at the 80 to 100 age group we see Arizona has shot 3 or around 42%

# # I would like to give three insights here:
# ## First the ones that were shot were not trying to flee, This indicates the pyschological effect of no crime commited
# ## Second there is strong corelation with body camera and shooting because we see that police man who did have a body camera most likely attacked and shot
# ## Third the most 20 to 40 age group of people were shot evident with 55%  of whole population of data provided

# # Thanks this was my first kaggle notebook, Please comment down any mistakes in insights
# ## Or enligthen me with the knowledge

# In[ ]:




