#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 30)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


candidates = pd.read_csv('/kaggle/input/campaign-contributions-19902016/candidates.csv')
candidates.head()


# In[ ]:


candidates.info()


# Some preprocessing

# In[ ]:


candidates['election_type'] = candidates['dist_id_run_for'].apply(lambda x : 'PRES' if str(x).upper() == 'PRES' else 'SENATE' if (str(x)[-2:] == 'S1' or str(x)[-2:] == 'S2') else 'DISTRICT')
candidates['election_type'].value_counts()


# In[ ]:


candidates['state'] = candidates[candidates['election_type'] != 'PRES']['dist_id_run_for'].apply(lambda x: str(x)[:2])
candidates['state'].value_counts()


# **Let's take a look at the raised amounts over the years**

# In[ ]:


by_cycle = candidates.groupby('cycle').agg({'raised_from_pacs': 'sum', 'raised_from_individuals': 'sum'}).sort_values('cycle')
by_cycle.plot(figsize=(20, 10))


# So, individual & PAC contributions have increased in the early 2000s, but decreased more recently. The change is more striking in individual contributions. Also, individual contributions have always been greater than PAC contrbutions.
# 
# **Let's see if this holds true for different type of elections**

# In[ ]:


by_election_type = candidates.groupby(['cycle', 'election_type']).agg({'raised_from_pacs': 'sum', 'raised_from_individuals': 'sum'}).sort_values('cycle').reset_index()
by_election_type.head()


# In[ ]:


by_election_type_pivoted = by_election_type.pivot(index='cycle', columns='election_type')
by_election_type_pivoted.head()


# In[ ]:


by_election_type_pivoted.loc[:, ['cycle', 'raised_from_pacs']].plot(figsize=(20, 10))


# In[ ]:


by_election_type_pivoted.loc[:, ['cycle', 'raised_from_individuals']].plot(figsize=(20, 10))


# In both the graphs, the Presidential elections have the most variability in terms of contributions.
# Also, PACS raise more in District-level elections than State Senate elections than Presidential elections, whereas on average individual contributions apears similar in all 3 elections.
# 
# **Let's see the same by party rather than election type**

# In[ ]:


candidates['party'].value_counts()


# Due to the disparity in counts between the parties, we'll have to normalize the contributions by the number of candidates

# In[ ]:


by_party = candidates.groupby(['cycle', 'party']).agg({'raised_from_pacs': np.mean, 'raised_from_individuals': np.mean}).sort_values('cycle').reset_index()
by_party.head()


# In[ ]:


by_party_pivoted = by_party.pivot(index='cycle', columns='party')
by_party_pivoted.head()


# In[ ]:


by_party_pivoted.loc[:, ['cycle', 'raised_from_pacs']].plot(figsize=(20, 10))


# In[ ]:


by_party_pivoted.loc[:, ['cycle', 'raised_from_individuals']].plot(figsize=(20, 10))


# So, the 2 major parties get the most contributions, although independent candidates also see spikes sometimes.
# 
# We see the contributions especially from PACs increase over time, although it seems to have decreased in the more recent years.

# **Let's see how the contributions vary every year by state using a map**

# In[ ]:


cycle_year_df = candidates.groupby(['cycle', 'state']).agg({'raised_from_pacs': np.sum, 'raised_from_individuals': np.sum, 'raised_total': np.sum}).reset_index()

cycle_year_df['text'] = "Total PAC contribution:" + cycle_year_df['raised_from_pacs'].astype('str') + "<br>" +    "Total Individual Contributions: " + cycle_year_df['raised_from_individuals'].astype('str')
cycle_year_df.head()


# In[ ]:


import plotly.express as px
fig = px.choropleth(cycle_year_df, locations="state", color="raised_total", hover_name="state", 
                    hover_data=['raised_from_pacs', 'raised_from_individuals'], animation_frame="cycle", 
                    locationmode='USA-states', scope='usa', title='Total Contributions raised by state over the years')
fig.show()


# In[ ]:


import plotly.express as px
fig = px.choropleth(cycle_year_df, locations="state", color="raised_from_individuals", hover_name="state", 
                    hover_data=['raised_from_pacs', 'raised_total'], animation_frame="cycle", 
                    locationmode='USA-states', scope='usa', title='Individual contributions raised by state over the years')
fig.show()


# In[ ]:


import plotly.express as px
fig = px.choropleth(cycle_year_df, locations="state", color="raised_from_pacs", hover_name="state", 
                    hover_data=['raised_from_individuals', 'raised_total'], animation_frame="cycle", 
                    locationmode='USA-states', scope='usa', title='PAC contributions raised by state over the years')
fig.show()

