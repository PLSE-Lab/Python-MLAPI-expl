#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv')


# Let's look at the names of the stadiums:

# In[ ]:


sorted(train['Stadium'].unique())


# Obviously there are some different names for the single stadium (for example: 'CenturyLink' and'CenturyLink Field). The number of "unique" stadiums (with errors) is: 

# In[ ]:


len(train['Stadium'].unique())


# Now let's fix it. Firstly, let's construct a dict. All errors are fixed by hands.

# In[ ]:


map_stad = {'Broncos Stadium at Mile High': 'Broncos Stadium At Mile High', 'CenturyField': 'CenturyLink Field', 'CenturyLink': 'CenturyLink Field', 'Everbank Field': 'EverBank Field', 'FirstEnergy': 'First Energy Stadium', 'FirstEnergy Stadium': 'First Energy Stadium', 'FirstEnergyStadium': 'First Energy Stadium', 'Lambeau field': 'Lambeau Field', 'Los Angeles Memorial Coliesum': 'Los Angeles Memorial Coliseum', 'M & T Bank Stadium': 'M&T Bank Stadium', 'M&T Stadium': 'M&T Bank Stadium', 'Mercedes-Benz Dome': 'Mercedes-Benz Superdome', 'MetLife': 'MetLife Stadium', 'Metlife Stadium': 'MetLife Stadium', 'NRG': 'NRG Stadium', 'Oakland Alameda-County Coliseum': 'Oakland-Alameda County Coliseum', 'Paul Brown Stdium': 'Paul Brown Stadium', 'Twickenham': 'Twickenham Stadium'}

for stad in train['Stadium'].unique():
    if stad in map_stad.keys():
        pass
    else:
        map_stad[stad]=stad


# Application to the dataset:

# In[ ]:


train['Stadium'] = train['Stadium'].map(map_stad)


# Look at the final list of unique stadiums:

# In[ ]:


sorted(train['Stadium'].unique())


# Looks much better. The number of them is:

# In[ ]:


len(sorted(train['Stadium'].unique()))


# So, we can see that this simple procedure cleans errors in the 'Stadium' field and reduces the number of unique stadiums from virtual 55 to the real 37.
