#!/usr/bin/env python
# coding: utf-8

# #### Import the essential libraries

# In[ ]:


import pandas as pd
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt


# #### Read the parties and candidates list

# In[ ]:


data=pd.read_csv('../input/NA Candidate List.csv')
parties=pd.read_csv('../input/parties.csv')


# In[ ]:


candidates_count=data.groupby('Constituency_title').count()


# #### Number of candidates in a constituency on average

# In[ ]:


candidates_count.Party.mean()


# #### Find out the constituencies with the maximum candidates

# In[ ]:


data.groupby(['Constituency_title'])['Seat'].count().nlargest(10)


# #### Find out the constituency with the lowest number of candidates

# In[ ]:


data.groupby(['Constituency_title'])['Seat'].count().nsmallest(10)


# In[ ]:


def resolveParty(x):
    max=0
    match=''
    for i in parties['Name of Political Party']:
        r=fuzz.ratio(x,i)
        if r>max:
            max=r
            match=i
    return match


# #### Find out the best match for a party name, resolving spelling mistakes

# In[ ]:


extracted_parties=data.Party.apply(lambda x: resolveParty(x))


# In[ ]:


data.Party=extracted_parties


# #### Find number of candidates per party

# In[ ]:


seats=data.groupby(['Party'])['Seat'].count()


# In[ ]:


party_position=seats.nlargest(11)


# In[ ]:


party_position


# #### Plot the parties with most number of candidates

# In[ ]:


party_position.iloc[1:].plot(kind='bar')
plt.show()


# #### Party with least number of seats

# In[ ]:


seats.nsmallest(10)


# In[ ]:




