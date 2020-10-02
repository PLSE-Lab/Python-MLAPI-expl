#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/database.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


states_death_penalty = ["Alabama","Arizona","Arkansas","California","Colorado","Florida",
                        "Georgia","Idaho","Indiana","Kansas","Kentucky","Louisiana","Mississippi",
                        "Missouri","Montana","Nebraska","Nevada","New Hampshire","North Carolina",
                        "Ohio","Oklahoma","Oregon","Pennsylvania","South Carolina","South Dakota",
                        "Tennessee","Texas","Utah","Virginia","Washington","Wyoming"]

states_no_death_penalty = [("Alaska",1957),("Connecticut",2012),("Delaware",2016),("Hawaii",1957),
                           ("Illinois",2011),("Iowa",1965),("Maine",1887),("Maryland",2013),
                           ("Massachusetts",1984),("Michigan",1846),("Minnesota",1911),
                           ("New Jersey",2007),("New Mexico",2009),("New York",2007),
                           ("North Dakota",1973),("Rhode Island",1984),("Vermont",1964),
                           ("West Virginia",1965),("Wisconsin",1853)]


# In[ ]:


df['Crime Type'].unique()


# In[ ]:


# Excluding manslaughter by negligence
incidents_by_state = df[df['Crime Type'] == 'Murder or Manslaughter'].groupby(['State'])['State'].count()

plt.rcParams['figure.figsize'] = (30,15)
plt.rcParams.update({'font.size': 22})

# Set the bar labels
bar_labels = incidents_by_state.keys()

# Create the x position of the bars
x_pos = list(range(len(incidents_by_state)))

# Create the plot bars
# In x position
plt.bar(x_pos,
        # using the data from the mean_values
        incidents_by_state, 
        # aligned in the center
        align='center',
        # with color
        color='#FFC222')

# set axes labels and title
plt.ylabel('Count')
plt.xticks(x_pos, bar_labels, rotation='vertical')
plt.title('Incidents by states')

plt.show()


# As you can see in the previous figure, California is by far the state with more incidents. Let's see if death penalty is related with the amount of incidents.

# In[ ]:


incidents_by_state_sorted = incidents_by_state.sort_values(ascending=False)
sample = list(incidents_by_state_sorted[0:10].keys())

amount_death_penalty = 0
amount_no_death_penalty = 0

for state in sample:
    if state in states_death_penalty:
        amount_death_penalty += df[df['State'] == state].shape[0]
    else:
        for index, row in df[df['State'] == state].iterrows():
            if row['Year'] < [item for item in states_no_death_penalty if item[0] == state][0][1]:
                amount_death_penalty += 1
            else:
                amount_no_death_penalty += 1


print("Murders or manslaughters in states with death penalty: {}".format(amount_death_penalty))
print("Murders or manslaughters in states without death penalty: {}".format(amount_no_death_penalty))


# We can conclude that criminals are not influenced by the death penalty. We can also conclude that the death penalty is not an effective deterrent.
