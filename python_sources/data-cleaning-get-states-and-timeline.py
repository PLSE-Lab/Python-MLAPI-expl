#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Remenber use parameters "encoding" and "thousands" while importing the data, otherwise there will be an error with encoding or you will get 1.000 (number 'one' with decimal point) instead of 1000 if you want to read the number "one thousand"

# In[ ]:


raw_data = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding = 'latin1', thousands = '.')


# Then convert spanish month into numerical month.

# In[ ]:


month_list = raw_data.groupby('month').size()
month_inspan = month_list.index.values.tolist()
month_innum = [4,8,12,2,1,7,6,5,3,11,10,9]
replace_dic = {}
for i in range(len(month_inspan)):
    replace_dic[month_inspan[i]] = month_innum[i]
fire_data = raw_data.replace({'month': replace_dic})
print('Converted Data looks like this:')
fire_data.head()


# In[ ]:


# sort the data, get the raw time line
fire_data = fire_data.sort_values(by = ['state','year','month'])


# It seems that we have dupicate rows with different 'number' item. It is because some states in Brazil share the same abbrevation. This problem cannot be solved in this dataset. What we can do is to remove all states with the same abbrevation, just keep the ones that could be analyze.
# 
# Or you can visit [This Discussion](https://www.kaggle.com/gustavomodelli/forest-fires-in-brazil/discussion/117901) for dataset with whole state name. I already put the seperated dataset in my github and attached that link in my comment.
# 
# Let's see what to be removed in the dataset provided by Kaggle:

# In[ ]:


state_rownum = fire_data.groupby('state').size()
print(state_rownum)


# We can simply remove 'Mato Grosso', 'Paraiba', 'Rio' because these have two or three times items more than the other abbrevitions toward states.
# 
# But what's wrong with 'Alagoas'?

# ## Remove abnormal data in state 'Alagoas'

# Split Alagoas from the raw dataset so you don't have to load the whole dataset for analyze.

# In[ ]:


Alagoas_data = fire_data.query("state == 'Alagoas'")
Alagoas_data.sort_values(by = ['year','month'])


# There are two ways to find out the abnormal data, one is to use set in python to find the duplicated variables automatically, another one is quite tricky.

# ### Solution 1. Using python set

# In[ ]:


temp = set()
for i in range(len(Alagoas_data)):
    # this is just for checking, so no matter how ugly the elements in temp set looks
    if (Alagoas_data.iloc[i]['year'] * 100 + Alagoas_data.iloc[i]['month']) not in temp:
        temp.add(Alagoas_data.iloc[i]['year'] * 100 + Alagoas_data.iloc[i]['month'])
    else:
        print('Duplicate Data')
        print(Alagoas_data.iloc[i])


# In[ ]:


Alagoas_data.loc[(Alagoas_data.year == 2017)&(Alagoas_data.month ==1)]


# 
# We can safely remove one of these two rows because they are exactly the same.

# ### Solution 2. Mathmetical T[](http://)rick

# First inspect both fire dataset and Alagoas data by using df.describe()

# In[ ]:


print('Description of the fire dataset')
print(fire_data.describe())
print('\n')
print('Description of Alagoas data')
print(Alagoas_data.describe())


# Than locating the exact year by simple calculation

# In[ ]:


2007.5 * 240 - 2007.461729 * 239


# Got it, the abnormal data will be either 2016 or 2017, let's inspect it.

# In[ ]:


data_16 = Alagoas_data.query("year == 2016")
print(data_16)


# In[ ]:


data_17 = Alagoas_data.query("year == 2017")
print(data_17)


# It is easy to find there are two month '1's in 2017, it is the duplicate month number we need to remove.

# Than we can drop this duplicate row in the fire dataset, as well as ambiguous state abbrevations.

# In[ ]:


fire_data = fire_data.drop(fire_data[(fire_data['state'] == 'Mato Grosso') | (fire_data['state'] == 'Paraiba') | (fire_data['state'] == 'Rio') ].index)
if fire_data.iloc[258].empty == False:
    fire_data = fire_data.drop(258)
print(fire_data.groupby('state').size())


# In[ ]:


fire_formatted = fire_data
# need to do reset index otherwise rows removed will be an issue
fire_formatted = fire_formatted.reset_index(drop = True)
for i in range(len(fire_formatted)):
    fire_formatted.at[i,'happened_date'] = str(fire_formatted.at[i, 'year']) + '-' + str(fire_formatted.at[i, 'month'])
fire_formatted


# Drop rows that is not useful:

# In[ ]:


dt = pd.DataFrame(fire_formatted, columns = ['state', 'number','happened_date'])
# group data by different states
groupby_state = dt.groupby('state')


# Now we can create a dataset with states and time line

# In[ ]:


fire = []
time = []
state_name = []

for i in range(len(dt)):
    if dt.iloc[i,2] not in time:
        time.append(dt.iloc[i,2])
    if dt.iloc[i,0] not in state_name:
        state_name.append(dt.iloc[i,0])

for state, item in groupby_state:
        fire.append(item.iloc[:,1].values.tolist())


fire_dt = pd.DataFrame(fire, columns = time)
fire_dt.insert(loc = 0, column = 'state', value = state_name)


# In[ ]:


print('Head:')
print(fire_dt.head())
print('\n')
print('Describe')
print(fire_dt.describe())
print('\n')
print('info')
print(fire_dt.info())


# In[ ]:


fire_dt.to_csv('state_timeline.csv')


# Than you can do other things like visualization by using this new dataset.

# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(50,25))
for i in range(1, fire_dt.shape[0]):
    plt.plot(fire_dt.columns[1:].tolist(), fire_dt.iloc[i,1:].values.tolist(), label = fire_dt.iloc[i,0], color = np.random.rand(3,))

plt.xlabel('Date')
plt.ylabel('Number')
plt.xticks(rotation=90)
plt.legend()
plt.show()


# In[ ]:




