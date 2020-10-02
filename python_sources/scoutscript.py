#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools
from subprocess import check_output

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.preprocessing import LabelEncoder


# ### Scout Script
# 
# Finally a dataset with Kaggle Kernels available. Full disclosure I've tried to work with [GTD](https://www.start.umd.edu/gtd/) but failed miserably due to my own incompetence.
# 
# Let's first load the dataset into pandas. We need to provide the encoding this time as the default of `UTF-8` does not work out well for us and causes the script to raise an error.

# In[ ]:


df = pd.read_csv('../input/attacks_data.csv',
                 encoding='latin1', parse_dates=['Date'],
                 infer_datetime_format=True,
                 index_col=1,
                )
df.info()


# Let's do a quick plot of how many people have been killed over time and how many injured.

# In[ ]:


plt.figure(figsize=(15, 5))

plt.subplot(121)
df.Killed.plot()
plt.title('Killed')

plt.subplot(122)
df.Injured.plot()
plt.title('Injured')


# We can see the counts picking up pace around 2015. Since a country column is available let's see who is getting hit the most and who is relatively safer. We do a simple histogram.

# In[ ]:


df.Country.value_counts().plot(kind='bar', figsize=(17, 7))
plt.title('Number of attacks by countries')


# As it turns out most of the first world countries are safe (exponentially safe one might add) in some sense if number of attacks are to be taken as any measure of safety.
# A better view of this data might be to plot a heatmap of the world and see where exactly these hits are taking place. Perhaps in another Notebook.
# 
# We might want to see which countries have seen maximum increase since previous years. This might help us get some idea of where things are headed.

# In[ ]:


(df['2015'].Country.value_counts() - 
df['2014'].Country.value_counts()).plot(kind='bar', figsize=(17, 7))


# In[ ]:


upto_month = str(df['2016':].index.month.max())
(
df['2016-' + upto_month].Country.value_counts() - 
df['2015-' + upto_month].Country.value_counts()
).plot(kind='bar', figsize=(17, 7))


# As expected, the year is not yet in. But these are simply attack counts. What matters is people. We need to see how many were killed and injured.

# In[ ]:


df.groupby('Country').sum()[['Killed', 'Injured']].plot(kind='bar', figsize=(17, 7), subplots=True)


# Let's see when exactly during a year do attacks occur and how they have been changing over time.

# In[ ]:


plt.figure(figsize=(15, 10))
years = list(set(df.index.year))
years.sort()
for index, year in enumerate(years):
    plt.subplot(4, 4, index+1)
    plt.hist(df[str(year)].index.month)
    plt.title(str(year))


# Most awful things happen in Jan - December. I suppose it's the holidays and so ... This notebook is getting difficult to write.
# I think this would be enough for now. Another notebook another time then.
