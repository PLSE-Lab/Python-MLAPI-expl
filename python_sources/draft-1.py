#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


youtube_file = '../input/youtube8m-2019/vocabulary.csv'
youtube_data = pd.read_csv(youtube_file)
youtube_data.head()


# In[ ]:


plt.figure(figsize=(20,10))
plt.title('Youtube Video Count by Categories')
plot = sns.scatterplot(x=youtube_data['Vertical1'], y=youtube_data['TrainVideoCount'])
plt.xticks(rotation=90)
plt.xlabel('Categories')
plt.ylabel('Video Count')
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
plt.title('Youtube Video Count by Categories')
sns.swarmplot(x=youtube_data['Vertical1'], y=youtube_data['TrainVideoCount'])
plt.xticks(rotation=90)
plt.xlabel('Categories')
plt.ylabel('Video Count')
plt.show()

