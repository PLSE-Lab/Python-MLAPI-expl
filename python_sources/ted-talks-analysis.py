#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/ted_main.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df[df.isnull().any(axis=1)]


# There are **six NaN** in the dataset. All NaN are in the column "**speaker_occupation**"

# ## How have TED talks evolved?

# In[ ]:


df['published_date'] = pd.to_datetime(df['published_date'], unit='s')


# In[ ]:


published_date_year = df.groupby(df['published_date'].dt.year)['published_date'].count()


# In[ ]:


bar_labels = published_date_year.keys()
x_pos = list(range(len(published_date_year)))

plt.bar(x_pos,
        # using the data from the mean_values
        published_date_year, 
        # aligned in the center
        align='center',
        # with color
        color='#FFC222')

plt.ylabel('Count')
plt.xticks(x_pos, bar_labels, rotation='vertical')
plt.title('Number of talks per year')

plt.show()


# ## What are the most viewed talks?

# In[ ]:


most_viewed = df.sort_values(by='views', ascending=False)
most_viewed


# ### What are the topics of the most viewed talks?

# In[ ]:


wordcloud = WordCloud(max_font_size=40).generate(' '.join(df['tags']))
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# ## Which aspects affect the visits the most?

# In[ ]:


df.corr()


# The number of views are correlated with:
# * Comments: if a video has more views it usually has more comments
# * Languages: the more languages a video has, the more people can understand what it says
