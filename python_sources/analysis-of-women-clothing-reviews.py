#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_rough= pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv")
df= df_rough.iloc[:,1:11]       #df.iloc[5:10,3:8] # rows 5-10 and columns 3-8


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# This dataset has 23486 entries and 10 columns. Some of the entries are missing like Title, Division Name, Department Name, and Class Name.
# 
# Is there any correlation between user's rating and reviews length ?

# In[ ]:


df['Review Text']=df['Review Text'].astype(str)
df['Review Length']=df['Review Text'].apply(len)


# In[ ]:


g = sns.FacetGrid(data=df, col='Rating')
g.map(plt.hist, 'Review Length', bins=50)


# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(data=df.corr(), annot= True, cbar=False, center= .8)


# the above table shows that there is very little correlation between feedback count and review length. lets draw the heat map to get the point visually.

# the above map shows that Age is not correlated with review length and even review length and positive feedback count have a correlation of .21 which shows low correlation between the two
# 
# lets divide the Age column into groups and make a stack bar chart with the ratings to see if certain age group has liking tgowards the clothings

# In[ ]:


AgeCat= pd.cut(df['Age'], np.arange(start=0, stop=100, step=10))
print(AgeCat.unique()) 


# In[ ]:


plt.figure(figsize=(2,10))
df.groupby(['Rating', AgeCat]).size().unstack(0).plot.bar(stacked=True)


# people who are between 30 and 40 years old gave most rating to differnet items. people below 20 and above 70 years do not care to give ratings.

# In[ ]:


df.groupby(['Department Name', pd.cut(df['Age'], np.arange(0,100,10))]).size().unstack(0).plot.bar(stacked=True)


# from the above figure we can say that overall in each age group 'Tops' have been given most ratings....which may imply that 'Tops' is the most loved and most profitable department for this Store, Dresses came next and then Bottoms. This can furthur be emphasized by the following chart

# In[ ]:


z=df.groupby(by=['Department Name'],as_index=False).count().sort_values(by='Class Name',ascending=False)
print(z.head())


# In[ ]:


sns.set_style("whitegrid")
ax = sns.barplot(x=z['Department Name'],y=z['Class Name'], data=z)
plt.ylabel("Count")
plt.title("Counts Vs Department Name")


# Now lets explore the Comments that user have given.......we will make a word cloud for the most used words in positive feedback and also in negative feedback.

# In[ ]:


a= df.groupby('Department Name')


# In[ ]:


a.describe().head()


# In[ ]:


from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[ ]:


import re
text = " ".join(review for review in df['Review Text'])
print ("There are {} words in the combination of all review.".format(len(text)))
text= re.sub('(\s+)(a|an|and|the|on|in|of|if|is|i|)(\s+)', '', text)  
print ("There are {} words in the combination of all review.".format(len(text)))


# In[ ]:


# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.add("fabric")

# Generate a word cloud image
wordcloud = WordCloud(background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


import collections
words = text.split()
print(collections.Counter(words))


# In[ ]:




