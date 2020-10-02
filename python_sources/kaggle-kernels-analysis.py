#!/usr/bin/env python
# coding: utf-8

# This notebook is dedicated to investigation of Kaggle popular kernels.  What I am going to investigate:
# 
# * Number of votes received
# * Most popular tags
# * Number of comments
# * Number of views
# * Number of forks
# * Script vs notebook
# * Python vs R (of course!)

# # Libraries Import

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from wordcloud import WordCloud, STOPWORDS


# # Data

# Let's load Kernels dataset: 

# In[ ]:


data = pd.read_csv("../input/voted-kaggle-kernels.csv")
print(data.head())

data.shape


# We see 971 rows in Kernels dataset.  Let's fill NANs with zeros and cast `Views` and `Forks` columns to integer values.

# In[ ]:


data[['Views', 'Forks']] = data[['Views', 'Forks']].fillna(0)

data[['Tags']] = data[['Tags']].fillna("")

data[['Views', 'Forks']] = data[['Views', 'Forks']].astype('int')


# # Plots

# In[ ]:


plt.figure(figsize=(15,10))
plt.hist(data['Votes'], 100)
plt.xticks(range(0, 2500, 250))
plt.show()


# In[ ]:


data['Votes'].max()


# In[ ]:


data['Votes'].idxmax()


# So the maximum number of votes one kernel has received is 2130 but usual kernel get less than 100 votes.

# Now let's investigate what tags are the most popular ones:

# In[ ]:


from collections import Counter
words = Counter()
data['Tags'].str.lower().str.split().apply(words.update)
print(words.most_common(10))

stopwords = set(STOPWORDS)

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(data['Tags']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# So "data" takes the first place, "visualization" takes the second place and "eda" takes the third place.

# Now let's study distribution of comments number:

# In[ ]:


plt.figure(figsize=(15,10))
plt.hist(data['Comments'], 100)
plt.xticks(range(0, 1000, 100))
plt.show()


# In[ ]:


data['Comments'].max()


# In[ ]:


data['Comments'].idxmax()


# So - kernels get less comments than votes which is not surprising! The most popular kernel gets 749 comments, average kernels less than 50 comments. By this moment you may start asking the question what is this kernel with 749 comments and 2130 votes? No wonder, it is Exploring Survival on the Titanic by Megan Risdal. 

# Let's plot the correlation between votes and comments to illustrate the same from another point of view.

# In[ ]:


plt.scatter(data['Votes'], data['Comments'], color="blue", linewidth=0.15)
plt.xlabel("Votes")
plt.ylabel("Comments")
plt.show()


# We see obviously positive correlation between votes and comments.

# Now let's plot number of views. 

# In[ ]:


plt.figure(figsize=(15,10))
plt.hist(data['Views'], 100)
plt.show()


# In[ ]:


data['Views'].max()


# In[ ]:


data['Views'].idxmax()


# And again Exploring Survival on the Titanic wins with 345590 views!  Now the final histogram - number of forks:

# In[ ]:


plt.figure(figsize=(15,10))
plt.hist(data['Forks'], 100)
plt.show()


# In[ ]:


data['Forks'].max()


# In[ ]:


data['Forks'].idxmax()


# Surprise! It is not  Exploring Survival on the Titanic but Titanic Data Science Solutions by Manav Sehgal ! Pay attention that distribution of views, comments, forks and votes is very much the same.

# What do you see more often on Kaggle - notebooks or scripts? Answer is below.

# In[ ]:


data['Code Type'].value_counts().plot(kind='pie')
plt.show()


# And now the final fight - Python vs R...

# In[ ]:


data['Language'].value_counts().plot(kind='pie')
plt.show()


# So Python wins!
