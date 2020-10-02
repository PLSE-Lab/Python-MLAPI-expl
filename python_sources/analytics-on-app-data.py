#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
print(os.listdir("../input"))


# In[ ]:


df=pd.read_csv("../input/AppleStore.csv")


# In[ ]:


mycol=list(df.columns)
mycol.remove("Unnamed: 0")


# In[ ]:


df=df[mycol]


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.shape[0]


# Apps having price 0:

# In[ ]:


df1=df[df['price']==0]


# In[ ]:


df1.shape[0]


# In[ ]:


df1.head()


# Total catagories ofa applications:

# In[ ]:


len(list(set(df1['prime_genre'])))


# Catagory of application where price is 0?

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(df1['prime_genre'])
plt.xticks(rotation=90)


# Ratingwise distribution of Catogory in app store of Apple:

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(df1['prime_genre'],hue='cont_rating',data=df1)
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1, 1))


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(df1['prime_genre'],hue='user_rating',data=df1)
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1, 1))


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(df1['prime_genre'],hue='user_rating_ver',data=df1)
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1, 1))


# Distribution of user_rating irrespective to catogory:

# In[ ]:


sns.violinplot(x='user_rating',data=df1)


# In[ ]:


sns.violinplot(x='user_rating_ver',data=df1,color='M')


# In[ ]:


plt.figure(figsize=(25,10))
sns.violinplot(x='user_rating_ver',y='prime_genre',data=df1,palette='dark')


# same plot as above just in different orrientation:

# In[ ]:


plt.figure(figsize=(25,10))
sns.violinplot(y='user_rating_ver',x='prime_genre',data=df1,palette='dark')


# Count Plots:

# In[ ]:


plt.figure(figsize=(20,10))
df1['prime_genre'].value_counts().plot.bar()


# Wordcloud of the catogory where price is 0:

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
import wordcloud
from wordcloud import WordCloud, STOPWORDS


mpl.rcParams['font.size']=12                #10 
mpl.rcParams['savefig.dpi']=100           #72 
mpl.rcParams['figure.subplot.bottom']=.1 


stopwords = set(STOPWORDS)
stopwords.add('dtype')
stopwords.add('object')
stopwords.add('prime_genre')
stopwords.add('Name')
stopwords.add('Length')
data = pd.read_csv("../input/AppleStore.csv")

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(data['prime_genre']))

print(wordcloud)
fig = plt.figure(1)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
sns.jointplot(x='user_rating', y='rating_count_tot',data=df1,kind='hex')


# Price distribution for applications:

# In[ ]:


plt.figure(figsize=(20,10))
df['price'].value_counts().plot.bar()


# In[ ]:


plt.figure(figsize=(20,10))
df['price'].value_counts().plot.barh()


# In[ ]:


plt.figure(figsize=(20,10))
df['price'].value_counts().plot.area()


# In[ ]:


plt.figure(figsize=(20,10))
df['price'].value_counts().plot()


# In[ ]:


plt.figure(figsize=(20,10))
df['price'].value_counts().plot.pie()


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(df['price'],hue='prime_genre',data=df)
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1, 1))


# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
import wordcloud
from wordcloud import WordCloud, STOPWORDS


mpl.rcParams['font.size']=12                #10 
mpl.rcParams['savefig.dpi']=100           #72 
mpl.rcParams['figure.subplot.bottom']=.1 


stopwords = set(STOPWORDS)
stopwords.add('dtype')
stopwords.add('object')
stopwords.add('prime_genre')
stopwords.add('Name')
stopwords.add('Length')
data = pd.read_csv("../input/appleStore_description.csv")

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(data['app_desc']))

print(wordcloud)
fig = plt.figure(1)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:


plt.figure(figsize=(40,20))
sns.pairplot(df.drop("id", axis=1), hue="prime_genre", size=3)


# Upvote if Helpful
