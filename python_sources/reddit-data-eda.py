#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('../input/dataisbeautiful/r_dataisbeautiful_posts.csv')


# In[ ]:


data.head()


# In[ ]:


data['title'][55]


# In[ ]:


data.score.nunique()


# In[ ]:


(data.isnull().sum() / len(data)) * 100


# In[ ]:


data.head()


# In[ ]:


del data['id']
del data['author_flair_text']
del data ['removed_by']
del data ['total_awards_received']
del data['awarders']
del data ['created_utc']
del data ['full_link']


# In[ ]:


data.head(100)


# In[ ]:


data.describe().T


# In[ ]:


len(data)


# In[ ]:


print(len(data[data['score'] < 20]), 'Posts with less than 20 votes')
print(len(data[data['score'] > 20]), 'Posts with more than 20 votes')


# In[ ]:


print(len(data[data['num_comments'] < 20]), 'Posts with less than 20 comments')
print(len(data[data['num_comments'] > 20]), 'Posts with more than 20 comments')


# In[ ]:


data[data['score'] == data['score'].max()]['title'].iloc[0]


# In[ ]:


from wordcloud import WordCloud, STOPWORDS


# In[ ]:


words = data['title'].values


# In[ ]:


type(words[25])


# In[ ]:


ls = []

for i in words:
    ls.append(str(i))


# In[ ]:


ls[325]


# In[ ]:


type(words[325])


# In[ ]:


ls[0]


# In[ ]:


# The wordcloud of Cthulhu/squidy thing for HP Lovecraft
plt.figure(figsize=(16,13))
wc = WordCloud(background_color="red", stopwords = STOPWORDS, max_words=2000, max_font_size= 300,  width=1600, height=800)
wc.generate(" ".join(ls))
plt.title("Most Discussed Terms", fontsize=20)
plt.imshow(wc.recolor( colormap= 'viridis' , random_state=17), alpha=0.98, interpolation="bilinear", )
plt.axis('off')


# In[ ]:


most_pop = data.sort_values('score', ascending  = False )[['title','score']].head(12)
most_pop['score1'] = most_pop['score']/1000


# In[ ]:


import matplotlib.style as style
style.available 


# In[ ]:


style.use('seaborn-notebook')


# In[ ]:


plt.figure(figsize = (20, 15))
sns.barplot(data = most_pop, y = 'title', x = 'score1', color = 'C')
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=21, rotation=0)
plt.xlabel('Votes in Thousands', fontsize = 21)
plt.ylabel('')
plt.title('Most popular posts', fontsize = 30)


# In[ ]:


data.head()


# In[ ]:




