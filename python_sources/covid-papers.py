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


# **JSON files**

# In[ ]:


import glob, json
root_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13'
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
all_json


# **Import data**

# In[ ]:


data=pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
print(data.shape)
data.head()


# In[ ]:


abstract=data['abstract'].dropna()
abstract


# In[ ]:


abstract=list(abstract)
abstract


# In[ ]:


publish=data['publish_time'].dropna()
publish


# In[ ]:


journals=data['journal'].dropna()
journals


# In[ ]:


#summary
data.describe()


# In[ ]:


# Nan values in all columns
data.isna().sum()


# In[ ]:


#Titles
title=data['title'].dropna()
title


# In[ ]:


get_ipython().system('git clone https://github.com/amueller/word_cloud.git')


# In[ ]:


import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt


# In[ ]:


#c:\intelpython3\lib\site-packages\matplotlib\__init__.py:
import warnings
warnings.filterwarnings("ignore")


# In[ ]:





# **For the first abstract**

# In[ ]:


# Create and generate a word cloud image:
wordcloud = WordCloud().generate(abstract[0])

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(abstract[0])
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


# Save the image in the img folder:
#wordcloud.to_file("img/first_review.png")


# In[ ]:


text = " ".join(review for review in abstract)


# In[ ]:


print ("There are {} words in the combination of all review.".format(len(text)))


# In[ ]:


# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["covid", "corona", "disease", "virus", "infection"])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




