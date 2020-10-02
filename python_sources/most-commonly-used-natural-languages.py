#!/usr/bin/env python
# coding: utf-8

# # Languages used in ISIS tweets
# 
# In this short notebook, I'm using [langid](https://github.com/saffsd/langid.py) to detect the language used in the tweets in this dataset. Most of them are English. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

import langid # identify languages based on tweets

df = pd.read_csv('../input/tweets.csv')

# identify languages
predicted_languages = [langid.classify(tweet) for tweet in df['tweets']]

lang_df = pd.DataFrame(predicted_languages, columns=['language','value'])

# show the top ten languages & their counts
print(lang_df['language'].value_counts().head(10))

# plot the counts for the top ten most commonly used languages
colors=sns.color_palette('hls', 10) 
pd.Series(lang_df['language']).value_counts().head(10).plot(kind = "bar",
                                                        figsize=(12,9),
                                                        color=colors,
                                                        fontsize=14,
                                                        rot=45,
                                                        title = "Top 10 most common languages")


# So it's obvious that English is the favored language in this dataset by a long shot. For reference, the codes used are [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) codes. The next most common languages are Arabic, French, Malay, Indonensian, Latin (?), Swahili, German, and Dutch. 
