#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sys
import codecs
import nltk
import re
import math

forum_posts = pd.read_csv("../input/meta-kaggle/ForumMessages.csv")


# In[ ]:


def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return(re.sub(clean, '', text))


# In[ ]:


text = forum_posts.Message.astype(str).str.cat(sep=' ')
text = remove_html_tags(text)


# In[ ]:


# tokenize
words = nltk.word_tokenize(text)

# Remove single-character tokens (mostly punctuation)
words = [word for word in words if len(word) > 1]

# Remove numbers
words = [word for word in words if not word.isnumeric()]

# remove non-breaking space
words = [word for word in words if word != "nbsp"]

# Lowercase all words (default_stopwords are lowercase too)
words = [word.lower() for word in words]


# Calculate frequency distribution
fdist = nltk.FreqDist(words)

# Output words
#for word, frequency in fdist.most_common():
#    print(u'{},{},{},{}'.format(word,
#                                frequency,
#                                math.log(frequency),
#                                math.log(frequency)/math.log(fdist.most_common(1)[0][1])))


# In[ ]:


with open("kaggle_lex_freq.csv", "w") as fp:
    fp.writelines("word,raw_freq,log_freq,saliency\n")
    for word, frequency in fdist.most_common():
        fp.write(u'{},{},{},{}'.format(word,
                                    frequency,
                                    math.log(frequency),
                                    math.log(frequency)/math.log(fdist.most_common(1)[0][1])))
        fp.write("\n")
    

