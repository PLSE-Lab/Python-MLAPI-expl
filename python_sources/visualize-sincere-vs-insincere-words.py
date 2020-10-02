#!/usr/bin/env python
# coding: utf-8

# ## Visualize Sincere vs Insincere words
# 
# This is my first kernel and i decided to visualize common words that appear in sincere sentences vs insincere sentences.

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

from nltk.corpus import stopwords
from wordcloud import WordCloud


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# ### Basic Stats
# 
# There are around 1.3 million questions and only 80k insincere ones, so the training dataset is very unbalanced.

# In[ ]:


print('total', train.shape[0])
print('sincere questions', train[train['target'] == 0].shape[0])
print('insincere questions', train[train['target'] == 1].shape[0])


# ### Vocabulary
# 
# We could just use one of the vectorizers and get the words used in the sincere and insincere sentences but i just decided to write my own. It counts the words in a sentence and builds a dict of word -> count. The tokenizer itself is a very simple split by space. 
# 
# We remove stopwords because they are common for both sincere as well as insincere words.

# In[ ]:


class Vocabulary(object):
    
    def __init__(self):
        self.vocab = {}
        self.STOPWORDS = set()
        self.STOPWORDS = set(stopwords.words('english'))
        
    def build_vocab(self, lines):
        for line in lines:
            for word in line.split(' '):
                word = word.lower()
                if (word in self.STOPWORDS):
                    continue
                if (word not in self.vocab):
                    self.vocab[word] = 0
                self.vocab[word] +=1 


# In[ ]:


sincere_vocab = Vocabulary()
sincere_vocab.build_vocab(train[train['target'] == 0]['question_text'])
sincere_vocabulary = sorted(sincere_vocab.vocab.items(), reverse=True, key=lambda kv: kv[1])
for word, count in sincere_vocabulary[:10]:
    print(word, count)


# In[ ]:


insincere_vocab = Vocabulary()
insincere_vocab.build_vocab(train[train['target'] == 1]['question_text'])
insincere_vocabulary = sorted(insincere_vocab.vocab.items(), reverse=True, key=lambda kv: kv[1])
for word, count in insincere_vocabulary[:10]:
    print(word, count)


# As we can clearly see there are certain words that count high in both sincere as well as insincere sentences and they are not in the stopword list. 
# 
# A simple metric to remove these is to use the ratio of sincere and insincere words and vice versa as the score for the word.

# In[ ]:


sincere_score = {}
for word, count in sincere_vocabulary:
    sincere_score[word] = count / insincere_vocab.vocab.get(word, 1)

wordcloud_sincere = WordCloud(width = 800, height = 800,background_color ='white', min_font_size = 10)
wordcloud_sincere.generate_from_frequencies(sincere_score) 
  
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud_sincere) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# In[ ]:


insincere_score = {}
for word, count in insincere_vocabulary:
    insincere_score[word] = count / sincere_vocab.vocab.get(word, 1)

wordcloud_insincere = WordCloud(width = 800, height = 800,background_color ='white', min_font_size = 10)
wordcloud_insincere.generate_from_frequencies(insincere_score) 
  
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud_insincere) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# ## Conclusion
# As we can clearly see there are certain words (swear words, discriminatory words based on race, political figures etc) that show up a lot in insincere sentences.
