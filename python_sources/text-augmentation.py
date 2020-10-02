#!/usr/bin/env python
# coding: utf-8

# ### Text augmentation with word2vec
# 
# The following script can be used for text augmentation.
# Basically, for each word in the corpus, a candidate list of synonyms is created by considering nearby words (as defined by the word2vec distance). This list is then filtered, only keeping those synonyms for which the POS tag is the same as the original word. 
# 
# The idea described above is adapted from the nicely written article https://towardsdatascience.com/data-augmentation-for-natural-language-processing-6ae928313a3f
# 
# 

# In[ ]:


import numpy as np 
import operator
import pandas as pd 
import os
from tqdm import tqdm_notebook as tqdm


# In[ ]:


import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
from nltk import pos_tag
import gensim
from gensim.models import KeyedVectors
import re
from nltk.corpus import wordnet as wn
import os


# In[ ]:


TEXT_COL = 'comment_text'
EMB_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', index_col='id')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv', index_col='id')


# In[ ]:


word2vec = KeyedVectors.load_word2vec_format(EMB_PATH)


# In[ ]:


# Inspired from https://towardsdatascience.com/data-augmentation-for-natural-language-processing-6ae928313a3f
def get_similar_words(word_list, word_index, threshold=0.8):
    word_tags = pos_tag(word_list)
    similar_words = {}
    for idx, word in tqdm(enumerate(word_list)):
        if word in word2vec.wv.vocab:
            #get words with highest cosine similarity
            replacements = word2vec.wv.most_similar(positive=word, topn=5)
            #keep only words that pass the threshold
            replacements = [replacements[i][0] for i in range(5) if replacements[i][1] > threshold]
            #check for POS tag equality, dismiss if unequal
            replacements = [elem for elem in replacements if pos_tag([elem.lower()])[0][1] == word_tags[idx][1]]
            # Dismiss upper-case similar replacements and OOV words.
            replacements = [elem for elem in replacements if elem.lower() != word.lower() and elem in word_index and elem in word2vec.wv]
            if len(replacements) > 0:
                similar_words[word] = replacements
            if idx % 500 == 0:
                print(word, replacements)
    # One could also reverse the dictionary or postprocess it further
    return similar_words


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=None, lower=True) #filters = ''
#tokenizer = text.Tokenizer(num_words=max_features)
print('fitting tokenizer')
tokenizer.fit_on_texts(list(train[TEXT_COL]) + list(test[TEXT_COL]))
word_index = tokenizer.word_index


# In[ ]:


sorted(word_index.items(), key=operator.itemgetter(1))[:100]


# In[ ]:


sample = list(word_index.keys())[:20]
print(sample)
for i in range(10):
    print(get_similar_words(sample, word_index, threshold=i * 0.1))
    print(f'threshold at: {i*0.1}')
    print('*'*50)


# In[ ]:


similar_words_dict = get_similar_words(list(word_index.keys())[:50_000], word_index, threshold=0.3)
len(similar_words_dict)


# In[ ]:


for word,synonyms in list(similar_words_dict.items())[:20]:
   print(word,synonyms)


# In[ ]:


np.random.seed(0)

def get_synonym_list(word, dictionary):
    if word in dictionary.keys():
        return dictionary[word]
    return [word]

def replace_text(text):
    return ' '.join([np.random.choice(get_synonym_list(word, similar_words_dict)) for word in text.split(' ')])    


# In[ ]:


replace_text('anyone likes 1 apple and 2 oranges ?')


# In[ ]:


train['comment_text_augmented'] = train['comment_text'].apply(lambda x: replace_text(x))
test['comment_text_augmented'] = test['comment_text'].apply(lambda x: replace_text(x))
train[['comment_text', 'comment_text_augmented']].head(50)


# In[ ]:


train.to_csv('augmented_train.csv', index=False)
test.to_csv('augmented_test.csv', index=False)

