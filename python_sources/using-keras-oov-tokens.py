#!/usr/bin/env python
# coding: utf-8

# # Using Keras OOV tokens
# 
# In this quick kernel I'm going to demonstrate how you can use an OOV token with Keras' tokenizer. If you are using an RNN hopefully this will give you a slight edge in training

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


from keras.preprocessing.text import Tokenizer


# So let's read in some data and then do what basically every public kernel is doing in the Quora Insincere Questions Classification competition is doing

# In[ ]:


df = pd.read_csv('../input/train.csv')


# Just a note on features - in this competition you see people using `max_features = 95000` a lot. If you're not aware already, `max_features` corresponds to the number of unique words you're interested in. 95000 is far less than the total number of unique words in the training set. This is over 200,000 words depending on how you do your cleaning.
# 
# So to train the tokenizer you would do this

# In[ ]:


max_features = 95000
maxlen = 60

# I'm just going to limit cleaning to lowering the string and putting spaces around stuff for now, you could do far more I guess
def clean_str(x):
    x = str(x)
    x = x.lower()
    
    specials = [',', '?']
    for s in specials:
        x = x.replace(s, f' {s} ')
        
    return x


df['question_text'] = df['question_text'].apply(clean_str)

tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(list(df['question_text'].values))


# OK, so what does that actually do? Let's tokenize a simple sentence with a misspelt word and a question mark

# In[ ]:


some_string = "burger king doesn't sell hamberders, does maccy ds?"
some_string = clean_str(some_string)


# In[ ]:


our_sent = tokenizer.texts_to_sequences([some_string])
our_sent


# Let's now use the tokenizer to return this vector back to a string and see what we get

# In[ ]:


tokenizer.sequences_to_texts(our_sent)


# There are two really important things here - firstly our misspelt and rare words are just gone. That's really bad, we're trying to judge if a sentence is sincere and part of Quora's critera is that the sentence is gramatically correct - we've just broken that. There is also information in the fact that the word was uncommon enough to not be in the tokenizer.
# 
# Another issue is the tokenizer has stripped `,` and `?`. We might not care so much about `,`s but part of the critera for a sincere question is it is in fact a question, a `?` undoubtably helps us here.
# 
# 
# ## Second attempt - use an OOV token
# 
# Keras lets us define an Out Of Vocab token - this will replace any unknown words with a token of our choosing. This is better than just throwing away unknown words since it tells our model there was information here.
# 
# Let's do that

# In[ ]:


tokenizer_2 = Tokenizer(num_words=max_features, oov_token='OOV')
tokenizer_2.fit_on_texts(list(df['question_text'].values))


# In[ ]:


our_sent_2 = tokenizer_2.texts_to_sequences([some_string])
our_sent_2


# In[ ]:


tokenizer_2.sequences_to_texts(our_sent_2)


# ## Third attempt - use question marks
# 
# Finally, let's fix the `?` issue. The `?` is being filtered out by the tokenizer, we can solve this by specifying the filters ourselves

# In[ ]:


tokenizer_3 = Tokenizer(num_words=max_features, oov_token='OOV', filters='!"#$%&()*+,-./:;<=>@[\]^_`{|}~ ')
tokenizer_3.fit_on_texts(list(df['question_text'].values))


# In[ ]:


our_sent_3 = tokenizer_3.texts_to_sequences([some_string])
our_sent_3


# In[ ]:


tokenizer_3.sequences_to_texts(our_sent_3)


# This looks much more like something we'd like to train against
