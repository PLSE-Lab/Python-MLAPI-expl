#!/usr/bin/env python
# coding: utf-8

# This notebook is currently a test site for augmenting data by looking at previous ideas and possibly trying new ones. Here are three ideas, one for adding translations, one for creating synthetic data, and one for interjecting noise/variations.
# 
# 
# ### Translations
# 
# This idea originally came from the first Toxic challenge in which translation was used as an encoder-decoder. Pavel Ostyakov's [A simple technique for extending dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48038) explains how translating an english comment to another language, and then back to english, can improve model accuracy. With this being a multilingual competition, there have been similar ideas implemented but maybe not in this exact way.
# 
# In this notebook I'm using Google Translator via the googletrans package.

# In[ ]:


get_ipython().system('pip install git+https://github.com/ssut/py-googletrans.git')


# In[ ]:


import os
import pandas as pd
from googletrans import Translator


# In[ ]:


text = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/'                        'jigsaw-toxic-comment-train.csv', 
                        nrows=10_000)


# TODO: Get the proportion of languages in the test set and set a randomized language per comment with np.random.choice()

translator = Translator()
for i,t in enumerate(text.comment_text[19:22]):
    try:
        encoded = translator.translate(t, dest='fr').text
        decoded = translator.translate(encoded, dest='en').text
        print(f"\nSet {i}\n"
              f"Original: {t}\n\n"
              f"Recoded: {decoded}\n")
    except: pass


# ### Synthetic comments
# 
# Here I'm using [Markovify](https://github.com/jsvine/markovify) to generate additional toxic commnents. This package uses Markov chains to string together new sequences of words based on previous sequences.

# In[ ]:


import markovify as mk


# In[ ]:


doc = text.loc[text.toxic == 1, 'comment_text'].tolist()
text_model = mk.Text(doc)
for i in range(10):
    print(text_model.make_sentence())


# Only a few lines of code and you too can sound like an angry 5th grader! Sometimes this technique produces a bit of nonsense but the toxic keywords are in there. It may be a way to add toxic comments and get a more balanced dataset.

# ### Various variations
# 
# The [nlpaug](https://github.com/makcedward/nlpaug) package contains a variety of augmentations to supplement text data and introduce noise that may help your model generalize. Here is a summary of a few functions:
# 
# <img src="https://github.com/makcedward/nlpaug/blob/master/res/textual_example.png?raw=true" width="600">
# 
# The package has many more augmentations -at the character, word, and sentence levels. 
# 
#  - Character Augmenter
#     - OCR
#     - Keyboard
#     - Random
#  - Word Augmenter
#     - Spelling
#     - Word Embeddings
#     - TF-IDF
#     - Contextual Word Embeddings
#     - Synonym
#     - Antonym
#     - Random Word
#     - Split
#  - Sentence Augmenter
#     - Contextual Word Embeddings for Sentence

# There's definiteiy more to explore here. Maybe these ideas can improve your model. Good luck!

# In[ ]:




