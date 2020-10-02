#!/usr/bin/env python
# coding: utf-8

# The purpose of the notebook is to walk through a few useful functions that can be used for text preprocessing.
# 1. For Preprocessing Text (with a couple of effective inclusions)
# 2. A slightly more intuitive way of building word vocabulary
# 3. Checking instances of mis-spelt or non-standardized words
# 4. Correcting mispelt words
# 5. Preparing the Embedding Matrix
# 
# Since other notebooks have covered visualizations and model building in great detail, I wont be emphasizing on those parts.

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


# In[ ]:


import pandas as pd
train = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")


# In[ ]:


train.info()


# In[ ]:


train.head(5)


# In[ ]:


# RETAINING THE 'TEXT' VARIABLE AS WE WILL BE WORKING OFF THAT 

train = train[['text']]


# In[ ]:


# SHALL USE CONTRACTIONS FOR DATA PRE-PROCESSING (EXPLAINED BELOW)
get_ipython().system(' pip install contractions')


# ### Function 1 `basic_preprocessor` - Preprocesses Text
# 
# **On Contractions**
# The Contraction library expands out words, for example *doesn't* is expanded *does not* and *won't* as *will not*. In my experience, LSTM models work well with contractions.
# 
# **Expanding Out `'s`**
# In the proprocessor below, I have also implemented the expansion of *apostrophe s*, for example the word *sister's* is expanded as *sister s*. In my experience, I have observed that LSTMs are able to process this information.
# 
# Besides these points, the function implements a set of standard regular expressions to systematically cleanse text

# In[ ]:


# FUNCTION TO IMPLEMENT BASIC TEXT PREPROCESSING : LOWER-CASING & REMOVING NUMBERS/PUNCTUATION

import re
import string
import contractions


def basic_preprocessor(text):
    
    """
    Input, text to be preprocessed in the string format
    returns the preprocessed text
    """
  
    import re
    import string
    import contractions
  

    # EXPANDING OUT CONTRACTIONS
    # e.g : don't -> do not
    text = contractions.fix(text)

    # TEXT TO LOWERCASE
    text = text.lower()

    # The syntax of re.sub() is:: re.sub(pattern, replace, source_string)
    
    # CODE TO HANDLE POSSESSIVES ('s) 
    # e.g: movie's -> movie s
    # THE REASON IS THAT LSTMs ARE ABLE TO PROCESS "s" 
    text = re.sub(r"(\w+)'s", r'\1 s', text)
  
    # HANDLING OTHER PUNCTUATION
    text = re.sub('\[.*#?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    return text


# Lambda functions greatly speed up text preprocessing. [This](https://medium.com/@chaimgluck1/have-messy-text-data-clean-it-with-simple-lambda-functions-645918fcc2fc) is a good introductory note of using these functions

# In[ ]:


# PREPROCESSING THE TRAINING SET
train['text'] = train.apply(lambda x: basic_preprocessor(x['text']),axis=1)


# ### Function 2 `build_vocab` - List of Sentences to Vocabulary
# 

# In[ ]:


# DEFINING A FUNCTION THAT CONVERTS A LIST OF SENTENCES TO A VOCABULARY DICTIONARY

from tqdm import tqdm # TQDM LIBRARY IS USED TO CREATE A PROGRESS BAR


def build_vocab(sentences):

    """
    Purpose, Converts a list of sentences into a vocabulary dictionary
    Input, a list of sentences
    Prints, the size (length) of the vocabulary
    Returns, a vocabulary dictionary -> {word: frequency}
    """

    length = len(sentences) # NO OF SENTENCES
    vocab ={} 

    for s in tqdm(sentences): # ACCESSING EACH SENTENCE IN A LIST OF SENTENCES       
        temp_list = [] 
        temp_list = s.split() # LIST OF WORDS IN A SENTENCE

        for word in temp_list:
            if word not in vocab.keys():
                vocab[word] = 1
            else:
                vocab[word] += 1

    print("The size of the vocabulary is : ", len(vocab))

    return vocab


# In[ ]:


# CONVERTING THE TEXT FIELD INTO A LIST OF SENTENCES

sentences = []
m = len(train['text'])

for i in range(m):
    sentences.append(train['text'].iloc[i])

print("Number of sentences : ", len(sentences))


# In[ ]:


# CHECKING IF THE LISTS OF LISTS HAS BEEN CREATED
for i in range(5):
    print("\n", sentences[i])


# In[ ]:


# IMPLEMENTING WORD FREQUENCY FUNCTION, STORING RESULTS IN master_vocab VARIABLE
master_vocab = build_vocab(sentences)


# ### Using Pre-Trained Embeddings to Determine Extent of Data Standardization

# In[ ]:


# IMPORTING THE GLOVE 100 DIMENSIONAL EMBEDDINGS
# THANKS TO: LAURENCE MORONEY, AI ADVOCATE AT GOOGLE  


get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt     -O /tmp/glove.6B.100d.txt')

embeddings_index = {};

with open('/tmp/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;


# In[ ]:


# EXPLORING THE EMBEDDINGS_INDEX DICTIONARY

print("Glove vocabulary size: ",len(embeddings_index))
print("Glove embedding dimensions: ",len(embeddings_index['the']))


# Just to recap:
# 1. We have two sets of dictionary: `master_vocab`, that representes the vocabuary of the processed tweets. `embeddings_index`, which represents the vocabulary of the GloVe embeddings
# 2. Given that the GloVe index has 400,000 words (which is huge), we can use this to benchmark our vocabulary (`master_vocab`). Ths is what the next function does

# ### Function 3 `check_overlap` - Determine the quality of vocabulary of the corpus (Tweet Vocabulary)

# In[ ]:


# WRITING A FUNCTION TO DETERMINE THE OVERLAP BETWEEN THE 2 DICTIONARIES

from tqdm import tqdm
import operator

def check_overlap(vocab,embeddings):
    
    """
    Purpose, to determine the overlap between the corpus and the embeddings disctionary
    Inputs, 2 dictionaries, one for the vocabulary and the other for the corpus
    Prints, a set of useful information on the overlap
    Returns, a sorted version dictionary with out_of_vocabulary words and their frequency
    """
  
    not_in_glove = {}
    total_corpus_words = 0
    total_oov_words = 0

    for i in tqdm(vocab.keys()):
        total_corpus_words += vocab[i] 

        if i not in embeddings.keys():
            not_in_glove[i] = vocab[i]
            total_oov_words += vocab[i]

    x = len(not_in_glove)
    y = len(vocab)
    z = len(embeddings)

    print("\n\nVOCABULARY INSIGHTS:")
    print("The vocabulary size is (unique word-count) : ",y)
    print(f"Embeddings found for {y-x} ({round((y-x)*100/y,2)})% of the words")
  
    print("\nCORPUS INSIGHTS:")
    print("The corpus size is (total, non-unique word-count) : ",total_corpus_words)
    print(f"{total_oov_words} word(s), representing, {round(total_oov_words*100/total_corpus_words,2)}% of the corpus vocabulary is unmapped")

    print("\n Top 20 words (by frequency) not present in embeddings dictionary :")
    p = min(len(not_in_glove), 20)
    print(sorted(not_in_glove, key = lambda x: (-not_in_glove[x], x))[0:p])

    not_in_glove_sorted = sorted(not_in_glove.items(), key=operator.itemgetter(1))[::-1]
  

    return not_in_glove_sorted


# In[ ]:


# IMPLEMENTING THE FUCNTION (oov -> out of vocabulary)

oov = check_overlap(master_vocab,embeddings_index)


# In[ ]:


# ACCESSING THE TOP 20 WORDS

oov[0:20]


# The next step is to correct some of these mis-spellings. Dr. Cristof Henkel, a Kaggle Grandmaster has written an excellent function for the same. Reproducing it here.
# 
# ### Function 4 `_get_mispell` and `replace_typical_misspell` - Correct mispelt words

# In[ ]:


# DEFINE A MIS-SPELLING DICTIONARY

mispell_dict = {
  'prebreak': 'pre break',
  'nowplaying': 'now playing',
  'typhoondevastated': 'typhoon devastated',
  'lmao': 'funny'
}


# In[ ]:


# AUTHOR: Dr. CRISTOF HENKEL

import re

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispellings, mispellings_re = _get_mispell(mispell_dict)



def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


# In[ ]:


# IMPLEMENTING THE FUNCTION 

train["text"] = train["text"].apply(lambda x: replace_typical_misspell(x))


# Next up, I will be tokenizing the sequence using Keras' inbuilt library. Since, tokenizing isn't a focus point for this notebook, I'll *rush through* this section and move to the part where the embeddings matrix is generated.

# In[ ]:


# SPECIFYING HYPERPARAMETERS 
# REPRESENTED IN CAPS AS PER CONVENTION
VOCAB_SIZE = 7000
EMBEDDING_DIM = 20
MAX_LENGTH = 15   
TRUNC_TYPE = 'post'
OOV_TOKEN = "<OOV>"

# IMPORTING LIBRARIES AND SETTING-UP TOKENIZER
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# IMPLEMENTING TOKENIZER
tokenizer = Tokenizer(num_words = VOCAB_SIZE, oov_token = OOV_TOKEN)
tokenizer.fit_on_texts(train['text'])

# I haven't included the rest of the Tokenizer code


# Next up, I will be accessing the `word_index` dictionary which is in the {Word: Code} format. The words that occur more frequently in the corpus have lower "code" values

# In[ ]:


word_index = tokenizer.word_index

print("\n Data-Type and Length of word_index :", type(word_index), len(word_index))
print("\n Value for a common word 'the' :", word_index['the'])
print("\n Value for a less common word 'india' :", word_index['india'])
print("\n Value for an oov word 'mtvhottest' :", word_index['mtvhottest'])


# ### Function 5 `gen_embeddings_matrix` - Generate Embeddings Matrix

# In[ ]:


# A FUNCTION TO GENERATE EMBEDDINGS_MATRIX


def gen_embeddings_matrix(max_features, word_index,embeddings_index):

    """
    Purpose, to generate embedding_matrix
    Inputs, no of features, word_index and embeddings_index
    Returns, embedding_matrix of size (max_features * embedding_dims)
    """

    # CONVERTING THE EMBEDDINGS DIMENSIONS TO NUMPY (100 d)
    all_emb_dims = np.stack(embeddings_index.values())

    # CALCULATING MEAN AND SD FOR THE DIMENSIONS
    emb_mean = all_emb_dims.mean()
    emb_std = all_emb_dims.std()

    # NO OF EMBEDDING DIMENSIONS
    embed_size = all_emb_dims.shape[1]

    # EFFECTIVE VOCAB SIZE
    no_words = min(max_features, len(word_index)) # no of words

    # CREATING AN EMBEDDINGS MATRIX
    # INITIALIZING IT WITH MEAN AND SD VALUES (EMBEDDINGS INDEX)
    np.random.seed(42)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (no_words, embed_size))

    # LOGIC TO CHECK AND POPULATE EMBEDDING MATRIX

    for word in word_index.keys():
    
        # IGNORE IF WORD IS "BEYOND" THE FREQ RANGE
        if word_index[word] >= max_features: 
            continue

        # CHECK IF WORD IS IN THE EMBEDDINGS_INDEX OR IS AN OOV WORD
        if word in embeddings_index.keys(): 
            embedding_vector = embeddings_index[word]
            embedding_matrix[word_index[word]] = embedding_vector

    return embedding_matrix


# In[ ]:


# IMPLEMENTING THE FUNCTION TO GENERATE EMBEDDINGS MATRIX

max_features = VOCAB_SIZE
word_index = tokenizer.word_index

embedding_matrix = gen_embeddings_matrix(max_features, word_index,embeddings_index)


# The next step is to coclusively establish the link between the `word_index`, the `embeddings_index` and `embeddings_matrix`

# In[ ]:


# TO CONCLUSIVELY ESTABLISH THE LINK BETWEEN word_index, embeddings_index AND embeddings_matrix

word_index['the'] # equals 2

# CHECK IF embeddings_index[2] = embedding_matrix[2] = embeddings FOR 'the' WORD
print(np.all(embeddings_index['the'] == embedding_matrix[2] ))


# This `embedding_matrix` can be used in the embedding layer of the sequence based model.

# In[ ]:


# keras.layers.Embedding(input_dim = vocab_size,
#                                          output_dim = embed_size,
#                                          input_length = sentence_length,
#                                          weights=[embedding_matrix], trainable= False)


# I hope that you found this notebook to be useful. [This](https://github.com/raamav/Text-Classification/blob/master/Twitter_for_PublicEmergencies_Part1.ipynb) notebook, located on my github page has the complete implementation details. 
# 
# **Thanks for reading**

# In[ ]:





# 
