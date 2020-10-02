#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The Quora Insincere Question Classification competition allows us to use the four embeddings:  glove.840B.300d (GloVe), paragram_300_sl999 (paragram),  wiki-news-300d-1M (wiki) and GoogleNews-vectors-negative300 (GoogleNews). In a kernel titled: ["How to: Preprocessing when Using Embeddings"](https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings), the author raises the issue of tokenization and its effect on how much of the training vocabulary is covered by words in an embedding. The author uses Google news embeddings to illustrate this point. ** In this kernel I expand on this point by exploring the effect of tokenization assumptions on the other three embeddings: GloVe, Paragram, and Wiki News.** 
# 
# Our base tokenization method simply defines words as sequences of letters, sequences of letters with an apostrophe somewhere in the sequence, or a puctuation mark. To reduce the amount of preprocessing, nothing was removed during the initial tokenization. More preprocessing is added gradually to improve coverage of the training vocabulary.

# # Setting Up

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import re

# Use tqdm to show progress of an pandas function we use
tqdm.pandas()

from gensim.models import KeyedVectors as kv
from gensim.scripts.glove2word2vec import glove2word2vec

embedding_path_dict= {'googlenews':{
                            'path':'../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin',
                            'format':'word2vec',
                            'binary': True
                      },
                      'glove':{
                            'path':'../input/embeddings/glove.840B.300d/glove.840B.300d.txt',
                            'format': 'glove',
                            'binary': ''
                      },
                      'glove_word2vec':{
                            'path':'../input/glove.840B.300d.txt.word2vec',
                            'format': 'word2vec',
                            'binary': False
                      },
                      'wiki':{
                            'path': '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',
                            'format': 'word2vec',
                            'binary': False
                      },
                      'paragram':{
                            'path': '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt',
                            'format': '',
                            'binary': False
                      }
                    }


# ## Get Training and Test Data

# In[ ]:


train=pd.read_csv("../input/train.csv")
test= pd.read_csv("../input/test.csv")
print("Train shape:", train.shape)
print("Test shape:", test.shape)


# In[ ]:


train.head()


# In[ ]:


train = train.loc[train.question_text.str.len()>100]


# In[ ]:


len(train.loc[train['target']==0])


# In[ ]:


num_pos= len(train.loc[train['target']==1])
print(num_pos)


# In[ ]:


len(train['target'])


# ## Functions
# Here I define functions that will be used repeatedly in this notebook. More functions will be added as we learn about the embeddings

# ### Functions: Embedding-Related Functions

# In[ ]:


# Get word embeddings
def get_embeddings(embedding_path_dict, emb_name):
    """
    :params embedding_path_dict: a dictionary containing the path, binary flag, and format of the desired embedding,
            emb_name: the name of the embedding to retrieve
    :return embedding index: a dictionary containing the embeddings"""
    
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    
    embeddings_index = {}
    if (emb_name == 'googlenews'):
        emb_path = embedding_path_dict[emb_name]['path']
        bin_flag = embedding_path_dict[emb_name]['binary']
        embeddings_index = kv.load_word2vec_format(emb_path, binary=bin_flag).vectors
    elif (emb_name in ['glove', 'wiki']):
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path_dict[emb_name]['path']) if len(o)>100)    
    elif (emb_name == 'paragram'):
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path_dict[emb_name]['path'], encoding="utf8", errors='ignore'))
    return embeddings_index

#Convert GLoVe format into word2vec format
def glove_to_word2vec(embedding_path_dict, emb_name='glove', output_emb='glove_word2vec'):
    """
    Convert the GLOVE embedding format to a word2vec format
    :params embedding_path_dict: a dictionary containing the path, binary flag, and format of the desired embedding,
            glove_path: the name of the GLOVE embedding
            output_file_path: the name of the converted embedding in embedding_path_dict. 
    :return output from the glove2word2vec script
    """
    glove_input_file = embedding_path_dict[emb_name]['path']
    word2vec_output_file = embedding_path_dict[output_emb]['path']                
    return glove2word2vec(glove_input_file, word2vec_output_file)


# In[ ]:


# Get stats of a given embeddings index
def get_emb_stats(embeddings_index):

    # Put all embeddings in a numpy matrix
    all_embs= np.stack(embeddings_index.values())

    # Get embedding stats
    emb_mean = all_embs.mean()
    emb_std = all_embs.std()
    
    num_embs = all_embs.shape[0]
    
    emb_size = all_embs.shape[1]
    
    return emb_mean,emb_std, num_embs, emb_size 


# ### Functions:  Tokenize Training Sentences

# In[ ]:


# Converts sentences into lists of tokens
# We use this function to allow more control over what constitutes a word
# It also allows us to explore ways to cover more the pre-defined word embeddings.

def tokenize(sentences, restrict_to_len=-1):
    """
    :params sentence_list: list of strings
    :returns tok_sentences: list of list of tokens
    """
    
    if restrict_to_len>0:
        tok_sentences = [re.findall(r"[\w]+[']*[\w]+|[\w]+|[.,!?;]", x )                          for x in sentences if len(x)>restrict_to_len]
    else:
       tok_sentences = [re.findall(r"[\w]+[']*[\w]+|[\w]+|[.,!?;]", x )                          for x in sentences] 
    return tok_sentences

#Build the vocabulary given a list of sentence words
def get_vocab(sentences, verbose= True):
    """
    :param sentences: a list of list of words
    :return: a dictionary of words and their frequency 
    """
    vocab={}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] +=1
            except KeyError:
                vocab[word] = 1
    return vocab

def repl(m):
    return '#' * len(m.group())

#Convert numerals to a # sign
def convert_num_to_pound(sentences):
    return sentences.progress_apply(lambda x: re.sub("[1-9][\d]+", repl, x)).values


# ### Functions: Compare Training and Embedding Vocabulary

# In[ ]:



#find words in common between a given embedding and our vocabulary
def compare_vocab_and_embeddings(vocab, embeddings_index):
    """
    :params vocab: our corpus vocabulary (a dictionary of word frquencies)
            embeddings_index: a genim object containing loaded embeddings.
    :returns in_common: words in common,
             in_common_freq: total frequency in the corpus vocabulary of 
                             all words in common
             oov: out of vocabulary words
             oov_frequency: total frequency in vocab of oov words
    """
    in_common={}
    oov=[]
    in_common=[]
    in_common_freq = 0
    oov_freq = 0
    
    # Compose the vocabulary given the sentence tokens
    vocab = get_vocab(sentences)

    for word in tqdm(vocab):
        if word in embeddings_index:
            in_common.append(word)
            in_common_freq += vocab[word]
        else: 
            oov.append(word)
            oov_freq += vocab[word]
    
    print('Found embeddings for {:.2%} of vocab'.format(len(in_common) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(in_common_freq / (in_common_freq + oov_freq)))

    return sorted(in_common)[::-1], sorted(oov)[::-1], in_common_freq, oov_freq, vocab

# print the list of out-of-vocabulary words sorted by their frequency in teh training text
def show_oov_words(oov, vocab,  num_to_show=15):
    # Sort oov words by their frequency in the text
    sorted_oov= sorted(oov, key =lambda x: vocab[x], reverse=True )

    # Show oov words and their frequencies
    if (len(sorted_oov)>0):
        print("oov words:")
        for word in sorted_oov[:num_to_show]:
            print("%s\t%s"%(word, vocab[word]))
    else:
        print("No words were out of vocabulary.")
        
    return len(sorted_oov);


# # Exploring Embeddings
# 
# We are now ready to explore each embedding and tokenization techniques that maximize its coverage of the training vocabulary

# ## GloVe

# ### Choose Embedding

# In[ ]:


embedding_name = 'glove'
embeddings_index= get_embeddings(embedding_path_dict, embedding_name)
import gc; gc.collect()


# In[ ]:


# Get embedding stats
emb_mean,emb_std, num_embs, emb_size = get_emb_stats(embeddings_index)
print("mean: %5.5f\nstd: %5.5f\nnumber of embeddings: %d\nembedding vector size:%d"       %(emb_mean,emb_std, num_embs, emb_size))


# ### Tokenize Training Text

# In[ ]:


question_text = train["question_text"]

# Get a list of token for each question text
# restrict_to_len is approximately the mean sentence length+ 0.5std
sentences = tokenize(question_text)


# In[ ]:


# Does our tokenization method produce a good match with 
# the words in the selected embedding type?

# Get words in common and out of vocabulary words
in_common, oov, in_common_freq, oov_freq, vocab = compare_vocab_and_embeddings(sentences, embeddings_index)

# Print a sorted list of the oov words
show_oov_words(oov, vocab)


# good coverage but the top missing words all have contractions. We deal with those next...

# In[ ]:


contr_dict={"I\'m": "I am",
            "won\'t": "will not",
            "\'s" : "", 
            "\'ll":"will",
            "\'ve":"have",
            "n\'t":"not",
            "\'re": "are",
            "\'d": "would",
            "y'all": "all of you"}

def replace_contractions(sentences, contr_dict=contr_dict):
    res_sentences=[]
    for sent in sentences:
        for contr in contr_dict:
            sent = sent.replace(contr, " "+contr_dict[contr])
        res_sentences.append(sent)
    return res_sentences


# In[ ]:


# start by replacing contractions
sentences = replace_contractions(question_text)

# Get a list of token for each question text
# restrict_to_len is approximately the mean sentence length+ 0.5std
sentences = tokenize(sentences)


# In[ ]:


# Does our tokenization method produce a good match with 
# the words in the selected embedding type?

# Get words in common and out of vocabulary words
in_common, oov, in_common_freq, oov_freq, vocab = compare_vocab_and_embeddings(sentences, embeddings_index)

# Print a sorted list of the oov words
show_oov_words(oov, vocab)


# Much better... Quorans is the top word missed now...I wonder if Quora or quora is in the embeddings_index vocabulary..

# In[ ]:


print("Is 'Quora' in the wiki embeddings index?",'Quora' in embeddings_index)
print("Is 'quora' in the wiki embeddings index?",'quora' in embeddings_index)


# We can replace Quorans with Quora contributors...

# In[ ]:


w_quoran_contr_dict={"I\'m": "I am",
                    "won\'t": "will not",
                    "\'s" : "", 
                    "\'ll":"will",
                    "\'ve":"have",
                    "n\'t":"not",
                    "\'re": "are",
                    "\'d": "would",
                    "y'all": "all of you",
                    "Quoran": "Quora contributor",
                    "quoran": "quora contributor"
                    }


# In[ ]:


# replace contractions using a contr dict containing replacement for Quoran
sentences = replace_contractions(question_text, contr_dict = w_quoran_contr_dict)

# Get a list of token for each question text
# restrict_to_len is approximately the mean sentence length+ 0.5std
sentences = tokenize(sentences)


# In[ ]:


# Does our tokenization method produce a good match with 
# the words in the selected embedding type?

# Get words in common and out of vocabulary words
in_common, oov, in_common_freq, oov_freq, vocab = compare_vocab_and_embeddings(sentences, embeddings_index)

# Print a sorted list of the oov words
show_oov_words(oov, vocab)


# No change. Probably because it is only a single word whose frequency is small compared to the size of the vocabulary.
# 
# what about heights? There are some tokens that mention height such as 5'2 and 6'4.. can we convert those to a longer, more compatible, format? 
# 
# (Note: Different height values  show up frequently further down the list. To see it in your own notebook use show_oov_words(oov, vocab, num_to_show=100). I am only showing a small list of oov words here to make the notebook more readable) 

# First, does the embeddings index contain digits? (Google News replaces numbers > 9 with # signs)

# In[ ]:


print("0 in embedding index?", ('0' in embeddings_index))
print("Other digits?", ('1' in embeddings_index) and ('2' in embeddings_index))


# Good...So let us replace all heights of the form \d\'\d such as 5'4 with the "5 foot 4"..

# In[ ]:


import re

def convert_height(sentences):
    res_sentences = []
    for sent in sentences:
        res_sent = re.sub( "(\d+)\'(\d+)", "\1 foot \2", sent)
        res_sentences.append(res_sent)
    return res_sentences


# In[ ]:


# start by converting heights such as 5'4 to longer format 5 foot 4
sentences = convert_height(question_text)

# replace contractions
sentences = replace_contractions(sentences)

# Get a list of token for each question text
# restrict_to_len is approximately the mean sentence length+ 0.5std
sentences = tokenize(sentences)


# In[ ]:


# Does our tokenization method produce a good match with 
# the words in the selected embedding type?

# Get words in common and out of vocabulary words
in_common, oov, in_common_freq, oov_freq, vocab = compare_vocab_and_embeddings(sentences, embeddings_index)

# Print a sorted list of the oov words
show_oov_words(oov, vocab)


# Very slight improvement. But overall, we managed to improve our coverage from 83.54% to 86.43%. For The GloVe embedding the main issue affecting our results was contractions. Replacing height with a longer format helped very slightly.

# ## Paragram

# ### Choose Embedding

# In[ ]:


embedding_name = 'paragram'
embeddings_index= get_embeddings(embedding_path_dict, embedding_name)
import gc; gc.collect()


# In[ ]:


# Get embedding stats
emb_mean,emb_std, num_embs, emb_size = get_emb_stats(embeddings_index)
print("mean: %5.5f\nstd: %5.5f\nnumber of embeddings: %d\nembedding vector size:%d"       %(emb_mean,emb_std, num_embs, emb_size))


# ### Tokenize Training Text

# In[ ]:


question_text = train["question_text"]

# Get a list of token for each question text
# restrict_to_len is approximately the mean sentence length+ 0.5std
sentences = tokenize(question_text)


# In[ ]:


# Does our tokenization method produce a good match with 
# the words in the selected embedding type?

# Get words in common and out of vocabulary words
in_common, oov, in_common_freq, oov_freq, vocab = compare_vocab_and_embeddings(sentences, embeddings_index)

# Print a sorted list of the oov words
show_oov_words(oov, vocab)


# Interesting! Very few vocabulary words are in common with the paragrams vocabulary. The top missing ones all have capital letters so let us convert to lower case

# In[ ]:


def convert_to_lower(sentences):
    res_sentences = []
    for sent in sentences:
        lower_sent = sent.lower()
        res_sentences.append(lower_sent)
    return res_sentences


# In[ ]:


# convert capitals to lowercase
sentences = convert_to_lower(question_text)

# Get a list of token for each question text
# restrict_to_len is approximately the mean sentence length+ 0.5std
sentences = tokenize(sentences)


# In[ ]:


# Does our tokenization method produce a good match with 
# the words in the selected embedding type?

# Get words in common and out of vocabulary words
in_common, oov, in_common_freq, oov_freq, vocab = compare_vocab_and_embeddings(sentences, embeddings_index)

# Print a sorted list of the oov words
show_oov_words(oov, vocab)


# Much better!  Lets deal with the contractions now...

# In[ ]:


# start by converting capitals to lowercase
sentences = convert_to_lower(question_text)

# replace contractions
sentences = replace_contractions(sentences)

# Get a list of token for each question text
# restrict_to_len is approximately the mean sentence length+ 0.5std
sentences = tokenize(sentences)


# In[ ]:


# Does our tokenization method produce a good match with 
# the words in the selected embedding type?

# Get words in common and out of vocabulary words
in_common, oov, in_common_freq, oov_freq, vocab = compare_vocab_and_embeddings(sentences, embeddings_index)

# Print a sorted list of the oov words
show_oov_words(oov, vocab)


# There are also frequent mentions of height..I wonder how that will affect the results...

# In[ ]:


# start by replacing heights such as 5'4 to a longer format (5 foot 4)
sentences = convert_height(question_text)

# convert capitals to lowercase
sentences = convert_to_lower(sentences)

# replace contractions
sentences = replace_contractions(sentences)

# Get a list of token for each question text
# restrict_to_len is approximately the mean sentence length+ 0.5std
sentences = tokenize(sentences)


# In[ ]:


# Does our tokenization method produce a good match with 
# the words in the selected embedding type?

# Get words in common and out of vocabulary words
in_common, oov, in_common_freq, oov_freq, vocab = compare_vocab_and_embeddings(sentences, embeddings_index)

# Print a sorted list of the oov words
show_oov_words(oov, vocab)


# Slightly better. 
# 
# We managed to improve the vocabulary coverage from a mere 50.03% to 85.97% by replacing capitals with lower case letters and expanding contractions. Replacing heights with a longer form improved results very slightly.

# ## Wiki

# ### Choose Embedding

# In[ ]:


embedding_name = 'wiki'
embeddings_index= get_embeddings(embedding_path_dict, embedding_name)
import gc; gc.collect()


# In[ ]:


# Get embedding stats
emb_mean,emb_std, num_embs, emb_size = get_emb_stats(embeddings_index)
print("mean: %5.5f\nstd: %5.5f\nnumber of embeddings: %d\nembedding vector size:%d"       %(emb_mean,emb_std, num_embs, emb_size))


# ### Tokenize Training Text

# In[ ]:


question_text = train["question_text"]

# Get a list of token for each question text
# restrict_to_len is approximately the mean sentence length+ 0.5std
sentences = tokenize(question_text)


# In[ ]:


# Does our tokenization method produce a good match with 
# the words in the selected embedding type?

# Get words in common and out of vocabulary words
in_common, oov, in_common_freq, oov_freq, vocab = compare_vocab_and_embeddings(sentences, embeddings_index)

# Print a sorted list of the oov words
show_oov_words(oov, vocab)


# Contractions seem to be the main issue here..

# In[ ]:


# start by replacing contractions
sentences = replace_contractions(question_text)

# Get a list of token for each question text
# restrict_to_len is approximately the mean sentence length+ 0.5std
sentences = tokenize(sentences)


# In[ ]:


# Does our tokenization method produce a good match with 
# the words in the selected embedding type?

# Get words in common and out of vocabulary words
in_common, oov, in_common_freq, oov_freq, vocab = compare_vocab_and_embeddings(sentences, embeddings_index)

# Print a sorted list of the oov words
show_oov_words(oov, vocab)


# Better! Quorans seems to be a repeatedly missed word in this embedding as well... 

# In[ ]:


print("Is 'Quora' in the wiki embeddings index?",'Quora' in embeddings_index)
print("Is 'quora' in the wiki embeddings index?",'quora' in embeddings_index)


# We can replace Quorans with Quora contributors...

# In[ ]:


# start by replacing contractions using the contractions dict containing replacements for Quoran
sentences = replace_contractions(question_text, contr_dict = w_quoran_contr_dict)

# Get a list of token for each question text
# restrict_to_len is approximately the mean sentence length+ 0.5std
sentences = tokenize(sentences)


# In[ ]:


# Does our tokenization method produce a good match with 
# the words in the selected embedding type?

# Get words in common and out of vocabulary words
in_common, oov, in_common_freq, oov_freq, vocab = compare_vocab_and_embeddings(sentences, embeddings_index)

# Print a sorted list of the oov words
show_oov_words(oov, vocab)


# Fixed that but with no effect on overall result since the qord frequency is small compared to the total vocabulary size. 
# 
# Lets now look at the effect of switching to heights.. First, does this embedding contain numbers as tokens?

# In[ ]:


print("0 in embedding index?", ('0' in embeddings_index))
print("Other digits?", ('1' in embeddings_index) and ('2' in embeddings_index))


# Lets convert heights to a longer format...

# In[ ]:


# start by converting height to longer form
sentences = convert_height(question_text)

# replace contractions
sentences = replace_contractions(sentences,  contr_dict = w_quoran_contr_dict)

# Get a list of token for each question text
# restrict_to_len is approximately the mean sentence length+ 0.5std
sentences = tokenize(sentences)


# In[ ]:


# Does our tokenization method produce a good match with 
# the words in the selected embedding type?

# Get words in common and out of vocabulary words
in_common, oov, in_common_freq, oov_freq, vocab = compare_vocab_and_embeddings(sentences, embeddings_index)

# Print a sorted list of the oov words
show_oov_words(oov, vocab)


# Slightly better!
# 
# Overall coverage for the Wiki embeddings improved from 79.36% to 82.09%. The main issue in this embedding was dealing with contractions. As with the other two embeddings, replacing height with a longer format had a minor effect on the overall result.

# ## Conclusion

# In this notebook We looked at three different embeddings: glove.840B.300d (GloVe), paragram_300_sl999 (paragram), and wiki-news-300d-1M (wiki) and how to best tokenize our Quora training text so that we maximize the percentage of words represented by the embeddings index.  In general, capetalization, contractions, and, to a lesser extent, height measurements, had the most impact on how much of the training vocabulary was covered by an embedding.
# 
# Our base tokenization method simply defined words as sequences of letters, sequences of letters with an apostrophe somewhere in the sequence, or a puctuation mark. To reduce the amount of preprocessing nothing was removed in this method. More preprocessing was done based on our observations of which words were missed by the embedding. Coverage of an embedding was improved by about 3 percentage point for the GloVe and Wiki embeddings. The most significant improvement was for the paragram embedding which improved  36 percentage points from 50.03% to 85.97%.
# 
# Awaerness of the intricacies of each embedding should help improve the accuracy of our Quora Insincere Questions Classification networks. 

# ### Acknowledgments

# * [https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings](http://https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings)
# * [https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings](http://https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings)
