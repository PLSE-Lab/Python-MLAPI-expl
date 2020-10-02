#!/usr/bin/env python
# coding: utf-8

# ### WARNING: Contains Obscene Words

# # Loading and Visualization of Data
# We first load the data using pandas and load it into data frame.

# In[ ]:


# importing the dependencies
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm # progress bar
import copy # perform deep copyong rather than referencing in python
import multiprocessing # for threading of word2vec model process

# importing classes helpfull for text processing
import nltk # general NLP
import re # regular expressions
import gensim.models.word2vec as w2v # word2vec model

import matplotlib.pyplot as plt # data visualization
get_ipython().run_line_magic('matplotlib', 'inline')

# Any results you write to the current directory are saved as output.


# In[ ]:


data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')


# In[ ]:


data_train.head(10)


# In[ ]:


data_test.head(10)


# In[ ]:


col_names = data_train.columns.values[2:]
col_names = col_names.tolist()
col_names.append('None')
x = [sum(data_train[y]) for y in data_train.columns.values[2:]]
x.append(len(data_train) - sum(x))


# ### Occurrence Plot
# We see how often the different catagories occur, thus giving us insight on how to pick up the data for training. As we can see that None of the catagories occur the most and thus we will sample ~10,000 sentences from it.

# In[ ]:


plt.figure(figsize = (10, 10))
plt.bar(np.arange(len(x)),x)
plt.xticks(np.arange(len(x)), col_names)
plt.xlabel('Catagories')
plt.ylabel('Occurrence')


# ## Making Vocabulary and Text Conversion
# We now need to make vocabulary and convert the text into usable information vectors.
# 
# For cleaning of sentences we first convert the sentences into a string that has all the sentences merged together. Then we convert the huge string into a sentence with each tokenized. After that we pass it through a regex filter which then returns each sentence converted into a list of tokens, and we append them to make a huge list of tokenized sentences.

# In[ ]:


train_sentences = data_train['comment_text'].values.tolist()
test_sentences = data_test['comment_text'].values.tolist()
# making a list of total sentences
total_ = copy.deepcopy(train_sentences)
total_.extend(test_sentences)
print('[*]Training Sentences:', len(train_sentences))
print('[*]Test Sentences:', len(test_sentences))
print('[*]Total Sentences:', len(total_))

# converting the text to lower
for i in tqdm(range(len(total_))):
    total_[i] = str(total_[i]).lower()


# The code in comments below wa previously used but was later discarded for poor performance, you can still go through it for learning more about the methods

# In[ ]:


'''
Won't be performing by this method rather will be using the crude way to do it
#initialize rawunicode, we'll add all text to this one big string
corpus_raw = u""
#for each sentence, read it, convert in utf 8 format, add it to the raw corpus
for i in tqdm(range(len(total_))):
    corpus_raw += str(total_[i])
print('[*]Corpus is now', len(corpus_raw), 'characters long')

# converting everything to small letter removing all the non caps
corpus_lower = corpus_raw.lower()

# we do no need a seperate tokenizer, sentence_to_wordlist function does it for us
# tokenization process will result in words
def tokenizer(sentences):
    temp = []
    for i in tqdm(range(len(sentences))):
        temp.append(sentences.split())
    return temp

# NLTKs tokenizer was giving me a lot of difficulties, so dropping it for now
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
'''


# In[ ]:


# convert into list of words remove unecessary characters, split into words,
# no hyphens and other special characters, split into words
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z0-9]"," ", raw)
    words = clean.split()
    return words


# ***Trying something else***
# 
# What if we take only the words that are lower and convert capitalised words to lower, how many tokens to we get then. This gives us ~1000 less tokens compared to when we were also using caps, which is not a difference of lot but will still help us.

# In[ ]:


# tokenising the lowered corpus
clean_lower = []
for i in tqdm(range(len(total_))):
    clean_lower.append(sentence_to_wordlist(total_[i]))


# Thus we see we have 322849 sentences in total, now we see the lengths of each of them, to decide our designing of the model. So some of the features of the data are as follows:

# In[ ]:


# tokens and its count
total_tokens_l = []
for s in clean_lower:
    total_tokens_l.extend(s)
unk_tokens_l = list(set(total_tokens_l))
print("[!]Total number of tokens:", len(total_tokens_l))
print("[!]Total number of unique tokens:", len(unk_tokens_l))


# In[ ]:


# while we convert each sentence into it's feature matrix, we need to have a consistency
# of size, else we will not be able to train the model efficiently. For that we need the
# length of largest sentence, and that of the smallest
maxlen = max([len(s) for s in clean_lower])
minlen = min([len(s) for s in clean_lower])
print(maxlen)
print(minlen)


# In[ ]:


# finding index where length is zero
index = [int(i) for i,s in enumerate(clean_lower) if len(s) == 0]
print("[*]No. of entries with 0 length:", len(index))


# In[ ]:


# as we can see that all those sentences exist in test data set
print('[*]Minimum index with length 0:',min(index))
print('[*]Length of training dataset:', len(train_sentences))

# so reducing the values of index by length of train_sentences
index_test = [i-len(train_sentences) for i in index]
# looking at those sentences with 0 length
# print(len(train_sentences) < index[0])
print(test_sentences[index_test[0]])
print(test_sentences[index_test[12]])
print(test_sentences[index_test[34]])


# In[ ]:


# we remove these indexes and in submission classify them as 0.5 for all catagories
clean_ = [c for i,c in enumerate(clean_lower) if i not in index]


# In[ ]:


print(clean_[10])


# ## Word-to-Vector
# For converting them into features we train a word2vec model which converts each word into it's corresponding vector. Word2Vec models are extremely efficient in finding temporal relations as they themselves are shallow neural networks.
# 
# We train the word2vec model on the entire corpus so that it learns the similarities in the text and can give us vectors for all the words, not just those that occur in training dataset.
# 
# You can perform the following operations and learn about word2vec model.

# In[ ]:


# hyper parameters of the word2vec model
num_features = 200 # dimensions of each word embedding
min_word_count = 1 # this is not advisable but since we need to extract
# feature vector for each word we need to do this
num_workers = multiprocessing.cpu_count() # number of threads running in parallel
context_size = 7 # context window length
downsampling = 1e-3 # downsampling for very frequent words
seed = 1 # seed for random number generator to make results reproducible


# In[ ]:


word2vec_ = thrones2vec = w2v.Word2Vec(
    sg = 1, seed = seed,
    workers = num_workers,
    size = num_features,
    min_count = min_word_count,
    window = context_size,
    sample = downsampling
)


# In[ ]:


# first we need to built the vocab
word2vec_.build_vocab(clean_)


# In[ ]:


# now we need to train the model
word2vec_.train(clean_, total_examples = word2vec_.corpus_count, epochs = word2vec_.iter)


# In[ ]:


word2vec_.wv.most_similar('male')


# In[ ]:


word2vec_.wv.most_similar('gay')


# In[ ]:


word2vec_.wv.most_similar('dick')


# In[ ]:


# how to get vector for each word
vec_ = word2vec_['male']
print('[*]Shape of vec_:', vec_.shape)


# The best practice is to save the model once the training is done, I cannot perform these on the Kaggle kernel, but following is the code on how to save it, following it is the code to load it.
# 

# In[ ]:


'''
 if not os.exists('trained'):
    os.makedirs('trained')
    
w2vector_.save(os.path.join('trained', 'w2vector_.w2v'))

w2vector_ = word2vec.Word2Vec.load(os.path.join('trained', 'w2vector_.w2v'))
'''


# ## Making Feature Matrices
# Now that we have our word2vec model trained we can convert each word into it's vector and make our numerical data.
# 
# We need to pad the sequences to make them of consistent length if we have to use Keras fixed LSTM network. for this we can use "PAD" and add it to the start of all the sequences which are less than maxlen. We also define the vector for PAD which will be the numerical information.
# 
# *As we can see this is a very slow process, might have to play around a bit to optimize the padding*

# In[ ]:


# adding 'PAD' to each sequence
print('[!]Adding \'PAD\' to each sequence...')
for i in tqdm(range(len(clean_))):
    sentence = clean_[i][::-1]
    for _ in range(maxlen - len(sentence)):
        sentence.append('PAD')
    clean_[i] = sentence[::-1]
print()

# defining 'PAD'
PAD = np.zeros(word2vec_['guy'].shape)


# In[ ]:


# first we make the training set
train_features = []
print('[!]Making training features...')
for i in tqdm(range(len(data_training))):
    sentence = clean_[i]
    temp = []
    for token in sentence:
        temp.append(word2vec_[token].tolist())
    train_features.append(temp)

# perform on local machine no need to waste kaggle resources
# train_data = np.array(train_features)
print()


# In[ ]:


# now we make the testing set
test_features = []
print('[!]Making training features...')
for i in tqdm(range(len(data_testing))):
    sentence = clean_[i+len(data_training)]
    temp = []
    for token in sentence:
        temp.append(word2vec_[token].tolist())
    test_features.append(temp)
    
# perform on local machine no need to waste kaggle resources
# test_data = np.array(testing_features)
print()


# In[ ]:


# saving the numpy arrays
print('[!]Saving training data file at:', PATH_SAVE_DATA_TRAIN, ' ...')
np.save(PATH_SAVE_DATA_TRAIN , train_data)

print('[!]Saving testing data file at:', PATH_SAVE_DATA_TEST, ' ...')
np.save(PATH_SAVE_DATA_TEST , test_data)


# ## Making the labels
# Now we have one last job to do, make the labels file and dump it so we can use it later

# In[ ]:




