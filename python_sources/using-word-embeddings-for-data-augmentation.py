#!/usr/bin/env python
# coding: utf-8

#  # Using Word Embeddings for Data Augmentation
# 
# ###  In this kernel, I'm going to show you a way to do data augmentation for texts, when you have word embeddings.
# 
# I will focus on augmenting texts labelled as 1, as this class is under-represented. Oversampling can help improve perfomances.
# 
# 
# #### Feel free to give any feedback, it is always appreciated.
# 
# ##### References :
# 
# Inspired by this kernel :
# > https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings#  by SRK
# 
# Continuation of :
# > https://www.kaggle.com/theoviel/dealing-with-class-imbalance-with-smote

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

np.random.seed(100)


# ## Step 1 : Loading Word Embeddings

# In[ ]:


def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

def load_embedding(file):
    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
    return embeddings_index


# In[ ]:


def make_embedding_matrix(embedding, tokenizer, len_voc):
    all_embs = np.stack(embedding.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    word_index = tokenizer.word_index
    embedding_matrix = np.random.normal(emb_mean, emb_std, (len_voc, embed_size))
    
    for word, i in word_index.items():
        if i >= len_voc:
            continue
        embedding_vector = embedding.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


# In[ ]:


glove = load_embedding('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')


# I'm using GloVe, because that's the embedding I got the best results with inbefore, but paragram is quite good as well.
# 
# ## Step 2 : Loading Data

# In[ ]:


df = pd.read_csv("../input/train.csv")
print("Number of texts: ", df.shape[0])


# In[ ]:


df.head()


# ### Class imbalance

# In[ ]:


plt.figure(figsize = (10, 8))
sns.countplot(df['target'])
plt.show()


# In[ ]:


print("Class repartition : ", (Counter(df['target'])))


# There is way more 0s than 1s in our dataset. As mentionned, we could use data augmentation to balance classes. Therefore the prediction task will be easier.

# ## Step 3: Tokenizing
# 
# I am using Keras' Tokenizer to apply some text processing and to limit the size of the vocabulary

# In[ ]:


len_voc = 100000


# #### Tokenizing

# In[ ]:


def make_tokenizer(texts, len_voc):
    from keras.preprocessing.text import Tokenizer
    t = Tokenizer(num_words=len_voc)
    t.fit_on_texts(texts)
    return t


# In[ ]:


tokenizer = make_tokenizer(df['question_text'], len_voc)


# In[ ]:


X = tokenizer.texts_to_sequences(df['question_text'])


# I also apply padding, mostly to store X as an array.

# In[ ]:


from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(X, 70)


# In[ ]:


y = df['target'].values


# For visualization, I'm gonna need to see which index corresponds to which word

# In[ ]:


index_word = {0: ''}
for word in tokenizer.word_index.keys():
    index_word[tokenizer.word_index[word]] = word


# #### Embedding Matrix

# In[ ]:


embed_mat = make_embedding_matrix(glove, tokenizer, len_voc)


# ## Step 3 : Making a Synonym Dictionary
# 
# Word vectors are made in a way that similar words have similar representation. Therefore we can use the $k$-nearest neighbours to get $k$ synonyms.
# 
# As the process takes a bit of time, I chose to compute 5 synonyms for the 20000 most frequent words.

# In[ ]:


from sklearn.neighbors import NearestNeighbors

synonyms_number = 5
word_number = 20000


# In[ ]:


nn = NearestNeighbors(n_neighbors=synonyms_number+1).fit(embed_mat) 


# #### We create Synonyms for the most frequent words

# In[ ]:


neighbours_mat = nn.kneighbors(embed_mat[1:word_number])[1]


# In[ ]:


synonyms = {x[0]: x[1:] for x in neighbours_mat}


# ### Checking our synonyms

# In[ ]:


for x in np.random.randint(1, word_number, 10):
    print(f"{index_word[x]} : {[index_word[synonyms[x][i]] for i in range(synonyms_number-1)]}")


# In[ ]:


index = np.random.randint(1, word_number, 9)
plt.figure(figsize=(20,10))

for k in range(len(index)):
    plt.subplot(3, 3, k+1)
    
    x = index[k]
    text = ' '.join([index_word[x]] + [index_word[synonyms[x][i]] for i in range(synonyms_number-1)]) 
    wordcloud = WordCloud(stopwords=[]).generate((text))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")


# #### Looks pretty good ! 

# ## Step 4 - Data Augmentation / Oversampling 
# 
# #### We work on 1 labelled texts. We apply the following algorithm to modify a sentence :
# 
# For each word in the sentence :
# * Keep it with probability $p$  (or if we don't have synonyms for it)
# * Randomly swap it with one of its synonyms with probability $1-p$

# In[ ]:


X_pos = X[y==1]


# In[ ]:


def modify_sentence(sentence, synonyms, p=0.5):
    for i in range(len(sentence)):
        if np.random.random() > p:
            try:
                syns = synonyms[sentence[i]]
                sentence[i] = np.random.choice(syns)
            except KeyError:
                pass
    return sentence


# ### Let us preview our function

# In[ ]:


indexes = np.random.randint(0, X_pos.shape[0], 10)


# In[ ]:


for x in X_pos[indexes]:
    sample =  np.trim_zeros(x)
    sentence = ' '.join([index_word[x] for x in sample])
    print(sentence)

    modified = modify_sentence(sample, synonyms)
    sentence_m = ' '.join([index_word[x] for x in modified])
    print(sentence_m)
    
    print(' ')


# Looks pretty good, we now generate some texts

# In[ ]:


n_texts = 30000


# In[ ]:


indexes = np.random.randint(0, X_pos.shape[0], n_texts)


# In[ ]:


X_gen = np.array([modify_sentence(x, synonyms) for x in X_pos[indexes]])
y_gen = np.ones(n_texts)


# 
#  
#  #### The next work to do is to find a good value for $p$, and a correct number of samples to generate, and then feed it into a network.
#  
#   ## Thanks for reading, hope it can be helpful to anyone !
# 
#  
