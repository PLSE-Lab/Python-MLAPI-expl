#!/usr/bin/env python
# coding: utf-8

# # **Importing Library**

# In[1]:


import numpy as np # linear algebra
import os
from nltk import *
from nltk.tokenize import word_tokenize,wordpunct_tokenize
from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.models import load_model
from nltk.corpus import stopwords
import h5py
filepath = '../input/'
filename = os.listdir(filepath)
print("Using these files : ",filename)
# Any results you write to the current directory are saved as output.


# # **Reading Datasets**

# In[3]:


blog = ''
news = ''
blog += open(filepath+filename[1],'r',encoding='utf8').read()
news += open(filepath+filename[2],'r',encoding='utf8').read()


# # **Text 2 Tokens**

# In[ ]:


blog_tokens = wordpunct_tokenize(blog)
news_tokens = wordpunct_tokenize(news)


# In[ ]:


blog_tokens[:2],news_tokens[:2]


# # **Remove single character word and Removing punctuation**

# In[ ]:


# List Tokenize word
tokenize_word_blog = []
tokenize_word_news = []
for word in blog_tokens:
    if len(word) >= 2 and word.isalpha() and word.lower()!='the':
        word = word.replace('?','')
        word = word.replace('.','')
        word = word.replace('!','')
        word = word.replace(';','')
        word = word.replace(':','')
        tokenize_word_blog.append(word.lower())

for word in news_tokens:
    if len(word) >= 2 and word.isalpha() and word.lower()!='the':
        word = word.replace('?','')
        word = word.replace('.','')
        word = word.replace('!','')
        word = word.replace(';','')
        word = word.replace(':','')
        tokenize_word_blog.append(word.lower())


# # **Taking 50k words from news datasets & 50k words from blog datasets**

# In[ ]:


final_tokenize_word = []
final_tokenize_word += tokenize_word_blog[:40000]
final_tokenize_word += tokenize_word_news[:40000]


# In[ ]:


final_tokenize_words = []
for i in final_tokenize_word:
    if i=='ve':
        final_tokenize_words.append('have')
    elif i=='re':
        final_tokenize_words.append('are')
    elif i=='ll':
        final_tokenize_words.append('will')
    else:
        final_tokenize_words.append(i)


# In[ ]:


# pickle.dump(final_tokenize_words,open('tokenized_words.pkl','wb'))


# 
# 
# # **Converting Text to sequences Using Tokenizer from Keras**
# ### Tokenizer will convert tokenized words to integer sequence
# ### Tokenizer will also store value of each word into Dictionary like {'where' : 1254, 'what' : 653}

# In[ ]:


tokenizer = Tokenizer() # creating object of Tokenizer()
tokenizer.fit_on_texts([final_tokenize_words])
encoded = tokenizer.texts_to_sequences([final_tokenize_words])[0]


# In[ ]:


encoded[:5]


# # **Determine the vocabulary size**
# ###  Determine number of Unique words

# In[ ]:


vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)


# # **Creating word sequence**
# # **Creating Bi-grams of words**

# In[ ]:


sequences = list()
for i in range(1, len(encoded)):
    sequences.append(encoded[i-1:i+1])
print('Total Sequences: %d' % len(sequences))


# In[ ]:


sequences[:5]


# # **Split sequence to Input as X & Output as Y**

# In[ ]:


sequences = np.array(sequences) # Converting list to numpy array
X, Y = sequences[:,0],sequences[:,1]


# # **Converting Y to Categorical for Calculating loss = 'categorical_crossentropy'**

# In[ ]:


Y = to_categorical(Y,num_classes=vocab_size)


# # **Defining Sequential model**
# ## Adding Embedding Layer for LookUp Table 
# ## Adding LSTM Layer of 200 nodes
# ## Adding Dense Layer ( Using 'softmax' Activation func )

# In[ ]:


model = Sequential()
model.add(Embedding(vocab_size, 400, input_length=1))
model.add(LSTM(400,return_sequences=True))
model.add(LSTM(400))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())


# # **Compile network**

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])


# # **Fitting model on X and Y Data**

# In[ ]:


model.fit(X, Y, epochs=200,batch_size=48, verbose=2)


# In[ ]:


# serialize model to HDF5
model.save("new_model.h5")
print("Saved model to disk")


# # **Generate a sequence from the model**

# In[ ]:


def generate_seq(word):
    in_text, result = word, word
    # generate a fixed number of words
    for _ in range(3):
        # encode the text as integer
        encode = tokenizer.texts_to_sequences([in_text])[0]
        encode = np.array(encode)
        # predict a word in the vocabulary
        yhat = model.predict_classes(encode, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text, result = out_word, result + ' ' + out_word
    print(result)


# In[ ]:


generate_seq('how')


# In[ ]:


generate_seq('so')


# In[ ]:


generate_seq('you')


# In[ ]:


generate_seq('what')


# In[ ]:


generate_seq('when')


# In[ ]:





# In[ ]:




