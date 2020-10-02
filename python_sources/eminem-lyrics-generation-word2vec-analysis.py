#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords


# In[2]:


import plotly
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.plotly as py
import plotly.graph_objs as go


# In[4]:


data = pd.read_csv("../input/songdata.csv")
data.head(5)


# In[4]:


del data['link']


# In[4]:


len(list(data['artist'].unique()))


# ## Let us focus on Eminem 

# In[5]:


eminem = data[(data['artist'] == 'Eminem')]
eminem.head(3)


# In[171]:


rap = list(eminem['text'])
word=[]
for i in range(0,len(rap)):
    kk = rap[i].replace("\n"," ")
    s=kk.split(' ')
    o = [x for x in s if x]
    word.append(o)


# In[98]:


word = [j for i in word for j in i]
word[:5]


# ### Remove stop words and other unnecessary words

# In[99]:


el = ["i'm","get","got"]
stop = set(stopwords.words('english'))
word = [word.lower() for word in word]
words = [i for i in word if i not in stop]
words = [i for i in words if i not in el]


# ### Removing punctuation from words in lyrics

# In[100]:


for i in range(0,len(words)):
    words[i] = re.sub(r'[^\w\s]','',words[i])


# ### Remove the empty elements in the list

# In[92]:


words = [x for x in words if x]


# In[11]:


from collections import Counter
labels, values = zip(*Counter(words).items())


# ### 20 commonly used words in the lyrics along with the number of times they occur

# In[13]:


w = Counter(words)
s = w.most_common(20)


# In[14]:


x , y = zip(*(s))


# In[159]:


data = [go.Bar(x=x,y=y)]
layout = go.Layout(
    title='Words in Eminem Lyrics ',
    xaxis=dict(
        title='Words Used',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Number of times it was used',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
iplot(go.Figure(data=data, layout = layout))


# ### Eminem seems to use the word LIKE a lot in his lyrics. He also used bad words frequently

# In[13]:


unique_words = sorted(set(words))


# In[15]:


len(unique_words)


# # Generating Eminem's Rap Lyric

# In[22]:


from keras.models import Sequential
from keras.layers.noise import GaussianNoise
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# ### Mapping each unique word in the lyric to a number (5204 unique words)

# In[17]:


rap = np.array(rap)


# In[18]:


lyric = (''.join(rap))


# In[21]:


l = set(lyric)


# In[22]:


len(l)


# ### Mapping words to numbers for easier training

# In[18]:


vocab= [k for k in l] 
char_ix={c:i for i,c in enumerate(vocab)}
ix_char={i:c for i,c in enumerate(vocab)}


# In[317]:


ix_char


# In[318]:


maxlen=40
vocab_size=len(vocab)


# ### Character wise Patterns 

# In[314]:


sentences=[]
next_char=[]
for i in range(len(lyric)-maxlen-1):
    sentences.append(lyric[i:i+maxlen])
    next_char.append(lyric[i+maxlen])
sentences


# In[320]:


X=np.zeros((len(sentences),maxlen,vocab_size))
y=np.zeros((len(sentences),vocab_size))
for ix in range(len(sentences)):
    y[ix,char_ix[next_char[ix]]]=1
    for iy in range(maxlen):
        X[ix,iy,char_ix[sentences[ix][iy]]]=1


# In[38]:


from keras.layers import Activation,LSTM,Dense
from keras.optimizers import Adam


# In[327]:


model=Sequential()
model.add(LSTM(128,input_shape=(maxlen,vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy')


# In[328]:


model.fit(X,y,epochs=5,batch_size=128)


# In[329]:


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")


# In[331]:


import random
generated=''
start_index=random.randint(0,len(lyric)-maxlen-1)
sent=lyric[start_index:start_index+maxlen]
generated+=sent
for i in range(1900):
    x_sample=generated[i:i+maxlen]
    x=np.zeros((1,maxlen,vocab_size))
    for j in range(maxlen):
        x[0,j,char_ix[x_sample[j]]]=1
    probs=model.predict(x)
    probs=np.reshape(probs,probs.shape[1])
    ix=np.random.choice(range(vocab_size),p=probs.ravel())
    generated+=ix_char[ix]


# ### Computer generated Character based eminem Lyric (Character based)

# In[334]:


generated.split("\n")


# ### Since this is character based, many words don't make sense....let's check out the word based model

# ## Word based Model

# In[13]:


from unidecode import unidecode
def get_tokenized_lines(df):
    words = []
    
    for index, row in df['text'].iteritems():
        row = str(row).lower()
        for line in row.split('\n'):
            new_words = re.findall(r"\b[a-z']+\b", unidecode(line))
            words = words + new_words
        
    return words


# In[16]:


all_lyric_lines = get_tokenized_lines(eminem)


# In[17]:


SEQ_LENGTH = 50 + 1
sequences = list()

for i in range(SEQ_LENGTH, len(all_lyric_lines)):
    seq = all_lyric_lines[i - SEQ_LENGTH: i]
    sequences.append(seq)

print('Total Sequences: %d' % len(sequences))


# In[18]:


vocab = set(all_lyric_lines)

word_to_index = {w: i for i, w in enumerate(vocab)}
index_to_word = {i: w for w, i in word_to_index.items()}
word_indices = [word_to_index[word] for word in vocab]
vocab_size = len(vocab)

print('vocabulary size: {}'.format(vocab_size))


# In[19]:


def get_tokenized_lines(lines, seq_len):
    tokenized = np.zeros((len(lines), seq_len))
    
    for r, line in enumerate(lines):
        for c, word in enumerate(line):
            tokenized[r, c] = word_to_index[word]

    return tokenized


# In[20]:


tokenized_seq = get_tokenized_lines(sequences, SEQ_LENGTH)


# In[21]:


tokenized_seq[:, -1].shape


# In[23]:


X, y = tokenized_seq[:, :-1], tokenized_seq[:, -1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = len(X[0])

print("X_shape", X.shape)
print("y_shape", y.shape)


# In[65]:


model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=128, epochs=10)


# ### I'm Using the Mockingbird lyric as the seed

# In[66]:


seed_text = "Hailie i know you miss your mom and i know you miss your dad well i'm gone but i'm trying to give you the life that i never had i can see you're sad even when you smile even when you laugh i can see it in your eyes deep inside"


# In[29]:


def texts_to_sequences(texts, word_to_index):
    indices = np.zeros((1, len(texts)), dtype=int)
    
    for i, text in enumerate(texts):
        indices[:, i] = word_to_index[text]
        
    return indices


# In[30]:


def my_pad_sequences(seq, maxlen):
    start = seq.shape[1] - maxlen
    
    return seq[:, start: start + maxlen]


# In[67]:


def generate_seq(model, word_to_index, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text

    for _ in range(n_words):
        encoded = texts_to_sequences(in_text.split()[1:], word_to_index)
        encoded = my_pad_sequences(encoded, maxlen=seq_length)
        
        yhat = model.predict_classes(encoded, verbose=0)
        out_word = ''
    
        for word, index in word_to_index.items():
            if index == yhat:
                out_word = word
                break
        
        in_text += ' ' + out_word
        result.append(out_word)
        
    return ' '.join(result)


# In[72]:


generated = generate_seq(model, word_to_index, seq_length, seed_text, 50)
print(generated)


# ### The most probable words seem to repeat in word level embedding ( Needs more epochs or data)
# eg:- The lyric "Hello" has 4 characters but is only one word
# so In word level language modelling the training data drastically reduces....which results in less accuracy when there is a lack of training data

# # Words2vec

# In[1]:


import gensim 


# In[2]:


for i in range(0,len(word)):
        word[i] = [word.lower() for word in word[i]]


# ### 70 songs are analysed

# In[199]:


len(word)


# In[200]:


model = gensim.models.Word2Vec(
        word,
        size=150,
        window=10,
        min_count=2,
        workers=10)
model.train(word, total_examples=len(word), epochs=10)


# In[201]:


print(model.similarity('eminem', 'rap'))


# In[202]:


print(model.similarity('eminem', 'marshall'))


# ### Sounds About Right

# In[203]:


model.most_similar('eminem')

