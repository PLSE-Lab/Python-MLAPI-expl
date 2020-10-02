#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

def load_data(path):
   
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')


# In[ ]:



codes = load_data('../input/cipher.txt')
plaintext = load_data('../input/plaintext.txt')


# In[ ]:


codes[:5]


# In[ ]:


plaintext[:5]


# In[ ]:


from keras.preprocessing.text import Tokenizer


def tokenize(x):
   
    x_tk = Tokenizer(char_level=True)
    x_tk.fit_on_texts(x)
   

    return x_tk.texts_to_sequences(x), x_tk

text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))


# In[ ]:


import numpy as np
from keras.preprocessing.sequence import pad_sequences


def pad(x, length=None):
   
    if length is None:
        length = max([len(sentence) for sentence in x])
    
    return pad_sequences(x,maxlen=length,padding='post')

test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))


# In[ ]:


from keras.preprocessing.text import Tokenizer


def tokenize(x):
    
    x_tk = Tokenizer(char_level=True)
    x_tk.fit_on_texts(x)
   

    return x_tk.texts_to_sequences(x), x_tk


# In[ ]:


def preprocess(x, y):
    
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_code_sentences, preproc_plaintext_sentences, code_tokenizer, plaintext_tokenizer =    preprocess(codes, plaintext)

print('Data Preprocessed')


# In[ ]:





# In[ ]:


from keras.layers import GRU, Input, Dense, TimeDistributed
from keras.models import Model
from keras.layers import Activation
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


def simple_model(input_shape, output_sequence_length, code_vocab_size, plaintext_vocab_size):
   
    learning_rate = 1e-3
    input_s =Input(input_shape[1:])
    rnn=GRU(64,return_sequences=True)(input_s)
    logits= TimeDistributed(Dense(plaintext_vocab_size))(rnn)
    
    model =Model(input_s,Activation('softmax')(logits))
    model.compile(loss=sparse_categorical_crossentropy,optimizer=Adam(learning_rate))
    
    

    
    return model


x = pad(preproc_code_sentences, preproc_plaintext_sentences.shape[1])
x = x.reshape((-1, preproc_plaintext_sentences.shape[-2], 1))


# In[ ]:


simple_rnn_model = simple_model(
    x.shape,
    preproc_plaintext_sentences.shape[1],
    len(code_tokenizer.word_index)+1,
    len(plaintext_tokenizer.word_index)+1)


# In[ ]:


simple_rnn_model.fit(x, preproc_plaintext_sentences, batch_size=32, epochs=15, validation_split=0.2)


# In[ ]:


def logits_to_text(logits, tokenizer):
 
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

print('`logits_to_text` function loaded.')

print(logits_to_text(simple_rnn_model.predict(x[2:3])[0], plaintext_tokenizer))


# In[ ]:


plaintext[2]


# In[ ]:




