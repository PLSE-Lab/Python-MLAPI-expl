#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import numpy  # linear algebra
from IPython.display import display, HTML
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import *
from keras.layers import Input, Dense, merge
from numpy import newaxis as na
from pickle import load
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import RNN

from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.utils import plot_model
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
from pickle import load
from numpy.random import shuffle
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# In[ ]:


# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs


# In[ ]:


# clean a list of lines
def clean_pairs(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [re_punc.sub('', w) for w in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)


# In[ ]:


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


# In[ ]:


# load dataset
filename = '../input/engliesh-german/deu.txt'
doc = load_doc(filename)
# split into english-german pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# save clean pairs to file
save_clean_data(clean_pairs, 'english-german.pkl')
# spot check
for i in range(2):
    print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))


# In[ ]:



# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))


# In[ ]:


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


# In[ ]:


# load dataset
raw_dataset = load_clean_sentences('english-german.pkl')
# reduce dataset size
n_sentences = 10000
dataset = raw_dataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
train, test = dataset[:9000], dataset[9000:]
# save
save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, 'english-german-train.pkl')
save_clean_data(test, 'english-german-test.pkl')


# In[ ]:


# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))


# In[ ]:


# load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
for i in train[1:5]:
    print(i)
test = load_clean_sentences('english-german-test.pkl')
for i in test[1:5]:
    print(i)


# In[ ]:


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# In[ ]:


# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)


# In[ ]:


# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])

eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % (ger_length))


# In[ ]:


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


# In[ ]:


# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


# In[ ]:


#define NMT model
#define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model
# compile model
    
   # model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    #inputs1 =  model.add(LSTM(n_units,return_sequences=True))
   # attention_probs = model.add((Dense(tar_vocab, activation='softmax')))
    #outputs1 = (merge([inputs1, attention_probs], name='attention_mul', mode='mul'))
    #model.add(RepeatVector(tar_timesteps))
    #model.add(LSTM(n_units, return_sequences=True))(outputs1)
   # model.add((Dense(tar_vocab, activation='softmax')))
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    #return model


# In[ ]:


def build_model():
    inputs = Input(shape=(input_dim,))

    # ATTENTION PART STARTS HERE
    attention_probs = Dense(input_dim, activation='softmax', name='attention_vec')(inputs)
    attention_mul = merge([inputs, attention_probs], output_shape=32, name='attention_mul')
    # ATTENTION PART FINISHES HERE

    attention_mul = Dense(64)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


# In[ ]:


# prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)


# In[ ]:


# prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)


# In[ ]:


# define model
model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)


# In[ ]:



#fit model
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1,
save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=300, batch_size=1000, validation_data=(testX, testY),
callbacks=[checkpoint], verbose=2)


# In[ ]:


model.save_weights("weights_NMT.h5")


# In[ ]:


= model.layers[0].get_weights()[0]


# In[ ]:


embedings5 = model.layers[0].get_weights()[0]


# In[ ]:


model.layers


# In[ ]:


embedings3[0]


# In[ ]:


# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        #print(i)
        
        word = word_for_id(i, tokenizer)
        #print(word)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)


# In[ ]:


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        
        if index == integer:
            
            return word
    return None
        


# In[ ]:


trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])


# In[ ]:


# evaluate the skill of the model
def evaluate_model(model,sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # translate encoded source text
        #print(source.shape[0])
        source = source.reshape((1, source.shape[0]))
        #print(source.shape)
        translation = predict_sequence(model, eng_tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        if i < 20:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# In[ ]:


evaluate_model(model, trainX[10:20], train[10:20])


# In[ ]:


#test[0:100]


# In[ ]:


plot_model(model, to_file='model.png')


# In[ ]:


model.layers[3].weights


# In[ ]:


first_layer_weights = model.layers[0].get_weights()[0]


# In[ ]:


second_layer_weights_xh = model.layers[1].get_weights()[0]
second_layer_weights_hh = model.layers[1].get_weights()[1]
second_layer_bias_xh = model.layers[1].get_weights()[2]


# In[ ]:


third_layer_weights_xh = model.layers[2].get_weights()[0]
third_layer_weights_hh = model.layers[2].get_weights()[1]
third_layer_bias_xh = model.layers[2].get_weights()[2]


# In[ ]:


fourth_layer_weights = model.layers[3].get_weights()[0]
fourth_layer_bias = model.layers[3].get_weights()[1]


# In[ ]:


model.weights


# In[ ]:


T = len(testX[0])
d   = int(second_layer_weights_xh.shape[1]/4) 
# initialize
gates_xh  = np.zeros((T, 4*d))  
gates_hh  = np.zeros((T, 4*d)) 
gates_pre = np.zeros((T, 4*d))  # gates i, g, f, o pre-activation
gates     = np.zeros((T, 4*d))  # gates i, g, f, o activation
h         = np.zeros((T+1, d))
c         = np.zeros((T+1, d))


# In[ ]:


gates_xh_2  = np.zeros((T, 4*d))  
gates_hh_2  = np.zeros((T, 4*d)) 
gates_pre_2 = np.zeros((T, 4*d))  # gates i, g, f, o pre-activation
gates_2     = np.zeros((T, 4*d))  # gates i, g, f, o activation
h_2         = np.zeros((T+1, d))
c_2         = np.zeros((T+1, d))


# In[ ]:


gates_xh_3  = np.zeros((T, 4*d))  
gates_hh_3  = np.zeros((T, 4*d)) 
gates_pre_3 = np.zeros((T, 4*d))  # gates i, g, f, o pre-activation
gates_3     = np.zeros((T, 4*d))  # gates i, g, f, o activation
h_3         = np.zeros((T+1, d))
c_3         = np.zeros((T+1, d))


# In[ ]:



y         = np.zeros((T+1,2309))


# In[ ]:


y_2         = np.zeros((T+1,2309))


# In[ ]:


e      = first_layer_weights.shape[1]                # word embedding dimension
x      = np.zeros((T, e))
#input1 = ["tom"]
#testX = encode_sequences(ger_tokenizer, ger_length, input1)
#testX
array1 = np.append(testX[50][0:4],testX[51][0:6])
array1
x[:,:] = first_layer_weights[trainX[19],:]


# **the below is the mathematical model for LSTM and dense layer, we are directly sending the word embedings of input into the for loop, where in the above cell we have taken the word embedings for a input sequence.**
# ****

# In[ ]:


idx    = np.hstack((np.arange(0,d), np.arange(d,2*d),np.arange(3*d,4*d))).astype(int) # indices of the gates i,f,o
for t in range(T):
    gates_xh[t]    = np.dot(np.transpose(second_layer_weights_xh),(x[t])) +  second_layer_bias_xh
    gates_hh[t]    =   np.dot(np.transpose(second_layer_weights_hh),(h[t-1]))  
    gates_pre[t]   = gates_xh[t] + gates_hh[t]
    gates[t,idx]   = 1.0/(1.0 + np.exp(- gates_pre[t,idx]))
    gates[t,2*d:3*d] = np.tanh(gates_pre[t,2*d:3*d]) 
    c[t]           = gates[t,d:2*d]*c[t-1] + gates[t,0:d]*gates[t,2*d:3*d]
    h[t]           = gates[t,3*d:4*d]*np.tanh(c[t])
   

for t in range(T):
    gates_xh_2[t]   = np.dot(np.transpose(third_layer_weights_xh),(h[t])) +  third_layer_bias_xh
    gates_hh_2[t]   =   np.dot(np.transpose(third_layer_weights_hh),(h_2[t-1]))  
    gates_pre_2[t]   = gates_xh_2[t] + gates_hh_2[t]
    gates_2[t,idx]   = 1.0/(1.0 + np.exp(- gates_pre_2[t,idx]))
    gates_2[t,2*d:3*d] = np.tanh(gates_pre_2[t,2*d:3*d]) 
    c_2[t]           = gates_2[t,d:2*d]*c_2[t-1] + gates_2[t,0:d]*gates_2[t,2*d:3*d]
    h_2[t]           = gates_2[t,3*d:4*d]*np.tanh(c_2[t])
    y[t]             = np.dot(np.transpose(fourth_layer_weights),h_2[t]) + fourth_layer_bias
    
"""for t in range(T):
    gates_xh_3[t]   = np.dot(np.transpose(third_layer_weights_xh),(h_2[t])) +  third_layer_bias_xh
    gates_hh_3[t]   =   np.dot(np.transpose(third_layer_weights_hh),(h_3[t-1]))  
    gates_pre_3[t]   = gates_xh_3[t] + gates_hh_3[t]
    gates_3[t,idx]   = 1.0/(1.0 + np.exp(- gates_pre_3[t,idx]))
    gates_3[t,2*d:3*d] = np.tanh(gates_pre_3[t,2*d:3*d]) 
    c_3[t]           = gates_3[t,d:2*d]*c_3[t-1] + gates_3[t,0:d]*gates_3[t,2*d:3*d]
    h_3[t]           = gates_3[t,3*d:4*d]*np.tanh(c_3[t])
    y_2[t]             = np.dot(np.transpose(fourth_layer_weights),h_3[t]) + fourth_layer_bias"""
    
    
            


# In[ ]:


print(model.png)


# In[ ]:


def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t)
    return a


# In[ ]:


target = list()
for vector in y:
    integer = argmax(soft_max(vector))
    print(integer)
    word = word_for_id(integer, eng_tokenizer)
    print(word)
    if word is None:
        break
    target.append(word)
    


# In[ ]:


for i in target:
    print(i)


# In[ ]:


def lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor, debug=False):
    """
    LRP for a linear layer with input dim D and output dim M.
    Args:
    - hin:            forward pass input, of shape (D,)
    - w:              connection weights, of shape (D, M)
    - b:              biases, of shape (M,)
    - hout:           forward pass output, of shape (M,) (unequal to np.dot(w.T,hin)+b if more than one incoming layer!)
    - Rout:           relevance at layer output, of shape (M,)
    - bias_nb_units:  number of lower-layer units onto which the bias/stabilizer contribution is redistributed
    - eps:            stabilizer (small positive number)
    - bias_factor:    for global relevance conservation set to 1.0, otherwise 0.0 to ignore bias redistribution
    Returns:
    - Rin:            relevance at layer input, of shape (D,)
    """
    #print("hout")
    #print(hout)
    #for i in hout[:5]:
      #  print(i)
    sign_out = np.where(hout[na,:]>=0, 1., -1.) # shape (1, M)
    #sign_out = hout[na,:]
    
    #print(na)
    #print(sign_out)
   

    
    numer    = (w* hin[:,na]) + ( (bias_factor*b[na,:]*1. + eps*sign_out*1.) * 1./bias_nb_units ) # shape (D, M)
    #print("Numerator")
    
   # print("################Numerator###################")
    #print(numer)
    denom    = hout[na,:] + (eps*sign_out*1.)   # shape (1, M)
    #print("denominator")
    #print(denom)
    message  = (numer/denom) * Rout[na,:]       # shape (D, M)
    
    Rin      = message.sum(axis=1)              # shape (D,)
    
    # Note: local  layer   relevance conservation if bias_factor==1.0 and bias_nb_units==D
    #       global network relevance conservation if bias_factor==1.0 (can be used for sanity check)
    if debug:
        print("local diff: ", Rout.sum() - Rin.sum())
    
    return Rin


# In[ ]:



T = len(x)
d   = int(second_layer_weights_xh.shape[1]/4) 
e      = first_layer_weights.shape[1]                # word embedding dimension
C      = 2309  # number of classes
idx    = np.hstack((np.arange(0,d), np.arange(d,2*d),np.arange(3*d,4*d))).astype(int)
LRP_class = argmax(soft_max(y[1]))
Rx = np.zeros(x.shape)
Rh  = np.zeros((T+1, d))
Rc  = np.zeros((T+1, d))
Rg  = np.zeros((T,   d)) # gate g only

Rx_2 = np.zeros(x.shape)
Rh_2  = np.zeros((T+1, d))
Rc_2  = np.zeros((T+1, d))
Rg_2  = np.zeros((T,   d)) # gate g only
Rout_mask  = np.zeros((C))
Rout_mask[LRP_class] = 1.0 
eps=0.1
#print(h[9])
bias_factor=1.0
Rh_2[T-1]  = lrp_linear(h_2[T-1],  fourth_layer_weights , fourth_layer_bias, y[1], y[1]*Rout_mask, 2*d, eps, bias_factor, debug=False)

for t in reversed(range(T)):
    Rc_2[t]   += Rh_2[t]
    Rc_2[t-1]  = lrp_linear(gates_2[t,d:2*d]*c_2[t-1],     np.identity(d), np.zeros((d)), c_2[t], Rc_2[t], 2*d, eps, bias_factor, debug=False)
    Rg_2[t]    = lrp_linear(gates_2[t,0:d]*gates_2[t,2*d:3*d], np.identity(d), np.zeros((d)),c_2[t], Rc_2[t], 2*d, eps, bias_factor, debug=False)
    Rh[t]   = lrp_linear(h[t],third_layer_weights_xh[t,2*d:3*d],third_layer_bias_xh[2*d:3*d],gates_pre_2[t,2*d:3*d], Rg_2[t], d+e, eps, bias_factor, debug=False)
    
    Rh_2[t-1]  = lrp_linear(h_2[t-1], third_layer_weights_hh[t,2*d:3*d], third_layer_bias_xh[2*d:3*d], gates_pre_2[t,2*d:3*d], Rg_2[t], d+e, eps, bias_factor, debug=False)
          
 
    Rc[t]   += Rh[t]
    Rc[t-1]  = lrp_linear(gates[t,d:2*d]*c[t-1],     np.identity(d), np.zeros((d)), c[t], Rc[t], 2*d, eps, bias_factor, debug=False)
    Rg[t]    = lrp_linear(gates[t,0:d]*gates[t,2*d:3*d], np.identity(d), np.zeros((d)),c[t], Rc[t], 2*d, eps, bias_factor, debug=False)
    Rx[t]   = lrp_linear(x[t],second_layer_weights_xh[t,2*d:3*d],second_layer_bias_xh[2*d:3*d],gates_pre[t,2*d:3*d], Rg[t], d+e, eps, bias_factor, debug=False)
    
    Rh[t-1]  = lrp_linear(h[t-1], second_layer_weights_hh[t,2*d:3*d], second_layer_bias_xh[2*d:3*d], gates_pre[t,2*d:3*d], Rg[t], d+e, eps, bias_factor, debug=False)
                    
 


# In[ ]:


R_words             = np.sum(Rx, axis=1)


# In[ ]:


print(R_words)


# In[ ]:


print(gates_pre[1,2*d:3*d])


# In[ ]:


def rotate(l, n):
    return np.append(l[n:] , l[:n])


# In[ ]:


#R_words = rotate(R_words,1)


# In[ ]:


R_words_previous             = np.sum(Rh, axis=1)


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt

def rescale_score_by_abs (score, max_score, min_score):
    """
    rescale positive score to the range [0.5, 1.0], negative score to the range [0.0, 0.5],
    using the extremal scores max_score and min_score for normalization
    """
    
    # CASE 1: positive AND negative scores occur --------------------
    if max_score>0 and min_score<0:
    
        if max_score >= abs(min_score):   # deepest color is positive
            if score>=0:
                return 0.5 + 0.5*(score/max_score)
            else:
                return 0.5 - 0.5*(abs(score)/max_score)

        else:                             # deepest color is negative
            if score>=0:
                return 0.5 + 0.5*(score/abs(min_score))
            else:
                return 0.5 - 0.5*(score/min_score)   
    
    # CASE 2: ONLY positive scores occur -----------------------------       
    elif max_score>0 and min_score>=0: 
        if max_score == min_score:
            return 1.0
        else:
            return 0.5 + 0.5*(score/max_score)
    
    # CASE 3: ONLY negative scores occur -----------------------------
    elif max_score<=0 and min_score<0: 
        if max_score == min_score:
            return 0.0
        else:
            return 0.5 - 0.5*(score/min_score)    
  
      
def getRGB (c_tuple):
    return "#%02x%02x%02x"%(int(c_tuple[0]*255), int(c_tuple[1]*255), int(c_tuple[2]*255))

     
def span_word (word, score, colormap):
    return "<span style=\"background-color:"+getRGB(colormap(score))+"\">"+word+"</span>"


def html_heatmap (words, scores, cmap_name="bwr"):
    
    colormap  = plt.get_cmap(cmap_name)
    print(len(words)) 
    assert len(words)==len(scores)
    max_s     = max(scores)
    min_s     = min(scores)
    
    output_text = ""
    
    for idx, w in enumerate(words):
        score       = rescale_score_by_abs(scores[idx], max_s, min_s)
        output_text = output_text + str(score) + span_word(w, score, colormap) + " "
    
    return output_text + "\n"


# In[ ]:


#print ("prediction scores: ",   scores)
print ("\nLRP target class:  ", LRP_class)
word = word_for_id(LRP_class, eng_tokenizer)
print(word)
print ("\nLRP relevances:")
target = list()
for integer in trainX[19]:
    print(integer)
    word = word_for_id(integer, ger_tokenizer)
    if word is None:
        break
    target.append(word)
print ("\nLRP heatmap:")    
display(HTML(html_heatmap(target, R_words[0: len(target)])))


# In[ ]:


#print ("prediction scores: ",   scores)
print ("\nLRP target class:  ", LRP_class)
word = word_for_id(LRP_class, eng_tokenizer)
print(word)
print ("\nLRP relevances:")
target = list()
for vector in y:
    integer = argmax(soft_max(vector))
    print(integer)
    word = word_for_id(integer, eng_tokenizer)
    if word is None:
        break
    target.append(word)
print ("\nLRP heatmap:")    
display(HTML(html_heatmap(target, R_words_previous[0: len(target)])))


# In[ ]:


def lrp( w, LRP_class, eps=0.001, bias_factor=1.0):
    """
    Update the hidden layer relevances by performing LRP for the target class LRP_class
    """
    # forward pass
     
    
    T      = len(w)
    d      = int(Wxh_Left.shape[0]/4)
    e      = self.E.shape[1] 
    C      = self.Why_Left.shape[0]  # number of classes
    idx    = np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) 
    
    # initialize
    Rx       = np.zeros(self.x.shape)
    Rx_rev   = np.zeros(self.x.shape)
    
    Rh_Left  = np.zeros((T+1, d))
    Rc_Left  = np.zeros((T+1, d))
    Rg_Left  = np.zeros((T,   d)) # gate g only
    Rh_Right = np.zeros((T+1, d))
    Rc_Right = np.zeros((T+1, d))
    Rg_Right = np.zeros((T,   d)) # gate g only
    
    Rout_mask            = np.zeros((C))
    Rout_mask[LRP_class] = 1.0  
    
    # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
    Rh_Left[T-1]  = lrp_linear(self.h_Left[T-1],  self.Why_Left.T , np.zeros((C)), self.s, self.s*Rout_mask, 2*d, eps, bias_factor, debug=False)
    Rh_Right[T-1] = lrp_linear(self.h_Right[T-1], self.Why_Right.T, np.zeros((C)), self.s, self.s*Rout_mask, 2*d, eps, bias_factor, debug=False)
    
    for t in reversed(range(T)):
        #print(Rc_Left[t])
        Rc_Left[t]   += Rh_Left[t]
        Rc_Left[t-1]  = lrp_linear(self.gates_Left[t,2*d:3*d]*self.c_Left[t-1],     np.identity(d), np.zeros((d)), self.c_Left[t], Rc_Left[t], 2*d, eps, bias_factor, debug=False)
        Rg_Left[t]    = lrp_linear(self.gates_Left[t,0:d]*self.gates_Left[t,d:2*d], np.identity(d), np.zeros((d)), self.c_Left[t], Rc_Left[t], 2*d, eps, bias_factor, debug=False)
        Rx[t]         = lrp_linear(self.x[t],        self.Wxh_Left[d:2*d].T, self.bxh_Left[d:2*d]+self.bhh_Left[d:2*d], self.gates_pre_Left[t,d:2*d], Rg_Left[t], d+e, eps, bias_factor, debug=False)
        Rh_Left[t-1]  = lrp_linear(self.h_Left[t-1], self.Whh_Left[d:2*d].T, self.bxh_Left[d:2*d]+self.bhh_Left[d:2*d], self.gates_pre_Left[t,d:2*d], Rg_Left[t], d+e, eps, bias_factor, debug=False)
        
        Rc_Right[t]  += Rh_Right[t]
        Rc_Right[t-1] = lrp_linear(self.gates_Right[t,2*d:3*d]*self.c_Right[t-1],     np.identity(d), np.zeros((d)), self.c_Right[t], Rc_Right[t], 2*d, eps, bias_factor, debug=False)
        Rg_Right[t]   = lrp_linear(self.gates_Right[t,0:d]*self.gates_Right[t,d:2*d], np.identity(d), np.zeros((d)), self.c_Right[t], Rc_Right[t], 2*d, eps, bias_factor, debug=False)
        Rx_rev[t]     = lrp_linear(self.x_rev[t],     self.Wxh_Right[d:2*d].T, self.bxh_Right[d:2*d]+self.bhh_Right[d:2*d], self.gates_pre_Right[t,d:2*d], Rg_Right[t], d+e, eps, bias_factor, debug=False)
        Rh_Right[t-1] = lrp_linear(self.h_Right[t-1], self.Whh_Right[d:2*d].T, self.bxh_Right[d:2*d]+self.bhh_Right[d:2*d], self.gates_pre_Right[t,d:2*d], Rg_Right[t], d+e, eps, bias_factor, debug=False)
               
    return Rx, Rx_rev[::-1,:], Rh_Left[-1].sum()+Rc_Left[-1].sum()+Rh_Right[-1].sum()+Rc_Right[-1].sum()

  

