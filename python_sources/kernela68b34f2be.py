#!/usr/bin/env python
# coding: utf-8

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


import numpy as np
import os
from os import path
import pandas as pd
import pickle
from pickle import dump
from keras.preprocessing import image, sequence,text
from keras.applications import inception_v3
from keras.layers import Input,Dense, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector, Concatenate
from keras.models import Sequential, Model,load_model
from keras.optimizers import Adam,Nadam,RMSprop
from keras.applications.inception_v3 import InceptionV3
from os import listdir
from keras.preprocessing.sequence import pad_sequences
from keras.layers.merge import add
from keras.utils import to_categorical,plot_model
from keras.callbacks import ModelCheckpoint
from numpy import array
from nltk import word_tokenize


# In[ ]:


train_data = pd.read_csv("../input/hindi-visual-genome-text/hindi-visual-genome-train.txt",sep = "\t", engine = "python",nrows=8000, header = None)
val_data = pd.read_csv("../input/hindi-visual-genome-text/hindi-visual-genome-dev.txt",sep = "\t", engine = "python", header = None)
test_data = pd.read_csv("../input/hindi-visual-genome-text/hindi-visual-genome-test.txt",sep = "\t", engine = "python", header = None)

features_train = pickle.load(open('../input/pickles-hindi-visual-genome/features_train.pkl', 'rb'))
features_test = pickle.load(open('../input/pickles-hindi-visual-genome/features_test.pkl', 'rb'))
features_val = pickle.load(open('../input/pickles-hindi-visual-genome/features_val.pkl', 'rb'))
#tokens = pickle.load(open('../input/pickles-hindi-visual-genome/tokens.pkl', 'rb'))
#tok2indx = pickle.load(open('../input/pickles-hindi-visual-genome/tok2indx.pkl', 'rb'))
#indx2tok = pickle.load(open('../input/pickles-hindi-visual-genome/indx2tok.pkl', 'rb'))
#embedding_matrix = pickle.load(open('../input/pickles-hindi-visual-genome/embedding_matrix.pkl', 'rb'))


# In[ ]:


print(len(train_data))
print(len(test_data))
print(len(val_data))


# In[ ]:



tokens = {'startseq', 'endseq'}
for i in range(len(train_data)):
     try:
         for token in word_tokenize(train_data[5][i]):
             tokens.add(token)
     except:
         pass
print(len(tokens)) 


# In[ ]:


skip=[]
for index, row in val_data.iterrows():
    for i in word_tokenize(row[5]):
        if i not in tokens:
            skip.append(index)
            continue
val_data.drop(skip, inplace = True)


# In[ ]:


print(len(val_data))
#printval_data.iloc[:792]


# In[ ]:




def load_desc(data):
    dictt=dict()
    for i in range(len(data)):
        if i not in skip:
            sent=data[5][i]
            if type(sent) == str:
                sent='startseq '+sent+' endseq'
            k=str(data[0][i])
            dictt[k] = sent
        #print(k,sent)
            dictt.update({k:sent})
        
    return dictt
train_desc=load_desc(train_data)
test_desc=load_desc(test_data)
val_desc=load_desc(val_data)


# In[ ]:


def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(descriptions[key].split())]
	return all_desc
vocabulary=to_vocabulary(train_desc)


# In[ ]:


print(len(vocabulary))


# In[ ]:





# In[ ]:



            


# In[ ]:





# In[ ]:



        


# In[ ]:



 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



def to_lines(descriptions):
    all_desc=list()
    for key in descriptions.keys():
        all_desc.append(descriptions[key])
    return all_desc

 
# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines=to_lines(descriptions)
    #print(lines)
    tokenizer=text.Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
    
    
    

 
# prepare tokenizer
tokenizer = create_tokenizer(train_desc)
print(tokenizer)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)


# In[ ]:


def create_sequences(tokenizer,vocab_size,max_length,descriptions,photos):
        x1,x2,y=list(),list(),list()
        
        #seq=[]
        for key,desc in descriptions.items():
            #print('yo')
            try:
                #print('yo')
                seq = tokenizer.texts_to_sequences([desc])[0]
                #print('yo')
                #print(len(seq))
                for i in range(1,len(seq)):
                    in_seq,out_seq=seq[:i],seq[i]
                    in_seq=pad_sequences([in_seq],maxlen=max_length)[0]
                    out_seq=to_categorical([out_seq],num_classes=vocab_size)[0]
                    x1.append(photos[key][0])
                    x2.append(in_seq)
                    y.append(out_seq)
            except:
                pass
        return array(x1),array(x2),array(y)


# In[ ]:


def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)
max_length=max_length(train_desc)
print(max_length)


# In[ ]:


def define_model(max_length,vocab_size):
        # feature extractor model
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        # sequence model
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size,
                                256,
                                mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        # decoder model
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        # tie it together [image, seq] [word]
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        opt= Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
        
        # summarize model
        print(model.summary())
        #plot_model(model, to_file='model.png', show_shapes=True)
        return model


# In[ ]:


X1train, X2train, ytrain = create_sequences(tokenizer,vocab_size, max_length, train_desc, features_train)


# In[ ]:


X1val, X2val, yval = create_sequences(tokenizer,vocab_size, max_length, val_desc, features_val)


# In[ ]:


model = define_model(max_length,vocab_size)
# define checkpoint callback
filepath = 'model_test4-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
# fit model
model.fit([X1train, X2train], ytrain, epochs=200, verbose=2, callbacks=[checkpoint], validation_data=([X1val, X2val], yval))


# In[ ]:


model = define_model(max_length,vocab_size)
""""features=features_train['2330902']
print(features)"""


# In[ ]:



def word_for_int_id(integer,tokenizer):
    for word,index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None
from numpy import argmax,argsort

def generate_desc(model,tokenizer,photo,max_length):
    in_text='startseq'
    for i in range(max_length):
        seq=tokenizer.texts_to_sequences([in_text])[0]
        seq=pad_sequences([seq],maxlen=max_length)
        model.load_weights('../input/h5-files2/model_test3-ep048-loss1.943-val_loss4.481.h5')
        pred=model.predict([photo,seq],verbose=1)
        pred=argmax(pred)
        word=word_for_int_id(pred,tokenizer)
        if word is None:
            break
        in_text+=' ' + word
        if word=='endseq':
            break
    return in_text
    


# In[ ]:


""""desc=generate_desc(model,tokenizer,features,max_length)"""


# In[ ]:


from nltk.translate.bleu_score import corpus_bleu


# In[ ]:


def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	for key, desc in descriptions.items():
		# generate description
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# store actual and predicted
		references = [desc.split()]
		actual.append(references)
		predicted.append(yhat.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# In[ ]:



evaluate_model(model, train_desc, features_train, tokenizer, max_length)


# In[ ]:





# In[ ]:


""""def load_desc(data):
        dictt=dict()
        for i in data.iterrows():
            sent=i[1][6]
            if type(sent) == str:
                sent='| '+sent+' `'
            k=str(i[1][0])
            dictt[k] = sent
            #print(k,sent)
            #dictt.update({k:sent})
        return dictt
train_desc=load_desc(train_data)
test_desc=load_desc(test_data)
val_desc=load_desc(val_data)"""


# In[ ]:



 


# In[ ]:



""""def create_sequences(vocab_size,max_length,descriptions,photos):
        x1,x2,y=list(),list(),list()
        seq=[]
        for tok in word_tokenize(descriptions):
            try:
                seq.append(tok2indx[tok])
                for i in range(len(seq)):
                    in_seq,out_seq=seq[:i],seq[i]
                    in_seq=pad_sequences([in_seq],padding='post',maxlen=max_length,value=tok2indx[''])[0]
                    out_seq=to_categorical([out_seq],num_classes=vocab_size)[0]
                    x1.append(photos)
                    x2.append(in_seq)
                    y.append(out_seq)
            except:
                pass
        return array(x1),array(x2),array(y)
    
        
def data_generator(descriptions, photos, vocab_size, max_length):
     while 1:
        for key,desc in descriptions.items():
             if(type(desc)!=str):
                 continue    
             photo=photos[key][0]
             in_img,in_seq,out_word=create_sequences(vocab_size,max_length,desc,photo)
             yield[[in_img,in_seq],out_word]


# In[ ]:


"""
def data_generator(descriptions, photos, vocab_size, max_length,batch_size):
    x1=[]
    x2=[]
    y=[]
    count=0
    while 1:
        for key,desc in descriptions.items():
            if(type(desc)!=str):
                continue 
            count=count+1
            photo=photos[key][0]
            in_img,in_seq,out_word=create_sequences(vocab_size,max_length,desc,photo)
            #print(type(in_img))
            x1.append(in_img)
            x2.append(in_seq)
            y.append(out_word)
            if count>batch_size:
                yield[[x1,x2],y]
                x1=[]
                x2=[]
                y=[]
                count=0
""""""


# In[ ]:



""""
 def data_generator(descriptions, photos, vocab_size, max_length,batch_size):
     while 1:
        for key,desc in descriptions.items():
             if(type(desc)!=str):
                 continue    
             photo=photos[key][0]
             in_img,in_seq,out_word=create_sequences(vocab_size,max_length,desc,photo)
             yield[[in_img,in_seq],out_word]

""""""


# In[ ]:






# In[ ]:


""""def define_model(max_length,vocab_size):
        # feature extractor model
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(300, activation='relu')(fe1)
        # sequence model
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size,
                                300,
                                weights=[embedding_train],
                                         input_length=40,
                                         trainable=False)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(300)(se2)
        # decoder model
        decoder1 = add([fe2, se3])
        decoder2 = Dense(300, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        # tie it together [image, seq] [word]
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        # summarize model
        print(model.summary())
        #plot_model(model, to_file='model.png', show_shapes=True)
        return model


# In[ ]:


""""model=define_model(max_length, vocab_size)


# In[ ]:


""""x1val,x2val,yval=create_sequences1(vocab_size,max_length,val_desc,features_val)


# In[ ]:



"""""generator = data_generator(train_desc,features_train, vocab_size, max_length,batch_size)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)


# In[ ]:


"""batch_size=2000
epochs = 20
steps = len(train_desc)//batch_size
steps1=len(val_desc)
filepath = 'model11-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
for i in range(epochs):
    train_generator=data_generator(train_desc,features_train,vocab_size,max_length,batch_size)
    
    #val_generator=data_generator(val_desc,features_val,vocab_size,max_length)
    model.fit_generator(train_generator,epochs=1,steps_per_epoch=steps,verbose=1,callbacks=[checkpoint],validation_data=([x1val,x2val],yval))
    
""""

