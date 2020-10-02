#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from lxml import etree
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer
from typing import Dict, List, Tuple
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from string import punctuation as punct
import numpy as np
from nltk.corpus import wordnet as wn


# In[ ]:


import os
os.listdir("../input/glove6b200d")

embeddings_index = {}
f = open(os.path.join('../input/glove6b200d', 'glove.6B.200d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[ ]:


def load_bn2wn_mapping(bn2wn_mapping_path: str, flip_dxn = False) -> Dict[str, str]:
    """
    :param bn2wn_mapping_path; Full path to the BabelNet to WordNet file mapping for filtering the corpora
    :return bn2wn_mapping; the BabelNet to WordNet mapping file encoded as a dictionary with the BabelNet IDs as keys
    """
    first = 0
    second = 1
    if (flip_dxn):
        first = 1
        second = 0
    bn2wn_mapping = dict()
    with open(bn2wn_mapping_path, 'r') as handle:
        for line in handle:
            line = line.strip().split("\t")
            if (line):
                bn2wn_mapping[line[first]] = line[second]

    return bn2wn_mapping

def load_bn2lex_mapping(bn2lex_mapping_path: str, flip_dxn = False) -> Dict[str, str]:
    """
    :param bn2wn_mapping_path; Full path to the BabelNet to WordNet file mapping for filtering the corpora
    :return bn2wn_mapping; the BabelNet to WordNet mapping file encoded as a dictionary with the BabelNet IDs as keys
    """
    first = 0
    second = 1
    if (flip_dxn):
        first = 1
        second = 0
    bn2lex_mapping = dict()
    with open(bn2lex_mapping_path, 'r') as handle:
        for line in handle:
            line = line.strip().split("\t")
            if (line):
                bn2lex_mapping[line[second]] = line[first]

    return bn2lex_mapping

def load_gold_data(gold_data_path: str) -> Dict[str, str]:
    gold_mapping = dict()
    with open(gold_data_path, 'r') as handle:
        for line in handle:
            line = line.strip().split(" ")
            if (line):
                gold_mapping[line[0]] = line[1]

    return gold_mapping


# In[ ]:


import re
from IPython.core.debugger import set_trace

sentence_count = 0

def check_punc(word):
    """ Checks if word does not contain only punctuations
    """
    if not re.match(r'^[_\W]+$', word):
        return True
    else:
        return False

def preprocess_semcor(file_path:str,gold_mapping_path: str, bn2wn_mapping_path: str,bn2lex_mapping_path:str):
    """ Parse xml file  provided in file_path and returns a sentence,
    babelnetId,lemma and anchors respectively for each sentences
    """
    sentences = []
    sequences = []
    sequences_pos = []
    sequences_lex = []
    sentence_count = 0
    
    print("Loading gold data...")
    gold_mapping = load_gold_data(gold_mapping_path)
    print("Loading WordNet mapping data...")
    bn2wn_mapping = load_bn2wn_mapping(bn2wn_mapping_path, True)
    print("Loading Lex mapping data...")
    bn2lex_mapping = load_bn2lex_mapping(bn2lex_mapping_path, True)
    
    print("Parsing Data...")
    context = etree.iterparse(file_path, events=('end',), tag='sentence')
    for event, elem in context:
        sentence_x = ""
        sentence_y = ""
        sentence_y2 = ""
        sentence_y3 = ""
        
        try:
            for e in elem.iter():
               
                if e.tag == "wf" and e.attrib.get('lemma') != False and e.text != None and (check_punc(e.text) == True) and (check_punc(e.attrib['lemma'].strip()) == True):
                    if (len(e.text.split()) > 1):
                        sentence_x = sentence_x + " " + e.text.replace(" ","_")
                    else:
                        sentence_x = sentence_x + " " + e.text
                    sentence_y = sentence_y + " " + "senseless"
                    sentence_y2 = sentence_y2 + " " + "senseless"
                    if (e.attrib.get('pos') != False):
                        sentence_y3 = sentence_y3 + " " + e.attrib['pos'].strip()
                    

                elif e.tag == "instance" and e.attrib.get('id') != False and e.text != None and (check_punc(e.text) == True) and (check_punc(e.attrib['lemma'].strip()) == True):
                        if (len(e.text.split()) > 1):
                            sentence_x = sentence_x + " " + e.text.replace(" ","_")
                        else:
                            sentence_x = sentence_x + " " + e.text

                        sense_key = gold_mapping[e.attrib['id']]
                        wn_synset = wn.lemma_from_key(sense_key).synset()
                        wn_synset_id = "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()
                        sentence_y = sentence_y + " " + bn2wn_mapping[wn_synset_id]
                        sentence_y2 = sentence_y2 + " " + bn2lex_mapping[bn2wn_mapping[wn_synset_id]]
                        if (e.attrib.get('pos') != False):
                            sentence_y3 = sentence_y3 + " " + e.attrib['pos'].strip()
                    
        
            if(len(sentence_x.split()) == len(sentence_y.split()) == len(sentence_y2.split()) == len(sentence_y3.split()) ):     
                sentences.append(sentence_x)
                sequences.append(sentence_y)
                sequences_lex.append(sentence_y2)
                sequences_pos.append(sentence_y3)
                sentence_count += 1

            else:
                print("Mismatch")

        except Exception:
            continue 
            
        print("{:,d} sentences extracted...".format(sentence_count), end="\r")
        # It's safe to call clear() here because no descendants will be accessed
        elem.clear()
        # Also eliminate now-empty references from the root node to <Title> 
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    print ("\nDone!")
    return sentences, sequences,sequences_pos,sequences_lex


# In[ ]:


val_x,val_y,val_y2,val_lex_y = preprocess_semcor("../input/evaluation-data/semeval2007.data.xml","../input/evaluation-data/semeval2007.gold.key.txt", "../input/babelnetdata/babelnet2wordnet.tsv","../input/babelnetdata/babelnet2lexnames.tsv")
sentence_x1,sequence_y1,sequence_y2,sequence_lex_y1 = preprocess_semcor("../input/semcor/semcor.data.xml","../input/semcor/semcor.gold.key.txt", "../input/babelnetdata/babelnet2wordnet.tsv","../input/babelnetdata/babelnet2lexnames.tsv")


# In[ ]:


def Punctuation(string): 
  
    # punctuation marks 
    punctuations = '''!()[]{}|;'"\,<>./?@#$%^&*~'''
  
    # traverse the given string and if any punctuation 
    # marks occur replace it with null 
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, "") 
    
    return string
            
clean_sentences = [Punctuation(s) for s in sentence_x1]
print(clean_sentences[0:2])

clean_sequences = [Punctuation(s) for s in sequence_y1]
print(clean_sequences[0:2])

clean_sequences2 = [Punctuation(s) for s in sequence_y2]
print(clean_sequences2[0:2])

clean_sequences3 = [s for s in sequence_lex_y1]
print(clean_sequences3[0:2])

clean_val_x = [Punctuation(s) for s in val_x]
print(clean_val_x[0:2])

clean_val_y = [Punctuation(s) for s in val_y]
print(clean_val_y[0:2])

clean_val_y2 = [Punctuation(s) for s in val_y2]
print(clean_val_y2[0:2])

clean_val_y3 = [s for s in val_lex_y]
print(clean_val_y3 [0:2])


# In[ ]:


from typing import Tuple, List, Dict
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping
from collections import Counter
from tensorflow.keras.layers import Dense, Input,Masking,LSTM, Embedding,Reshape, Dropout, Activation,TimeDistributed,Bidirectional,concatenate, GlobalMaxPool1D
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
import pickle
from tensorflow.keras.optimizers import SGD
from  tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical 
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import collections
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


# In[ ]:


def tokenize_dataset(sentences,sequences,sequence_pos,sequence_lex):
    """
    Converts each sentences to words

    :param sentences: sentences
    :param sequences: sequences
    :return: encoded X  training set
    """
    all_sentences,all_sentence_tags,all_sentence_pos,all_sentence_lex = [],[],[],[]
    for s,t,p,l in zip(sentences,sequences,sequence_pos,sequence_lex):
        sentence,tags,pos,lex = s.split(),t.split(),p.split(),l.split()
        if (len(sentence) == len(tags) == len(pos) == len(lex)):
            all_sentences.append(np.array(sentence))
            all_sentence_tags.append(np.array(tags))
            all_sentence_pos.append(np.array(pos))
            all_sentence_lex.append(np.array(lex))
        else:
            print("Mismatch!")
    return all_sentences,all_sentence_tags,all_sentence_pos,all_sentence_lex


# In[ ]:


all_sentences,all_sentence_tags,all_sentence_tags2,all_sentence_tags3 = tokenize_dataset(clean_sentences,clean_sequences,clean_sequences2,clean_sequences3)
all_val_x,all_val_y,all_val_y2,all_val_y3 = tokenize_dataset(clean_val_x,clean_val_y,clean_val_y2,clean_val_y3)
print(all_sentences[0])
print(all_sentence_tags[0])
print(all_sentence_tags2[0])
print(all_sentence_tags3[0])
print(all_val_x[0])
print(all_val_y[0])
print(all_val_y2[0])
print(all_val_y3[0])


# In[ ]:


from sklearn.model_selection import train_test_split

train_sentences, train_tags,test_sentences, test_tags = all_sentences, all_sentence_tags,all_val_x,all_val_y
train_tags2,train_tags3 = all_sentence_tags2,all_sentence_tags3
test_tags2,test_tags3 = all_val_y2,all_val_y3
train_sentences_X, train_tags_y,train_tags_y2,train_tags_y3,test_sentences_X,test_tags_y = [], [], [],[],[],[]
test_tags_y2,test_tags_y3 = [],[]


# In[ ]:


def vocabulary():  
    """
    This is the function to build the vocabulary of the dataset.
    
    :param unigram_path: The path to the file that contains the unigrams
    :return: None
    """
    words, tags,tags2,tags3 = set([]), set([]), set([]),set([])
 
    for s in train_sentences:
        for w in s:
            words.add(w.lower())

    for ts in train_tags:
        for t in ts:
            tags.add(t)
    
    for ts in train_tags2:
        for t in ts:
            tags2.add(t)
            
    for ts in train_tags3:
        for t in ts:
            tags3.add(t)

    word2index = {w: i + 2 for i, w in enumerate(list(words))}
    word2index['-PAD-'] = 0  # The special value used for padding
    word2index['-OOV-'] = 1  # The special value used for OOVs

    tag2index = {t: i + 2 for i, t in enumerate(list(tags))}
    tag2index['-PAD-'] = 0  # The special value used to paddin
    tag2index['-OOV-'] = 1  # The special value used to paddin
    
    tag2index2 = {t: i + 2 for i, t in enumerate(list(tags2))}
    tag2index2['-PAD-'] = 0  # The special value used to paddin
    tag2index2['-OOV-'] = 1  # The special value used to paddin
    
    tag2index3 = {t: i + 2 for i, t in enumerate(list(tags3))}
    tag2index3['-PAD-'] = 0  # The special value used to paddin
    tag2index3['-OOV-'] = 1  # The special value used to paddin
    
 
    for s in train_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])

        train_sentences_X.append(s_int)
 
    for s in test_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])

        test_sentences_X.append(s_int)

    for s in train_tags:
        t_int = []
        for t in s:
            try:
                t_int.append(tag2index[t])
            except KeyError:
                t_int.append(tag2index['-OOV-'])
                
        train_tags_y.append(t_int)
        
    for s in train_tags2:
        t_int = []
        for t in s:
            try:
                t_int.append(tag2index2[t])
            except KeyError:
                t_int.append(tag2index2['-OOV-'])
                
        train_tags_y2.append(t_int)
        
    for s in train_tags3:
        t_int = []
        for t in s:
            try:
                t_int.append(tag2index3[t])
            except KeyError:
                t_int.append(tag2index3['-OOV-'])
                
        train_tags_y3.append(t_int)
                                 
    for s in test_tags:
        t_int = []
        for t in s:
            try:
                t_int.append(tag2index[t])
            except KeyError:
                t_int.append(tag2index['-OOV-'])
                
        test_tags_y.append(t_int)
        
    for s in test_tags2:
        t_int = []
        for t in s:
            try:
                t_int.append(tag2index2[t])
            except KeyError:
                t_int.append(tag2index2['-OOV-'])
                
        test_tags_y2.append(t_int)
        
    for s in test_tags3:
        t_int = []
        for t in s:
            try:
                t_int.append(tag2index3[t])
            except KeyError:
                t_int.append(tag2index3['-OOV-'])
                
        test_tags_y3.append(t_int)
        
        
    print("Train: ",train_sentences_X[0])
    print("Train tags: ",train_tags_y[0])
    print("Train tags2: ",train_tags_y2[0])
    print("Train tags3: ",train_tags_y3[0])
    print("Test: ",test_sentences_X[0])
    print("Test tags: ",test_tags_y[0])
    print("Test tags2: ",test_tags_y2[0])
    print("Test tags3: ",test_tags_y3[0])
 
    return word2index,tag2index,tag2index2,tag2index3
#     vocab = dict()
#     for line in file_path_x:
#         words = line.split()
#         for word in words:
#             word = word.lower()
#             if word not in vocab:
#                 vocab[word] = 1
#             else:
#                 vocab[word] += 1
#     return vocab


# In[ ]:


word_len , tag_len ,tag_len2,tag_len3 = vocabulary()
print(len(word_len))
print(len(tag_len))
print(len(tag_len2))
print(len(tag_len3))

import json
json.dump( word_len, open( "words_vocab.json", 'w' ) )

json.dump( tag_len, open( "sense_vocab.json", 'w' ) )

json.dump( tag_len2, open( "sense_vocab2.json", 'w' ) )

json.dump( tag_len2, open( "sense_vocab3.json", 'w' ) )


# In[ ]:


# embedding_matrix = np.zeros((len(word_len) + 1, 128))
# for word, i in word_len.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector

EMBEDDING_DIM = embeddings_index.get('a').shape[0]
num_words = min(len(word_len), len(word_len)) + 1
embedding_matrix = np.zeros((len(word_len), EMBEDDING_DIM))
for word, i in word_len.items():
    if i > len(word_len):
        continue
    embedding_vector = embeddings_index.get(word) ## This references the loaded embeddings dictionary
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
MAX_LENGTH = len(max(train_sentences_X, key=len))
print(MAX_LENGTH)  # 27


# a = np.array(train_sentences_X)
# print(a.shape)


# In[ ]:


from keras.preprocessing.sequence import pad_sequences
 
train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')

train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
train_tags_y2 = pad_sequences(train_tags_y2, maxlen=MAX_LENGTH, padding='post')
train_tags_y3 = pad_sequences(train_tags_y3, maxlen=MAX_LENGTH, padding='post')

test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y2 = pad_sequences(test_tags_y2, maxlen=MAX_LENGTH, padding='post')
test_tags_y3 = pad_sequences(test_tags_y3, maxlen=MAX_LENGTH, padding='post')
 
print(train_sentences_X[0])
print(train_tags_y[0])
print(train_tags_y2[0])
print(train_tags_y3[0])
print("Test")
print(test_sentences_X[0])
print(test_tags_y[0])
print(test_tags_y2[0])
print(test_tags_y3[0])


# In[ ]:


from numpy import newaxis

reshaped_train_tags_y = train_tags_y[:, :, newaxis]
reshaped_train_tags_y.shape

reshaped_train_tags_y2 = train_tags_y2[:, :, newaxis]
reshaped_train_tags_y2.shape

reshaped_train_tags_y3 = train_tags_y3[:, :, newaxis]
reshaped_train_tags_y3.shape

reshaped_test_tags_y = test_tags_y[:, :, newaxis]
reshaped_test_tags_y.shape

reshaped_test_tags_y2 = test_tags_y2[:, :, newaxis]
reshaped_test_tags_y2.shape

reshaped_test_tags_y3 = test_tags_y3[:, :, newaxis]
reshaped_test_tags_y3.shape


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation,concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
 
visible = Input(shape=(None,))
# x = Embedding(len(word_len),128)(visible)
x = Embedding(len(word_len),EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_LENGTH,
                            trainable=False)(visible)
lstm_out = Bidirectional(LSTM(512, return_sequences=True,dropout=0.3),merge_mode="sum")(x)
lstm_out2 = Bidirectional(LSTM(512, return_sequences=True,dropout=0.3),merge_mode="sum")(lstm_out)
output1 = TimeDistributed(Dense(len(tag_len),activation='softmax'))(lstm_out2)
output2 = TimeDistributed(Dense(len(tag_len2),activation='softmax'))(lstm_out)
output3 = TimeDistributed(Dense(len(tag_len3),activation='softmax'))(lstm_out)
model = Model(inputs=visible, outputs=[output1,output2,output3])

# model = Sequential()
# model.add(InputLayer(input_shape=(MAX_LENGTH, )))
# model.add(Embedding(len(word_len), 128))
# # model.add(ElmoEmbeddingLayer())
# model.add(Bidirectional(LSTM(256, return_sequences=True)))
# model.add(TimeDistributed(Dense(len(tag_len))))
# model.add(Activation('softmax'))
 
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(0.001))

filepath="model_wsd_senseless.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
 
model.summary()


# In[ ]:


# history = model.fit(train_sentences_X, [reshaped_train_tags_y,reshaped_train_tags_y2,reshaped_train_tags_y3], batch_size=32, epochs=5,callbacks=callbacks_list,validation_data=(test_sentences_X, [reshaped_test_tags_y,reshaped_test_tags_y2,reshaped_test_tags_y3]))
history = model.fit(train_sentences_X,[reshaped_train_tags_y,reshaped_train_tags_y2,reshaped_train_tags_y3], batch_size=64, epochs=10,callbacks=callbacks_list,shuffle=True,validation_data=(test_sentences_X,[reshaped_test_tags_y,reshaped_test_tags_y2,reshaped_test_tags_y3]))


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss+-
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[ ]:


# from IPython.display import FileLink
# FileLink('model_wsd_senseless.hdf5')


# In[ ]:


# from IPython.display import FileLink
# FileLink('sense_vocab.json')


# In[ ]:


# from IPython.display import FileLink
# FileLink('words_vocab.json')

