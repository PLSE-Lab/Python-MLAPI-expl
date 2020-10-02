#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import to_categorical
import re
import time

# Input data fil8es are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/binaryfinal"))

start = time.time()
# Any results you write to the current directory are saved as output.

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield " ".join(l[i:i + n])
           


# In[ ]:


def iterative_levenshtein(s, t):
    """ 
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings 
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein 
        distance between the first i characters of s and the 
        first j characters of t
    """
    rows = len(s)+1
    cols = len(t)+1
    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings 
    # by deletions:
    for i in range(1, rows):
        dist[i][0] = i
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for i in range(1, cols):
        dist[0][i] = i
        
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                 dist[row][col-1] + 1,      # insertion
                                 dist[row-1][col-1] + cost) # substitution
    #for r in range(rows):
    #    print(dist[r])
    
 
    return dist[row][col]


# In[ ]:


batch_size = 256  # Batch size for training.
nbatch_size_train = 1024
nbatch_size = 2048 # [1-2:256]
latent_dim = 1024  # Latent dimensionality of the encoding space.
vocab = 10000
context = 128
equal = False
continuing = True
ordno = 6
beam = 10
noisy = False
noisiness = 0.01 #1-2: 0.0, 3-4 :0.01, 5:0
debug = False
full = True
shuffling = True
testing = True
testing_teacher = False
file_to_model = "binaryfinal"
worktimes = [2,2.15,2.25]
num_samples = 50000  # Number of samples to train on.

dropout_rate = 0.5

uh = "$"
um = "$"    
rep = "$"
pause = "$"


# In[ ]:


# Path to the data txt file on disk.
data_path = '../input/dickens/Combined.txt'
data_path = "../input/micase/MICASE.txt"

# Vectorize the data.
input_texts = []
target_texts = []

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read()


import re
lines = re.sub("\n|\r", " ", lines)
lines  = re.sub("[^.]\n\s*(?!\.)", " .\n", lines)
lines  = re.sub("[^.]\n\s*\.", " .\n", lines)
lines  = re.sub("\n", "", lines)
lines  = re.sub("\?|!", ".", lines)
lines  = re.sub("\.", " .", lines)
lines  = re.sub(" +\.", " .", lines)
lines  = re.sub("\.", ".\n", lines)
lines  = re.sub(",|\;|:", "", lines)
lines  = re.sub('"|-|\(|\)', "", lines)

orig = len(lines.split("\n")) 
orig_w = sum([len(x.split()) for x in lines.split("\n")])
if equal:

    lines = lines.split()
    lines = list(chunks(lines, context))[0:-1]
    
else:
    lines = lines.split("\n")
    bins = np.percentile([len(x.split()) for x in lines], [20, 80])
    lines = [x for x in lines if (len(x.split()) > bins[0]) & (len(x.split()) < bins[1])]
    
    bins = np.histogram([len(x.split()) for x in lines])
       

lines = [x for x in lines if not all([y in ["uhuh", ".", "yeah", "mhmm", "yes"] for y in x.split()])]
lines = [re.sub("-", "",x) for x in lines]

lines = [x for x in lines if len(x.split()) > 2]

print("Datasize: %i/%i sentences" % (len(lines), orig))
print("Datasize: %i/%i words" % (sum([len(x.split()) for x in lines]), orig_w))


# In[ ]:


from keras.preprocessing.text import Tokenizer, text_to_word_sequence

#lines = [x for x in lines if len(x.split())<10 and len(x.split()>5)]
if full == True:
    num_samples = len(lines)-1
    
lines = lines[: min(num_samples, len(lines) - 1)]
from collections import Counter
cnt = Counter()
[cnt.update(x.split()) for x in lines]
non_hapaxes = [x for x in cnt if cnt[x] > 4]
print("Non-hapaxes: %i/%i" % (len(non_hapaxes), len(cnt)))
print("Data: %i" % sum([len(x.split()) for x in lines]))
tokenizer = Tokenizer(min(10000, len(non_hapaxes)), oov_token="<unk>", filters='!"#%()*+,-/:;<>?[\\]^_`{}\t\n')
tokenizer.fit_on_texts(lines + ["@", "|", "$", "~", "=", "&"]*100000)



def filt(line):
    line = re.sub("( |^)(um|uh|speechpause) ", " ", line)
    line = re.sub("\||@|&|$|=|~", "", line)
    line = line.split()+["0"]
    line = [line[x] for x in range(len(line)-1) if line[x] != line[x+1]] # 1 word repetitions
    
    line = line + 4*["0"]    
    line = [" ".join([line[x], line[x+1]]) for x in range(len(line)-1)]

    temp = []
    x = 0
    while x < (len(line)-3):
        if line[x] == line[x+2]:
            del line[x]
            del line[x]           
        else:
            temp.append(line[x].split()[0])
            x += 1

    line = temp
    
    line = line + 6*["0"]
    line = [" ".join([line[x], line[x+1], line[x+2]]) for x in range(len(line)-2)]
    temp = []
    x = 0
    while x < (len(line)-4):
        if line[x] == line[x+3]:
            del line[x]
            del line[x]
            del line[x]            
        else:
            temp.append(line[x].split()[0])
            x += 1
    line = temp            
            
    return(" ".join(line))

def markup(line):
    line = re.sub("\||@|&|$|=|~", "", line)
    
    line = re.sub("( |^)um ", " "+um+" ", line)
    line = re.sub("( |^)uh ", " "+uh+" ", line)
    line = re.sub("( |^)speechpause ", " "+pause+" ", line)
    
    line = line.split()+["0"]
    line = [line[x] if line[x] != line[x+1] else rep for x in range(len(line)-1)]
        
    line = line + 4*["0"]
    line = [" ".join([line[x], line[x+1]]) for x in range(len(line)-1)]
    temp = []
    x = 0
    while x < (len(line)-3):
        if line[x] == line[x+2]:
            del line[x]
            del line[x]
            temp.append(rep)            
        else:
            temp.append(line[x].split()[0])
            x += 1
    line = temp

    line = line + 6*["0"]
    line = [" ".join([line[x], line[x+1], line[x+2]]) for x in range(len(line)-2)]
    temp = []
    x = 0
    while x < (len(line)-4):
        if line[x] == line[x+3]:
            del line[x]
            del line[x]
            del line[x]
            temp.append("$")            
        else:
            temp.append(line[x].split()[0])
            x += 1
    line = temp  
    
    line = " ".join(line)
    line = re.sub("(\$ )+", "$ ", line)
    return(line)

def pad(line, mx): 
    return(line+[0]*(mx-len(line)))

inlines = [filt(x) for x in lines]
outlines = ["| "+markup(x) + " @" for x in lines]

#vc = [x.split() for x in outlines]
#print(len(set([item for sublist in vc for item in sublist])))


inlines = tokenizer.texts_to_sequences(inlines)
inlines = tokenizer.sequences_to_texts(inlines)
print(inlines[0])

inlines = [[x for x in y] for y in inlines]
m_inlinelen = max([len(x) for x in inlines])+1 #+1 for the padding placeholder

outlines = tokenizer.texts_to_sequences(outlines)
outlines = tokenizer.sequences_to_texts(outlines)
print(outlines[0])

outlines = [[x for x in y] for y in outlines]
m_outlinelen = max([len(x) for x in outlines])+1 #+1 for the padding placeholder


# In[ ]:


testlines = [letter for sentence in outlines for letter in sentence]

print("Sentences: %i" % len([x for x in testlines if x == "@"]))
print("uh: %i" % len([x for x in testlines if x == "="]))
print("um: %i" % len([x for x in testlines if x == "$"]))
print("pause: %i" % len([x for x in testlines if x == "~"]))
print("repetition: %i" % len([x for x in testlines if x == "&"]))
print("Data disfluency rate: %i/%i - %.1f%%" % (len([x for x in testlines if x in set(["&","~","$","="])]), sum([len(x.split()) for x in lines]), 100.0*len([x for x in testlines if x in set(["&","~","$","="])])/sum([len(x.split()) for x in lines])))


# In[ ]:


charsIn = set([item for sublist in inlines for item in sublist])
charsOut = set([item for sublist in outlines for item in sublist])

chars = charsIn.union(charsOut)
chars = chars.union(set(["=", "|", "~", "@", "&", "$"]))
vocab = len(chars)+1

charToInt = {}
intToChar = {}
i = 1
intToChar[0] = "_"
charToInt["_"] = 0
for ch in chars:
    charToInt[ch] = i
    intToChar[i] = ch
    i+= 1
    
inlines = [[charToInt[x] for x in y] for y in inlines]
outlines = [[charToInt[x] for x in y] for y in outlines] 

print("Done")
#inlines = np.array([pad(x, m_inlinelen) for x in inlines])
#outlines = np.array([pad(x, m_outlinelen) for x in outlines])


# In[ ]:


import random
i = random.randint(0, len(inlines))
preview = inlines[i]
print("".join([intToChar[x] for x in preview]))
preview = outlines[i]
print("".join([intToChar[x] for x in preview]))


# In[ ]:


preview = [x for x in outlines if (len(x) <30) and (sum([y==charToInt["$"] for y in x]) == 2)]
for i in range(min(20, len(preview))):
    print("".join([intToChar[x] for x in preview[i]]))


# In[ ]:


class inpGen(object):

    def __init__(self, dt, bsize, vsize):
        from keras.utils import to_categorical
        self.dt = dt
        #self.dt = np.array([x[1:] for x in dt])
        self.bsize = bsize
        self.vsize = vsize
        self.index = 0
    def generate(self):
        while True:

            if self.index + self.bsize > len(self.dt):
                self.index = 0
                
            mx = max([len(x) for x in self.dt[self.index:self.index+self.bsize]])
            
            mbatch = np.array([pad(x, mx) for x in self.dt[self.index:self.index+self.bsize]])            
            batch = np.zeros([self.bsize, mbatch.shape[1], self.vsize])
            
            for i in range(self.bsize):
                batch[i,:,:] = to_categorical(mbatch[i,:], num_classes=self.vsize)
            self.index += self.bsize
            yield batch
            
class tarGen(object):

    def __init__(self, dt, bsize, vsize):
        from keras.utils import to_categorical   
        self.dt = [x[1:] for x in dt]
        #self.dt = np.array([x[1:] for x in dt])
        self.bsize = bsize
        self.vsize = vsize
        self.index = 0
        
    def generate(self):
        while True:
            if self.index + self.bsize > len(self.dt):
                self.index = 0
                
            mx = max([len(x) for x in self.dt[self.index:self.index+self.bsize]])
            
            mbatch = np.array([pad(x, mx) for x in self.dt[self.index:self.index+self.bsize]])            
            batch = np.zeros([self.bsize, mbatch.shape[1], self.vsize])
            
            for i in range(self.bsize):
                batch[i,:,:] = to_categorical(mbatch[i,:], num_classes=self.vsize)
            self.index += self.bsize
            
            yield batch            
    
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(inlines, outlines, random_state=1991, test_size=0.2)

testX, finalX, testY, finalY = train_test_split(testX, testY, random_state=1987, test_size=0.5)
print("Train - valid - test: %i/%i/%i" % (len(trainX),len(testX),len(finalX)))
print("Train - valid - test: %.1f/%.1f/%.1f" % (len(trainX)/num_samples,len(testX)/num_samples,len(finalX)/num_samples))

tg_tr = tarGen(trainY, batch_size, vocab)
inpen_tr = inpGen(trainX, batch_size, vocab)
inpde_tr = inpGen(trainY, batch_size, vocab)

ien = inpen_tr.generate()
ide = inpde_tr.generate()
o = tg_tr.generate()

tg_val = tarGen(testY, batch_size*2, vocab)
inpen_val = inpGen(testX, batch_size*2, vocab)
inpde_val = inpGen(testY, batch_size*2, vocab)

ien_t = inpen_val.generate()
ide_t = inpde_val.generate()
o_t = tg_val.generate()

tg_final = tarGen(finalY, len(finalX), vocab)
inpen_final = inpGen(finalX, len(finalX), vocab)

ien_f = inpen_final.generate()
o_f = tg_final.generate()

#dec_tokenizer.text_to_word_sequence("\n".join(lines[: min(num_samples, len(lines) - 1)]))


# In[ ]:


from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Masking, GaussianNoise, Dropout, Bidirectional, CuDNNGRU, Concatenate
from keras.optimizers import RMSprop

import gc
gc.collect()

#Shared_Embedding = Embedding(vocab, latent_dim, mask_zero=True)
encoder_inputs = Input(shape=(None, vocab))
encoder_layer1 = Bidirectional(CuDNNGRU(int(latent_dim/2), return_state=True, return_sequences=True))
encoder_layer2 = CuDNNGRU(latent_dim, return_state=True, return_sequences=True)
encoder_layer3 = CuDNNGRU(latent_dim, return_state=True)

if noisy == True:        
    noisy_encoder_inputs = GaussianNoise(noisiness)(encoder_inputs)
    encoder_outputs_layer1, enc_state_h_layer1_forward, enc_state_h_layer1_backward = encoder_layer1(noisy_encoder_inputs)
    encoder_outputs_layer2, enc_state_h_layer2 = encoder_layer2(encoder_outputs_layer1)
    encoder_outputs_layer3, enc_state_h_layer3 = encoder_layer3(encoder_outputs_layer2)
else:
    encoder_outputs_layer1,  enc_state_h_layer1_forward, enc_state_h_layer1_backward = encoder_layer1(encoder_inputs)
    encoder_outputs_layer2, enc_state_h_layer2 = encoder_layer2(encoder_outputs_layer1)
    encoder_outputs_layer3, enc_state_h_layer3 = encoder_layer3(encoder_outputs_layer2)
    
enc_state_h_layer1 = Concatenate()([enc_state_h_layer1_forward, enc_state_h_layer1_backward])    
# We discard `encoder_outputs` and only keep the states.
encoder_states = [enc_state_h_layer1, enc_state_h_layer2,enc_state_h_layer3]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, vocab))
decoder_layer1 = CuDNNGRU(latent_dim, return_sequences=True,return_state=True)
decoder_layer2 = CuDNNGRU(latent_dim, return_sequences=True,return_state=True)
decoder_layer3 = CuDNNGRU(latent_dim, return_sequences=True,return_state=True)


if noisy == True:
    noisy_decoder_inputs = GaussianNoise(noisiness)(decoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the 
# return states in the training model, but we will use them in inference.
    decoder_outputs_layer1, _ = decoder_layer1(noisy_decoder_inputs,
                                     initial_state=enc_state_h_layer1)
    decoder_outputs_layer2, _ = decoder_layer2(decoder_outputs_layer1,
                                     initial_state=enc_state_h_layer2)    
    decoder_outputs_layer3, _ = decoder_layer3(decoder_outputs_layer2,
                                     initial_state=enc_state_h_layer3)    
    
else:
    decoder_outputs_layer1, _ = decoder_layer1(decoder_inputs,
                                     initial_state=enc_state_h_layer1)
    decoder_outputs_layer2, _ = decoder_layer2(decoder_outputs_layer1,
                                     initial_state=enc_state_h_layer2)    
    decoder_outputs_layer3, _ = decoder_layer3(decoder_outputs_layer2,
                                     initial_state=enc_state_h_layer3)    

decoder_dropout = Dropout(rate=dropout_rate)
decoder_outputs_layer4 = decoder_dropout(decoder_outputs_layer3)
decoder_dense = Dense(vocab, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs_layer4)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

vallos = []
valacc = []
trainlos = []
f1uh = []
f1um = []
f1rep = []
f1pause = []
f1disf = []

performances = {"vallos": [],
               "valacc": [],
                "trainlos": [],
                "f1uh": [],
                "f1um": [],
                "f1rep": [],
                "f1pause": [],
                "f1disf": []
               }

if continuing == True:
    
    model.load_weights("../input/"+file_to_model+"/model_weights" + str(ordno-1) + ".h5")
    import pickle
    with open("../input/"+file_to_model+"/ien" + str(ordno-1) + ".pkl", "rb") as f:
        inpen_tr = pickle.load(f) 
        inpen_tr.bsize = nbatch_size_train
    with open("../input/"+file_to_model+"/ide" + str(ordno-1) + ".pkl", "rb") as f:
        inpde_tr = pickle.load(f)
        inpde_tr.bsize = nbatch_size_train
    with open("../input/"+file_to_model+"/o" + str(ordno-1) + ".pkl", "rb") as f:
        tg_tr = pickle.load(f)
        tg_tr.bsize = nbatch_size_train
    with open("../input/"+file_to_model+"/ien_t" + str(ordno-1) + ".pkl", "rb") as f:
        inpen_val = pickle.load(f)
        inpen_val.bsize = nbatch_size
    with open("../input/"+file_to_model+"/ide_t" + str(ordno-1) + ".pkl", "rb") as f:
        inpde_val = pickle.load(f)
        inpde_val.bsize = nbatch_size
    with open("../input/"+file_to_model+"/o_t" + str(ordno-1) + ".pkl", "rb") as f:
        tg_val = pickle.load(f)
        tg_val.bsize = nbatch_size
        
    with open("../input/"+file_to_model+"/ien_f" + str(ordno-1) + ".pkl", "rb") as f:
        inpen_final = pickle.load(f)
    with open("../input/"+file_to_model+"/o_f" + str(ordno-1) + ".pkl", "rb") as f:       
        tg_final = pickle.load(f)
    
    with open("../input/"+file_to_model+"/tokenizer" + str(ordno-1) + ".pkl", "rb") as f:
        tokenizer = pickle.load(f)        
    with open("../input/"+file_to_model+"/charToInt.pkl", "rb") as f:
        charToInt = pickle.load(f)  
    with open("../input/"+file_to_model+"/intToChar.pkl", "rb") as f:
        intToChar = pickle.load(f)          
        
    with open("../input/"+file_to_model+"/performances.pkl", "rb") as f:
        performances = pickle.load(f)    
                
    if shuffling == True:
        inds = [x for x in range(len(inpen_tr.dt))]
        from random import shuffle
    
        shuffle(inds)
        inpen_tr.dt = [inpen_tr.dt[x] for x in inds]
        inpde_tr.dt = [inpde_tr.dt[x] for x in inds]
        tg_tr.dt = [tg_tr.dt[x] for x in inds]
    
    ien = inpen_tr.generate()
    ide = inpde_tr.generate()
    o = tg_tr.generate()    
    
    ien_t = inpen_val.generate()
    ide_t = inpde_val.generate()
    o_t = tg_val.generate()   

    ien_f = inpen_final.generate()
    o_f = tg_final.generate()        
    
# Run training
model.compile(optimizer=RMSprop(lr=(1/(1+0.00001*len(performances["trainlos"])))*0.001, rho=0.9, epsilon=None, decay=0.00001), loss='categorical_crossentropy')


# In[ ]:


print(model.summary())


# enc1 = model.layers[1]
# enc2 = model.layers[5]
# enc3 = model.layers[7]
# 
# dropout_enc1 = Dropout(rate=dropout_rate)
# dropout_enc2 = Dropout(rate=dropout_rate)
# 
# if noisy == True:        
#     encoder_outputs_layer1, enc_state_h_layer1_forward, enc_state_h_layer1_backward = enc1.output
#     encoder_outputs_layer1 = dropout_enc1(encoder_outputs_layer1)
#     encoder_outputs_layer2, enc_state_h_layer2 = encoder_layer2(encoder_outputs_layer1)
#     encoder_outputs_layer2 = dropout_enc2(encoder_outputs_layer2)
#     encoder_outputs_layer3, enc_state_h_layer3 = encoder_layer3(encoder_outputs_layer2)
#     
# else:
#     encoder_outputs_layer1,  enc_state_h_layer1_forward, enc_state_h_layer1_backward = enc1.output
#     encoder_outputs_layer1 = dropout_enc1(encoder_outputs_layer1)    
#     encoder_outputs_layer2, enc_state_h_layer2 = encoder_layer2(encoder_outputs_layer1)
#     encoder_outputs_layer2 = dropout_enc2(encoder_outputs_layer2)
#     encoder_outputs_layer3, enc_state_h_layer3 = encoder_layer3(encoder_outputs_layer2)
#     
# enc_state_h_layer1 = Concatenate()([enc_state_h_layer1_forward, enc_state_h_layer1_backward])    
# # We discard `encoder_outputs` and only keep the states.
# encoder_states = [enc_state_h_layer1, enc_state_h_layer2, enc_state_h_layer3]
# 
# #dec1 = model.layers[4]
# dec2 = model.layers[6]
# 
# dropout_dec1 = Dropout(rate=dropout_rate)
# dropout_dec2 = Dropout(rate=dropout_rate)
# 
# if noisy == True:
#     noisy_decoder_inputs = GaussianNoise(noisiness)(decoder_inputs)
# # We set up our decoder to return full output sequences,
# # and to return internal states as well. We don't use the 
# # return states in the training model, but we will use them in inference.
#     decoder_outputs_layer1, _ = model.layers[4].get_output_at(-1)
#     decoder_outputs_layer1 = dropout_dec1(decoder_outputs_layer1)
#     decoder_outputs_layer2, _ = decoder_layer2(decoder_outputs_layer1,
#                                      initial_state=enc_state_h_layer2)
#     decoder_outputs_layer2 = dropout_dec2(decoder_outputs_layer2)    
#     
#     decoder_outputs_layer3, _ = decoder_layer3(decoder_outputs_layer2,
#                                      initial_state=enc_state_h_layer3)    
#     
# else:
#     decoder_outputs_layer1, _ = model.layers[4].get_output_at(-1)
#     decoder_outputs_layer1 = dropout_dec1(decoder_outputs_layer1)
#     decoder_outputs_layer2, _ = decoder_layer2(decoder_outputs_layer1,
#                                      initial_state=enc_state_h_layer2)  
#     decoder_outputs_layer2 = dropout_dec2(decoder_outputs_layer2)        
#     decoder_outputs_layer3, _ = decoder_layer3(decoder_outputs_layer2,
#                                      initial_state=enc_state_h_layer3)    
# 
# decoder_dropout = Dropout(rate=dropout_rate)
# decoder_outputs_layer4 = decoder_dropout(decoder_outputs_layer3)
# decoder_final = model.layers[-1]
# 
# decoder_outputs = decoder_final(decoder_outputs_layer4)    
#     
# model2 = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# model2.compile(optimizer=RMSprop(lr=(1/(1+0.00001*len(performances["trainlos"])))*0.001, rho=0.9, epsilon=None, decay=0.00001), loss='categorical_crossentropy')
# 

# print(model2.summary())

# In[ ]:


from matplotlib import pyplot as plt
plt.plot(performances["trainlos"])
plt.show()


# In[ ]:


from IPython.display import SVG
from keras.utils import plot_model
plot_model(model, to_file='modelTrain.png', show_layer_names=False)
#SVG(model_to_dot(model).create(prog='dot', format='svg'))


# from IPython.display import SVG
# from keras.utils import plot_model
# plot_model(model2, to_file='model2Train.png', show_layer_names=False)

# In[ ]:


#debug = True


# print("".join([intToChar[x] for x in np.argmax(i_x[0,0:10,:],1)]))
# print("".join([intToChar[x] for x in np.argmax(o_x[0,0:10,:],1)]))
# print("".join([intToChar[x] for x in np.argmax(d_x[0,0:10,:],1)]))

# In[ ]:


if (debug == True) and (testing == False):
    times = [0.05, 0.005, 0.01]
else:
    if testing == True:
        if debug == True:
            times = [0, 0.005, 0.01]
        elif not testing_teacher:
            times = [0, 3, 5]
        else:
            times = [0,0,0]
    else:
        times = worktimes
        
start = time.time()


# In[ ]:


# WITHOUT DROPOUT
from keras.metrics import categorical_accuracy

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

i = -1    #print((time.time()-start)/3600)
while ((time.time()-start)/3600) < times[0]:
    i += 1
    if i%200 !=0:
        if i%50 == 0:
            l = model.train_on_batch([next(ien), next(ide)[:,:-1]], next(o))
            trainlos.append(l)
            print("\tBatch %i\t loss %f" % (i,l))
            gc.collect()
        else:
            l = model.train_on_batch([next(ien), next(ide)[:,:-1]], next(o))
            trainlos.append(l)

    else:
      #  if i%100 == 0:
      #      from keras.metrics import categorical_accuracy
      #      y_pred =  model.predict_on_batch([next(ien_t), next(ide_t)])
      #      print("Accuracy at batch %i: %f" % (i, categorical_accuracy(next(o_t), y_pred))) 
       # else:
        from IPython.display import clear_output
        clear_output()
        if shuffling == True:
            inds = [x for x in range(len(inpen_tr.dt))]
            from random import shuffle
    
            shuffle(inds)
            inpen_tr.dt = [inpen_tr.dt[x] for x in inds]
            inpde_tr.dt = [inpde_tr.dt[x] for x in inds]
            tg_tr.dt = [tg_tr.dt[x] for x in inds]
            ien = inpen_tr.generate()
            ide = inpde_tr.generate()
            o = tg_tr.generate()
            
        l = model.test_on_batch([next(ien_t), next(ide_t)[:,:-1]], next(o_t))
        vallos.append(l)
        print("ValLoss at batch %i: %f" % (i, l))  
        preds = model.predict_on_batch([next(ien_t), next(ide_t)[:,:-1]])

        y_pred = []
        y_txt = []
        for x in range(len(preds)):
            pred = [np.argmax(x) for x in preds[x]]    
            stop = charToInt["@"]
            if stop in pred:
                st = [x for x in range(len(pred)) if pred[x] == stop]
                pred = pred[0:st[0]+1]+[0]*(len(pred)-(st[0]+1))
            else:
                st = [len(pred)-1]
            y_pred.append(pred)
            y_txt.append(pred[0:st[0]+1])
        
        truths = next(o_t)
        y_true = []
        y_ttxt = []
        
        for x in range(len(truths)):
            pred = [np.argmax(x) for x in truths[x]]
            stop = charToInt["@"]
            if stop in pred:
                st = [x for x in range(len(pred)) if pred[x] == stop]
                y_ttxt.append(pred[0:st[0]+1])
            else:
                st = [len(pred)-1]
                y_ttxt.append(pred[0:st[0]+1])                   
            y_true.append(pred)

        from sklearn.metrics import accuracy_score
        acc = accuracy_score([item for sublist in y_true for item in sublist], [item for sublist in y_pred for item in sublist])

        uh = charToInt["="]
        um = charToInt["$"]    
        rep = charToInt["&"] 
        pause = charToInt["~"]
        
        f1_uh = f1_score([x==uh for x in [item for sublist in y_true for item in sublist]], [x==uh for x in [item for sublist in y_pred for item in sublist]])
        f1_um = f1_score([x==um for x in [item for sublist in y_true for item in sublist]], [x==um for x in [item for sublist in y_pred for item in sublist]])        
        f1_rep = f1_score([x==rep for x in [item for sublist in y_true for item in sublist]], [x==rep for x in [item for sublist in y_pred for item in sublist]])        
        f1_pause = f1_score([x==pause for x in [item for sublist in y_true for item in sublist]], [x==pause for x in [item for sublist in y_pred for item in sublist]])        
        f1_disf = f1_score([x in [rep, uh, um, pause] for x in [item for sublist in y_true for item in sublist]], [x in [rep, uh, um, pause] for x in [item for sublist in y_pred for item in sublist]])
        
        f1uh.append(f1_uh)
        f1um.append(f1_um)
        f1rep.append(f1_rep)
        f1pause.append(f1_pause)
        f1disf.append(f1_disf)
        
        valacc.append(acc)
        print("\tValAcc:   %f" % (acc))
        print("\tF1-uh:    %f" % (f1_uh))
        print("\tF1-um:    %f" % (f1_um))
        print("\tF1-rep:   %f" % (f1_rep))
        print("\tF1-pause: %f" % (f1_pause))
        print("__________________________")
        print("\tF1-disf:  %f" % (f1_disf))
        
        print(["".join([intToChar[x] for x in y]) for y in y_txt[0:2]])
        print(["".join([intToChar[x] for x in y]) for y in y_ttxt[0:2]])
        
        plt.plot(trainlos)
        plt.title("Training loss")
        plt.show()

        plt.plot(valacc)
        plt.title("Validation acc")
        plt.show()

        
        #print("TrainLoss at batch %i: %f" % (i, model.test_on_batch([next(ien), next(ide)], next(o))))  
        #steps=len(inlines)//batch_size,)
                   # ,
#validation_split=0.2)



# class inpGen(object):
# 
#     def __init__(self, dt, bsize, vsize):
#         from keras.utils import to_categorical
#         self.dt = dt
#         #self.dt = np.array([x[1:] for x in dt])
#         self.bsize = bsize
#         self.vsize = vsize
#         self.index = 0
#     def generate(self):
#         while True:
# 
#             if self.index + self.bsize > len(self.dt):
#                 self.index = 0
#                 
#             mx = max([len(x) for x in self.dt[self.index:self.index+self.bsize]])
#             
#             mbatch = np.array([pad(x, mx) for x in self.dt[self.index:self.index+self.bsize]])            
#             batch = np.zeros([self.bsize, mbatch.shape[1], self.vsize])
#             
#             for i in range(self.bsize):
#                 batch[i,:,:] = to_categorical(mbatch[i,:], num_classes=self.vsize)
#             self.index += self.bsize
#             yield batch
#             
# class tarGen(object):
# 
#     def __init__(self, dt, bsize, vsize):
#         from keras.utils import to_categorical   
#         self.dt = [x[1:] for x in dt]
#         #self.dt = np.array([x[1:] for x in dt])
#         self.bsize = bsize
#         self.vsize = vsize
#         self.index = 0
#         
#     def generate(self):
#         while True:
#             if self.index + self.bsize > len(self.dt):
#                 self.index = 0
#                 
#             mx = max([len(x) for x in self.dt[self.index:self.index+self.bsize]])
#             
#             mbatch = np.array([pad(x, mx) for x in self.dt[self.index:self.index+self.bsize]])            
#             batch = np.zeros([self.bsize, mbatch.shape[1], self.vsize])
#             
#             for i in range(self.bsize):
#                 batch[i,:,:] = to_categorical(mbatch[i,:], num_classes=self.vsize)
#             self.index += self.bsize
#             
#             yield batch
#             
# 
# finalY = [[inpen_final.dt[0][0]]+x for x in tg_final.dt]
# inpde_final = inpGen(finalY, tg_final.bsize, tg_final.vsize)
# 
# inpen_final.bsize = 1141
# inpde_final.bsize = 1141
# tg_final.bsize = 1141
# 
# ien_f = inpen_final.generate()
# ide_f = inpde_final.generate()
# o_f = tg_final.generate()   
# 
# 

# len(inpen_final.dt)

#     acc = []
#     f1_uh = []
#     f1_um = []
#     f1_rep = []
#     f1_pause = []
#     f1_disf = []
#     for i in range(7):  
#  #   for i in range(int(round(len(inpde_final.dt)/inpde_final.bsize))):
#         l = model.test_on_batch([next(ien_f), next(ide_f)[:,:-1]], next(o_f))
#         vallos.append(l)
#         print("ValLoss at batch %i: %f" % (i, l))  
#         preds = model.predict_on_batch([next(ien_f), next(ide_f)[:,:-1]])
# 
#         y_pred = []
#         y_txt = []
#         for x in range(len(preds)):
#             pred = [np.argmax(x) for x in preds[x]]    
#             stop = charToInt["@"]
#             if stop in pred:
#                 st = [x for x in range(len(pred)) if pred[x] == stop]
#                 pred = pred[0:st[0]+1]+[0]*(len(pred)-(st[0]+1))
#             else:
#                 st = [len(pred)-1]
#             y_pred.append(pred)
#             y_txt.append(pred[0:st[0]+1])
#         
#         truths = next(o_f)
#         y_true = []
#         y_ttxt = []
#         
#         for x in range(len(truths)):
#             pred = [np.argmax(x) for x in truths[x]]
#             stop = charToInt["@"]
#             if stop in pred:
#                 st = [x for x in range(len(pred)) if pred[x] == stop]
#                 y_ttxt.append(pred[0:st[0]+1])
#             else:
#                 st = [len(pred)-1]
#                 y_ttxt.append(pred[0:st[0]+1])                   
#             y_true.append(pred)
# 
#         from sklearn.metrics import accuracy_score
#         acc.append(accuracy_score([item for sublist in y_true for item in sublist], [item for sublist in y_pred for item in sublist]))
# 
#         uh = charToInt["="]
#         um = charToInt["$"]    
#         rep = charToInt["&"] 
#         pause = charToInt["~"]
#         
#         f1_uh.append(f1_score([x==uh for x in [item for sublist in y_true for item in sublist]], [x==uh for x in [item for sublist in y_pred for item in sublist]]))
#         f1_um.append(f1_score([x==um for x in [item for sublist in y_true for item in sublist]], [x==um for x in [item for sublist in y_pred for item in sublist]]))  
#         f1_rep.append(f1_score([x==rep for x in [item for sublist in y_true for item in sublist]], [x==rep for x in [item for sublist in y_pred for item in sublist]]))       
#         f1_pause.append(f1_score([x==pause for x in [item for sublist in y_true for item in sublist]], [x==pause for x in [item for sublist in y_pred for item in sublist]]))        
#         f1_disf.append(f1_score([x in [rep, uh, um, pause] for x in [item for sublist in y_true for item in sublist]], [x in [rep, uh, um, pause] for x in [item for sublist in y_pred for item in sublist]]))
#                 

#     print("ACC: %f" % ((sum(acc)/len(acc))*100.0))
#     print("F1_uh: %f" % ((sum(f1_uh)/len(acc))*100.0))
#     print("f1_um: %f" % ((sum(f1_um)/len(acc))*100.0))
#     print("f1_rep: %f" % ((sum(f1_rep)/len(acc))*100.0))
#     print("f1_pause: %f" % ((sum(f1_pause)/len(acc))*100.0))
#     print("f1_disf: %f" % ((sum(f1_disf)/len(acc))*100.0))

#     print("ACC: {}".format(acc))
#     print("F1_uh: {}".format(f1_uh))
#     print("f1_um: {}".format(f1_um))
#     print("f1_rep: {}".format(f1_rep))
#     print("f1_pause: {}".format(f1_pause))
#     print("f1_disf: {}".format(f1_disf))
# 

# # WITH DROPOUT
# from keras.metrics import categorical_accuracy
# 
# from sklearn.metrics import f1_score
# import matplotlib.pyplot as plt
# 
# i = -1    #print((time.time()-start)/3600)
# while ((time.time()-start)/3600) < times[0]:
#     i += 1
#     if i%200 !=0:
#         if i%50 == 0:
#             l = model2.train_on_batch([next(ien), next(ide)[:,:-1]], next(o))
#             trainlos.append(l)
#             print("\tBatch %i\t loss %f" % (i,l))
#             gc.collect()
#         else:
#             l = model2.train_on_batch([next(ien), next(ide)[:,:-1]], next(o))
#             trainlos.append(l)
# 
#     else:
#       #  if i%100 == 0:
#       #      from keras.metrics import categorical_accuracy
#       #      y_pred =  model.predict_on_batch([next(ien_t), next(ide_t)])
#       #      print("Accuracy at batch %i: %f" % (i, categorical_accuracy(next(o_t), y_pred))) 
#        # else:
#         from IPython.display import clear_output
#         clear_output()
#         if shuffling == True:
#             inds = [x for x in range(len(inpen_tr.dt))]
#             from random import shuffle
#     
#             shuffle(inds)
#             inpen_tr.dt = [inpen_tr.dt[x] for x in inds]
#             inpde_tr.dt = [inpde_tr.dt[x] for x in inds]
#             tg_tr.dt = [tg_tr.dt[x] for x in inds]
#             ien = inpen_tr.generate()
#             ide = inpde_tr.generate()
#             o = tg_tr.generate()
#             
#         l = model2.test_on_batch([next(ien_t), next(ide_t)[:,:-1]], next(o_t))
#         vallos.append(l)
#         print("ValLoss at batch %i: %f" % (i, l))  
#         preds = model2.predict_on_batch([next(ien_t), next(ide_t)[:,:-1]])
# 
#         y_pred = []
#         y_txt = []
#         for x in range(len(preds)):
#             pred = [np.argmax(x) for x in preds[x]]    
#             stop = charToInt["@"]
#             if stop in pred:
#                 st = [x for x in range(len(pred)) if pred[x] == stop]
#                 pred = pred[0:st[0]+1]+[0]*(len(pred)-(st[0]+1))
#             else:
#                 st = [len(pred)-1]
#             y_pred.append(pred)
#             y_txt.append(pred[0:st[0]+1])
#         
#         truths = next(o_t)
#         y_true = []
#         y_ttxt = []
#         
#         for x in range(len(truths)):
#             pred = [np.argmax(x) for x in truths[x]]
#             stop = charToInt["@"]
#             if stop in pred:
#                 st = [x for x in range(len(pred)) if pred[x] == stop]
#                 y_ttxt.append(pred[0:st[0]+1])
#             else:
#                 st = [len(pred)-1]
#                 y_ttxt.append(pred[0:st[0]+1])                   
#             y_true.append(pred)
# 
#         from sklearn.metrics import accuracy_score
#         acc = accuracy_score([item for sublist in y_true for item in sublist], [item for sublist in y_pred for item in sublist])
# 
#         uh = charToInt["="]
#         um = charToInt["$"]    
#         rep = charToInt["&"] 
#         pause = charToInt["~"]
#         
#         f1_uh = f1_score([x==uh for x in [item for sublist in y_true for item in sublist]], [x==uh for x in [item for sublist in y_pred for item in sublist]])
#         f1_um = f1_score([x==um for x in [item for sublist in y_true for item in sublist]], [x==um for x in [item for sublist in y_pred for item in sublist]])        
#         f1_rep = f1_score([x==rep for x in [item for sublist in y_true for item in sublist]], [x==rep for x in [item for sublist in y_pred for item in sublist]])        
#         f1_pause = f1_score([x==pause for x in [item for sublist in y_true for item in sublist]], [x==pause for x in [item for sublist in y_pred for item in sublist]])        
#         f1_disf = f1_score([x in [rep, uh, um, pause] for x in [item for sublist in y_true for item in sublist]], [x in [rep, uh, um, pause] for x in [item for sublist in y_pred for item in sublist]])
#         
#         f1uh.append(f1_uh)
#         f1um.append(f1_um)
#         f1rep.append(f1_rep)
#         f1pause.append(f1_pause)
#         f1disf.append(f1_disf)
#         
#         valacc.append(acc)
#         print("\tValAcc:   %f" % (acc))
#         print("\tF1-uh:    %f" % (f1_uh))
#         print("\tF1-um:    %f" % (f1_um))
#         print("\tF1-rep:   %f" % (f1_rep))
#         print("\tF1-pause: %f" % (f1_pause))
#         print("__________________________")
#         print("\tF1-disf:  %f" % (f1_disf))
#         
#         print(["".join([intToChar[x] for x in y]) for y in y_txt[0:2]])
#         print(["".join([intToChar[x] for x in y]) for y in y_ttxt[0:2]])
#         
#         plt.plot(trainlos)
#         plt.title("Training loss")
#         plt.show()
# 
#         plt.plot(valacc)
#         plt.title("Validation acc")
#         plt.show()
# 
#         
#         #print("TrainLoss at batch %i: %f" % (i, model.test_on_batch([next(ien), next(ide)], next(o))))  
#         #steps=len(inlines)//batch_size,)
#                    # ,
# #validation_split=0.2)
# 
# 
# 

# In[ ]:


print((time.time()-start)/3600)


# In[ ]:


print(len(performances["trainlos"]))


# In[ ]:


performances["vallos"] += vallos
performances["valacc"] += valacc
performances["trainlos"] += trainlos
performances["f1uh"] += f1uh
performances["f1um"] += f1um
performances["f1rep"] += f1rep
performances["f1pause"] += f1pause
performances["f1disf"] += f1disf


# In[ ]:


from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm


# In[ ]:


if not testing:
    print("Training summary:")


    print("TRAINING LOSS")

    results = OLS(trainlos, sm.add_constant(np.arange(len(trainlos)))).fit()
    print("\tIntercept: %.2f\n\tSlope:%.5f" % (results.params[0]*100,results.params[1]*100))
    plt.plot(trainlos)
    X_plot = np.linspace(0,len(trainlos),100)
    plt.plot(X_plot, X_plot*results.params[1] + results.params[0])

    plt.show()


# In[ ]:


if not testing:
    print("VALIDATION LOSS")

    results = OLS(vallos, sm.add_constant(np.arange(len(vallos)))).fit()
    print("\tIntercept: %.2f\n\tSlope:%.5f" % (results.params[0]*100,results.params[1]*100))
    plt.plot(vallos)
    X_plot = np.linspace(0,len(vallos),100)
    plt.plot(X_plot, X_plot*results.params[1] + results.params[0])

    plt.show()


# In[ ]:


if not testing:
    print("VALIDATION ACCURACY")

    results = OLS(valacc, sm.add_constant(np.arange(len(valacc)))).fit()
    print("\tIntercept: %.2f\n\tSlope:%.5f" % (results.params[0]*100,results.params[1]*100))
    plt.plot(valacc)
    X_plot = np.linspace(0,len(valacc),100)
    plt.plot(X_plot, X_plot*results.params[1] + results.params[0])

    plt.show()


# In[ ]:


if not testing:    
    print("F1 um")

    results = OLS(f1um, sm.add_constant(np.arange(len(f1um)))).fit()
    print("\tIntercept: %.2f\n\tSlope:%.5f" % (results.params[0]*100,results.params[1]*100))
    plt.plot(f1um)
    X_plot = np.linspace(0,len(f1um),100)
    plt.plot(X_plot, X_plot*results.params[1] + results.params[0])

    plt.show()


# In[ ]:


if not testing:   
    print("F1 uh")

    results = OLS(f1uh, sm.add_constant(np.arange(len(f1uh)))).fit()
    print("\tIntercept: %.2f\n\tSlope:%.5f" % (results.params[0]*100,results.params[1]*100))
    plt.plot(f1uh)
    X_plot = np.linspace(0,len(f1uh),100)
    plt.plot(X_plot, X_plot*results.params[1] + results.params[0])

    plt.show()


# In[ ]:


if not testing:    
    print("F1 repetitions")

    results = OLS(f1rep, sm.add_constant(np.arange(len(f1rep)))).fit()
    print("\tIntercept: %.2f\n\tSlope:%.5f" % (results.params[0]*100,results.params[1]*100))
    plt.plot(f1rep)
    X_plot = np.linspace(0,len(f1rep),100)
    plt.plot(X_plot, X_plot*results.params[1] + results.params[0])

    plt.show()


# In[ ]:


if not testing:    
    print("F1 pauses")

    results = OLS(f1pause, sm.add_constant(np.arange(len(f1pause)))).fit()
    print("\tIntercept: %.2f\n\tSlope:%.5f" % (results.params[0]*100,results.params[1]*100))
    plt.plot(f1pause)
    X_plot = np.linspace(0,len(f1pause),100)
    plt.plot(X_plot, X_plot*results.params[1] + results.params[0])

    plt.show()


# In[ ]:


if not testing:    
    print("F1 disfluencies")

    results = OLS(f1disf, sm.add_constant(np.arange(len(f1disf)))).fit()
    print("\tIntercept: %.2f\n\tSlope:%.5f" % (results.params[0]*100,results.params[1]*100))
    plt.plot(f1disf)
    X_plot = np.linspace(0,len(f1disf),100)
    plt.plot(X_plot, X_plot*results.params[1] + results.params[0])

    plt.show()


# In[ ]:


print("Overall F1 disfluencies")
print("After %i updates" % len(performances["trainlos"]))

f1 = performances["f1disf"]

results = OLS(f1, sm.add_constant(np.arange(len(f1)))).fit()
print("\tIntercept: %.2f\n\tSlope:%.5f" % (results.params[0]*100,results.params[1]*100))
plt.plot(f1)
X_plot = np.linspace(0,len(f1),100)
plt.plot(X_plot, X_plot*results.params[1] + results.params[0])

plt.show()


# In[ ]:



l = model.test_on_batch([next(ien_t), next(ide_t)[:,:-1]], next(o_t))
vallos.append(l)
print("ValLoss at batch %i: %f" % (i, l))  
preds = model.predict_on_batch([next(ien_t), next(ide_t)[:,:-1]])

y_pred = []
y_txt = []
for x in range(len(preds)):
    pred = [np.argmax(x) for x in preds[x]]    
    stop = charToInt["@"]
    if stop in pred:
        st = [x for x in range(len(pred)) if pred[x] == stop]
        pred = pred[0:st[0]+1]+[0]*(len(pred)-(st[0]+1))
    else:
        st = [len(pred)-1]
    y_pred.append(pred)
    y_txt.append(pred[0:st[0]+1])

truths = next(o_t)
y_true = []
y_ttxt = []

for x in range(len(truths)):
    pred = [np.argmax(x) for x in truths[x]]
    stop = charToInt["@"]
    if stop in pred:
        st = [x for x in range(len(pred)) if pred[x] == stop]
        y_ttxt.append(pred[0:st[0]+1])
    else:
        st = [len(pred)-1]
        y_ttxt.append(pred[0:st[0]+1])                   
    y_true.append(pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score([item for sublist in y_true for item in sublist], [item for sublist in y_pred for item in sublist])

uh = charToInt["="]
um = charToInt["$"]    
rep = charToInt["&"] 
pause = charToInt["~"]

f1_uh = f1_score([x==uh for x in [item for sublist in y_true for item in sublist]], [x==uh for x in [item for sublist in y_pred for item in sublist]])
f1_um = f1_score([x==um for x in [item for sublist in y_true for item in sublist]], [x==um for x in [item for sublist in y_pred for item in sublist]])        
f1_rep = f1_score([x==rep for x in [item for sublist in y_true for item in sublist]], [x==rep for x in [item for sublist in y_pred for item in sublist]])        
f1_pause = f1_score([x==pause for x in [item for sublist in y_true for item in sublist]], [x==pause for x in [item for sublist in y_pred for item in sublist]])        
f1_disf = f1_score([x in [rep, uh, um, pause] for x in [item for sublist in y_true for item in sublist]], [x in [rep, uh, um, pause] for x in [item for sublist in y_pred for item in sublist]])

f1uh.append(f1_uh)
f1um.append(f1_um)
f1rep.append(f1_rep)
f1pause.append(f1_pause)
f1disf.append(f1_disf)

valacc.append(acc)
print("\tValAcc:   %f" % (acc))
print("\tF1-uh:    %f" % (f1_uh))
print("\tF1-um:    %f" % (f1_um))
print("\tF1-rep:   %f" % (f1_rep))
print("\tF1-pause: %f" % (f1_pause))
print("__________________________")
print("\tF1-disf:  %f" % (f1_disf))

print(["".join([intToChar[x] for x in y]) for y in y_txt[0:2]])
print(["".join([intToChar[x] for x in y]) for y in y_ttxt[0:2]])

#print("TrainLoss at batch %i: %f" % (i, model.test_on_batch([next(ien), next(ide)], next(o))))  
#steps=len(inlines)//batch_size,)


# In[ ]:


if not testing:    
    import pickle
    with open("charToInt.pkl", "wb+") as f:
        pickle.dump(charToInt, f)

    with open("intToChar.pkl", "wb+") as f:
        pickle.dump(intToChar, f) 

    model2.save("model" + str(ordno) + ".h5")
    model2.save_weights("model_weights" + str(ordno) + ".h5")

    with open("ien" + str(ordno) + ".pkl", "wb+") as f:
        pickle.dump(inpen_tr, f)

    with open("ide" + str(ordno) + ".pkl", "wb+") as f:
        pickle.dump(inpde_tr, f)

    with open("o" + str(ordno) + ".pkl", "wb+") as f:
        pickle.dump(tg_tr, f)

    with open("ien_t" + str(ordno) + ".pkl", "wb+") as f:
        pickle.dump(inpen_val, f)

    with open("ide_t" + str(ordno) + ".pkl", "wb+") as f:
        pickle.dump(inpde_val, f)

    with open("o_t" + str(ordno) + ".pkl", "wb+") as f:
        pickle.dump(tg_val, f)    

    with open("ien_f" + str(ordno) + ".pkl", "wb+") as f:
        pickle.dump(inpen_final, f)

    with open("o_f" + str(ordno) + ".pkl", "wb+") as f:
        pickle.dump(tg_final, f)        

    with open("tokenizer" + str(ordno) + ".pkl", "wb+") as f:
        pickle.dump(tokenizer, f)    

    with open("performances.pkl", "wb+") as f:
        pickle.dump(performances, f)        


# In[ ]:


print(model.summary())


# In[ ]:


#WITHOUT DROPOUT

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h_layer1 = Input(shape=(latent_dim,))
decoder_state_input_h_layer2 = Input(shape=(latent_dim,))
decoder_state_input_h_layer3 = Input(shape=(latent_dim,))

decoder_states_inputs = [decoder_state_input_h_layer1,
                        decoder_state_input_h_layer2,
                        decoder_state_input_h_layer3]

#decoder_masked = decoder_masking(decoder_inputs)
#decoder_embedded = decoder_embedding(decoder_masked)
#decoder_embedded = Shared_Embedding(decoder_inputs)
decoder_outputs_layer1, state_h_layer1 = decoder_layer1(decoder_inputs, initial_state=decoder_states_inputs[0])
decoder_outputs_layer2, state_h_layer2 = decoder_layer2(decoder_outputs_layer1, initial_state=decoder_states_inputs[1])    
decoder_outputs_layer3, state_h_layer3 = decoder_layer3(decoder_outputs_layer2, initial_state=decoder_states_inputs[2]) 

decoder_states = [state_h_layer1, state_h_layer2, state_h_layer3]

decoder_outputs = decoder_dense(decoder_outputs_layer3)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


# # WITH DROPOUT
# encoder_model = Model(encoder_inputs, encoder_states)
# 
# decoder_state_input_h_layer1 = Input(shape=(latent_dim,))
# decoder_state_input_h_layer2 = Input(shape=(latent_dim,))
# decoder_state_input_h_layer3 = Input(shape=(latent_dim,))
# 
# decoder_states_inputs = [decoder_state_input_h_layer1,
#                         decoder_state_input_h_layer2,
#                         decoder_state_input_h_layer3]
# 
# #decoder_masked = decoder_masking(decoder_inputs)
# #decoder_embedded = decoder_embedding(decoder_masked)
# #decoder_embedded = Shared_Embedding(decoder_inputs)
# decoder_outputs_layer1, state_h_layer1 = model2.layers[4](decoder_inputs, initial_state=decoder_states_inputs[0])
# decoder_outputs_layer1 = dropout_dec1(decoder_outputs_layer1)
# decoder_outputs_layer2, state_h_layer2 = model2.layers[8](decoder_outputs_layer1, initial_state=decoder_states_inputs[1])    
# decoder_outputs_layer2 = dropout_dec1(decoder_outputs_layer2)
# decoder_outputs_layer3, state_h_layer3 = model2.layers[12](decoder_outputs_layer2, initial_state=decoder_states_inputs[2]) 
# 
# decoder_states = [state_h_layer1, state_h_layer2, state_h_layer3]
# 
# decoder_outputs = decoder_final(decoder_outputs_layer3)
# decoder_model = Model(
#     [decoder_inputs] + decoder_states_inputs,
#     [decoder_outputs] + decoder_states)

# In[ ]:


encoder_model.save("encoder_model" + str(ordno) + ".h5")
encoder_model.save_weights("encoder_model_weights" + str(ordno) + ".h5")

decoder_model.save("decoder_model" + str(ordno) + ".h5")
decoder_model.save_weights("decoder_model_weights" + str(ordno) + ".h5")


# In[ ]:


print(encoder_model.summary())


# In[ ]:


print(decoder_model.summary())


# In[ ]:


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    #print(input_seq.shape)
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1, vocab))
    # Populate the first character of target sequence with the start character.
    target_seq[0,0, charToInt['|']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h_layer1, h_layer2, h_layer3 = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        # sampled_char = tokenizer.sequences_to_texts([[sampled_token_index]])
        # print(sampled_char)
        decoded_sentence.append(sampled_token_index)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_token_index == charToInt['@'] or
            len(decoded_sentence) >= m_outlinelen-1):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1, vocab))
        target_seq[0,0, sampled_token_index] = 1.

        # Update states
        states_value = [h_layer1, h_layer2, h_layer3]

    #decoded_sentence = [argmax(decoded_sentence[1,x,:]) for x in range(len(decoded_sentence[1]))]
    return decoded_sentence


# In[ ]:


def decode_sequence_beam(input_seq, beam=1):
    # Encode the input as state vectors.
    #print(input_seq.shape)
    states_value = encoder_model.predict(input_seq)
    states_value = [states_value for x in range(beam)]
    
    # Generate empty target sequence of length 1.
    target_seq = [np.zeros((1,1, vocab)) for x in range(beam)]
    # Populate the first character of target sequence with the start character.
    for x in range(beam):
        target_seq[x][0,0, charToInt['|']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    stops = set([charToInt['@'], charToInt['.']])
    
    dead_beam = [False]*beam
    beam_prob = [0.5]*beam
    output_tokens = [[]]*beam
    h_layer1 = [[]]*beam    
    h_layer2 = [[]]*beam    
    h_layer3 = [[]]*beam
   
    
    sampled_token_index = [[]]*beam
    decoded_sentence = [[]]*beam
    
    while not stop_condition:
        for x in range(beam):
            #print("_____")
            #print(x)
            #print(output_tokens[x])
            if dead_beam[x] == False:
                output_tokens[x], h_layer1[x], h_layer2[x], h_layer3[x] = decoder_model.predict([target_seq[x]] + states_value[x])
            
        # Sample a token
                sampled_token_index[x] = np.argsort(output_tokens[x][0, -1, :])[-beam:]
                sampled_token_index[x] = [(x, i, beam_prob[x]*(0.001-np.log(output_tokens[x][0,-1,i]))) for i in sampled_token_index[x]]
               # print("___")
                #print(sampled_token_index[x])
                #input()
            else:
                sampled_token_index[x] = [sampled_token_index[x]]
        #prune beams
        sampled_token_index = [item for sublist in sampled_token_index for item in sublist]
        sampled_token_index = sorted(sampled_token_index, key=lambda tup: tup[2])[0:beam]
#        print(sampled_token_index)
        
        for x in range(beam):
            h_layer1[x] = h_layer1[sampled_token_index[x][0]]
            h_layer2[x] = h_layer2[sampled_token_index[x][0]]
            h_layer3[x] = h_layer3[sampled_token_index[x][0]]
            #h[x] = h[sampled_token_index[x][0]]
            #c[x] = c[sampled_token_index[x][0]]
           # print(decoded_sentence)
            if dead_beam[x] == False:
                decoded_sentence[x] = decoded_sentence[sampled_token_index[x][0]] + [sampled_token_index[x][1]]

            beam_prob[x] = sampled_token_index[x][2]
        # sampled_char = tokenizer.sequences_to_texts([[sampled_token_index]])
        # print(sampled_char)
            if sampled_token_index[x][1] in stops:
                dead_beam[x] = True
                
            target_seq[x] = np.zeros((1,1, vocab))
            target_seq[x][0,0, sampled_token_index[x][1]] = 1.

        # Update states
            states_value[x] = [h_layer1[x], h_layer2[x], h_layer3[x]]
            
        if all(dead_beam) or len(decoded_sentence[0]) >= m_outlinelen-1:
            stop_condition = True
        
#        decoded_sentence[].append(sampled_token_index)

            
        # Exit condition: either hit max length
        # or find stop character.
 #       if (sampled_token_index == charToInt['@'] or
  #          len(decoded_sentence) >= m_outlinelen-1):
   #         stop_condition = True

        # Update the target sequence (of length 1).
    #    target_seq = np.zeros((1,1, vocab))
     #   target_seq[0,0, sampled_token_index] = 1.

        # Update states
      #  states_value = [h, c]

    #decoded_sentence = [argmax(decoded_sentence[1,x,:]) for x in range(len(decoded_sentence[1]))]
    return decoded_sentence[np.argmax(beam_prob)]


# In[ ]:


plot_model(decoder_model, to_file='modeltest.png', show_layer_names=False)


# In[ ]:


import gc
#debug = True
y_pred = []
inp = next(ien_f)
if debug == True:
    start = time.time()
    times = [0,0.002,1]
    beam = 5
    
for sent in range(len(inp)):
    #print((time.time()-start)/3600)
    if ((time.time()-start)/3600) > times[1] or len(y_pred) == len(inp):
        break
        #pass
    gc.collect()
    if beam != None:
        predicted = decode_sequence_beam(inp[sent:sent+1], beam=beam)
    else:
        predicted = decode_sequence(inp[sent:sent+1])
    predicted = pad(predicted, m_outlinelen-1)
    y_pred.append(predicted)


# In[ ]:


truths = next(o_f)

y_true = []
y_ttxt = []

for x in range(len(truths)):
   # if ((time.time()-start)/3600) > times[2] or (len(y_true) == len(y_pred)):
    #    break
    pred = pad([np.argmax(x) for x in truths[x]], m_outlinelen-1)
    stop = charToInt["@"]
    if stop in pred:
        st = [x for x in range(len(pred)) if pred[x] == stop]
        y_ttxt.append(pred[0:st[0]+1])
    else:
        st = [len(pred)-1]
        y_ttxt.append(pred[0:st[0]+1])                   
    y_true.append(pred)

mx = min(len(y_true), len(y_pred))
y_true = y_true[:mx]
y_pred = y_pred[:mx]
print("Tested on %i sentences." % mx)


# In[ ]:


i = random.randint(0, min(len(y_pred), len(y_true)))

p=["".join([intToChar[y] for y in x]) for x in y_pred[i:i+min(mx,5)]]
t=["".join([intToChar[y] for y in x]) for x in y_true[i:i+min(mx,5)]]

_ = [[print(x[0]), print(x[1]), print("")] for x in zip(p,t)]


# In[ ]:


def crop(true, pred):

    stops = set([charToInt["@"],charToInt["."]])
    
    true_end = [x for x in range(len(true)) if true[x] in stops]
    pred_end = [x for x in range(len(pred)) if pred[x] in stops]
    
    if true_end == []:
        true_end = [len(true)]
    if pred_end == []:
        pred_end = [len(pred)]
   
    end = max(true_end[0], pred_end[0])
    true = true[:end]
    pred = pred[:end]
    return([true, pred])

precrop = sum([len(x) for x in y_true]) + sum([len(x) for x in y_pred]) 

for x in range(mx):
    y_true[x], y_pred[x] = crop(y_true[x], y_pred[x])
postcrop = sum([len(x) for x in y_true]) + sum([len(x) for x in y_pred])
print("Cropped %.1f%%" % (100*(1-(postcrop/precrop))))


# In[ ]:


#print(y_true[0])
##print(y_pred[0])
#print(pad(y_pred[0], m_outlinelen-1))
#print(len(y_true[0]))
#print(len(y_pred[0]))
#print(len(pad(y_pred[0], m_outlinelen-1)))
#print([[intToChar[x] for x in y] for y in y_pred[0:2]])
#print([[intToChar[x] for x in y] for y in y_true[0:2]])
#m_outlinelen


# In[ ]:


def precision(y_pred, y_true):
    tps = sum([True if (x==y and x==True) else False for x,y in zip(y_pred, y_true)])
    fps = sum([True if (x!=y and x==True) else False for x,y in zip(y_pred, y_true)])
    if (tps + fps) == 0:
        fps = 1
    return(tps/(1.0*(tps+fps)))

def recall(y_pred, y_true):
    tps = sum([True if (x==y and x==True) else False for x,y in zip(y_pred, y_true)])
    fns = sum([True if (x!=y and x==False) else False for x,y in zip(y_pred, y_true)])
    if (tps + fns) == 0:
        fns = 1    
    return(tps/(1.0*(tps+fns)))


# In[ ]:


from re import sub
lv_ready = [[sub("_", "", "".join([intToChar[P] for P in p])), sub("_", "", "".join([intToChar[T] for T in t]))] for p,t in zip(y_pred, y_true)]
lvs = [iterative_levenshtein(p,t)/max(len(p), len(t)) for p,t in lv_ready]


# In[ ]:


print("Levensthein mean: %f" % np.mean(lvs))
print("Levensthein sd: %f" % np.std(lvs))
print("Levensthein median: %f" % np.median(lvs))


# In[ ]:


from sklearn.metrics import accuracy_score, f1_score
acc = accuracy_score([item for sublist in y_true for item in sublist], [item for sublist in y_pred for item in sublist])

uh = charToInt["="]
um = charToInt["$"]    
rep = charToInt["&"] 
pause = charToInt["~"]
        
f1_disf = f1_score([x in [rep, uh, um, pause] for x in [item for sublist in y_true for item in sublist]], [x in [rep, uh, um, pause] for x in [item for sublist in y_pred for item in sublist]])
prec_disf = precision([x in [rep, uh, um, pause] for x in [item for sublist in y_true for item in sublist]], [x in [rep, uh, um, pause] for x in [item for sublist in y_pred for item in sublist]])
rec_disf = recall([x in [rep, uh, um, pause] for x in [item for sublist in y_true for item in sublist]], [x in [rep, uh, um, pause] for x in [item for sublist in y_pred for item in sublist]])



print("\tValAcc:   %f\n" % (acc))

print("\tF1-disf:  %f" % (f1_disf))
print("\tprecision-disf:  %f" % (prec_disf))
print("\trecall-disf:  %f" % (rec_disf))


# In[ ]:




