#!/usr/bin/env python
# coding: utf-8

# # DeepCTCFLoop getMotifs

# based on https://github.com/BioDataLearning/DeepCTCFLoop/

# ## Dataset : common cell types GM12878, K562, HeLa

# In[ ]:


import os
print(os.listdir('../input/ctcfloop-data/Data'))


# ## utils

# In[ ]:


from Bio import SeqIO
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras import backend as K
import tensorflow as tf
np.random.seed(12345)


# In[ ]:


def get_num(fasta_name, fasta_name2):
    num = 0
    for seq_record in SeqIO.parse(fasta_name,"fasta"):
        if(not(re.search('N',str(seq_record.seq.upper())))):
            num+=1
    for seq_record2 in SeqIO.parse(fasta_name2,"fasta"):
        if(not(re.search('N',str(seq_record2.seq.upper())))):
            num+=1
    return num


# In[ ]:


'''Convert the input sequences into binary matrixs'''
def get_seq_matrix(fasta_name,seqmatrix,rank): 
    labellist = []
    for seq_record in SeqIO.parse(fasta_name,"fasta"):
        label = seq_record.id
        sequence = seq_record.seq.upper()
        if(re.search('N',str(sequence))):
            continue
        Acode = np.array(get_code(sequence,'A'),dtype=int)
        Tcode = np.array(get_code(sequence,'T'),dtype=int)
        Gcode = np.array(get_code(sequence,'G'),dtype=int)
        Ccode = np.array(get_code(sequence,'C'),dtype=int)
        seqcode = np.vstack((Acode,Tcode,Gcode,Ccode))
        labellist.append(label)
        seqmatrix[rank] = seqcode
        rank +=1
    return seqmatrix,labellist,rank


# In[ ]:


def get_code(seq,nt):
    nts = ['A','T','G','C']
    nts.remove(nt)
    codes = str(seq).replace(nt,str(1))
    for i in range(0,len(nts)):
        codes = codes.replace(nts[i],str(0))
    coding = list(codes)
    for i in range(0,len(coding)):
        coding[i] = float(coding[i])
    return coding


# In[ ]:


'''Get the train, validation and test set from the input'''
def get_data(infile,infile2):
    rank = 0
    num = get_num(infile,infile2)
    seqmatrix = np.zeros((num,4,1038))
    (seqmatrix, poslab, rank) = get_seq_matrix(infile,seqmatrix,rank)
    (seqmatrix, neglab, rank) = get_seq_matrix(infile2,seqmatrix,rank)
    X = seqmatrix
    Y = np.array(poslab + neglab,dtype = int)
    validation_size = 0.10
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation_size)
    return np.transpose(X_train,axes=(0,2,1)), np.transpose(X_val,axes=(0,2,1)), np.transpose(X_test,axes=(0,2,1)), Y_train, Y_val, Y_test


# In[ ]:


'''Calculate ROC AUC during model training, obtained from <https://github.com/nathanshartmann/NILC-at-CWI-2018>'''
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    N = K.sum(1 - y_true)
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    P = K.sum(y_true)
    TP = K.sum(y_pred * y_true)
    return TP/P

def roc_auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)


# In[ ]:


def one_hot_to_seq(matrix):
    nts = ['A','T','G','C']
    seqs = []
    index = [np.where(r==1)[0][0] for r in matrix]
    for i in index:
        seqs.append(nts[i])
    seq = ''.join(seqs)
    return seq


# ## Prepare Dataset

# In[ ]:


infile   = '../input/ctcfloop-data/Data/GM12878_pos_seq.fasta'
secondin = '../input/ctcfloop-data/Data/GM12878_neg_seq.fasta'


# In[ ]:


X_train,X_val,X_test,Y_train,Y_val,Y_test = get_data(infile,secondin)


# In[ ]:


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# ## Build Model

# In[ ]:


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Permute, Lambda
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.layers import multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# In[ ]:


labels = ['negative', 'positive']
label=0


# In[ ]:


dense_unit = 112
droprate_cnn  = 0.4279770967884926
droprate_lstm = 0.05028428952624636
filter_unit = 208
l2_reg = 5.2164660610264974e-05
#learning_rate = 0.00010199140620075788
lstm_unit = 64
pool_size = 4
window_size = 13


# In[ ]:


## Build Model
inputs = Input(shape = (1038, 4))

# 1st Conv1D
cnn_out = Conv1D(filter_unit, window_size,
                 kernel_regularizer=l2(l2_reg),
                 activation="relu")(inputs)
pooling_out = MaxPooling1D(pool_size=pool_size, strides=pool_size)(cnn_out)
dropout1 = Dropout(droprate_cnn)(pooling_out)

# 2nd Conv1D
cnn_out2 = Conv1D(filter_unit, window_size,
                  kernel_regularizer=l2(l2_reg),
                  activation="relu")(dropout1)
pooling_out2 = MaxPooling1D(pool_size=pool_size, strides=pool_size)(cnn_out2)
dropout2 = Dropout(droprate_cnn)(pooling_out2)

# LSTM
lstm_out = Bidirectional(LSTM(lstm_unit, return_sequences=True, 
                              kernel_regularizer=l2(l2_reg)
                             ),merge_mode = 'concat')(dropout2)

# Attention
a = Permute((2, 1))(lstm_out)
a = Dense(61, activation='softmax')(a)
a_probs = Permute((2, 1), name='attention_vec')(a)

attention_out = multiply([lstm_out, a_probs])
attention_out = Lambda(lambda x: K.sum(x, axis=1))(attention_out)
dropout3 = Dropout(droprate_lstm)(attention_out)

# FC
dense_out = Dense(dense_unit, 
                  kernel_regularizer=l2(l2_reg),
                  activation='relu')(dropout3)

output = Dense(1, activation='sigmoid')(dense_out)

model = Model(inputs=[inputs], outputs=output)

model.summary()


# ## Load Model weight

# In[ ]:


model.load_weights('../input/ctcfloop-data/Data/GM12878.bestmodel.hdf5')


# ## Evaluate Model

# In[ ]:


y_pred = model.predict(X_test)
rounded = [round(y[0]) for y in y_pred]


# In[ ]:


cm = confusion_matrix(Y_test, rounded)
print(cm)


# In[ ]:


print(classification_report(Y_test, rounded, target_names=labels))


# In[ ]:


TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
specificity = TN / float( TN + FP)
sensitivity = TP / float(FN + TP)
print('Specificity:',specificity)
print('Sensitivity:',sensitivity)


# ## Visualize Motifs

# In[ ]:


layer_names = [l.name for l in model.layers]
print(layer_names)


# In[ ]:


conv_layer_index = layer_names.index('conv1d_1')
conv_layer = model.layers[conv_layer_index]
num_motifs = conv_layer.filters
window     = conv_layer.kernel_size[0]
conv_output= conv_layer.get_output_at(0)

print(num_motifs)
print(window)


# In[ ]:


X = X_test
Y = Y_test


# In[ ]:


f = K.function([model.input], [K.max(K.max(conv_output, axis=1), axis=0)])
f_seq = K.function([model.input], [K.argmax(conv_output, axis=1), K.max(conv_output, axis=1)])
f_act = K.function([model.input],[conv_output])


# In[ ]:


motifs = np.zeros((num_motifs, window, 4))
nsites = np.zeros(num_motifs)
nseqs = np.zeros(num_motifs)
Y_pos = [i for i,e in enumerate(Y) if e ==label]
X_pos = X[Y_pos]

nsamples = len(X_pos)
mean_acts = np.zeros((num_motifs, nsamples))
print(mean_acts.shape)


# In[ ]:


z = f([X_pos])
max_motif = z[0]
thr_per = 0.5
z_seq = f_seq([X_pos])
max_inds = z_seq[0]
max_acts = z_seq[1]
z_act = f_act([X_pos])
acts = z_act[0]

for m in range(num_motifs):
    for n in range(len(X_pos)):
        if max_acts[n, m] > thr_per*max_motif[m]:
            nseqs[m] +=1


# In[ ]:


output_dir = './'
output_dir2= './'
##get the filter activity and locations on the input sequence
act_file = open(output_dir+'motifs'+str(label)+'_act', 'w')
loc_file = open(output_dir+'motifs'+str(label)+'_loc', 'w')
for m in range(num_motifs):
    for n in range(len(X_pos)):
        for j in range(acts.shape[1]):
            weight = (519-abs(j-519))/519
            mean_acts[m,n] += acts[n,j,m]*weight
            if acts[n, j, m] > thr_per * max_motif[m]:
                nsites[m] += 1
                motifs[m] += X_pos[n, j:j+window, :]
                loc_file.write("M%i %i %i\n" % (m, j, j+window))


# In[ ]:


for m in range(num_motifs):
    act_file.write("M%i" % (m))
    for n in range(len(X_pos)):
        act_file.write("\t%0.4f" % (mean_acts[m,n]))
    act_file.write("\n")


# In[ ]:


for m in range(num_motifs):
    seqfile = open(output_dir2+'motif'+str(m)+'_seq.fasta', 'w')
    for n in range(len(X_pos)):
        for j in range(acts.shape[1]): 
            if acts[n, j, m] > thr_per * max_motif[m]:
                nsites[m] += 1
                motifs[m] += X_pos[n, j:j+window, :]
                kmer = one_hot_to_seq(X_pos[n, j:j+window, :])         
                seqfile.write('>%d_%d' % (n,j))
                seqfile.write('\n')
                seqfile.write(kmer)
                seqfile.write('\n')


# In[ ]:


print('Making motifs')
motifs = motifs[:, :, [0, 3, 2, 1]]
motifs_file = open(output_dir+'motifs'+str(label)+'.txt', 'w')
motifs_file.write('MEME version 4.9.0\n\n'
              'ALPHABET= ACGT\n\n'
              'strands: + -\n\n'
              'Background letter frequencies (from uniform background):\n'
              'A 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n')
for m in range(num_motifs):
    if nsites[m] == 0:
        continue
    motifs_file.write('MOTIF M%i O%i\n' % (m, m))
    motifs_file.write("letter-probability matrix: alength= 4 w= %i nsites= %i nseqs= %i E= 1337.0e-6\n" % (window, nsites[m], nseqs[m]))
    for j in range(window):
        motifs_file.write("%f %f %f %f\n" % tuple(1.0 * motifs[m, j, 0:4] / np.sum(motifs[m, j, 0:4])))
    motifs_file.write('\n')


# In[ ]:




