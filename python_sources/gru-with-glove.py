#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('wget http://nlp.stanford.edu/data/glove.6B.zip')
get_ipython().system('unzip glove.6B.zip -d glove')

import numpy as np
import pandas as pd
import os
import nltk
from gensim.utils import tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# ## Load train, test data

# In[ ]:


su = 0
ma = 0
MAX_LEN = 500
NUM_WORDS = 100000
EOS = '</s>'
UNK = '</u>'
df = pd.read_csv('../input/dataset/train.csv')
word_count = dict()
X = []
Y = []
for i in range(len(df)):
    if(type(df.loc[i,'reviewText']) != type("")):
        continue
    X.append(nltk.word_tokenize(df.loc[i,'reviewText']))
#     X.append(list(tokenize(df.loc[i,'reviewText'], lowercase=True)))
    if(len(X[-1]) <= MAX_LEN-1):
        X[-1] = X[-1] + ([EOS]*(MAX_LEN-len(X[-1])))
    else:
        X[-1] = X[-1][:MAX_LEN-1] + [EOS]
    for word in X[-1]:
        if(word not in word_count.keys()):
            word_count[word] = 1
        else:
            word_count[word] += 1
#     tmp = np.zeros(5)
#     tmp[int(df.loc[i, 'overall'])-1] = 1
#     Y.append(tmp)
    Y.append(df.loc[i, 'overall'])
    su += len(X[-1])
    ma = max(ma, len(X[-1]))
    if(i%10000 == 0):
        print(i,'/',len(df))

word_to_num = dict()
num_to_word = dict()
ct = 2
word_to_num[UNK] = 0
num_to_word[0] = UNK
word_to_num[EOS] = 1
num_to_word[1] = EOS
word_list = sorted([(word_count[word], word) for word in word_count.keys()], reverse=True)
word_filtered = dict([(x[1],x[0]) for x in word_list[:NUM_WORDS - 2]])

for i in range(len(X)):
    for j in range(len(X[i])):
        if(X[i][j] in word_filtered.keys()):
            if(X[i][j] not in word_to_num.keys()):
                word_to_num[X[i][j]] = ct
                num_to_word[ct] = X[i][j]
                ct += 1
            X[i][j] = word_to_num[X[i][j]]
            
        else:
            X[i][j] = word_to_num[UNK]
X = np.asarray(X)
Y = np.asarray(Y)


# In[ ]:


for i in range(len(X)):
    if(len(X[i])!=MAX_LEN):
            print(len(X[i]))


# In[ ]:


df = pd.read_csv('../input/dataset/val.csv')
X_test = []
Y_test = []
for i in range(len(df)):
    if(type(df.loc[i,'reviewText']) != type("")):
        continue
    X_test.append(nltk.word_tokenize(df.loc[i,'reviewText']))
#     X_test.append(list(tokenize(df.loc[i,'reviewText'], lowercase=True)))
    if(len(X_test[-1]) < MAX_LEN-1):
        X_test[-1] = X_test[-1] + [EOS]*(MAX_LEN-len(X_test[-1]))
    else:
        X_test[-1] = X_test[-1][:MAX_LEN-1] + [EOS]
#     tmp = np.zeros(5)
#     tmp[int(df.loc[i, 'overall'])-1] = 1
#     Y_test.append(tmp)
    Y_test.append(df.loc[i, 'overall'])
    if(i%10000 == 0):
        print(i,'/',len(df))
        
for i in range(len(X_test)):
    for j in range(len(X_test[i])):
        if(X_test[i][j] in word_to_num.keys()):
            X_test[i][j] = word_to_num[X_test[i][j]]
        else:
            X_test[i][j] = word_to_num[UNK]

X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)


# In[ ]:


filenames = ['glove/glove.6B.300d.txt']
embeddings_index = dict()
for filename in filenames:
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# In[ ]:


# create a weight matrix for words in training docs
embedding_matrix = np.zeros((NUM_WORDS, 300))
for word, i in word_to_num.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[ ]:


from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, Dense, Reshape, Dropout, Flatten, GRU, Bidirectional
from keras.optimizers import Adam
model = Sequential()
model.add(Embedding(NUM_WORDS, 300, weights=[embedding_matrix], input_length=MAX_LEN, trainable=True, name='e1'))
model.add(Bidirectional(GRU(32, activation='relu', return_sequences=True, name='g1')))
model.add(Bidirectional(GRU(32, activation='relu', return_sequences=True, name='g2')))
model.add(Bidirectional(GRU(32, activation='relu', return_sequences=False, name='g3')))
model.add(Dense(512, activation='relu',name='d1'))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu', name='d2'))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu', name='d3'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='relu', name='o'))
model.compile(Adam(), loss='mse', metrics=['accuracy'])


# In[ ]:


from keras.legacy import interfaces
import keras.backend as K
from keras.optimizers import Optimizer

class Adam_lr_mult(Optimizer):
    """Adam optimizer.
    Adam optimizer, with learning rate multipliers built on Keras implementation
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
        
    AUTHOR: Erik Brorson
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False,
                 multipliers=None, debug_verbose=False,**kwargs):
        super(Adam_lr_mult, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.multipliers = multipliers
        self.debug_verbose = debug_verbose

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            # Learning rate multipliers
            if self.multipliers:
                multiplier = [mult for mult in self.multipliers if mult in p.name]
            else:
                multiplier = None
            if multiplier:
                new_lr_t = lr_t * self.multipliers[multiplier[0]]
                if self.debug_verbose:
                    print('Setting {} to learning rate {}'.format(multiplier[0], new_lr_t))
                    print(K.get_value(new_lr_t))
            else:
                new_lr_t = lr_t
                if self.debug_verbose:
                    print('No change in learning rate {}'.format(p.name))
                    print(K.get_value(new_lr_t))
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - new_lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - new_lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad,
                  'multipliers':self.multipliers}
        base_config = super(Adam_lr_mult, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[ ]:


from keras.callbacks import EarlyStopping, ModelCheckpoint
muldict = {}
muldict['e1'] = 5e-2
muldict['g1'] = 1
muldict['g2'] = 1
muldict['g3'] = 1
muldict['d1'] = 1
muldict['d2'] = 1
muldict['d3'] = 1
opt = Adam_lr_mult(lr = 5e-5, multipliers = muldict)
model.compile(opt, loss='mse', metrics=['accuracy'])
es = EarlyStopping(verbose = 2, patience = 5)
mc = ModelCheckpoint(filepath='./weight.hdf5', save_best_only=True, verbose = 2)
model.fit(X,Y,batch_size = 1024, shuffle=True, validation_split = 0.1, epochs=100, callbacks = [es,mc])


# In[ ]:


model.load_weights('./weight.hdf5')
# model.evaluate(X_test,Y_test)


# In[ ]:


print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

y_pred = model.predict(X_test)
y_pred_2 = np.asarray(np.round(y_pred.flatten()), dtype=np.int)
y_test_2 = np.asarray(np.round(Y_test), dtype=np.int)


# In[ ]:


for i in range(len(y_pred_2)):
    if(abs(y_pred_2[i] - y_test_2[i]) >= 2):
        for word in X_test[i]:
            if num_to_word[word] == EOS:
                break
            print(num_to_word[word], end=' ')
        print("\n")
        print("Pred ",y_pred_2[i], "Real", y_test_2[i])


# In[ ]:


f1_score(y_test_2, y_pred_2, average='weighted')


# In[ ]:


for i in range(len(y_pred_2)):
    y_pred_2[i] = max(0,y_pred_2[i])
    y_pred_2[i] = min(5,y_pred_2[i])


# In[ ]:


# Plot normalized confusion matrix
classes = ["1.0", "2.0", "3.0", "4.0", "5.0"]
plot_confusion_matrix(y_test_2, y_pred_2, classes=classes, normalize=False,title='Confusion matrix')
print(classification_report(y_test_2, y_pred_2, target_names=classes))
plt.show()

