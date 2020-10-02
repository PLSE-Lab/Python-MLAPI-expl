#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook is focus on [Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification) . This problem is basically **text classification** where we have to classify quora topics to be either *sincere* or *insincere*. To have a glance of the example quora topics see this [kernel](https://www.kaggle.com/konohayui/topic-modeling-on-quora-insincere-questions) and this [kernel](https://www.kaggle.com/thebrownviking20/analyzing-quora-for-the-insinceres). Note that this problem is 'kernel-only' submission with limited time of 2 hours.
#  
# **update Nov, 16, 2018** -- Add F1 directly as an evaluation metric. Credit this [kernel](https://www.kaggle.com/applecer/use-f1-to-select-model-lstm-based).
# **update Nov, 20, 2018** -- Add more potential ideas to fight with overfitting
# 
# ![Picture from flickr.com with ](https://c2.staticflickr.com/4/3781/9200265759_897d96f81c_b.jpg)
# 
# This work is mainly based on [SRK's excellent kernel "A look at different embeding"](https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings). In SRK's work, we can see that the 3 pretrained embedding matrices + GRU give good performance  (and their ensemble improves the final performance). Therefore, it is interesting to study them further, and see how far can the model go. This analysis can also be apply to any RNN types e.g. LSTM or a stack of GRU/LSTM and so on.
# 
# Here, we would like to explore the limit of this neural architecture (we choose GloVe as the representative embedding since all of the embedding methods have similar performances) by answering the following questions :
# 
# ## 1) Is this architecture's capacity overfit or underfit the problem? 
# - Can we improve the performance by running more and more epochs?
# - Should we increase the capacity of the network, i.e. increases more latent dimension or add more layers? (underfitting case)
# - In case of overfitting (as will be seen soon), what should we do?
# 
# ## 2) At its best, what kind of predictions do the network trying to make?
# Since the 'insincere' class has only a small number of data, our network has to be careful when it will predict 'class 1'.
# - Does it try to make only a sure prediction? (try a small number of class-1 prediction, but each one of them is precisely correct)
# - OR, Does it make class-1 prediction a lot (in order to cover most class 1 data), but hopefully not make too much mistakes to predict class 0 as class 1. 
# 
# ## 3) At its best, what kind of errors do the network make?
# - what are *insincere topics* where the network strongly believe to be *sincere* ?
# - what are *sincere topics* where the network strongly believe to be *insincere* ?
# - what are *insincere topics* where the network are most uncertain how to classify ?
# - what are *sincere topics* where the network are most uncertain how to classify?
# - (Not error, but good to see) what are *insincere topics* where the network strongly believe *correctly* ?
# 
# ## 4) How can we improve performance further based on the above error analysis?
# - I would like to hear your opinions on regarding this point!

# In[ ]:


import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


# In[ ]:


# credit : https://www.kaggle.com/applecer/use-f1-to-select-model-lstm-based
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[ ]:


import tensorflow as tf


def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)


# We do exactly the same step as SRK's :
#  * Split the training dataset into train and val sample. Cross validation is a time consuming process and so let us do simple train val split.
#  * Fill up the missing values in the text column with '_na_'
#  * Tokenize the text column and convert them to vector sequences
#  * Pad the sequence as needed - if the number of words in the text is greater than 'max_len' trunacate them to 'max_len' or if the number of words in the text is lesser than 'max_len' add zeros for remaining values.

# In[ ]:


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

## split to train and val

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen,padding='post',truncating='post')
val_X = pad_sequences(val_X, maxlen=maxlen,padding='post',truncating='post')
test_X = pad_sequences(test_X, maxlen=maxlen,padding='post',truncating='post')

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values


# Next, loading the glove embedding matrix.

# In[ ]:


EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        


# ## 1) Is this architecture's capacity overfit or underfit the problem? 
# 
# First of all, we need to have a benchmark. We would like to understand the best performance this network can give us. 
# In SRK's original notebook, we run the network for only 2 epochs and get good result. So, can we improve the performance by running more and more epochs?
# Or the network will more and more overfit the data?  
# 
# ![](https://raw.githubusercontent.com/alexeygrigorev/wiki-figures/master/ufrt/kddm/overfitting-logreg-ex.png)
# 
# Understanding this topic is central of how we can make a justification of how to improve the network. In the case of **overfitting**, it means that **the network has too much capacity, so it is not so useful to increase the number of parameters** such as increasing more layers or increase the RNN's latent dimension. In this case we say that the network have much variance so we should reduce it by using more training data or applying special techniques such as increasing dropout rate (or add more dropout layer) or using ensemble (More detailed below). 
# 
# On the other hand, in the case of **underfitting**, i.e. **the network cannot accurately fit the training data**, we have to do a usual routine : add more dimensions or add more layers.
# 
# You can learn more about bias/variance and overfitting/underfitting from [Andrew Ng's lectures](https://www.youtube.com/watch?v=dFX8k1kXhOw&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b)
# 
# So let us start by creating the Glove+GRU network and train it for 50 epochs. Note that the purpose of this notebook is not to make a high score submission; we want *insights*. Therefore, we do not care about the 2-hours time limit. We will try to make the network learn from data as much as possible.

# In[ ]:


inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1])
model.compile(loss=f1_loss, optimizer='adam', metrics=['accuracy',f1])
print(model.summary())


# In[ ]:


history = model.fit(train_X, train_y, batch_size=512, epochs=10, validation_data=(val_X, val_y))


# In[ ]:


history = model.fit(train_X, train_y, batch_size=512, epochs=5, validation_data=(val_X, val_y))


# After a while, we finally get the training and validation results! Let us plot their loss and accuracy.

# In[ ]:


#codes from machinelearningmastery.com
import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())
# summarize history for accuracy
# codes from machinelearningmastery.com
def print_hist(history):
    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.title('model af1')
    plt.ylabel('f1')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    


# In[ ]:


print_hist(history)


# ### What we learn from the #1 result
# 
# - The result is quite surprising (at least to me) : even though the network is considerably small (1 layer of GRU with only 64 dimensions of latent space) we **overfit** the data very quickly! Only 5 epochs are enough to get the best result. So roughly the 2-hour limitation makes sense as you do not need 6 hours to fit the data.
# 
# - Therefore, **there should be no clear advantage of building a deeper network**. what is most promising is how can we reduce its variance. This explains why using ensemble method works well here. 
# 
# ### What can we do to fight overfitting ?
# 
# ![](https://cdn-images-1.medium.com/max/1600/1*XWh6hd8BgI3RKhd8bkrbuw.png)
# (picture credit : https://hackernoon.com/memorizing-is-not-learning-6-tricks-to-prevent-overfitting-in-machine-learning-820b091dc42)
# 
# - **Ensemble** : this combines a lot of classifier to stabilize the prediction. This method was already implemented by SRK.
# 
# - **Dropout** : this is effectively similar to ensembling. In each training batch, the network will drop different set of nodes, so we can think that we have one unique classifier for each subset of selected nodes. But at test time, it will use all the nodes to combine all there predictions. Therefore, it can be view as a version of ensemble. Standard dropout layer is easy to implement in Keras, and is already done in SRK's code. Nevertheless, Keras has many types of dropout implementation e.g. spatial dropout, and maybe some of these may suit this problem.
# 
# (**Note** that unfortunately, CuDNNLSTM / CuDNNGRU do not support *recurrent_dropout*, and standard LSTM/GRU may be too time consuming to try in the limited 2 hours) 
# 
# -**Classic regularization** : we can use classic regularization such as L1 or L2 for each neural network layer. This could work, but you have to find appropriate values of regularization parameters. See [Keras manual](https://keras.io/regularizers/) for details.
# 
# -**Dimensionality reduction** : Beside reduce overfitting, dimensionality reduction can also help speed up the entire learning process.  PCA is a standard approach to reduce dimensionality. Other non-linear dimensionality reductions  which is popular in Deep Learning era is auto-encoder. In Deep Learning practice, however, the most simple but effective method is to simply apply 'Dense' layer to the input layer to reduce the dimension.
# 
# 
# - **Early stopping** : if run too many epochs will overfit the data, we will have to stop the training process early. I think we all have done tuning this hyperparameters :)
# 
# - **Adjust batch size** : this can be help also! Indeed, when we create each batch, we are randomly select them from the total data. Compared to the gradient of the whole data, the gradient of each batch can be seen as an added noisy direction.  This randomness can sometimes regularize our training process, not to overfit the training set. See [more excellent explanation here](https://www.quora.com/Intuitively-how-does-mini-batch-size-affect-the-performance-of-stochastic-gradient-descent)
# 
# - **Increase more training data** : If possible this will let your network understand more about the nature of the problem, and will help it to generalize better. Unfortunately, this competition doesn't allow external dataset. Another method is to create artificial data (which have the same characteristic as the real data), the so-called **Data Augmentation** method. There is a discussion on Data Augmentation [here](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/71083), but I still cannot think of ways to apply it. If you have any ideas please share :) 
# 
# - **Use of unlabeled data** : what if we cannot find more *labeled* training data?? Perhaps *unlabeled data* can also help! The method of employing unlabeled data with labeled data can be broadly called as *semi-supervised learning*. At least three methods are possible and exploit successfully in literatures. 
# 
# (1) *transductive learning* : Here, perhaps we try to exploit the test data itself which is unlabeled. Even though, we have no label of them, some of their information (data manifold) can be useful for our learners. See this [discussion](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557) as one example of how to employ test data (they called it *pseudo labelling*) 
# 
# (2) *language model pretrained and finetuning* : This is a well-known approach which is very successful in academic literatures, i.e. we use all (potentially infinite) unlabeled data to train an auxiliary task first (here, language model training). And then we use the pretrained network to fine-tune to our classification problem at hand. Most state-of-the-art results such as ELMO, BERT, ULMFiT and OpenAI's LM+Transformer all employ this methodology. Nevertheless, with the 2-hour constraints of this competition, it is challenging to find a way to employ this approach.
# 
# (3) *multi-task learning* : similar to (2), but instead of pretraining and fine-tuning, we can learn both the auxiliary task and the main classification task simultaneously!. See [Ruder's blog](http://ruder.io/multi-task-learning-nlp/) post for much more comprehensive details.
# 
# - **BONUS** : cleaning up original data. This may not be overfitting fighting, but it will definitely help! See [Dieter's excellent kernel here](https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings).
# 
# Is there any more methods I missed, if you have more ideas, please share!

# ## 2) At its best, what kind of predictions do the network trying to make?
# 
# Since the 'insincere' class has only a small number of data, our network has to be careful when it will predict 'class 1'.
# - Does it try to make only a sure prediction? (try a small number of class-1 prediction, but each one of them is precisely correct)
# - OR, Does it make class-1 prediction a lot (in order to cover most class 1 data), but hopefully not make too much mistakes to predict class 0 as class 1. 
# 
# To answer this question, let us build the network at its best, i.e. fitting the training data for 5 epochs.

# In[ ]:


inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1])
print(model.summary())

history = model.fit(train_X, train_y, batch_size=512, epochs=5, validation_data=(val_X, val_y))


# In[ ]:


print_hist(history)


# Finish!
# 
# Then, with little modification of SRK's code again : let us see the best threshold which makes the best prediction for training set and validation set. Not surprisingly, F1 score on training set is greater than F1 score on validation set.

# In[ ]:


pred_glove_train_y = model.predict([train_X], batch_size=1024, verbose=1)
scores = []
thresholds = np.arange(0.1, 0.501, 0.01)
for thresh in thresholds:
    thresh = np.round(thresh, 2)
    scores.append(metrics.f1_score(train_y, (pred_glove_train_y>thresh).astype(int)))
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(train_y, (pred_glove_train_y>thresh).astype(int))))
idx = np.argmax(np.array(scores))
thresh_train = thresholds[idx]


# In[ ]:


print(thresh_train)


# In[ ]:


pred_glove_val_y = model.predict([val_X], batch_size=1024, verbose=1)
scores = []
thresholds = np.arange(0.1, 0.501, 0.01)
for thresh in thresholds:
    thresh = np.round(thresh, 2)
    scores.append(metrics.f1_score(val_y, (pred_glove_val_y>thresh).astype(int)))
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_glove_val_y>thresh).astype(int))))
idx = np.argmax(np.array(scores))
thresh_valid = thresholds[idx]


# In[ ]:


print(thresh_valid)


# We can see that the best threshold for training data is around 0.35 - 0.4 for validation data (there is some randomness from our environment). Thaaat is only 35% -40% confidence is enough to classify the topic as insincere. Note that paradoxically **we don't need more than 51% confidence to say that the topic is insincere.**
# 
# We can understand its predictions better by looking at its accuray and confusion matrix.

# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix

print('total valid data is ',train_y.shape[0], 'where', np.sum(train_y==1),'is insincere' )
print(accuracy_score(train_y, pred_glove_train_y>thresh_train))
confusion_matrix(train_y, pred_glove_train_y>thresh_train, labels=[0, 1])


# In[ ]:



print('total valid data is ',val_y.shape[0], 'where', np.sum(val_y==1),'is insincere' )
print(accuracy_score(val_y, pred_glove_val_y>thresh_valid))
confusion_matrix(val_y, pred_glove_val_y>thresh_valid, labels=[0, 1])


# In[ ]:


print('This is the precision I got from validation set : ', 5712/(5712+3027))
print('This is the recall I got from validation set : ', 5712/(5712+2413))


# At 5 epochs, the network doesn't overfit the data as we can see from the fact that training accuracy and validation accuracy are quite similar.
# 
# Look at validation performance, in my run, we can see that : (note that you can get slightly different results due to randomness)
#  - Precision is aroud 65.X%
#  - Recall is also around = 70.X%
#  Since F1 is somewhat an average of the two score, that is why we get F1 validatin score around 67%.
#  
#  **Interpretation :**  we can see that the network balances its *insincere* prediction quite well. It indeed has a good precision (65% versus 6% of randomly guessing -- a 10X boost!). On the other hand, it doesn't 
# make a small number of class-1 prediction because it covers around 70% of the true insincere class. Our network indeed do a good job.
# 
# So far so good. But can we get more insights on its prediction behavior? Let us explore more on this on the next section.

# ## 3) At its best, what kind of errors do the network make?
# 
# Continue from the previous section, now we will try to dig deeper by going through error analysis as taught by [Andrew Ng](https://www.youtube.com/watch?v=JoAxZsdw_3w&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b&index=13) and [Fast.ai](https://www.kaggle.com/hortonhearsafoo/fast-ai-lesson-1)
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/9/94/Contingency_table.png)
# 
# We intend to understand the following :
# - what are *insincere topics* where the network strongly believes to be *sincere* ?
# - what are *sincere topics* where the network strongly believes to be *insincere* ?
# - what are *insincere topics* where the network are most uncertain how to classify ?
# - what are *sincere topics* where the network are most uncertain how to classify?
# - (Not error, but good to see) what are *insincere topics* where the network strongly believes *correctly* ?
# 
# By understanding this kind of errors, perhaps we will get new ideas of how to improve the performance.

# First, let us define these functions which will do the jobs: 

# In[ ]:


def most_false_negative(pred_y,true_y,thresh, N ):
    pred_y_round = (pred_y>thresh)
    
    flag = (pred_y_round == 0) & (true_y == 1)[:,np.newaxis]
    
    idxs = np.where(flag)[0] # ignore the np.newaxis dimension
    newidx = np.argsort(pred_y[idxs,0])

    return idxs[newidx[:N]], pred_y[idxs[newidx[:N]]]

def most_uncertain_negative(pred_y,true_y,thresh, N ):
    pred_y_round = (pred_y>thresh)
    flag = (pred_y_round == 0) & (true_y == 1)[:,np.newaxis]
    
    idxs = np.where(flag)[0] # ignore the np.newaxis dimension
    newidx = np.argsort(pred_y[idxs,0])
    return idxs[newidx[-N:]], pred_y[idxs[newidx[-N:]]]

def most_uncertain_positive(pred_y,true_y,thresh, N ):
    pred_y_round = (pred_y>thresh)
    
    flag = (pred_y_round == 1) & (true_y == 0)[:,np.newaxis]
    
    idxs = np.where(flag)[0] # ignore the np.newaxis dimension
    newidx = np.argsort(pred_y[idxs,0])

    return idxs[newidx[:N]], pred_y[idxs[newidx[:N]]]

def most_false_positive(pred_y,true_y,thresh, N ):
    pred_y_round = (pred_y>thresh)
    flag = (pred_y_round == 1) & (true_y == 0)[:,np.newaxis]
    
    idxs = np.where(flag)[0] # ignore the np.newaxis dimension
    newidx = np.argsort(pred_y[idxs,0])
    return idxs[newidx[-N:]], pred_y[idxs[newidx[-N:]]]

def most_true_positive(pred_y,true_y,thresh, N ):
    pred_y_round = (pred_y>thresh)
    
    flag = (pred_y_round == 1) & (true_y == 1)[:,np.newaxis]
    
    idxs = np.where(flag)[0] # ignore the np.newaxis dimension
    newidx = np.argsort(pred_y[idxs,0])

    return idxs[newidx[-N:]], pred_y[idxs[newidx[-N:]]]


# ###  what are *insincere topics* where the network strongly believe to be *sincere* ?
# This kind of predictions  is the most errorneous that our network made, so let see which kinds of topics are they. You can change the constant *NN* below in order to see more examples. 

# In[ ]:


NN = 20
idx, prob = most_false_positive( pred_glove_val_y, val_y,thresh_valid,NN)
for i in range(NN):
    print('class',val_y[idx[i]],prob[i],val_df["question_text"].values[idx[i]])


# The results are unexpected! As you can see our network strongly believes the above topics are insincere with high probabilities ( >90%), but the true class is sincere. Nevertheless, by looking at topics it can be seen that these topics should be class 1 instead!  So this should be the mislabels in the dataset. In fact, in the dataset page, quora does say that
# 
# > The ground-truth labels contain some amount of noise: they are not guaranteed to be perfect.
# 
# The mislabeling make the problem more difficult since our network are doing good, but is forced to overfit on these kind of topics. 

# ### what are *sincere topics* where the network strongly believe to be *insincere* ?
# This is similar to the above subsection, but in the opposite direction.

# In[ ]:


idx, prob = most_false_negative( pred_glove_val_y, val_y,thresh_valid,NN)
for i in range(NN):
    print('class',val_y[idx[i]],prob[i],val_df["question_text"].values[idx[i]])


# It seems that we get the noise again! According to our human knowledge, these topics look very sincere (and our network also strongly believes that), but the labels say that they are *insincere*. Again, this supports the evidence in #1 that our network will eventually overfit the data.
# 
# Next, let us consider the following two groups together.
# 
# ### what are *insincere topics* where the network are most uncertain how to classify ?, and
# ### what are *sincere topics* where the network are most uncertain how to classify?
# 
# Remember that the best threshold for validation data is :

# In[ ]:


print('our best threshold is',thresh_valid)


# In[ ]:


idx, prob = most_uncertain_positive( pred_glove_val_y, val_y,thresh_valid,NN)
for i in range(NN):
    print('class',val_y[idx[i]],prob[i],val_df["question_text"].values[idx[i]])
print('\n')
print('\n')
idx, prob = most_uncertain_negative( pred_glove_val_y, val_y,thresh_valid,NN)
for i in range(NN):
    print('class',val_y[idx[i]],prob[i],val_df["question_text"].values[idx[i]])


# What do you think of these topics where our network feels super unsure. To me, except for a few,  these topics are very difficult to judge whether they are sincere or not. Therefore, even the human baseline might not do a good job on this dataset.
# 
# ### Bonus : what are *insincere topics* where the network strongly believes *correctly* ?
# It is interesting to see what kind of topics both of our network and the labeller agree with.

# In[ ]:


idx, prob = most_true_positive( pred_glove_val_y, val_y,thresh_valid,NN)
for i in range(NN):
    print('class',val_y[idx[i]],prob[i],val_df["question_text"].values[idx[i]])


# It is clear here! These topics are unacceptable and very easy to classify due to their uses of impolite words.
# 
# 
# That's all for now! In fact, for the following last section, I would like to hear your great ideas! I hope that this kernel can be helpful.

# 
# 
# ## 4) How can we improve performance further based on the above error analysis?
# Please discuss :)
