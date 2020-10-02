#!/usr/bin/env python
# coding: utf-8

# In this kernel we are focused on doing two major things:
# 1. Seeing if we can disambiguate words using parts of speech and if this is useful to our model
# 2. Analyzing errors and understanding why they may have occurred

# In[ ]:


# all the standard imports + spacy
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
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, concatenate
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers



import matplotlib.pylab as plt


# # Disambiguation
# So what do I mean when I say dismbiguate the words using the parts of speech? If you want to dig in for real check out the wiki page for [word sense disambiguation ](https://en.wikipedia.org/wiki/Word-sense_disambiguation) , but I'll try my best to give a rudimentary example. 
# 
# If you look in the dictionary many words have multiple definitions. What our word embeddings currently capture is a happy medium between the multiple different usages of the word. In theory it is likely skewing toward whatever the most common sense of the word is and just being confused by the other connotations. So for example if the word is bank then it is trying to find a reasonable location for both bank as in the financial bank and bank as in a river bank and maybe also a bank shot like in sports. Obviously these are all relatively unrelated concepts and our embedding is hopelessy trying to capture all in one 300 dimension vector.
# 
# It would be great if we had a way to disambiguate these various meanings so that they each have their own embedding that can move independently. The problem is how do we determine which sense is being used when? We don't have a perfect way of discerning this and even as humans sometimes we have misunderstandings of what specifically someone is referring to, but one potential way to make this determination is the part of speech. One thing you will notice in the dictionary is that words have their parts of speech listed and then the corresponding definitions under them. And we're in luck, automatic parts of speech tagging has been very thoroughly explored and we have libraries like spacy that can do it quickly and with a reported accuracy of 92%+. We'll do the parts of speech tagging and then I'll explain how to utilize this information further.
# 
# 

# In[ ]:


#we will import spacy and then disable the additional modules we dont need because they take a lot of compute time
import spacy
nlp = spacy.load('en', disable = ["parser", "ner", "textcat", "tagger"])


# **Load data**

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)


# Here we will use the spacy pipeline just to tokenize. There are plenty of other faster/simpler tokenizers, but for the sake of consistency I am using spacy's built in. 

# In[ ]:


tokens = []
for doc in tqdm(nlp.pipe(train_df["question_text"].values, n_threads = 16)):
    tokens.append(" ".join([n.text for n in doc]))
train_df["question_text"] = tokens


# In[ ]:


results = set()
train_df['question_text'].str.lower().str.split().apply(results.update)
print("Number of unique words before pos tagging:", len(results))


# In[ ]:


print("What the text looks like before tagging:",  train_df["question_text"][0])


# Here we will use the spacy pipeline we just created in order to tag the parts of speech and then return back a string that has each word concatenated with the part of speech that was found.  

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


nlp = spacy.load('en', disable = ["parser", "ner", "textcat"])
tokens = []
for doc in tqdm(nlp.pipe(train_df["question_text"].values, n_threads = 16)):
    tokens.append(" ".join([n.text + "_"  + n.pos_ for n in doc]))
train_df["question_text"] = tokens


# In[ ]:


print("after tagging:",  train_df["question_text"][0])
results = set()
train_df['question_text'].str.lower().str.split().apply(results.update)
print("Number of unique words after pos tagging:", len(results))


# In[ ]:


#number of words we have added to our vocabulary
284281 - 219231


# In[ ]:


tokens = []
pos = []
for doc in tqdm(nlp.pipe(test_df["question_text"].values, n_threads = 16)):
    tokens.append([n.text + "_"  + n.pos_ for n in doc])
test_df["tokens"] = tokens


# **Setting up validation and training dataset**

# In[ ]:


# Cross validation - create training and testing dataset
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)


# The below is pretty much all the same standard stuff available in other public kernels. Only major thing I have changed is I removed the underscore character from the filters of the tokenizer so that our pos and word are not separated into their own tokens.

# In[ ]:


# Preprocess the data
## some config values                                                                                                                                                                                       
embed_size = 300 # how big is each word vector                                                                                                                                                              
max_features = 100000 # how many unique words to use (i.e num rows in embedding vector)                                                                                                                      
maxlen = 20 # max number of words in a question to use                                                                                                                                                     

## fill up the missing values                                                                                                                                                                               
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences                                                                                                                                                                                   
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~')
tokenizer.fit_on_texts(list(train_X))


# In[ ]:


train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences                                                                                                                                                                                        
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values                                                                                                                                                                                    
train_y = train_df['target'].values
val_y = val_df['target'].values


# **Load the pretrained embeddings**

# In[ ]:


EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))


# # This is where the magic happens.
# What we are doing here is only subtley different from what everyone else has been doing. Typically our matrix would have one entry for each word and we would fill it up by looking for that word in our pretrained embeddings. Now what we are doing is creating one entry for each word_pos pair and then looking up just the word. This will initialize bank_noun and bank_verb as the same embedding but now they will be able to be independently moved in the embedding space during training. 
# 
# ![WSD2](https://i.imgur.com/akeenqu.png)
# 
# As you can see from my lovely paint work, now our word "bank" can be pushed in to separate directions for when it is used as a verb or as a noun. Might not be the best example of this, but it at least conveys the concept hopefully. Visualization is a screenshot from [projector.tensorflow.org](projector.tensorflow.org)

# In[ ]:


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
vocab_dict = {}
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= nb_words: continue
    word_part = word.split("_")[0]
    embedding_vector = embeddings_index.get(word_part)
    if embedding_vector is not None: 
        if word_part in vocab_dict:
            vocab_dict[word_part].append((word, i))
        else:
            vocab_dict[word_part] = [(word, i)]
        embedding_matrix[i] = embedding_vector
    


# In[ ]:


#filter for words that with more than one part of speech
vocab_dict = {i:vocab_dict[i] for i in vocab_dict if len(vocab_dict[i]) > 1}


# In[ ]:


new_vocab_indexes = []
new_word_list = []
for word in vocab_dict.values():
    for pos in word:
        new_vocab_indexes.append(pos[1])
        new_word_list.append(pos[0])


# In[ ]:


from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
tsvd = TruncatedSVD(n_components=3)
# tsne = TSNE(n_components=3)
X_embedded = tsvd.fit_transform(embedding_matrix[new_vocab_indexes])
# X_embedded = tsne.fit_transform(X_embedded)
X_embedded.shape


# Using this awesome plotly interactive graph we can look at what our embeddings currently look like and see that we in fact have words with different parts of speech that are initialized to the same spot. We will check on where they are at after the embedding training. 

# In[ ]:


word_count = 5000


# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import numpy as np
trace1 = go.Scatter3d(
    x=X_embedded[:word_count,0],
    y=X_embedded[:word_count,1],
    z=X_embedded[:word_count,2],
    mode='markers',
    text = new_word_list[:word_count],
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='simple-3d-scatter')


# In[ ]:


from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min', restore_best_weights=True)


# An insane CNN model I made just for kicks. Uncomment for an interesting graph you can show to people who doubt the complexity of your work. 

# In[ ]:


from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Flatten, Lambda, SpatialDropout1D
import keras.backend as K
# nb_filter = 32
# inp = Input(shape=(maxlen,))

# x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
# x = SpatialDropout1D(.4)(x)
# rev_x = Lambda(lambda x: K.reverse(x,axes=-1))(x)
# conv = Convolution1D(nb_filter=nb_filter, filter_length=1,
#                      border_mode='valid', activation='relu')(x)
# conv1 = Convolution1D(nb_filter=nb_filter, filter_length=2,
#                      border_mode='valid', activation='relu')(x)
# conv2 = Convolution1D(nb_filter=nb_filter, filter_length=3,
#                      border_mode='valid', activation='relu')(x)
# conv3 = Convolution1D(nb_filter=nb_filter, filter_length=4,
#                      border_mode='valid', activation='relu')(x)
# convs = [conv, conv1, conv2, conv3]
# convs2 = []
# for layer in convs:
#     conv = Convolution1D(nb_filter=nb_filter, filter_length=1,
#                      border_mode='valid', activation='relu')(layer)
#     conv1 = Convolution1D(nb_filter=nb_filter, filter_length=2,
#                          border_mode='valid', activation='relu')(layer)
#     conv2 = Convolution1D(nb_filter=nb_filter, filter_length=3,
#                          border_mode='valid', activation='relu')(layer)
#     conv3 = Convolution1D(nb_filter=nb_filter, filter_length=4,
#                          border_mode='valid', activation='relu')(layer)
#     conv = MaxPooling1D()(conv)
#     conv1 = MaxPooling1D()(conv1)
#     conv2 = MaxPooling1D()(conv2)
#     conv3 = MaxPooling1D()(conv3)
#     convs2.append(concatenate([conv, conv1, conv2, conv3], axis = 1))

    
# conv4 = concatenate(convs2, axis = 1)


# rev_conv = Convolution1D(nb_filter=nb_filter, filter_length=1,
#                      border_mode='valid', activation='relu')(rev_x)
# rev_conv1 = Convolution1D(nb_filter=nb_filter, filter_length=2,
#                      border_mode='valid', activation='relu')(rev_x)
# rev_conv2 = Convolution1D(nb_filter=nb_filter, filter_length=3,
#                      border_mode='valid', activation='relu')(rev_x)
# rev_conv3 = Convolution1D(nb_filter=nb_filter, filter_length=4,
#                      border_mode='valid', activation='relu')(rev_x)
# rev_convs = [rev_conv, rev_conv1, rev_conv2, rev_conv3]
# rev_convs2 = []
# for layer in rev_convs:
#     rev_conv = Convolution1D(nb_filter=nb_filter, filter_length=1,
#                      border_mode='valid', activation='relu')(layer)
#     rev_conv1 = Convolution1D(nb_filter=nb_filter, filter_length=2,
#                          border_mode='valid', activation='relu')(layer)
#     rev_conv2 = Convolution1D(nb_filter=nb_filter, filter_length=3,
#                          border_mode='valid', activation='relu')(layer)
#     rev_conv3 = Convolution1D(nb_filter=nb_filter, filter_length=4,
#                          border_mode='valid', activation='relu')(layer)
#     rev_conv = MaxPooling1D()(rev_conv)
#     rev_conv1 = MaxPooling1D()(rev_conv1)
#     rev_conv2 = MaxPooling1D()(rev_conv2)
#     rev_conv3 = MaxPooling1D()(rev_conv3)
    
#     rev_convs2.append(concatenate([rev_conv, rev_conv1, rev_conv2, rev_conv3], axis = 1))

    
# rev_conv4 = concatenate(rev_convs2, axis = 1)
# conv4 = concatenate([rev_conv4, conv4], axis = 1)

# conv5 = Flatten()(conv4)

# z = Dropout(0.5)(Dense(64, activation='relu')(conv5))
# z = Dropout(0.5)(Dense(64, activation='relu')(z))

# pred = Dense(1, activation='sigmoid', name='output')(z)

# model = Model(inputs=inp, outputs=pred)

# model.compile(loss='binary_crossentropy', optimizer='adam',
#               metrics=['accuracy'])


# from keras.utils.vis_utils import model_to_dot
# from IPython.display import Image

# Image(model_to_dot(model, show_shapes=True).create(prog='dot', format='png'))


# In[ ]:


#the model we will actually use to measure performance and analyze errors on
inp = Input(shape=(maxlen,))
x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable = False)(inp)
x = SpatialDropout1D(.4)(x)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# We will start with frozen weights and then unfreeze them for some more tuning

# In[ ]:


model.fit(train_X, train_y, batch_size=3000, epochs=8, validation_data=(val_X, val_y), callbacks = [es])


# In[ ]:


for layer in model.layers:
    layer.trainable = False
    if "embedding" in layer.name:
        layer.trainable = True
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.fit(train_X, train_y, batch_size=3000, epochs=8, validation_data=(val_X, val_y), callbacks = [es])


# In[ ]:


for layer in model.layers:
    if "embedding" in layer.name:
        new_embed = layer.get_weights()[0]


# Now that the embeddings have been trained lets check on how they were moved and if the different POS versions were disambiguated. 

# In[ ]:



X_embedded = tsvd.transform(new_embed[new_vocab_indexes])
# X_embedded = tsne.fit_transform(X_embedded)
X_embedded.shape


# Now we can look at the cosine similarity between two versions of a word and see if they have moved. We will look at the average across all similar base words. A cosine similarity of 1 is an identical vector which is what we expect from the original embeddings

# In[ ]:





# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
def sum_cos_sim(embedding_matrix):
    tot_sim = 0
    sim_dict = {}
    for word, word_parts in vocab_dict.items():
        cos_sim = 0
        first_embed = embedding_matrix[word_parts[0][1]].reshape(1, -1)
        for word_part in word_parts[1:]:
            next_embed = embedding_matrix[word_part[1]].reshape(1, -1)
            cos_sim += cosine_similarity(first_embed, next_embed)/(len(word_parts) -1)
        sim_dict[word] = cos_sim
        tot_sim += cos_sim/len(vocab_dict.items())
    return tot_sim, sim_dict


# In[ ]:


sim_val, sim_dict = sum_cos_sim(embedding_matrix)
print("average cosine similarity of different word senses:" , sim_val)


# In[ ]:


sim_val, sim_dict = sum_cos_sim(new_embed)
print("average cosine similarity of different word senses after embedding tuning:" , sim_val)


# Looks like they were moved but only a tiny bit. Probably not enough to be useful. It is interesting to see that with more data this may be a useful method to separate the different senses of words. 
# 
# Below is the same visualization as before but now with the adjusted embeddings.

# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import numpy as np
trace1 = go.Scatter3d(
    x=X_embedded[:word_count,0],
    y=X_embedded[:word_count,1],
    z=X_embedded[:word_count,2],
    mode='markers',
    text = new_word_list[:word_count],
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='simple-3d-scatter')


# **Prediction on validation dataset**

# In[ ]:


pred_noemb_val_y = model.predict([val_X], batch_size=1024, verbose=1)

thresholds = np.arange(0.1, 0.501, 0.01)
f1s = np.zeros(thresholds.shape[0])

for ind, thresh in np.ndenumerate(thresholds):
    f1s[ind[0]] = metrics.f1_score(val_y, (pred_noemb_val_y > np.round(thresh, 2)).astype(int))


# In[ ]:


np.round(thresholds[np.argmax(f1s)], 2)


# In[ ]:


import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    # else:
    # print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[ ]:


pred_noemb_val_y[:, 0].shape


# # Error analysis
# Now that we've made predictions and found the optimal threshold we can start analyzing what is going wrong. We will start with a confusion matrix and checking out precision, recall and f1 score

# In[ ]:


pred_noemb_val_y1 = pred_noemb_val_y[:, 0]
y_test = val_y


# In[ ]:


opt_thresh = np.round(thresholds[np.argmax(f1s)], 2)
# y_test = val_y
y_pred = (pred_noemb_val_y > opt_thresh).astype(int)

cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Sincere','Insincere'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Sincere','Insincere'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[ ]:


opt_thresh = np.round(thresholds[np.argmax(f1s)], 2)
y_test = val_y
y_pred = (pred_noemb_val_y > opt_thresh).astype(int)

cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Sincere','Insincere'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Sincere','Insincere'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[ ]:


precision = cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[0,1])
recall = cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0])
print("Precision: " + str(np.round(precision, 3)))
print("Recall: " + str(np.round(recall, 3)))


# In[ ]:


"F1 Score:", (2 * precision * recall)/(precision + recall)


# In[ ]:


pred_noemb_test_y = model.predict([test_X], batch_size=1024, verbose=1)


# In[ ]:


original_text = val_df["question_text"].fillna("_na_").values


# Some fancy pants code my teammate Alex made for the toxic comment challenge that I've expanded on and adapted to this challenge. The gist of it is to print off the original text, then the text after embedding because that is actually what reaches our model. Everything else is just dropped, so with this we can see if potentially a distinction wasnt made because a word wasnt in our embeddings vocabulary or if it was lost in a preprocessing step. In this kernel we dont do any preprocessing, but this is a good litmus test to see exactly how that might be affecting your predictions. One other thing we can glean from this is if our questions are getting cut off by the padding size we used. In this specific case it is only 20 so that it can run quickly so many are cut off. 
# 
# Originally I set this up to look at the 500 with the highest loss, but eventually I realized it might be more useful to look at the comments that are near the decision boundary threshold but not on the correct side. In theory it should be easier to move the decisions a little bit instead of taking something our model is confident is sincere all the way across the boundary to insincere.
# 
# This function prints off the correct label, the models prediction value, the loss, the original text and then the text after embedding.  

# In[ ]:


import operator
from tqdm import tqdm
def analyze_model(model, num_results, reverse = False):
    #let's see on which comments we get the biggest loss
    train_predictions = model.predict([val_X], batch_size=250, verbose=1)
    inverted_word_index = dict([[v,k] for k,v in word_index.items()])
    pd.DataFrame(train_predictions).hist()
    results = []
    eps = 0.1 ** 64
    for i in tqdm(range(0, len(val_y))):
        metric = 0

        for j in range(len([val_y[i]])):
            p = train_predictions[i][j]
            y = [val_y[i]][j]
            metric +=  -(y * math.log(p + eps) + (1 - y) * math.log(1 - p + eps))
        if p < opt_thresh and y == 1:
            results.append((original_text[i], metric, val_y[i], train_predictions[i], val_X[i]))
    results.sort(key=operator.itemgetter(1), reverse=reverse)  
    
    for i in range(num_results):
        inverted_text = ""
        for index in results[i][4]:
            if index > 0:
                word = inverted_word_index[index]
                if not np.any(embedding_matrix[index]):
                    word = "_" + word + "_"
                inverted_text += word + " "


        print(str(results[i][2]) + "\t" + str(results[i][3]) + "\t" + str(results[i][1]))
        print("Original Text")
        print( str(results[i][0]))
        print("---------------------------")
        print("Text that reached the model")
        print(inverted_text)
        print("")


# In[ ]:


#500 highest loss comments
#Correct Label | Model Output | Loss
#Original Text
#===========
#Text that reached the model after preprocessing, tokenizing and embedding
analyze_model(model, 500, False)


# It appears the bigger issue is questions being cut off rather than them being missing words, but I'll let you look through them and come to your own conclusions. It seems like many of the questions have the issue of being a legitimate question, but falling under the category of being easily googled so there aren't any clear indicators in terms of vocabulary or meaning. 

# In[ ]:





# In[ ]:




