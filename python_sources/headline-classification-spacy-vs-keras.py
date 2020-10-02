#!/usr/bin/env python
# coding: utf-8

# ## Intro
# 
# In this notebook, I approach the task of headline classification on a subset of the categories. I compare SpaCy's `TextCategorizer` feature inroduced in v2.0 with a custom CNN architecture inspired by [this](http://www.aclweb.org/anthology/D14-1181) paper and implemented in Keras. The point of this exercise is really just to see if I can beat SpaCy's model.

# In[ ]:


import json
import keras.layers as layers
import numpy as np
import pandas as pd
import spacy
from gensim.corpora import Dictionary
from keras.models import Model
from keras.preprocessing import sequence
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from spacy.util import minibatch


# In[ ]:


nlp = spacy.load('en')
data = pd.read_json('../input/News_Category_Dataset.json', lines=True)


# In[ ]:


categories = data.groupby('category').size().sort_values(ascending=False)
categories


# I'll limit my effort to the top seven of these categories. The reason for this is to avoid sparser and perhaps less clearly defined categories (e.g. why is ARTS & CULTURE separate from ARTS?).

# In[ ]:


TOP_N_CATEGORIES = 7
data = data[data.category.apply(lambda x: x in categories.index[:TOP_N_CATEGORIES]) &            (data.headline.apply(len) > 0)]
data_train, data_test = train_test_split(data, test_size=.1, random_state=31)


# ## SpaCy Baseline
# 
# SpaCy has a builtin `TextCategorizer` module that does multilabel classification. The docs for that are [here](https://spacy.io/api/textcategorizer), and while the specifics of the model are not documented, the docs do give a high level description:
# 
# > The document tensor is then summarized by concatenating max and mean pooling, and a multilayer perceptron is used to predict an output vector of length `nr_class`, before a logistic activation is applied elementwise. The value of each output neuron is the probability that some class is present.
# 
# I'll train this model and then later compare results with a custom CNN architecture. Since the model does multilabel classification, it will be interesting to see if just a multiclass setup with softmax instead of logistic activation on the output layer can boost performance.
# 
# Additionally, the only SpaCy model currently available in the Kaggle kernel environment is `en_web_core_sm` which does not have pretrained word embeddings. Therefore, I assume that the embedding layer in this model is randomly initialized and trainable. This will be important when training my custom model later.
# 
# The code below was adapted from the official SpaCy example [here](https://github.com/explosion/spacy/blob/master/examples/training/train_textcat.py).

# In[ ]:


# check if `textcat` is already in the pipe, add if not
if 'textcat' not in nlp.pipe_names:
    textcat = nlp.create_pipe('textcat')
    nlp.add_pipe(textcat, last=True)
else:
    textcat = nlp.get_pipe('textcat')

# add labels to the model    
for label in categories.index[:TOP_N_CATEGORIES]:
    textcat.add_label(label)

# preprocess training data
data_train_spacy = list(
    zip(data_train.headline,
        data_train.category.apply(
            lambda cat: {'cats': {c: float(c == cat)
                                  for c in categories.index[:TOP_N_CATEGORIES]}}))
)

# train the model
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for i in range(5):
        print('Epoch %d' % i)
        losses = {}
        batches = minibatch(data_train_spacy, size=128)
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                       losses=losses)
        with textcat.model.use_params(optimizer.averages):
            docs = [nlp.tokenizer(h) for h in data_test.headline]
            test_pred = np.array(
                [sorted(doc.cats.items(), key=lambda x: -x[1])[0][0]
                 for doc in textcat.pipe(docs)])
            print('Test Acc: %.4f' %
                  (pd.Series(test_pred == data_test.category.values).sum() / data_test.shape[0]))


# In[ ]:


spacy_y_pred = [sorted(doc.cats.items(), key=lambda x: -x[1])[0][0]
                for doc in nlp.pipe(data_test.headline)]
print(classification_report(data_test.category, spacy_y_pred))


# ## Simple CNN using Keras
# 
# Next, I'll train my own model. I pad (or clip) the sentences to have length 20 and limit the vocabulary to words that appear in at least three documents. Then, I start with an embedding layer, stack convolutional layers with different filter sizes, and finally end with a dense softmax output.

# In[ ]:


MAX_SEQUENCE_LEN = 20
UNK = 'UNK'
PAD = 'PAD'

def text_to_id_list(text, dictionary):
    return [dictionary.token2id.get(tok, dictionary.token2id.get(UNK))
            for tok in text_to_tokens(text)]

def texts_to_input(texts, dictionary):
    return sequence.pad_sequences(
        list(map(lambda x: text_to_id_list(x, dictionary), texts)), maxlen=MAX_SEQUENCE_LEN,
        padding='post', truncating='post', value=dictionary.token2id.get(PAD))

def text_to_tokens(text):
    return [tok.text.lower() for tok in nlp.tokenizer(text)
            if not (tok.is_punct or tok.is_quote)]

def build_dictionary(texts):
    d = Dictionary(text_to_tokens(t)for t in texts)
    d.filter_extremes(no_below=3, no_above=1)
    d.add_documents([[UNK, PAD]])
    d.compactify()
    return d


# In[ ]:


dictionary = build_dictionary(data.headline)


# In[ ]:


x_train = texts_to_input(data_train.headline, dictionary)
x_test = texts_to_input(data_test.headline, dictionary)


# In[ ]:


lb = LabelBinarizer()
lb.fit(categories.index[:TOP_N_CATEGORIES])
y_train = lb.transform(data_train.category)
y_test = lb.transform(data_test.category)


# In[ ]:


EMBEDDING_DIM = 50

inp = layers.Input(shape=(MAX_SEQUENCE_LEN,), dtype='float32')
emb = layers.Embedding(len(dictionary), EMBEDDING_DIM, input_length=MAX_SEQUENCE_LEN)(inp)
filters = []
for kernel_size in [2, 3, 4]:
    conv = layers.Conv1D(32, kernel_size, padding='same', activation='relu', strides=1)(emb)
    pooled = layers.MaxPooling1D(pool_size=MAX_SEQUENCE_LEN-kernel_size+1)(conv)
    filters.append(pooled)

stacked = layers.Concatenate()(filters)
flatten = layers.Flatten()(stacked)
drop = layers.Dropout(0.2)(flatten)
out = layers.Dense(7, activation='softmax')(drop)

model = Model(inputs=inp, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))


# In[ ]:


y_test_pred = [lb.classes_[i] for i in np.argmax(model.predict(x_test), axis=1)]
print(classification_report(data_test.category, y_test_pred))


# ## Conclusion
# 
# The SpaCy model is pretty similar in performance to the simple CNN.
# 
# ### Todo
# * Experiment further with the CNN parameters
# * Add pretrained embeddings and see how that affects performance
# * Experiment with other methods in [that](http://www.aclweb.org/anthology/D14-1181) paper
# * Look into more advanced CNN architectures
