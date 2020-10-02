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


import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense, Dropout, Concatenate, Lambda, Flatten
from keras.layers import GlobalMaxPool1D
from keras.models import Model


import tqdm


# # Combinations
# This kernel would contain a combination of previousle tested models. For example, it may be useful to combine pretrained embeddings with ones that were trained on this particular datase.

# # Embeddings

# In[ ]:


MAX_SEQUENCE_LENGTH = 60
MAX_WORDS = 75000
EMBEDDINGS_TRAINED_DIMENSIONS = 100
EMBEDDINGS_LOADED_DIMENSIONS = 300


# ## Custom
# Train our own embeddings on the training data

# In[ ]:


import gensim, logging
from nltk.tokenize import sent_tokenize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class SentenceGenerator(object):
    def __init__(self, texts):
        self.texts = texts
    def __iter__(self):
        for text in self.texts:
            sentences = sent_tokenize(text)
            for sent in sentences:
                yield sent
 

def train_w2v(texts, epochs=5):
    sent_gen = SentenceGenerator(texts)
    model_path = "quora_w2v" +        f"_{EMBEDDINGS_TRAINED_DIMENSIONS}dimenstions" +        f"_{str(epochs)}epochs" +        f"_{MAX_WORDS}words" +        ".model"

    if (os.path.isfile(model_path)):
        model = gensim.models.Word2Vec.load(model_path)
        print("Word2Vec loaded from " + model_path)
    else:
        model = gensim.models.Word2Vec(sent_gen, size=EMBEDDINGS_TRAINED_DIMENSIONS, workers=4, max_final_vocab=MAX_WORDS, iter=epochs)
        model.save(model_path)
        print("Word2Vec saved to " + model_path)
        
    return model


# ## Pretrained
# Load (one of) the embeddings

# In[ ]:


def load_embeddings(file):
    embeddings = {}
    with open(file) as f:
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings = dict(get_coefs(*line.split(" ")) for line in f if len(line)>100)
        
    print('Found %s word vectors.' % len(embeddings))
    return embeddings


# # NLP Features
# Find Part of Speech tags and named entities in the questions. Tokenize them and use them later in the model.

# In[ ]:


import spacy

nlp = spacy.load("en_core_web_sm", disable=['parser'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))
print(f"spaCy pipes: {nlp.pipe_names}")


# Find POS and NER tags
# Entity types from https://spacy.io/api/annotation#named-entities
pos_tags = nlp.tokenizer.vocab.morphology.tag_map.keys()
pos_tags_count = len(pos_tags)
entity_types = ["PERSON", "NORP", "FAC", "ORG", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE",
                "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
entity_types_count = len(entity_types)


pos_tokenizer = Tokenizer(num_words=pos_tags_count, lower=False)
pos_tokenizer.fit_on_texts(pos_tags)
default_filter_without_underscore = '!"#$%&()*+,-./:;<=>?@[\]^`{|}~'
entity_tokenizer = Tokenizer(num_words=entity_types_count,
                             lower=False, oov_token='0',
                             filters=default_filter_without_underscore)
entity_tokenizer.fit_on_texts(list(entity_types))
entity_types_count = len(entity_tokenizer.index_word) + 1

def token_encoded_pos_getter(token):
    if token.tag_ in pos_tokenizer.word_index:
        return pos_tokenizer.word_index[token.tag_]
    else:
        return 0

def token_encoded_ent_getter(token):
    if token.ent_type_ in entity_tokenizer.word_index:
        return entity_tokenizer.word_index[token.ent_type_]
    else:
        return 0

spacy.tokens.token.Token.set_extension('encoded_pos', force=True, getter=token_encoded_pos_getter)
spacy.tokens.token.Token.set_extension('encoded_ent', force=True, getter=token_encoded_ent_getter)
spacy.tokens.doc.Doc.set_extension('encoded_pos', force=True, getter=lambda doc: [token._.encoded_pos for token in doc])
spacy.tokens.doc.Doc.set_extension('encoded_ent', force=True, getter=lambda doc: [token._.encoded_ent for token in doc])

def make_nlp_features(texts):
    '''
    A simple greedy function that generates one-hot encodings for the NLP features of each word in each question.
    '''
    pos_encodings = []
    ent_encodings = []
    for doc in tqdm.tqdm(nlp.pipe(texts, batch_size=100, n_threads=4), total=len(texts)):
        pos_encodings.append(doc._.encoded_pos)
        ent_encodings.append(doc._.encoded_ent)

    pos_encodings = np.array(pos_encodings)
    pos_encodings = pad_sequences(pos_encodings, maxlen=MAX_SEQUENCE_LENGTH)

    ent_encodings = np.array(ent_encodings)
    ent_encodings = pad_sequences(ent_encodings, maxlen=MAX_SEQUENCE_LENGTH)

    return pos_encodings, ent_encodings


# # Data
# Load the data.

# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# In[ ]:


BATCH_SIZE = 512
Q_FRACTION = 1
questions = df_train.sample(frac=Q_FRACTION)
question_texts = questions["question_text"].values
question_targets = questions["target"].values
test_texts = df_test["question_text"].fillna("_na_").values

print(f"Working on {len(questions)} questions")


# In[ ]:


tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(list(df_train["question_text"].values))


# In[ ]:


custom_embeddings = train_w2v(question_texts, epochs=5)
pretrained_embeddings = load_embeddings("../input/embeddings/glove.840B.300d/glove.840B.300d.txt")


# In[ ]:


from collections import defaultdict

def create_embedding_weights(tokenizer, embeddings, dimensions):
    not_embedded = defaultdict(int)
    
    word_index = tokenizer.word_index
    words_count = min(len(word_index), MAX_WORDS)
    embeddings_matrix = np.zeros((words_count, dimensions))
    for word, i in word_index.items():
        if i >= MAX_WORDS:
            continue
        if word not in embeddings:
            not_embedded[word] = not_embedded[word] + 1
            continue
        embedding_vector = embeddings[word]
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
            
    print(sorted(not_embedded, key=not_embedded.get)[:10])
    return embeddings_matrix


# In[ ]:


custom_emb_weights = create_embedding_weights(tokenizer, custom_embeddings, EMBEDDINGS_TRAINED_DIMENSIONS)
pretrained_emb_weights = create_embedding_weights(tokenizer, pretrained_embeddings, EMBEDDINGS_LOADED_DIMENSIONS)


# # Model
# Construct the model to use, e.g. a simple NN

# In[ ]:


from keras.layers import Conv1D, Conv2D, Reshape, MaxPool1D, MaxPool2D

filter_size = 5
num_filters = 45

def create_model(embeddings_weights):
    tok_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name="tok_input")
    ent_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name="ent_input", dtype='uint8')
    pos_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name="pos_input", dtype='uint8')

    trained = Embedding(MAX_WORDS,
                        EMBEDDINGS_TRAINED_DIMENSIONS,
                        weights=[custom_emb_weights],
                        trainable=True)(tok_input)
    pretrained = Embedding(MAX_WORDS,
                          EMBEDDINGS_LOADED_DIMENSIONS,
                          weights=[pretrained_emb_weights],
                          trainable=True)(tok_input)
    
    trained = GlobalMaxPool1D()(trained)
    trained = Dropout(0.7)(trained)
    trained = Dense(10)(trained)
    pretrained = GlobalMaxPool1D()(pretrained)
    pretrained = Dropout(0.7)(pretrained)
    pretrained = Dense(10)(pretrained)
    

    x_ent = Lambda(
        keras.backend.one_hot,
        arguments={"num_classes": entity_types_count},
        output_shape = (MAX_SEQUENCE_LENGTH, entity_types_count, 1))(ent_input)
    x_ent = Reshape((MAX_SEQUENCE_LENGTH, entity_types_count, 1))(x_ent)
    conv_0 = Conv2D(num_filters, kernel_size=(filter_size, entity_types_count), kernel_initializer='he_normal', activation='tanh')(x_ent)
    maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_size + 1, 1))(conv_0)
    x_ent = Flatten()(x_ent)
    x_ent = Dropout(0.1)(x_ent)
    x_ent = Dense(10)(x_ent)
    
    x_pos = Lambda(
        keras.backend.one_hot,
        arguments={"num_classes": pos_tags_count},
        output_shape = (MAX_SEQUENCE_LENGTH, pos_tags_count))(pos_input)
    x_pos = Reshape((MAX_SEQUENCE_LENGTH, pos_tags_count, 1))(x_pos)
    conv_0 = Conv2D(num_filters, kernel_size=(filter_size, pos_tags_count), kernel_initializer='he_normal', activation='tanh')(x_pos)
    maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_size + 1, 1))(conv_0)
    x_pos = Flatten()(x_pos)
    x_pos = Dropout(0.1)(x_pos)
    x_pos = Dense(10)(x_pos)
    
    x = Concatenate()([trained, pretrained, x_pos, x_ent])
    x = Dropout(0.7)(x)
    x = Dense(10)(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[tok_input, ent_input, pos_input], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    return model


# # Model evaluation
# 
# 
# 

# In[ ]:


import sklearn
import keras
import matplotlib.pyplot as plt

THRESHOLD = 0.35

class F1EpochCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.f1s = []
        
    def on_epoch_end(self, batch, logs={}):
        predictions = self.model.predict(self.validation_data[0])
        predictions = (predictions > THRESHOLD).astype(int)
        predictions = np.asarray(predictions)
        targets = self.validation_data[1]
        f1 = sklearn.metrics.f1_score(targets, predictions)
        print(f"validation_f1: {f1}")
        self.f1s.append(f1)
        return
    
def display_model_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

def display_model_f1(f1_callback):
    plt.plot(f1_callback.f1s)
    plt.title('F1')
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.legend(['F1 score'], loc='upper right')
    plt.show()


# # Training
# Train the model. Also, experiment with different versions

# ## Prepare the data first
# E.g. the tokenized words as well as the nlp features

# In[ ]:


(pos_encodings, ent_encodings) = make_nlp_features(question_texts)

train_X = pad_sequences(tokenizer.texts_to_sequences(question_texts),
                        maxlen=MAX_SEQUENCE_LENGTH)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model = create_model(custom_emb_weights)\nf1_callback = F1EpochCallback()\n#     x={"pos_input": pos_encodings, "ent_input": ent_encodings, "tok_input": train_X},\nhistory = model.fit(\n    x=[train_X, ent_encodings, pos_encodings],\n    y=question_targets,\n    batch_size=512, epochs=15, validation_split=0.015) #callbacks=[f1_callback],')


# In[ ]:


display_model_history(history)
# display_model_f1(f1_callback)


# # Results

# In[ ]:


(test_pos_encodings, test_ent_encodings) = make_nlp_features(test_texts)

test_word_tokens = pad_sequences(tokenizer.texts_to_sequences(test_texts),
                       maxlen=MAX_SEQUENCE_LENGTH)

pred_test = model.predict([test_word_tokens, test_ent_encodings, test_pos_encodings], batch_size=1024, verbose=1)
pred_test = (pred_test > THRESHOLD).astype(int)

df_out = pd.DataFrame({"qid":df_test["qid"].values})
df_out['prediction'] = pred_test
df_out.to_csv("submission.csv", index=False)


# In[ ]:




