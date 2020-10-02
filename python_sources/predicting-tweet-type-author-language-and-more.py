#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import tensorflow as tf
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pylab as pl

from tqdm import tqdm
tqdm.pandas()


# # Objectives
# 
# The main objective is to show how to do basic supervised learning with tensorflow for NLP data categorisation
# 
# * Basic NLP tasks with tensorflow
# * word base VS chars based, embeddings
# * Categorical prediction with tf
# * Tracking how well your model works: train VS validation performance, confusion matrix
# 

# # Data loading
# Small enough so we can load everything in RAM

# In[ ]:


frames = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print('Loading', os.path.join(dirname, filename))
        frames.append(pd.read_csv(os.path.join(dirname, filename)))
df = pd.concat(frames)
df = df.sample(frac=1).reset_index(drop=True)  # shuffles the rows, just in case the files contain a certain order etc...


# # Setup code

# ## Datasets building functions
# 
# Generic code to transform the dataframe into inputs and labels ingestible by tensorflow.  
# Mostly:
# 
# * Spliting the data into train, validation and test
# * String labels to categorical data mapping
# * Overall, converting data to tf usable format
# 
# One important note: what the following code also does is, when picking categorical data (author, language, type tc...), balancing the classes so we have as many examples in each. This is to prevent categories inbalance. Ex: if we had 10000 examples for English, 100 for Russian and 100 for Italian, we could make a very simple model reaching 98% accuracy by always predicting English for everything.

# In[ ]:


def make_train_test_val(key, topn, train_ratio=0.7, val_ratio=0.15):
    topn_df = df[key].value_counts().head(topn)
    min_nb = min(topn_df)
    train_size = int(train_ratio * min_nb)
    val_size = int(val_ratio * min_nb)
    test_size = min_nb - train_size - val_size
    # print('\nFor each %s: TRAIN=%s, VAL=%s, TEST=%s' % (key, train_size, val_size, test_size))
    unique_keys = topn_df.index
    
    training_frames = []
    val_frames = []
    test_frames = []

    for elem in topn_df.index:
        # print('Building', key, '|', elem, 'dataset (train/val/test)')
        sub_df = df[df[key] == elem]
        training_frames.append(sub_df[0: train_size - 1])
        val_frames.append(sub_df[train_size: train_size + val_size - 1])
        test_frames.append(sub_df[-test_size:])
    train_data = pd.concat(training_frames).sample(frac=1).reset_index(drop=True)
    val_data = pd.concat(val_frames).sample(frac=1).reset_index(drop=True)
    test_data = pd.concat(test_frames).sample(frac=1).reset_index(drop=True)
    return train_data, val_data, test_data, unique_keys


# In[ ]:


def unique2categ(unique_keys):
    elem2categ = dict(zip(unique_keys, range(len(unique_keys))))
    categ2elem = dict(zip(range(len(unique_keys)), unique_keys))
    print(elem2categ)
    print(categ2elem)
    return elem2categ, categ2elem


# In[ ]:


def tokenize_dataset(train_data, val_data, test_data, tokenizer):
    tokenizer.fit_on_texts(train_data.content)
    tokenizer.fit_on_texts(val_data.content)
    tokenizer.fit_on_texts(test_data.content)
    total_words = len(tokenizer.word_index) + 1
    
    train_sequences = tokenizer.texts_to_sequences(train_data.content)
    val_sequences = tokenizer.texts_to_sequences(val_data.content)
    test_sequences = tokenizer.texts_to_sequences(test_data.content)
    
    max_sequence_len_train = max([len(x) for x in train_sequences])
    max_sequence_len_val = max([len(x) for x in val_sequences])
    max_sequence_len_test = max([len(x) for x in test_sequences])
    max_sequence_len = max([max_sequence_len_train, max_sequence_len_val, max_sequence_len_test])

    train_input = np.array(pad_sequences(train_sequences, maxlen=max_sequence_len, padding='pre'))
    val_input = np.array(pad_sequences(val_sequences, maxlen=max_sequence_len, padding='pre'))
    test_input = np.array(pad_sequences(test_sequences, maxlen=max_sequence_len, padding='pre'))
    return train_input, val_input, test_input, max_sequence_len


# ## Tensorflow model
# 
# We're going to make use of a model with different types of Layers and activations. I am not looking for the best model, since this one model is a bit too complex and tends to overfit the data quite fast given how "little" (compared to the model complexity) we have.
# 
# Notably, we'll use the folowing layers:
# 
# * Embedding (report to the last section where I dive deeper into what are embeddings if interested)
# * Dense
# * LSTM
# * Dropout
# 
# I am not going to explain all the layers and activations since there are plenty of ressources online doing this better than I could.

# In[ ]:


def build_model(vocab_size, input_length, nb_classes, embedding_dim, units_Bi_LSTM=50, units_LSTM_2=30, dropout=0.3):
    model = tf.keras.Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=input_length))
    model.add(Bidirectional(LSTM(units_Bi_LSTM, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(LSTM(units_LSTM_2))
    model.add(Dense(vocab_size / 2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


# ## Model performance visualization
# 
# Since we will me working with categorical data I am making graphs of two informations I like to have to evaluate my performance (in addition to my basic model accuracy etc...)
# 
# * **confusion matrix**: for inputs associated with a given category, which categories does the model tends to predict. Useful to see if the model struggles to discernate between a subset of categories etc... Here we do the confusion matrix on the validation data.
# * **accuracy and loss evolution per epoch**: useful to track how well the model learns, as in "am I underfitting or overfitting?", by comparing scores on train and validation. Also useful to tune early-stopping.

# In[ ]:


def plot_confusion_matrix(model, val_input, val_expected, xlabel, ylabel, title, suptitle, labels):
    val_predicted = model.predict_classes(val_input, verbose=0)
    cm = tf.math.confusion_matrix(
        val_expected,
        val_predicted,
    ) #/ val_size
    fig = pl.figure(num=None, figsize=(16, 14), dpi=80, facecolor='w', edgecolor='k')
    hm = sns.heatmap(cm, annot=True, fmt="d", linewidths=.5)
    hm.set_yticklabels(labels, rotation=0)
    hm.set_xticklabels(labels, rotation=90)
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.title(title, size=20)
    pl.suptitle(suptitle, size=40)


# In[ ]:


def plot_accuracy_loss(history):
    acc     = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss    = history.history['loss']
    val_loss= history.history['val_loss']
    epochs  = range(len(acc))

    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Validation Accuracy"])
    plt.figure()

    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation Loss"])
    plt.figure()


# # Let's predict stuff!

# ## Predicting tweet category using words

# In[ ]:


accountCateg_detect_train, accountCateg_detect_val, accountCateg_detect_test, unique_accountCateg = make_train_test_val('account_category', 10)
accountCateg2categ, categ2accountCateg = unique2categ(unique_accountCateg)

accountCateg_detect_train['content'] = accountCateg_detect_train['content'].str.lower()
accountCateg_detect_val['content'] = accountCateg_detect_val['content'].str.lower()
accountCateg_detect_test['content'] = accountCateg_detect_test['content'].str.lower()

accountCateg_detect_train['label'] = accountCateg_detect_train['account_category'].apply(lambda r: accountCateg2categ[r])
accountCateg_detect_val['label'] = accountCateg_detect_val['account_category'].apply(lambda r: accountCateg2categ[r])
accountCateg_detect_test['label'] = accountCateg_detect_test['account_category'].apply(lambda r: accountCateg2categ[r])


# In[ ]:


nb_words = 30000
tokenizer = Tokenizer(num_words=nb_words, oov_token='<OOV_TOK>')
train_input, val_input, test_input, max_sequence_len = tokenize_dataset(accountCateg_detect_train, accountCateg_detect_val, accountCateg_detect_test, tokenizer)

nb_classes = len(unique_accountCateg)
train_labels = ku.to_categorical(accountCateg_detect_train.label, num_classes=nb_classes)
val_labels = ku.to_categorical(accountCateg_detect_val.label, num_classes=nb_classes)
test_labels = ku.to_categorical(accountCateg_detect_test.label, num_classes=nb_classes)


# In[ ]:


model = build_model(
    vocab_size=nb_words,
    input_length=max_sequence_len,
    embedding_dim=50,
    nb_classes=nb_classes,
)


# In[ ]:


history = model.fit(
    train_input,
    train_labels,
    epochs=5,
    validation_data=(val_input, val_labels),
    verbose=1
)


# In[ ]:


plot_confusion_matrix(
    model,
    val_input,
    accountCateg_detect_val.label,
    xlabel='Expected Account Category',
    ylabel='Predicted Account Category',
    title='Using a Bi-LSTM with word embeddings',
    suptitle='Account Category detection confusion matrix (Validation)',
    labels=unique_accountCateg,
)


# In[ ]:


plot_accuracy_loss(history)


# What do we get from these plots?
# 
# * We struggle a bit more to differentiate between *Right Troll* and *Left Troll*.
# * The model also struggles a bit with the *Unknown* category, probably because it isn't as homogeneous as the rest.
# * We're pretty good with *comercial* data.
# * The model learns pretty fast but also overfits quite fast (you can see the accuracy/loss keeps improving on training but not validation). When I have more time I'll try to address this issue.

# ## Predicting tweet language using words

# In[ ]:


lang_detect_train, lang_detect_val, lang_detect_test, unique_languages = make_train_test_val('language', 12)
lang2categ, categ2lang = unique2categ(unique_languages)

lang_detect_train['content'] = lang_detect_train['content'].str.lower()
lang_detect_val['content'] = lang_detect_val['content'].str.lower()
lang_detect_test['content'] = lang_detect_test['content'].str.lower()

lang_detect_train['label'] = lang_detect_train['language'].apply(lambda r: lang2categ[r])
lang_detect_val['label'] = lang_detect_val['language'].apply(lambda r: lang2categ[r])
lang_detect_test['label'] = lang_detect_test['language'].apply(lambda r: lang2categ[r])

nb_words = 30000
tokenizer = Tokenizer(num_words=nb_words, oov_token='<OOV_TOK>')
train_input, val_input, test_input, max_sequence_len = tokenize_dataset(lang_detect_train, lang_detect_val, lang_detect_test, tokenizer)

nb_classes = len(unique_languages)
train_labels = ku.to_categorical(lang_detect_train.label, num_classes=nb_classes)
val_labels = ku.to_categorical(lang_detect_val.label, num_classes=nb_classes)
test_labels = ku.to_categorical(lang_detect_test.label, num_classes=nb_classes)

model = build_model(
    vocab_size=nb_words,
    input_length=max_sequence_len,
    embedding_dim=50,
    nb_classes=nb_classes,
)

history = model.fit(
    train_input,
    train_labels,
    epochs=10,
    validation_data=(val_input, val_labels),
    verbose=1
)


# In[ ]:


plot_confusion_matrix(
    model,
    val_input,
    lang_detect_val.label,
    xlabel='Expected language',
    ylabel='Predicted language',
    title='Using a Bi-LSTM with token embeddings',
    suptitle='Language detection confusion matrix (Validation)',
    labels=unique_languages,
)


# In[ ]:


plot_accuracy_loss(history)


# ## Predicting the language of a tweet using characters

# In[ ]:


train_chars = set.union(*lang_detect_train.content.apply(lambda r: set(r)))
val_chars = set.union(*lang_detect_val.content.apply(lambda r: set(r)))
test_chars = set.union(*lang_detect_test.content.apply(lambda r: set(r)))
chars_vocab = sorted(set.union(train_chars, val_chars, test_chars))

print(chars_vocab)
print ('{} unique characters'.format(len(chars_vocab)))


# In[ ]:


char2idx = {u:i for i, u in enumerate(chars_vocab)}
idx2char = np.array(chars_vocab)
for _df in [lang_detect_train, lang_detect_test, lang_detect_val]:
    _df['encoded_chars'] = _df.content.apply(lambda r: np.array([char2idx[c] for c in r][:100]))

train_input = np.array(pad_sequences(lang_detect_train.encoded_chars, maxlen=100, padding='pre'))
val_input = np.array(pad_sequences(lang_detect_val.encoded_chars, maxlen=100, padding='pre'))
test_input = np.array(pad_sequences(lang_detect_test.encoded_chars, maxlen=100, padding='pre'))


# In[ ]:


model = build_model(
    vocab_size=len(chars_vocab),
    embedding_dim=256,
    input_length=100,
    dropout=0.2,
    nb_classes=nb_classes,
)


# In[ ]:


history = model.fit(
    train_input,
    train_labels,
    epochs=16,
    validation_data=(val_input, val_labels),
    verbose=1
)


# In[ ]:


plot_confusion_matrix(
    model,
    val_input,
    lang_detect_val.label,
    xlabel='Expected language',
    ylabel='Predicted language',
    title='Using a Bi-LSTM with character embeddings',
    suptitle='Language detection confusion matrix (Validation)',
    labels=unique_languages,
)


# In[ ]:


plot_accuracy_loss(history)


# ## Predicting a tweet author using words

# In[ ]:


author_detect_train, author_detect_val, author_detect_test, unique_authors = make_train_test_val('author', 20)
author2categ, categ2author = unique2categ(unique_authors)

author_detect_train['content'] = author_detect_train['content'].str.lower()
author_detect_val['content'] = author_detect_val['content'].str.lower()
author_detect_test['content'] = author_detect_test['content'].str.lower()

author_detect_train['label'] = author_detect_train['author'].apply(lambda r: author2categ[r])
author_detect_val['label'] = author_detect_val['author'].apply(lambda r: author2categ[r])
author_detect_test['label'] = author_detect_test['author'].apply(lambda r: author2categ[r])


# In[ ]:


nb_words = 30000
tokenizer = Tokenizer(num_words=nb_words, oov_token='<OOV_TOK>')
train_input, val_input, test_input, max_sequence_len = tokenize_dataset(author_detect_train, author_detect_val, author_detect_test, tokenizer)

nb_classes = len(unique_authors)
train_labels = ku.to_categorical(author_detect_train.label, num_classes=nb_classes)
val_labels = ku.to_categorical(author_detect_val.label, num_classes=nb_classes)
test_labels = ku.to_categorical(author_detect_test.label, num_classes=nb_classes)


# In[ ]:


model = build_model(
    vocab_size=nb_words,
    input_length=max_sequence_len,
    embedding_dim=50,
    nb_classes=nb_classes,
)


# In[ ]:


history = model.fit(
    train_input,
    train_labels,
    epochs=5,
    validation_data=(val_input, val_labels),
    verbose=1
)


# In[ ]:


plot_confusion_matrix(
    model,
    val_input,
    author_detect_val.label,
    xlabel='Expected Author',
    ylabel='Predicted Author',
    title='Using a Bi-LSTM with word embeddings',
    suptitle='Author detection confusion matrix (Validation)',
    labels=unique_authors,
)


# In[ ]:


plot_accuracy_loss(history)


# # Building a custom embedding
# 
# Embeddings are a way to represent tokens as vectors (It can be different/more than words, such as expressions, sentences...), the vector encoding the word/token meaning.   
# Again I won't dwelve too deep on the idea behind word embeddings, but two points you can keep in mind:
# 
# ### How does the model learns the "meaning" of a word? 
# 
# Here we're going to use the word2vec approach (you can also check out GloVe, FastText ...)  
# Let's say we're trying to understand the meaning of the word XXXXXX, as humans.  
# Consider the following sentences:
# 
# > XXXXXX is very sweet and tasty  
# > You can find XXXXXX growing on trees  
# > People use XXXXXX to make alcohol  
# 
# Even if you never heard about XXXXXX before, you can understand we are talking about some kind of fruit.  
# To simplify a lot, Word2Vec generalizes this idea to a whole corpus of text: for each word it will use the word context (the words around it) to infer its meaning.

# In[ ]:


from nltk.tokenize import word_tokenize
from tqdm import tqdm
tqdm.pandas()
sub_df = df[df.language == 'English'].head(100000)
sub_df['tokens'] = sub_df.content.progress_apply(lambda r: [i.lower() for i in word_tokenize(r)])


# ### Word2Vec model building
# 
# A quick overview of the parameters
# 
# * `size`: embedding (vector) size for each token
# * `sg`: algorithm to use, here skip-gram (else: CBOW). CBOW is more similar to the approach explained earlier, skipgram is a bit more complex and tends to be slower but to perform better especially with little data like here.
# * `window`: when looking at a word (to learn its meaning), defines the **context** size (words to the right and the left)
# * `min_count`: we ignore any token present less than min_count times in the corpus. If we have too little examples of a word it will be hard to get a good representation
# * `workers`: unerlated to word2vec, use to parallelise on 10 cores
# 
# One again this is a simplified overview, but if you are interested you can find plenty more littlerature about word2vec online. Don't hesitate to play with the parameters and train your own model!

# In[ ]:


from gensim.models import Word2Vec
model = Word2Vec(sub_df.tokens, size=100, sg=1, window=10, min_count=20, workers=10)


# In[ ]:


print(len(model.wv.vocab))


# ### So what makes word embeddings so interesting
# 
# Well, since you try to encode the words / tokens semantics as a vector, you can actually use mathematical operations on these vectors that hold meaning when you look at the associated words. It will be clearer with examples.

# #### Compute similarity between words
# 
# This, for a pair of words, compares their vector using cosine similarity. The higher the number, the most similar.

# In[ ]:


print('man', 'boy =', model.wv.similarity('man', 'boy'))
print('man', 'woman =', model.wv.similarity('man', 'woman'))
print('man', 'cat =', model.wv.similarity('man', 'cat'))
print('man', 'idea =', model.wv.similarity('man', 'idea'))
print('man', '5 =', model.wv.similarity('man', '5'))


# #### Find the words with the closest meaning
# 
# Same idea as the previous point: for a given word, it will rank the words in the vocabulary from most similar to least similar.  
# Note that:
# 
# * this model is trained on little data so it isn't perfect
# * it is very hard to compare words even for us humans. For example would you say that the word "wine" has more in common with "alcohol" or with "grapes"?

# In[ ]:


model.wv.most_similar('republicans', topn=10)


# In[ ]:


model.wv.most_similar('russia', topn=10)


# #### Some more complex operations
# 
# If for "man" you have the word "boy" when younger, what is the equivalent for "woman"?  
# This can be modelized as:
# 
# ```
# v(WOMAN) + v(BOY) - V(MAN)
# ```
# 
# ![](http://)Meaning, we want a notion that as to do with woman and boy but NOT man, which roughly translates to `"female + adult + male + young - made - adult"` (once again it is a bit more complex but you get the general idea)

# In[ ]:


model.wv.most_similar(positive=['boy', 'woman'], negative=['man'], topn=1)


# An other example: from singular to plural

# In[ ]:


model.wv.most_similar(positive=['men', 'woman'], negative=['man'], topn=1)


# #### One last graph: PCA
# 
# PCA (Principal component analysis) will allow us to project a set of vectors of the N dimensions we have (here 100) to an other number, here 2 for visualization.  
# We're obviously losing a lot of information, but we can still gain some indsight and understanding of what are word embeddings.
# 
# I highlighted some couples, so you can see how close they end up:
# ```
# {'man', 'woman'}
# {'trump', 'obama'}
# {'week', 'year'}
# {'love', 'like'}
# ```

# In[ ]:


from sklearn.decomposition import PCA
import numpy as np

def plot_labels_PCA(model, labels, light_up_labels):
    fig = pl.figure(num=None, figsize=(30, 30), dpi=80, facecolor='w', edgecolor='k')
    X = model[labels]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    fig, ax = pl.subplots(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
    ax.scatter(result[:, 0], result[:, 1])
    for x, y, label in zip(result[:, 0], result[:, 1], labels):
        if label in light_up_labels:
            pl.annotate(
                label,  # this is the text
                (x, y),  # this is the point to label
                textcoords="offset points",  # how to position the text
                xytext=(0, 10),  # distance from text to points (x,y)
                ha='center',
                size=16,
                #color='red' if label in light_up_labels else 'black'
            )  # horizontal alignment can be left, right or center
    pl.show()


# In[ ]:


labels = [i for i in model.wv.vocab if model.wv.vocab[i].count > 500]  # we only keep the most common labels, and this is already a LOT of information to plot... too much
plot_labels_PCA(model, labels, light_up_labels={'man', 'woman', 'trump', 'obama', 'week', 'year', 'love', 'like'})

