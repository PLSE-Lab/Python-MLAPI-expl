#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.io as pio

# Standard plotly imports
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected=True)


# In[ ]:


from collections import Counter
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import MultiLabelBinarizer
import operator 
from collections import defaultdict
import string
from nltk.tokenize.treebank import TreebankWordTokenizer


# In[ ]:





# In[ ]:





# In[ ]:



def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

def build_vocab(sentences, verbose =  True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix


# In[ ]:


def create_aggregrated_plots(feature_name, target, bubble_size_adj=0.1 , aggregration='mean'):

    _tmp_material = train.groupby([target])[feature_name].mean()
    fig = go.Figure()
    color_values = list(_tmp_material.values.astype(int))
    tmp_trace = []

    fig.add_trace(go.Scatter(
        x=_tmp_material.index, y=_tmp_material.values.astype(int),
        marker=dict(
            size=_tmp_material.values.astype(int)*bubble_size_adj,
            color=color_values,
            colorbar=dict(
                title="Colorbar"
            ),
            colorscale="Viridis",
        ),
        mode="lines+markers")
    )

    fig.update_layout(
        title=go.layout.Title(
            text="Average "+feature_name + "/" + target,
        ),
        yaxis=dict(
                title='Average '+feature_name,
            ),
            xaxis=dict(
                title=target,
            )
    )

    fig.show()


# In[ ]:


def target_distribution(df , target, top_counts=None):
    if top_counts:
        topic_counts = df[target].astype(str).value_counts()[:top_counts]
    else:        
        topic_counts = df[target].astype(str).value_counts()
    
    fig = go.Figure([go.Bar(x=topic_counts.index, y=topic_counts.values, 
                            text=topic_counts.values,
                            textposition='auto',
                           marker_color='indianred')])
    fig.update_layout(
            title=go.layout.Title(
                text="Topic Distribution",
            ),
            yaxis=dict(
                    title='Count',
                ),
                xaxis=dict(
                    title="Topic",
                )
        )
    fig.show()


# In[ ]:


def create_wordcloud(df, feature_names, target, target_filter):
    plt.figure(figsize=(25,10))
    for ei, feature_name in enumerate(feature_names):
        text = " ".join(review for review in df[df[target] == target_filter][feature_name])
        #print ("There are {} words in the combination of all {}.".format(len(text), feature_name))

        # Create the wordcloud object
        wordcloud = WordCloud(width=1024, height=480, margin=0).generate(text)

        # Display the generated image:
        plt.subplot(1,2,ei+1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(feature_name)
        plt.margins(x=0, y=0)
    plt.show()


# In[ ]:


train_raw = pd.read_csv('/kaggle/input/amazon-hiring-challenge/2901c100-b-dataset/Dataset/train.csv')
test_raw = pd.read_csv('/kaggle/input/amazon-hiring-challenge/2901c100-b-dataset/Dataset/test.csv')


# In[ ]:


train = train_raw.copy()
test = test_raw.copy()


# In[ ]:


train.shape, test.shape


# In[ ]:


train.head()


# In[ ]:


test.head()


# # Handling Missing Values:

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# # Feature Engineering

# In[ ]:


train['info'] = train['Review Text'] + " " + train['Review Title']
test['info'] = test['Review Text'] + " " + test['Review Title']


train['review_length'] = train['Review Text'].apply(lambda x: len(x))
train['review_word_count'] = train['Review Text'].apply(lambda x: len(x.split()))

train['title_length'] = train['Review Title'].apply(lambda x: len(x))
train['title_word_count'] = train['Review Title'].apply(lambda x: len(x.split()))


# In[ ]:


train.head()


# In[ ]:


train.shape, test.shape


# # TARGET DISTRIBUTION

# In[ ]:


target_distribution(train, 'topic')


# # Length of Review Text , Review Title

# In[ ]:


create_aggregrated_plots('review_length','topic',0.2)


# In[ ]:


create_aggregrated_plots('title_length', 'topic',1)


# # Word Count of Review Text , Review Title

# In[ ]:


create_aggregrated_plots('review_word_count', 'topic', 1)


# In[ ]:


create_aggregrated_plots('title_word_count', 'topic',10)


# In[ ]:


train.title_word_count.max(),train.title_word_count.min() , train.title_word_count.mean()


# # Important words in Review Text, Review Title for 'Bad Taste/Flavor'

# In[ ]:


create_wordcloud(train, ["Review Text", "Review Title"], "topic", 'Bad Taste/Flavor')


# # Important words in Review Text, Review Title for 'Packaging'

# In[ ]:


create_wordcloud(train,['Review Text', 'Review Title'],'topic', 'Packaging')


# # Important words in Review Text, Review Title for 'Expiry'

# In[ ]:


create_wordcloud(train, ['Review Text', 'Review Title'],'topic', 'Expiry')


# # Explore data more...

# In[ ]:


repeated_reviews = train['Review Text'].value_counts()


# In[ ]:


repeated_reviews.head()


# In[ ]:


train[train['Review Text'] == repeated_reviews.index[0]]


# From the above output it is very clear that for same *Review Text* and *Review Title* we have different topics. So we can assume that it is a **multi label classification problem.**

# # Make training data suitable for MULTI LABEL CLASSIFICATION problem.

# In[ ]:


multi_label_data = defaultdict(list)

for key, grp in train.groupby(['Review Text','Review Title']):
    multi_label_data['Review Text'].append(key[0])
    multi_label_data['Review Title'].append(key[1])
    multi_label_data['topic'].append(list(grp.topic.values))


# In[ ]:


multi_label_df = pd.DataFrame(multi_label_data)


# In[ ]:


multi_label_df['info'] = multi_label_df['Review Text'] +" "+ multi_label_df['Review Title']


# In[ ]:


multi_label_df.head()


# In[ ]:


mul_binarizer = MultiLabelBinarizer()
mul_binarizer.fit(multi_label_df.topic)


# In[ ]:


mul_binarizer.classes_


# In[ ]:


y_trans = mul_binarizer.transform(multi_label_df.topic)


# In[ ]:


y_trans


# In[ ]:


target_distribution(multi_label_df, 'topic', 30)


# In[ ]:





# # Pre-trained Embeddings

# In[ ]:


GLOVE_EMBEDDING_FILE = '/kaggle/input/glove840b300dtxt/glove.840B.300d.txt'
FASTTEXT_EMBEDDING_FILE = '/kaggle/input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
WIKI_EMBEDDING_FILE =     '/kaggle/input/wikinews300d1mvec/wiki-news-300d-1M.vec'


# In[ ]:


EMBEDDING_FILES = [
    GLOVE_EMBEDDING_FILE,
    FASTTEXT_EMBEDDING_FILE,
    WIKI_EMBEDDING_FILE
]


# # Do pre-processing by comparing Embedding Coverage

# In[ ]:


glove_embeddings = load_embeddings(GLOVE_EMBEDDING_FILE)
print(f'loaded {len(glove_embeddings)} word vectors ')


# In[ ]:


vocab = build_vocab(list(multi_label_df['info'].apply(lambda x:x.split())))
oov = check_coverage(vocab,glove_embeddings)
oov[:10]


# We can see html tags, let's remove those.

# In[ ]:


def preprocessing(text):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', text)
    return cleantext.split()


# In[ ]:


vocab = build_vocab(list(multi_label_df['info'].apply(preprocessing)))
oov = check_coverage(vocab,glove_embeddings)
oov[:10]


# In[ ]:


tokenizer = TreebankWordTokenizer()


# In[ ]:


def preprocess_2(x):
    x = ' '.join(preprocessing(x))
    x = tokenizer.tokenize(x)
    x = ' '.join(x)
    return x.split()


# In[ ]:


preprocess_2("I didn't know")


# In[ ]:


vocab = build_vocab(list(multi_label_df['info'].apply(preprocess_2)))
oov = check_coverage(vocab,glove_embeddings)
oov[:10]


# In[ ]:


"protein.".replace(".","")


# In[ ]:


def preprocess_3(x):
    x = ' '.join(preprocessing(x))
    x = x.replace('.',' ')
    x = tokenizer.tokenize(x)
    x = ' '.join(x)
    return x.split()


# In[ ]:


vocab = build_vocab(list(multi_label_df['info'].apply(preprocess_3)))
oov = check_coverage(vocab,glove_embeddings)
oov[:10]


# In[ ]:


def preprocess_4(x):
    x = ' '.join(preprocessing(x))
    x = x.replace('.',' ')
    x = tokenizer.tokenize(x)
    x = ' '.join(x)
    x = re.sub('[+/-]', ' ', x)
    return x.split()


# In[ ]:


preprocess_4('pill/capsule/softgel/etc')


# In[ ]:


preprocess_4('softgel-')


# In[ ]:


vocab = build_vocab(list(multi_label_df['info'].apply(preprocess_4)))
oov = check_coverage(vocab,glove_embeddings)
oov[:10]


# In[ ]:


white_list = string.ascii_letters + string.digits + ' '
white_list += "'"


# In[ ]:


white_list


# In[ ]:


glove_chars = ''.join([c for c in tqdm(glove_embeddings) if len(c) == 1])
glove_symbols = ''.join([c for c in glove_chars if not c in white_list])
glove_symbols


# In[ ]:


review_chars = build_vocab(list(multi_label_df["info"]))
review_symbols = ''.join([c for c in review_chars if not c in white_list])
review_symbols


# In[ ]:


symbols_to_delete = ''.join([c for c in review_symbols if not c in glove_symbols])
symbols_to_delete


# In[ ]:





# In[ ]:


def preprocess_5(x):
    x = ' '.join(preprocessing(x))
    x = x.replace('.',' ')
    x = tokenizer.tokenize(x)
    x = ' '.join(x)
    x = re.sub('[+/-]', ' ', x)

    x = "".join([ x[i] for i in range(len(x)) if x[i] not in symbols_to_delete])
    return x


# In[ ]:


preprocess_5("hello i'm")


# In[ ]:


vocab = build_vocab(list(multi_label_df['info'].apply(preprocess_5).str.split()))
oov = check_coverage(vocab,glove_embeddings)
oov[:10]


# In[ ]:





# In[ ]:


unique_topics = mul_binarizer.classes_


# In[ ]:


train = multi_label_df
test = test[~test[['Review Text','Review Title']].duplicated()]


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:





# In[ ]:


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


# In[ ]:





# In[ ]:


X_train = train['info'].apply(preprocess_5)
X_test = test['info'].apply(preprocess_5)


# In[ ]:


X_train.head()


# In[ ]:


from keras.preprocessing import text, sequence


# In[ ]:





# In[ ]:


n_unique_words = None
tokenizer = text.Tokenizer(num_words = n_unique_words)


# In[ ]:





# In[ ]:


tokenizer.fit_on_texts(list(X_train) + list(X_test))


# In[ ]:


X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


# In[ ]:


X_train_len = [len(x) for x in X_train]


# In[ ]:


pd.Series(X_train_len).describe()


# In[ ]:


MAX_LEN = 100

X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_LEN)


# In[ ]:


from keras.utils.np_utils import to_categorical


# In[ ]:


y = y_trans


# In[ ]:


y


# In[ ]:


len(tokenizer.word_index)


# In[ ]:





# In[ ]:


EMBEDDING_FILES


# In[ ]:


embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES[:]], axis=-1)


# In[ ]:


embedding_matrix.shape


# In[ ]:


from keras.models import Sequential, Input, Model
from keras.layers import Dense, Flatten, Dropout, SpatialDropout1D, LSTM, GRU
from keras.layers import add, concatenate, Conv1D, MaxPooling1D, merge, CuDNNLSTM, CuDNNGRU
from keras.layers import Embedding 
from keras.callbacks import ModelCheckpoint 
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.utils import plot_model
from keras.initializers import Constant


# In[ ]:


MAX_WORDS = len(tokenizer.word_index) + 1


# In[ ]:


input_layer = Input(shape=(MAX_LEN,))
embed_layer = Embedding(MAX_WORDS, embedding_matrix.shape[1],   weights=[embedding_matrix], input_length=MAX_LEN , trainable=False)(input_layer)
x = Bidirectional(GRU(64, return_sequences=True, dropout=0.5,recurrent_dropout=0.5))(embed_layer)
x = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)

avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)

x = concatenate([ avg_pool, max_pool])

preds = Dense(len(unique_topics), activation="sigmoid")(x)

model = Model(input_layer, preds)
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

print(model.summary())


# In[ ]:


EPOCHS = 100


# In[ ]:


from keras.callbacks import EarlyStopping, ModelCheckpoint

earlystop = EarlyStopping(monitor='val_loss',patience=5, min_delta=0.0001, verbose=1)
model_checkpoint = ModelCheckpoint(filepath='./model-weights.hdf5', save_best_only=True, monitor='val_loss', verbose=1)

callbacks = [
    earlystop, 
    model_checkpoint
]


# In[ ]:


history = model.fit(X_train, y, 
          batch_size=16, 
          epochs=EPOCHS, 
          verbose=1, 
          validation_split=0.2,
          callbacks= callbacks)


# In[ ]:


plot_model(model, to_file='model.png')


# In[ ]:


fig = go.Figure()
tmp_trace = []

fig.add_trace(go.Scatter(
    x=history.epoch, 
    y=history.history['categorical_accuracy'],
    name='train',

    mode="lines+markers")
)
fig.add_trace(go.Scatter(
    x=history.epoch, 
    y=history.history['val_categorical_accuracy'],
    name='validation',
    mode="lines+markers")
)


fig.update_layout(
    title=go.layout.Title(
        text="Model Accuracy",
    ),
    yaxis=dict(
            title='Categorical Accuracy',
        ),
        xaxis=dict(
            title="Epoch",
        )
)

fig.show()


# In[ ]:


fig = go.Figure()
tmp_trace = []

fig.add_trace(go.Scatter(
    x=history.epoch, 
    y=history.history['loss'],
    name='train',

    mode="lines+markers")
)
fig.add_trace(go.Scatter(
    x=history.epoch, 
    y=history.history['val_loss'],
    name='validation',
    mode="lines+markers")
)


fig.update_layout(
    title=go.layout.Title(
        text="Model Loss",
    ),
    yaxis=dict(
            title='Loss',
        ),
        xaxis=dict(
            title="Epoch",
        )
)

fig.show()


# In[ ]:


model.load_weights('model-weights.hdf5')


# In[ ]:


predict_topics = model.predict(X_test, verbose=1)


# # How to handle top N ?
# > If that review occurs once in the test data, submit the most correlated prediction. If the occurrence is 2, submit the top 2 topics. And so on.

# In[ ]:


output = test_raw[['Review Text', 'Review Title']].copy()
output['topic'] = np.nan


# In[ ]:


for i in range(test.shape[0]):
    review = test.iloc[i]['Review Text']
    title = test.iloc[i]['Review Title']
    output_filter = output[(output['Review Text'] == review) & (output['Review Title'] == title)]
    test_pred = predict_topics[i]
    test_topic = np.argsort(test_pred)[::-1]

    p_topics = [ unique_topics[_] for _ in test_topic][:output_filter.shape[0]]
    
    output.loc[output_filter.index, 'topic'] = p_topics


# In[ ]:


output.head()


# In[ ]:


output.topic.value_counts()


# In[ ]:


output.to_csv("./topic_prediction_multilabel_final_model.csv", index=False)


# In[ ]:




