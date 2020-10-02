#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import GroupKFold
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Input, Embedding, SpatialDropout1D, concatenate, Conv2D, Reshape
from keras.layers import MaxPool2D, PReLU, AvgPool2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from gensim.models.keyedvectors import KeyedVectors
import tensorflow as tf
from functools import partial
import scipy as sp
import glob
import json

from sklearn.metrics import cohen_kappa_score
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
print(os.listdir("../input"))


# In[ ]:


# Load the general data
data_path = "../input/petfinder-adoption-prediction/"
train_df = pd.read_csv(data_path + "train/train.csv")
test_df = pd.read_csv(data_path + "test/test.csv")

# https://www.kaggle.com/christofhenkel/extract-image-features-from-pretrained-nn
train_img = pd.read_csv("../input/petfindermy-dense121-extracted-image-features/train_img_features.csv")
test_img = pd.read_csv("../input/petfindermy-dense121-extracted-image-features/test_img_features.csv")
train_img.rename(columns=lambda i: f"img_{i}" ,inplace=True)
test_img.rename(columns=lambda i: f"img_{i}" ,inplace=True)

train_df = pd.concat([train_df, train_img], axis=1)
test_df = pd.concat([test_df, test_img], axis=1)
df = pd.concat([train_df, test_df], axis=0)
df.head(2)


# In[ ]:


# Load the metadata and sentiment data
train_metadata_files = sorted(glob.glob(data_path + 'train_metadata/*.json'))
train_sentiment_files = sorted(glob.glob(data_path + 'train_sentiment/*.json'))
train_metadata_df = pd.DataFrame(train_metadata_files, columns=["filename"])
train_sentiment_df = pd.DataFrame(train_sentiment_files, columns=["filename"])

test_metadata_files = sorted(glob.glob(data_path + 'test_metadata/*.json'))
test_sentiment_files = sorted(glob.glob(data_path + 'test_sentiment/*.json'))
test_metadata_df = pd.DataFrame(test_metadata_files, columns=["filename"])
test_sentiment_df = pd.DataFrame(test_sentiment_files, columns=["filename"])

sentiment_df = pd.concat([train_sentiment_df, test_sentiment_df])
metadata_df = pd.concat([train_metadata_df, test_metadata_df])
del train_sentiment_df; del test_sentiment_df
del train_metadata_df; del test_metadata_df

def assign_pet_id(df):
    df["PetID"] = df["filename"].apply(lambda x: x.split("/")[-1].split("-")[0].split(".")[0])

assign_pet_id(metadata_df)
assign_pet_id(sentiment_df)


# In[ ]:


def extract_sentiments(row):
    with open(row["filename"], 'r') as f:
        file = json.load(f)
    file_sentiment = file['documentSentiment']
    file_entities = [x['name'] for x in file['entities']]
    file_entities = ' '.join(file_entities)

    file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]

    file_sentences_sentiment = pd.DataFrame.from_dict(
        file_sentences_sentiment, orient='columns').sum()
    file_sentences_sentiment = file_sentences_sentiment.add_prefix('document_').to_dict()

    file_sentiment.update(file_sentences_sentiment)
    
    for key, value in file_sentiment.items():
        row["sentiment_"+key] = value
    return row

sentiment_df = sentiment_df.apply(extract_sentiments, axis=1)
sentiment_df.head()


# In[ ]:


def extract_metadata(row):
    with open(row["filename"], 'r') as f:
        file = json.load(f)
        
    file_keys = list(file.keys())
        
    if 'labelAnnotations' in file_keys:
        file_annots = file['labelAnnotations'][:int(len(file['labelAnnotations']) * 0.3)]
        file_top_score = np.asarray([x['score'] for x in file_annots]).mean()
    else:
        file_top_score = np.nan
        file_top_desc = ['']

    file_colors = file['imagePropertiesAnnotation']['dominantColors']['colors']
    file_crops = file['cropHintsAnnotation']['cropHints']

    file_color_score = np.asarray([x['score'] for x in file_colors]).mean()
    file_color_pixelfrac = np.asarray([x['pixelFraction'] for x in file_colors]).mean()

    file_crop_conf = np.asarray([x['confidence'] for x in file_crops]).mean()

    if 'importanceFraction' in file_crops[0].keys():
        file_crop_importance = np.asarray([x['importanceFraction'] for x in file_crops]).mean()
    else: file_crop_importance = np.nan

    metadata = {
        'annots_score': file_top_score,
        'color_score': file_color_score,
        'color_pixelfrac': file_color_pixelfrac,
        'crop_conf': file_crop_conf,
        'crop_importance': file_crop_importance,
    }
    
    for key, value in metadata.items():
        row["metadata_"+key] = value
        
    return row

metadata_df = metadata_df.apply(extract_metadata, axis=1)
metadata_df.head()


# In[ ]:


# Aggregate sentiments and metadata based on PetID
aggregates = ["sum", "mean"]

metadata_agg = metadata_df.drop(["filename"], axis=1).groupby(["PetID"]).agg(aggregates)
metadata_agg.columns = pd.Index([f"metadata_{c[0]}_{c[1].upper()}" for c in metadata_agg.columns.tolist()])
metadata_agg = metadata_agg.reset_index()

sentiment_agg = sentiment_df.drop(["filename"], axis=1).groupby(["PetID"]).agg(aggregates)
sentiment_agg.columns = pd.Index([f"sentiment_{c[0]}_{c[1].upper()}" for c in sentiment_agg.columns.tolist()])
sentiment_agg = sentiment_agg.reset_index()


# In[ ]:


df = df.merge(metadata_agg, how="left", on="PetID")
df = df.merge(sentiment_agg, how="left", on="PetID")
df.head()


# In[ ]:


word_vec_size = 300
max_words = 100
max_word_features = 25000

def transform_text(text, tokenizer):
    tokenizer.fit_on_texts(text)
    text_emb = tokenizer.texts_to_sequences(text)
    text_emb = sequence.pad_sequences(text_emb, maxlen=max_words)
    return text_emb

desc_tokenizer = text.Tokenizer(num_words=max_word_features)
desc_embs = transform_text(df["Description"].astype(str), desc_tokenizer)


# In[ ]:


text_mode = "word2vec"

if text_mode == "fasttext":
    embedding_file = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"

    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embedding_file))

    word_index = desc_tokenizer.word_index
    print('Word index len:', len(word_index))
    vocabulary_size = min(max_word_features, len(word_index)) + 1
    text_embs = np.zeros((vocabulary_size, word_vec_size))
    for word, i in word_index.items():
        if i >= max_word_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: text_embs[i] = embedding_vector
            
    del(embeddings_index)

elif text_mode == "word2vec":
    embedding_file = "../input/word2vec-google/GoogleNews-vectors-negative300.bin"
    print("Loading word vectors...")
    word_vectors = KeyedVectors.load_word2vec_format(embedding_file, binary=True)

    print("Matching word vectors...")
    EMBEDDING_DIM=300
    word_index = desc_tokenizer.word_index
    vocabulary_size=min(len(word_index)+1,max_word_features)
    text_embs = np.zeros((vocabulary_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i>=max_word_features:
            continue
        try:
            embedding_vector = word_vectors[word]
            text_embs[i] = embedding_vector
        except KeyError:
            text_embs[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)

    del(word_vectors)


# In[ ]:


cat_vars = ["Type", "Breed1", "Breed2", "Color1", "Color2", "Color3", "Gender", "MaturitySize",
            "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "State"]
cont_vars = ["Fee", "PhotoAmt", "VideoAmt", "Age", "Quantity", "NameLength"]

#cont_vars += [c for c in list(df.columns) if "metadata_" in c]
#cont_vars += [c for c in list(df.columns) if "sentiment_" in c]

def preproc(df):
    global cont_vars
#     df["DescriptionLength"] = df["Description"].astype(str).apply(len)
    df["NameLength"] = df["Name"].astype(str).apply(len)
    
    for var in cat_vars:
        df[var] = LabelEncoder().fit_transform(df[var])
        
    for var in cont_vars:
        df[var] = MinMaxScaler().fit_transform(df[var].values.reshape(-1,1))
        df[var].fillna(df[var].mean(), inplace=True)
    
    return df


# In[ ]:


df = preproc(df)
train_df = df[:len(train_df)]
test_df = df[len(train_df):]
len(train_df), len(test_df)
del df
train_df.head(2)


# In[ ]:


def get_keras_data(df, description_embeds):
    X = {var: df[var].values for var in cont_vars+cat_vars}
    X["description"] = description_embeds
    for i in range(256): X[f"img_{i}"] = df[f"img_{i}"]
    return X


# In[ ]:


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


# In[ ]:


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']


# In[ ]:


from keras.callbacks import *

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
    
class QWKEvaluation(Callback):
    def __init__(self, train_data=(), validation_data=(), measure_train=False, interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.history = []
        self.X_val, self.y_val = validation_data
        self.X_train, self.y_train = train_data
        self.measure_train = measure_train
        
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, batch_size=3000, verbose=0)
            y_pred = eval_predict(self.y_val, y_pred)
            val_score = quadratic_weighted_kappa(self.y_val, y_pred)
            if self.measure_train:
                y_pred = self.model.predict(self.X_train, batch_size=3000, verbose=0)
                y_pred = eval_predict(self.y_train, y_pred)
                train_score = quadratic_weighted_kappa(self.y_train, y_pred)
            else: train_score = 0
            print("QWK - epoch: %d - train_score: %.6f, val_score: %.6f \n" % (epoch+1, train_score, val_score))
            self.history.append(val_score)
            if val_score >= max(self.history): self.model.save('checkpoint.h5')

def eval_predict(y=[], y_pred=[], coeffs=None, ret_coeffs=False):
    optR = OptimizedRounder()
    if not coeffs:
        optR.fit(y_pred.reshape(-1,), y)
        coeffs = optR.coefficients()
    if ret_coeffs: return optR.coefficients()
    return optR.predict(y_pred, coeffs).reshape(-1,)


# In[ ]:


# Model inspiration from https://www.kaggle.com/c/avito-demand-prediction/discussion/59917

def rmse(y, y_pred):
    return K.sqrt(K.mean(K.square(y-y_pred), axis=-1))

def get_model(emb_n=10, dout=.25, batch_size=1000):
    inps = []
    embs = [] # Embedding for Categoricals
    nums = [] # Numerical Features
    
    for var in cat_vars:
        inp = Input(shape=[1], name=var)
        inps.append(inp)
        embs.append((Embedding(train_df[var].max()+1, emb_n)(inp)))
    
    for var in cont_vars:
        inp = Input(shape=[1], name=var)
        inps.append(inp)
        nums.append((inp))
    
    desc_inp = Input(shape=(max_words,), name="description")
    inps.append(desc_inp)
    emb_desc = Embedding(vocabulary_size, word_vec_size, weights=[text_embs])(desc_inp)
    emb_desc = SpatialDropout1D(.4)(emb_desc)
    emb_desc = Reshape((max_words, word_vec_size, 1))(emb_desc)
    
    filter_sizes=[1,3]
    convs = []
    for filter_size in filter_sizes:
        conv = Conv2D(32, kernel_size=(filter_size, word_vec_size), 
                        kernel_initializer="normal", activation="relu")(emb_desc)
        convs.append(MaxPool2D(pool_size=(max_words-filter_size+1, 1))(conv))
        
    img_fts = []
    for i in range(256):
        inp = Input(shape=[1], name=f"img_{i}")
        inps.append(inp)
        img_fts.append((inp))
        
    img_fts = concatenate(img_fts)
#     img_fts = Dropout(.25)(img_fts)
    img_fts = BatchNormalization()(img_fts)
    img_fts = Dense(64, activation="relu", kernel_initializer="he_normal")(img_fts)
    img_fts = Dropout(.2)(img_fts)
        
    convs = concatenate(convs)
    convs = Flatten()(convs)
#     convs = Dropout(.25)(convs)
    convs = BatchNormalization()(convs)
    
    
    embs = Flatten()(concatenate(embs))
    embs = Dropout(dout)(Dense(64, activation="relu", kernel_initializer="he_normal")(embs))

    nums = concatenate(nums)
    nums = Dense(32, activation="relu")(nums)
    
    x = concatenate([embs, nums, convs, img_fts])
    x = BatchNormalization()(x)
    
    dense_n = [256, 64]
    for n in dense_n:
        x = BatchNormalization()(x)
        x = Dense(n, activation="relu", kernel_initializer="he_normal")(x)
        
    x = BatchNormalization()(x)
    x = Dropout(.3)(x)
    out = Dense(1, activation="linear")(x)
    
    model = Model(inputs=inps, outputs=out)
    opt = Adam()
    model.compile(optimizer=opt, loss=rmse,)
    return model
    


# In[ ]:


nfolds=5
folds = GroupKFold(n_splits=nfolds)
avg_train_kappa = 0
avg_valid_kappa = 0
batch_size=1000
coeffs=None

X_test = get_keras_data(test_df, desc_embs[len(train_df):])
submission_df = test_df[["PetID"]]
adoptions = np.zeros((len(test_df),))

for train_idx, valid_idx in folds.split(train_df[cat_vars+cont_vars], train_df["AdoptionSpeed"], groups=train_df["RescuerID"]):
    X_train = get_keras_data(train_df.iloc[train_idx], desc_embs[train_idx])
    X_valid = get_keras_data(train_df.iloc[valid_idx], desc_embs[valid_idx])
    y_train, y_valid = train_df["AdoptionSpeed"][train_idx].values, train_df["AdoptionSpeed"][valid_idx].values
    
    model = get_model()
    clr_tri = CyclicLR(base_lr=2e-3, max_lr=4e-2, step_size=len(train_df)//batch_size, mode="triangular2")
    qwk_eval = QWKEvaluation(train_data=(X_train, y_train),validation_data=(X_valid, y_valid), 
                             measure_train=False, interval=1)
    history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_valid, y_valid), 
                        epochs=5, callbacks=[clr_tri, qwk_eval])
    model.load_weights('checkpoint.h5')

    # Softmax prediction to one hot encoding
    y_train_pred = eval_predict(y_train, model.predict(X_train, batch_size=1000))
    y_valid_pred = eval_predict(y_valid, model.predict(X_valid, batch_size=1000))
    avg_train_kappa += quadratic_weighted_kappa(y_train_pred, y_train)/nfolds
    avg_valid_kappa += quadratic_weighted_kappa(y_valid_pred, y_valid)/nfolds
    coeffs = eval_predict(y_valid, model.predict(X_valid, batch_size=1000), ret_coeffs=True)
    adoptions += model.predict(X_test, batch_size=batch_size).reshape(-1,)/nfolds
    
print("\navg train kappa:", avg_train_kappa,)
print("\navg valid kappa:", avg_valid_kappa,)
# Last avg. 0.35371759658545565


# In[ ]:


import matplotlib.pyplot as plt

f = plt.figure(figsize=(10,3))
ax = f.add_subplot(121)
ax2 = f.add_subplot(122)

ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_title('Model loss')
ax.set_xlabel('epoch')
ax.legend(['train', 'valid'], loc='upper left')

ax2.plot(clr_tri.history['iterations'], clr_tri.history['lr'])
ax2.set_title('Learning rate')
ax2.set_xlabel('iteration')


# In[ ]:


# Coeffs gotten from here: https://www.kaggle.com/skooch/petfinder-simple-lgbm-baseline
coeffs[0] = 1.645
coeffs[1] = 2.115
coeffs[3] = 2.84
submission_df["AdoptionSpeed"] = eval_predict(y_pred=adoptions, coeffs=list(coeffs)).astype(int)
submission_df.to_csv("submission.csv", index=False)
submission_df.head()


# In[ ]:




