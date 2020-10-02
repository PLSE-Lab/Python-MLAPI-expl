#!/usr/bin/env python
# coding: utf-8

# Thanks for this https://www.kaggle.com/risntforpirates/petfinder-simple-lgbm and this https://www.kaggle.com/skooch/petfinder-simple-lgbm-baseline main parts.
# 
# And for this picture part: https://www.kaggle.com/christofhenkel/extract-image-features-from-pretrained-nn.
# 
# Just combined two notebooks together to show how to add image features to your lbgm.

# In[ ]:


import json

import scipy as sp
import pandas as pd
import numpy as np

from functools import partial
from math import sqrt

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from collections import Counter

import lightgbm as lgb
np.random.seed(369)


# In[ ]:


# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
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
    
def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# ## Image Features

# In[ ]:


import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm, tqdm_notebook
from keras.applications.densenet import preprocess_input, DenseNet121

train_df = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
img_size = 256
batch_size = 16


# In[ ]:


pet_ids = train_df['PetID'].values
n_batches = len(pet_ids) // batch_size + 1


# In[ ]:


def resize_to_square(im):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im

def load_image(path, pet_id):
    image = cv2.imread(f'{path}{pet_id}-1.jpg')
    new_image = resize_to_square(image)
    new_image = preprocess_input(new_image)
    return new_image


# In[ ]:


from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
inp = Input((256,256,3))
backbone = DenseNet121(input_tensor = inp, 
                       weights="../input/densenet-keras/DenseNet-BC-121-32-no-top.h5",
                       include_top = False)
x = backbone.output
x = GlobalAveragePooling2D()(x)
x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
x = AveragePooling1D(4)(x)
out = Lambda(lambda x: x[:,:,0])(x)

m = Model(inp,out)


# In[ ]:


features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_pets = pet_ids[start:end]
    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
    for i,pet_id in enumerate(batch_pets):
        try:
            batch_images[i] = load_image("../input/petfinder-adoption-prediction/train_images/", pet_id)
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,pet_id in enumerate(batch_pets):
        features[pet_id] = batch_preds[i]


# In[ ]:


train_feats = pd.DataFrame.from_dict(features, orient='index')
train_feats.columns = ['pic_'+str(i) for i in range(train_feats.shape[1])]


# In[ ]:


test_df = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')

pet_ids = test_df['PetID'].values
n_batches = len(pet_ids) // batch_size + 1

features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_pets = pet_ids[start:end]
    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
    for i,pet_id in enumerate(batch_pets):
        try:
            batch_images[i] = load_image("../input/petfinder-adoption-prediction/test_images/", pet_id)
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,pet_id in enumerate(batch_pets):
        features[pet_id] = batch_preds[i]


# In[ ]:


test_feats = pd.DataFrame.from_dict(features, orient='index')
test_feats.columns = ['pic_'+str(i) for i in range(test_feats.shape[1])]


# In[ ]:


test_feats = test_feats.reset_index()
test_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)

train_feats = train_feats.reset_index()
train_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)

test_feats.head()


# In[ ]:


print('Train')
train = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")
print(train.shape)

print('Test')
test = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")
print(test.shape)

print('Breeds')
breeds = pd.read_csv("../input/petfinder-adoption-prediction/breed_labels.csv")
print(breeds.shape)

print('Colors')
colors = pd.read_csv("../input/petfinder-adoption-prediction/color_labels.csv")
print(colors.shape)

print('States')
states = pd.read_csv("../input/petfinder-adoption-prediction/state_labels.csv")
print(states.shape)

target = train['AdoptionSpeed']
train_id = train['PetID']
test_id = test['PetID']


train = pd.merge(train, train_feats, how='left', on='PetID')
test = pd.merge(test, test_feats, how='left', on='PetID')

train.drop(['AdoptionSpeed', 'PetID'], axis=1, inplace=True)
test.drop(['PetID'], axis=1, inplace=True)


# In[ ]:


doc_sent_mag = []
doc_sent_score = []
nf_count = 0
for pet in train_id:
    try:
        with open('../input/petfinder-adoption-prediction/train_sentiment/' + pet + '.json', 'r') as f:
            sentiment = json.load(f)
        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
        doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)

train.loc[:, 'doc_sent_mag'] = doc_sent_mag
train.loc[:, 'doc_sent_score'] = doc_sent_score

doc_sent_mag = []
doc_sent_score = []
nf_count = 0
for pet in test_id:
    try:
        with open('../input/petfinder-adoption-prediction/test_sentiment/' + pet + '.json', 'r') as f:
            sentiment = json.load(f)
        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
        doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)

test.loc[:, 'doc_sent_mag'] = doc_sent_mag
test.loc[:, 'doc_sent_score'] = doc_sent_score


# In[ ]:


## WITHOUT ERROR FIXED
train_desc = train.Description.fillna("none").values
test_desc = test.Description.fillna("none").values

tfv = TfidfVectorizer(min_df=3,  max_features=10000,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words = 'english')
    
# Fit TFIDF
tfv.fit(list(train_desc))
X =  tfv.transform(train_desc)
X_test = tfv.transform(test_desc)
print("X (tfidf):", X.shape)

svd = TruncatedSVD(n_components=200)
svd.fit(X)
# print(svd.explained_variance_ratio_.sum())
# print(svd.explained_variance_ratio_)
X = svd.transform(X)
print("X (svd):", X.shape)

X = pd.DataFrame(X, columns=['svd_{}'.format(i) for i in range(200)])
train = pd.concat((train, X), axis=1)
X_test = svd.transform(X_test)
X_test = pd.DataFrame(X_test, columns=['svd_{}'.format(i) for i in range(200)])
test = pd.concat((test, X_test), axis=1)

print("train:", train.shape)


# In[ ]:


vertex_xs = []
vertex_ys = []
bounding_confidences = []
bounding_importance_fracs = []
dominant_blues = []
dominant_greens = []
dominant_reds = []
dominant_pixel_fracs = []
dominant_scores = []
label_descriptions = []
label_scores = []
nf_count = 0
nl_count = 0
for pet in train_id:
    try:
        with open('../input/petfinder-adoption-prediction/train_metadata/' + pet + '-1.json', 'r') as f:
            data = json.load(f)
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_xs.append(vertex_x)
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        vertex_ys.append(vertex_y)
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_confidences.append(bounding_confidence)
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        bounding_importance_fracs.append(bounding_importance_frac)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_blues.append(dominant_blue)
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_greens.append(dominant_green)
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_reds.append(dominant_red)
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_pixel_fracs.append(dominant_pixel_frac)
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        dominant_scores.append(dominant_score)
        if data.get('labelAnnotations'):
            label_description = data['labelAnnotations'][0]['description']
            label_descriptions.append(label_description)
            label_score = data['labelAnnotations'][0]['score']
            label_scores.append(label_score)
        else:
            nl_count += 1
            label_descriptions.append('nothing')
            label_scores.append(-1)
    except FileNotFoundError:
        nf_count += 1
        vertex_xs.append(-1)
        vertex_ys.append(-1)
        bounding_confidences.append(-1)
        bounding_importance_fracs.append(-1)
        dominant_blues.append(-1)
        dominant_greens.append(-1)
        dominant_reds.append(-1)
        dominant_pixel_fracs.append(-1)
        dominant_scores.append(-1)
        label_descriptions.append('nothing')
        label_scores.append(-1)

print(nf_count)
print(nl_count)
train.loc[:, 'vertex_x'] = vertex_xs
train.loc[:, 'vertex_y'] = vertex_ys
train.loc[:, 'bounding_confidence'] = bounding_confidences
train.loc[:, 'bounding_importance'] = bounding_importance_fracs
train.loc[:, 'dominant_blue'] = dominant_blues
train.loc[:, 'dominant_green'] = dominant_greens
train.loc[:, 'dominant_red'] = dominant_reds
train.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
train.loc[:, 'dominant_score'] = dominant_scores
train.loc[:, 'label_description'] = label_descriptions
train.loc[:, 'label_score'] = label_scores


vertex_xs = []
vertex_ys = []
bounding_confidences = []
bounding_importance_fracs = []
dominant_blues = []
dominant_greens = []
dominant_reds = []
dominant_pixel_fracs = []
dominant_scores = []
label_descriptions = []
label_scores = []
nf_count = 0
nl_count = 0
for pet in test_id:
    try:
        with open('../input/petfinder-adoption-prediction/test_metadata/' + pet + '-1.json', 'r') as f:
            data = json.load(f)
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_xs.append(vertex_x)
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        vertex_ys.append(vertex_y)
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_confidences.append(bounding_confidence)
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        bounding_importance_fracs.append(bounding_importance_frac)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_blues.append(dominant_blue)
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_greens.append(dominant_green)
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_reds.append(dominant_red)
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_pixel_fracs.append(dominant_pixel_frac)
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        dominant_scores.append(dominant_score)
        if data.get('labelAnnotations'):
            label_description = data['labelAnnotations'][0]['description']
            label_descriptions.append(label_description)
            label_score = data['labelAnnotations'][0]['score']
            label_scores.append(label_score)
        else:
            nl_count += 1
            label_descriptions.append('nothing')
            label_scores.append(-1)
    except FileNotFoundError:
        nf_count += 1
        vertex_xs.append(-1)
        vertex_ys.append(-1)
        bounding_confidences.append(-1)
        bounding_importance_fracs.append(-1)
        dominant_blues.append(-1)
        dominant_greens.append(-1)
        dominant_reds.append(-1)
        dominant_pixel_fracs.append(-1)
        dominant_scores.append(-1)
        label_descriptions.append('nothing')
        label_scores.append(-1)

print(nf_count)
test.loc[:, 'vertex_x'] = vertex_xs
test.loc[:, 'vertex_y'] = vertex_ys
test.loc[:, 'bounding_confidence'] = bounding_confidences
test.loc[:, 'bounding_importance'] = bounding_importance_fracs
test.loc[:, 'dominant_blue'] = dominant_blues
test.loc[:, 'dominant_green'] = dominant_greens
test.loc[:, 'dominant_red'] = dominant_reds
test.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
test.loc[:, 'dominant_score'] = dominant_scores
test.loc[:, 'label_description'] = label_descriptions
test.loc[:, 'label_score'] = label_scores


# In[ ]:


train.drop(['Name', 'RescuerID', 'Description'], axis=1, inplace=True)
test.drop(['Name', 'RescuerID', 'Description'], axis=1, inplace=True)


# In[ ]:


numeric_cols = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 
                'doc_sent_mag', 'doc_sent_score', 'dominant_score', 'dominant_pixel_frac', 
                'dominant_red', 'dominant_green', 'dominant_blue', 'bounding_importance', 
                'bounding_confidence', 'vertex_x', 'vertex_y', 'label_score'] +\
               [col for col in train.columns if col.startswith('pic') or col.startswith('svd')]
cat_cols = list(set(train.columns) - set(numeric_cols))
train.loc[:, cat_cols] = train[cat_cols].astype('category')
test.loc[:, cat_cols] = test[cat_cols].astype('category')
print(train.shape)
print(test.shape)

# get the categorical features
foo = train.dtypes
cat_feature_names = foo[foo == "category"]
cat_features = [train.columns.get_loc(c) for c in train.columns if c in cat_feature_names]


# In[ ]:


train = train.drop('label_description',axis=1)
test= test.drop('label_description', axis=1)


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


col_vals_dict= {col: list(train[col].unique()) for col in train.select_dtypes(['category'])}
embed_cols = []
for c in col_vals_dict:
    if len(col_vals_dict[c])>=2:
        embed_cols.append(c)
        print(c + ': %d values' % len(col_vals_dict[c])) #look at value counts to know the embedding dimensions  


# In[ ]:


num_train = train.select_dtypes(['float64','int64'])
num_train.head()


# In[ ]:


pic_cols = [col for col in num_train.columns if col.startswith('pic')]
svd_cols = [col for col in num_train.columns if col.startswith('svd')]
meta_cols = ['vertex_x', 'vertex_y', 'bounding_confidence', 'bounding_importance',
             'dominant_blue','dominant_green', 'dominant_red' ,'dominant_pixel_frac' , 'dominant_score','label_score']
sent_cols = ['doc_sent_mag','doc_sent_score']
othernum_cols = list(set(num_train.columns) - set(meta_cols)- set(pic_cols)- set(svd_cols)- set(sent_cols))


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
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.history = []
        self.X_val, self.y_val = validation_data
        
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, batch_size=1000, verbose=0)
            y_pred = eval_predict(self.y_val, y_pred)
            score = quadratic_weighted_kappa(self.y_val, y_pred)
            print("QWK - epoch: %d - score: %.6f \n" % (epoch+1, score))
            self.history.append(score)
            if score >= max(self.history): self.model.save('checkpoint.h5')
def eval_predict(y=[], y_pred=[], coeffs=None, ret_coeffs=False):
    optR = OptimizedRounder()
    if not coeffs:
        optR.fit(y_pred.reshape(-1,), y)
        coeffs = optR.coefficients()
    if ret_coeffs: return optR.coefficients()
    return optR.predict(y_pred, coeffs).reshape(-1,)


# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dropout, Dense, BatchNormalization, Lambda, concatenate, GRU, Embedding, Flatten, add, multiply, GlobalMaxPool1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
from keras import optimizers
from keras import initializers
import tensorflow as tf

seed = 2019
np.random.seed(seed)
tf.set_random_seed(seed)

def get_model(dr=0.15, seed=2019):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    sess = tf.Session()
    K.set_session(sess)
    
    #embedding layers
    inputs = []
    embeddings = []
    
    input_type = Input(shape=[1], name='Type')
    embedding = Embedding(2,1, input_length=1)(input_type)
    embedding = Reshape(target_shape=(1,))(embedding)
    inputs.append(input_type)
    embeddings.append(embedding)
    
    input_breed1 = Input(shape=[1],name= 'Breed1')
    embedding = Embedding(176,10, input_length=1)(input_breed1)
    embedding = Reshape(target_shape=(10,))(embedding)
    inputs.append(input_breed1)
    embeddings.append(embedding)
    
    input_breed2 = Input(shape=[1], name= 'Breed2')
    embedding = Embedding(135,10, input_length=1)(input_breed2)
    embedding = Reshape(target_shape=(10,))(embedding)
    inputs.append(input_breed2)
    embeddings.append(embedding)
    
    input_gender = Input(shape=[1], name= 'Gender')
    embedding = Embedding(3,2, input_length=1)(input_gender)
    embedding = Reshape(target_shape=(2,))(embedding)
    inputs.append(input_gender)
    embeddings.append(embedding)
    
    input_color1 = Input(shape=[1], name= 'Color1')
    embedding = Embedding(7,4, input_length=1)(input_color1)
    embedding = Reshape(target_shape=(4,))(embedding)
    inputs.append(input_color1)
    embeddings.append(embedding)
    
    input_color2 = Input(shape=[1], name= 'Color2')
    embedding = Embedding(7,4, input_length=1)(input_color2)
    embedding = Reshape(target_shape=(4,))(embedding)
    inputs.append(input_color2)
    embeddings.append(embedding)
    
    input_color3 = Input(shape=[1], name= 'Color3')
    embedding = Embedding(6,3, input_length=1)(input_color3)
    embedding = Reshape(target_shape=(3,))(embedding)
    inputs.append(input_color3)
    embeddings.append(embedding)
    
    input_maturity_size = Input(shape=[1], name= 'MaturitySize')
    embedding = Embedding(4,2, input_length=1)(input_maturity_size)
    embedding = Reshape(target_shape=(2,))(embedding)
    inputs.append(input_maturity_size)
    embeddings.append(embedding)
    
    input_fur_length = Input(shape=[1], name= 'FurLength')
    embedding = Embedding(3,2, input_length=1)(input_fur_length)
    embedding = Reshape(target_shape=(2,))(embedding)
    inputs.append(input_fur_length)
    embeddings.append(embedding)
    
    input_vaccinated = Input(shape=[1], name= 'Vaccinated')
    embedding = Embedding(3,2, input_length=1)(input_vaccinated)
    embedding = Reshape(target_shape=(2,))(embedding)
    inputs.append(input_vaccinated)
    embeddings.append(embedding)
    
    input_dewormed = Input(shape=[1], name='Dewormed')
    embedding = Embedding(3,2, input_length=1)(input_vaccinated)
    embedding = Reshape(target_shape=(2,))(embedding)
    inputs.append(input_dewormed)
    embeddings.append(embedding)
    
    input_sterilized = Input(shape=[1], name='Sterilized')
    embedding = Embedding(3,2, input_length=1)(input_sterilized)
    embedding = Reshape(target_shape=(2,))(embedding)
    inputs.append(input_sterilized)
    embeddings.append(embedding)
    
    input_health = Input(shape=[1], name='Health')
    embedding = Embedding (3,2, input_length=1)(input_health)
    embedding = Reshape(target_shape=(2,))(embedding)
    inputs.append(input_health)
    embeddings.append(embedding)
    
    input_state = Input(shape=[1], name='State')
    embedding = Embedding (14,7, input_length=1)(input_health)
    embedding = Reshape(target_shape=(7,))(embedding)
    inputs.append(input_state)
    embeddings.append(embedding)
    
    # numeric cols & batch normalization   
    input_pic = Input(shape=[num_train[pic_cols].shape[1]], name='pic_col')
    bn_pic = BatchNormalization()(input_pic)
    inputs.append(input_pic)
    embeddings.append(bn_pic)
    
    input_svd = Input(shape=[num_train[svd_cols].shape[1]], name='svd_col')
    bn_svd = BatchNormalization()(input_svd)
    inputs.append(input_svd)
    embeddings.append(bn_svd)
    
    input_meta = Input(shape=[num_train[meta_cols].shape[1]], name='meta_col')
    bn_meta = BatchNormalization()(input_meta)
    inputs.append(input_meta)
    embeddings.append(bn_meta)
    
    input_sent = Input(shape=[num_train[sent_cols].shape[1]], name='sent_col')
    bn_sent = BatchNormalization()(input_sent)
    inputs.append(input_sent)
    embeddings.append(bn_sent)
    
    input_other = Input(shape=[num_train[othernum_cols].shape[1]], name='other_col')
    bn_other = BatchNormalization()(input_other)
    inputs.append(input_other)
    embeddings.append(bn_other)
    
    # model compiling 
    
    main_l= Concatenate()(embeddings)
    main_l = BatchNormalization()(main_l)
    main_l= Dropout(dr)(Dense(250, activation = 'relu')(main_l))
    #main_l = BatchNormalization()(main_l)
    main_l= Dropout(dr)(Dense(100, activation = 'relu')(main_l))
    #main_l = BatchNormalization()(main_l)
    main_l = Dense(50, activation='relu')(main_l)
    #main_l = BatchNormalization()(main_l)
    main_l= Dense(10, activation = 'relu')(main_l)
    #main_l = BatchNormalization()(main_l)
           
    #output
    output = Dense(1, activation = 'linear')(main_l)
    
    #model
    model = Model(inputs, output)
    #optimizer = optimizers.Adam(lr=0.01, decay=3e-5)
    optimizer= optimizers.Adam()
    model.compile(loss='logcosh', optimizer=optimizer)

    return model, sess


# In[ ]:


def preproc(df):
    input_list = []
    #the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(df[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i       
        input_list.append(df[c].map(val_map).fillna(0).values)
     
    #the rest of the columns
  
    input_list.append(df[pic_cols].values)
    input_list.append(df[svd_cols].values)
    input_list.append(df[meta_cols].values)
    input_list.append(df[sent_cols].values)
    input_list.append(df[othernum_cols].values)

    return input_list


# In[ ]:


epochs = 5
BATCH_SIZE = 1000
N_SPLITS = 5
model, sess = get_model()
K.set_session(sess)


def run_cv_model(train, target, model_fn, eval_fn=None, label='model'):
    kf = StratifiedKFold(n_splits=N_SPLITS, random_state=1906, shuffle=True)
    fold_splits = kf.split(train, target)
    cv_scores = []
    qwk_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0],))
    all_coefficients = np.zeros((N_SPLITS, 4))
    feature_importance_df = pd.DataFrame()
    i = 1
    for train_index, val_index in fold_splits:
        print('Started ' + label + ' fold ' + str(i) + '/' + str(N_SPLITS))
        X_train = preproc(train.iloc[train_index])
        X_val = preproc(train.iloc[val_index])
        y_train = target.iloc[train_index].values
        y_val = target.iloc[val_index].values
        X_test = preproc(test.copy())
        pred_val_y, pred_test_y, coefficients, qwk = model_fn(X_train, X_val, y_train, y_val,X_test)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        all_coefficients[i-1, :] = coefficients
        if eval_fn is not None:
            cv_score = eval_fn(y_val, pred_val_y)
            cv_scores.append(cv_score)
            qwk_scores.append(qwk)
            print(label + ' cv score {}: RMSE {} QWK {}'.format(i, cv_score, qwk))           
        i += 1
    print('{} cv RMSE scores : {}'.format(label, cv_scores))
    print('{} cv mean RMSE score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std RMSE score : {}'.format(label, np.std(cv_scores)))
    print('{} cv QWK scores : {}'.format(label, qwk_scores))
    print('{} cv mean QWK score : {}'.format(label, np.mean(qwk_scores)))
    print('{} cv std QWK score : {}'.format(label, np.std(qwk_scores)))
    pred_full_test = pred_full_test / float(N_SPLITS)
    results = {'label': label,
               'train': pred_train,'cv': cv_scores, 'qwk': qwk_scores,'test': pred_full_test,
               'coefficients': all_coefficients}
    return results


def runNN(X_train, X_val, y_train, y_val, X_test):
    model, sess = get_model()
    K.set_session(sess)
    
    clr_tri = CyclicLR(base_lr=2e-3, max_lr=4e-2, step_size=len(X_train)//batch_size, mode="triangular2")
    qwk_eval = QWKEvaluation(validation_data=(X_val, y_val), interval=1)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),callbacks=[clr_tri, qwk_eval])
            
    print('Predict 1/2')
    y_pred = model.predict(X_val,batch_size=BATCH_SIZE).ravel()
    optR = OptimizedRounder()
    optR.fit(y_pred, y_val)
    coefficients = optR.coefficients()
    y_pred_k = optR.predict(y_pred, coefficients)
    print("Valid Counts = ", Counter(y_val))
    print("Predicted Counts = ", Counter(y_pred_k))
    print("Coefficients = ", coefficients)
    qwk = quadratic_weighted_kappa(y_val, y_pred_k)
    print("QWK = ", qwk)
    print('Predict 2/2')
    test_pred = model.predict(X_test,batch_size=BATCH_SIZE).ravel()
    return y_pred, test_pred, coefficients, qwk

results = run_cv_model(train, target, runNN, rmse, 'NN')


# In[ ]:


optR = OptimizedRounder()
coefficients_ = np.mean(results['coefficients'], axis=0)
print(coefficients_)


# In[ ]:


coefficients_[0] = 1.645
coefficients_[1] = 2.115
coefficients_[3] = 2.84
train_predictions = results['train']
train_predictions = optR.predict(train_predictions, coefficients_).astype(int)
Counter(train_predictions)
print(quadratic_weighted_kappa(train_predictions,target))


# In[ ]:


optR = OptimizedRounder()
coefficients_ = np.mean(results['coefficients'], axis=0)
print(coefficients_)
# manually adjust coefs
coefficients_[0] = 1.645
coefficients_[1] = 2.115
coefficients_[3] = 2.84
test_predictions = results['test']
test_predictions = optR.predict(test_predictions, coefficients_).astype(int)
Counter(test_predictions)


# In[ ]:


print("True Distribution:")
print(pd.value_counts(target, normalize=True).sort_index())
print("Test Predicted Distribution:")
print(pd.value_counts(test_predictions, normalize=True).sort_index())
print("Train Predicted Distribution:")
print(pd.value_counts(train_predictions, normalize=True).sort_index())


# In[ ]:


pd.DataFrame(sk_cmatrix(target, train_predictions), index=list(range(5)), columns=list(range(5)))


# In[ ]:


quadratic_weighted_kappa(target, train_predictions)
rmse(target, results['train'])
submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)

