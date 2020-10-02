#!/usr/bin/env python
# coding: utf-8

# ## 13-th place solution: 0.44091 on private LB
# 
# I will share my 13-th place solution. 
# At first, sorry but the code is not so organized & quite messy (it contains all my effort during 3 months).
# 
# Summary of the approach is written at [13-th place solution summary 0.44091 (65-th on public LB: 0.459~0.467)](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/87733). Please refer it for the detailed explanation.
# 
# 
# ### Model
# This is ensemble of these 5 models:
#  - XGBoost
#  - LightGBM
#  - CatBoost
#  - xlearn
#  - Neural Network: xDeepFM based model
#    - xDeepFM is a network for sparse, categorical dataset.
#    - Refer: "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems", https://arxiv.org/abs/1803.05170
#  
# ### External dataset
# 
# 
# ### Library
# I used some library which is/was not supported in the kaggle default docker.
# 
#  - [xlearn](https://github.com/aksnzhy/xlearn): As explained in https://www.kaggle.com/bminixhofer/xlearn, we can install library as an "external dataset".
#  - [optuna](https://github.com/pfnet/optuna): I used it for hyper parameter tuning during local development (not used in the final code). It seems [now supported](https://github.com/Kaggle/docker-python/pull/486).
#  - [chainer_chemistry](https://github.com/pfnet-research/chainer-chemistry): I used it as an extension of Chainer, for writing neural network part. I am sending [PR to support it in default docker](https://github.com/Kaggle/docker-python/pull/447).
#  In this kernel, I copied some of the module/code from the library to use it.
# 
#  - [pfnet-research/sngan_projection](https://github.com/pfnet-research/sngan_projection): Some of the codes are copied from this repository to use spectral normalization for regularization of Neural Network.
# 
# ### External dataset
#  - Petfinder.my: this competition's dataset
#  - Glove embedding feature: text embedding extraction
#  - Keras DenseNet Weights: image feature extraction
#  - Cat and dog breeds parameters: breed feature extraction
#  - xlearn: to install xlearn library
#  
#  - Malaysia GDP & population: hard coded.
# 

# In[2]:


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


# In[3]:


# To check the total kernel running time in detail, 2 hours limit for GPU kernel for petfinder.my competition.
from time import time

start_time = time()


# In[ ]:





# Install xlearn: it takes several minutes.

# In[4]:


import os
import subprocess

os.environ['USER'] = 'root'
os.system('pip install ../input/xlearn/xlearn/xlearn-0.40a1/')
#result = subprocess.check_output('pip install ../input/xlearn/xlearn/xlearn-0.40a1/', shell=True)

import xlearn as xl


# In[5]:


xl.hello()


# In[6]:


debug = False

print(os.listdir("../input/petfinder-adoption-prediction"))


# In[7]:


is_kaggle_kernel = True
pet_dir = '../input/petfinder-adoption-prediction'
bert_dir = '../input/uncased-l12-h768-a12-bert-config-json'
vgg16_dir = '../input/vgg16-chainercv'
json_dir = '../input/cat-and-dog-breeds-parameters'
glove_dir = '../input/glove-global-vectors-for-word-representation'
densenet_dir = '../input/densenet-keras'
cute_dir = '../input/cute-cats-and-dogs-from-pixabaycom'


# In[8]:


# --- utils ---


# In[9]:


from contextlib import contextmanager
from time import perf_counter
import os

import numpy


@contextmanager
def timer(name):
    t0 = perf_counter()
    yield
    t1 = perf_counter()
    print('[{}] done in {:.3f} s'.format(name, t1-t0))


def _check_path_exist(filepath):
    if not os.path.exists(filepath):
        raise IOError('{} not found'.format(filepath))


def save_npz(filepath, datasets):
    if not isinstance(datasets, (list, tuple)):
        datasets = (datasets, )
    numpy.savez(filepath, *datasets)


def load_npz(filepath):
    _check_path_exist(filepath)
    load_data = numpy.load(filepath)
    result = []
    i = 0
    while True:
        key = 'arr_{}'.format(i)
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    if len(result) == 1:
        result = result[0]
    return result


# In[10]:


# --- preprocessing ---


# In[11]:


import json
import os
from glob import glob
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score


import sys
import os



def prepare_df(debug=True, animal_type=None):
    train = pd.read_csv(os.path.join(pet_dir, "train/train.csv"))
    print('Train', train.shape)

    test = pd.read_csv(os.path.join(pet_dir, "test/test.csv"))
    print('Test', test.shape)

    breeds = pd.read_csv(os.path.join(pet_dir, "breed_labels.csv"))
    print('Breeds', breeds.shape)

    colors = pd.read_csv(os.path.join(pet_dir, "color_labels.csv"))
    print('Colors', colors.shape)

    states = pd.read_csv(os.path.join(pet_dir, "state_labels.csv"))
    print('States', states.shape)

    if debug:
        train = train[:1000]
        test = test[:500]

    if animal_type is not None:
        assert animal_type == 1 or animal_type == 2
        # Only train dog or cat...
        print('Only use type = {}'.format(animal_type))
        train = train[train['Type'] == animal_type]
        test = test[test['Type'] == animal_type]

    return train, test, breeds, colors, states


def process_sentiment(petid, dataset_type='train', num_text=0):
    """preprocessing for sentiment json

    Args:
        petid (str): petid
        dataset_type (str): train or test
        num_text (int): First `num_text` text sentiment is extracted.
            If 0, only global sentiment is extracted.

    Returns:
        doc_features (list): [`doc_sent_mag`, `doc_sent_score`] which stores
            magnitude & score of sentiment.
    """
    json_filepath = '{}/{}_sentiment/{}.json'.format(pet_dir, dataset_type, petid)
    ndim = 2 * (num_text + 1)
    feat = np.zeros((ndim,), dtype=np.float32)
    if os.path.exists(json_filepath):
        with open(json_filepath, 'r', encoding='utf-8') as f:
            sentiment = json.load(f)
        doc_sent = sentiment['documentSentiment']
        feat[0] = doc_sent['magnitude']
        feat[1] = doc_sent['score']
        doc_sent_list = sentiment['sentences']
        for i in range(num_text):
            if i == len(doc_sent_list):
                break
            current_index = 2 * (i + 1)
            feat[current_index] = doc_sent_list[i]['sentiment']['magnitude']
            feat[current_index + 1] = doc_sent_list[i]['sentiment']['score']
    return feat


def process_metadata(petid, dataset_type='train'):
    """preprocess for image metadata json
    Args:
        petid (str): petid
        dataset_type (str): train or test

    Returns:
        metadata_features (list):
            [vertex_x, vertex_y, bounding_confidence, bounding_importance_frac,
             dominant_blue, dominant_green, dominant_red, dominant_pixel_frac, dominant_score,
             label_description, label_score]
    """
    # Only check first image...
    json_filepath = '/{}_metadata/{}-1.json'.format(pet_dir, dataset_type, petid)
    if os.path.exists(json_filepath):
        with open(json_filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        crop_hints0 = metadata['cropHintsAnnotation']['cropHints'][0]
        vertex = crop_hints0['boundingPoly']['vertices'][2]
        vertex_x = vertex['x']
        vertex_y = vertex['y']
        bounding_confidence = crop_hints0['confidence']
        bounding_importance_frac = crop_hints0.get('importanceFraction', -1)
        colors0 = metadata['imagePropertiesAnnotation']['dominantColors']['colors'][0]
        dominant_color = colors0['color']
        dominant_blue = dominant_color['blue']
        dominant_green = dominant_color['green']
        dominant_red = dominant_color['red']
        dominant_pixel_frac = colors0['pixelFraction']
        dominant_score = colors0['score']
        if metadata.get('labelAnnotations'):
            label_description = metadata['labelAnnotations'][0]['description']
            label_score = metadata['labelAnnotations'][0]['score']
        else:
            label_description = 'nothing'
            label_score = -1

        # TODO: how to return label_description??
        return [vertex_x, vertex_y, bounding_confidence, bounding_importance_frac,
                dominant_blue, dominant_green, dominant_red, dominant_pixel_frac, dominant_score,
                label_score]         # label_description
    else:
        return [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


def add_gdp(df):
    """Add GDP & population inplace."""
    # Copied from https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/78040
    # state GDP: https://en.wikipedia.org/wiki/List_of_Malaysian_states_by_GDP
    state_gdp = {
        41336: 116.679,
        41325: 40.596,
        41367: 23.02,
        41401: 190.075,
        41415: 5.984,
        41324: 37.274,
        41332: 42.389,
        41335: 52.452,
        41330: 67.629,
        41380: 5.642,
        41327: 81.284,
        41345: 80.167,
        41342: 121.414,
        41326: 280.698,
        41361: 32.270
    }

    # state population: https://en.wikipedia.org/wiki/Malaysia
    state_population = {
        41336: 33.48283,
        41325: 19.47651,
        41367: 15.39601,
        41401: 16.74621,
        41415: 0.86908,
        41324: 8.21110,
        41332: 10.21064,
        41335: 15.00817,
        41330: 23.52743,
        41380: 2.31541,
        41327: 15.61383,
        41345: 32.06742,
        41342: 24.71140,
        41326: 54.62141,
        41361: 10.35977
    }
    df["state_gdp"] = df.State.map(state_gdp)
    df["state_population"] = df.State.map(state_population)


def add_tfidf(train, test, tfidf_svd_components=120):
    train_desc = train.Description.values
    test_desc = test.Description.values
    # TF-IDF value
    s = perf_counter()
    tfv = TfidfVectorizer(
        min_df=2, max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, )
    # tfv = TfidfVectorizer(
    #     min_df=3,  max_features=10000,
    #     strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
    #     ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
    #     stop_words='english')
    tfv.fit(list(train_desc))
    train_x_tfidf = tfv.transform(train_desc)
    test_x_tfidf = tfv.transform(test_desc)
    e = perf_counter()
    print('Tfidfvectorizer done {} sec train_desc {}, test_desc {}'
          .format(e - s, train_x_tfidf.shape, test_x_tfidf.shape))

    s = perf_counter()
    svd = TruncatedSVD(n_components=tfidf_svd_components)
    svd.fit(train_x_tfidf)
    train_x_tfidf_svd = svd.transform(train_x_tfidf).astype(np.float32)
    test_x_tfidf_svd = svd.transform(test_x_tfidf).astype(np.float32)
    e = perf_counter()
    print('TruncatedSVD done {} sec, train_x_tfidf_svd {}'
          .format(e - s, train_x_tfidf_svd.shape))
    return train_x_tfidf_svd, test_x_tfidf_svd


def parse_image_size():
    # https://www.kaggle.com/ranjoranjan/single-xgboost-model
    from PIL import Image
    # train_df_ids = train[['PetID']]
    # test_df_ids = test[['PetID']]

    train_image_files = sorted(glob('{}/train_images/*.jpg'.format(pet_dir)))
    test_image_files = sorted(glob('{}/test_images/*.jpg'.format(pet_dir)))
    split_char = '/'

    train_df_imgs = pd.DataFrame(train_image_files)
    train_df_imgs.columns = ['image_filename']
    train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])

    test_df_imgs = pd.DataFrame(test_image_files)
    test_df_imgs.columns = ['image_filename']
    test_imgs_pets = test_df_imgs['image_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])

    train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)
    test_df_imgs = test_df_imgs.assign(PetID=test_imgs_pets)

    def getSize(filename):
        st = os.stat(filename)
        return st.st_size

    def getDimensions(filename):
        img_size = Image.open(filename).size
        return img_size

    from joblib import Parallel, delayed
    n_jobs = 16
    results = Parallel(n_jobs, verbose=1)(
        delayed(getDimensions)(filepath) for filepath in train_image_files)
    results = np.asarray(results)
    print('results', results.shape)

    train_df_imgs['image_size'] = train_df_imgs['image_filename'].apply(getSize)
    # train_df_imgs['temp_size'] = train_df_imgs['image_filename'].apply(getDimensions)
    # train_df_imgs['width'] = train_df_imgs['temp_size'].apply(lambda x: x[0])
    # train_df_imgs['height'] = train_df_imgs['temp_size'].apply(lambda x: x[1])
    train_df_imgs['width'] = results[:, 0]
    train_df_imgs['height'] = results[:, 1]
    # train_df_imgs = train_df_imgs.drop(['temp_size'], axis=1)

    results = Parallel(n_jobs, verbose=1)(
        delayed(getDimensions)(filepath) for filepath in test_image_files)
    results = np.asarray(results)

    test_df_imgs['image_size'] = test_df_imgs['image_filename'].apply(getSize)
    # test_df_imgs['temp_size'] = test_df_imgs['image_filename'].apply(getDimensions)
    # test_df_imgs['width'] = test_df_imgs['temp_size'].apply(lambda x: x[0])
    # test_df_imgs['height'] = test_df_imgs['temp_size'].apply(lambda x: x[1])
    test_df_imgs['width'] = results[:, 0]
    test_df_imgs['height'] = results[:, 1]
    # test_df_imgs = test_df_imgs.drop(['temp_size'], axis=1)

    aggs = {
        'image_size': ['sum', 'mean', 'var'],
        'width': ['sum', 'mean', 'var'],
        'height': ['sum', 'mean', 'var'],
    }

    agg_train_imgs = train_df_imgs.groupby('PetID').agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_train_imgs.columns = new_columns
    agg_train_imgs = agg_train_imgs.reset_index()

    agg_test_imgs = test_df_imgs.groupby('PetID').agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_test_imgs.columns = new_columns
    agg_test_imgs = agg_test_imgs.reset_index()

    # agg_imgs = pd.concat([agg_train_imgs, agg_test_imgs], axis=0).reset_index(drop=True)
    # return agg_imgs
    return agg_train_imgs, agg_test_imgs


def preprocessing(train, test, breeds, colors, states,
                  mode='regression', debug=False, use_tfidf=True, use_tfidf_cache=True,
                  use_sentiment=True, use_metadata=False, cat2num=True,
                  use_rescuer_id_count=True, use_name_feature=True, use_target_encoding=True,
                  tfidf_svd_components=120, animal_type=None, num_sentiment_text=0):

    # nan handling...
    train['Name'].fillna('none', inplace=True)
    train['Description'].fillna('none', inplace=True)
    test['Name'].fillna('none', inplace=True)
    test['Description'].fillna('none', inplace=True)

    train['dataset_type'] = 'train'
    test['dataset_type'] = 'test'
    all_data = pd.concat([train, test], axis=0, sort=True)
    train_indices = (all_data['dataset_type'] == 'train').values
    test_indices = (all_data['dataset_type'] == 'test').values

    numeric_cols = ['Age', 'MaturitySize', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']
    if debug:
        cat_cols = ['Type', 'Breed1', 'Gender', 'Color1', 'Vaccinated', 'Dewormed', 'Sterilized',
                    'State', 'FurLength', 'Health']
    else:
        cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
                    'Vaccinated', 'Dewormed', 'Sterilized', 'State', 'FurLength', 'Health']

    # TODO: process these features properly later...
    remove_cols = ['RescuerID', 'PetID', 'AdoptionSpeed']
    other_cols = ['Name', 'Description']

    if mode == 'regression':
        target = train['AdoptionSpeed'].values.astype(np.float32)[:, None]
    elif mode == 'classification':
        target = train['AdoptionSpeed'].values.astype(np.int32)
    else:
        raise ValueError("[ERROR] Unexpected value mode={}".format(mode))

    # --- Feature engineering ---
    # GDP & population
    use_gdp = False
    if use_gdp:
        add_gdp(train)
        add_gdp(test)

        # Scaling...
        train["state_gdp"] = train["state_gdp"] / train["state_gdp"].max()
        test["state_gdp"] = test["state_gdp"] / train["state_gdp"].max()
        train["state_population"] = train["state_population"] / train["state_population"].max()
        test["state_population"] = test["state_population"] / train["state_population"].max()
        numeric_cols.append('state_gdp')
        numeric_cols.append('state_population')

    if use_rescuer_id_count:
        print('use_rescuer_id_count...')
        cutoff_count = 20
        train['RescuerIDCount'] = train.groupby('RescuerID')[
            'RescuerID'].transform(lambda s: s.count())
        test['RescuerIDCount'] = test.groupby('RescuerID')[
            'RescuerID'].transform(lambda s: s.count())

        train.loc[train['RescuerIDCount'] >= cutoff_count, 'RescuerIDCount'] = cutoff_count
        test.loc[test['RescuerIDCount'] >= cutoff_count, 'RescuerIDCount'] = cutoff_count

        # "is_first_time" feature
        train['is_first_time'] = (train['RescuerIDCount'] == 1).astype(np.float32)
        test['is_first_time'] = (test['RescuerIDCount'] == 1).astype(np.float32)
        numeric_cols.append('RescuerIDCount')
        numeric_cols.append('is_first_time')

    if use_name_feature:
        print('create name feature...')
        # create name feature
        # 1. no name or not
        train['No_name'] = 0
        train.loc[train['Name'] == 'none', 'No_name'] = 1
        test['No_name'] = 0
        test.loc[test['Name'] == 'none', 'No_name'] = 1

        # 2. weired name or not
        train['name_under2'] = train['Name'].apply(lambda x: len(str(x)) < 3).values.astype(np.float32)
        test['name_under2'] = test['Name'].apply(lambda x: len(str(x)) < 3).values.astype(np.float32)

        # 3. puppy, puppies, kitten, kitty, baby flag.
        train['is_kitty'] = train['Name'].apply(lambda x: 'kitty' in str(x).lower()).values.astype(np.float32)
        test['is_kitty'] = test['Name'].apply(lambda x: 'kitty' in str(x).lower()).values.astype(np.float32)

        train['is_kitten'] = train['Name'].apply(lambda x: 'kitten' in str(x).lower()).values.astype(np.float32)
        test['is_kitten'] = test['Name'].apply(lambda x: 'kitten' in str(x).lower()).values.astype(np.float32)

        train['is_puppy'] = train['Name'].apply(lambda x: 'puppy' in str(x).lower()).values.astype(np.float32)
        test['is_puppy'] = test['Name'].apply(lambda x: 'puppy' in str(x).lower()).values.astype(np.float32)

        train['is_puppies'] = train['Name'].apply(lambda x: 'puppies' in str(x).lower()).values.astype(np.float32)
        test['is_puppies'] = test['Name'].apply(lambda x: 'puppies' in str(x).lower()).values.astype(np.float32)

        numeric_cols.append('name_under2')
        numeric_cols.append('is_kitty')
        numeric_cols.append('is_kitten')
        numeric_cols.append('is_puppy')
        numeric_cols.append('is_puppies')

    if use_target_encoding:
        print('create target encoding feature...')
        # 1. --- Breed target encoding ---
        # breed1 = train.groupby('Breed1')['AdoptionSpeed']
        # train['breed1_mean'] = breed1.transform(np.mean)
        # train['breed1_median'] = breed1.transform(np.median)
        breed1 = all_data.groupby('Breed1')['AdoptionSpeed']
        all_data['breed1_mean'] = breed1.transform(np.mean)
        # all_data['breed1_q1'] = breed1.transform(lambda x: np.quantile(x, 0.25))
        breed2 = all_data.groupby('Breed2')['AdoptionSpeed']
        all_data['breed2_mean'] = breed2.transform(np.mean)
        # all_data['breed2_median'] = breed2.transform(np.median)

        # 2. --- State target encoding ---
        state = all_data.groupby('State')['AdoptionSpeed']
        all_data['state_mean'] = state.transform(np.mean)

        # Assign values into `train` and `test`...
        for col in ['breed1_mean', 'breed2_mean', 'state_mean']:
            train[col] = all_data[train_indices][col]
            test[col] = all_data[test_indices][col]
            numeric_cols.append(col)

    # --- is_xxx flag ---
    train['is_free'] = 0
    train.loc[train['Fee'] == 0, 'is_free'] = 1
    test['is_free'] = 0
    test.loc[test['Fee'] == 0, 'is_free'] = 1
    numeric_cols.append('is_free')

    train['has_photo'] = 0
    train.loc[train['PhotoAmt'] > 0, 'has_photo'] = 1
    test['has_photo'] = 0
    test.loc[test['PhotoAmt'] > 0, 'has_photo'] = 1
    numeric_cols.append('has_photo')

    train['age_unknown'] = 0
    train.loc[train['Age'] == 255, 'age_unknown'] = 1
    test['age_unknown'] = 0
    test.loc[test['Age'] == 255, 'age_unknown'] = 1
    numeric_cols.append('age_unknown')

    # def quantize_age(df):
    #     # quantize age to multiple of 6 (half year...).
    #     # DataFrame is replaced inplace.
    #     age = df['Age'].values
    #
    #     # (13, 18)  # seems better not to quantize..
    #     quantize_list = [(19, 24), (25, 30), (31, 36), (37, 48),
    #                      (49, 60), (61, 72), (73, 84), (85, 96)]
    #     for low, high in quantize_list:
    #         condition = (age >= low) & (age < high)
    #         print('{} => {} matched for age between {} to {}. quantize to {}...'
    #               .format(np.sum(condition), np.sum(age == high), low, high-1, high))
    #         df.loc[condition, 'Age'] = high
    # print('quantize_age: train')
    # quantize_age(train)
    # print('quantize_age: test')
    # quantize_age(test)

    # --- Cutoff ---
    # 'Age', max was 255.
    # TODO: we may need to deal with "255" as special (I think this is "unknown")
    # age_cutoff = 54  # TODO: hyperparameter tuning. This affects a lot!!!
    age_cutoff = 60  # TODO: hyperparameter tuning. This affects a lot!!!
    print('age_cutoff', age_cutoff)
    train.loc[train['Age'] >= age_cutoff, 'Age'] = age_cutoff
    test.loc[test['Age'] >= age_cutoff, 'Age'] = age_cutoff
    # 'Quantity', max was 20, but most of them are ~10.
    quantity_cutoff = 11
    train.loc[train['Quantity'] >= quantity_cutoff, 'Quantity'] = quantity_cutoff
    test.loc[test['Quantity'] >= quantity_cutoff, 'Quantity'] = quantity_cutoff
    # 'VideoAmt', max was 8, but most of them are ~1.
    video_amt_cutoff = 4
    train.loc[train['VideoAmt'] >= video_amt_cutoff, 'VideoAmt'] = video_amt_cutoff
    test.loc[test['VideoAmt'] >= video_amt_cutoff, 'VideoAmt'] = video_amt_cutoff
    # 'PhotoAmt', max was 30, but most of them are ~11.
    photo_amt_cutoff = 12
    train.loc[train['PhotoAmt'] >= photo_amt_cutoff, 'PhotoAmt'] = photo_amt_cutoff
    test.loc[test['PhotoAmt'] >= photo_amt_cutoff, 'PhotoAmt'] = photo_amt_cutoff
    # 'Fee', max was 3000, but most of them are ~300 or ~500
    # 300 or 500
    fee_cutoff = 500
    train.loc[train['Fee'] >= fee_cutoff, 'Fee'] = fee_cutoff
    test.loc[test['Fee'] >= fee_cutoff, 'Fee'] = fee_cutoff

    # --- Numeric value processing ---
    print('numeric value preprocessing...')
    # There is no nan value, but this is just for make sure no nan exist.
    train_x_numeric = train[numeric_cols].fillna(0).values.astype(np.float32)
    test_x_numeric = test[numeric_cols].fillna(0).values.astype(np.float32)

    # --- MinMax scaling ---
    xmax = np.max(train_x_numeric, axis=0)
    xmin = np.min(train_x_numeric, axis=0)
    print('xmax', xmax)
    print('xmin', xmin)
    inds = xmax != xmin  # Non-zero indices
    train_x_numeric[:, inds] = (train_x_numeric[:, inds] - xmin[inds]) / (xmax[inds] - xmin[inds])
    test_x_numeric[:, inds] = (test_x_numeric[:, inds] - xmin[inds]) / (xmax[inds] - xmin[inds])

    if use_sentiment:
        print('create sentiment feature...')
        n_jobs = 16
        s = perf_counter()
        # Multiprocessing: around 2 sec. Multithreading: 10 sec. Singlethreading 63 sec.
        # train_x_sent = Parallel(n_jobs, backend='threading')(
        #     delayed(process_sentiment, check_pickle=False)
        #     (petid, 'train') for petid in train['PetID'].values)
        train_x_sent = Parallel(n_jobs)(
            delayed(process_sentiment)
            (petid, 'train', num_sentiment_text) for petid in train['PetID'].values)
        test_x_sent = Parallel(n_jobs)(
            delayed(process_sentiment)
            (petid, 'test', num_sentiment_text) for petid in test['PetID'].values)
        e = perf_counter()
        print('sentiment {} sec, n_jobs {}'.format(e-s, n_jobs))
        train_x_sent = np.array(train_x_sent, dtype=np.float32)
        test_x_sent = np.array(test_x_sent, dtype=np.float32)
        print('train_x_numeric {}, train_x_sent {}'
              .format(train_x_numeric.shape, train_x_sent.shape))
        train_x_numeric = np.concatenate([train_x_numeric, train_x_sent], axis=1)
        test_x_numeric = np.concatenate([test_x_numeric, test_x_sent], axis=1)

    if use_metadata:
        print('create metadata feature...')
        n_jobs = 16
        s = perf_counter()
        train_x_metadata = Parallel(n_jobs)(
            delayed(process_metadata)
            (petid, 'train') for petid in train['PetID'].values)
        test_x_metadata = Parallel(n_jobs)(
            delayed(process_metadata)
            (petid, 'test') for petid in test['PetID'].values)
        e = perf_counter()
        print('metadata {} sec, n_jobs {}'.format(e-s, n_jobs))
        train_x_metadata = np.array(train_x_metadata, dtype=np.float32)
        test_x_metadata = np.array(test_x_metadata, dtype=np.float32)
        print('train_x_numeric {}, train_x_metadata {}'
              .format(train_x_numeric.shape, train_x_metadata.shape))
        train_x_numeric = np.concatenate([train_x_numeric, train_x_metadata], axis=1)
        test_x_numeric = np.concatenate([test_x_numeric, test_x_metadata], axis=1)

    os.makedirs('./cache', exist_ok=True)
    if use_tfidf:
        if animal_type is not None:
            cache_filepath = './cache/x_tfidf_svd{}_debug{}_animaltype{}.npz'.format(
                tfidf_svd_components, int(debug), animal_type)
        else:
            cache_filepath = './cache/x_tfidf_svd{}_debug{}.npz'.format(
                tfidf_svd_components, int(debug))
        if use_tfidf_cache and os.path.exists(cache_filepath):
            print('load from {}'.format(cache_filepath))
            train_x_tfidf_svd, test_x_tfidf_svd = load_npz(cache_filepath)
        else:
            print('create tfidf feature...')
            train_x_tfidf_svd, test_x_tfidf_svd = add_tfidf(train, test, tfidf_svd_components=tfidf_svd_components)
            save_npz(cache_filepath, (train_x_tfidf_svd, test_x_tfidf_svd))
            print('saved to {}'.format(cache_filepath))
        train_x_numeric = np.concatenate([train_x_numeric, train_x_tfidf_svd], axis=1)
        test_x_numeric = np.concatenate([test_x_numeric, test_x_tfidf_svd], axis=1)

    # --- Category value processing ---
    if cat2num:
        # convert category value to one-hot vector or other values.
        # cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
        #             'Vaccinated', 'Dewormed', 'Sterilized', 'State', 'FurLength', 'Health']
        # all_cat = all_data[cat_cols].astype('category')
        # all_cat_id = all_cat.apply(lambda x: x.cat.codes)

        # --- 1. category to one-hot vector ---
        # checked include State or not --> include State seems better.
        one_hot_cols = ['Type', 'Gender', 'Vaccinated',
                        'Dewormed', 'Sterilized', 'State', 'FurLength', 'Health']
        one_hot_list = [pd.get_dummies(all_data[col]).values for col in one_hot_cols]
        one_hot_array = np.concatenate(one_hot_list, axis=1).astype(np.float32)

        # --- 2. breed ---
        # deal "Unspecified" and "Unknown" as same for 2nd breed
        all_data['Breed2'][all_data['Breed2'] == 0] = 307
        # is_mixed = (all_data['Breed2'] != 0) & (all_data['Breed2'] != 307)
        is_mixed = (all_data['Breed2'] != 307).astype(np.float32)[:, None]
        b1 = all_data['Breed1'].value_counts()
        major_breeds = b1[b1 >= 100].index.values  # 18 species remain
        # major_breeds = b1[b1 >= 1000].index.values  # 3 species remain
        print('major_breeds', major_breeds)

        def breed_onehot(x):
            if x not in major_breeds:
                # rare breed
                return len(major_breeds)
            else:
                # major (non-rare) breed
                breed_id = np.argwhere(x == major_breeds)[0, 0]
                # return x
                return breed_id

        b1r = all_data['Breed1'].apply(breed_onehot)
        b2r = all_data['Breed2'].apply(breed_onehot)
        breed_ones = np.eye(len(major_breeds) + 1, dtype=np.float32)
        breed_array = (1.0 * breed_ones[b1r] + 0.7 * breed_ones[b2r]).astype(np.float32)
        # breed_array = (1.0 * breed_ones[b1r] + 1.0 * breed_ones[b2r]).astype(np.float32)

        # --- 3. color ---
        # 0 unspecified, 1 black, ... , 7 white.
        color_ones = np.eye(8)
        color1_onehot = color_ones[all_data['Color1'].values]
        color2_onehot = color_ones[all_data['Color2'].values]
        color3_onehot = color_ones[all_data['Color3'].values]
        color_array = (1.0 * color1_onehot + 0.7 * color2_onehot + 0.5 * color3_onehot).astype(np.float32)
        # color_array = (1.0 * color1_onehot + 1.0 * color2_onehot + 1.0 * color3_onehot).astype(np.float32)

        x_cat2num_array = np.concatenate([one_hot_array, is_mixed, breed_array, color_array], axis=1)
        print('one_hot_array', one_hot_array.shape,
              'is_mixed', is_mixed.shape,
              'breed_array', breed_array.shape,
              'color_array', color_array.shape,
              'x_cat2num_array', x_cat2num_array.shape, x_cat2num_array.dtype)

        train_x_cat = x_cat2num_array[train_indices]
        test_x_cat = x_cat2num_array[test_indices]
        num_cat_id = -1
    else:
        all_cat = all_data[cat_cols].astype('category')
        all_cat_id = all_cat.apply(lambda x: x.cat.codes)
        all_x_cat = all_cat_id.values.astype(np.int32)

        train_x_cat = all_x_cat[train_indices]
        test_x_cat = all_x_cat[test_indices]

        num_cat_id = np.max(all_x_cat, axis=0) + 1
        print('train_x_cat', train_x_cat.shape, 'test_x_cat', test_x_cat.shape, 'num_cat_id', num_cat_id)
    return train_x_numeric, train_x_cat, target, test_x_numeric, test_x_cat, num_cat_id


# In[12]:


# --- preprocessing image feature ---


# In[13]:


import cv2
import os
import numpy as np
import pandas as pd
from chainer import cuda
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from keras.applications.densenet import preprocess_input, DenseNet121
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K

import sys
import os


def resize_to_square(im, img_size=256):
    old_size = im.shape[:2]
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    return new_im


def load_image(path, pet_id, image_index=1, img_size=256):
    image = cv2.imread(f'{path}{pet_id}-{image_index}.jpg')
    new_image = resize_to_square(image, img_size=img_size)
    new_image = preprocess_input(new_image)
    return new_image


def read_image_keras(filepath, img_size=256):
    image = cv2.imread(filepath)
    new_image = resize_to_square(image, img_size=img_size)
    new_image = preprocess_input(new_image)
    return new_image


def prepare_model_densenet():
    inp = Input((256, 256, 3))
    backbone = DenseNet121(input_tensor=inp,
                           weights=f"{densenet_dir}/DenseNet-BC-121-32-no-top.h5",
                           include_top=False)
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
    # x = AveragePooling1D(4)(x)  # Remove feature reduction!!
    out = Lambda(lambda x: x[:, :, 0])(x)
    m = Model(inp, out)
    return m


def preprocess_image_densenet(train, test, img_size=256, batch_size=256,
                              n_components=32, method='svd'):
    use_cache = True
    if use_cache:
        try:
            # local
            train_feats = read_feather("../input/densenet-keras/train_image_densenet_5.feather")
            test_feats = read_feather("../input/densenet-keras/test_image_densenet_5.feather")
        except:
            # kaggle kernel
            train_feats = read_feather("./train_image_densenet_10_1024.feather")
            test_feats = read_feather("./test_image_densenet_10_1024.feather")
    else:
        m = prepare_model_densenet()
        pet_ids = train['PetID'].values
        n_batches = len(pet_ids) // batch_size + 1

        features = {}
        for b in tqdm(range(n_batches)):
            start = b * batch_size
            end = (b + 1) * batch_size
            batch_pets = pet_ids[start:end]
            batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
            for i, pet_id in enumerate(batch_pets):
                try:
                    batch_images[i] = load_image(
                        f"{pet_dir}/train_images/",
                        pet_id, img_size=img_size)
                except:
                    pass
            batch_preds = m.predict(batch_images)
            for i, pet_id in enumerate(batch_pets):
                features[pet_id] = batch_preds[i]

        train_feats = pd.DataFrame.from_dict(features, orient='index')
        train_feats.columns = [f'pic_{i}' for i in range(train_feats.shape[1])]

        pet_ids = test['PetID'].values
        n_batches = len(pet_ids) // batch_size + 1

        features = {}
        for b in tqdm(range(n_batches)):
            start = b * batch_size
            end = (b + 1) * batch_size
            batch_pets = pet_ids[start:end]
            batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
            for i, pet_id in enumerate(batch_pets):
                try:
                    batch_images[i] = load_image(
                        f"{pet_dir}/test_images/",
                        pet_id, img_size=img_size)
                except:
                    pass
            batch_preds = m.predict(batch_images)
            for i, pet_id in enumerate(batch_pets):
                features[pet_id] = batch_preds[i]
        test_feats = pd.DataFrame.from_dict(features, orient='index')
        test_feats.columns = [f'pic_{i}' for i in range(test_feats.shape[1])]

        train_feats = train_feats.reset_index()
        train_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)

        test_feats = test_feats.reset_index()
        test_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)

    hdim = train_feats.shape[1] - 1
    fcols = [f'pic_{i}' for i in range(hdim)]
    print('keras densenet feature: hdim', hdim)

    train_feats.fillna(0, inplace=True)
    test_feats.fillna(0, inplace=True)
    features_df = pd.concat([train_feats, test_feats], axis=0, sort=False, ignore_index=True)
    features = features_df[fcols].values

    if n_components is not None:
        if method == 'svd':
            svd_ = TruncatedSVD(n_components=n_components, random_state=1337)
            svd_col = svd_.fit_transform(features)
            train_x_image_array = svd_col[:len(train)]
            test_x_image_array = svd_col[len(train):]
        else:
            raise ValueError("[ERROR] Unexpected value method={}".format(method))
    else:
        train_x_image_array = train_feats[fcols].values
        test_x_image_array = test_feats[fcols].values

    num_image = 1
    num_hidden = train_x_image_array.shape[-1]
    xp = cuda.get_array_module(train_x_image_array)
    train_x_image_feat = xp.reshape(train_x_image_array, (len(train), num_image, num_hidden))
    test_x_image_feat = xp.reshape(test_x_image_array, (len(test), num_image, num_hidden))
    # train_x_image_feat = xp.reshape(train_x_image_array, (len(train_x_image_array), num_image, num_hidden))
    # test_x_image_feat = xp.reshape(test_x_image_array, (len(test_x_image_array), num_image, num_hidden))

    print('train_feats', train_feats.shape)
    print('test_feats', test_feats.shape)
    assert np.alltrue(train_feats['PetID'].values == train['PetID'].values)
    assert np.alltrue(test_feats['PetID'].values == test['PetID'].values)
    return train_x_image_feat, test_x_image_feat


# 

# In[ ]:


# --- preprocessing image ---


# In[ ]:


"""
Image preprocessing using classification model.
"""
import numpy as np

# import cv2

import chainer
# from chainer.iterators import SerialIterator
from chainer import cuda
import chainer.functions as F
from chainer.dataset import concat_examples
from chainer.iterators import MultithreadIterator, SerialIterator
from sklearn.decomposition import PCA, TruncatedSVD
from tqdm import tqdm

import sys
import os


# sys.path.append(os.pardir)
# sys.path.append(os.path.join(os.pardir, os.pardir))
# from chainercv.links.model.vgg import VGG16
# from chainercv.links.model.senet.se_resnext import SEResNeXt50
# from chainercv.links.model.feature_predictor import FeaturePredictor
# from chainercv.utils.image.read_image import read_image
# from chainer_chemistry.links.scaler.standard_scaler import StandardScaler
# from src.models.mlp2 import MLP2
# from src.rating_eda import load_json
# from src.pet_image_dataset import PetImageDataset
# from src.utils import timer, save_npz, load_npz
# from src.configs import pet_dir, is_kaggle_kernel, vgg16_dir


def prepare_model(arch='vgg16', device=-1):
    print('prepare_model {}...'.format(arch))
    if arch == 'vgg16':
        if is_kaggle_kernel:
            base_model = VGG16(pretrained_model=os.path.join(
                vgg16_dir, 'vgg16_imagenet_converted_2017_07_18.npz'))
        else:
            base_model = VGG16()
        # base_model.pick = ['fc7']  # 'fc6'
        # base_model.pick = ['conv5_3']
        base_model.pick = ['conv3_3']
        print('base_model.pick', base_model.pick)
    elif arch == 'seresnext50':
        if is_kaggle_kernel:
            raise NotImplementedError
            # seresnext50_dir = '../input/seresnext50-chainercv'
            # base_model = SEResNeXt50(pretrained_model=os.path.join(seresnext50_dir, 'vgg16_imagenet_converted_2017_07_18.npz'))
        else:
            base_model = SEResNeXt50()
        # base_model.pick = ['pool5']  # 'fc6'
        base_model.pick = ['res3']  # 'fc6'
        print('base_model.pick', base_model.pick)
    else:
        raise ValueError("[ERROR] Unexpected value arch={}"
                         .format(arch))

    model = FeaturePredictor(
        base_model, crop_size=224, scale_size=256, crop='center')
    if device >= 0:
        chainer.backends.cuda.get_device_from_id(device).use()
        model.to_gpu()
    return model


# def resize_to_square(im, img_size=256):
#     old_size = im.shape[:2] # old_size is in (height, width) format
#     ratio = float(img_size)/max(old_size)
#     new_size = tuple([int(x*ratio) for x in old_size])
#     # new_size should be in (width, height) format
#     im = cv2.resize(im, (new_size[1], new_size[0]))
#     delta_w = img_size - new_size[1]
#     delta_h = img_size - new_size[0]
#     top, bottom = delta_h//2, delta_h-(delta_h//2)
#     left, right = delta_w//2, delta_w-(delta_w//2)
#     color = [0, 0, 0]
#     new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
#     return new_im


def load_image(petid, data_type, num_image=2):
    image_list = []
    for index in range(1, num_image + 1):
        filepath = '{}/{}_images/{}-{}.jpg'.format(pet_dir, data_type, petid, index)
        if os.path.exists(filepath):
            image = read_image(filepath)
            # print('image', type(image), image.dtype, image.shape)
        else:
            print('{} not exit'.format(filepath))
            # image not exist
            ch = 3
            h = 256
            w = 256
            image = np.zeros((ch, h, w), dtype=np.float32)
        image_list.append(image)
    return image_list


def calc_image_features(model, image_list, batch_size=16):
    pred_list = []
    for i in tqdm(range(0, len(image_list), batch_size)):
        pred = model.predict(image_list[i:i+batch_size])
        # print('pred', pred[0].shape)
        pred_list.append(pred[0])
    return np.concatenate(pred_list, axis=0)


def preprocess_image(train, test, num_image=2, device=-1, arch='vgg16',
                     n_components=None, use_cache=True, animal_type=None,
                     method='pooling', mode='chainercv'):
    """

    Args:
        train (): train df
        test (): test df
        num_image (int): number of image to extract feature
        device (int):
        arch (str): architecture
        n_components (int or None): if int specified, PCA is applied to
            reduce dimension.
        animal_type (None or int): if specified, only use dog or cat images.

    Returns:
        train_x_image_feat (array): (batch_size, num_image, hidden_dim)
        test_x_image_feat (array): (batch_size, num_image, hidden_dim)
    """
    os.makedirs('./cache', exist_ok=True)
    if animal_type is None:
        cache_filepath = './cache/x_image_array_arch{}_numimage{}_size{}.npz'            .format(arch, num_image, len(train))
    else:
        cache_filepath = './cache/x_image_array_arch{}_numimage{}_size{}_type{}.npz'             .format(arch, num_image, len(train), animal_type)
    if use_cache and os.path.exists(cache_filepath):
        # load from cache
        print('loading from cache {}'.format(cache_filepath))
        train_x_image_array, test_x_image_array = load_npz(cache_filepath)
    else:
        # prepare pretrained model
        if arch == 'densenet':
            try:
                from src.preprocessing_image_keras import prepare_model_densenet
            except:
                pass
            model = prepare_model_densenet()
        else:
            model = prepare_model(arch=arch, device=device)

        # n_jobs = 8
        # print('n_jobs', n_jobs)
        batch_size = 32

        # extract pet id
        def calc_x_image_array(petids, data_type, num_image, batch_size):
            image_dataset = PetImageDataset(petids, data_type, num_image, mode=mode)
            # iterator = MultiprocessIterator(
            #     image_dataset, batch_size, repeat=False, shuffle=False)
            iterator = MultithreadIterator(
                image_dataset, batch_size, repeat=False, shuffle=False)
            x_image_array = None
            current_index = 0
            for batch in tqdm(iterator, total=len(image_dataset) // batch_size):
                has_image_indices = np.argwhere(np.array([elem is not None for elem in batch]))[:, 0]
                image_list = [elem for elem in batch if elem is not None]
                if mode == 'keras':
                    feats = model.predict(np.array(image_list))
                else:
                    if arch in ['vgg16', 'seresnext50']:
                        feats = model.predict(image_list)[0]
                        # feats = F.average(feats, axis=(2, 3)).array
                        feats = np.mean(feats, axis=(2, 3))
                        # feats = np.max(feats, axis=(2, 3))
                    else:
                        feats = model.predict(image_list)[0]
                if x_image_array is None:
                    feat_dim = feats.shape[1]
                    x_image_array = np.zeros((len(image_dataset), feat_dim), dtype=np.float32)
                x_image_array[has_image_indices + current_index] = feats
                current_index += batch_size
            return x_image_array

        train_x_image_array = calc_x_image_array(train['PetID'].values, 'train', num_image, batch_size)
        test_x_image_array = calc_x_image_array(test['PetID'].values, 'test', num_image, batch_size)

        save_npz(cache_filepath, (train_x_image_array, test_x_image_array))
        print('saved to {}'.format(cache_filepath))

    # TODO: investigate why it sometimes contains nan...
    if np.sum(np.isnan(train_x_image_array)) > 0:
        print('train_x_image_array contains {} nan... replace to 0.'
              .format(np.sum(np.isnan(train_x_image_array))))
    train_x_image_array[np.isnan(train_x_image_array)] = 0.
    test_x_image_array[np.isnan(test_x_image_array)] = 0.

    # # --- DEBUG: cosine distance exp ---
    import cupy
    # preprocess_image_cute(device=0, arch='vgg16')
    x_cute_image_array = train_x_image_array

    print('x_cute_image_array', x_cute_image_array.shape)
    x_cute_image_array = cupy.asarray(x_cute_image_array)
    xp = cupy

    # cosine distance experiment...
    print('calc norm')
    x_cute_image_array_normlized = x_cute_image_array / xp.sqrt(
        xp.sum(xp.square(x_cute_image_array), axis=1, keepdims=True))
    target_index = 0
    print('calc cosine dist')
    cosine_distance = xp.sum(
        x_cute_image_array_normlized[target_index:target_index+1, None, :] * x_cute_image_array_normlized[None, :, :], axis=2)
    print('cosine_distance', cosine_distance.shape, cosine_distance)
    import IPython; IPython.embed()
    # # --- DEBUD end ---

    # train_x_image_array, test_x_image_array = contraction_by_mlp(
    #     train_x_image_array, test_x_image_array, model_dir='./result', arch=arch)
    if n_components is not None:
        train_x_image_array, test_x_image_array = contraction(
            train_x_image_array, test_x_image_array, n_components,
            method=method)

    num_hidden = train_x_image_array.shape[-1]
    xp = cuda.get_array_module(train_x_image_array)
    train_x_image_feat = xp.reshape(train_x_image_array, (len(train), num_image, num_hidden))
    test_x_image_feat = xp.reshape(test_x_image_array, (len(test), num_image, num_hidden))
    return train_x_image_feat, test_x_image_feat


def contraction_by_mlp(train_x_image_array, test_x_image_array, model_dir, device=0,
                       arch='vgg16'):
    print('contraction_by_mlp')
    args_dict = load_json(os.path.join(model_dir, 'args.json'))
    # scaler = StandardScaler()
    scaler = None
    mlp = MLP2(out_dim=1, layer_dims=[args_dict['unit']], use_bn=True,
               use_sn=False, use_residual=True, scaler=scaler)
    cute_filepath = os.path.join(model_dir, f'cute_mlp_{arch}.npz')
    print(f'loading model from {cute_filepath} ...')
    chainer.serializers.load_npz(cute_filepath, mlp)
    if device >= 0:
        mlp.to_gpu(device)
        chainer.cuda.get_device_from_id(device).use()  # Make a specified GPU current
    print('forwarding train_x_image_array', train_x_image_array.shape)
    batch_size = 10240
    train_h_image = forward(mlp, train_x_image_array, batch_size=batch_size, device=device)
    test_h_image = forward(mlp, test_x_image_array, batch_size=batch_size, device=device)
    # train_h_image = mlp.calc(train_x_image_array, extract_hidden=True)
    # test_h_image = mlp.calc(test_x_image_array, extract_hidden=True)
    return _to_ndarray(train_h_image), _to_ndarray(test_h_image)


def forward(model, x, batch_size=1024, device=0):
    iter = SerialIterator(x, batch_size=batch_size, repeat=False, shuffle=False)
    output_list = []
    for batch in iter:
        inputs = concat_examples(batch, device=device)
        outputs = model.calc(inputs, extract_hidden=True)
        outputs_array = cuda.to_cpu(outputs.array)
        output_list.append(outputs_array)
        if np.sum(np.isnan(outputs_array)) > 0:
            print('nan detected in forward')
            import IPython; IPython.embed()
    return np.concatenate(output_list, axis=0)


def _to_ndarray(x):
    if isinstance(x, chainer.Variable):
        x = x.array
    return cuda.to_cpu(x)


def contraction(train_x_image_array, test_x_image_array, n_components,
                method='pca', retain_dim=0):
    contraction_method = method
    print('contraction_method', contraction_method)
    if contraction_method == 'pca':
        with timer('pca image fit n_components={}'.format(n_components)):
            print('before', train_x_image_array.shape)
            pca = PCA(n_components=n_components)
            pca.fit(np.concatenate([train_x_image_array, test_x_image_array], axis=0))
        with timer('pca image train feature n_components={}'.format(n_components)):
            train_x_image_array = pca.transform(train_x_image_array).astype(np.float32)
            print('after', train_x_image_array.shape)
        with timer('pca image test feature n_components={}'.format(n_components)):
            test_x_image_array = pca.transform(test_x_image_array).astype(np.float32)
    elif contraction_method == 'svd':
        n_test = len(test_x_image_array)
        svd_ = TruncatedSVD(n_components=n_components, random_state=1337)
        x_image_array = svd_.fit_transform(np.concatenate([train_x_image_array, test_x_image_array], axis=0))
        train_x_image_array = x_image_array[:len(train_x_image_array)]
        test_x_image_array = x_image_array[len(train_x_image_array):]
        assert len(test_x_image_array) == n_test
        # with timer('svd image fit n_components={}'.format(n_components)):
        #     print('before', train_x_image_array.shape)
        #     svd_.fit(np.concatenate([train_x_image_array, test_x_image_array], axis=0))
        # with timer('svd image train feature n_components={}'.format(n_components)):
        #     train_x_image_array = svd_.transform(train_x_image_array).astype(np.float32)
        #     print('after', train_x_image_array.shape)
        # with timer('svd image test feature n_components={}'.format(n_components)):
        #     test_x_image_array = svd_.transform(test_x_image_array).astype(np.float32)
    elif contraction_method == 'pooling':
        def pooling(x, n_components):
            # retain_dim = 6 for detection model, num_person, num_animal, bbox features are remain...
            x1 = x[:, :retain_dim]  #
            x2 = x[:, retain_dim:]  # other features.
            bs, ch = x2.shape
            p = ch // n_components
            print('x2', x2.shape, 'n_components', n_components, 'p', p)
            x2 = np.mean(x2[:, :(n_components * p)].reshape(bs, n_components, p), axis=2)
            return np.concatenate([x1, x2], axis=1)
        train_x_image_array = pooling(train_x_image_array, n_components)
        test_x_image_array = pooling(test_x_image_array, n_components)
    return train_x_image_array, test_x_image_array


if not is_kaggle_kernel and __name__ == '__main__':
    from src.preprocessing import prepare_df

    debug = False
    train, test, breeds, colors, states = prepare_df(debug)
    train_x_image, test_x_image = preprocess_image(
        train, test, num_image=1, device=0, arch='seresnext50', use_cache=False)
    print('train_x_image', type(train_x_image), train_x_image.dtype,
          train_x_image.shape)
    print('test_x_image', test_x_image.shape)
    import IPython; IPython.embed()


# In[ ]:





# In[ ]:





# ## Chainer Chemistry
# 
# Codes copied from [chainer_chemistry](https://github.com/pfnet-research/chainer-chemistry), MIT license.
# 
# I hope it is supported in kaggle default docker: [PR to support it in default docker](https://github.com/Kaggle/docker-python/pull/447).

# In[14]:


import chainer
from chainer.functions import relu
from chainer import links


class MLP(chainer.Chain):

    """Basic implementation for MLP

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_layers (int): number of layers
        activation (chainer.functions): activation function
    """

    def __init__(self, out_dim, hidden_dim=16, n_layers=2, activation=relu):
        super(MLP, self).__init__()
        if n_layers <= 0:
            raise ValueError('n_layers must be a positive integer, but it was '
                             'set to {}'.format(n_layers))
        layers = [links.Linear(None, hidden_dim) for i in range(n_layers - 1)]
        with self.init_scope():
            self.layers = chainer.ChainList(*layers)
            self.l_out = links.Linear(None, out_dim)
        self.activation = activation

    def __call__(self, x):
        h = x
        for l in self.layers:
            h = self.activation(l(h))
        h = self.l_out(h)
        return h


# In[15]:


import copy
from logging import getLogger

import numpy

import chainer
from chainer import cuda
from chainer.dataset import convert
from chainer import reporter
from chainer.training.extensions import Evaluator


def _get_1d_numpy_array(v):
    """Convert array or Variable to 1d numpy array

    Args:
        v (numpy.ndarray or cupy.ndarray or chainer.Variable): array to be
            converted to 1d numpy array

    Returns (numpy.ndarray): Raveled 1d numpy array

    """
    if isinstance(v, chainer.Variable):
        v = v.data
    return cuda.to_cpu(v).ravel()


class BatchEvaluator(Evaluator):

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, metrics_fun=None,
                 name=None, logger=None):
        super(BatchEvaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func)
        self.name = name
        self.logger = logger or getLogger()

        if callable(metrics_fun):
            # TODO(mottodora): use better name or infer
            self.metrics_fun = {"evaluation": metrics_fun}
        elif isinstance(metrics_fun, dict):
            self.metrics_fun = metrics_fun
        else:
            raise TypeError('Unexpected type metrics_fun must be Callable or '
                            'dict.')

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        y_total = []
        t_total = []
        for batch in it:
            in_arrays = self.converter(batch, self.device)
            with chainer.no_backprop_mode(), chainer.using_config('train',
                                                                  False):
                y = eval_func(*in_arrays[:-1])
            t = in_arrays[-1]
            y_data = _get_1d_numpy_array(y)
            t_data = _get_1d_numpy_array(t)
            y_total.append(y_data)
            t_total.append(t_data)

        y_total = numpy.concatenate(y_total).ravel()
        t_total = numpy.concatenate(t_total).ravel()
        # metrics_value = self.metrics_fun(y_total, t_total)
        metrics = {key: metric_fun(y_total, t_total) for key, metric_fun in
                   self.metrics_fun.items()}

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(metrics, self._targets['main'])
        return observation


# In[16]:


import os
import six

import numpy

# from chainer_chemistry.dataset.indexers.numpy_tuple_dataset_feature_indexer import NumpyTupleDatasetFeatureIndexer  # NOQA


class NumpyTupleDataset(object):

    """Dataset of a tuple of datasets.

    It combines multiple datasets into one dataset. Each example is represented
    by a tuple whose ``i``-th item corresponds to the i-th dataset.
    And each ``i``-th dataset is expected to be an instance of numpy.ndarray.

    Args:
        datasets: Underlying datasets. The ``i``-th one is used for the
            ``i``-th item of each example. All datasets must have the same
            length.

    """

    def __init__(self, *datasets):
        if not datasets:
            raise ValueError('no datasets are given')
        length = len(datasets[0])
        for i, dataset in enumerate(datasets):
            if len(dataset) != length:
                raise ValueError(
                    'dataset of the index {} has a wrong length'.format(i))
        self._datasets = datasets
        self._length = length
#         self._features_indexer = NumpyTupleDatasetFeatureIndexer(self)
        self._features_indexer = None  # Hacking. not use it.

    def __getitem__(self, index):
        batches = [dataset[index] for dataset in self._datasets]
        if isinstance(index, (slice, list, numpy.ndarray)):
            length = len(batches[0])
            return [tuple([batch[i] for batch in batches])
                    for i in six.moves.range(length)]
        else:
            return tuple(batches)

    def __len__(self):
        return self._length

    def get_datasets(self):
        return self._datasets

    @property
    def features(self):
        """Extract features according to the specified index.

        - axis 0 is used to specify dataset id (`i`-th dataset)
        - axis 1 is used to specify feature index

        .. admonition:: Example

           >>> import numpy
           >>> from chainer_chemistry.datasets import NumpyTupleDataset
           >>> x = numpy.array([0, 1, 2], dtype=numpy.float32)
           >>> t = x * x
           >>> numpy_tuple_dataset = NumpyTupleDataset(x, t)
           >>> targets = numpy_tuple_dataset.features[:, 1]
           >>> print('targets', targets)  # We can extract only target value
           targets [0, 1, 4]

        """
        return self._features_indexer

    @classmethod
    def save(cls, filepath, numpy_tuple_dataset):
        """save the dataset to filepath in npz format

        Args:
            filepath (str): filepath to save dataset. It is recommended to end
                with '.npz' extension.
            numpy_tuple_dataset (NumpyTupleDataset): dataset instance

        """
        if not isinstance(numpy_tuple_dataset, NumpyTupleDataset):
            raise TypeError('numpy_tuple_dataset is not instance of '
                            'NumpyTupleDataset, got {}'
                            .format(type(numpy_tuple_dataset)))
        numpy.savez(filepath, *numpy_tuple_dataset._datasets)

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            return None
        load_data = numpy.load(filepath)
        result = []
        i = 0
        while True:
            key = 'arr_{}'.format(i)
            if key in load_data.keys():
                result.append(load_data[key])
                i += 1
            else:
                break
        return NumpyTupleDataset(*result)


# In[17]:


import pickle

import chainer
from chainer import cuda
from chainer.dataset.convert import concat_examples
from chainer.iterators import SerialIterator
from chainer import link
import numpy


def _to_tuple(x):
    if not isinstance(x, tuple):
        x = (x,)
    return x


def _extract_numpy(x):
    if isinstance(x, chainer.Variable):
        x = x.data
    return cuda.to_cpu(x)


class BaseForwardModel(link.Chain):

    """A base model which supports forward functionality.

    It also supports `device` id management and pickle save/load functionality.

    Args:
        device (int): GPU device id of this model to be used.
            -1 indicates to use in CPU.

    Attributes:
        _dev_id (int): Model's current device id

    """

    def __init__(self):
        super(BaseForwardModel, self).__init__()

        self.inputs = None
        self._dev_id = None

    def get_device(self):
        return self._dev_id

    def initialize(self, device=-1):
        """Initialization of the model.

        It must be executed **after** the link registration
        (often done by `with self.init_scope()` finished.

        Args:
            device (int): GPU device id of this model to be used.
            -1 indicates to use in CPU.

        """
        self.update_device(device=device)

    def update_device(self, device=-1):
        if self._dev_id is None or self._dev_id != device:
            # reset current state
            self.to_cpu()

            # update the model to specified device id
            self._dev_id = device
            if device >= 0:
                chainer.cuda.get_device_from_id(device).use()
                self.to_gpu()  # Copy the model to the GPU

    def _forward(self, data, fn, batchsize=16,
                 converter=concat_examples, retain_inputs=False,
                 preprocess_fn=None, postprocess_fn=None):
        """Forward data by iterating with batch

        Args:
            data: "train_x array" or "chainer dataset"
            fn (Callable): Main function to forward. Its input argument is
                either Variable, cupy.ndarray or numpy.ndarray, and returns
                Variable.
            batchsize (int): batch size
            converter (Callable): convert from `data` to `inputs`
            retain_inputs (bool): If True, this instance keeps inputs in
                `self.inputs` or not.
            preprocess_fn (Callable): Its input is numpy.ndarray or
                cupy.ndarray, it can return either Variable, cupy.ndarray or
                numpy.ndarray
            postprocess_fn (Callable): Its input argument is Variable,
                but this method may return either Variable, cupy.ndarray or
                numpy.ndarray.

        Returns (tuple or numpy.ndarray): forward result

        """
        input_list = None
        output_list = None
        it = SerialIterator(data, batch_size=batchsize, repeat=False,
                            shuffle=False)
        for batch in it:
            inputs = converter(batch, self._dev_id)
            inputs = _to_tuple(inputs)

            if preprocess_fn:
                inputs = preprocess_fn(*inputs)
                inputs = _to_tuple(inputs)

            outputs = fn(*inputs)
            outputs = _to_tuple(outputs)

            # Init
            if retain_inputs:
                if input_list is None:
                    input_list = [[] for _ in range(len(inputs))]
                for j, input in enumerate(inputs):
                    input_list[j].append(cuda.to_cpu(input))
            if output_list is None:
                output_list = [[] for _ in range(len(outputs))]

            if postprocess_fn:
                outputs = postprocess_fn(*outputs)
                outputs = _to_tuple(outputs)
            for j, output in enumerate(outputs):
                output_list[j].append(_extract_numpy(output))

        if retain_inputs:
            self.inputs = [numpy.concatenate(
                in_array) for in_array in input_list]

        result = [numpy.concatenate(output) for output in output_list]
        if len(result) == 1:
            return result[0]
        else:
            return result

    def save_pickle(self, filepath, protocol=None):
        """Save the model to `filepath` as a pickle file

        This function send the parameters to CPU before saving the model so
        that the pickled file can be loaded with in CPU-only environment. 
        After the model is saved, it is sent back to the original device.

        Saved pickle file can be loaded with `load_pickle` static method.

        Note that the transportability of the saved file follows the
        specification of `pickle` module, namely serialized data depends on the
        specific class or attribute structure when saved. The file may not be
        loaded in different environment (version of python or dependent
        libraries), or after large refactoring of the pickled object class.
        If you want to avoid it, use `chainer.serializers.save_npz`
        method instead to save only model parameters.

    .. admonition:: Example

       >>> from chainer_chemistry.models import BaseForwardModel
       >>> class DummyForwardModel(BaseForwardModel):
       >>> 
       >>>     def __init__(self, device=-1):
       >>>         super(DummyForwardModel, self).__init__()
       >>>         with self.init_scope():
       >>>             self.l = chainer.links.Linear(3, 10)
       >>>         self.initialize(device)
       >>> 
       >>>     def __call__(self, x):
       >>>         return self.l(x)
       >>>
       >>> model = DummyForwardModel()
       >>> filepath = 'model.pkl'
       >>> model.save_pickle(filepath)  

        Args:
            filepath (str): file path of pickle file.
            protocol (int or None): protocol version used in `pickle`.
                Use 2 if you need python2/python3 compatibility.
                3 or higher is used for python3.
                Please refer the official document [1] for more details.
                [1]: https://docs.python.org/3.6/library/pickle.html#module-interface

        """  # NOQA
        current_device = self.get_device()

        # --- Move the model to CPU for saving ---
        self.update_device(-1)
        with open(filepath, mode='wb') as f:
            pickle.dump(self, f, protocol=protocol)

        # --- Revert the model to original device ---
        self.update_device(current_device)

    @staticmethod
    def load_pickle(filepath, device=-1):
        """Load the model from `filepath` of pickle file, and send to `device`

        The file saved by `save_pickle` method can be loaded, but it may fail
        to load when loading from different develop environment or after
        updating library version.
        See `save_pickle` method for the transportability of the saved file.

    .. admonition:: Example

       >>> from chainer_chemistry.models import BaseForwardModel
       >>> filepath = 'model.pkl'
       >>> # `load_pickle` is static method, call from Class to get an instance
       >>> model = BaseForwardModel.load_pickle(filepath)

        Args:
            filepath (str): file path of pickle file.
            device (int): GPU device id of this model to be used.
                -1 indicates to use in CPU.

        """
        with open(filepath, mode='rb') as f:
            model = pickle.load(f)

        if not isinstance(model, BaseForwardModel):
            raise TypeError('Unexpected type {}'.format(type(model)))

        # --- Revert the model to specified device ---
        model.initialize(device)
        return model


# In[18]:


import numpy

import chainer
from chainer.dataset.convert import concat_examples
from chainer import cuda, Variable  # NOQA
from chainer import reporter
#from chainer_chemistry.models.prediction.base import BaseForwardModel


class Regressor(BaseForwardModel):
    """A simple regressor model.

    This is an example of chain that wraps another chain. It computes the
    loss and metrics based on a given input/label pair.

    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        metrics_fun (function or dict or None): Function that computes metrics.
        label_key (int or str): Key to specify label variable from arguments.
            When it is ``int``, a variable in positional arguments is used.
            And when it is ``str``, a variable in keyword arguments is used.
        device (int): GPU device id of this Regressor to be used.
            -1 indicates to use in CPU.

    Attributes:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        metrics (dict): Metrics computed in last minibatch
        compute_metrics (bool): If ``True``, compute metrics on the forward
            computation. The default value is ``True``.

    """

    compute_metrics = True

    def __init__(self, predictor,
                 lossfun=chainer.functions.mean_squared_error,
                 metrics_fun=None, label_key=-1, device=-1):
        if not (isinstance(label_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' %
                            type(label_key))
        super(Regressor, self).__init__()
        self.lossfun = lossfun
        if metrics_fun is None:
            self.compute_metrics = False
            self.metrics_fun = {}
        elif callable(metrics_fun):
            self.metrics_fun = {'metrics': metrics_fun}
        elif isinstance(metrics_fun, dict):
            self.metrics_fun = metrics_fun
        else:
            raise TypeError('Unexpected type metrics_fun must be None or '
                            'Callable or dict. actual {}'
                            .format(type(metrics_fun)))
        self.y = None
        self.loss = None
        self.metrics = None
        self.label_key = label_key

        with self.init_scope():
            self.predictor = predictor

        # `initialize` must be called after `init_scope`.
        self.initialize(device)

    def _convert_to_scalar(self, value):
        """Converts an input value to a scalar if its type is a Variable,
        numpy or cupy array, otherwise it returns the value as it is.
        """
        if isinstance(value, Variable):
            value = value.array
        if numpy.isscalar(value):
            return value
        if type(value) is not numpy.array:
            value = cuda.to_cpu(value)
        return numpy.asscalar(value)

    def __call__(self, *args, **kwargs):
        """Computes the loss value for an input and label pair.

        It also computes metrics and stores it to the attribute.

        Args:
            args (list of ~chainer.Variable): Input minibatch.
            kwargs (dict of ~chainer.Variable): Input minibatch.

        When ``label_key`` is ``int``, the correpoding element in ``args``
        is treated as ground truth labels. And when it is ``str``, the
        element in ``kwargs`` is used.
        The all elements of ``args`` and ``kwargs`` except the ground trush
        labels are features.
        It feeds features to the predictor and compare the result
        with ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """

        # --- Separate `args` and `t` ---
        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]
        else:
            raise TypeError('Label key type {} not supported'
                            .format(type(self.label_key)))

        self.y = None
        self.loss = None
        self.metrics = None
        self.y = self.predictor(*args, **kwargs)
        self.loss = self.lossfun(self.y, t)

        # When the reported data is a numpy array, the loss and metrics values
        # are scalars. When the reported data is a cupy array, sometimes the
        # same values become arrays instead. This seems to be a bug inside the
        # reporter class, which needs to be addressed and fixed. Until then,
        # the reported values will be converted to numpy arrays.
        reporter.report(
            {'loss': self._convert_to_scalar(self.loss)}, self)

        if self.compute_metrics:
            # Note: self.metrics_fun is `dict`,
            # which is different from original chainer implementation
            self.metrics = {key: self._convert_to_scalar(value(self.y, t))
                            for key, value in self.metrics_fun.items()}
            reporter.report(self.metrics, self)
        return self.loss

    def predict(
            self, data, batchsize=16, converter=concat_examples,
            retain_inputs=False, preprocess_fn=None, postprocess_fn=None):
        """Predict label of each category by taking .

        Args:
            data: input data
            batchsize (int): batch size
            converter (Callable): convert from `data` to `inputs`
            preprocess_fn (Callable): Its input is numpy.ndarray or
                cupy.ndarray, it can return either Variable, cupy.ndarray or
                numpy.ndarray
            postprocess_fn (Callable): Its input argument is Variable,
                but this method may return either Variable, cupy.ndarray or
                numpy.ndarray.
            retain_inputs (bool): If True, this instance keeps inputs in
                `self.inputs` or not.

        Returns (tuple or numpy.ndarray): Typically, it is 1-dimensional int
            array with shape (batchsize, ) which represents each examples
            category prediction.

        """
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            predict_labels = self._forward(
                data, fn=self.predictor, batchsize=batchsize,
                converter=converter, retain_inputs=retain_inputs,
                preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
        return predict_labels


# In[ ]:





# In[ ]:





# ## utils for this task

# Codes from [pfnet-research/sngan_projection](https://github.com/pfnet-research/sngan_projection), under MIT license.
# Some of the codes are copied from this repository to use spectral normalization for regularization of Neural Network.

# In[19]:


# --- models.sn.max_sv ---


# In[20]:


import chainer.functions as F
from chainer import cuda


def _l2normalize(v, eps=1e-12):
    norm = cuda.reduce('T x', 'T out',
                       'x * x', 'a + b', 'out = sqrt(a)', 0,
                       'norm_sn')
    div = cuda.elementwise('T x, T norm, T eps',
                           'T out',
                           'out = x / (norm + eps)',
                           'div_sn')
    return div(v, norm(v), eps)


def max_singular_value(W, u=None, Ip=1):
    """
    Apply power iteration for the weight parameter
    """
    if not Ip >= 1:
        raise ValueError("The number of power iterations should be positive integer")

    xp = cuda.get_array_module(W.data)
    if u is None:
        u = xp.random.normal(size=(1, W.shape[0])).astype(xp.float32)
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(xp.dot(_u, W.data), eps=1e-12)
        _u = _l2normalize(xp.dot(_v, W.data.transpose()), eps=1e-12)
    sigma = F.sum(F.linear(_u, F.transpose(W)) * _v)
    return sigma, _u, _v


def max_singular_value_fully_differentiable(W, u=None, Ip=1):
    """
    Apply power iteration for the weight parameter (fully differentiable version)
    """
    if not Ip >= 1:
        raise ValueError("The number of power iterations should be positive integer")

    xp = cuda.get_array_module(W.data)
    if u is None:
        u = xp.random.normal(size=(1, W.shape[0])).astype(xp.float32)
    _u = u
    for _ in range(Ip):
        _v = F.normalize(F.matmul(_u, W), eps=1e-12)
        _u = F.normalize(F.matmul(_v, F.transpose(W)), eps=1e-12)
    _u = F.matmul(_v, F.transpose(W))
    norm = F.sqrt(F.sum(_u ** 2))
    return norm, _l2normalize(_u.data), _v


# In[21]:


# --- sn_linear & sn_mlp ---


# In[22]:


import chainer
import numpy as np
from chainer import cuda
from chainer.functions.array.broadcast import broadcast_to
from chainer.functions.connection import linear
from chainer.links.connection.linear import Linear

#from src.models.sn.max_sv import max_singular_value


class SNLinear(Linear):
    """Linear layer with Spectral Normalization.

    Args:
        in_size (int): Dimension of input vectors. If ``None``, parameter
            initialization will be deferred until the first forward datasets pass
            at which time the size will be determined.
        out_size (int): Dimension of output vectors.
        wscale (float): Scaling factor of the weight matrix.
        bias (float): Initial bias value.
        nobias (bool): If ``True``, then this function does not use the bias.
        initialW (2-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        use_gamma (bool): If true, apply scalar multiplication to the
            normalized weight (i.e. reparameterize).
        Ip (int): The number of power iteration for calculating the spcetral
            norm of the weights.
        factor (float) : constant factor to adjust spectral norm of W_bar.

    .. seealso:: :func:`~chainer.functions.linear`

    Attributes:
        W (~chainer.Variable): Weight parameter.
        W_bar (~chainer.Variable): Spectrally normalized weight parameter.
        b (~chainer.Variable): Bias parameter.
        u (~numpy.array): Current estimation of the right largest singular vector of W.
        (optional) gamma (~chainer.Variable): the multiplier parameter.
        (optional) factor (float): constant factor to adjust spectral norm of W_bar.

    """

    def __init__(self, in_size, out_size, use_gamma=False, nobias=False,
                 initialW=None, initial_bias=None, Ip=1, factor=None):
        self.Ip = Ip
        self.use_gamma = use_gamma
        self.factor = factor
        super(SNLinear, self).__init__(
            in_size, out_size, nobias, initialW, initial_bias
        )
        self.u = np.random.normal(size=(1, out_size)).astype(dtype="f")
        self.register_persistent('u')

    @property
    def W_bar(self):
        """
        Spectral Normalized Weight
        """
        sigma, _u, _ = max_singular_value(self.W, self.u, self.Ip)
        if self.factor:
            sigma = sigma / self.factor
        sigma = broadcast_to(sigma.reshape((1, 1)), self.W.shape)
        self.u[:] = _u
        if hasattr(self, 'gamma'):
            return broadcast_to(self.gamma, self.W.shape) * self.W / sigma
        else:
            return self.W / sigma

    def _initialize_params(self, in_size):
        super(SNLinear, self)._initialize_params(in_size)
        if self.use_gamma:
            _, s, _ = np.linalg.svd(cuda.to_cpu(self.W.data))
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1))

    def __call__(self, x):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """
        if self.W.data is None:
            self._initialize_params(x.size // x.shape[0])
        return linear.linear(x, self.W_bar, self.b)


# In[23]:


from chainer.functions.connection import embed_id
from chainer.initializers import normal
from chainer import link
from chainer import variable
from chainer.functions.array.broadcast import broadcast_to
import numpy as np

# from src.models.sn.max_sv import max_singular_value


class SNEmbedID(link.Link):
    """Efficient linear layer for one-hot input.
    This is a link that wraps the :func:`~chainer.functions.embed_id` function.
    This link holds the ID (word) embedding matrix ``W`` as a parameter.
    Args:
        in_size (int): Number of different identifiers (a.k.a. vocabulary
            size).
        out_size (int): Size of embedding vector.
        initialW (2-D array): Initial weight value. If ``None``, then the
            matrix is initialized from the standard normal distribution.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        ignore_label (int or None): If ``ignore_label`` is an int value,
            ``i``-th column of return value is filled with ``0``.
        Ip (int): The number of power iteration for calculating the spcetral
            norm of the weights.
        factor (float) : constant factor to adjust spectral norm of W_bar.
    .. seealso:: :func:`chainer.functions.embed_id`
    Attributes:
        W (~chainer.Variable): Embedding parameter matrix.
        W_bar (~chainer.Variable): Spectrally normalized weight parameter.
        u (~numpy.array): Current estimation of the right largest singular vector of W.
        (optional) gamma (~chainer.Variable): the multiplier parameter.
        (optional) factor (float): constant factor to adjust spectral norm of W_bar.
    """

    ignore_label = None

    def __init__(self, in_size, out_size, initialW=None, ignore_label=None, Ip=1, factor=None):
        super(SNEmbedID, self).__init__()
        self.ignore_label = ignore_label
        self.Ip = Ip
        self.factor = factor
        with self.init_scope():
            if initialW is None:
                initialW = normal.Normal(1.0)
            self.W = variable.Parameter(initialW, (in_size, out_size))

        self.u = np.random.normal(size=(1, in_size)).astype(dtype="f")
        self.register_persistent('u')

    @property
    def W_bar(self):
        """
        Spectral Normalized Weight
        """
        sigma, _u, _ = max_singular_value(self.W, self.u, self.Ip)
        if self.factor:
            sigma = sigma / self.factor
        sigma = broadcast_to(sigma.reshape((1, 1)), self.W.shape)
        self.u[:] = _u
        return self.W / sigma

    def __call__(self, x):
        """Extracts the word embedding of given IDs.
        Args:
            x (~chainer.Variable): Batch vectors of IDs.
        Returns:
            ~chainer.Variable: Batch of corresponding embeddings.
        """
        return embed_id.embed_id(x, self.W_bar, ignore_label=self.ignore_label)


# Neural Network model definition...

# In[28]:


# --- mlp ---


# In[29]:


import chainer
from chainer.functions import relu
import chainer.functions as F
from chainer import links

class MLP(chainer.Chain):

    """Basic implementation for MLP

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_layers (int): number of layers
        activation (chainer.functions): activation function
    """

    def __init__(self, out_dim, hidden_dim=16, n_layers=2, activation=relu,
                 use_bn=False, use_sn=False, use_gamma=True, use_residual=False,
                 dropout_ratio=0):
        super(MLP, self).__init__()
        if n_layers <= 0:
            raise ValueError('n_layers must be a positive integer, but it was '
                             'set to {}'.format(n_layers))
        if use_sn:
            layers = [SNLinear(None, hidden_dim, use_gamma=use_gamma) for i in range(n_layers - 1)]
        else:
            layers = [links.Linear(None, hidden_dim) for i in range(n_layers - 1)]
        with self.init_scope():
            self.layers = chainer.ChainList(*layers)
            if use_sn:
                self.l_out = SNLinear(None, out_dim, use_gamma=use_gamma)
            else:
                self.l_out = links.Linear(None, out_dim)
            if use_bn:
                self.bn_layers = chainer.ChainList(
                    *[links.BatchNormalization(hidden_dim) for i in range(n_layers - 1)])

        self.activation = activation
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.dropout_ratio = dropout_ratio

    def __call__(self, x):
        h = x
        if self.use_bn:
            for l, bn in zip(self.layers, self.bn_layers):
                if self.dropout_ratio > 0:
                    h = F.dropout(x, self.dropout_ratio)
                h2 = self.activation(bn(l(h)))
                if self.use_residual and h.shape == h2.shape:
                    h = h + h2
                else:
                    h = h2
        else:
            for l in self.layers:
                if self.dropout_ratio > 0:
                    h = F.dropout(x, self.dropout_ratio)
                h2 = self.activation(l(h))
                if self.use_residual and h.shape == h2.shape:
                    h = h + h2
                else:
                    h = h2
        h = self.l_out(h)
        return h


class ProjectionMLP(chainer.Chain):

    """Basic implementation for MLP

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_layers (int): number of layers
        activation (chainer.functions): activation function
    """

    def __init__(self, out_dim, hidden_dim=16, n_layers=2, activation=relu,
                 use_bn=False, use_sn=False, use_gamma=True, use_residual=False):
        super(ProjectionMLP, self).__init__()
        if n_layers <= 0:
            raise ValueError('n_layers must be a positive integer, but it was '
                             'set to {}'.format(n_layers))
        if use_sn:
            layers = [SNLinear(None, hidden_dim, use_gamma=use_gamma) for i in range(n_layers - 1)]
        else:
            layers = [links.Linear(None, hidden_dim) for i in range(n_layers - 1)]
        with self.init_scope():
            self.layers = chainer.ChainList(*layers)
            if use_sn:
                self.l_out = SNLinear(None, out_dim, use_gamma=use_gamma)
            else:
                self.l_out = links.Linear(None, out_dim)
            if use_bn:
                self.bn_layers = chainer.ChainList(
                    *[links.BatchNormalization(hidden_dim) for i in range(n_layers - 1)])
            cat_size = 2  # dog and cat
            self.embed = links.EmbedID(cat_size, hidden_dim)

        self.activation = activation
        self.use_bn = use_bn
        self.use_residual = use_residual

    def __call__(self, x, cat):
        h = x
        h_cat = self.embed(cat)
        if self.use_bn:
            for i, (l, bn) in enumerate(zip(self.layers, self.bn_layers)):
                h2 = self.activation(bn(l(h)))
                if self.use_residual and h.shape == h2.shape:
                    h = h + h2
                else:
                    h = h2
                # if i == 0:
                #     h = h * h_cat
        else:
            for i, l in enumerate(self.layers):
                h2 = self.activation(l(h))
                if self.use_residual and h.shape == h2.shape:
                    h = h + h2
                else:
                    h = h2
                # if i == 0:
                #     h = h * h_cat
        # h_cat = self.embed(cat)
        h = h + h * h_cat
        h = self.l_out(h)
        return h


# In[30]:


# --- mlp2 ---

import chainer
from chainer.functions import relu
import chainer.functions as F
from chainer import links


class MLP2(chainer.Chain):

    """Basic implementation for MLP

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_layers (int): number of layers
        activation (chainer.functions): activation function
    """

    def __init__(self, out_dim, layer_dims=None, activation=relu,
                 use_bn=False, use_sn=False, use_gamma=True, use_residual=False,
                 dropout_ratio=0, scaler=None):
        super(MLP2, self).__init__()
        if layer_dims is None:
            layer_dims = [16, 16]
        n_layers = len(layer_dims) + 1
        if n_layers <= 0:
            raise ValueError('n_layers must be a positive integer, but it was '
                             'set to {}'.format(n_layers))
        if use_sn:
            layers = [SNLinear(None, layer_dims[i], use_gamma=use_gamma) for i in range(n_layers - 1)]
        else:
            layers = [links.Linear(None, layer_dims[i]) for i in range(n_layers - 1)]
        with self.init_scope():
            self.layers = chainer.ChainList(*layers)
            if use_sn:
                self.l_out = SNLinear(None, out_dim, use_gamma=use_gamma)
            else:
                self.l_out = links.Linear(None, out_dim)
            if use_bn:
                self.bn_layers = chainer.ChainList(
                    *[links.BatchNormalization(layer_dims[i]) for i in range(n_layers - 1)])
            if scaler is not None:
                self.scaler = scaler
        if scaler is None:
            self.scaler = None

        self.activation = activation
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.dropout_ratio = dropout_ratio

    def calc(self, x, extract_hidden=False):
        if self.scaler is None:
            h = x
        else:
            h = self.scaler.transform(x)
        if self.use_bn:
            for l, bn in zip(self.layers, self.bn_layers):
                if self.dropout_ratio > 0:
                    h = F.dropout(x, self.dropout_ratio)
                h2 = self.activation(bn(l(h)))
                if self.use_residual and h.shape == h2.shape:
                    h = h + h2
                else:
                    h = h2
        else:
            for l in self.layers:
                if self.dropout_ratio > 0:
                    h = F.dropout(x, self.dropout_ratio)
                h2 = self.activation(l(h))
                if self.use_residual and h.shape == h2.shape:
                    h = h + h2
                else:
                    h = h2
        if extract_hidden:
            return h
        h = self.l_out(h)
        return h

    def __call__(self, x):
        return self.calc(x)


# In[31]:


# --- blendnet ---


# In[32]:


import chainer

import chainer.functions as F
import chainer.links as L
from chainer import functions


def lrelu(x):
    return functions.leaky_relu(x, slope=0.05)


class BlendNet(chainer.Chain):

    def __init__(self, num_cat_id=None, out_dim=1, activation=lrelu,
                 dropout_ratio=-1, use_bn=False, use_residual=False,
                 numeric_hidden_dim=96, embed_dim=10, bert_hidden_dim=32,
                 image_hidden_dim=96, mlp_hidden_dim=128, mlp_n_layers=6,
                 cat_hidden_dim=32):
        """

        Args:
            num_cat_id:
            out_dim:
            activation:
            dropout_ratio:
            use_bn:
            numeric_hidden_dim: numerical feature
            embed_dim: category feature, this is not used when `cat2num=True` at preprocessing.
            bert_hidden_dim:
            image_hidden_dim:
            mlp_hidden_dim:
            mlp_n_layers:
        """
        super(BlendNet, self).__init__()
        print('num_cat_id', num_cat_id)  # len(num_cat_id)
        projection = False
        self.projection = projection
        print('projection', projection)
        self.use_embed = not isinstance(num_cat_id, (int, float))
        with self.init_scope():
            if num_cat_id is not None:
                if self.use_embed:
                    self.embed_list = chainer.ChainList(
                        *[L.EmbedID(insize, embed_dim) for insize in num_cat_id])
                else:
                    self.l_cat = L.Linear(None, cat_hidden_dim)
            self.l_num = L.Linear(None, numeric_hidden_dim)
            self.l_bert = L.Linear(None, bert_hidden_dim)
            self.l_image = L.Linear(None, image_hidden_dim)
            if projection:
                self.mlp = ProjectionMLP(
                    out_dim=out_dim, hidden_dim=mlp_hidden_dim,
                    n_layers=mlp_n_layers, activation=activation,
                    use_bn=use_bn, use_residual=use_residual)
            else:
                self.mlp = MLP(
                    out_dim=out_dim, hidden_dim=mlp_hidden_dim,
                    n_layers=mlp_n_layers, activation=activation,
                    use_bn=use_bn, use_residual=use_residual)
            # if use_bn:
            #     self.bn1 = L.BatchNormalization()
        self.activation = activation
        self.bert_hidden_dim = bert_hidden_dim
        self.dropout_ratio = dropout_ratio
        self.num_cat_id = num_cat_id
        self.use_bn = use_bn

    def forward(self, x_numeric, x_cat=None, x_bert=None, x_image=None):
        h_num = self.l_num(x_numeric)
        h_feat_list = [h_num]
        if x_cat is not None:
            if self.use_embed:
                h_cat_list = [l_cat(x_cat[:, i]) for i, l_cat in enumerate(self.embed_list)]
                h_feat_list.extend(h_cat_list)
            else:
                h_feat_list.append(self.l_cat(x_cat))
        if x_bert is not None:
            # x_bert (bs, num_extract_seq, hdim)
            # --- 1. simply take linear, it will reshape  ---
            # h_bert = self.l_bert(x_bert)
            # --- 2. take linear for each element and sum it. ---
            bs, num_sent, hdim = x_bert.shape
            h_bert = F.reshape(self.l_bert(F.reshape(x_bert, (bs*num_sent, hdim))),
                               (bs, num_sent, self.bert_hidden_dim))
            h_bert = F.sum(h_bert, axis=1)
            # h_bert (bs, bert_hidden_dim)

            # print('x_bert', x_bert.shape, 'h_bert', h_bert.shape)
            h_feat_list.append(h_bert)
        if x_image is not None:
            h_image = self.l_image(x_image)
            h_feat_list.append(h_image)

        h = F.concat(h_feat_list, axis=1)
        # if self.use_bn:
        #     h = self.bn1(h)
        if self.dropout_ratio > 0:
            h = F.dropout(h, ratio=self.dropout_ratio)
        h = self.activation(h)
        if self.projection:
            h = self.mlp(h, x_cat[:, 0].astype(self.xp.int32))
        else:
            h = self.mlp(h)
        return h


# In[33]:


# blendnet regressor


# In[34]:


import chainer

import chainer.functions as F
import chainer.links as L
from chainer.dataset.convert import concat_examples
from chainer import functions, reporter

# from chainer_chemistry.models import Regressor
# from src.models.mlp import MLP


def lrelu(x):
    return functions.leaky_relu(x, slope=0.05)


class BlendNetRegressor(Regressor):

    def __init__(self, predictor,
                 lossfun=chainer.functions.mean_squared_error,
                 metrics_fun=None, label_key=-1, device=-1,
                 x_numeric_dim=None, sgnet=None, lam_image_recon=1.0, use_sn=False,
                 image_encoder_layers=1, image_encoder_hdim=32, mode='normal', image_input_dim=0,
                 dropout_ratio=0, image_encode=True):
        assert isinstance(x_numeric_dim, int)
        super(BlendNetRegressor, self).__init__(
            predictor, lossfun=lossfun, metrics_fun=metrics_fun,
            label_key=label_key, device=device)

        cat_size = 2  # dog and cat
        image_projection = True
        self.image_projection = image_projection
        bert_projection = True
        self.bert_projection = bert_projection
        print('image_projection', image_projection, 'dropout_ratio', dropout_ratio)
        print('bert_projection', bert_projection)
        with self.init_scope():
            # TODO: compare with MLP
            # self.image_encoder = L.Linear(None, 64)
            # self.image_decoder = L.Linear(None, x_numeric_dim)
            if image_encode:
                self.image_encoder = MLP(
                    out_dim=image_encoder_hdim, hidden_dim=image_encoder_hdim,
                    n_layers=image_encoder_layers, use_sn=use_sn, dropout_ratio=dropout_ratio)
                self.image_decoder = MLP(out_dim=x_numeric_dim, hidden_dim=image_encoder_hdim, n_layers=image_encoder_layers, use_sn=use_sn)
            if self.image_projection:
                self.embed_image_projection = L.EmbedID(cat_size, image_input_dim)  # TODO: remove hard coding...
            if self.bert_projection:
                self.embed_bert_projection = L.EmbedID(cat_size, 768)  # TODO: remove hard coding...
            if sgnet is not None:
                self.sgnet = sgnet

        if sgnet is None:
            self.sgnet = sgnet

        self.image_encode = image_encode
        self.lam_image_recon = lam_image_recon
        print('self.lam_image_recon', self.lam_image_recon)
        self.mode = mode
        print('self.mode', self.mode)
        self.initialize(device)
        print('initialize', device)

    def __call__(self, *args, **kwargs):
        if self.mode == 'normal':
            x_numeric, x_cat, x_bert, x_image, t = args
        else:
            assert self.mode == 'mean'
            x_numeric, x_cat, x_bert, x_image, t_, indices, target_mean = args
            t = target_mean

        self.y = None
        self.loss = None
        self.metrics = None
        y, h_image = self.calc(x_numeric, x_cat, x_bert, x_image, *args,
                               return_all=True)

        if self.lam_image_recon > 0.:
            x_numeric_recon = self.image_decoder(h_image)
            recon_loss = F.mean_squared_error(x_numeric_recon, x_numeric)
        else:
            recon_loss = 0.
        reg_loss = self.lossfun(self.y, t)
        recon_loss = self.lam_image_recon * recon_loss

        self.loss = reg_loss + recon_loss
        reporter.report(
            {'loss': self.loss, 'reg_loss': reg_loss, 'recon_loss': recon_loss},
            self)

        if self.compute_metrics:
            # Note: self.metrics_fun is `dict`,
            # which is different from original chainer implementation
            self.metrics = {key: self._convert_to_scalar(value(self.y, t))
                            for key, value in self.metrics_fun.items()}
            reporter.report(self.metrics, self)
        return self.loss

    def calc(self, x_numeric, x_cat, x_bert, x_image, *args,
             return_all=False):
        if self.sgnet is not None:
            x_numeric, x_cat, x_bert, x_image = self.sgnet(
                x_numeric, x_cat, x_bert, x_image)

        if self.image_projection and x_image is not None:
            # TODO: support when num_image > 1
            x_image = x_image * self.embed_image_projection(x_cat[:, 0].astype(self.xp.int32))[:, None, :]
        if self.bert_projection and x_bert is not None:
            # TODO: support when num_sentence > 1
            x_bert = x_bert * self.embed_bert_projection(x_cat[:, 0].astype(self.xp.int32))[:, None, :]

        if x_image is None:
            h_image = None
        elif self.image_encode:
            h_image = self.image_encoder(x_image)
        else:
            h_image = x_image
        self.y = self.predictor(x_numeric, x_cat, x_bert, h_image)
        if self.mode == 'mean':
            assert len(args) == 0
            indices = args[0]
            self.y = self.calc_attention_mean(self.y, indices)
        if return_all:
            return self.y, h_image
        else:
            return self.y

    def scatter_softmax(self, a, indices):
        # TODO: subtract by scatter_max for computation stability
        # Currently, just use overall max of `a`.
        a = a - F.max(a)
        a = F.exp(a)
        #  self.xp.zeros(indices.shape, dtype=self.xp.float32)
        alpha = self.xp.zeros((int(self.xp.max(indices))+1,), dtype=self.xp.float32)
        alpha = a / (F.scatter_add(alpha, indices, a)[indices] + 1e-16)
        return alpha

    def calc_attention_mean(self, y, indices):
        assert y.ndim == 2
        assert indices.ndim == 1
        assert y.shape[0] == indices.shape[0]
        assert y.shape[1] == 2
        # --- scatter attention by softmax part ---
        a = y[:, 0]
        alpha = self.scatter_softmax(a, indices)

        result = self.xp.zeros((int(self.xp.max(indices))+1,), dtype=self.xp.float32)
        b = y[:, 1]
        result = F.scatter_add(result, indices, alpha * b)
        return result[:, None]  # Add second axis, which is same shape with `target_mean`.

    def predict(
            self, data, batchsize=16, converter=concat_examples,
            retain_inputs=False, preprocess_fn=None, postprocess_fn=None):
        """Predict label of each category by taking .

        Args:
            data: input data
            batchsize (int): batch size
            converter (Callable): convert from `data` to `inputs`
            preprocess_fn (Callable): Its input is numpy.ndarray or
                cupy.ndarray, it can return either Variable, cupy.ndarray or
                numpy.ndarray
            postprocess_fn (Callable): Its input argument is Variable,
                but this method may return either Variable, cupy.ndarray or
                numpy.ndarray.
            retain_inputs (bool): If True, this instance keeps inputs in
                `self.inputs` or not.

        Returns (tuple or numpy.ndarray): Typically, it is 1-dimensional int
            array with shape (batchsize, ) which represents each examples
            category prediction.

        """
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            predict_labels = self._forward(
                data, fn=self.calc, batchsize=batchsize,
                converter=converter, retain_inputs=retain_inputs,
                preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
        return predict_labels


# In[ ]:


# blendnet mean regressor


# In[ ]:


import chainer

import chainer.functions as F
import chainer.links as L
from chainer.dataset.convert import concat_examples
from chainer import functions, reporter

# from chainer_chemistry.models import Regressor
# from src.models.mlp import MLP


def lrelu(x):
    return functions.leaky_relu(x, slope=0.05)


class BlendNetMeanRegressor(Regressor):

    def __init__(self, predictor, mean_predictor,
                 lossfun=chainer.functions.mean_squared_error,
                 metrics_fun=None, label_key=-1, device=-1,
                 x_numeric_dim=None, sgnet=None, lam_image_recon=1.0, use_sn=False,
                 image_encoder_layers=1, image_encoder_hdim=32, image_input_dim=0,
                 dropout_ratio=0, image_encode=True):
        assert isinstance(x_numeric_dim, int)
        super(BlendNetMeanRegressor, self).__init__(
            predictor, lossfun=lossfun, metrics_fun=metrics_fun,
            label_key=label_key, device=device)

        # self.lam_mean = 10.0
        self.lam_mean = 0.50
        # self.lam_mean = 0.3
        image_projection = True
        cat_size = 2  # dog and cat
        self.image_projection = image_projection
        print('image_projection', image_projection)
        with self.init_scope():
            # TODO: compare with MLP
            # self.image_encoder = L.Linear(None, 64)
            # self.image_decoder = L.Linear(None, x_numeric_dim)
            self.mean_predictor = mean_predictor
            hdim = 16
            # self.set_mlp_block = SetMLPBlock(out_dim=hdim, hidden_dim=hdim, n_layers=0,
            #                                  use_sn=use_sn)
            self.mean_predictor_post_mlp = MLP(out_dim=1, hidden_dim=hdim, n_layers=1, use_sn=use_sn)
            if image_encode:
                self.image_encoder = MLP(out_dim=image_encoder_hdim, hidden_dim=image_encoder_hdim,
                                         n_layers=image_encoder_layers, use_sn=use_sn, dropout_ratio=dropout_ratio)
                self.image_decoder = MLP(out_dim=x_numeric_dim, hidden_dim=image_encoder_hdim,
                                         n_layers=image_encoder_layers, use_sn=use_sn)
            if self.image_projection:
                self.embed_image_projection = L.EmbedID(cat_size, image_input_dim)
            if sgnet is not None:
                self.sgnet = sgnet

        if sgnet is None:
            self.sgnet = sgnet

        self.image_encode = image_encode
        self.lam_image_recon = lam_image_recon
        print('self.lam_image_recon', self.lam_image_recon)
        self.initialize(device)
        print('initialize', device)

    def __call__(self, *args, **kwargs):
        x_numeric, x_cat, x_bert, x_image, t, indices, target_mean = args

        self.y = None
        self.loss = None
        self.metrics = None

        y, h_image, y_mean = self.calc(x_numeric, x_cat, x_bert, x_image, indices, return_all=True)

        # 1. ImageEncoder loss
        if h_image is not None:
            if self.lam_image_recon > 0.:
                x_numeric_recon = self.image_decoder(h_image)
                recon_loss = F.mean_squared_error(x_numeric_recon, x_numeric)
            else:
                recon_loss = 0.
        else:
            recon_loss = 0.

        # 2. target_mean loss
        # reg_loss_mean = self.lossfun(y_mean, target_mean)
        reg_loss_mean = self.lossfun(y_mean[indices], target_mean[indices])
        reg_loss_mean = self.lam_mean * reg_loss_mean
        # 3. t loss
        reg_loss = self.lossfun(self.y, t)
        recon_loss = self.lam_image_recon * recon_loss

        self.loss = reg_loss + reg_loss_mean + recon_loss
        reporter.report(
            {'loss': self.loss, 'reg_loss': reg_loss, 'recon_loss': recon_loss,
             'reg_loss_mean': reg_loss_mean},
            self)

        if self.compute_metrics:
            # Note: self.metrics_fun is `dict`,
            # which is different from original chainer implementation
            self.metrics = {key: self._convert_to_scalar(value(self.y, t))
                            for key, value in self.metrics_fun.items()}
            reporter.report(self.metrics, self)
        return self.loss

    def calc(self, x_numeric, x_cat, x_bert, x_image, indices, *args,
             return_all=False):
        assert len(args) == 0
        # if len(args) != 0:
        #     print('args ', len(args))
        #     import IPython; IPython.embed()
        if self.sgnet is not None:
            x_numeric, x_cat, x_bert, x_image = self.sgnet(
                x_numeric, x_cat, x_bert, x_image)
        if x_image is None:
            h_image = None
        elif self.image_encode:
            if self.image_projection:
                h_image = x_image[:, 0, :] * self.embed_image_projection(x_cat[:, 0].astype(self.xp.int32))
            h_image = self.image_encoder(h_image)
        else:
            h_image = x_image

        # --- 1st step: predict target_mean ---
        h_mean = self.mean_predictor(x_numeric, x_cat, x_bert, h_image, indices)
        # h_mean = self.mean_predictor(x_numeric, x_cat, x_bert, h_image)
        # calc softmax mean
        h_mean_agg = self.calc_attention_mean(h_mean, indices)
        # h_mean_agg = self.set_mlp_block.calc_agg(h_mean, indices)
        y_mean = self.mean_predictor_post_mlp(h_mean_agg)
        # `t` is target_mean in this case...

        # --- 2nd step: predict t ---
        # h_numeric_agg = F.concat([h_mean_agg, y_mean], axis=1)  # (bs, 1+hdim)
        h_numeric_agg = y_mean
        # print('h_numeric_agg', h_numeric_agg.shape, 'indices', indices.shape)
        h_numeric = h_numeric_agg[indices]
        # print('h_numeric', h_numeric.shape)
        h_numeric = F.concat([x_numeric, h_numeric], axis=1)
        # h_numeric = x_numeric

        self.y = self.predictor(h_numeric, x_cat, x_bert, h_image)
        if return_all:
            return self.y, h_image, y_mean
        else:
            return self.y

    def scatter_softmax_1d(self, a, indices):
        """
        Args:
            a: 1d-array (num_examples,)
            indices: (num_rescuer_id,)

        Returns:
            alpha: Array shape is same with `a`, (num_rescuer_id,)
        """
        # TODO: subtract by scatter_max for computation stability
        # Currently, just use overall max of `a`.
        a = a - F.max(a)
        a = F.exp(a)
        #  self.xp.zeros(indices.shape, dtype=self.xp.float32)
        z = self.xp.zeros((int(self.xp.max(indices))+1,), dtype=self.xp.float32)
        alpha = a / (F.scatter_add(z, indices, a)[indices] + 1e-16)
        return alpha

    def scatter_softmax_2d(self, a, indices):
        """
        Args:
            a: 2d-array (num_examples, hdim)
            indices: (num_rescuer_id,)

        Returns:
            alpha: Array shape is same with `a`, (num_rescuer_id, hdim)
        """
        # TODO: subtract by scatter_max for computation stability
        # Currently, just use overall max of `a`.
        a = a - F.max(a, axis=1, keepdims=True)
        a = F.exp(a)
        #  self.xp.zeros(indices.shape, dtype=self.xp.float32)
        hdim = a.shape[1]
        z = self.xp.zeros((int(self.xp.max(indices))+1, hdim), dtype=self.xp.float32)
        alpha = a / (F.scatter_add(z, indices, a)[indices] + 1e-16)
        return alpha

    def calc_attention_mean(self, y, indices):
        assert y.ndim == 2
        assert indices.ndim == 1
        assert y.shape[0] == indices.shape[0]
        # assert y.shape[1] % 2 == 0  # sometimes this does not guaranteed...
        # --- scatter attention by softmax part ---
        outdim = y.shape[1] // 2
        a = y[:, :outdim]
        # alpha = self.scatter_softmax(a, indices)
        alpha = self.scatter_softmax_2d(a, indices)

        result = self.xp.zeros((int(self.xp.max(indices))+1, outdim), dtype=self.xp.float32)
        b = y[:, outdim:outdim*2]
        result = F.scatter_add(result, indices, alpha * b)
        return result

    def predict(
            self, data, batchsize=16, converter=concat_examples,
            retain_inputs=False, preprocess_fn=None, postprocess_fn=None):
        """Predict label of each category by taking .

        Args:
            data: input data
            batchsize (int): batch size
            converter (Callable): convert from `data` to `inputs`
            preprocess_fn (Callable): Its input is numpy.ndarray or
                cupy.ndarray, it can return either Variable, cupy.ndarray or
                numpy.ndarray
            postprocess_fn (Callable): Its input argument is Variable,
                but this method may return either Variable, cupy.ndarray or
                numpy.ndarray.
            retain_inputs (bool): If True, this instance keeps inputs in
                `self.inputs` or not.

        Returns (tuple or numpy.ndarray): Typically, it is 1-dimensional int
            array with shape (batchsize, ) which represents each examples
            category prediction.

        """
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            predict_labels = self._forward(
                data, fn=self.calc, batchsize=batchsize,
                converter=converter, retain_inputs=retain_inputs,
                preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
        return predict_labels


# In[35]:


# sn blendnet


# In[ ]:


import chainer

import chainer.functions as F
import chainer.links as L
from chainer import functions

# from src.models.mlp import MLP, ProjectionMLP
# from src.models.sn.sn_embed_id import SNEmbedID
# from src.models.sn.sn_linear import SNLinear


def lrelu(x):
    return functions.leaky_relu(x, slope=0.05)


class SNBlendNet(chainer.Chain):
    """BlendNet with Spectral Normalization"""

    def __init__(self, num_cat_id=None, out_dim=1, activation=lrelu,
                 dropout_ratio=-1, use_bn=False,
                 numeric_hidden_dim=96, embed_dim=10, bert_hidden_dim=32,
                 image_hidden_dim=96, mlp_hidden_dim=128, mlp_n_layers=6,
                 cat_hidden_dim=32, use_gamma=True, use_residual=False):
        """

        Args:
            num_cat_id:
            out_dim:
            activation:
            dropout_ratio:
            use_bn:
            numeric_hidden_dim: numerical feature
            embed_dim: category feature, this is not used when `cat2num=True` at preprocessing.
            bert_hidden_dim:
            image_hidden_dim:
            mlp_hidden_dim:
            mlp_n_layers:
        """
        super(SNBlendNet, self).__init__()
        print('num_cat_id', num_cat_id)  # len(num_cat_id)
        print('spectral normalizaiton ON, use_gamma {}'.format(use_gamma))
        projection = False
        self.projection = projection
        print('projection', projection)
        self.use_embed = not isinstance(num_cat_id, (int, float))
        with self.init_scope():
            if num_cat_id is not None:
                if self.use_embed:
                    self.embed_list = chainer.ChainList(
                        *[SNEmbedID(insize, embed_dim) for insize in num_cat_id])
                else:
                    self.l_cat = SNLinear(None, cat_hidden_dim, use_gamma=use_gamma)
            self.l_num = SNLinear(None, numeric_hidden_dim, use_gamma=use_gamma)
            self.l_bert = SNLinear(None, bert_hidden_dim, use_gamma=use_gamma)
            self.l_image = SNLinear(None, image_hidden_dim, use_gamma=use_gamma)
            if projection:
                self.mlp = ProjectionMLP(
                    out_dim=out_dim, hidden_dim=mlp_hidden_dim,
                    n_layers=mlp_n_layers, activation=activation,
                    use_bn=use_bn, use_sn=True, use_gamma=use_gamma,
                    use_residual=use_residual)
            else:
                self.mlp = MLP(
                    out_dim=out_dim, hidden_dim=mlp_hidden_dim,
                    n_layers=mlp_n_layers, activation=activation,
                    use_bn=use_bn, use_sn=True, use_gamma=use_gamma,
                    use_residual=use_residual)
            # if use_bn:
            #     self.bn1 = L.BatchNormalization()
        self.activation = activation
        self.bert_hidden_dim = bert_hidden_dim
        self.dropout_ratio = dropout_ratio
        self.num_cat_id = num_cat_id
        self.use_bn = use_bn

    def forward(self, x_numeric, x_cat=None, x_bert=None, x_image=None):
        h_num = self.l_num(x_numeric)
        h_feat_list = [h_num]
        if x_cat is not None:
            if self.use_embed:
                h_cat_list = [l_cat(x_cat[:, i]) for i, l_cat in enumerate(self.embed_list)]
                h_feat_list.extend(h_cat_list)
            else:
                h_feat_list.append(self.l_cat(x_cat))
        if x_bert is not None:
            # x_bert (bs, num_extract_seq, hdim)
            # --- 1. simply take linear, it will reshape  ---
            # h_bert = self.l_bert(x_bert)
            # --- 2. take linear for each element and sum it. ---
            bs, num_sent, hdim = x_bert.shape
            h_bert = F.reshape(self.l_bert(F.reshape(x_bert, (bs*num_sent, hdim))),
                               (bs, num_sent, self.bert_hidden_dim))
            h_bert = F.sum(h_bert, axis=1)
            # h_bert (bs, bert_hidden_dim)

            # print('x_bert', x_bert.shape, 'h_bert', h_bert.shape)
            h_feat_list.append(h_bert)
        if x_image is not None:
            h_image = self.l_image(x_image)
            h_feat_list.append(h_image)

        h = F.concat(h_feat_list, axis=1)
        # if self.use_bn:
        #     h = self.bn1(h)
        if self.dropout_ratio > 0:
            h = F.dropout(h, ratio=self.dropout_ratio)
        h = self.activation(h)
        if self.projection:
            h = self.mlp(h, x_cat[:, 0].astype(self.xp.int32))
        else:
            h = self.mlp(h)
        return h


# In[ ]:


# set sn blendnet


# In[ ]:


import chainer

import chainer.functions as F
import chainer.links as L
from chainer import functions
from chainer.functions.activation import relu

# from src.models.mlp import MLP
# from src.models.sn.sn_embed_id import SNEmbedID
# from src.models.sn.sn_linear import SNLinear


def lrelu(x):
    return functions.leaky_relu(x, slope=0.05)


def scatter_softmax_2d(xp, a, indices):
    """
    Args:
        a: 2d-array (num_examples, hdim)
        indices: (num_rescuer_id,)

    Returns:
        alpha: Array shape is same with `a`, (num_rescuer_id, hdim)
    """
    # TODO: subtract by scatter_max for computation stability
    # Currently, just use overall max of `a`.
    a = a - F.max(a, axis=1, keepdims=True)
    a = F.exp(a)
    #  self.xp.zeros(indices.shape, dtype=self.xp.float32)
    hdim = a.shape[1]
    z = xp.zeros((int(xp.max(indices))+1, hdim), dtype=xp.float32)
    alpha = a / (F.scatter_add(z, indices, a)[indices] + 1e-16)
    return alpha


def calc_num_examples(xp, indices):
    ones = xp.ones((indices.shape[0], 1), dtype=xp.float32)
    num_examples = xp.zeros((int(xp.max(indices)) + 1, 1), dtype=xp.float32)
    num_examples = F.scatter_add(num_examples, indices, ones)
    return num_examples


def calc_attention_mean(xp, y, indices):
    assert y.ndim == 2
    assert indices.ndim == 1
    assert y.shape[0] == indices.shape[0]
    # assert y.shape[1] % 2 == 0
    # --- scatter attention by softmax part ---
    outdim = y.shape[1] // 2
    a = y[:, :outdim]
    # alpha = self.scatter_softmax(a, indices)
    alpha = scatter_softmax_2d(xp, a, indices)

    result = xp.zeros((int(xp.max(indices))+1, outdim), dtype=xp.float32)
    b = y[:, outdim:2*outdim]
    result = F.scatter_add(result, indices, alpha * b)
    return result


class SetMLPBlock(chainer.Chain):
    def __init__(self, out_dim, hidden_dim=16, n_layers=2, activation=relu,
                 use_bn=False, use_sn=False, use_gamma=True, use_residual=False):
        super(SetMLPBlock, self).__init__()
        # assert out_dim % 2 == 0, 'out_dim is {}'.format(out_dim)
        with self.init_scope():
            if n_layers > 0:
                self.mlp = MLP(
                    out_dim // 2, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation,
                    use_bn=use_bn, use_sn=use_sn, use_gamma=use_gamma, use_residual=use_residual)
            self.i_layer = L.Linear(None, out_dim // 2)
            self.j_layer = L.Linear(None, out_dim // 2)
        if n_layers <= 0:
            self.mlp = None

    def calc_agg(self, y, indices):
        assert y.ndim == 2
        assert indices.ndim == 1
        assert y.shape[0] == indices.shape[0]
        # --- scatter part ---
        h = F.sigmoid(self.i_layer(y)) * self.j_layer(y)
        result = self.xp.zeros((int(self.xp.max(indices))+1, h.shape[1]), dtype=self.xp.float32)
        result = F.scatter_add(result, indices, h)

        num_examples = calc_num_examples(self.xp, indices)
        # ones = self.xp.ones((h.shape[0], 1), dtype=self.xp.float32)
        # num_examples = self.xp.zeros((int(self.xp.max(indices)) + 1, 1), dtype=self.xp.float32)
        # num_examples = F.scatter_add(num_examples, indices, ones)

        # Take mean of features...
        # return result / num_examples
        # concat `num_examples` as feature...
        return F.concat([result / num_examples, num_examples, num_examples], axis=1)

    def __call__(self, x, indices):
        if self.mlp is not None:
            h = self.mlp(x)
        else:
            h = x
        # H = self.calc_agg(h, indices)
        H = calc_attention_mean(self.xp, h, indices)
        num_examples = calc_num_examples(self.xp, indices)
        return F.concat([h, H[indices], num_examples[indices]], axis=1)


class SetMLP(chainer.Chain):
    def __init__(self, out_dim, hidden_dim=16, n_layers=2, activation=relu,
                 use_bn=False, use_sn=False, use_gamma=True, use_residual=False):
        super(SetMLP, self).__init__()
        assert out_dim % 2 == 0, 'out_dim is {}'.format(out_dim)
        with self.init_scope():
            self.mlp1 = SetMLPBlock(
                hidden_dim, hidden_dim=hidden_dim, n_layers=1, activation=activation,
                use_bn=use_bn, use_sn=use_sn, use_gamma=use_gamma, use_residual=use_residual)
            self.mlp2 = SetMLPBlock(
                out_dim, hidden_dim=hidden_dim, n_layers=2, activation=activation,
                use_bn=use_bn, use_sn=use_sn, use_gamma=use_gamma, use_residual=use_residual)
            # self.mlp3 = SetMLPBlock(
            #     out_dim, hidden_dim=hidden_dim, n_layers=2, activation=activation,
            #     use_bn=use_bn, use_sn=use_sn, use_gamma=use_gamma, use_residual=use_residual)

    def __call__(self, x, indices):
        h = self.mlp1(x, indices)
        h = self.mlp2(h, indices)
        # h = self.mlp3(h, indices)
        return h


class SetSNBlendNet(chainer.Chain):
    """BlendNet with Spectral Normalization"""

    def __init__(self, num_cat_id=None, out_dim=1, activation=lrelu,
                 dropout_ratio=-1, use_bn=False,
                 numeric_hidden_dim=96, embed_dim=10, bert_hidden_dim=32,
                 image_hidden_dim=96, mlp_hidden_dim=128, mlp_n_layers=6,
                 cat_hidden_dim=32, use_gamma=True, use_residual=False):
        """

        Args:
            num_cat_id:
            out_dim:
            activation:
            dropout_ratio:
            use_bn:
            numeric_hidden_dim: numerical feature
            embed_dim: category feature, this is not used when `cat2num=True` at preprocessing.
            bert_hidden_dim:
            image_hidden_dim:
            mlp_hidden_dim:
            mlp_n_layers:
        """
        super(SetSNBlendNet, self).__init__()
        print('num_cat_id', num_cat_id)  # len(num_cat_id)
        print('spectral normalizaiton ON, use_gamma {}'.format(use_gamma))
        self.use_embed = not isinstance(num_cat_id, (int, float))
        with self.init_scope():
            if num_cat_id is not None:
                if self.use_embed:
                    self.embed_list = chainer.ChainList(
                        *[SNEmbedID(insize, embed_dim) for insize in num_cat_id])
                else:
                    self.l_cat = SNLinear(None, cat_hidden_dim, use_gamma=use_gamma)
            self.l_num = SNLinear(None, numeric_hidden_dim, use_gamma=use_gamma)
            self.l_bert = SNLinear(None, bert_hidden_dim, use_gamma=use_gamma)
            self.l_image = SNLinear(None, image_hidden_dim, use_gamma=use_gamma)
            self.mlp = SetMLP(
                out_dim=out_dim, hidden_dim=mlp_hidden_dim,
                n_layers=mlp_n_layers, activation=activation,
                use_bn=use_bn, use_sn=True, use_gamma=use_gamma,
                use_residual=use_residual)
            # if use_bn:
            #     self.bn1 = L.BatchNormalization()
        self.activation = activation
        self.bert_hidden_dim = bert_hidden_dim
        self.dropout_ratio = dropout_ratio
        self.num_cat_id = num_cat_id
        self.use_bn = use_bn

    def forward(self, x_numeric, x_cat=None, x_bert=None, x_image=None, indices=None):
        h_num = self.l_num(x_numeric)
        h_feat_list = [h_num]
        if x_cat is not None:
            if self.use_embed:
                h_cat_list = [l_cat(x_cat[:, i]) for i, l_cat in enumerate(self.embed_list)]
                h_feat_list.extend(h_cat_list)
            else:
                h_feat_list.append(self.l_cat(x_cat))
        if x_bert is not None:
            # x_bert (bs, num_extract_seq, hdim)
            # --- 1. simply take linear, it will reshape  ---
            # h_bert = self.l_bert(x_bert)
            # --- 2. take linear for each element and sum it. ---
            bs, num_sent, hdim = x_bert.shape
            h_bert = F.reshape(self.l_bert(F.reshape(x_bert, (bs*num_sent, hdim))),
                               (bs, num_sent, self.bert_hidden_dim))
            h_bert = F.sum(h_bert, axis=1)
            # h_bert (bs, bert_hidden_dim)

            # print('x_bert', x_bert.shape, 'h_bert', h_bert.shape)
            h_feat_list.append(h_bert)
        if x_image is not None:
            h_image = self.l_image(x_image)
            h_feat_list.append(h_image)

        h = F.concat(h_feat_list, axis=1)
        # if self.use_bn:
        #     h = self.bn1(h)
        if self.dropout_ratio > 0:
            h = F.dropout(h, ratio=self.dropout_ratio)
        h = self.activation(h)
        h = self.mlp(h, indices)
        return h


class SetBlendNet(chainer.Chain):

    def __init__(self, num_cat_id=None, out_dim=1, activation=lrelu,
                 dropout_ratio=-1, use_bn=False, use_residual=False,
                 numeric_hidden_dim=96, embed_dim=10, bert_hidden_dim=32,
                 image_hidden_dim=96, mlp_hidden_dim=128, mlp_n_layers=6,
                 cat_hidden_dim=32):
        """

        Args:
            num_cat_id:
            out_dim:
            activation:
            dropout_ratio:
            use_bn:
            numeric_hidden_dim: numerical feature
            embed_dim: category feature, this is not used when `cat2num=True` at preprocessing.
            bert_hidden_dim:
            image_hidden_dim:
            mlp_hidden_dim:
            mlp_n_layers:
        """
        super(SetBlendNet, self).__init__()
        print('num_cat_id', num_cat_id)  # len(num_cat_id)
        self.use_embed = not isinstance(num_cat_id, (int, float))
        with self.init_scope():
            if num_cat_id is not None:
                if num_cat_id > 0:
                    self.embed_list = chainer.ChainList(
                        *[L.EmbedID(insize, embed_dim) for insize in num_cat_id])
                else:
                    self.l_cat = L.Linear(None, cat_hidden_dim)
            self.l_num = L.Linear(None, numeric_hidden_dim)
            self.l_bert = L.Linear(None, bert_hidden_dim)
            self.l_image = L.Linear(None, image_hidden_dim)
            self.mlp = SetMLP(
                out_dim=out_dim, hidden_dim=mlp_hidden_dim,
                n_layers=mlp_n_layers, activation=activation,
                use_bn=use_bn, use_sn=False,
                use_residual=use_residual)
            # if use_bn:
            #     self.bn1 = L.BatchNormalization()
        self.activation = activation
        self.bert_hidden_dim = bert_hidden_dim
        self.dropout_ratio = dropout_ratio
        self.num_cat_id = num_cat_id
        self.use_bn = use_bn

    def forward(self, x_numeric, x_cat=None, x_bert=None, x_image=None, indices=None):
        h_num = self.l_num(x_numeric)
        h_feat_list = [h_num]
        if x_cat is not None:
            if self.use_embed:
                h_cat_list = [l_cat(x_cat[:, i]) for i, l_cat in enumerate(self.embed_list)]
                h_feat_list.extend(h_cat_list)
            else:
                h_feat_list.append(self.l_cat(x_cat))
        if x_bert is not None:
            # x_bert (bs, num_extract_seq, hdim)
            # --- 1. simply take linear, it will reshape  ---
            # h_bert = self.l_bert(x_bert)
            # --- 2. take linear for each element and sum it. ---
            bs, num_sent, hdim = x_bert.shape
            h_bert = F.reshape(self.l_bert(F.reshape(x_bert, (bs*num_sent, hdim))),
                               (bs, num_sent, self.bert_hidden_dim))
            h_bert = F.sum(h_bert, axis=1)
            # h_bert (bs, bert_hidden_dim)

            # print('x_bert', x_bert.shape, 'h_bert', h_bert.shape)
            h_feat_list.append(h_bert)
        if x_image is not None:
            h_image = self.l_image(x_image)
            h_feat_list.append(h_image)

        h = F.concat(h_feat_list, axis=1)
        # if self.use_bn:
        #     h = self.bn1(h)
        if self.dropout_ratio > 0:
            h = F.dropout(h, ratio=self.dropout_ratio)
        h = self.activation(h)
        h = self.mlp(h, indices)
        return h


# In[36]:


"""
Compressed Interaction Network used in xDeepFM

Ref:
 - https://arxiv.org/pdf/1803.05170.pdf
 - https://data.gunosy.io/entry/deep-factorization-machines-2018
"""
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import functions

# from src.configs import is_kaggle_kernel
# from src.models.sn.sn_linear import SNLinear


class CINBlock(chainer.Chain):
    def __init__(self, hk_dim, hk_prev_dim=None, h0_dim=None,
                 layer_type='linear', use_sn=False, use_gamma=True):
        """

        Args:
            num_field (int): `D` in paper. number of
        """
        super(CINBlock, self).__init__()
        if hk_prev_dim is None or h0_dim is None:
            in_size = None
        else:
            in_size = hk_prev_dim * h0_dim
        with self.init_scope():
            if layer_type == 'linear':
                if use_sn:
                    self.l = SNLinear(in_size, hk_dim, use_gamma=use_gamma)
                else:
                    self.l = L.Linear(in_size, hk_dim)
            elif layer_type == 'gru':
                self.l = L.GRU(in_size, hk_dim)
            else:
                raise ValueError("[ERROR] Unexpected value layer_type={}".format(layer_type))
        self.hk_dim = hk_dim
        self.hk_prev_dim = hk_prev_dim
        self.h0_dim = h0_dim

    def forward(self, hk_prev, h0):
        """

        Args:
            hk_prev: (bs, hdim=D, hk_prev_dim)
            h0: (bs, hdim=D, h0_dim=m)

        Returns:
            hk: (bs, hdim=D, hk_dim)
        """
        h_prod = hk_prev[:, :, :, None] * h0[:, :, None, :]
        bs, D, hk_prev_dim, h0_dim = h_prod.shape
        h_prod = F.reshape(h_prod, (bs * D, hk_prev_dim * h0_dim))
        hk = self.l(h_prod)
        bs_D, hk_dim = hk.shape
        hk = F.reshape(hk, (bs, D, hk_dim))
        return hk

    def reset_state(self):
        if hasattr(self.l, 'reset_state'):
            self.l.reset_state()


class CIN(chainer.Chain):
    def __init__(self, out_dim, hk_dims, use_sn=False, use_gamma=True,
                 dropout_ratio=0):
        super(CIN, self).__init__()
        # hk_dims = [m, ..., hk]
        pooled_dim = 0
        for hk in hk_dims[1:]:
            pooled_dim += hk
        with self.init_scope():
            self.cin_layers = chainer.ChainList(
                *[CINBlock(hk_dims[i], hk_dims[i-1], hk_dims[0],
                           use_sn=use_sn, use_gamma=use_gamma)
                  for i in range(1, len(hk_dims))])
            self.fc = L.Linear(pooled_dim, out_dim)
        self.out_dim = out_dim
        self.hk_dims = hk_dims
        self.dropout_ratio = dropout_ratio

    def forward(self, x):
        # x (bs, hdim=D, num_field=m)
        h = x
        out_list = []
        for cin_layer in self.cin_layers:
            h = cin_layer(h, x)
            # sum pooling
            # print('[DEBUG]', h.shape)
            out = F.sum(h, axis=1)
            out_list.append(out)
        out = F.concat(out_list, axis=1)
        if self.dropout_ratio > 0:
            out = F.dropout(out, ratio=self.dropout_ratio)
        out = self.fc(out)
        return out


class CINTying(chainer.Chain):
    def __init__(self, out_dim, h0_dim, n_layers, layer_type='linear',
                 use_sn=False, use_gamma=True, activation=functions.identity,
                 dropout_ratio=0):
        super(CINTying, self).__init__()
        print('CINTying: out_dim {}, h0_dim {}, n_layers {} layer_type {}'
              .format(out_dim, h0_dim, n_layers, layer_type))
        # hk_dims = [m, ..., hk]
        pooled_dim = h0_dim * n_layers
        with self.init_scope():
            self.cin_layer = CINBlock(
                h0_dim, h0_dim, h0_dim, layer_type=layer_type,
                use_sn=use_sn, use_gamma=use_gamma)
            self.fc = L.Linear(pooled_dim, out_dim)
            # self.fc = MLP(out_dim, hidden_dim=16, n_layers=2)
        self.out_dim = out_dim
        self.h0_dim = h0_dim
        self.n_layers = n_layers
        self.activation = activation
        self.dropout_ratio = dropout_ratio

    def forward(self, x):
        # x (bs, hdim=D, num_field=m)
        self.reset_state()
        h = x
        out_list = []
        for i in range(self.n_layers):
            h = self.cin_layer(h, x)
            h = self.activation(h)
            # sum pooling
            # print('[DEBUG]', h.shape)
            out = F.sum(h, axis=1)
            out_list.append(out)
        out = F.concat(out_list, axis=1)
        if self.dropout_ratio > 0:
            out = F.dropout(out, ratio=self.dropout_ratio)
        out = self.fc(out)

        return out

    def reset_state(self):
        self.cin_layer.reset_state()


# In[37]:


import chainer

import chainer.functions as F
import chainer.links as L
from chainer import functions

def lrelu(x):
    return functions.leaky_relu(x, slope=0.05)


class BlendNetXDeepFM(chainer.Chain):

    def __init__(self, num_cat_id=None, out_dim=1, activation=lrelu,
                 dropout_ratio=-1, use_bn=False, use_residual=False,
                 numeric_hidden_dim=96, embed_dim=10, bert_hidden_dim=32,
                 image_hidden_dim=96, mlp_hidden_dim=128, mlp_n_layers=6,
                 cat_hidden_dim=32, use_sn=False, use_gamma=True,
                 weight_tying=False):
        """

        Args:
            num_cat_id:
            out_dim:
            activation:
            dropout_ratio:
            use_bn:
            numeric_hidden_dim: numerical feature
            embed_dim: category feature, this is not used when `cat2num=True` at preprocessing.
            bert_hidden_dim:
            image_hidden_dim:
            mlp_hidden_dim:
            mlp_n_layers:
        """
        super(BlendNetXDeepFM, self).__init__()
        print('num_cat_id', num_cat_id)  # len(num_cat_id)
        projection = False
        self.projection = projection
        print('projection', projection)
        self.use_embed = not isinstance(num_cat_id, (int, float))

        num_numeric_feat = 0
        num_image_feat = 0
        h0_dim = len(num_cat_id) + num_numeric_feat + num_image_feat  # num & image
        self.num_numeric_feat = num_numeric_feat
        self.num_image_feat = num_image_feat
        print('h0_dim', h0_dim)
        with self.init_scope():
            # if num_cat_id is not None:
            if self.use_embed:
                self.embed_list = chainer.ChainList(
                    *[L.EmbedID(insize, embed_dim) for insize in num_cat_id])
            else:
                if use_sn:
                    self.l_cat = SNLinear(None, cat_hidden_dim, use_gamma=use_gamma)
                else:
                    self.l_cat = L.Linear(None, cat_hidden_dim)

            if use_sn:
                self.l_num = SNLinear(None, numeric_hidden_dim, use_gamma=use_gamma)
                self.l_bert = SNLinear(None, bert_hidden_dim, use_gamma=use_gamma)
                self.l_image = SNLinear(None, image_hidden_dim, use_gamma=use_gamma)
                self.embed_num = SNLinear(None, embed_dim * num_numeric_feat, use_gamma=use_gamma)
                self.embed_bert = SNLinear(None, embed_dim, use_gamma=use_gamma)
                self.embed_image = SNLinear(None, embed_dim * num_image_feat, use_gamma=use_gamma)
            else:
                self.l_num = L.Linear(None, numeric_hidden_dim)
                self.l_bert = L.Linear(None, bert_hidden_dim)
                self.l_image = L.Linear(None, image_hidden_dim)
                self.embed_num = L.Linear(None, embed_dim * num_numeric_feat)
                self.embed_bert = L.Linear(None, embed_dim)
                self.embed_image = L.Linear(None, embed_dim * num_image_feat)

            if weight_tying:
                self.cin = CINTying(out_dim, h0_dim, n_layers=2, use_sn=True,
                                    dropout_ratio=dropout_ratio)  #
            else:
                self.cin = CIN(out_dim, hk_dims=[h0_dim, h0_dim], use_sn=True,
                               dropout_ratio=dropout_ratio)  # [h0_dim, 10, 10]
            if projection:
                self.mlp = ProjectionMLP(
                    out_dim=out_dim, hidden_dim=mlp_hidden_dim,
                    n_layers=mlp_n_layers, activation=activation,
                    use_bn=use_bn, use_residual=use_residual)
            else:
                self.mlp = MLP(
                    out_dim=out_dim, hidden_dim=mlp_hidden_dim,
                    n_layers=mlp_n_layers, activation=activation,
                    use_bn=use_bn, use_residual=use_residual)
            # if use_bn:
            #     self.bn1 = L.BatchNormalization()
        self.activation = activation
        self.bert_hidden_dim = bert_hidden_dim
        self.dropout_ratio = dropout_ratio
        self.num_cat_id = num_cat_id
        self.use_bn = use_bn
        self.embed_dim = embed_dim

    def forward(self, x_numeric, x_cat=None, x_bert=None, x_image=None):
        bs = x_numeric.shape[0]
        h_num = self.l_num(x_numeric)
        h_num_embed = self.embed_num(x_numeric)
        h_feat_list = [h_num]
        if x_cat is not None:
            if self.use_embed:
                h_cat_list = [l_cat(x_cat[:, i]) for i, l_cat in enumerate(self.embed_list)]
                h_feat_list.extend(h_cat_list)
            else:
                h_feat_list.append(self.l_cat(x_cat))
                raise NotImplementedError
        # h_cat_list.append(h_num_embed)
        h_cat_var = F.stack(h_cat_list, axis=2)

        if x_bert is not None:
            # x_bert (bs, num_extract_seq, hdim)
            # --- 1. simply take linear, it will reshape  ---
            # h_bert = self.l_bert(x_bert)
            # --- 2. take linear for each element and sum it. ---
            bs, num_sent, hdim = x_bert.shape
            h_bert = F.reshape(self.l_bert(F.reshape(x_bert, (bs*num_sent, hdim))),
                               (bs, num_sent, self.bert_hidden_dim))
            h_bert = F.sum(h_bert, axis=1)
            # h_bert (bs, bert_hidden_dim)

            # print('x_bert', x_bert.shape, 'h_bert', h_bert.shape)
            h_feat_list.append(h_bert)

            h_bert_embed = F.reshape(self.embed_bert(F.reshape(x_bert, (bs*num_sent, hdim))),
                                     (bs, num_sent, self.bert_hidden_dim))
            h_bert_embed = F.sum(h_bert_embed, axis=1)
            # h_cat_list.append(h_bert_embed)
        if x_image is not None:
            h_image = self.l_image(x_image)
            h_feat_list.append(h_image)
            h_image_embed = self.embed_image(x_image)
            # h_cat_list.append(h_image_embed)

        # --- CIN part ---
        h_num_embed = F.reshape(h_num_embed, (bs, self.embed_dim, self.num_numeric_feat))
        h_image_embed = F.reshape(h_image_embed, (bs, self.embed_dim, self.num_image_feat))
        h_embed = F.concat([h_num_embed, h_cat_var, h_image_embed], axis=2)
        h_embed = self.cin(h_embed)

        # --- DNN part ---
        h = F.concat(h_feat_list, axis=1)
        # if self.use_bn:
        #     h = self.bn1(h)
        if self.dropout_ratio > 0:
            h = F.dropout(h, ratio=self.dropout_ratio)
        h = self.activation(h)
        if self.projection:
            h = self.mlp(h, x_cat[:, 0].astype(self.xp.int32))
        else:
            h = self.mlp(h)

        # FC
        # h = self.fc(F.concat([h, h_embed], axis=1))
        h = h + h_embed
        return h


# In[ ]:





# In[ ]:


# rescuer id mean dataset


# In[ ]:


import chainer
import numpy as np

import sys
import os

# from chainer_chemistry.datasets import NumpyTupleDataset

# sys.path.append(os.pardir)
# sys.path.append(os.path.join(os.pardir, os.pardir))
# from src.preprocessing import prepare_df


def preprocessing_target_mean(train):
    train = train.reset_index()
    train['AdoptionSpeedMean'] = train.groupby('RescuerID')[
        'AdoptionSpeed'].transform(np.mean)
    target_mean = train['AdoptionSpeedMean'].values.astype(np.float32)[:, None]
    rescuer_id_list = []
    rescuer_id_index_list = []
    for rescuer_id, df in train.groupby('RescuerID'):
        rescuer_id_list.append(rescuer_id)
        rescuer_id_index_list.append(df.index.values)
    return target_mean, rescuer_id_list, rescuer_id_index_list


class RescuerIDMeanDataset(chainer.dataset.DatasetMixin):
    def __init__(self, train, dataset, mode='train'):
        """
        Args:
            train: train DataFrame
            dataset: dataset for this
        """
        print('RescuerIDMeanDataset')
        target_mean, rescuer_id_list, rescuer_id_index_list = preprocessing_target_mean(train)
        print('target_mean', target_mean.shape, 'rescuer_id_index_list', len(rescuer_id_index_list))
        self.rescuer_id = train['RescuerID'].values
        self.rescuer_id_list = rescuer_id_list  # list of str, unique rescuer_id list
        self.rescuer_id_index_list = rescuer_id_index_list  # list of list, each element is rescuer's index list
        self.target_mean = target_mean
        self.dataset = dataset
        assert len(target_mean) == len(dataset)
        self.max_num_sample = 10
        self.mode = mode

    def __len__(self):
        """return length of this dataset"""
        if self.mode == 'train':
            return len(self.target_mean)
        else:
            return len(self.rescuer_id_list)

    def get_example(self, i):
        """Return i-th data"""
        if self.mode == 'train':
            rescuer_id = self.rescuer_id[i]
        else:
            rescuer_id = self.rescuer_id_list[i]
        index_list = self.rescuer_id_index_list[self.rescuer_id_list.index(rescuer_id)]
        if self.mode == 'train':
            # sample datasets from `index_list`.
            size = min(len(index_list), self.max_num_sample)
            random_indices = np.random.choice(len(index_list), size, replace=False)
            target_indices = index_list[random_indices]
            assert self.target_mean[i] == self.target_mean[index_list[0]]
        else:
            target_indices = index_list
        t = self.target_mean[index_list[0]]
        # print('[DEBUG] rescuer_id', rescuer_id, 'target_indices', target_indices)
        return self.dataset[target_indices], t


# if __name__ == '__main__':
#     debug = False
#     train, test, breeds, colors, states = prepare_df(debug)
#     # target_mean, rescuer_id_list, rescuer_id_index_list = preprocessing_target_mean(train)
#     dummy_dataset = NumpyTupleDataset(train[['Age', 'MaturitySize']].values, train[['AdoptionSpeed']].values)
#     d = RescuerIDMeanDataset(train, dummy_dataset)
#     data0, t0 = d[0]
#     import IPython; IPython.embed()


# util function to tune optimizer's learning rate during NN training.

# In[ ]:


# myutils.schedule_value


# In[ ]:


import os
import numpy

import chainer
from chainer.training import Trainer


def schedule_optimizer_value(epoch_list, value_list, optimizer_name='main',
                             attr_name='__auto'):
    """Set optimizer's hyperparameter according to value_list, scheduled on epoch_list. 
    
    Example usage:
    trainer.extend(schedule_optimizer_value([2, 4, 7], [0.008, 0.006, 0.002]))
    
    or 
    trainer.extend(schedule_optimizer_value(2, 0.008))

    Args:
        epoch_list (list, int or float): list of int. epoch to invoke this extension. 
        value_list (list, int or float): list of float. value to be set.
        optimizer_name: optimizer's name on trainer
        attr_name: attr name of optimizer to change value.
           if '__auto' is set, it will automatically infer learning rate attr name. 

    Returns (callable): extension function

    """
    if isinstance(epoch_list, list):
        if len(epoch_list) != len(value_list):
            raise ValueError('epoch_list length {} and value_list length {} '
                             'must be same!'
                             .format(len(epoch_list), len(value_list)))
    else:
        assert isinstance(epoch_list, float) or isinstance(epoch_list, int)
        assert isinstance(value_list, float) or isinstance(value_list, int)
        epoch_list = [epoch_list, ]
        value_list = [value_list, ]


    trigger = chainer.training.triggers.ManualScheduleTrigger(epoch_list,
                                                              'epoch')
    count = 0
    _attr_name = attr_name

    @chainer.training.extension.make_extension(trigger=trigger)
    def set_value(trainer: Trainer):
        nonlocal count, _attr_name
        value = value_list[count]
        optimizer = trainer.updater.get_optimizer(optimizer_name)

        # Infer attr name
        if count == 0 and _attr_name == '__auto':
            if isinstance(optimizer, chainer.optimizers.Adam):
                _attr_name = 'alpha'
            else:
                _attr_name = 'lr'

        print('updating {} to {}'.format(_attr_name, value))
        setattr(optimizer, _attr_name, value)
        count += 1

    return set_value


def schedule_target_value(epoch_list, value_list, target, attr_name):
    """Set optimizer's hyperparameter according to value_list, scheduled on epoch_list. 

    target is None -> use main optimizer

    Example usage:
    trainer.extend(schedule_target_value([2, 4, 7], [0.008, 0.006, 0.002], iterator, 'batch_size'))
    """
    if isinstance(epoch_list, list):
        if not isinstance(value_list, list):
            assert isinstance(value_list, float) or isinstance(value_list, int)
            value_list = [value_list, ]
        if len(epoch_list) != len(value_list):
            raise ValueError('epoch_list length {} and value_list length {} '
                             'must be same!'
                             .format(len(epoch_list), len(value_list)))
    else:
        assert isinstance(epoch_list, float) or isinstance(epoch_list, int)
        assert isinstance(value_list, float) or isinstance(value_list, int)
        epoch_list = [epoch_list, ]
        value_list = [value_list, ]

    trigger = chainer.training.triggers.ManualScheduleTrigger(epoch_list,
                                                              'epoch')
    count = 0

    @chainer.training.extension.make_extension(trigger=trigger)
    def set_value(trainer: Trainer):
        nonlocal count
        value = value_list[count]

        print('updating {} to {}'.format(attr_name, value))
        setattr(target, attr_name, value)
        count += 1

    return set_value


# ### optimized rounder
# 
# I added some implementation from [public kernel](https://www.kaggle.com/fiancheto/petfinder-simple-lgbm-baseline-lb-0-399) to align prediction data to specified histogram (Final submission is aligned with `train` dataset label histogram)

# In[40]:


# optimized rounder


# In[41]:


"""
ref: https://www.kaggle.com/fiancheto/petfinder-simple-lgbm-baseline-lb-0-399
"""
from collections import Counter
from functools import partial

import numpy as np
import numba as nb
import scipy as sp
from scipy.optimize import differential_evolution
from sklearn.metrics import cohen_kappa_score


class OptimizedRounder(object):
    def __init__(self, num_class=5, method='differential_evolution'):
        """

        Args:
            num_class:
            method: 'nelder-mead' or 'differential_evolution'
        """
        self.coef_ = 0
        self.num_class = num_class
        self.method = method

    def _kappa_loss(self, coef, X, y):
        # X_p = calc_threshold_label(X, coef)
        X_p = calc_threshold_label_numba(X, coef.astype(np.float32))
        # print('X', X.shape, X.dtype, 'coef', coef.shape, coef.dtype)
        # print('X_p', X_p.shape, X_p.dtype)
        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        # print('coef', coef, 'y', y, 'X_p', X_p, 'll', ll)
        # ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        if X.ndim > 1:
            X = X.ravel()
        if y.ndim > 1:
            y = y.ravel()
        X = X.astype(np.float32)
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        if self.method == 'nelder-mead':
            initial_coef = self.get_initial_coef()
            print('fit... initial_coef', initial_coef)
            self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
            # print('coef_', self.coef_)
            # print('coef_ x', self.coef_['x'])
        elif self.method == 'differential_evolution':
            bounds = [(0., self.num_class-1) for _ in range(self.num_class - 1)]
            result = differential_evolution(loss_partial, bounds)
            self.coef_ = result
        else:
            raise ValueError("[ERROR] Unexpected value method={}".format(self.method))

    def get_initial_coef(self):
        return np.arange(self.num_class - 1) + 0.5

    def predict(self, X, coef=None):
        if X.ndim > 1:
            X = X.ravel()
        if coef is None:
            coef = self.get_initial_coef()
            print('[WARNING] set initial coef {}'.format(coef))
        # X_p = calc_threshold_label(X, np.asarray(coef, dtype=np.float32))
        X_p = calc_threshold_label_numba(X.astype(np.float32),
                                         np.asarray(coef, dtype=np.float32))
        return X_p

    def calc_histogram_coef(self, X, y_count):
        """

        Args:
            X: target predicted values to calculate threshold.
            y_count (Counter or list): list of each class occurrence count.

        Returns:
            coef:
        """
        size_array = np.zeros((self.num_class,), dtype=np.int32)
        for i in range(self.num_class):
            # Counter `y_count[i]` returns i-th count.
            size_array[i] = y_count[i]
        size_array = size_array.cumsum()
        hist_array = size_array / size_array[-1]
        x_size_array = (hist_array * len(X)).astype(np.int32)
        coef = np.sort(X)[x_size_array[:-1]]
        return coef

    def fit_and_predict_by_histgram(self, X, y):
        """QWK is not used.
        simply calculate coefficients to match same histgram with `y`.
        Args:
            X: predicted label
            y: ground truth label
        Returns:
        """
        if X.ndim > 1:
            X = X.ravel()
        if y.ndim > 1:
            y = y.ravel()
        y_count = Counter(y)
        print('train counter', y_count)
        coef = self.calc_histogram_coef(X, y)
        self.coef_ = {'x': coef}
        print('coefficient', self.coef_)
        return self.predict(X, coef=self.coefficients())

    def coefficients(self):
        return self.coef_['x']


def calc_threshold_label(X, coef):
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


@nb.jit(nb.int32[:](nb.float32[:], nb.float32[:]), nopython=True, nogil=True)
def calc_threshold_label_numba(X, coef):
    X_p = np.empty(X.shape, dtype=np.int32)
    for i, pred in enumerate(X):
        if pred < coef[0]:
            X_p[i] = 0
        elif pred < coef[1]:
            X_p[i] = 1
        elif pred < coef[2]:
            X_p[i] = 2
        elif pred < coef[3]:
            X_p[i] = 3
        else:
            X_p[i] = 4
    return X_p



# calculate QWK score during NN training, as extension

# In[42]:


import numpy

from chainer.dataset import convert
from sklearn.metrics import cohen_kappa_score

#from chainer_chemistry.training.extensions.batch_evaluator import BatchEvaluator  # NOQA


def _to_list(a):
    """convert value `a` to list

    Args:
        a: value to be convert to `list`

    Returns (list):

    """
    if isinstance(a, (int, float)):
        return [a, ]
    else:
        # expected to be list or some iterable class
        return a


class QuadraticWeightedKappaEvaluator(BatchEvaluator):

    """Evaluator which calculates quadratic weighted kappa

    Note that this Evaluator is only applicable to binary classification task.

    Args:
        iterator: Dataset iterator for the dataset to calculate ROC AUC score.
            It can also be a dictionary of iterators. If this is just an
            iterator, the iterator is registered by the name ``'main'``.
        target: Link object or a dictionary of links to evaluate. If this is
            just a link object, the link is registered by the name ``'main'``.
        converter: Converter function to build input arrays and true label.
            :func:`~chainer.dataset.concat_examples` is used by default.
            It is expected to return input arrays of the form
            `[x_0, ..., x_n, t]`, where `x_0, ..., x_n` are the inputs to
            the evaluation function and `t` is the true label.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        eval_hook: Function to prepare for each evaluation process. It is
            called at the beginning of the evaluation. The evaluator extension
            object is passed at each call.
        eval_func: Evaluation function called at each iteration. The target
            link to evaluate as a callable is used by default.
        name (str): name of this extension. When `name` is None,
            `default_name='validation'` which is defined in super class
            `Evaluator` is used as extension name. This name affects to the
            reported key name.
        pos_labels (int or list): labels of the positive class, other classes
            are considered as negative.
        ignore_labels (int or list or None): labels to be ignored.
            `None` is used to not ignore all labels.
        raise_value_error (bool): If `False`, `ValueError` caused by
            `roc_auc_score` calculation is suppressed and ignored with a
            warning message.
        logger:

    Attributes:
        converter: Converter function.
        device: Device to which the training data is sent.
        eval_hook: Function to prepare for each evaluation process.
        eval_func: Evaluation function called at each iteration.
        pos_labels (list): labels of the positive class
        ignore_labels (list): labels to be ignored.

    """

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, name=None,
                 pos_labels=1, ignore_labels=None, raise_value_error=True,
                 logger=None):
        metrics_fun = {'qwk': self.quadratic_weighted_kappa}
        super(QuadraticWeightedKappaEvaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func, metrics_fun=metrics_fun,
            name=name, logger=logger)

        # self.pos_labels = _to_list(pos_labels)
        # self.ignore_labels = _to_list(ignore_labels)
        # self.raise_value_error = raise_value_error

    def quadratic_weighted_kappa(self, y_total, t_total):
        # --- ignore labels if specified ---
        # if self.ignore_labels:
        #     valid_ind = numpy.in1d(t_total, self.ignore_labels, invert=True)
        #     y_total = y_total[valid_ind]
        #     t_total = t_total[valid_ind]

        y_total = numpy.round(y_total)
        score = cohen_kappa_score(y_total, t_total, weights='quadratic')
        return score


# BlendConverter for data augmentation during NN training.
# 
# It includes 
# 
# - permutation augmentation (used in final submission)
# - mixup data augmentation (not used in final submission)
# - gaussian noise data augmentation (not used in final submission)

# In[43]:


# --- blend converter ---


# In[44]:


import numpy
import numba as nb

import chainer
from chainer.dataset import concat_examples, to_device


@nb.jit(nb.void(nb.float32[:, :], nb.int64), nopython=True, nogil=True)
def permute_cols_2d(x, num_cols):
    """Permute cols of 2d-array `x` inplace."""
    bs, ndim = x.shape
    col_indices = numpy.random.choice(ndim, num_cols, replace=False)
    for col_index in col_indices:
        perm = numpy.random.permutation(bs)
        x[:, int(col_index)] = x[perm, int(col_index)]


@nb.jit(nb.void(nb.int32[:, :], nb.int64), nopython=True, nogil=True)
def permute_cols_2d_int(x, num_cols):
    """Permute cols of 2d-array `x` inplace."""
    bs, ndim = x.shape
    col_indices = numpy.random.choice(ndim, num_cols, replace=False)
    for col_index in col_indices:
        perm = numpy.random.permutation(bs)
        x[:, int(col_index)] = x[perm, int(col_index)]


@nb.jit(nb.void(nb.float32[:, :, :], nb.int64), nopython=True, nogil=True)
def permute_cols_3d(x, num_cols):
    """Permute cols of 3d-array `x` inplace."""
    bs, n_image, ndim = x.shape
    col_indices = numpy.random.choice(ndim, num_cols, replace=False)
    for col_index in col_indices:
        perm = numpy.random.permutation(bs)
        x[:, :, int(col_index)] = x[perm, :, int(col_index)]


@nb.jit(nb.float32[:, :](nb.float32[:, :], nb.int64[:], nb.int64[:], nb.float32[:]),
        nopython=True, nogil=True)
def mixup_2d(x, ind1, ind2, lam):
    """Mixup `x` between ind1 and ind2, with weight lam. return new `x`"""
    # num_mixup = lam.shape[0]
    # lam2 = numpy.ascontiguousarray(lam).reshape((num_mixup, 1))
    lam2 = numpy.expand_dims(lam, axis=1)
    x_mix = (lam2 * x[ind1] + (1. - lam2) * x[ind2]).astype(numpy.float32)
    return numpy.concatenate((x, x_mix), axis=0)


@nb.jit(nb.float32[:, :, :](nb.float32[:, :, :], nb.int64[:], nb.int64[:], nb.float32[:]),
        nopython=True, nogil=True)
def mixup_3d(x, ind1, ind2, lam):
    """Mixup `x` between ind1 and ind2, with weight lam. return new `x`"""
    # num_mixup = lam.shape[0]
    # lam3 = numpy.ascontiguousarray(lam).reshape((num_mixup, 1, 1))
    lam3 = numpy.expand_dims(numpy.expand_dims(lam, axis=1), axis=1)
    x_mix = (lam3 * x[ind1] + (1 - lam3) * x[ind2]).astype(numpy.float32)
    return numpy.concatenate((x, x_mix), axis=0)


@nb.jit(nb.void(nb.float32[:, :], nb.float32[:], nb.float64), nopython=True, nogil=True)
def add_noise_2d(x, std, noise_ratio):
    """Add Gaussian noise with scale `std * noise_ratio`, Value `x` is overwritten inplace"""
    noise = numpy.random.normal(0, 1, size=x.shape).astype(numpy.float32)
    std = numpy.expand_dims(std, axis=0)
    x[:] = x + noise * std * noise_ratio


@nb.jit(nb.void(nb.float32[:, :, :], nb.float32[:], nb.float64), nopython=True, nogil=True)
def add_noise_3d(x, std, noise_ratio):
    """Add Gaussian noise with scale `std * noise_ratio`, Value `x` is overwritten inplace"""
    noise = numpy.random.normal(0, 1, size=x.shape).astype(numpy.float32)
    std = numpy.expand_dims(std, axis=0)
    std = numpy.expand_dims(std, axis=0)
    x[:] = x + noise * std * noise_ratio


class BlendConverter(object):

    def __init__(self, use_cat=True, use_bert=True, use_image=True,
                 augmentation=True, permute_col_ratio_list=0.10, num_cols_choice=False,
                 mixup_ratio=0., alpha=10., std_list=None, noise_ratio_list=None,
                 mode='normal', use_embed=False):
        self.use_cat = use_cat
        self.use_bert = use_bert
        self.use_image = use_image
        self.mode = mode
        self.use_embed = use_embed

        # --- Permute augmentation ---
        self.augmentation = augmentation
        if isinstance(permute_col_ratio_list, float):
            permute_col_ratio_list = [permute_col_ratio_list for _ in range(4)]
        self.permute_col_ratio_list = permute_col_ratio_list
        # If True, `num_cols` is calculated as choice, if `False` always `permute_col_ratio` is used.
        self.num_cols_choice = num_cols_choice

        # --- Mixup ---
        # ref: http://wazalabo.com/mixup_1.html, https://qiita.com/yu4u/items/70aa007346ec73b7ff05
        self.mixup_ratio = mixup_ratio
        self.alpha = alpha

        # --- Gaussian Noise ---
        self.std_list = std_list
        self.noise_ratio_list = noise_ratio_list

        print('BlendConverter: augmentation {} ratio {}, choice {}, mixup_ratio {} alpha {}, '
              'noise_ratio_list {}, use_embed {}'
              .format(augmentation, self.permute_col_ratio_list, self.num_cols_choice,
                      self.mixup_ratio, self.alpha, noise_ratio_list, self.use_embed))

        self.extract_inputs = False
        self._count = 0

    def __call__(self, batch, device=None):
        # concat in CPU at first
        # batch_list: [x_num, x_cat, x_bert, x_image, target]
        if self.mode == 'mean':
            from itertools import chain
            data_list = [b[0] for b in batch]
            data_len_list = [len(d) for d in data_list]
            data_flatten_list = list(chain.from_iterable(data_list))
            batch_list = list(concat_examples(data_flatten_list, device=-1))

            indices = numpy.zeros((len(data_flatten_list),), dtype=numpy.int32)
            index = 0
            val = 0
            for k in data_len_list:
                indices[index:index+k] = val
                index += k
                val += 1
            assert index == len(indices)
            # batch_list.append(indices)
            target_mean = numpy.array([b[1] for b in batch])
        else:
            batch_list = list(concat_examples(batch, device=-1))
        if not self.use_cat:
            batch_list.insert(1, None)
        if not self.use_bert:
            batch_list.insert(2, None)
        if not self.use_image:
            batch_list.insert(3, None)

        if self.augmentation and chainer.config.train:
            # --- Gaussian noise augmentation ---
            if self.noise_ratio_list is not None:
                for i, x in enumerate(batch_list[:4]):
                    if self.use_embed and i == 1:
                        continue
                    if x is None or self.noise_ratio_list[i] <= 0.:
                        continue
                    elif x.ndim == 2:
                        # noise = numpy.random.normal(0, 1, size=x.shape).astype(numpy.float32)
                        # batch_list[i] = x + noise * self.std_list[i][None, :] * self.noise_ratio_list[i]
                        add_noise_2d(x, self.std_list[i], self.noise_ratio_list[i])
                    elif x.ndim == 3:
                        # noise = numpy.random.normal(0, 1, size=x.shape).astype(numpy.float32)
                        # batch_list[i] = x + noise * self.std_list[i][None, None, :] * self.noise_ratio_list[i]
                        add_noise_3d(x, self.std_list[i], self.noise_ratio_list[i])
                    else:
                        raise ValueError("[ERROR] Unexpected value x.shape={}"
                                         .format(x.shape))

            # --- mixup ---
            if self.mixup_ratio > 0:
                bs = batch_list[0].shape[0]
                num_mixup = int(bs * self.mixup_ratio)
                ind1 = numpy.random.choice(bs, num_mixup)
                ind2 = numpy.random.choice(bs, num_mixup)
                lam = numpy.random.beta(self.alpha, self.alpha, num_mixup).astype(numpy.float32)
                if self._count <= 1:
                    print('num_mixup', num_mixup, 'alpha', self.alpha,
                          'ind1', ind1.shape, 'ind2', ind2.shape)
                    print('lam', lam.shape, lam[:10])
                for i, x in enumerate(batch_list):
                    # Need to mixup labels as well...
                    if self.use_embed and i == 1:
                        continue
                    if x is None:
                        continue
                    elif x.ndim == 2:
                        if x.dtype is numpy.int32:
                            raise NotImplementedError('x.dtype {} for {}-th feature'.format(x.dtype, i))
                        # lam2 = lam.reshape((num_mixup, 1))
                        # x_mix = lam2 * x[ind1] + (1-lam2) * x[ind2]
                        # batch_list[i] = numpy.concatenate([x, x_mix], axis=0)
                        batch_list[i] = mixup_2d(x, ind1, ind2, lam)
                    elif x.ndim == 3:
                        # lam3 = lam.reshape((num_mixup, 1, 1))
                        # x_mix = lam3 * x[ind1] + (1-lam3) * x[ind2]
                        # batch_list[i] = numpy.concatenate([x, x_mix], axis=0)
                        batch_list[i] = mixup_3d(x, ind1, ind2, lam)
                    else:
                        raise ValueError("[ERROR] Unexpected value x.shape={}"
                                         .format(x.shape))

            # --- permutation augmentation ---
            # x_num, x_cat, x_bert, x_image = batch_list[:4]
            for i, x in enumerate(batch_list[:4]):
                if x is None:
                    continue
                elif x.ndim == 2:
                    bs, ndim = x.shape
                    # num_cols = 1
                    num_cols = int(ndim * self.permute_col_ratio_list[i])
                    if self.num_cols_choice and num_cols > 0:
                        num_cols = numpy.random.choice(num_cols)
                    if self.use_embed and i == 1:
                        permute_cols_2d_int(x, num_cols)
                    else:
                        permute_cols_2d(x, num_cols)
                    if self._count <= 1:
                        print('i', i, 'num_cols', num_cols)
                elif x.ndim == 3:
                    # n_image for image or n_sentence for bert feature
                    bs, n_image, ndim = x.shape
                    # num_cols = 1
                    num_cols = int(ndim * self.permute_col_ratio_list[i])
                    if self.num_cols_choice and num_cols > 0:
                        num_cols = numpy.random.choice(num_cols)
                    permute_cols_3d(x, num_cols)
                    if self._count <= 1:
                        print('i', i, 'num_cols', num_cols)
                else:
                    raise ValueError("[ERROR] Unexpected value x.shape={}"
                                     .format(x.shape))
        else:
            # print('skip augmentation...')
            pass

        # send to device...
        batch_list = [to_device(device, x) for x in batch_list]

        self._count += 1
        if self.mode == 'mean':
            indices = to_device(device, indices)
            if self.extract_inputs:
                # During inference, do not return label
                return tuple(batch_list[:-1]) + (indices, )
            else:
                # During training, return label as well
                target_mean = to_device(device, target_mean)
                return tuple(batch_list) + (indices, target_mean)
        elif self.mode == 'normal':
            if self.extract_inputs:
                # During inference, do not return label
                return tuple(batch_list[:-1])
            else:
                # During training, return label as well
                return tuple(batch_list)


# In[45]:


# --- fasttext ---


# In[ ]:


import os

import numpy
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm


def extract_line(line, i, vec_array):
    line = line.rstrip()
    # print('line:', line)
    tmp = line.split(' ')
    vocab = tmp[0]
    vec_array[i:i + 1] = numpy.array(tmp[1:], dtype=numpy.float32)
    return vocab


def construct_fasttext_vocab_list(filepath):
    """

    Args:
        filepath (str): fast text embedding file path

    Returns:
        vocab_list (list): length is `num_vocab` in file
        vec_array (numpy.ndarray): (num_vocab, hdim)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        a = f.readline()
        num_vocab, hdim = a.split(' ')
        num_vocab, hdim = int(num_vocab), int(hdim)
        print('num_vocab', num_vocab, 'hdim', hdim)
        all_lines = f.readlines()
        # print('all_lines:', all_lines)
        print('all_lines extracted on RAM')

    vec_array = numpy.empty((num_vocab, hdim), dtype=numpy.float32)
    # from joblib import Parallel, delayed
    # n_jobs = 16
    # results = Parallel(n_jobs, backend='threading', verbose=1)(
    #     delayed(extract_line)(line, i, vec_array) for i, line in enumerate(all_lines))
    vocab_list = [extract_line(line, i, vec_array) for i, line in tqdm(enumerate(all_lines), total=len(all_lines))]
    return vocab_list, vec_array


def calc_fasttext_feature(corpus, vocab_list, vec_array,
                          out_dim=None, col_name=None, method='tfidf',
                          source='fasttext', contraction_method='svd'):
    h_sent = None
    cache_filepath = None
    if col_name is not None:
        cache_filepath = f'./cache/{source}_{col_name}_{method}_{len(corpus)}.npz'
        if os.path.exists(cache_filepath):
            h_sent = load_npz(cache_filepath)
    if h_sent is None:
        if method == 'count':
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(corpus)
            print('X', type(X), X.shape)  # sparse matrix
            num_sent, num_feat = X.shape
            feat_names = vectorizer.get_feature_names()
            print('feat_names', len(feat_names))
            assert num_feat == len(feat_names)
        elif method == 'tfidf':
            tfv = TfidfVectorizer(
                min_df=2, max_features=None,
                strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                ngram_range=(1, 1), use_idf=True, smooth_idf=True, sublinear_tf=False)  # min_df=2, ngram_range=(1, 3)
            with timer('fasttext tfidf fit_transform'):
                X = tfv.fit_transform(corpus)
            print('X', type(X), X.shape)  # sparse matrix (num_text, num_vocab)
            num_sent, num_feat = X.shape
            feat_names = tfv.get_feature_names()
            print('feat_names', len(feat_names))
            assert num_feat == len(feat_names)
        else:
            raise ValueError("[ERROR] Unexpected value method={}".format(method))

        unknown_count = 0
        num_vocab, hdim = vec_array.shape
        feat_array = numpy.empty((num_feat, hdim), dtype=numpy.float32)
        for i, name in enumerate(feat_names):
            try:
                ind = vocab_list.index(name)
            except ValueError:
                ind = len(vocab_list) - 1  # unknown word index???
                unknown_count += 1
                # print(f'unknown word: {name}, unknown_count {unknown_count}')
            feat_array[i] = vec_array[ind]

        # X (num_sentence, num_feat), feat_array (num_feat, hdim)
        # h_sent = numpy.matmul(X, feat_array)
        h_sent = X * feat_array
        assert h_sent.shape == (num_sent, hdim)
        if cache_filepath is not None:
            save_npz(cache_filepath, h_sent)

    if out_dim is not None:
        if contraction_method == 'pooling':
            num_sent, hdim = h_sent.shape
            pool_size = hdim // out_dim
            h_sent = numpy.mean(
                h_sent[:, :out_dim*pool_size].reshape(num_sent, out_dim, pool_size),
                axis=2)
        elif contraction_method == 'svd':
            svd_ = TruncatedSVD(
                n_components=out_dim, random_state=1337)
            h_sent = svd_.fit_transform(h_sent)
        else:
            raise ValueError("[ERROR] Unexpected value contraction_method={}".format(contraction_method))
    return h_sent


if not is_kaggle_kernel and __name__ == '__main__':
    use_cache = True
    npz_filepath = 'crawl-300d-2M.npz'
    if use_cache and os.path.exists(npz_filepath):
        a = numpy.load(npz_filepath)
        vec_array = a['vec_array']
        vocab_list = list(a['vocab_list'])
    else:
        filepath = './crawl-300d-2M.vec'
        # filepath = './test_sample.vec'
        vocab_list, vec_array = construct_fasttext_vocab_list(filepath)
        numpy.savez_compressed(npz_filepath, vocab_list=vocab_list, vec_array=vec_array)

    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    h_sent = calc_fasttext_feature(corpus, vocab_list, vec_array)
    print('h_sent', h_sent.shape)
    print(h_sent)


# ### Glove text feature embedding extraction

# In[46]:


# --- glove_exp ---


# In[47]:


import os

import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

import sys
import os

# sys.path.append(os.pardir)
# sys.path.append(os.path.join(os.pardir, os.pardir))
# from src.configs import is_kaggle_kernel
# from src.utils import save_npz, load_npz, timer
# from src.fasttext.fasttext_exp import calc_fasttext_feature, extract_line


def construct_glove_vocab_list(filepath, hdim=50):
    """

    Args:
        filepath (str): fast text embedding file path

    Returns:
        vocab_list (list): length is `num_vocab` in file
        vec_array (numpy.ndarray): (num_vocab, hdim)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        # a = f.readline()
        # num_vocab, hdim = a.split(' ')
        # num_vocab, hdim = int(num_vocab), int(hdim)
        # print('num_vocab', num_vocab, 'hdim', hdim)
        all_lines = f.readlines()
        # print('all_lines:', all_lines)
        print('all_lines extracted on RAM')
    num_vocab = len(all_lines)
    print('num_vocab', num_vocab, 'hdim', hdim)

    vec_array = numpy.empty((num_vocab, hdim), dtype=numpy.float32)
    # from joblib import Parallel, delayed
    # n_jobs = 16
    # results = Parallel(n_jobs, backend='threading', verbose=1)(
    #     delayed(extract_line)(line, i, vec_array) for i, line in enumerate(all_lines))
    vocab_list = [extract_line(line, i, vec_array) for i, line in tqdm(enumerate(all_lines), total=len(all_lines))]
    return vocab_list, vec_array


if not is_kaggle_kernel and __name__ == '__main__':
    use_cache = True
    npz_filepath = 'glove-50d-6B.npz'
    if use_cache and os.path.exists(npz_filepath):
        a = numpy.load(npz_filepath)
        vec_array = a['vec_array']
        vocab_list = list(a['vocab_list'])
    else:
        filepath = './glove.6B.50d.txt'
        # filepath = './test_sample.vec'
        vocab_list, vec_array = construct_glove_vocab_list(filepath)
        numpy.savez_compressed(npz_filepath, vocab_list=vocab_list, vec_array=vec_array)

    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    h_sent = calc_fasttext_feature(corpus, vocab_list, vec_array)
    print('h_sent', h_sent.shape)
    print(h_sent)


# In[48]:


# --- pet finder parser ---


# In[49]:


"""
Scripts from: https://www.kaggle.com/wrosinski/baselinemodeling
"""
from glob import glob
import os
from logging import getLogger

import feather
import pandas as pd
import json
import numpy as np
from PIL import Image
from joblib import Parallel, delayed

class PetFinderParser(object):

    def __init__(self, debug=False):

        self.debug = debug
        self.sentence_sep = ' '

        # Does not have to be extracted because main DF already contains description
        self.extract_sentiment_text = False
        # version 1 is from `baselinemodeling` kernel
        # version 2 is from `simple xgboost model` kernel
        self.version = 2

    def open_metadata_file(self, filename):
        """
        Load metadata file.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            metadata_file = json.load(f)
        return metadata_file

    def open_sentiment_file(self, filename):
        """
        Load sentiment file.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            sentiment_file = json.load(f)
        return sentiment_file

    def open_image_file(self, filename):
        """
        Load image file.
        """
        image = np.asarray(Image.open(filename))
        return image

    def parse_sentiment_file(self, file):
        """
        Parse sentiment file. Output DF with sentiment features.
        """

        file_sentiment = file['documentSentiment']
        file_entities = [x['name'] for x in file['entities']]
        file_entities = self.sentence_sep.join(file_entities)

        if self.version == 1:
            if self.extract_sentiment_text:
                file_sentences_text = [x['text']['content'] for x in file['sentences']]
                file_sentences_text = self.sentence_sep.join(file_sentences_text)
            file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]

            file_sentences_sentiment = pd.DataFrame.from_dict(
                file_sentences_sentiment, orient='columns').sum()
            file_sentences_sentiment = file_sentences_sentiment.add_prefix('document_').to_dict()

            file_sentiment.update(file_sentences_sentiment)
            df_sentiment = pd.DataFrame.from_dict(file_sentiment, orient='index').T
            if self.extract_sentiment_text:
                df_sentiment['text'] = file_sentences_text

            df_sentiment['entities'] = file_entities
            df_sentiment = df_sentiment.add_prefix('sentiment_')
        elif self.version == 2:
            file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]
            file_sentences_sentiment = pd.DataFrame.from_dict(
                file_sentences_sentiment, orient='columns')
            file_sentences_sentiment_df = pd.DataFrame(
                {
                    'magnitude_sum': file_sentences_sentiment['magnitude'].sum(axis=0),
                    'score_sum': file_sentences_sentiment['score'].sum(axis=0),
                    'magnitude_mean': file_sentences_sentiment['magnitude'].mean(axis=0),
                    'score_mean': file_sentences_sentiment['score'].mean(axis=0),
                    'magnitude_var': file_sentences_sentiment['magnitude'].var(axis=0),
                    'score_var': file_sentences_sentiment['score'].var(axis=0),
                }, index=[0]
            )
            df_sentiment = pd.DataFrame.from_dict(file_sentiment, orient='index').T
            df_sentiment = pd.concat([df_sentiment, file_sentences_sentiment_df], axis=1)

            df_sentiment['entities'] = file_entities
            df_sentiment = df_sentiment.add_prefix('sentiment_')
        else:
            raise ValueError("[ERROR] Unexpected value self.version={}".format(self.version))
        return df_sentiment

    def parse_metadata_file(self, file):
        """
        Parse metadata file. Output DF with metadata features.
        """

        file_keys = list(file.keys())

        if 'labelAnnotations' in file_keys:
            if self.version == 1:
                file_annots = file['labelAnnotations'][:int(len(file['labelAnnotations']) * 0.3)]
            else:
                file_annots = file['labelAnnotations']
            # file_top_score = np.asarray([float(x['score']) for x in file_annots]).mean()
            file_top_score = np.asarray([x['score'] for x in file_annots]).mean()
            file_top_desc = [x['description'] for x in file_annots]
        else:
            file_top_score = np.nan
            file_top_desc = ['']

        file_colors = file['imagePropertiesAnnotation']['dominantColors']['colors']
        file_crops = file['cropHintsAnnotation']['cropHints']

        file_color_score = np.asarray([float(x['score']) for x in file_colors]).mean()
        file_color_pixelfrac = np.asarray([x['pixelFraction'] for x in file_colors]).mean()

        file_crop_conf = np.asarray([x['confidence'] for x in file_crops]).mean()

        if 'importanceFraction' in file_crops[0].keys():
            file_crop_importance = np.asarray([x['importanceFraction'] for x in file_crops]).mean()
        else:
            file_crop_importance = np.nan

        df_metadata = {
            'annots_score': file_top_score,
            'color_score': file_color_score,
            'color_pixelfrac': file_color_pixelfrac,
            'crop_conf': file_crop_conf,
            'crop_importance': file_crop_importance,
            'annots_top_desc': self.sentence_sep.join(file_top_desc)
        }

        df_metadata = pd.DataFrame.from_dict(df_metadata, orient='index').T
        df_metadata = df_metadata.add_prefix('metadata_')

        return df_metadata


# Helper function for parallel data processing:
def extract_additional_features(pet_parser, pet_id, mode='train'):
    sentiment_filename = '{}/{}_sentiment/{}.json'.format(pet_dir, mode, pet_id)
    try:
        sentiment_file = pet_parser.open_sentiment_file(sentiment_filename)
        try:
            df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)
            df_sentiment['PetID'] = pet_id
        except Exception:
            df_sentiment = []
    except FileNotFoundError:
        df_sentiment = []

    dfs_metadata = []
    metadata_filenames = sorted(glob('{}/{}_metadata/{}*.json'.format(pet_dir, mode, pet_id)))
    if len(metadata_filenames) > 0:
        for f in metadata_filenames:
            try:
                metadata_file = pet_parser.open_metadata_file(f)
                df_metadata = pet_parser.parse_metadata_file(metadata_file)
                df_metadata['PetID'] = pet_id
                dfs_metadata.append(df_metadata)
            except Exception:
                pass
        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)
    dfs = [df_sentiment, dfs_metadata]

    return dfs


class SentimentMetadataPreprocessor(object):
    def __init__(self):
        self.train_dfs_sentiment, self.train_dfs_metadata = None, None
        self.test_dfs_sentiment, self.test_dfs_metadata = None, None
        self.train_sentiment_gr, self.train_metadata_gr = None, None
        self.test_sentiment_gr, self.test_metadata_gr = None, None
        self.train_sentiment_desc, self.train_metadata_desc = None, None
        self.test_sentiment_desc, self.test_metadata_desc = None, None
        # version 1 is from `baselinemodeling` kernel
        # version 2 is from `simple xgboost model` kernel
        self.version = 2

    def preprocess_sentiment_and_metadata(self, train, test, n_jobs=16, use_cache=True):
        key = 'train{}_test{}_version{}'.format(len(train), len(test), self.version)
        if use_cache:
            result = self.load(key)
            if result:
                return

        # Unique IDs from train and test:
        train_pet_ids = train.PetID.unique()
        test_pet_ids = test.PetID.unique()

        pet_parser = PetFinderParser()
        # Train set:
        # Parallel processing of data:
        dfs_train = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(extract_additional_features)(pet_parser, i, mode='train') for i in train_pet_ids)

        # Extract processed data and format them as DFs:
        train_dfs_sentiment = [x[0] for x in dfs_train if isinstance(x[0], pd.DataFrame)]
        train_dfs_metadata = [x[1] for x in dfs_train if isinstance(x[1], pd.DataFrame)]

        train_dfs_sentiment = pd.concat(train_dfs_sentiment, ignore_index=True, sort=False)
        train_dfs_metadata = pd.concat(train_dfs_metadata, ignore_index=True, sort=False)

        print(train_dfs_sentiment.shape, train_dfs_metadata.shape)

        # Test set:
        # Parallel processing of data:
        dfs_test = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(extract_additional_features)(pet_parser, i, mode='test') for i in test_pet_ids)

        # Extract processed data and format them as DFs:
        test_dfs_sentiment = [x[0] for x in dfs_test if isinstance(x[0], pd.DataFrame)]
        test_dfs_metadata = [x[1] for x in dfs_test if isinstance(x[1], pd.DataFrame)]

        test_dfs_sentiment = pd.concat(test_dfs_sentiment, ignore_index=True, sort=False)
        test_dfs_metadata = pd.concat(test_dfs_metadata, ignore_index=True, sort=False)

        print(test_dfs_sentiment.shape, test_dfs_metadata.shape)

        self.train_dfs_sentiment, self.train_dfs_metadata = train_dfs_sentiment, train_dfs_metadata
        self.test_dfs_sentiment, self.test_dfs_metadata = test_dfs_sentiment, test_dfs_metadata
        # --- Group extracted features by PetID ---
        # Extend aggregates and improve column naming
        if self.version == 1:
            aggregates = ['mean', 'sum']
            sent_agg = ['mean', 'sum']
        elif self.version == 2:
            aggregates = ['mean', 'sum', 'var']
            sent_agg = ['sum']

        # Train
        train_metadata_desc = train_dfs_metadata.groupby(['PetID'])['metadata_annots_top_desc'].unique()
        train_metadata_desc = train_metadata_desc.reset_index()
        train_metadata_desc[
            'metadata_annots_top_desc'] = train_metadata_desc[
            'metadata_annots_top_desc'].apply(lambda x: ' '.join(x))

        prefix = 'metadata'
        train_metadata_gr = train_dfs_metadata.drop(['metadata_annots_top_desc'], axis=1)
        for i in train_metadata_gr.columns:
            if 'PetID' not in i:
                train_metadata_gr[i] = train_metadata_gr[i].astype(float)
        train_metadata_gr = train_metadata_gr.groupby(['PetID']).agg(aggregates)
        if self.version == 1:
            train_metadata_gr.columns = pd.Index(['{}_{}_{}'.format(
                prefix, c[0], c[1].upper()) for c in train_metadata_gr.columns.tolist()])
        elif self.version == 2:
            train_metadata_gr.columns = pd.Index([f'{c[0]}_{c[1].upper()}' for c in train_metadata_gr.columns.tolist()])
        train_metadata_gr = train_metadata_gr.reset_index()

        train_sentiment_desc = train_dfs_sentiment.groupby(['PetID'])['sentiment_entities'].unique()
        train_sentiment_desc = train_sentiment_desc.reset_index()
        train_sentiment_desc[
            'sentiment_entities'] = train_sentiment_desc[
            'sentiment_entities'].apply(lambda x: ' '.join(x))

        prefix = 'sentiment'
        train_sentiment_gr = train_dfs_sentiment.drop(['sentiment_entities'], axis=1)
        for i in train_sentiment_gr.columns:
            if 'PetID' not in i:
                train_sentiment_gr[i] = train_sentiment_gr[i].astype(float)
        train_sentiment_gr = train_sentiment_gr.groupby(['PetID']).agg(sent_agg)
        if self.version == 1:
            train_sentiment_gr.columns = pd.Index(['{}_{}_{}'.format(
                prefix, c[0], c[1].upper()) for c in train_sentiment_gr.columns.tolist()])
        elif self.version == 2:
            train_sentiment_gr.columns = pd.Index([f'{c[0]}' for c in train_sentiment_gr.columns.tolist()])
        train_sentiment_gr = train_sentiment_gr.reset_index()

        # Test
        test_metadata_desc = test_dfs_metadata.groupby(['PetID'])['metadata_annots_top_desc'].unique()
        test_metadata_desc = test_metadata_desc.reset_index()
        test_metadata_desc[
            'metadata_annots_top_desc'] = test_metadata_desc[
            'metadata_annots_top_desc'].apply(lambda x: ' '.join(x))

        prefix = 'metadata'
        test_metadata_gr = test_dfs_metadata.drop(['metadata_annots_top_desc'], axis=1)
        for i in test_metadata_gr.columns:
            if 'PetID' not in i:
                test_metadata_gr[i] = test_metadata_gr[i].astype(float)
        test_metadata_gr = test_metadata_gr.groupby(['PetID']).agg(aggregates)
        if self.version == 1:
            test_metadata_gr.columns = pd.Index(['{}_{}_{}'.format(
                prefix, c[0], c[1].upper()) for c in test_metadata_gr.columns.tolist()])
        elif self.version == 2:
            test_metadata_gr.columns = pd.Index([f'{c[0]}_{c[1].upper()}'
                                                 for c in test_metadata_gr.columns.tolist()])
        test_metadata_gr = test_metadata_gr.reset_index()

        test_sentiment_desc = test_dfs_sentiment.groupby(['PetID'])['sentiment_entities'].unique()
        test_sentiment_desc = test_sentiment_desc.reset_index()
        test_sentiment_desc[
            'sentiment_entities'] = test_sentiment_desc[
            'sentiment_entities'].apply(lambda x: ' '.join(x))

        prefix = 'sentiment'
        test_sentiment_gr = test_dfs_sentiment.drop(['sentiment_entities'], axis=1)
        for i in test_sentiment_gr.columns:
            if 'PetID' not in i:
                test_sentiment_gr[i] = test_sentiment_gr[i].astype(float)
        test_sentiment_gr = test_sentiment_gr.groupby(['PetID']).agg(sent_agg)
        if self.version == 1:
            test_sentiment_gr.columns = pd.Index(['{}_{}_{}'.format(
                prefix, c[0], c[1].upper()) for c in test_sentiment_gr.columns.tolist()])
        elif self.version == 2:
            test_sentiment_gr.columns = pd.Index([f'{c[0]}' for c in test_sentiment_gr.columns.tolist()])
        test_sentiment_gr = test_sentiment_gr.reset_index()
        # return train_dfs_sentiment, train_dfs_metadata, test_dfs_sentiment, test_dfs_metadata
        self.train_sentiment_gr, self.train_metadata_gr = train_sentiment_gr, train_metadata_gr
        self.test_sentiment_gr, self.test_metadata_gr = test_sentiment_gr, test_metadata_gr
        self.train_sentiment_desc, self.train_metadata_desc = train_sentiment_desc, train_metadata_desc
        self.test_sentiment_desc, self.test_metadata_desc = test_sentiment_desc, test_metadata_desc

        self.save(key)
        return

    def save(self, key):
        os.makedirs('cache', exist_ok=True)
        for i, df in enumerate([self.train_dfs_sentiment, self.train_dfs_metadata,
                   self.test_dfs_sentiment, self.test_dfs_metadata,
                   self.train_sentiment_gr, self.train_metadata_gr,
                   self.test_sentiment_gr, self.test_metadata_gr,
                   self.train_sentiment_desc, self.train_metadata_desc,
                   self.test_sentiment_desc, self.test_metadata_desc]):
            df.to_feather('cache/smp_{}_{:03}.feather'.format(key, i))

    def load(self, key):
        df_list = []
        for i in range(12):
            if os.path.exists('cache/smp_{}_{:03}.feather'.format(key, i)):
                df = read_feather('cache/smp_{}_{:03}.feather'.format(key, i))
                df_list.append(df)
            else:
                return False
        self.train_dfs_sentiment, self.train_dfs_metadata,         self.test_dfs_sentiment, self.test_dfs_metadata,         self.train_sentiment_gr, self.train_metadata_gr,         self.test_sentiment_gr, self.test_metadata_gr,         self.train_sentiment_desc, self.train_metadata_desc,         self.test_sentiment_desc, self.test_metadata_desc = tuple(df_list)
        return True


def read_feather(filepath):
    return feather.read_dataframe(filepath)



# In[ ]:





# Feature extraction of `rating.json` from external dataset for Breed feature.

# In[50]:


# --- rating eda ---
import json
import os
import pandas as pd
import numpy as np


def load_json(filepath):
    """Load params, which is stored in json format.

    Args:
        filepath (str): filepath to json file to load.

    Returns (dict or list): params
    """
    with open(filepath, 'r') as f:
        params = json.load(f)
    return params


def calc_breed_rating_feat(breeds, thresh=70):
    json_path = os.path.join(json_dir, 'rating.json')
    params = load_json(json_path)
    cat_df = pd.DataFrame(params['cat_breeds']).T
    dog_df = pd.DataFrame(params['dog_breeds']).T
    print('cat_df', cat_df.shape, 'dog_df', dog_df.shape)

    def _extract_rating_feature(breed_name, df):
        names_in_rating = list(df.index.values)
        if breed_name in names_in_rating:
            return df.loc[breed_name, :]
        else:
            exist = 0
            for breed_name_elem in breed_name.split(' '):
                exist += np.array([(breed_name_elem in names) for names in names_in_rating])
            if np.sum(exist) > 0:
                result = df[exist > 0].mean(axis=0)
                result.name = breed_name
                return result
            else:
                # return all df's mean...
                result = df.mean(axis=0)
                result.name = breed_name
                return result

    def extract_rating_feature(row):
        # print('row', row)
        animal_type = row['Type']
        breed_name = row['BreedName']
        if animal_type == 1:
            return _extract_rating_feature(breed_name, dog_df)
        else:
            assert animal_type == 2
            return _extract_rating_feature(breed_name, cat_df)

    # debug
    feat_list = []
    # breeds = breeds.iloc[:5]
    for index, row in breeds.iterrows():
        # print('row', type(row))
        feat = extract_rating_feature(row)
        feat_list.append(feat)
    # a = breeds.loc[:3, ['Type', 'BreedName']].apply(extract_rating_feature)
    feat_df = pd.concat(feat_list, axis=1, sort=False).T
    # thresh = 60  # 67 is big threshold for dog/cat
    if thresh is not None:
        feat_df = feat_df.loc[:, feat_df.isna().sum() < thresh]
    return feat_df


# Language feature extraction
# 
# Various kinds of people are lived in Malaysia (Malay, Chinese), thus I thought it is important feature.

# In[51]:


from langdetect import detect
from joblib import Parallel, delayed
import pandas as pd


import sys
import os

def detect_wrapper(s):
    try:
        lang = detect(s)
    except Exception:
        lang = 'unknown'
    if lang.startswith('zh-'):
        lang = 'zh-cn'  # consider chinese char as same.
    if lang not in ['en', 'id', 'da', 'de', 'zh-cn', 'unknown']:
        lang = 'others'
    return lang


def process_lang_df(train, test, use_cache=True):
    train_cache_filepath = './cache/train_lang_df.feather'
    test_cache_filepath = './cache/test_lang_df.feather'
    if use_cache and os.path.exists(train_cache_filepath) and os.path.exists(test_cache_filepath):
        train_df = read_feather(train_cache_filepath)
        test_df = read_feather(test_cache_filepath)
    else:
        n_jobs = 16
        train_results = Parallel(n_jobs, verbose=1)(
            delayed(detect_wrapper)(s) for s in train['Description'].values)
        test_results = Parallel(n_jobs, verbose=1)(
            delayed(detect_wrapper)(s) for s in test['Description'].values)
        train_df = pd.DataFrame(train_results, columns=['lang'])
        test_df = pd.DataFrame(test_results, columns=['lang'])
        train_df.to_feather(train_cache_filepath)
        test_df.to_feather(test_cache_filepath)
    return train_df, test_df


# In[ ]:





# In[52]:


# --- preprocessing 2 ---


# In[53]:


from copy import deepcopy
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
import os

import sys
import os

class Preprocessor(object):
    def __init__(self, arch='xgb'):
        if arch == 'xlearn':
            self.numeric_cols = []
            self.cat_cols = [
                'Age', 'MaturitySize', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt',
                'FurLength', 'Health', 'Vaccinated', 'Dewormed', 'Sterilized',
                'Gender', 'Color1', 'Color2', 'Color3', 'Type', 'Breed1', 'Breed2', 'State']
        elif arch == 'xgb':
            # self.numeric_cols = [
            #     'PhotoAmt', 'Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
            #     'Vaccinated', 'Dewormed', 'Sterilized', 'State', 'FurLength', 'Health',
            #     'Age', 'MaturitySize', 'Quantity', 'Fee', 'VideoAmt']
            # self.cat_cols = []
            self.numeric_cols = ['Age', 'MaturitySize', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt',
                                 'FurLength', 'Health', 'Vaccinated', 'Dewormed', 'Sterilized',
                                 'Gender', 'Color1', 'Color2', 'Color3', 'Type', 'Breed1', 'Breed2', 'State']
            self.cat_cols = []
            # self.numeric_cols = ['Age', 'MaturitySize', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']
            # self.cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
            #                  'Vaccinated', 'Dewormed', 'Sterilized', 'State', 'FurLength', 'Health']
        elif arch == 'nn':
            # nn
            self.numeric_cols = ['Age', 'MaturitySize', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']
            self.cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
                             'Vaccinated', 'Dewormed', 'Sterilized', 'State', 'FurLength', 'Health']
            # self.numeric_cols = ['Age', ]
            # self.cat_cols = ['Fee', 'MaturitySize', 'Quantity', 'PhotoAmt', 'VideoAmt',
            #     'FurLength', 'Health', 'Vaccinated', 'Dewormed', 'Sterilized',
            #     'Gender', 'Color1', 'Color2', 'Color3', 'Type', 'Breed1', 'Breed2', 'State']
        else:
            # lgbm, cb
            self.numeric_cols = ['Age', 'MaturitySize', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']
            self.cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
                             'Vaccinated', 'Dewormed', 'Sterilized', 'State', 'FurLength', 'Health']

        # TODO: process these features properly later...
        remove_cols = ['RescuerID', 'PetID', 'AdoptionSpeed']
        other_cols = ['Name', 'Description']
        self.train_indices = None
        self.test_indices = None
        self.target = None
        self.num_cat_id = None
        self.arch = arch

    def remove_cols(self, cols):
        if isinstance(cols, str):
            cols = [cols]

        for col in cols:
            if col in self.cat_cols:
                self.cat_cols.remove(col)
                print(f'removing {col} from cat_cols...')
            if col in self.numeric_cols:
                self.numeric_cols.remove(col)
                print(f'removing {col} from numeric_cols...')

    def preprocess(self, train, test, breeds, colors, states,
                   debug=False, use_tfidf=True, use_tfidf_cache=True,
                   use_sentiment=True, use_metadata=False, cat2num=True,
                   use_rescuer_id_count=True, use_name_feature=True, use_target_encoding=False,
                   tfidf_svd_components=120, animal_type=None, num_sentiment_text=0,
                   use_gdp=False, arch='lgbm',
                   use_sentiment2=True, use_metadata2=True, use_text=True, use_fasttext=True,
                   embedding_type_list=None,
                   use_image_size=True, use_breed_feature=True, add_pred=None,
                   use_custom_boolean_feature=True):
        categorical_cols = deepcopy(self.cat_cols)

        print('preproces... arch={}'.format(arch))
        train['Name'].fillna('none', inplace=True)
        test['Name'].fillna('none', inplace=True)
        # train['Description'].fillna('none', inplace=True)
        # test['Description'].fillna('none', inplace=True)

        train['dataset_type'] = 'train'
        test['dataset_type'] = 'test'
        all_data = pd.concat([train, test], axis=0, sort=False)
        train_indices = (all_data['dataset_type'] == 'train').values
        test_indices = (all_data['dataset_type'] == 'test').values
        self.train_indices = train_indices
        self.test_indices = test_indices
        assert self.train_indices[len(train):].sum() == 0
        assert self.test_indices[:len(train)].sum() == 0
        target = train['AdoptionSpeed'].values.astype(np.float32)[:, None]
        self.target = target

        if arch in ['xlearn', 'nn']:  #
            # self.cat_cols = [
            #     'Age', 'MaturitySize', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt',
            #     'FurLength', 'Health', 'Vaccinated', 'Dewormed', 'Sterilized',
            #     'Gender', 'Color1', 'Color2', 'Color3', 'Type', 'Breed1', 'Breed2', 'State']
            # --- cutoff ---
            age_cutoff = 84  # TODO: hyperparameter tuning. This affects a lot!!!
            print('age_cutoff', age_cutoff)
            all_data.loc[all_data['Age'] >= age_cutoff, 'Age'] = age_cutoff
            # 'Quantity', max was 20, but most of them are ~10.
            quantity_cutoff = 15
            all_data.loc[all_data['Quantity'] >= quantity_cutoff, 'Quantity'] = quantity_cutoff
            # 'VideoAmt', max was 8, but most of them are ~1.
            video_amt_cutoff = 5
            all_data.loc[all_data['VideoAmt'] >= video_amt_cutoff, 'VideoAmt'] = video_amt_cutoff
            # 'PhotoAmt', max was 30, but most of them are ~11.
            # photo_amt_cutoff = 12
            photo_amt_cutoff = 15
            all_data.loc[all_data['PhotoAmt'] >= photo_amt_cutoff, 'PhotoAmt'] = photo_amt_cutoff
            # 'Fee', max was 3000, but most of them are ~300 or ~500
            # 300 or 500
            fee_cutoff = 500
            all_data.loc[all_data['Fee'] >= fee_cutoff, 'Fee'] = fee_cutoff

            # --- discretize to bin ---
            print('discretize...')

            age_cut_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24, 30, 36, 48, 60, 72, 84]
            age_array = all_data['Age'].values
            for i in range(len(age_cut_list) - 1):
                age_th = age_cut_list[i]
                age_th2 = age_cut_list[i+1]
                all_data.loc[(age_array >= age_th) & (age_array < age_th2), 'Age'] = age_th
            fee_cut_list = [0, 1, 10, 50, 100, 500]
            fee_array = all_data['Fee'].values
            for i in range(len(fee_cut_list) - 1):
                fee_th = fee_cut_list[i]
                fee_th2 = fee_cut_list[i+1]
                all_data.loc[(fee_array >= fee_th) & (fee_array < fee_th2), 'Fee'] = fee_th
            # all_data.loc[:, 'Age'] = pd.cut(
            #     all_data['Age'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24, 30, 36, 48, 60, 72, 84], labels=False)
            # all_data.loc[:, 'Fee'] = pd.cut(
            #     all_data['Fee'], [0, 1, 10, 50, 100, 500], labels=False)

            train = all_data.iloc[train_indices].copy()
            test = all_data.iloc[test_indices].copy()

        if use_gdp:
            add_gdp(train)
            add_gdp(test)

            # Scaling... max value must be calculated beforehand!!
            state_gdp_max = train["state_gdp"].max()
            state_pop_max = train["state_population"].max()
            train["state_gdp"] = train["state_gdp"] / state_gdp_max
            test["state_gdp"] = test["state_gdp"] / state_gdp_max
            train["state_population"] = train["state_population"] / state_pop_max
            test["state_population"] = test["state_population"] / state_pop_max

            self.numeric_cols.append('state_gdp')
            self.numeric_cols.append('state_population')

        if use_rescuer_id_count:
            cutoff_count = 20  # TODO: check which is better. not use it for now...
            # cutoff_count = -1  # not cutoff it for now...
            print(f'use_rescuer_id_count... cutoff {cutoff_count}')
            rescuer_id_count_col_name = 'RescuerID_COUNT'
            train[rescuer_id_count_col_name] = train.groupby('RescuerID')[
                'RescuerID'].transform(lambda s: s.count())
            test[rescuer_id_count_col_name] = test.groupby('RescuerID')[
                'RescuerID'].transform(lambda s: s.count())

            if arch == ['xlearn']:  #, 'nn'
                self.cat_cols.append(rescuer_id_count_col_name)
            else:
                self.numeric_cols.append(rescuer_id_count_col_name)

            if cutoff_count > 0:
                train.loc[train[rescuer_id_count_col_name] >= cutoff_count, rescuer_id_count_col_name] = cutoff_count
                test.loc[test[rescuer_id_count_col_name] >= cutoff_count, rescuer_id_count_col_name] = cutoff_count

            # "is_first_time" feature
            use_is_first_time = False  # TODO: check
            if use_is_first_time:
                train['is_first_time'] = (train[rescuer_id_count_col_name] == 1).astype(np.float32)
                test['is_first_time'] = (test[rescuer_id_count_col_name] == 1).astype(np.float32)
                if arch == 'xlearn':
                    self.cat_cols.append('is_first_time')
                else:
                    self.numeric_cols.append('is_first_time')

        if use_name_feature:
            print('create name feature...')
            # create name feature
            # 1. no name or not
            train['No_name'] = 0
            train.loc[train['Name'] == 'none', 'No_name'] = 1
            test['No_name'] = 0
            test.loc[test['Name'] == 'none', 'No_name'] = 1

            # 2. weired name or not
            train['name_under2'] = train['Name'].apply(lambda x: len(str(x)) < 3).values.astype(np.float32)
            test['name_under2'] = test['Name'].apply(lambda x: len(str(x)) < 3).values.astype(np.float32)

            # 3. puppy, puppies, kitten, kitty, baby flag.
            train['is_kitty'] = train['Name'].apply(lambda x: 'kitty' in str(x).lower()).values.astype(np.float32)
            test['is_kitty'] = test['Name'].apply(lambda x: 'kitty' in str(x).lower()).values.astype(np.float32)

            train['is_kitten'] = train['Name'].apply(lambda x: 'kitten' in str(x).lower()).values.astype(np.float32)
            test['is_kitten'] = test['Name'].apply(lambda x: 'kitten' in str(x).lower()).values.astype(np.float32)

            train['is_puppy'] = train['Name'].apply(lambda x: 'puppy' in str(x).lower()).values.astype(np.float32)
            test['is_puppy'] = test['Name'].apply(lambda x: 'puppy' in str(x).lower()).values.astype(np.float32)

            train['is_puppies'] = train['Name'].apply(lambda x: 'puppies' in str(x).lower()).values.astype(np.float32)
            test['is_puppies'] = test['Name'].apply(lambda x: 'puppies' in str(x).lower()).values.astype(np.float32)

            if arch in ['xlearn', 'nn']:
                self.cat_cols.append('No_name')
                self.cat_cols.append('name_under2')
                self.cat_cols.append('is_kitty')
                self.cat_cols.append('is_kitten')
                self.cat_cols.append('is_puppy')
                self.cat_cols.append('is_puppies')
            else:
                self.numeric_cols.append('No_name')
                self.numeric_cols.append('name_under2')
            # self.numeric_cols.append('is_kitty')
            # self.numeric_cols.append('is_kitten')
            # self.numeric_cols.append('is_puppy')
            # self.numeric_cols.append('is_puppies')

        # --- Breed preprocessing ---
        # deal "Unspecified" and "Unknown" as same for 2nd breed
        # --> seems 307 is for mixed type!! and 0 is for unspecified. we should treat different way!
        breed2_preprocessing = False
        if breed2_preprocessing:
            print('breed2_preprocessing')
            all_data.loc[all_data['Breed2'] == 0, 'Breed2'] = 307
            train.loc[train['Breed2'] == 0, 'Breed2'] = 307
            test.loc[test['Breed2'] == 0, 'Breed2'] = 307

        breed12_preprocessing = True
        if breed12_preprocessing:
            print('breed12_preprocessing')
            # # https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/77898
            train['Breed1'] = np.where((train['Breed1'] == 0) & (train['Breed2'] != 0),
                                       train['Breed2'], train['Breed1'])
            train['Breed2'] = np.where((train['Breed1'] == train['Breed2']), 0, train['Breed2'])

        use_is_mixed = False
        if use_is_mixed:
            # is_mixed = (all_data['Breed2'] != 0) & (all_data['Breed2'] != 307)
            is_mixed = (all_data['Breed2'] != 307).astype(np.int32)
            all_data['is_mixed'] = is_mixed
            train['is_mixed'] = is_mixed[train_indices]
            test['is_mixed'] = is_mixed[test_indices]
            self.cat_cols.append('is_mixed')

        if use_breed_feature:
            train_breed_main = train[['Breed1']].merge(
                breeds, how='left',
                left_on='Breed1', right_on='BreedID',
                suffixes=('', '_main_breed'))

            train_breed_main = train_breed_main.iloc[:, 2:]
            train_breed_main = train_breed_main.add_prefix('main_breed_')

            train_breed_second = train[['Breed2']].merge(
                breeds, how='left',
                left_on='Breed2', right_on='BreedID',
                suffixes=('', '_second_breed'))

            train_breed_second = train_breed_second.iloc[:, 2:]
            train_breed_second = train_breed_second.add_prefix('second_breed_')

            train = pd.concat(
                [train, train_breed_main, train_breed_second], axis=1, sort=False)

            test_breed_main = test[['Breed1']].merge(
                breeds, how='left',
                left_on='Breed1', right_on='BreedID',
                suffixes=('', '_main_breed'))

            test_breed_main = test_breed_main.iloc[:, 2:]
            test_breed_main = test_breed_main.add_prefix('main_breed_')

            test_breed_second = test[['Breed2']].merge(
                breeds, how='left',
                left_on='Breed2', right_on='BreedID',
                suffixes=('', '_second_breed'))

            test_breed_second = test_breed_second.iloc[:, 2:]
            test_breed_second = test_breed_second.add_prefix('second_breed_')

            test = pd.concat(
                [test, test_breed_main, test_breed_second], axis=1, sort=False)
            print(train.shape, test.shape)

            X = pd.concat([train, test], ignore_index=True, axis=0, sort=False)
            X_temp = X.copy()
            categorical_columns = ['main_breed_BreedName', 'second_breed_BreedName']
            for i in categorical_columns:
                X_temp.loc[:, i] = pd.factorize(X_temp.loc[:, i])[0]
                # Becareful!! must assign `.values` otherwise index does not match and nan is inserted for test!!!
                train.loc[:, i] = X_temp.loc[train_indices, i].values
                test.loc[:, i] = X_temp.loc[test_indices, i].values
            breed_feat_cols = [
                'main_breed_Type',
                'main_breed_BreedName',
                'second_breed_Type',
                'second_breed_BreedName']

            # dtypes does not match due to nan existent... align to same dtypes.
            for col in breed_feat_cols:
                test[col] = test[col].astype(train[col].dtype)

            if arch == 'xgb':
                self.numeric_cols.extend(breed_feat_cols)
            else:
                self.cat_cols.extend(breed_feat_cols)
            # print('breed_feat_cols', breed_feat_cols)
            # print(train[breed_feat_cols].dtypes)
            # print(test[breed_feat_cols].dtypes)

            add_rating_json = True
            if add_rating_json:
                # --- rating.json ---
                feat_df = calc_breed_rating_feat(breeds, thresh=60)
                # How to fillna? mean is better?? try both way.
                feat_df.fillna(-1, inplace=True)
                # feat_df.fillna(feat_df.mean(), inplace=True)

                train_breed_main = train[['Breed1']].merge(
                    breeds, how='left',
                    left_on='Breed1', right_on='BreedID',)
                train_breed_main_feat = train_breed_main[['BreedName']].merge(
                    feat_df, how='left',
                    left_on='BreedName', right_index=True, )
                train_breed_main_feat = train_breed_main_feat.iloc[:, 1:]
                train_breed_main_feat = train_breed_main_feat.add_prefix('main_breed_feat_')

                train_breed_second = train[['Breed2']].merge(
                    breeds, how='left',
                    left_on='Breed2', right_on='BreedID',)
                train_breed_second_feat = train_breed_second[['BreedName']].merge(
                    feat_df, how='left',
                    left_on='BreedName', right_index=True, )
                train_breed_second_feat = train_breed_second_feat.iloc[:, 1:]
                train_breed_second_feat = train_breed_second_feat.add_prefix('second_breed_feat_')

                test_breed_main = test[['Breed1']].merge(
                    breeds, how='left',
                    left_on='Breed1', right_on='BreedID',)
                test_breed_main_feat = test_breed_main[['BreedName']].merge(
                    feat_df, how='left',
                    left_on='BreedName', right_index=True, )
                test_breed_main_feat = test_breed_main_feat.iloc[:, 1:]
                test_breed_main_feat = test_breed_main_feat.add_prefix('main_breed_feat_')

                test_breed_second = test[['Breed2']].merge(
                    breeds, how='left',
                    left_on='Breed2', right_on='BreedID',)
                test_breed_second_feat = test_breed_second[['BreedName']].merge(
                    feat_df, how='left',
                    left_on='BreedName', right_index=True, )
                test_breed_second_feat = test_breed_second_feat.iloc[:, 1:]
                test_breed_second_feat = test_breed_second_feat.add_prefix('second_breed_feat_')

                train = pd.concat(
                    [train, train_breed_main_feat, train_breed_second_feat], axis=1, sort=False)
                test = pd.concat(
                    [test, test_breed_main_feat, test_breed_second_feat], axis=1, sort=False)

                breed_rating_feat_cols = list(train_breed_main_feat.columns.values)
                self.numeric_cols.extend(breed_rating_feat_cols)

        use_color_feat = False
        if use_color_feat:
            color_array = np.array([
                [0, 0, 0],  # unspecified
                [0, 0, 0],  # black
                [165, 42, 42],  # brown
                [255, 215, 0],  # golden
                [255, 255, 0],  # yellow
                [255, 248, 220],  # cream
                [128, 128, 128],  # gray
                [255, 255, 255],  # white
            ], dtype=np.float32) / 255.
            for tmp_df in [train, test]:
                c1 = tmp_df['Color1'].values
                c2 = tmp_df['Color2'].values
                c3 = tmp_df['Color3'].values
                num_color = (c1 > 0).astype(np.float32) + (c2 > 0).astype(np.float32) + (c3 > 0).astype(np.float32)
                color = (color_array[c1] + color_array[c2] + color_array[c3]) / num_color[:, None]
                tmp_df['color_r'] = color[:, 0]
                tmp_df['color_g'] = color[:, 1]
                tmp_df['color_b'] = color[:, 2]
                tmp_df['num_color'] = num_color
            self.numeric_cols.extend(['color_r', 'color_g', 'color_b', 'num_color'])
            # self.numeric_cols.extend(['num_color'])

        if cat2num:
            breed_cutoff = -1  # Do not cutoff here, it will be cutoff later...
        else:
            if arch in ['xlearn', 'nn']:
                breed_cutoff = 3  # TODO: check it was best value...
            else:
                breed_cutoff = 6  # TODO: check it was best value...
        if breed_cutoff > 0:
            # b1 = all_data['Breed1'].value_counts().values
            breed_counts = pd.concat([all_data['Breed1'], all_data['Breed2']], axis=0).value_counts()
            minor_breeds = breed_counts[breed_counts < breed_cutoff].index.values
            # minor_breeds 178/201 for threshold 100.
            # minor_breeds 123/201 for threshold 10.
            # minor_breeds 101/201 for threshold 7.
            # minor_breeds 80/201 for threshold 5.
            print('minor_breeds {}/{}'.format(len(minor_breeds), len(breed_counts)))
            # Assign new "minor breed" id=308 for minor breeds...
            all_data['Breed1'] = all_data['Breed1'].apply(lambda s: 308 if s in minor_breeds else s)
            all_data['Breed2'] = all_data['Breed2'].apply(lambda s: 308 if s in minor_breeds else s)

        if use_target_encoding:
            # Becareful, it makes info leak and only fits validation data but not in test data!!!
            print('create target encoding feature...')
            # 1. --- Breed target encoding ---
            # breed1 = train.groupby('Breed1')['AdoptionSpeed']
            # train['breed1_mean'] = breed1.transform(np.mean)
            # train['breed1_median'] = breed1.transform(np.median)
            breed1 = all_data.groupby('Breed1')['AdoptionSpeed']
            all_data['breed1_mean'] = breed1.transform(np.mean)
            # all_data['breed1_q1'] = breed1.transform(lambda x: np.quantile(x, 0.25))
            breed2 = all_data.groupby('Breed2')['AdoptionSpeed']
            all_data['breed2_mean'] = breed2.transform(np.mean)
            # all_data['breed2_median'] = breed2.transform(np.median)

            # 2. --- State target encoding ---
            state = all_data.groupby('State')['AdoptionSpeed']
            all_data['state_mean'] = state.transform(np.mean)

            # Assign values into `train` and `test`...
            for col in ['breed1_mean', 'breed2_mean', 'state_mean']:
                train[col] = all_data[train_indices][col]
                test[col] = all_data[test_indices][col]
                self.numeric_cols.append(col)

        if use_custom_boolean_feature:
            # --- is_xxx flag ---
            train['is_free'] = 0
            train.loc[train['Fee'] == 0, 'is_free'] = 1
            test['is_free'] = 0
            test.loc[test['Fee'] == 0, 'is_free'] = 1

            train['has_photo'] = 0
            train.loc[train['PhotoAmt'] > 0, 'has_photo'] = 1
            test['has_photo'] = 0
            test.loc[test['PhotoAmt'] > 0, 'has_photo'] = 1

            train['age_unknown'] = 0
            train.loc[train['Age'] == 255, 'age_unknown'] = 1
            test['age_unknown'] = 0
            test.loc[test['Age'] == 255, 'age_unknown'] = 1
            if arch == 'xgb':
                self.numeric_cols.append('is_free')
                self.numeric_cols.append('has_photo')
                self.numeric_cols.append('age_unknown')
            else:
                self.cat_cols.append('is_free')
                self.cat_cols.append('has_photo')
                self.cat_cols.append('age_unknown')

        # --- Cutoff ---
        apply_cutoff = (arch not in ['xlearn'])  # 'nn',
        if apply_cutoff:
            # 'Age', max was 255.
            # TODO: we may need to deal with "255" as special (I think this is "unknown")
            # age_cutoff = 54  # TODO: hyperparameter tuning. This affects a lot!!!
            # age_cutoff = 60  # TODO: hyperparameter tuning. This affects a lot!!!
            # age_cutoff = 72  # TODO: hyperparameter tuning. This affects a lot!!!
            age_cutoff = 84  # TODO: hyperparameter tuning. This affects a lot!!!
            print('age_cutoff', age_cutoff)
            train.loc[train['Age'] >= age_cutoff, 'Age'] = age_cutoff
            test.loc[test['Age'] >= age_cutoff, 'Age'] = age_cutoff
            # 'Quantity', max was 20, but most of them are ~10.
            # quantity_cutoff = 11
            quantity_cutoff = 15
            train.loc[train['Quantity'] >= quantity_cutoff, 'Quantity'] = quantity_cutoff
            test.loc[test['Quantity'] >= quantity_cutoff, 'Quantity'] = quantity_cutoff
            # 'VideoAmt', max was 8, but most of them are ~1.
            # video_amt_cutoff = 4
            video_amt_cutoff = 5
            train.loc[train['VideoAmt'] >= video_amt_cutoff, 'VideoAmt'] = video_amt_cutoff
            test.loc[test['VideoAmt'] >= video_amt_cutoff, 'VideoAmt'] = video_amt_cutoff
            # 'PhotoAmt', max was 30, but most of them are ~11.
            # photo_amt_cutoff = 12
            photo_amt_cutoff = 15
            train.loc[train['PhotoAmt'] >= photo_amt_cutoff, 'PhotoAmt'] = photo_amt_cutoff
            test.loc[test['PhotoAmt'] >= photo_amt_cutoff, 'PhotoAmt'] = photo_amt_cutoff
            # 'Fee', max was 3000, but most of them are ~300 or ~500
            # 300 or 500
            fee_cutoff = 500
            train.loc[train['Fee'] >= fee_cutoff, 'Fee'] = fee_cutoff
            test.loc[test['Fee'] >= fee_cutoff, 'Fee'] = fee_cutoff

        if arch == 'nn':
            # Neural Network preprocessing...
            # --- Numeric value processing ---
            print('numeric value preprocessing...')
            # There is no nan value, but this is just for make sure no nan exist.
            scale_cols = deepcopy(self.numeric_cols)

        if use_sentiment:
            print('create sentiment feature...')
            n_jobs = 16
            s = perf_counter()
            # Multiprocessing: around 2 sec. Multithreading: 10 sec. Singlethreading 63 sec.
            # train_x_sent = Parallel(n_jobs, backend='threading')(
            #     delayed(process_sentiment, check_pickle=False)
            #     (petid, 'train') for petid in train['PetID'].values)
            train_x_sent = Parallel(n_jobs)(
                delayed(process_sentiment)
                (petid, 'train', num_sentiment_text) for petid in train['PetID'].values)
            test_x_sent = Parallel(n_jobs)(
                delayed(process_sentiment)
                (petid, 'test', num_sentiment_text) for petid in test['PetID'].values)
            e = perf_counter()
            print('sentiment {} sec, n_jobs {}'.format(e - s, n_jobs))
            train_x_sent = np.array(train_x_sent, dtype=np.float32)
            test_x_sent = np.array(test_x_sent, dtype=np.float32)
            print('train_x_sent {}'
                  .format(train_x_sent.shape))
            # TODO: update feature engineering and its name...
            sentiment_cols = ['sent01', 'sent02']
            train_sentiment_df = pd.DataFrame(train_x_sent, columns=sentiment_cols)
            test_sentiment_df = pd.DataFrame(test_x_sent, columns=sentiment_cols)
            train = pd.concat([train, train_sentiment_df], axis=1)
            test = pd.concat([test, test_sentiment_df], axis=1)
            self.numeric_cols.extend(sentiment_cols)

        if use_metadata:
            print('create metadata feature...')
            n_jobs = 16
            s = perf_counter()
            train_x_metadata = Parallel(n_jobs)(
                delayed(process_metadata)
                (petid, 'train') for petid in train['PetID'].values)
            test_x_metadata = Parallel(n_jobs)(
                delayed(process_metadata)
                (petid, 'test') for petid in test['PetID'].values)
            e = perf_counter()
            print('metadata {} sec, n_jobs {}'.format(e-s, n_jobs))
            train_x_metadata = np.array(train_x_metadata, dtype=np.float32)
            test_x_metadata = np.array(test_x_metadata, dtype=np.float32)
            metadata_cols = [
                'meta_vertex_x', 'meta_vertex_y', 'meta_bounding_confidence', 'meta_bounding_importance_frac',
                'meta_dominant_blue', 'meta_dominant_green', 'meta_dominant_red',
                'meta_dominant_pixel_frac', 'meta_dominant_score', 'meta_label_score']
            train_meta_df = pd.DataFrame(train_x_metadata, columns=metadata_cols)
            test_meta_df = pd.DataFrame(test_x_metadata, columns=metadata_cols)
            train = pd.concat([train, train_meta_df], axis=1, sort=False)
            test = pd.concat([test, test_meta_df], axis=1, sort=False)
            self.numeric_cols.extend(metadata_cols)

        # --- feature engineering in https://www.kaggle.com/wrosinski/baselinemodeling ---
        if use_sentiment2 or use_metadata2 or use_text:
            smp = SentimentMetadataPreprocessor()
            smp.preprocess_sentiment_and_metadata(train, test)
            # It is necessary to fillna after df merged, because some PetID is missing.
            # if arch == 'nn':
            #     # nan handling...
            #     smp.train_sentiment_gr.fillna(0., inplace=True)
            #     smp.train_metadata_gr.fillna(0., inplace=True)
            #     smp.test_sentiment_gr.fillna(0., inplace=True)
            #     smp.test_metadata_gr.fillna(0., inplace=True)
            train = train.merge(
                smp.train_sentiment_gr, how='left', on='PetID')
            train = train.merge(
                smp.train_metadata_gr, how='left', on='PetID')
            train = train.merge(
                smp.train_metadata_desc, how='left', on='PetID')
            train = train.merge(
                smp.train_sentiment_desc, how='left', on='PetID')
            test = test.merge(
                smp.test_sentiment_gr, how='left', on='PetID')
            test = test.merge(
                smp.test_metadata_gr, how='left', on='PetID')
            test = test.merge(
                smp.test_metadata_desc, how='left', on='PetID')
            test = test.merge(
                smp.test_sentiment_desc, how='left', on='PetID')
            if use_sentiment2:
                sentiment2_cols = list(smp.train_sentiment_gr.columns.values)
                sentiment2_cols.remove('PetID')
                if arch == 'nn':
                    # nan handling...
                    for col in sentiment2_cols:
                        train[col].fillna(-1., inplace=True)
                        test[col].fillna(-1., inplace=True)
                self.numeric_cols.extend(sentiment2_cols)
            if use_metadata2:
                metadata2_cols = list(smp.train_metadata_gr.columns.values)
                metadata2_cols.remove('PetID')
                if arch == 'nn':
                    # nan handling...
                    for col in metadata2_cols:
                        train[col].fillna(-1., inplace=True)
                        test[col].fillna(-1., inplace=True)
                self.numeric_cols.extend(metadata2_cols)

            if use_text:
                text_columns = ['Description', 'metadata_annots_top_desc', 'sentiment_entities']
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.decomposition import SparsePCA, TruncatedSVD, LatentDirichletAllocation, NMF

                # n_components = 5  # version 1
                # text_methods = ['svd', 'nmf']  # version 1

                # n_components = 16  # version 2
                # text_methods = ['svd']  # version 2

                n_components_list = [16, 8, 8]  # version 3 (self-customized)
                text_methods = ['svd']  # version 2

                text_features = []
                # Generate text features:
                for i, n_components in zip(text_columns, n_components_list):
                    # Initialize decomposition methods:
                    print('generating features from: {}'.format(i))
                    # train.loc[:, i].fillna('<MISSING>', inplace=True)
                    # test.loc[:, i].fillna('<MISSING>', inplace=True)
                    train.loc[:, i].fillna('none', inplace=True)
                    test.loc[:, i].fillna('none', inplace=True)
                    x_text = np.concatenate([
                        train.loc[:, i].values, test.loc[:, i].values], axis=0)

                    # tfv = TfidfVectorizer()
                    tfv = TfidfVectorizer(min_df=2, max_features=None,
                                          strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)
                    # tfidf_col = TfidfVectorizer().fit_transform(X_text.loc[:, i].values)
                    tfidf_col = tfv.fit_transform(x_text)
                    if 'svd' in text_methods and n_components > 0:
                        svd_ = TruncatedSVD(
                            n_components=n_components, random_state=1337)
                        svd_col = svd_.fit_transform(tfidf_col)
                        svd_col = pd.DataFrame(svd_col)
                        svd_col = svd_col.add_prefix('SVD_{}_'.format(i))
                        text_features.append(svd_col)

                    if 'nmf' in text_methods and n_components > 0:
                        nmf_ = NMF(
                            n_components=n_components, random_state=1337)
                        nmf_col = nmf_.fit_transform(tfidf_col)
                        nmf_col = pd.DataFrame(nmf_col)
                        nmf_col = nmf_col.add_prefix('NMF_{}_'.format(i))
                        text_features.append(nmf_col)

                    add_length_feat = True  # added version 2
                    if add_length_feat:
                        length_col = 'Length_{}'.format(i)
                        if arch == 'nn':
                            scale_cols.append(length_col)

                        length_df = pd.DataFrame({length_col: [len(elem) for elem in x_text]})
                        text_features.append(length_df)

                # Combine all extracted features:
                text_features_df = pd.concat(text_features, axis=1, sort=False)
                print('text_features', type(text_features_df), text_features_df.shape)
                # Concatenate with main DF:
                train_text_df = text_features_df.iloc[:len(train)].reset_index(drop=True)
                test_text_df = text_features_df.iloc[len(train):].reset_index(drop=True)
                train = pd.concat([train, train_text_df], axis=1, sort=False)
                test = pd.concat([test, test_text_df], axis=1, sort=False)

                text_feat_cols = list(text_features_df.columns.values)
                self.numeric_cols.extend(text_feat_cols)

        if use_fasttext:
            if embedding_type_list is None:
                embedding_type_list = ['glove200d']
                # embedding_type_list = ['fasttext', 'glove200d']
        else:
            embedding_type_list = []

        for embedding_type in embedding_type_list:
            use_fasttext_cache = True
            # embedding_type = 'fasttext'  # 'fasttext'
            if embedding_type == 'glove50d':
                npz_filepath = './cache/glove-50d-6B.npz'
            elif embedding_type == 'glove200d':
                npz_filepath = './cache/glove-200d-6B.npz'
            elif embedding_type == 'fasttext':
                npz_filepath = './cache/crawl-300d-2M.npz'
            else:
                raise ValueError("[ERROR] Unexpected value embedding_type={}".format(embedding_type))

            t0 = perf_counter()
            if use_fasttext_cache and os.path.exists(npz_filepath):
                a = np.load(npz_filepath)
                vec_array = a['vec_array']
                vocab_list = list(a['vocab_list'])
            else:
                if embedding_type == 'glove50d':
                    filepath = f'{glove_dir}/glove.6B.50d.txt'
                    vocab_list, vec_array = construct_glove_vocab_list(filepath, hdim=50)
                elif embedding_type == 'glove200d':
                    filepath = f'{glove_dir}/glove.6B.200d.txt'
                    vocab_list, vec_array = construct_glove_vocab_list(filepath, hdim=200)
                else:
                    filepath = f'{glove_dir}/crawl-300d-2M.vec'
                    vocab_list, vec_array = construct_fasttext_vocab_list(filepath)
                os.makedirs(os.path.dirname(npz_filepath), exist_ok=True)
                np.savez_compressed(npz_filepath, vocab_list=vocab_list, vec_array=vec_array)

            t1 = perf_counter()
            print('fasttext construct_fasttext_vocab_list took {:.3} sec'.format(t1-t0))
            # out_dim_list = [64, 16]
            # fasttext_columns = ['metadata_annots_top_desc', 'sentiment_entities']  # 'Description'
            # out_dim_list = [64, 16, 16]
            # fasttext_columns = ['metadata_annots_top_desc', 'sentiment_entities', 'Description']  # 'Description'
            out_dim_list = [64]
            fasttext_columns = ['metadata_annots_top_desc']  # 'Description'
            # fasttext_columns = ['metadata_annots_top_desc', 'sentiment_entities', 'Description']  # 'Description'
            for i, out_dim in zip(fasttext_columns, out_dim_list):
                # Initialize decomposition methods:
                print('generating fasttext features from: {}'.format(i))
                train.loc[:, i].fillna('none', inplace=True)
                test.loc[:, i].fillna('none', inplace=True)
                x_fasttext = np.concatenate([
                    train.loc[:, i].values, test.loc[:, i].values], axis=0)
                t0 = perf_counter()
                h_fasttext = calc_fasttext_feature(
                    x_fasttext, vocab_list, vec_array,
                    out_dim=out_dim, col_name=i, method='tfidf', source=embedding_type)
                t1 = perf_counter()
                print('h_fasttext', h_fasttext.shape)
                print('fasttext calc_fasttext_feature for {} took {:.3} sec'.format(i, t1 - t0))
                fasttext_df = pd.DataFrame(h_fasttext)
                fasttext_df = fasttext_df.add_prefix(f'{embedding_type}_{i}_')

                # Concatenate with main DF:
                train_fasttext_df = fasttext_df.iloc[:len(train)].reset_index(drop=True)
                test_fasttext_df = fasttext_df.iloc[len(train):].reset_index(drop=True)
                train = pd.concat([train, train_fasttext_df], axis=1, sort=False)
                test = pd.concat([test, test_fasttext_df], axis=1, sort=False)

                fasttext_feat_cols = list(fasttext_df.columns.values)
                self.numeric_cols.extend(fasttext_feat_cols)

        os.makedirs('./cache', exist_ok=True)
        if use_tfidf:
            if animal_type is not None:
                cache_filepath = './cache/x_tfidf_svd{}_debug{}_animaltype{}.npz'.format(
                    tfidf_svd_components, int(debug), animal_type)
            else:
                cache_filepath = './cache/x_tfidf_svd{}_debug{}.npz'.format(
                    tfidf_svd_components, int(debug))
            if use_tfidf_cache and os.path.exists(cache_filepath):
                print('load from {}'.format(cache_filepath))
                train_x_tfidf_svd, test_x_tfidf_svd = load_npz(cache_filepath)
            else:
                print('create tfidf feature...')
                train_x_tfidf_svd, test_x_tfidf_svd = add_tfidf(train, test, tfidf_svd_components=tfidf_svd_components)
                save_npz(cache_filepath, (train_x_tfidf_svd, test_x_tfidf_svd))
                print('saved to {}'.format(cache_filepath))

            tfidf_cols = ['tfidf_{:04}'.format(i) for i in range(tfidf_svd_components)]
            train_tfidf_df = pd.DataFrame(train_x_tfidf_svd, columns=tfidf_cols)
            test_tfidf_df = pd.DataFrame(test_x_tfidf_svd, columns=tfidf_cols)
            train = pd.concat([train, train_tfidf_df], axis=1, sort=False)
            test = pd.concat([test, test_tfidf_df], axis=1, sort=False)
            self.numeric_cols.extend(tfidf_cols)

        if use_image_size:
            cache_filepath_train = './cache/x_image_size_debug{}_train.feather'.format(int(debug))
            cache_filepath_test = './cache/x_image_size_debug{}_test.feather'.format(int(debug))
            if os.path.exists(cache_filepath_train) and os.path.exists(cache_filepath_test):
                agg_train_imgs = read_feather(cache_filepath_train)
                agg_test_imgs = read_feather(cache_filepath_test)
            else:
                agg_train_imgs, agg_test_imgs = parse_image_size()
                agg_train_imgs.to_feather(cache_filepath_train)
                agg_test_imgs.to_feather(cache_filepath_test)
            train = train.merge(
                agg_train_imgs, how='left', on='PetID')
            test = test.merge(
                agg_test_imgs, how='left', on='PetID')
            image_size_cols = list(agg_train_imgs.columns)
            image_size_cols.remove('PetID')
            for col in image_size_cols:
                assert col in agg_test_imgs.columns
                if arch == 'nn':
                    # nan handling...
                    train[col].fillna(0., inplace=True)
                    test[col].fillna(0., inplace=True)
            self.numeric_cols.extend(image_size_cols)
            if arch == 'nn':
                scale_cols.extend(image_size_cols)

        if add_pred is not None:
            # add_pred = ['xlearn', 'nn', 'xgb']
            for model_name in add_pred:
                print(f'Adding {model_name} model pred...')
                filepath = 'predict_{}_train.csv'.format(model_name)
                df_train_pred = pd.read_csv(filepath)
                train = train.merge(
                    df_train_pred, how='left', on='PetID')
                filepath = 'predict_{}_test.csv'.format(model_name)
                df_test_pred = pd.read_csv(filepath)
                test = test.merge(
                    df_test_pred, how='left', on='PetID')
                train.rename({'y': f'y_{model_name}'}, axis=1, inplace=True)
                test.rename({'y': f'y_{model_name}'}, axis=1, inplace=True)
                self.numeric_cols.append(f'y_{model_name}')

        if arch == 'nn':
            # Neural Network preprocessing...
            # --- Numeric value processing ---
            print('numeric value preprocessing...')
            # There is no nan value, but this is just for make sure no nan exist.
            print('scale_cols', scale_cols)
            train_x_numeric = train[scale_cols].fillna(0).values.astype(np.float32)
            test_x_numeric = test[scale_cols].fillna(0).values.astype(np.float32)

            # --- MinMax scaling ---
            xmax = np.max(train_x_numeric, axis=0)
            xmin = np.min(train_x_numeric, axis=0)
            print('xmax', xmax)
            print('xmin', xmin)
            inds = xmax != xmin  # Non-zero indices
            train_x_numeric[:, inds] = (train_x_numeric[:, inds] - xmin[inds]) / (xmax[inds] - xmin[inds])
            test_x_numeric[:, inds] = (test_x_numeric[:, inds] - xmin[inds]) / (xmax[inds] - xmin[inds])
            train.loc[:, scale_cols] = train_x_numeric
            test.loc[:, scale_cols] = test_x_numeric

        add_lang = True
        if add_lang:
            train_lang_df, test_lang_df = process_lang_df(train, test)
            all_data['lang'] = np.concatenate([
                train_lang_df['lang'].values, test_lang_df['lang'].values], axis=0)

        # --- Category value processing ---
        if cat2num:
            # TODO: numeric cols append for lgbm, cat_cols override for nn...
            # convert category value to one-hot vector or other values.
            # cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
            #             'Vaccinated', 'Dewormed', 'Sterilized', 'State', 'FurLength', 'Health']
            # all_cat = all_data[cat_cols].astype('category')
            # all_cat_id = all_cat.apply(lambda x: x.cat.codes)

            # --- 1. category to one-hot vector ---
            # checked include State or not --> include State seems better.
            one_hot_cols = ['Type', 'Gender', 'Vaccinated',
                            'Dewormed', 'Sterilized', 'State', 'FurLength', 'Health']
            if add_lang:
                one_hot_cols.append('lang')
            one_hot_list = [pd.get_dummies(all_data[col]).values for col in one_hot_cols]
            one_hot_array = np.concatenate(one_hot_list, axis=1).astype(np.float32)
            self.remove_cols(one_hot_cols)

            # --- 2. breed ---
            # deal "Unspecified" and "Unknown" as same for 2nd breed
            # all_data['Breed2'][all_data['Breed2'] == 0] = 307
            # all_data.loc[all_data['Breed2'] == 0, 'Breed2'] = 307
            # is_mixed = (all_data['Breed2'] != 0) & (all_data['Breed2'] != 307)
            is_mixed = (all_data['Breed2'] != 0).astype(np.float32)[:, None]
            b1 = all_data['Breed1'].value_counts()
            major_breeds = b1[b1 >= 100].index.values  # 18 species remain
            # major_breeds = b1[b1 >= 1000].index.values  # 3 species remain
            print('major_breeds', major_breeds)

            def breed_onehot(x):
                if x not in major_breeds:
                    # rare breed
                    return len(major_breeds)
                else:
                    # major (non-rare) breed
                    breed_id = np.argwhere(x == major_breeds)[0, 0]
                    # return x
                    return breed_id

            b1r = all_data['Breed1'].apply(breed_onehot)
            b2r = all_data['Breed2'].apply(breed_onehot)
            breed_ones = np.eye(len(major_breeds) + 1, dtype=np.float32)
            # breed_array = (1.0 * breed_ones[b1r] + 0.7 * breed_ones[b2r]).astype(np.float32)
            breed_array = (1.0 * breed_ones[b1r] + 1.0 * breed_ones[b2r]).astype(np.float32)
            # self.remove_cols(['Breed1', 'Breed2'])

            # --- 3. color ---
            # 0 unspecified, 1 black, ... , 7 white.
            color_ones = np.eye(8)
            color1_onehot = color_ones[all_data['Color1'].values]
            color2_onehot = color_ones[all_data['Color2'].values]
            color3_onehot = color_ones[all_data['Color3'].values]
            # color_array = (1.0 * color1_onehot + 0.7 * color2_onehot + 0.5 * color3_onehot).astype(np.float32)
            color_array = (1.0 * color1_onehot + 1.0 * color2_onehot + 1.0 * color3_onehot).astype(np.float32)
            # self.remove_cols(['Color1', 'Color2', 'Color3'])

            x_cat2num_array = np.concatenate([one_hot_array, is_mixed, breed_array, color_array], axis=1)
            one_hot_cols = ['one_hot_{:04}'.format(i) for i in range(one_hot_array.shape[1])]
            is_mixed_cols = ['is_mixed_{:04}'.format(i) for i in range(is_mixed.shape[1])]
            breed_cols = ['breed_{:04}'.format(i) for i in range(breed_array.shape[1])]
            color_cols = ['color_{:04}'.format(i) for i in range(color_array.shape[1])]

            cat2num_cols = one_hot_cols + is_mixed_cols + breed_cols + color_cols
            train_cat2num_df = pd.DataFrame(x_cat2num_array[train_indices], columns=cat2num_cols)
            test_cat2num_df = pd.DataFrame(x_cat2num_array[test_indices], columns=cat2num_cols)
            train = pd.concat([train, train_cat2num_df], axis=1)
            test = pd.concat([test, test_cat2num_df], axis=1)
            if arch == 'nn':
                self.cat_cols = cat2num_cols
            else:
                self.numeric_cols.extend(cat2num_cols)
                # self.numeric_cols.extend(breed_cols + is_mixed_cols)
            print('one_hot_array', one_hot_array.shape,
                  'is_mixed', is_mixed.shape,
                  'breed_array', breed_array.shape,
                  'color_array', color_array.shape,
                  'x_cat2num_array', x_cat2num_array.shape, x_cat2num_array.dtype)
            self.num_cat_id = -1
        else:
            if add_lang:
                categorical_cols.append('lang')
                train['lang'] = 0
                test['lang'] = 0
                self.cat_cols.append('lang')
            if len(categorical_cols) > 0:
                all_cat = all_data[categorical_cols].astype('category')
                all_cat_id = all_cat.apply(lambda x: x.cat.codes)
                all_x_cat = all_cat_id.values.astype(np.int32)
                # all_cat_id[['Breed1', 'Breed2']].groupby(['Breed1', 'Breed2']).size()

                train_x_cat = all_x_cat[train_indices]
                test_x_cat = all_x_cat[test_indices]
                train.loc[:, categorical_cols] = train_x_cat
                test.loc[:, categorical_cols] = test_x_cat
                num_cat_id = np.max(all_x_cat, axis=0) + 1
                print('train_x_cat', train_x_cat.shape, 'test_x_cat', test_x_cat.shape, 'num_cat_id', num_cat_id)
                print('cat dtypes: ', train.loc[:, categorical_cols].dtypes)
                self.num_cat_id = num_cat_id

        if arch == 'xlearn':
            # for col in [self.cat_cols + self.numeric_cols]:
            for col in self.cat_cols:
                train.loc[:, col] = train.loc[:, col].values.astype(np.int64)
                test.loc[:, col] = test.loc[:, col].values.astype(np.int64)
        elif arch == 'nn':
            # for col in [self.cat_cols + self.numeric_cols]:
            for col in self.cat_cols:
                train.loc[:, col] = train.loc[:, col].values.astype(np.int32)
                test.loc[:, col] = test.loc[:, col].values.astype(np.int32)
        return train, test
        # return train_x_numeric, train_x_cat, target, test_x_numeric, test_x_cat, num_cat_id

    def preprocess_bert(self, train, test, num_extract_sentence=1, layer_indices=[-1, ], device=-1,
                        use_cache=True, animal_type=None):
        train_x_bert, test_x_bert = preprocess_bert(
            train, test, num_extract_sentence=num_extract_sentence, layer_indices=layer_indices, device=device,
            use_cache=use_cache, animal_type=animal_type)
        bs, num_sentence, hdim = train_x_bert.shape
        bs_test, num_sentence, hdim = test_x_bert.shape
        train_x_bert = np.reshape(train_x_bert, (bs, num_sentence * hdim))
        test_x_bert = np.reshape(test_x_bert, (bs, num_sentence * hdim))
        bert_cols = ['bert_{:05}'.format(i) for i in range(train_x_bert.shape[1])]
        train_bert_df = pd.DataFrame(train_x_bert, columns=bert_cols)
        test_bert_df = pd.DataFrame(test_x_bert, columns=bert_cols)
        train = pd.concat([train, train_bert_df], axis=1, sort=False)
        test = pd.concat([test, test_bert_df], axis=1, sort=False)
        self.numeric_cols.extend(bert_cols)
        return train, test

    def preprocess_image_cosine(self, train, test, train_type, test_type, arch='densenet'):
        # --- debug ---
        assert arch == 'densenet'
        train_x_image_full, test_x_image_full = preprocess_image_densenet(
            train, test, n_components=None)
        x_cute_image_array = preprocess_image_cute(device=0, arch=arch, batch_size=32, debug=False)
        d = CuteImageDataset(debug=False)
        labels_df = d.labels_df
        all_x_image_full = np.concatenate([train_x_image_full, test_x_image_full], axis=0)
        all_type = np.concatenate([train_type, test_type], axis=0)
        all_x_disthists = process_cosine_distance(
            all_x_image_full, all_type, x_cute_image_array, labels_df)
        train_x_disthists = all_x_disthists[:len(train)]
        test_x_disthists = all_x_disthists[len(train):]
        assert len(test_x_disthists) == len(test)
        # add numerical features and assign values to df...
        disthist_cols = ['image_disthist_{:05}'.format(i) for i in range(test_x_disthists.shape[1])]
        train_disthist_df = pd.DataFrame(train_x_disthists, columns=disthist_cols)
        test_disthist_df = pd.DataFrame(test_x_disthists, columns=disthist_cols)
        train = pd.concat([train, train_disthist_df], axis=1, sort=False)
        test = pd.concat([test, test_disthist_df], axis=1, sort=False)
        self.numeric_cols.extend(disthist_cols)
        return train, test
        # train_x_dists = process_cosine_distance(
        #     train_x_image_full, train_type, x_cute_image_array, labels_df)
        # test_x_dists = process_cosine_distance(
        #     test_x_image_full, test_type, x_cute_image_array, labels_df)

    def preprocess_image(self, train, test, num_image=2, device=-1, arch='vgg16',
                         n_components=None, use_cache=True, animal_type=None, method='pooling'):
        if arch == 'densenet':
            train_x_image, test_x_image = preprocess_image_densenet(
                train, test, n_components=n_components, method=method)
        else:
            train_x_image, test_x_image = preprocess_image(
                train, test, num_image=num_image, device=device, arch=arch,
                n_components=n_components, animal_type=animal_type, use_cache=use_cache,
                method=method)
        bs, num_image, hdim = train_x_image.shape
        bs_test, num_image, hdim = test_x_image.shape
        train_x_image = np.reshape(train_x_image, (bs, num_image * hdim))
        test_x_image = np.reshape(test_x_image, (bs_test, num_image * hdim))
        image_cols = ['image_{}_{:05}'.format(arch, i) for i in range(train_x_image.shape[1])]
        train_image_df = pd.DataFrame(train_x_image, columns=image_cols)
        test_image_df = pd.DataFrame(test_x_image, columns=image_cols)
        train = pd.concat([train, train_image_df], axis=1, sort=False)
        test = pd.concat([test, test_image_df], axis=1, sort=False)
        self.numeric_cols.extend(image_cols)
        return train, test

    def preprocess_image_det(self, train, test, num_image=1,
                             device=-1, arch='faster_rcnn_vgg16', use_cache=True,
                             n_components=None, animal_type=None):

        train_x_image, test_x_image = preprocess_image_det(
            train, test, num_image=num_image,
            device=device, arch=arch, use_cache=use_cache,
            n_components=n_components, animal_type=animal_type)
        bs, num_image, hdim = train_x_image.shape
        bs_test, num_image, hdim = test_x_image.shape

        method = 'animal_mean'
        if num_image > 1 and method == 'animal_mean':
            print('num_image', num_image)
            def calc_valid_mean(x_image):
                x_exist_animal = x_image[:, :, 1] > 0
                num_valid_images = np.sum(x_exist_animal, axis=1)
                x_image_mean = np.sum(np.where(x_exist_animal[:, :, None], x_image, 0), axis=1) / num_valid_images[:, None]
                return x_image_mean
            train_x_image = calc_valid_mean(train_x_image)
            test_x_image = calc_valid_mean(test_x_image)
            print('calc animal_mean done.', train_x_image.shape, test_x_image.shape)
        else:
            train_x_image = np.reshape(train_x_image, (bs, num_image * hdim))
            test_x_image = np.reshape(test_x_image, (bs_test, num_image * hdim))

        # # --- Try contraction by pearsonr ---  # this makes information leak only for checking maximum performance.
        # pearson_list = [pearson_corr(train_x_image[:, i], train.AdoptionSpeed.values)
        #                 for i in range(train_x_image.shape[1])]
        # pearson_arr = np.array(pearson_list)
        # inds = np.argsort(np.abs(pearson_arr))[::-1]
        # print('pearson_arr', pearson_arr[inds[:n_components]])
        # train_x_image = train_x_image[:, inds[:n_components]]
        # test_x_image = test_x_image[:, inds[:n_components]]

        image_cols = ['image_{:05}'.format(i) for i in range(train_x_image.shape[1])]
        train_image_df = pd.DataFrame(train_x_image, columns=image_cols)
        test_image_df = pd.DataFrame(test_x_image, columns=image_cols)
        train = pd.concat([train, train_image_df], axis=1, sort=False)
        test = pd.concat([test, test_image_df], axis=1, sort=False)
        self.numeric_cols.extend(image_cols)
        return train, test


def pearson_corr(x, y):
    x_diff = x - np.mean(x)
    y_diff = y - np.mean(y)
    denom = (np.sqrt(np.sum(x_diff ** 2)) * np.sqrt(np.sum(y_diff ** 2)))
    if denom < 1e-8:
        return 0
    else:
        return np.dot(x_diff, y_diff) / denom


# In[54]:


# --- pet image dataset ---


# In[57]:


import os

import chainer

class PetImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, petids, data_type, num_image=1, mode='chainercv'):
        self.petids = petids
        self.data_type = data_type
        self.num_image = num_image
        self.mode = mode

    def __len__(self):
        """return length of this dataset"""
        return len(self.petids) * self.num_image

    def get_example(self, i):
        """Return i-th data"""
        pet_index, image_index = divmod(i, self.num_image)
        filepath = '{}/{}_images/{}-{}.jpg'.format(
            pet_dir, self.data_type, self.petids[pet_index], image_index+1)
        if os.path.exists(filepath):
            if self.mode == 'chainercv':
                image = read_image(filepath)
            elif self.mode == 'keras':
                image = read_image_keras(filepath)
            else:
                raise ValueError("[ERROR] Unexpected value mode={}".format(self.mode))
            # print('image', type(image), image.dtype, image.shape)
            return image
        else:
            # print('[DEBUG] {} not exist'.format(filepath))
            return None


# ## Start script code
# 
# It is already very long... But above is the "library code" scripts.
# 
# Let's start actual calculation with some scripts from here!

# ### Densenet image feature extraction...
# 
# At first, calculate image feature and save it with DataFrame using feather.
# 
# The saved files (extracted image feature) are loaded every time I train different models.

# In[ ]:





# In[64]:


from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K

img_size = 256
batch_size = 256

import tensorflow as tf
#tf_device_str = "/gpu:0"  # It seems we face cudnn error with the latest version of docker image... It was fine during competition.
tf_device_str = "/cpu:0"  # it works but very slow...
with tf.device(tf_device_str):
    m = prepare_model_densenet()


# In[ ]:


import os
import numpy as np
from itertools import chain

import chainer
from chainer import cuda
from chainer.iterators import MultiprocessIterator, MultithreadIterator

from joblib import delayed, Parallel

import sys
import os

from sklearn.decomposition import PCA, TruncatedSVD
from tqdm import tqdm


def calc_x_image_array(petids, data_type, num_image, batch_size):
    image_dataset = PetImageDataset(petids, data_type, num_image, mode='keras')
    # iterator = MultiprocessIterator(
    #     image_dataset, batch_size, repeat=False, shuffle=False)
    iterator = MultithreadIterator(
        image_dataset, batch_size, repeat=False, shuffle=False)
    x_image_array = None
    current_index = 0
    for batch in tqdm(iterator, total=len(image_dataset) // batch_size):
        has_image_indices = np.argwhere(np.array([elem is not None for elem in batch]))[:, 0]
        image_list = [elem for elem in batch if elem is not None]
        #feats = model.predict(image_list)[0]
        feats = model.predict(np.array(image_list))
        if x_image_array is None:
            feat_dim = feats.shape[1]
            x_image_array = np.zeros((len(image_dataset), feat_dim), dtype=np.float32)
        x_image_array[has_image_indices + current_index] = feats
        current_index += batch_size
    return x_image_array


train, test, breeds, colors, states = prepare_df(debug)
num_image = 10
model = m

#train_x_image_array = calc_x_image_array(train['PetID'].values, 'train', num_image, batch_size)
#test_x_image_array = calc_x_image_array(test['PetID'].values, 'test', num_image, batch_size)

with tf.device(tf_device_str):
    train_x_image_array = calc_x_image_array(train['PetID'].values, 'train', num_image, batch_size)
    test_x_image_array = calc_x_image_array(test['PetID'].values, 'test', num_image, batch_size)


# In[60]:


train_image_feat_list = []
for i in tqdm(range(len(train))):
    pet_image_feat = 0
    pet_image_count = 0
    for j in range(num_image):
        index = i * num_image + j
        this_feat = train_x_image_array[index]
        if np.sum(this_feat) == 0:
            pet_image_feat += this_feat
        else:
            pet_image_feat += this_feat
            pet_image_count += 1
    pet_image_feat_mean = pet_image_feat / pet_image_count
    if i < 5:
        print(f'i, {i} pet_image_count {pet_image_count}')
    train_image_feat_list.append(pet_image_feat_mean)
    
test_image_feat_list = []
for i in tqdm(range(len(test))):
    pet_image_feat = 0
    pet_image_count = 0
    for j in range(num_image):
        index = i * num_image + j
        this_feat = test_x_image_array[index]
        if np.sum(this_feat) == 0:
            pet_image_feat += this_feat
        else:
            pet_image_feat += this_feat
            pet_image_count += 1
    pet_image_feat_mean = pet_image_feat / pet_image_count
    if i < 5:
        print(f'i, {i} pet_image_count {pet_image_count}')
    test_image_feat_list.append(pet_image_feat_mean)
    


# In[ ]:


features = {}
for i,pet_id in enumerate(train['PetID'].values):
    features[pet_id] = train_image_feat_list[i]
train_feats = pd.DataFrame.from_dict(features, orient='index')
train_feats.columns = [f'pic_{i}' for i in range(train_feats.shape[1])]
train_feats.head()


# In[ ]:


features = {}
for i,pet_id in enumerate(test['PetID'].values):
    features[pet_id] = test_image_feat_list[i]

test_feats = pd.DataFrame.from_dict(features, orient='index')
test_feats.columns = [f'pic_{i}' for i in range(test_feats.shape[1])]


# In[ ]:


train_feats = train_feats.reset_index()
train_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)

test_feats = test_feats.reset_index()
test_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)


# In[ ]:


test_feats.head()


# In[ ]:


train_feats.to_feather(f'train_image_densenet_{num_image}_1024.feather')
test_feats.to_feather(f'test_image_densenet_{num_image}_1024.feather')


# In[ ]:


import gc

del m
del train_image_feat_list
del test_image_feat_list
del train_x_image_array
del test_x_image_array
del features
del train_feats
del test_feats
gc.collect()


# > ## Main for GBT training...

# In[ ]:


# --- train lgbm ---


# In[ ]:


import argparse
from collections import Counter
from copy import deepcopy

import numpy
import pandas as pd
from distutils.util import strtobool

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import sys
import os


def permutation_augmentation(df, permute_cols_ratio=0.1, augment_scale=4):
    df_list = [df]

    bs, ndim = df.shape
    num_permute_cols = int(ndim * permute_cols_ratio)
    for i in range(augment_scale):
        df_tmp = deepcopy(df)
        col_indices = numpy.random.choice(ndim, num_permute_cols, replace=False)
        for col_index in col_indices:
            perm = numpy.random.permutation(bs)
            df_tmp.iloc[:, col_index] = df_tmp.iloc[:, col_index].values[perm]
        df_list.append(df_tmp)
    return pd.concat(df_list, axis=0)


def fit_xgb(train, test, pp, train_indices, val_indices,
            optr=None, predict_y=None, step=0, **kwargs):
    metric = 'rmse'
    xgb_params = {
        'eval_metric': metric,
        'seed': 1337,
        'eta': 0.0123,
        'tree_method': 'gpu_hist',
        'device': 'gpu',
        'silent': 1,
        'depth': 5,
        'min_child_weight': 1,
        'subsample': 0.9,
        # 'subsample': 0.8,
        'colsample_bytree': 0.85,
        # 'colsample_bylevel': 0.2,
        'lambda': 0.6,
        # 'alpha': 0.4
    }
    print('xgb_params', xgb_params)

    # Additional parameters:
    verbose_eval = 100
    early_stop = 300
    # num_rounds = 10000
    num_rounds = 2500
    # early_stop = 500
    # num_rounds = 60000

    target = pp.target
    # train_x_numeric = x_numeric[train_indices]
    train_target = target[train_indices]

    # val_x_numeric = x_numeric[val_indices]
    val_target = target[val_indices]

    feat_cols = pp.numeric_cols + pp.cat_cols
    X_tr = train.loc[train_indices, feat_cols].copy()
    if step == 0:
        print('feat_cols', feat_cols)
        print('X_tr before augment', X_tr.shape)

    augment_scale = 0
    if augment_scale > 0:
        X_tr = permutation_augmentation(
            X_tr, permute_cols_ratio=0.03,
            augment_scale=augment_scale)
        print('X_tr after augment', X_tr.shape)
    X_val = train.loc[val_indices, feat_cols].copy()
    # print('X_tr shape {} dtypes {}'.format(X_tr.shape, X_tr.dtypes.values))
    y_tr = numpy.tile(train_target[:, 0], augment_scale + 1)
    y_val = val_target[:, 0]

    d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
    d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)

    if step == 0:
        print('d_train')
        for i, (k, v) in enumerate(zip(d_train.feature_names, d_train.feature_types)):
            print(i, k, v)

    # https://stackoverflow.com/questions/45006341/xgboost-how-to-use-mae-as-objective-function
    def huber_approx_obj(preds, dtrain):
        # d = preds - dtrain.get_labels()  # remove .get_labels() for sklearn
        d = preds - dtrain.get_label()
        h = 1  # h is delta in the graphic
        scale = 1 + (d / h) ** 2
        scale_sqrt = numpy.sqrt(scale)
        grad = d / scale_sqrt
        hess = 1 / scale / scale_sqrt
        return grad, hess

    print('training XGB:')
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                      early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=xgb_params,
                      )  # obj=huber_approx_obj

    os.makedirs('tmp', exist_ok=True)
    # To release GPU memory, we need to delete object.
    # save & load is necessary for future prediction...
    # https://github.com/dmlc/xgboost/issues/3045
    best_score = model.best_score
    best_ntree_limit = model.best_ntree_limit
    model.save_model('tmp/xgb.model')
    model.__del__()
    model = xgb.Booster()
    model.load_model('tmp/xgb.model')
    model.best_ntree_limit = best_ntree_limit

    if optr is not None:
        with timer('predict'):
            val_y = model.predict(
                xgb.DMatrix(X_val, feature_names=X_val.columns),
                ntree_limit=model.best_ntree_limit)
        if predict_y is not None:
            predict_y[val_indices] = val_y
        with timer('optr.fit'):
            optr.fit(val_y, val_target)
        coefficients = optr.coefficients()

        pred_y1_rank = optr.predict(val_y, coefficients)
        # pred_y1_rank = optr.predict(pred_y1, [0.5, 2.2, 3.2, 3.3])
        score = cohen_kappa_score(pred_y1_rank, val_target,
                                  labels=numpy.arange(optr.num_class),
                                  weights='quadratic')
        print('optimized score', score, 'coefficients', coefficients,)
    else:
        print('optr is not set, skip optimizing threshold...')
        coefficients = None
        score = None
    log = {'valid_rmse': best_score}

    return model, coefficients, log, score


def fit_lgbm(train, test, pp, train_indices, val_indices,
             optr=None, predict_y=None, step=0, **kwargs):

    # metric = 'huber'  # 'rmse'
    metric = 'rmse'
    params = {'application': 'regression',
              'boosting': 'gbdt',
              'metric': metric,
              'num_leaves': 70,
              'max_depth': 9,
              'learning_rate': 0.01,
              'bagging_fraction': 0.85,
              'feature_fraction': 0.8,
              'min_split_gain': 0.02,
              'min_child_samples': 150,
              'min_child_weight': 0.02,
              'lambda_l2': 0.0475,
              'verbosity': -1,
              'data_random_seed': 17,
              # 'device': 'gpu',  # need to compile...
              # 'gpu_device_id': 0
              }

    # Additional parameters:
    # early_stop = 500
    early_stop = 300
    verbose_eval = 100
    num_rounds = 10000

    target = pp.target
    # train_x_numeric = x_numeric[train_indices]
    train_target = target[train_indices]

    # val_x_numeric = x_numeric[val_indices]
    val_target = target[val_indices]

    feat_cols = pp.numeric_cols + pp.cat_cols
    X_tr = train.loc[train_indices, feat_cols]
    if step == 0:
        print('feat_cols', feat_cols)
        print('X_tr before augment', X_tr.shape)
    augment_scale = 0
    if augment_scale > 0:
        X_tr = permutation_augmentation(
            X_tr, permute_cols_ratio=0.15,
            augment_scale=augment_scale)
        print('X_tr after augment', X_tr.shape)
    X_val = train.loc[val_indices, feat_cols]
    # print('X_tr shape {} dtypes {}'.format(X_tr.shape, X_tr.dtypes.values))
    y_tr = numpy.tile(train_target[:, 0], augment_scale + 1)
    y_val = val_target[:, 0]
    d_train = lgb.Dataset(X_tr, label=y_tr, categorical_feature=pp.cat_cols)
    d_valid = lgb.Dataset(X_val, label=y_val, categorical_feature=pp.cat_cols)
    watchlist = [d_train, d_valid]

    print('training LGB:')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)

    if optr is not None:
        with timer('predict'):
            val_y = model.predict(X_val, num_iteration=model.best_iteration)
        if predict_y is not None:
            predict_y[val_indices] = val_y
        with timer('optr.fit'):
            optr.fit(val_y, val_target)
        coefficients = optr.coefficients()

        pred_y1_rank = optr.predict(val_y, coefficients)
        # pred_y1_rank = optr.predict(pred_y1, [0.5, 2.2, 3.2, 3.3])
        score = cohen_kappa_score(pred_y1_rank, val_target,
                                  labels=numpy.arange(optr.num_class),
                                  weights='quadratic')
        print('optimized score', score, 'coefficients', coefficients,)
    else:
        print('optr is not set, skip optimizing threshold...')
        coefficients = None
        score = None

    log = {'training_rmse': model.best_score['training'][metric],
           'valid_rmse': model.best_score['valid_1'][metric]}
    return model, coefficients, log, score


def fit_cb(train, test, pp, train_indices, val_indices,
           optr=None, predict_y=None, step=0, **kwargs):
    # metric = 'huber'  # 'rmse'
    metric = 'RMSE'  # 'rmse'
    params = {
        'loss_function': metric,
        # 'learning_rate': 0.01,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 9,  # 1 ~ 9
        'iterations': 10000,
        'od_type': 'Iter',
        'od_wait': 300,  # early stopping
        'random_seed': 1337,
        'task_type': 'GPU',
    }
    params['devices'] = '0'

    target = pp.target
    train_target = target[train_indices]
    val_target = target[val_indices]

    feat_cols = pp.numeric_cols + pp.cat_cols
    X_tr = train.loc[train_indices, feat_cols]
    if step == 0:
        print('feat_cols', feat_cols)
        print('X_tr before augment', X_tr.shape)

    augment_scale = 0
    if augment_scale > 0:
        X_tr = permutation_augmentation(
            X_tr, permute_cols_ratio=0.15,
            augment_scale=augment_scale)
        if step == 0:
            print('X_tr after augment', X_tr.shape)
    X_val = train.loc[val_indices, feat_cols]
    # print('X_tr shape {} dtypes {}'.format(X_tr.shape, X_tr.dtypes.values))
    y_tr = numpy.tile(train_target[:, 0], augment_scale + 1)
    y_val = val_target[:, 0]

    categorical_features_indices = [
        X_tr.columns.get_loc(cat_col) for cat_col in pp.cat_cols]
    model = cb.CatBoostRegressor(**params)
    print('training CatBoost:')
    model.fit(X_tr, y_tr, cat_features=categorical_features_indices,
              eval_set=(X_val, y_val), plot=False,
              use_best_model=True, metric_period=100, early_stopping_rounds=300)

    if optr is not None:
        with timer('predict'):
            val_y = model.predict(X_val)
        if predict_y is not None:
            predict_y[val_indices] = val_y
        with timer('optr.fit'):
            optr.fit(val_y, val_target)
        coefficients = optr.coefficients()

        pred_y1_rank = optr.predict(val_y, coefficients)
        # pred_y1_rank = optr.predict(pred_y1, [0.5, 2.2, 3.2, 3.3])
        score = cohen_kappa_score(pred_y1_rank, val_target,
                                  labels=numpy.arange(optr.num_class),
                                  weights='quadratic')
        print('optimized score', score, 'coefficients', coefficients,)
    else:
        print('optr is not set, skip optimizing threshold...')
        coefficients = None
        score = None
    log = {'training_rmse': model.best_score_['learn'][metric],
           'valid_rmse': model.best_score_['validation_0'][metric]}
    return model, coefficients, log, score


def main_gbm(debug, device, use_bert, use_image, num_image,
             use_tfidf=False, animal_type=None,
             cat2num=True, use_cat=True, image_type='clf',
             arch='lgbm', fold=4, use_fasttext=True, embedding_type_list=None,
             postfix='', use_cosine_dist=False):
    """

    Args:
        debug:
        device:
        use_bert:
        use_image:
        num_image:
        use_tfidf:
        animal_type (int): if specified, only train specific animal_type = 'dog' or 'cat'.
            If Negative value, both animals are trained same time.
        cat2num (bool): If True categorical values are converted to numeric values.
        use_cat (bbol): Use category specific Linear layer or not.
        mode (str): Training mode type.
            normal - normal training, regression of AdoptionSpeed.
            mean - regression of mean AdoptionSpeed on RescuerID.

    Returns:

    """
    num_class = 5
    print('debug', debug)
    print('Train data_type {}'.format(animal_type))

    if animal_type < 0:
        animal_type = None
    use_cat = not cat2num

    # --- dataset ---

    train, test, breeds, colors, states = prepare_df(debug, animal_type=animal_type)
    train_type = train['Type'].values.copy()
    test_type = test['Type'].values.copy()
    pp = Preprocessor(arch=arch)
    if arch == 'xlearn':
        train, test = pp.preprocess(
            train, test, breeds, colors, states, debug=debug, use_tfidf=use_tfidf,
            use_metadata=False, use_sentiment=False, use_gdp=False,
            use_rescuer_id_count=True, use_name_feature=True, use_target_encoding=False,
            cat2num=cat2num, animal_type=animal_type, use_tfidf_cache=True,
            tfidf_svd_components=16, num_sentiment_text=0,
            use_sentiment2=True, use_metadata2=True, use_text=False, use_fasttext=use_fasttext,
            add_pred=None, use_image_size=False, arch=arch, use_custom_boolean_feature=True,
            embedding_type_list=embedding_type_list)
    else:
        train, test = pp.preprocess(
            train, test, breeds, colors, states, debug=debug, use_tfidf=use_tfidf,
            use_metadata=False, use_sentiment=False, use_gdp=True,
            use_rescuer_id_count=True, use_name_feature=True, use_target_encoding=False,
            cat2num=cat2num, animal_type=animal_type, use_tfidf_cache=True,
            tfidf_svd_components=16, num_sentiment_text=0,
            use_sentiment2=True, use_metadata2=True, use_text=True, use_fasttext=use_fasttext,
            use_image_size=True, arch=arch, add_pred=None, use_custom_boolean_feature=False,
            embedding_type_list=embedding_type_list)
    target = pp.target

#     if use_bert:
#         print('preprocess bert feature...')
#         train, test = pp.preprocess_bert(
#             train, test, num_extract_sentence=2, layer_indices=[-1, ], device=device,
#             use_cache=True, animal_type=animal_type)
#     else:
#         print('skip bert feature')
#         # train_x_bert, test_x_bert = None, None

    if use_image:
        print('preprocess image feature... {}'.format(image_type))
        if image_type == 'clf':
            clf_arch = 'densenet'
            # clf_arch = 'vgg16'
            # clf_arch = 'seresnext50'
            n_components = 32
            print(f'using clf_arch {clf_arch}, n_components {n_components}')
            train, test = pp.preprocess_image(
                train, test, num_image=num_image, device=device, arch=clf_arch,
                n_components=n_components, animal_type=animal_type, use_cache=True,
                method='svd')
        elif image_type == 'det':
            det_arch = 'faster_rcnn_vgg16'  # fpn50
            n_components = 32  # 506 was worse...
            train, test = pp.preprocess_image_det(
                train, test, num_image=num_image, device=device, arch=det_arch, use_cache=True,
                n_components=n_components)
        else:
            raise ValueError("[ERROR] Unexpected value image_type={}".format(image_type))
    else:
        print('skip image feature...')
        train_x_image, test_x_image = None, None

    if use_cosine_dist:
        clf_arch = 'densenet'
        print('use_cosine_dist, {}'.format(clf_arch))
        train, test = pp.preprocess_image_cosine(
            train, test, train_type, test_type, arch=clf_arch)

    print('fillna -1')
    train = train.fillna(-1)
    test = test.fillna(-1)

    # --- Setup CV ---
    num_split = fold
    kfold_method = 'group'
    # kfold_method = 'stratified'
    random_state = 1337

    if kfold_method == 'stratified':
        print('StratifiedKFold...')
        kf = StratifiedKFold(n_splits=num_split, random_state=random_state, shuffle=True)
        # fold_splits = kf.split(train, target)
        fold_splits = kf.split(numpy.arange(len(train)), target)
    elif kfold_method == 'group':
        print('GroupKFold...')
        kf = GroupKFold(num_split)
        # kf.random_state = 42
        groups = train['RescuerID'].astype('category').cat.codes.values
        fold_splits = kf.split(numpy.arange(len(train)), target, groups)
    else:
        raise ValueError("[ERROR] Unexpected value kfold_method={}".format(kfold_method))

    optr = OptimizedRounder(
        num_class=num_class,
        # method='differential_evolution'
        method='nelder-mead')

    regressor_list = []
    coefficients_list = []
    log_list = []
    score_list = []

    # target is (num_rows, 1), predict_y is (num_rows,)
    predict_y = numpy.ones((len(target),), dtype=numpy.float32) * -1
    # std_list = calc_std_list(train_x_numeric, train_x_cat, train_x_bert, train_x_image)
    xlearn_dataset = None
    if arch == 'lgbm':
        fit_fn = fit_lgbm
    elif arch == 'xgb':
        fit_fn = fit_xgb
    elif arch == 'cb':
        fit_fn = fit_cb
    elif arch == 'xlearn':
        fit_fn = fit_xlearn
        # --- convert to dataframe to series... ---
        feat_cols = pp.numeric_cols + pp.cat_cols
        xlearn_dataset = XlearnDataset(
            train.loc[:, feat_cols + ['AdoptionSpeed']],
            test.loc[:, feat_cols],
        )
        print('xlearn_dataset', xlearn_dataset)
    else:
        raise ValueError("[ERROR] Unexpected value arch={}".format(arch))

    for k, (train_indices, val_indices) in enumerate(fold_splits):
        print('---- {} fold / {} ---'.format(k, num_split))
        regressor, coefficients, log, score = fit_fn(
            train, test, pp, train_indices, val_indices, optr=optr,
            predict_y=predict_y, step=k, xlearn_dataset=xlearn_dataset)
        regressor_list.append(regressor)
        coefficients_list.append(coefficients)
        log_list.append(log)
        score_list.append(score)

    log_df = pd.DataFrame(log_list)
    print('log_df mean\n{}'.format(log_df.mean()))
    if score_list[0] is not None:
        opt_score_mean = numpy.array(score_list).mean()
        print('opt_score_mean', opt_score_mean, 'score_list', score_list)

    print('Number of un-predicted example: ', numpy.sum(predict_y <= -1))
    with timer('optr.fit'):
        optr.fit(predict_y, target)
    coefficients = optr.coefficients()
    pred_y1_rank = optr.predict(predict_y, coefficients)
    score = cohen_kappa_score(pred_y1_rank, target,
                              labels=numpy.arange(optr.num_class),
                              weights='quadratic')
    print('Total: optimized score', score, 'coefficients', coefficients, )

    # --- create submission ---
    flag_create_submission = True
    if flag_create_submission:
        feat_cols = pp.numeric_cols + pp.cat_cols
        # print('feat_cols', feat_cols)

        test_dataset = test.loc[:, feat_cols]
        if arch == 'lgbm':
            with timer('test predict'):
                test_predict_list = [reg.predict(test_dataset, num_iteration=reg.best_iteration)
                                     for reg in regressor_list]
        elif arch == 'xgb':
            test_dataset = xgb.DMatrix(test_dataset, feature_names=test_dataset.columns)
            with timer('test predict'):
                test_predict_list = [reg.predict(test_dataset, ntree_limit=reg.best_ntree_limit)
                                     for reg in regressor_list]
        elif arch == 'cb':
            with timer('test predict'):
                test_predict_list = [reg.predict(test_dataset)
                                     for reg in regressor_list]
        else:
            assert arch == 'xlearn'
            test_predict_list = [reg.predict(xlearn_dataset.test_ffm_series)
                                 for reg in regressor_list]

        print('train counter', Counter(target[:, 0]))
        test_predict_mean = numpy.mean(numpy.array(test_predict_list), axis=0)
        print('test_predict_mean', test_predict_mean.shape)

        test_id = test['PetID']
        # --- 0. raw float predictions ---
        predict_df = pd.DataFrame({'PetID': train['PetID'], 'y': predict_y, 't': target.ravel()})
        predict_df.to_csv('predict_{}{}_train.csv'.format(arch, postfix), index=False)
        print(f'predict_{arch}{postfix}_train.csv created.')
        predict_df = pd.DataFrame({'PetID': test_id, 'y': test_predict_mean})
        predict_df.to_csv('predict_{}{}_test.csv'.format(arch, postfix), index=False)
        print(f'predict_{arch}{postfix}_test.csv created.')

        # --- 1. mean coefficients ---
        coefficients_mean = numpy.mean(numpy.array(coefficients_list), axis=0)
        print('coefficients_mean', coefficients_mean)
        test_predictions = optr.predict(test_predict_mean, coefficients_mean).astype(int)
        print('test_predictions counter', Counter(test_predictions))
        submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
        submission.to_csv(f'submission_mean_{arch}{postfix}.csv', index=False)
        print(f'submission_mean_{arch}{postfix}.csv created.')

        # --- 2. validation all coefficients ---
        print('coefficients from all validation ', coefficients)
        test_predictions = optr.predict(test_predict_mean, coefficients).astype(int)
        print('test_predictions counter', Counter(test_predictions))
        submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
        submission.to_csv(f'submission_valcoef_{arch}{postfix}.csv', index=False)
        print(f'submission_valcoef_{arch}{postfix}.csv created.')

        coefficients3 = coefficients.copy()
        coefficients3[0] = 1.66
        coefficients3[1] = 2.13
        coefficients3[3] = 2.85
        test_predictions = optr.predict(test_predict_mean, coefficients3).astype(int)
        submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
        submission.to_csv(f'submission_valcoef3_{arch}{postfix}.csv', index=False)
        print('test_predictions counter', Counter(test_predictions))
        print(f'submission_valcoef3_{arch}{postfix}.csv created.')

        # --- 3. same histgram with train... ---
        test_predictions = optr.fit_and_predict_by_histgram(test_predict_mean, target)
        print('coefficients to align with train target', optr.coefficients())
        print('test_predictions counter', Counter(test_predictions))
        submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
        submission.to_csv(f'submission_{arch}{postfix}.csv', index=False)
        print(f'submission_{arch}{postfix}.csv created.')


# In[ ]:


import argparse
from collections import Counter
from copy import deepcopy

import numpy
import pandas as pd
from distutils.util import strtobool

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

import lightgbm as lgb

import sys
import os

device = 0
epoch = 60
print('debug', debug)
print('epoch', epoch)
#print('args', vars(args))

num_class = 5
use_bert = False
use_image = True
num_image = 1
out = 'results/tmp'
animal_type = None

use_tfidf=False
# cat2num=True
use_cat=True
image_type='clf'
mode='normal'
# arch = 'lgbm'
batchsize=256


# In[ ]:


num_class = 5
print('debug', debug)
print('Train data_type {}'.format(animal_type))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# train_xlearn


# In[ ]:


import os
import shutil

import numpy
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import xlearn as xl
from sklearn.metrics import cohen_kappa_score
'''
https://www.kaggle.com/mpearmain/pandas-to-libffm
Another CTR comp and so i suspect libffm will play its part, after all it is an atomic bomb for this kind of stuff.
A sci-kit learn inspired script to convert pandas dataframes into libFFM style data.

The script is fairly hacky (hey thats Kaggle) and takes a little while to run a huge dataset.
The key to using this class is setting up the features dtypes correctly for output (ammend transform to suit your needs)

'''


class FFMFormatPandas:
    """
    Convert pandas DataFrame into libffm format (consists of "field:feature:value") used in xlearn.
    DataFrame dtype 'i' and 'O' are treated as categorical value (different feature for each value).
    'f' are treated as value (same feature for each value).
    """
    def __init__(self):
        self.field_index_ = None
        self.feature_index_ = None
        self.y = None

    def fit(self, df, y=None):
        self.y = y
        df_ffm = df[df.columns.difference([self.y])]
        if self.field_index_ is None:
            self.field_index_ = {col: i for i, col in enumerate(df_ffm)}

        if self.feature_index_ is not None:
            last_idx = max(list(self.feature_index_.values()))

        if self.feature_index_ is None:
            self.feature_index_ = dict()
            last_idx = 0

        for col in df.columns:
            if df[col].dtype.kind in ['O', 'i']:
                print(f'{col} is treated as categorical...')
                vals = df[col].unique()
                for val in vals:
                    if pd.isnull(val):
                        continue
                    name = '{}_{}'.format(col, val)
                    if name not in self.feature_index_:
                        self.feature_index_[name] = last_idx
                        last_idx += 1
            else:
                print(f'{col} is treated as float value...')
            self.feature_index_[col] = last_idx
            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, row, t):
        ffm = []
        if self.y != None and self.y in row.index:
            ffm.append(str(row.loc[row.index == self.y][0]))
        if self.y is None:
            ffm.append(str(0))

        for col, val in row.loc[row.index != self.y].to_dict().items():
            col_type = t[col]
            if isinstance(val, float):
                val_int = int(val)
            name = '{}_{}'.format(col, val_int)
            # ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            if col_type.kind == 'O':
                ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            elif col_type.kind == 'i':
                ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            elif col_type.kind == 'f':
                ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
                # print('col_type.kind == f, append', '{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
        return ' '.join(ffm)

    def transform(self, df):
        t = df.dtypes.to_dict()
        return pd.Series({idx: self.transform_row_(row, t) for idx, row in df.iterrows()})


class XlearnDataset(object):
    def __init__(self, train_df, test_df, ycol='AdoptionSpeed'):
        self.train_df = train_df
        self.test_df = test_df
        self.ffm_format = FFMFormatPandas()

        all_df = pd.concat([train_df, test_df], axis=0, sort=False)
        all_df.reset_index(inplace=True)
        all_df[ycol].fillna(-1, inplace=True)
        ffm_series = self.ffm_format.fit_transform(all_df, y=ycol)
        assert len(ffm_series) == len(all_df), f'ffm_series {len(ffm_series)}, all_df {len(all_df)}'
        self.train_ffm_series = ffm_series.iloc[:len(train_df)]
        self.test_ffm_series = ffm_series.iloc[len(train_df):]

    def get_train(self, indices=None):
        if indices is None:
            return self.train_ffm_series
        else:
            return self.train_ffm_series[indices]


class XlearnModel(object):
    def __init__(self, params, method='ffm', model_filepath='tmp/model.out',
                 model_txt_filepath=None):  # 'tmp/model.txt'
        print(f'XlearnModel method={method}')
        self.params = params
        self.model_filepath = model_filepath
        self.model_txt_filepath = model_txt_filepath
        if method == 'ffm':
            self.model = xl.create_ffm()
        elif method == 'fm':
            self.model = xl.create_fm()
        else:
            raise NotImplementedError

    def fit(self, train_ffm_series, val_ffm_series, train_filepath='tmp/train.txt', val_filepath='tmp/val.txt',
            use_cache=False):
        train_ffm_series.to_csv(train_filepath, index=False)
        val_ffm_series.to_csv(val_filepath, index=False)

        self.model.setTrain(train_filepath)  # Training data
        self.model.setValidate(val_filepath)  # Validation data

        if self.model_txt_filepath is not None:
            self.model.setTXTModel(self.model_txt_filepath)
        print('fit start...')
        self.model.fit(self.params, self.model_filepath)
        print('fit end...')
        return self

    def predict(self, ffm_series, filepath='tmp/test.txt', out_filepath='tmp/output.txt'):
        # self.create_ffm_series(df, filepath)
        ffm_series.to_csv(filepath, index=False)
        # Prediction task
        self.model.setTest(filepath)  # Test data
        self.model.predict(self.model_filepath, out_filepath)
        y = pd.read_csv(out_filepath, header=None).values[:, 0]
        return y


def fit_xlearn(train, test, pp, train_indices, val_indices,
           optr=None, predict_y=None, step=0, xlearn_dataset=None, **kwargs):
    # param:
    #  0. binary classification
    #  1. learning rate: 0.2
    #  2. regular lambda: 0.002
    #  3. evaluation metric: accuracy
    params = {'task': 'reg', 'lr': 0.20,
              'lambda': 0.00002, 'metric': 'rmse',
              'opt': 'adagrad', 'k': 16,
              'init': 0.50,
              # 'epoch': 100,
              # 'stop_window': 5
              }  # adagrad

    model = XlearnModel(params)

    target = pp.target
    val_target = target[val_indices]

    feat_cols = pp.numeric_cols + pp.cat_cols
    feat_y_cols = feat_cols + ['AdoptionSpeed']
    print('feat_cols', feat_cols)
    X_tr = xlearn_dataset.train_ffm_series[train_indices]
    X_val = xlearn_dataset.train_ffm_series[val_indices]
    model.fit(X_tr, X_val)

    if optr is not None:
        with timer('predict'):
            # Start to predict
            # The output result will be stored in output.txt
            val_y = model.predict(X_val)
        if predict_y is not None:
            predict_y[val_indices] = val_y
        with timer('optr.fit'):
            optr.fit(val_y, val_target)
        coefficients = optr.coefficients()

        pred_y1_rank = optr.predict(val_y, coefficients)
        # pred_y1_rank = optr.predict(pred_y1, [0.5, 2.2, 3.2, 3.3])
        score = cohen_kappa_score(pred_y1_rank, val_target,
                                  labels=numpy.arange(optr.num_class),
                                  weights='quadratic')
        print('optimized score', score, 'coefficients', coefficients,)
    else:
        print('optr is not set, skip optimizing threshold...')
        coefficients = None
        score = None
    # log = {'training_rmse': model.best_score_['learn'][metric],
    #        'valid_rmse': model.best_score_['validation_0'][metric]}
    log = {'dummy': 0}
    return model, coefficients, log, score


# In[ ]:


if debug:
    fold = 3
else:
    fold = 10


# In[ ]:


# --- train lgbm ---
# use_fasttext takes time.
main_gbm(debug, device, use_bert, use_image, num_image,
         use_tfidf=False, animal_type=-1,
         cat2num=False, use_cat=True, image_type='clf',
         arch='lgbm', fold=10, use_fasttext=False)


# In[ ]:


gc.collect()


# In[ ]:


# --- train cb ---
main_gbm(debug, device, use_bert, use_image, num_image,
         use_tfidf=False, animal_type=-1,
         cat2num=False, use_cat=True, image_type='clf',
         arch='cb', fold=10, use_fasttext=True)


# In[ ]:


gc.collect()


# In[ ]:


# --- train xgb ---
main_gbm(debug, device, use_bert, use_image, num_image,
         use_tfidf=False, animal_type=-1,
         cat2num=True, use_cat=True, image_type='clf',
         arch='xgb', fold=fold, use_fasttext=True)


# In[ ]:


gc.collect()


# # Xlearn

# In[ ]:


# --- train xlearn ---
os.makedirs('tmp', exist_ok=True)
main_gbm(debug, device, use_bert, use_image, num_image,
         use_tfidf=False, animal_type=-1,
         cat2num=False, use_cat=True, image_type='clf',
         arch='xlearn', fold=10, use_fasttext=False)


# In[ ]:


gc.collect()


# # Chainer xDeepFM

# In[ ]:


import argparse
from collections import Counter

import numpy
import pandas as pd
from itertools import chain

from chainer import functions as F, functions
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.optimizer_hooks import WeightDecay
from chainer.training import extensions as E

from distutils.util import strtobool

from chainer.training.extensions import observe_lr
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, GroupKFold


# from chainer_chemistry.datasets import NumpyTupleDataset
import sys
import os

def lrelu05(x):
    return functions.leaky_relu(x, slope=0.05)


def lrelu20(x):
    return functions.leaky_relu(x, slope=0.20)


def setup_predictor(num_cat_id=None, dropout_ratio=0.1, use_bn=False, use_sn=False,
                    numeric_hidden_dim=96, embed_dim=10, bert_hidden_dim=32,
                    image_hidden_dim=96, mlp_hidden_dim=128, mlp_n_layers=6, activation=lrelu05,
                    cat_hidden_dim=32, use_residual=False, out_dim=1, use_set=False,
                    use_xdeepfm=True):
    print('setup_predictor num_cat_id {} dropout_ratio {}, use_bn {}, use_sn {} embed_dim {}'
          .format(num_cat_id, dropout_ratio, use_bn, use_sn, embed_dim))
    kwargs = dict(dropout_ratio=dropout_ratio, use_bn=use_bn,
                  numeric_hidden_dim=numeric_hidden_dim, embed_dim=embed_dim,
                  bert_hidden_dim=bert_hidden_dim,
                  image_hidden_dim=image_hidden_dim, mlp_hidden_dim=mlp_hidden_dim,
                  mlp_n_layers=mlp_n_layers, activation=activation,
                  cat_hidden_dim=cat_hidden_dim, use_residual=use_residual,
                  out_dim=out_dim)
    if use_xdeepfm:
        print('BlendNetXDeepFM built!')
        predictor = BlendNetXDeepFM(num_cat_id, use_sn=use_sn, weight_tying=True, **kwargs)
    else:
        if use_sn:
            if use_set:
                predictor = SetSNBlendNet(num_cat_id, **kwargs)
                # predictor = SNBlendNet(num_cat_id, **kwargs)
            else:
                predictor = SNBlendNet(num_cat_id, **kwargs)
        else:
            if use_set:
                predictor = SetBlendNet(num_cat_id, **kwargs)
            else:
                predictor = BlendNet(num_cat_id, **kwargs)
    return predictor


def calc_std_list(x_numeric, x_cat, x_bert, x_image):
    std_list = [None, None, None, None]
    if x_numeric is not None:
        std_list[0] = numpy.std(x_numeric, axis=0, dtype=numpy.float32)
    if x_cat is not None and x_cat.dtype is not numpy.int32:
        std_list[1] = numpy.std(x_cat, axis=0, dtype=numpy.float32)
    if x_bert is not None:
        bs, num_sent, ndim = x_bert.shape
        x_bert_2dim = numpy.reshape(x_bert, (bs * num_sent, ndim))
        std_list[2] = numpy.std(x_bert_2dim, axis=0, dtype=numpy.float32)
    if x_image is not None:
        bs, num_image, ndim = x_image.shape
        x_image_2dim = numpy.reshape(x_image, (bs * num_image, ndim))
        std_list[3] = numpy.std(x_image_2dim, axis=0, dtype=numpy.float32)
    return std_list


def fit_nn(x_numeric, x_cat, target, num_cat_id, train_indices, val_indices,
        x_bert=None, x_image=None, batchsize=256, device=-1, epoch=20,
        out='./results/tmp', debug=False, optr=None, converter=None, decay_rate=1e-4,
        use_sn=False, dropout_ratio=0.1, use_bn=False, use_residual=False,
        check_error=False, mode='normal', train_df=None, predict_y=None):
    assert converter is not None
    if debug:
        epoch = 2
    # Set up the iterators.
    train_x_numeric = x_numeric[train_indices]
    train_target = target[train_indices]

    val_x_numeric = x_numeric[val_indices]
    val_target = target[val_indices]

    extra_train_features = []
    extra_val_features = []

    use_cat = (x_cat is not None)
    use_bert = (x_bert is not None)
    use_image = (x_image is not None)
    if use_cat:
        extra_train_features.append(x_cat[train_indices])
        extra_val_features.append(x_cat[val_indices])
    if use_bert:
        train_x_bert = x_bert[train_indices]
        val_x_bert = x_bert[val_indices]
        extra_train_features.append(train_x_bert)
        extra_val_features.append(val_x_bert)
    if use_image:
        train_x_image = x_image[train_indices]
        val_x_image = x_image[val_indices]
        extra_train_features.append(train_x_image)
        extra_val_features.append(val_x_image)

    train = NumpyTupleDataset(*([train_x_numeric] + extra_train_features + [train_target]))
    valid = NumpyTupleDataset(*([val_x_numeric] + extra_val_features + [val_target]))
    if mode == 'mean':
        print('creating RescuerIDMeanDataset...')
        # train = RescuerIDMeanDataset(train_df.iloc[train_indices].copy(), train, mode='train')
        # valid = RescuerIDMeanDataset(train_df.iloc[val_indices].copy(), valid, mode='eval')
        train = RescuerIDMeanDataset(train_df.iloc[train_indices].copy(), train, mode='eval')
        valid = RescuerIDMeanDataset(train_df.iloc[val_indices].copy(), valid, mode='eval')

    train_iter = iterators.SerialIterator(train, batchsize)
    valid_iter = iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)

    # Set up the regressor.
    image_encode = False
    print('image_encode', image_encode)

    metrics_fun = {'mae': F.mean_absolute_error}
    out_dim = 1
    kwargs = dict(
        # activation=lrelu20, cat_hidden_dim=64, image_hidden_dim=32,
        # mlp_n_layers=6, mlp_hidden_dim=512, numeric_hidden_dim=96,
        activation=lrelu20, cat_hidden_dim=32, image_hidden_dim=32,
        mlp_n_layers=1, mlp_hidden_dim=16, numeric_hidden_dim=32,
        embed_dim=16,
        # activation=lrelu20, cat_hidden_dim=64, image_hidden_dim=66,
        # mlp_n_layers=4, mlp_hidden_dim=557, numeric_hidden_dim=96,
        # activation=lrelu20, cat_hidden_dim=38, image_hidden_dim=66,
        # mlp_n_layers=3, mlp_hidden_dim=557, numeric_hidden_dim=86,
        )
    predictor = setup_predictor(
        num_cat_id=num_cat_id, use_bn=use_bn, use_sn=use_sn,
        dropout_ratio=dropout_ratio, use_residual=use_residual,
        out_dim=out_dim, use_xdeepfm=True,
        **kwargs)
    # lam_image_recon = 0.010558
    lam_image_recon = 0.0
    image_input_dim = train_x_image.shape[2] if train_x_image is not None else 0
    # if image_encode:

    def lossfun_huber(y, t):
        return F.mean(F.huber_loss(y, t, delta=1.0))

    def lossfun_custom(y, t):
        diff = F.absolute_error(y, t)
        # print('y', y.shape, t.shape, diff.shape)
        return F.mean(F.leaky_relu(diff - 0.5, slope=0.2)) + 0.1

    # F.mean_absolute_error
    if mode == 'mean':
        mean_predictor = setup_predictor(
            num_cat_id=num_cat_id, use_bn=False, use_sn=use_sn,
            dropout_ratio=dropout_ratio, use_residual=use_residual,
            out_dim=64, use_set=True, **kwargs)
        regressor = BlendNetMeanRegressor(
            predictor, mean_predictor, lossfun=F.mean_squared_error,  # F.mean_squared_error,
            metrics_fun=metrics_fun, device=device, x_numeric_dim=x_numeric.shape[1],
            use_sn=use_sn, image_encoder_hdim=54, image_encoder_layers=2,
            lam_image_recon=lam_image_recon,
            image_input_dim=image_input_dim, image_encode=image_encode,
            dropout_ratio=dropout_ratio)
        batch_eval_func = regressor.calc
    else:
        assert mode == 'normal'
        regressor = BlendNetRegressor(
            predictor, lossfun=F.mean_squared_error,   # lossfun_huber
            metrics_fun=metrics_fun, device=device, x_numeric_dim=x_numeric.shape[1],
            use_sn=use_sn, mode=mode, lam_image_recon=lam_image_recon,
            image_input_dim=image_input_dim, image_encode=image_encode,
            dropout_ratio=dropout_ratio)  #dropout_ratio
        batch_eval_func = regressor.calc
    # else:
    #     regressor = Regressor(predictor, lossfun=F.mean_squared_error,
    #                           metrics_fun=metrics_fun, device=device)
    #     batch_eval_func = predictor

    # Set up the optimizer.
    optimizer = optimizers.Adam(alpha=0.001)
    # optimizer = optimizers.MomentumSGD(lr=0.001)
    # optimizer = optimizers.AdaDelta()
    # optimizer = optimizers.AdaGrad()
    # optimizer = optimizers.RMSprop()

    print('optimizer', type(optimizer))
    optimizer.setup(regressor)
    if decay_rate > 0.:
        print('add WeightDecay={}'.format(decay_rate))
        optimizer.add_hook(WeightDecay(decay_rate))

    # Set up the updater.
    updater = training.StandardUpdater(
        train_iter, optimizer, device=device, converter=converter)

    # Set up the trainer.
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)
    trainer.extend(E.Evaluator(valid_iter, regressor, device=device,
                               converter=converter))

    def batch_converter(batch, device=None):
        if mode == 'normal':
            return converter(batch, device)
        else:
            assert mode == 'mean'
            x_num, x_cat, x_bert, x_image, t, indices, target_mean = converter(batch, device)
            return x_num, x_cat, x_bert, x_image, indices, t
    trainer.extend(QuadraticWeightedKappaEvaluator(
        valid_iter, regressor, device=device, eval_func=batch_eval_func, name='val',
        converter=batch_converter))
    # trainer.extend(QuadraticWeightedKappaEvaluator(train_iter, regressor, device=device,))
    # trainer.extend(E.snapshot(), trigger=(epoch, 'epoch'))

    trainer.extend(observe_lr())
    trainer.extend(schedule_optimizer_value(
        # [2, 10, 20, 40], [0.005, 0.003, 0.001]))
        # [2, 10, 25, 45, 50, 55], [0.005, 0.003, 0.001, 0.0003, 0.0001, 0.00001]))
        [2, 10, 25, 45, 50, 55], [0.005, 0.003, 0.001, 0.0003, 0.0001, 0.00001]))
        # [2, 10, 25, 45, 50, 55], [0.001, 0.001, 0.001, 0.0003, 0.0001, 0.00001]))
    log_report = E.LogReport()
    trainer.extend(log_report)
    if image_encode:
        if mode == 'mean':
            trainer.extend(E.PrintReport([
                'epoch', 'main/loss', 'main/reg_loss', 'main/reg_loss_mean', 'main/recon_loss', 'main/mae',
                'validation/main/loss', 'validation/main/reg_loss', 'validation/main/reg_loss_mean',
                'validation/main/recon_loss',
                'validation/main/mae', 'val/main/qwk', 'lr',
                'elapsed_time']))
        else:
            trainer.extend(E.PrintReport([
                'epoch', 'main/loss', 'main/reg_loss', 'main/recon_loss', 'main/mae',
                'validation/main/loss', 'validation/main/reg_loss', 'validation/main/recon_loss',
                'validation/main/mae', 'val/main/qwk', 'lr',
                'elapsed_time']))
    else:
        if mode == 'mean':
            trainer.extend(E.PrintReport([
                'epoch', 'main/loss', 'main/reg_loss', 'main/reg_loss_mean', 'main/mae',
                'validation/main/loss', 'validation/main/reg_loss', 'validation/main/reg_loss_mean',
                'validation/main/mae', 'val/main/qwk', 'lr', 'elapsed_time']))
        else:
            trainer.extend(E.PrintReport([
                'epoch', 'main/loss', 'main/mae',
                'validation/main/loss', 'validation/main/mae', 'val/main/qwk',
                'elapsed_time']))

    if not is_kaggle_kernel:
        trainer.extend(E.ProgressBar())
    trainer.run()

    # check_error = True
    if check_error:
        val_pred = regressor.predict(valid, converter=converter, batchsize=batchsize)
        if mode == 'mean':
            assert isinstance(valid, RescuerIDMeanDataset) and valid.mode == 'eval'
            permute_indices = numpy.argsort(numpy.array(list(chain.from_iterable(valid.rescuer_id_index_list))))
            val_pred = val_pred[permute_indices]
        diff = val_pred - val_target
        diff = diff[:, 0]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.scatter(val_target[:, 0], val_pred[:, 0])
        y = val_target[:, 0]
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        plt.savefig(os.path.join(out, 'scatter.png'))
        import seaborn as sns
        df = pd.DataFrame({'pred': val_pred[:, 0], 'actual': val_target[:, 0]})
        ax = sns.violinplot(x='actual', y='pred', data=df)
        plt.savefig(os.path.join(out, 'pred_violin.png'))

        print('saved to ', os.path.join(out, 'pred_violin.png'))
        # from biggest error, descending order
        indices = numpy.argsort(diff)
        # Top 3-error for both side
        print('val_indices', val_indices[indices[:10]])
        print('val_indices', val_indices[indices[::-1][:10]])

    # --- optr ---
    if optr is not None:
        with timer('predict'):
            converter.extract_inputs = True  # HACKING
            val_y = regressor.predict(valid, converter=converter)
            if mode == 'mean':
                # scatter operations...
                assert isinstance(valid, RescuerIDMeanDataset) and valid.mode == 'eval'
                permute_indices = numpy.argsort(numpy.array(list(chain.from_iterable(valid.rescuer_id_index_list))))
                val_y = val_y[permute_indices]
                # val_y = val_y[list(chain.from_iterable(valid.rescuer_id_index_list))]
                # val_y = valid.target_mean  # DEBUG, this is answer...
            converter.extract_inputs = False
            if predict_y is not None:
                predict_y[val_indices] = val_y
        with timer('optr.fit'):
            optr.fit(val_y, val_target)
        coefficients = optr.coefficients()

        pred_y1_rank = optr.predict(val_y, coefficients)
        # pred_y1_rank = optr.predict(pred_y1, [0.5, 2.2, 3.2, 3.3])
        score = cohen_kappa_score(pred_y1_rank, val_target,
                                  labels=numpy.arange(optr.num_class),
                                  weights='quadratic')
        print('optimized score', score, 'coefficients', coefficients,)
    else:
        print('optr is not set, skip optimizing threshold...')
        coefficients = None

    return regressor, coefficients, log_report.log, score


def main_nn(debug, device, epoch, use_bert, use_image, num_image,
            use_selection_gate=False, use_tfidf=False, use_sn=False,
            dropout_ratio=0.1, decay_rate=1e-4, use_bn=False, animal_type=None,
            cat2num=False, use_cat=True, image_type='clf', use_residual=False,
            mode='normal', batchsize=1024, fold=4):
    """

    Args:
        debug:
        device:
        epoch:
        use_bert:
        use_image:
        num_image:
        use_selection_gate:
        use_tfidf:
        use_sn:
        dropout_ratio:
        decay_rate:
        use_bn:
        animal_type (int): if specified, only train specific animal_type = 'dog' or 'cat'.
            If Negative value, both animals are trained same time.
        cat2num (bool): If True categorical values are converted to numeric values.
        use_cat (bbol): Use category specific Linear layer or not.
        mode (str): Training mode type.
            normal - normal training, regression of AdoptionSpeed.
            mean - regression of mean AdoptionSpeed on RescuerID.

    Returns:

    """
    num_class = 5
    print('debug', debug)
    print('epoch', epoch, 'dropout_ratio', dropout_ratio, 'decay_rate', decay_rate)
    print('Train data_type {}'.format(animal_type))

    if animal_type < 0:
        animal_type = None
    # use_cat = not cat2num

    # --- dataset ---

    train, test, breeds, colors, states = prepare_df(debug, animal_type=animal_type)
    pp = Preprocessor(arch='nn')
    train, test = pp.preprocess(
        train, test, breeds, colors, states, debug=debug, use_tfidf=use_tfidf,
        use_metadata=False, use_sentiment=False, use_gdp=True,
        use_rescuer_id_count=True, use_name_feature=True, use_target_encoding=False,
        cat2num=cat2num, animal_type=animal_type, use_tfidf_cache=True,
        tfidf_svd_components=16, num_sentiment_text=0,
        use_sentiment2=True, use_metadata2=True, use_text=True, use_fasttext=True,
        use_image_size=True, arch='nn', use_custom_boolean_feature=True,
        add_pred=None)
    target = pp.target
    print('target', target.shape)
    train_x_numeric = train.loc[:, pp.numeric_cols].values.astype(numpy.float32)
    test_x_numeric = test.loc[:, pp.numeric_cols].values.astype(numpy.float32)
    # print('num nan in numeric: ', numpy.sum(numpy.isnan(train_x_numeric), axis=0))
    cat_dtype = numpy.float32 if cat2num else numpy.int32
    train_x_cat = train.loc[:, pp.cat_cols].values.astype(cat_dtype)
    # print('num nan in cat', numpy.sum(numpy.isnan(train_x_cat), axis=0))
    test_x_cat = test.loc[:, pp.cat_cols].values.astype(cat_dtype)
    num_cat_id = pp.num_cat_id
    print(f'numeric_cols {pp.numeric_cols}, cat_cols {pp.cat_cols}')

    # train_x_numeric, train_x_cat, target, test_x_numeric, test_x_cat, num_cat_id = preprocessing(
    #     train, test, breeds, colors, states, debug=debug, use_tfidf=use_tfidf, use_metadata=True,
    #     cat2num=cat2num, animal_type=animal_type, use_tfidf_cache=True,
    #     tfidf_svd_components=128, num_sentiment_text=0)
    if use_bert:
        print('preprocess bert feature...')
        train_x_bert, test_x_bert = preprocess_bert(
            train, test, num_extract_sentence=2, layer_indices=[-1, ], device=device,
            use_cache=True, animal_type=animal_type)
    else:
        print('skip bert feature')
        train_x_bert, test_x_bert = None, None

    if use_image:
        print('preprocess image feature... {}'.format(image_type))
        if image_type == 'clf':

            # clf_arch = 'vgg16'
            clf_arch = 'densenet'
            # clf_arch = 'seresnext50'
            # n_components = None if debug else 506
            # n_components = 512
            # n_components = 256
            n_components = 32
            # n_components = 16
            if clf_arch == 'densenet':
                train_x_image, test_x_image = preprocess_image_densenet(
                    train, test, n_components=n_components, method='svd')
                train_x_image = train_x_image.astype(numpy.float32)
                test_x_image = test_x_image.astype(numpy.float32)
            else:
                train_x_image, test_x_image = preprocess_image(
                    train, test, num_image=num_image, device=device, arch=clf_arch,
                    n_components=n_components, animal_type=animal_type, use_cache=True,
                    method='svd')
        elif image_type == 'det':
            det_arch = 'faster_rcnn_vgg16'  # fpn50
            # n_components = None  # 506 was worse...
            # n_components = 32  # 506 was worse...
            n_components = 16  # 506 was worse...
            train_x_image, test_x_image = preprocess_image_det(
                train, test, num_image=num_image, device=device, arch=det_arch, use_cache=True,
                n_components=n_components)
            # train_x_image[numpy.isnan(train_x_image)] = 0.
            # test_x_image[numpy.isnan(test_x_image)] = 0.
        else:
            raise ValueError("[ERROR] Unexpected value image_type={}".format(image_type))
    else:
        print('skip image feature...')
        train_x_image, test_x_image = None, None

    # --- Setup CV ---
    num_split = fold
    random_state = 42
    kfold_method = 'group'
    # kfold_method = 'stratified'

    # num_split = 5
    # random_state = 1337
    # kfold_method = 'stratified'

    if kfold_method == 'stratified':
        print('StratifiedKFold...')
        kf = StratifiedKFold(n_splits=num_split, random_state=random_state, shuffle=True)
        # fold_splits = kf.split(train, target)
        fold_splits = kf.split(train_x_numeric, target)
    elif kfold_method == 'group':
        print('GroupKFold...')
        kf = GroupKFold(num_split)
        # kf.random_state = 42
        groups = train['RescuerID'].astype('category').cat.codes.values
        fold_splits = kf.split(train_x_numeric, target, groups)
    else:
        raise ValueError("[ERROR] Unexpected value kfold_method={}".format(kfold_method))

    optr = OptimizedRounder(num_class=num_class, method='nelder-mead')  # differential_evolution

    regressor_list = []
    coefficients_list = []
    log_list = []
    score_list = []

    if use_selection_gate:
        fit_fn = fit_nn_sg
    else:
        fit_fn = fit_nn

    std_list = calc_std_list(train_x_numeric, train_x_cat, train_x_bert, train_x_image)
    converter = BlendConverter(
        use_cat=use_cat,
        use_bert=use_bert,
        use_image=use_image,
        augmentation=True,
        # permute_col_ratio_list=[0.12930218, 0.197565899, 0.0, 0.023249633],
        permute_col_ratio_list=[0.15930218, 0.197565899, 0.0, 0.023249633],
        # permute_col_ratio_list=[0.07930218, 0.107565899, 0.0, 0.],
        # permute_col_ratio_list=[0., 0., 0.0, 0.],
        # permute_col_ratio_list=[0.15930218, 0.107565899, 0.0, 0.],
        num_cols_choice=False,
        mixup_ratio=0.,
        std_list=std_list,
        # noise_ratio_list=[0.01, 0., 0., 0.01],
        noise_ratio_list=[0.0, 0., 0., 0.0],
        # noise_ratio_list=[0.1, 0.1, 0.0, 0.0],
        mode=mode,  # [0.1, 0.1, 0.1, 0.1]
        use_embed=not cat2num
    )

    predict_y = numpy.ones(target.shape, dtype=numpy.float32) * -1
    for k, (train_indices, val_indices) in enumerate(fold_splits):
        print('---- {} fold / {} ---'.format(k, num_split))
        regressor, coefficients, log, score = fit_fn(
            train_x_numeric, train_x_cat, target, num_cat_id,
            train_indices, val_indices,
            x_bert=train_x_bert, x_image=train_x_image, debug=debug, device=device,
            epoch=epoch, optr=optr, out='{}_{}'.format(out, k), converter=converter,
            use_sn=use_sn, decay_rate=decay_rate, dropout_ratio=dropout_ratio,
            use_bn=use_bn, use_residual=use_residual,
            mode=mode, train_df=train, batchsize=batchsize, predict_y=predict_y)

        regressor_list.append(regressor)
        coefficients_list.append(coefficients)
        log_list.append(log)
        score_list.append(score)

    log_df = pd.DataFrame([log[-1] for log in log_list])
    print('log_df mean\n{}'.format(log_df.mean()))
    opt_score_mean = numpy.array(score_list).mean()
    print('opt_score_mean', opt_score_mean, 'score_list', score_list)

    print('Number of un-predicted example: ', numpy.sum(predict_y <= -1))
    with timer('optr.fit'):
        optr.fit(predict_y, target)
    coefficients = optr.coefficients()
    pred_y1_rank = optr.predict(predict_y, coefficients)
    score = cohen_kappa_score(pred_y1_rank, target,
                              labels=numpy.arange(optr.num_class),
                              weights='quadratic')
    print('Total: optimized score', score, 'coefficients', coefficients, )

    # --- create submission ---
    flag_create_submission = True
    if flag_create_submission:
        extra_test_features = []
        if test_x_cat is not None:
            extra_test_features.append(test_x_cat)
        if test_x_bert is not None:
            extra_test_features.append(test_x_bert)
        if test_x_image is not None:
            extra_test_features.append(test_x_image)

        test_dataset = NumpyTupleDataset(*([test_x_numeric] + extra_test_features))
        coefficients_ = numpy.mean(numpy.array(coefficients_list), axis=0)
        print('coefficients_', coefficients_)
        # train_predictions = [r[0] for r in results['train']]
        # train_predictions = optr.predict(train_predictions, coefficients_).astype(int)
        # Counter(train_predictions)

        with timer('test predict'):
            if mode == 'normal':
                test_predict_list = [reg.predict(test_dataset, converter=converter)
                                     for reg in regressor_list]
            elif mode == 'mean':
                test_dataset = RescuerIDMeanDataset(test.copy(), test_dataset, mode='eval')
                assert isinstance(test_dataset, RescuerIDMeanDataset) and test_dataset.mode == 'eval'
                permute_indices = numpy.argsort(numpy.array(list(chain.from_iterable(test_dataset.rescuer_id_index_list))))
                raise NotImplementedError
                import IPython; IPython.embed()
                test_predict_list = [reg.predict(test_dataset, converter=converter, batchsize=batchsize)[permute_indices]
                                     for reg in regressor_list]
            else:
                raise ValueError("[ERROR] Unexpected value mode={}".format(mode))
        test_predict_mean = numpy.mean(numpy.array(test_predict_list), axis=0)
        print('test_predict_mean', test_predict_mean.shape)
        test_id = test['PetID']

        # --- 0. raw float predictions ---
        predict_df = pd.DataFrame({'PetID': train['PetID'], 'y': predict_y.ravel(), 't': target.ravel()})
        predict_df.to_csv('predict_nn_train.csv', index=False)
        print('predict_nn_train.csv created.')
        predict_df = pd.DataFrame({'PetID': test_id, 'y': test_predict_mean.ravel()})
        predict_df.to_csv('predict_nn_test.csv', index=False)
        print('predict_nn.csv created.')

        # --- 1. mean coefficients ---
        coefficients_mean = numpy.mean(numpy.array(coefficients_list), axis=0)
        print('coefficients_mean', coefficients_mean)
        test_predictions = optr.predict(test_predict_mean, coefficients_mean).astype(int)
        print('test_predictions counter', Counter(test_predictions))
        submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
        submission.to_csv('submission_mean.csv', index=False)
        print('submission_mean.csv created.')

        # --- 2. validation all coefficients ---
        print('coefficients from all validation ', coefficients)
        test_predictions = optr.predict(test_predict_mean, coefficients).astype(int)
        print('test_predictions counter', Counter(test_predictions))
        submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
        submission.to_csv('submission_valcoef.csv', index=False)
        print('submission.csv created.')

        # --- 3. same histgram with train... ---
        test_predictions = optr.fit_and_predict_by_histgram(test_predict_mean, target)
        print('coefficients to align with train target', optr.coefficients())
        print('test_predictions counter', Counter(test_predictions))
        submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
        submission.to_csv('submission.csv', index=False)
        print('submission.csv created.')


# In[ ]:


gc.collect()


# In[ ]:


use_sn = False
dropout_ratio = 0.08
decay_rate = 1e-3
fold = 10
use_bn = False
use_residual = True

main_nn(debug, device, epoch, use_bert, use_image, num_image,
        use_selection_gate=False, use_tfidf=use_tfidf, use_sn=use_sn,
        dropout_ratio=dropout_ratio, decay_rate=decay_rate, use_bn=use_bn,
        image_type=image_type, animal_type=-1, use_residual=use_residual,
        mode=mode, batchsize=batchsize, fold=fold)


# In[ ]:


gc.collect()


# # --- ensemble ---

# In[ ]:


from collections import Counter

import numpy
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score

import sys
import os

class MeanPredictor(object):
    def __init__(self):
        self.coef_ = None

    def fit(self, x, y):
        # dummy, no internal parameter
        pass

    def predict(self, x):
        if x.ndim == 2:
            return numpy.mean(x, axis=1)
        else:
            raise ValueError("[ERROR] Unexpected value x.shape={}".format(x.shape))

            
optr = OptimizedRounder(method='nelder-mead')
train, test, breeds, colors, states = prepare_df(debug)
dog_train_indices = (train['Type'] == 1).values
cat_train_indices = (train['Type'] == 2).values
dog_test_indices = (test['Type'] == 1).values
cat_test_indices = (test['Type'] == 2).values

ensemble_archs = ['nn', 'lgbm', 'xgb', 'cb', 'xlearn']
# ensemble_archs = ['lgbm', 'xgb', 'cb', 'xlearn']
# ensemble_archs = ['nn', 'lgbm', 'xgb', 'cb']
# ensemble_archs = ['lgbm', 'xgb', 'cb']
# ensemble_archs = ['xgb', 'cb']

# --- Train/validation check ---
pet_id = None
target = None
y_list = []
for model_name in ensemble_archs:
    filepath = 'predict_{}_train.csv'.format(model_name)
    df = pd.read_csv(filepath)
    if pet_id is None:
        pet_id = df['PetID'].values
        target = df['t'].values
    else:
        assert numpy.alltrue(pet_id == df['PetID'].values)
        assert numpy.alltrue(target == df['t'].values)
    y = df['y'].values
    rmse = numpy.sqrt(numpy.mean(numpy.square(y - target)))
    print('model_name {}, rmse {}'.format(model_name, rmse))
    y_list.append(y)

# --- 1. Just take mean ---
predict_y = numpy.mean(numpy.array(y_list), axis=0)
rmse = numpy.sqrt(numpy.mean(numpy.square(predict_y - target)))
print('mean ensemble, rmse {}'.format(rmse))

# --- 2. User ridge regression ---
# ridge = Ridge()
ridge = MeanPredictor()
X = numpy.array(y_list).T
print('X', X.shape)
ridge.fit(X, target)
predict_y = ridge.predict(X)
rmse = numpy.sqrt(numpy.mean(numpy.square(predict_y - target)))
print('ridge ensemble, rmse {}'.format(rmse))
print('ridge coef_ {}'.format(ridge.coef_))

print('--- optimize all ---')
with timer('optr.fit'):
    optr.fit(predict_y, target)
coefficients1 = optr.coefficients()
print('coefficients1', coefficients1)
pred_y1_rank = optr.predict(predict_y, coefficients1)
score = cohen_kappa_score(pred_y1_rank, target,
                          labels=numpy.arange(optr.num_class),
                          weights='quadratic')
print('Total: optimized score', score, 'coefficients1', coefficients1)
print('predicted histogram: dog', Counter(pred_y1_rank[dog_train_indices]),
      'cat', Counter(pred_y1_rank[cat_train_indices]))

# --- if same with train histogram... ---
print('--- train histogram ---')
pred_y1_rank = optr.fit_and_predict_by_histgram(predict_y, target)
score = cohen_kappa_score(pred_y1_rank, target,
                          labels=numpy.arange(optr.num_class),
                          weights='quadratic')
print('Total: train_histgram score', score)
print('predicted histogram: dog', Counter(pred_y1_rank[dog_train_indices]),
      'cat', Counter(pred_y1_rank[cat_train_indices]))

# --- test dataset ---
print('--- test ---')
print('test dog', (test['Type'] == 1).sum(), 'cat', (test['Type'] == 2).sum())

test_id = None
y_list = []
for model_name in ensemble_archs:
    filepath = 'predict_{}_test.csv'.format(model_name)
    df = pd.read_csv(filepath)
    if test_id is None:
        test_id = df['PetID'].values
    else:
        assert numpy.alltrue(test_id == df['PetID'].values)
    y = df['y'].values
    y_list.append(y)

# predict_y = numpy.mean(numpy.array(y_list), axis=0)
X_test = numpy.array(y_list).T
print('X_test', X_test.shape)
predict_y_test = ridge.predict(X_test)
print('predict_y_test', predict_y_test.shape, predict_y_test)
print('squared error', numpy.mean(numpy.square(X_test - predict_y_test[:, None]), axis=0))
print('abs error', numpy.mean(numpy.abs(X_test - predict_y_test[:, None]), axis=0))

print('--- test with coefficients1, optimized by all val data ---')
test_predictions = optr.predict(predict_y_test, coefficients1)
submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
submission.to_csv('submission_coef1.csv', index=False)
print('test_predictions counter', Counter(test_predictions), test_predictions)
print('submission_coef1.csv created.')

print('--- test with coefficients3, manual tuned from coefficients1 ---')
coefficients3 = coefficients1.copy()
coefficients3[0] = 1.66
coefficients3[1] = 2.13
coefficients3[3] = 2.85
test_predictions = optr.predict(predict_y_test, coefficients3)
submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
submission.to_csv('submission_coef3.csv', index=False)
print('test_predictions counter', Counter(test_predictions), test_predictions)
print('submission_coef3.csv created.')

# coefficients_ = coefficients.copy()
# pred_y1_rank = optr.predict(predict_y_test, coefficients_)
print('--- test align histogram all ---')
test_predictions = optr.fit_and_predict_by_histgram(predict_y_test, target)
print('coefficients to align with train target', optr.coefficients())
print('test_predictions counter', Counter(test_predictions), test_predictions)
submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
submission.to_csv('submission_histogram.csv', index=False)
print('submission_histogram.csv created.')


# # calculate threshold for target histogram

# In[ ]:


# https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/87037
# train counter Counter({4.0: 4197, 2.0: 4037, 3.0: 3259, 1.0: 3090, 0.0: 410})

# coefficients_ = coefficients.copy()
# pred_y1_rank = optr.predict(predict_y_test, coefficients_)
print('--- test align histogram all ---')
coef = optr.calc_histogram_coef(predict_y_test, [410, 3090, 4037, 3259, 4197])
test_predictions = optr.predict(predict_y_test, coef)
print('coefficients to align with train target', coef)
print('test_predictions counter', Counter(test_predictions), test_predictions)
submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
submission.to_csv('submission.csv', index=False)
print('submission.csv created.')


# In[ ]:





# In[ ]:


submission


# In[ ]:


end_time = time()
print('total took {} sec'.format(end_time - start_time))


# In[ ]:




