#!/usr/bin/env python
# coding: utf-8

# #  Forked from [Baseline Modeling](https://www.kaggle.com/wrosinski/baselinemodeling)

# ## Added Image features from [Extract Image features from pretrained NN](https://www.kaggle.com/christofhenkel/extract-image-features-from-pretrained-nn)

# ## Added Image size features from [Extract Image Features](https://www.kaggle.com/kaerunantoka/extract-image-features)

# In[ ]:


import datetime
datetime.datetime.now()


# In[ ]:


import gc
import glob
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
import warnings
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from joblib import Parallel, delayed
from tqdm import tqdm, tqdm_notebook

get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(seed=1337)
warnings.filterwarnings('ignore')

split_char = '/'
os.listdir('../input')
train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
sample_submission = pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')


# ## Image features

# In[ ]:


import cv2
import os
from keras.applications.densenet import preprocess_input, DenseNet121
def resize_to_square(im):
    old_size = im.shape[:2]
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
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
img_size = 256
batch_size = 256

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

pet_ids = train['PetID'].values
n_batches = len(pet_ids) // batch_size + 1

features = {}
for b in tqdm(range(n_batches)):
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
train_feats.columns = [f'pic_{i}' for i in range(train_feats.shape[1])]


# In[ ]:


pet_ids = test['PetID'].values
n_batches = len(pet_ids) // batch_size + 1

features = {}
for b in tqdm(range(n_batches)):
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
        
test_feats = pd.DataFrame.from_dict(features, orient='index')
test_feats.columns = [f'pic_{i}' for i in range(test_feats.shape[1])]     

train_feats = train_feats.reset_index()
train_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)

test_feats = test_feats.reset_index()
test_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)

all_ids = pd.concat([train, test], axis=0, ignore_index=True, sort=False)[['PetID']]
all_ids.shape


# In[ ]:


n_components = 16
svd_ = TruncatedSVD(n_components=n_components, random_state=1337)

features_df = pd.concat([train_feats, test_feats], axis=0)
features = features_df[[f'pic_{i}' for i in range(256)]].values

svd_col = svd_.fit_transform(features)
svd_col = pd.DataFrame(svd_col)
svd_col = svd_col.add_prefix('IMG_SVD_')
features = features_df[[f'pic_{i}' for i in range(256)]]
#img_features = pd.concat([all_ids, svd_col,features.reset_index(drop=True, inplace=True)], axis=1)
img_features = pd.concat([all_ids, svd_col], axis=1)

gc.collect()


# ## About metadata and sentiment

# In[ ]:


labels_breed = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
labels_state = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')
labels_color = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')

train_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'))
train_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_metadata/*.json'))
train_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_sentiment/*.json'))

print(f'num of train images files: {len(train_image_files)}')
print(f'num of train metadata files: {len(train_metadata_files)}')
print(f'num of train sentiment files: {len(train_sentiment_files)}')


test_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg'))
test_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_metadata/*.json'))
test_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_sentiment/*.json'))

print(f'num of test images files: {len(test_image_files)}')
print(f'num of test metadata files: {len(test_metadata_files)}')
print(f'num of test sentiment files: {len(test_sentiment_files)}')


# ### Train test

# In[ ]:


# Images:
train_df_ids = train[['PetID']]
print(train_df_ids.shape)

# Metadata:
train_df_ids = train[['PetID']]
train_df_metadata = pd.DataFrame(train_metadata_files)
train_df_metadata.columns = ['metadata_filename']
train_metadata_pets = train_df_metadata['metadata_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])
train_df_metadata = train_df_metadata.assign(PetID=train_metadata_pets)
print(len(train_metadata_pets.unique()))

pets_with_metadatas = len(np.intersect1d(train_metadata_pets.unique(), train_df_ids['PetID'].unique()))
print(f'fraction of pets with metadata: {pets_with_metadatas / train_df_ids.shape[0]:.3f}')

# Sentiment:
train_df_ids = train[['PetID']]
train_df_sentiment = pd.DataFrame(train_sentiment_files)
train_df_sentiment.columns = ['sentiment_filename']
train_sentiment_pets = train_df_sentiment['sentiment_filename'].apply(lambda x: x.split(split_char)[-1].split('.')[0])
train_df_sentiment = train_df_sentiment.assign(PetID=train_sentiment_pets)
print(len(train_sentiment_pets.unique()))

pets_with_sentiments = len(np.intersect1d(train_sentiment_pets.unique(), train_df_ids['PetID'].unique()))
print(f'fraction of pets with sentiment: {pets_with_sentiments / train_df_ids.shape[0]:.3f}')


# Images:
test_df_ids = test[['PetID']]
print(test_df_ids.shape)

# Metadata:
test_df_metadata = pd.DataFrame(test_metadata_files)
test_df_metadata.columns = ['metadata_filename']
test_metadata_pets = test_df_metadata['metadata_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])
test_df_metadata = test_df_metadata.assign(PetID=test_metadata_pets)
print(len(test_metadata_pets.unique()))

pets_with_metadatas = len(np.intersect1d(test_metadata_pets.unique(), test_df_ids['PetID'].unique()))
print(f'fraction of pets with metadata: {pets_with_metadatas / test_df_ids.shape[0]:.3f}')

# Sentiment:
test_df_sentiment = pd.DataFrame(test_sentiment_files)
test_df_sentiment.columns = ['sentiment_filename']
test_sentiment_pets = test_df_sentiment['sentiment_filename'].apply(lambda x: x.split(split_char)[-1].split('.')[0])
test_df_sentiment = test_df_sentiment.assign(PetID=test_sentiment_pets)
print(len(test_sentiment_pets.unique()))

pets_with_sentiments = len(np.intersect1d(test_sentiment_pets.unique(), test_df_ids['PetID'].unique()))
print(f'fraction of pets with sentiment: {pets_with_sentiments / test_df_ids.shape[0]:.3f}')


# ## Extract features from json

# In[ ]:


class PetFinderParser(object):
    
    def __init__(self, debug=False):
        
        self.debug = debug
        self.sentence_sep = ' '
        
        self.extract_sentiment_text = False
    
    def open_json_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            json_file = json.load(f)
        return json_file
        
    def parse_sentiment_file(self, file):
        """
        Parse sentiment file. Output DF with sentiment features.
        """
        
        file_sentiment = file['documentSentiment']
        file_entities = [x['name'] for x in file['entities']]
        file_entities = self.sentence_sep.join(file_entities)
        
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
        
        return df_sentiment
    
    def parse_metadata_file(self, file):
        """
        Parse metadata file. Output DF with metadata features.
        """
        
        file_keys = list(file.keys())
        
        if 'labelAnnotations' in file_keys:
            file_annots = file['labelAnnotations']
            file_top_score = np.asarray([x['score'] for x in file_annots]).mean()
            file_top_desc = [x['description'] for x in file_annots]
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
    

def extract_additional_features(pet_id, mode='train'):
    
    sentiment_filename = f'../input/petfinder-adoption-prediction/{mode}_sentiment/{pet_id}.json'
    try:
        sentiment_file = pet_parser.open_json_file(sentiment_filename)
        df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)
        df_sentiment['PetID'] = pet_id
    except FileNotFoundError:
        df_sentiment = []

    dfs_metadata = []
    metadata_filenames = sorted(glob.glob(f'../input/petfinder-adoption-prediction/{mode}_metadata/{pet_id}*.json'))
    if len(metadata_filenames) > 0:
        for f in metadata_filenames:
            metadata_file = pet_parser.open_json_file(f)
            df_metadata = pet_parser.parse_metadata_file(metadata_file)
            df_metadata['PetID'] = pet_id
            dfs_metadata.append(df_metadata)
        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)
    dfs = [df_sentiment, dfs_metadata]
    
    return dfs


pet_parser = PetFinderParser()


# In[ ]:


debug = False
train_pet_ids = train.PetID.unique()
test_pet_ids = test.PetID.unique()

if debug:
    train_pet_ids = train_pet_ids[:1000]
    test_pet_ids = test_pet_ids[:500]


dfs_train = Parallel(n_jobs=-1, verbose=1)(
    delayed(extract_additional_features)(i, mode='train') for i in train_pet_ids)

train_dfs_sentiment = [x[0] for x in dfs_train if isinstance(x[0], pd.DataFrame)]
train_dfs_metadata = [x[1] for x in dfs_train if isinstance(x[1], pd.DataFrame)]

train_dfs_sentiment = pd.concat(train_dfs_sentiment, ignore_index=True, sort=False)
train_dfs_metadata = pd.concat(train_dfs_metadata, ignore_index=True, sort=False)

print(train_dfs_sentiment.shape, train_dfs_metadata.shape)


dfs_test = Parallel(n_jobs=-1, verbose=1)(
    delayed(extract_additional_features)(i, mode='test') for i in test_pet_ids)

test_dfs_sentiment = [x[0] for x in dfs_test if isinstance(x[0], pd.DataFrame)]
test_dfs_metadata = [x[1] for x in dfs_test if isinstance(x[1], pd.DataFrame)]

test_dfs_sentiment = pd.concat(test_dfs_sentiment, ignore_index=True, sort=False)
test_dfs_metadata = pd.concat(test_dfs_metadata, ignore_index=True, sort=False)

print(test_dfs_sentiment.shape, test_dfs_metadata.shape)


# ### group extracted features by PetID:

# In[ ]:


aggregates = ['sum', 'mean', 'var', 'nunique', 'max']
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
test_metadata_gr.columns = pd.Index([f'{c[0]}_{c[1].upper()}' for c in test_metadata_gr.columns.tolist()])
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
test_sentiment_gr.columns = pd.Index([f'{c[0]}' for c in test_sentiment_gr.columns.tolist()])
test_sentiment_gr = test_sentiment_gr.reset_index()


# ### merge processed DFs with base train/test DF:

# In[ ]:


# Train merges:
train_proc = train.copy()
train_proc = train_proc.merge(
    train_sentiment_gr, how='left', on='PetID')
train_proc = train_proc.merge(
    train_metadata_gr, how='left', on='PetID')
train_proc = train_proc.merge(
    train_metadata_desc, how='left', on='PetID')
train_proc = train_proc.merge(
    train_sentiment_desc, how='left', on='PetID')

# Test merges:
test_proc = test.copy()
test_proc = test_proc.merge(
    test_sentiment_gr, how='left', on='PetID')
test_proc = test_proc.merge(
    test_metadata_gr, how='left', on='PetID')
test_proc = test_proc.merge(
    test_metadata_desc, how='left', on='PetID')
test_proc = test_proc.merge(
    test_sentiment_desc, how='left', on='PetID')

print(train_proc.shape, test_proc.shape)
assert train_proc.shape[0] == train.shape[0]
assert test_proc.shape[0] == test.shape[0]


# In[ ]:


train_breed_main = train_proc[['Breed1']].merge(
    labels_breed, how='left',
    left_on='Breed1', right_on='BreedID',
    suffixes=('', '_main_breed'))

train_breed_main = train_breed_main.iloc[:, 2:]
train_breed_main = train_breed_main.add_prefix('main_breed_')

train_breed_second = train_proc[['Breed2']].merge(
    labels_breed, how='left',
    left_on='Breed2', right_on='BreedID',
    suffixes=('', '_second_breed'))

train_breed_second = train_breed_second.iloc[:, 2:]
train_breed_second = train_breed_second.add_prefix('second_breed_')


train_proc = pd.concat(
    [train_proc, train_breed_main, train_breed_second], axis=1)


test_breed_main = test_proc[['Breed1']].merge(
    labels_breed, how='left',
    left_on='Breed1', right_on='BreedID',
    suffixes=('', '_main_breed'))

test_breed_main = test_breed_main.iloc[:, 2:]
test_breed_main = test_breed_main.add_prefix('main_breed_')

test_breed_second = test_proc[['Breed2']].merge(
    labels_breed, how='left',
    left_on='Breed2', right_on='BreedID',
    suffixes=('', '_second_breed'))

test_breed_second = test_breed_second.iloc[:, 2:]
test_breed_second = test_breed_second.add_prefix('second_breed_')


test_proc = pd.concat(
    [test_proc, test_breed_main, test_breed_second], axis=1)

print(train_proc.shape, test_proc.shape)


# In[ ]:


X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False)


# In[ ]:


X_temp = X.copy()

text_columns = ['Description', 'metadata_annots_top_desc', 'sentiment_entities']
categorical_columns = ['main_breed_BreedName', 'second_breed_BreedName']

to_drop_columns = ['PetID', 'Name', 'RescuerID']


# In[ ]:


def is_popular_name(name):
    most_popular_names = ["Max", "Sam", "Lady", "Bear", "Smokey", "Shadow", "Kitty", "Molly", "Buddy", "Brandy", "Ginger", "Baby", "Misty", "Missy", "Pepper", "Jake", "Bandit", "Tiger", "Samantha", "Lucky", "Muffin", "Princess", "Maggie", "Charlie", "Sheba", "Rocky", "Patches", "Tigger", "Rusty", "Buster", "Lucy", "Luna", "Sadie", "Sophie", "Jack"]
    return int(name in most_popular_names)


# In[ ]:





# In[ ]:


X_temp['isPopularName'] = X_temp['Name'].apply(is_popular_name)
rescuer_count = X.groupby(['RescuerID'])['PetID'].count().reset_index()
rescuer_count.columns = ['RescuerID', 'RescuerID_COUNT']

X_temp = X_temp.merge(rescuer_count, how='left', on='RescuerID')
for i in categorical_columns:
    X_temp.loc[:, i] = pd.factorize(X_temp.loc[:, i])[0]
X_text = X_temp[text_columns]

for i in X_text.columns:
    X_text.loc[:, i] = X_text.loc[:, i].fillna('none')
    
def set_pet_breed(b1, b2):
    #print(b1, b2)
    if (b1 in  (0, 307)) & (b2 in  (0, 307)):
        return 4
    elif (b1 ==  307) & (b2 not in  (0, 307)):
        return 3
    elif (b2 ==  307) & (b1 not in  (0, 307)):
        return 3
    elif (b1 not in  (0, 307)) & (b2 not in  (0, 307)) & (b1 != b2):
        return 2
    elif (b1 == 0) & (b2 not in  (0, 307)):
        return 1
    elif (b2 == 0) & (b1 not in  (0, 307)):
        return 1
    elif (b1 not in  (0, 307)) & (b2 not in  (0, 307)) & (b1 == b2):
        return 0
    else:
        return 3
X_temp["Pet_Breed"] = X_temp.apply(lambda x: set_pet_breed(x['Breed1'], x['Breed2']), axis=1)


# In[ ]:


import nltk
from nltk import word_tokenize
def avg_sen_len(sent_text):
    sum_len = 0
    for sent in sent_text:
        tokens = word_tokenize(sent)
        sum_len += len(tokens)
    return sum_len/len(sent_text)
def amount_of_words(text):
    tp = type(text) is str
    if not tp:
        return -1
    bag_of_words = nltk.word_tokenize(text)
    return len(bag_of_words)
def num_senten(text):
    tp = type(text) is str
    if not tp:
        return -1    
    sent_text = nltk.sent_tokenize(text) 
    return len(sent_text)


# In[ ]:


X_temp['num_senten'] = X_text['Description'].apply(num_senten)
X_temp['amount_of_words'] = X_text['Description'].apply(amount_of_words)

X_temp['Length_Description'] = X_text['Description'].map(len)
X_temp['Length_metadata_annots_top_desc'] = X_text['metadata_annots_top_desc'].map(len)
X_temp['Lengths_sentiment_entities'] = X_text['sentiment_entities'].map(len)


# In[ ]:


'''
from keras.preprocessing import text, sequence

word_vec_size = 300
max_words = 100
max_word_features = 25000

def transform_text(text, tokenizer):
    tokenizer.fit_on_texts(text)
    text_emb = tokenizer.texts_to_sequences(text)
    text_emb = sequence.pad_sequences(text_emb, maxlen=max_words)
    return text_emb

desc_tokenizer = text.Tokenizer(num_words=max_word_features)
desc_embs = transform_text(X_temp["Description"].astype(str), desc_tokenizer)
text_mode = "fasttext"

if text_mode == "fasttext":
    embedding_file = "../input/fasttext-english-word-vectors-including-subwords/wiki-news-300d-1M-subword.vec"

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
    gc.collect()
    
def text2features(embeddings_index, text):
    vec_stack = []
    for w in text:
        v = embeddings_index[w]
        if v is not None:
            vec_stack.append(v)
    if len(vec_stack) != 0:
        v_mean = np.mean(vec_stack, axis=0)
    else:
        v_mean = np.zeros(300)
    return v_mean
 
def df_to_embed_features():
    X = np.zeros((desc_embs.shape[0], word_vec_size), dtype='float32')
    for i, text in tqdm(enumerate(desc_embs)):
        X[i] = text2features(text_embs, text)    
    return X
text_emb = df_to_embed_features()
text_emb_df = pd.DataFrame(text_emb)
text_emb_df.columns = [f'fastext_feat{i}' for i in range(text_emb.shape[1])]

svd_ = TruncatedSVD(  n_components=16, random_state=1337)

svd_col = svd_.fit_transform(text_emb_df)
svd_col = pd.DataFrame(svd_col)
svd_col = svd_col.add_prefix('TFIDF_fasttext{}_'.format(i))

nmf_ = NMF(n_components=8, random_state=1337)
nmf_col = nmf_.fit_transform(text_emb_df)
nmf_col = pd.DataFrame(nmf_col)
nmf_col = nmf_col.add_prefix('NMF_{}_'.format(i))

X_temp = pd.concat([X_temp,svd_col, nmf_col], axis=1)
del text_emb_df,text_embs,desc_embs
gc.collect()
'''


# ### TFIDF

# In[ ]:


n_components = 16
text_features = []

# Generate text features:
for i in X_text.columns:
    
    # Initialize decomposition methods:
    print(f'generating features from: {i}')
    tfv = TfidfVectorizer(min_df=2,  max_features=None, #stop_words='english',
                          strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)
    svd_ = TruncatedSVD(
        n_components=n_components, random_state=1337)
    
    tfidf_col = tfv.fit_transform(X_text.loc[:, i].values)
    
    svd_col = svd_.fit_transform(tfidf_col)
    svd_col = pd.DataFrame(svd_col)
    svd_col = svd_col.add_prefix('TFIDF_{}_'.format(i))
    
    text_features.append(svd_col)
    
text_features = pd.concat(text_features, axis=1)

X_temp = pd.concat([X_temp, text_features], axis=1)

for i in X_text.columns:
    X_temp = X_temp.drop(i, axis=1)


# ### Merge image features

# In[ ]:





# In[ ]:


X_temp = X_temp.merge(img_features, how='left', on='PetID')


# ### Add image_size features

# In[ ]:


from PIL import Image
train_df_ids = train[['PetID']]
test_df_ids = test[['PetID']]

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

train_df_imgs['image_size'] = train_df_imgs['image_filename'].apply(getSize)
train_df_imgs['temp_size'] = train_df_imgs['image_filename'].apply(getDimensions)
train_df_imgs['width'] = train_df_imgs['temp_size'].apply(lambda x : x[0])
train_df_imgs['height'] = train_df_imgs['temp_size'].apply(lambda x : x[1])
train_df_imgs = train_df_imgs.drop(['temp_size'], axis=1)

test_df_imgs['image_size'] = test_df_imgs['image_filename'].apply(getSize)
test_df_imgs['temp_size'] = test_df_imgs['image_filename'].apply(getDimensions)
test_df_imgs['width'] = test_df_imgs['temp_size'].apply(lambda x : x[0])
test_df_imgs['height'] = test_df_imgs['temp_size'].apply(lambda x : x[1])
test_df_imgs = test_df_imgs.drop(['temp_size'], axis=1)

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

agg_imgs = pd.concat([agg_train_imgs, agg_test_imgs], axis=0).reset_index(drop=True)


# In[ ]:


X_temp = X_temp.merge(agg_imgs, how='left', on='PetID')


# ### Drop ID, name and rescuerID

# In[ ]:


#X_temp = X_temp.drop(to_drop_columns, axis=1)


# In[ ]:


X_train = X_temp.loc[np.isfinite(X_temp.AdoptionSpeed), :]
X_test = X_temp.loc[~np.isfinite(X_temp.AdoptionSpeed), :]

X_test = X_test.drop(['AdoptionSpeed'], axis=1)

assert X_train.shape[0] == train.shape[0]
assert X_test.shape[0] == test.shape[0]

train_cols = X_train.columns.tolist()
train_cols.remove('AdoptionSpeed')

test_cols = X_test.columns.tolist()

assert np.all(train_cols == test_cols)


# In[ ]:


X_train_non_null = X_train.fillna(-1000)
X_test_non_null = X_test.fillna(-1000)


# In[ ]:


X_train_non_null.isnull().any().any(), X_test_non_null.isnull().any().any()


# In[ ]:


X_train_non_null.shape, X_test_non_null.shape


# In[ ]:


import scipy as sp

from collections import Counter
from functools import partial
from math import sqrt

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix


# FROM: https://www.kaggle.com/myltykritik/simple-lgbm-image-features

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


# ### OptimizeRounder from [OptimizedRounder() - Improved](https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved)

# In[ ]:


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0
    
    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return -cohen_kappa_score(y, preds, weights='quadratic')
    
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
    
    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return preds
    
    def coefficients(self):
        return self.coef_['x']


# ## Train model

# In[ ]:


import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

xgb_params = {
    'eval_metric': 'rmse',
    'seed': 1337,
    'eta': 0.0123,
    'subsample': 0.8,
    'colsample_bytree': 0.85,
    'tree_method': 'gpu_hist',
    'device': 'gpu',
    'max_depth':6,
    'silent': 1,
}

# for xgboost here is my steps, usually i can reach almost good parameters in a few steps,
# 
# initialize parameters such: eta = 0.1, depth= 10, subsample=1.0, min_child_weight = 5, col_sample_bytree = 0.2 (depends on feature size), set proper objective for the problem (reg:linear, reg:logistic or count:poisson for regression, binary:logistic or rank:pairwise for classification)
# 
# split %20 for validation, and prepare a watchlist for train and test set, set num_round too high such as 1000000 so you can see the valid prediction for any round value, if at some point test prediction error rises you can terminate the program running,
# 
# i) play to tune depth parameter, generally depth parameter is invariant to other parameters, i start from 10 after watching best error rate for initial parameters then i can compare the result for different parameters, change it 8, if error is higher then you can try 12 next time, if for 12 error is lower than 10 , so you can try 15 next time, if error is lower for 8 you would try 5 and so on.
# 
# ii) after finding best depth parameter, i tune for subsample parameter, i started from 1.0 then change it to 0.8 if error is higher then try 0.9 if still error is higher then i use 1.0, and so on.
# 
# iii) in this step i tune for min child_weight, same approach above,
# 
# iv) then i tune for col_Sample_bytree
# 
# v) now i descrease the eta to 0.05, and leave program running then get the optimum num_round (where error rate start to increase in watchlist progress),
# 
# after these step you can get roughly good parameters (i dont claim best ones), then you can play around these parameters.
# 


# In[ ]:


print('MODEL LEARNING STAGE')


# In[ ]:



from sklearn.utils import shuffle

def run_xgb(params, X_train, X_test):
    n_splits = 10
    verbose_eval = 1000
    num_rounds = 60000
    early_stop = 500

    #kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
    #X_train = shuffle(X_train, random_state=0)

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    
    


    i = 0
    kf = GroupKFold(n_splits=n_splits)

    for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values, X_train['RescuerID']):
        if 'RescuerID' in X_train.columns:
            X_train = X_train.drop(to_drop_columns, axis=1)
            X_test = X_test.drop(to_drop_columns, axis=1)
        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)

        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)
        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred

        i += 1
    return model, oof_train, oof_test


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model, oof_train, oof_test = run_xgb(xgb_params, X_train_non_null, X_test_non_null)')


# In[ ]:


optR = OptimizedRounder()
optR.fit(oof_train, X_train['AdoptionSpeed'].values)
coefficients = optR.coefficients()
valid_pred = optR.predict(oof_train, coefficients)
qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)
print("QWK = ", qwk)


# In[ ]:


coefficients_ = coefficients.copy()
coefficients_[0] = 1.66
coefficients_[1] = 2.13
coefficients_[3] = 2.85
train_predictions = optR.predict(oof_train, coefficients_).astype(np.int8)
print(f'train pred distribution: {Counter(train_predictions)}')
test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_).astype(np.int8)
print(f'test pred distribution: {Counter(test_predictions)}')


# In[ ]:


print (Counter(train_predictions), Counter(test_predictions))

submission1 = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions})
submission1.to_csv('submission1.csv', index=False)


# ## MODEL 2 for bleding xgboost like 456 pb kernel

# In[ ]:


def run_xgb(params, X_train, X_test):
    n_splits = 10
    verbose_eval = 1000
    num_rounds = 60000
    early_stop = 500
    if 'RescuerID' in X_train.columns:
        X_train = X_train.drop(to_drop_columns, axis=1)
        X_test = X_test.drop(to_drop_columns, axis=1)
        
    if 'Pet_Breed' in X_train.columns:
        X_train = X_train.drop(['Pet_Breed', 'RescuerID_COUNT'], axis=1)
        X_test = X_test.drop(['Pet_Breed', 'RescuerID_COUNT'], axis=1)
        
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))

    i = 0

    for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):

        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)

        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)
        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred

        i += 1
    return model, oof_train, oof_test


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model, oof_train, oof_test = run_xgb(xgb_params, X_train_non_null, X_test_non_null)')


# In[ ]:


optR = OptimizedRounder()
optR.fit(oof_train, X_train['AdoptionSpeed'].values)
coefficients = optR.coefficients()
valid_pred = optR.predict(oof_train, coefficients)
qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)
print("QWK = ", qwk)

coefficients_ = coefficients.copy()
coefficients_[0] = 1.66
coefficients_[1] = 2.13
coefficients_[3] = 2.85
train_predictions = optR.predict(oof_train, coefficients_).astype(np.int8)
print(f'train pred distribution: {Counter(train_predictions)}')
test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_).astype(np.int8)
print(f'test pred distribution: {Counter(test_predictions)}')


print (Counter(train_predictions), Counter(test_predictions))

submission2 = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions})
submission2.to_csv('submission2.csv', index=False)


# ## model 3 LIGHTGBM

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nimport gc\nimport glob\nimport json\nimport matplotlib.pyplot as plt\n\nimport numpy as np\nimport pandas as pd\nimport scipy as sp\nimport lightgbm as lgb\n\nfrom collections import Counter\nfrom functools import partial\nfrom math import sqrt\nfrom joblib import Parallel, delayed\nfrom tqdm import tqdm\nfrom PIL import Image\nfrom sklearn.model_selection import KFold\nfrom sklearn.model_selection import StratifiedKFold\nfrom sklearn.metrics import cohen_kappa_score, mean_squared_error\nfrom sklearn.metrics import confusion_matrix as sk_cmatrix\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.decomposition import SparsePCA, TruncatedSVD, LatentDirichletAllocation, NMF\n\n# basic datasets\ntrain = pd.read_csv(\'../input/petfinder-adoption-prediction/train/train.csv\')\ntest = pd.read_csv(\'../input/petfinder-adoption-prediction/test/test.csv\')\nsample_submission = pd.read_csv(\'../input/petfinder-adoption-prediction/test/sample_submission.csv\')\nlabels_breed = pd.read_csv(\'../input/petfinder-adoption-prediction/breed_labels.csv\')\nlabels_state = pd.read_csv(\'../input/petfinder-adoption-prediction/color_labels.csv\')\nlabels_color = pd.read_csv(\'../input/petfinder-adoption-prediction/state_labels.csv\')\n\ntrain_image_files = sorted(glob.glob(\'../input/petfinder-adoption-prediction/train_images/*.jpg\'))\ntrain_metadata_files = sorted(glob.glob(\'../input/petfinder-adoption-prediction/train_metadata/*.json\'))\ntrain_sentiment_files = sorted(glob.glob(\'../input/petfinder-adoption-prediction/train_sentiment/*.json\'))\ntest_image_files = sorted(glob.glob(\'../input/petfinder-adoption-prediction/test_images/*.jpg\'))\ntest_metadata_files = sorted(glob.glob(\'../input/petfinder-adoption-prediction/test_metadata/*.json\'))\ntest_sentiment_files = sorted(glob.glob(\'../input/petfinder-adoption-prediction/test_sentiment/*.json\'))\n\n# extract datasets\n# https://www.kaggle.com/christofhenkel/extract-image-features-from-pretrained-nn\ntrain_img_features = train_feats\ntest_img_features = test_feats\n\n# img_features columns set names\ncol_names =["PetID"] + ["{}_img_feature".format(_) for _ in range(256)]\ntrain_img_features.columns = col_names\ntest_img_features.columns = col_names\n\n# ref: https://www.kaggle.com/wrosinski/baselinemodeling\nclass PetFinderParser(object):\n    \n    def __init__(self, debug=False):\n        \n        self.debug = debug\n        self.sentence_sep = \' \'\n        \n        # Does not have to be extracted because main DF already contains description\n        self.extract_sentiment_text = False\n        \n        \n    def open_metadata_file(self, filename):\n        """\n        Load metadata file.\n        """\n        with open(filename, \'r\') as f:\n            metadata_file = json.load(f)\n        return metadata_file\n            \n    def open_sentiment_file(self, filename):\n        """\n        Load sentiment file.\n        """\n        with open(filename, \'r\') as f:\n            sentiment_file = json.load(f)\n        return sentiment_file\n            \n    def open_image_file(self, filename):\n        """\n        Load image file.\n        """\n        image = np.asarray(Image.open(filename))\n        return image\n        \n    def parse_sentiment_file(self, file):\n        """\n        Parse sentiment file. Output DF with sentiment features.\n        """\n        \n        file_sentiment = file[\'documentSentiment\']\n        file_entities = [x[\'name\'] for x in file[\'entities\']]\n        file_entities = self.sentence_sep.join(file_entities)\n\n        if self.extract_sentiment_text:\n            file_sentences_text = [x[\'text\'][\'content\'] for x in file[\'sentences\']]\n            file_sentences_text = self.sentence_sep.join(file_sentences_text)\n        file_sentences_sentiment = [x[\'sentiment\'] for x in file[\'sentences\']]\n        \n        file_sentences_sentiment = pd.DataFrame.from_dict(\n            file_sentences_sentiment, orient=\'columns\').sum()\n        file_sentences_sentiment = file_sentences_sentiment.add_prefix(\'document_\').to_dict()\n        \n        file_sentiment.update(file_sentences_sentiment)\n        \n        df_sentiment = pd.DataFrame.from_dict(file_sentiment, orient=\'index\').T\n        if self.extract_sentiment_text:\n            df_sentiment[\'text\'] = file_sentences_text\n            \n        df_sentiment[\'entities\'] = file_entities\n        df_sentiment = df_sentiment.add_prefix(\'sentiment_\')\n        \n        return df_sentiment\n    \n    def parse_metadata_file(self, file):\n        """\n        Parse metadata file. Output DF with metadata features.\n        """\n        \n        file_keys = list(file.keys())\n        \n        if \'labelAnnotations\' in file_keys:\n            file_annots = file[\'labelAnnotations\'][:int(len(file[\'labelAnnotations\']) * 0.3)]\n            file_top_score = np.asarray([x[\'score\'] for x in file_annots]).mean()\n            file_top_desc = [x[\'description\'] for x in file_annots]\n        else:\n            file_top_score = np.nan\n            file_top_desc = [\'\']\n        \n        file_colors = file[\'imagePropertiesAnnotation\'][\'dominantColors\'][\'colors\']\n        file_crops = file[\'cropHintsAnnotation\'][\'cropHints\']\n\n        file_color_score = np.asarray([x[\'score\'] for x in file_colors]).mean()\n        file_color_pixelfrac = np.asarray([x[\'pixelFraction\'] for x in file_colors]).mean()\n\n        file_crop_conf = np.asarray([x[\'confidence\'] for x in file_crops]).mean()\n        \n        if \'importanceFraction\' in file_crops[0].keys():\n            file_crop_importance = np.asarray([x[\'importanceFraction\'] for x in file_crops]).mean()\n        else:\n            file_crop_importance = np.nan\n\n        df_metadata = {\n            \'annots_score\': file_top_score,\n            \'color_score\': file_color_score,\n            \'color_pixelfrac\': file_color_pixelfrac,\n            \'crop_conf\': file_crop_conf,\n            \'crop_importance\': file_crop_importance,\n            \'annots_top_desc\': self.sentence_sep.join(file_top_desc)\n        }\n        \n        df_metadata = pd.DataFrame.from_dict(df_metadata, orient=\'index\').T\n        df_metadata = df_metadata.add_prefix(\'metadata_\')\n        \n        return df_metadata\n    \n\n# Helper function for parallel data processing:\ndef extract_additional_features(pet_id, mode=\'train\'):\n    \n    sentiment_filename = \'../input/petfinder-adoption-prediction/{}_sentiment/{}.json\'.format(mode, pet_id)\n    try:\n        sentiment_file = pet_parser.open_sentiment_file(sentiment_filename)\n        df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)\n        df_sentiment[\'PetID\'] = pet_id\n    except FileNotFoundError:\n        df_sentiment = []\n\n    dfs_metadata = []\n    metadata_filenames = sorted(glob.glob(\'../input/petfinder-adoption-prediction/{}_metadata/{}*.json\'.format(mode, pet_id)))\n    if len(metadata_filenames) > 0:\n        for f in metadata_filenames:\n            metadata_file = pet_parser.open_metadata_file(f)\n            df_metadata = pet_parser.parse_metadata_file(metadata_file)\n            df_metadata[\'PetID\'] = pet_id\n            dfs_metadata.append(df_metadata)\n        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)\n    dfs = [df_sentiment, dfs_metadata]\n    \n    return dfs\n\ndef agg_features(df_metadata, df_sentiment):\n    # Extend aggregates and improve column naming\n    aggregates = [\'mean\', "median", \'sum\', "var", "std", "min", "max", "nunique"]\n    \n    metadata_desc = df_metadata.groupby([\'PetID\'])[\'metadata_annots_top_desc\'].unique()\n    metadata_desc = metadata_desc.reset_index()\n    metadata_desc[\'metadata_annots_top_desc\'] = metadata_desc[\'metadata_annots_top_desc\'].apply(lambda x: \' \'.join(x))\n    \n    prefix = \'metadata\'\n    metadata_gr = df_metadata.drop([\'metadata_annots_top_desc\'], axis=1)\n    for i in metadata_gr.columns:\n        if \'PetID\' not in i:\n            metadata_gr[i] = metadata_gr[i].astype(float)\n    metadata_gr = metadata_gr.groupby([\'PetID\']).agg(aggregates)\n    metadata_gr.columns = pd.Index([\'{}_{}_{}\'.format(prefix, c[0], c[1].upper()) for c in metadata_gr.columns.tolist()])\n    metadata_gr = metadata_gr.reset_index()\n    \n    sentiment_desc = df_sentiment.groupby([\'PetID\'])[\'sentiment_entities\'].unique()\n    sentiment_desc = sentiment_desc.reset_index()\n    sentiment_desc[\'sentiment_entities\'] = sentiment_desc[\'sentiment_entities\'].apply(lambda x: \' \'.join(x))\n    \n    prefix = \'sentiment\'\n    sentiment_gr = df_sentiment.drop([\'sentiment_entities\'], axis=1)\n    for i in sentiment_gr.columns:\n        if \'PetID\' not in i:\n            sentiment_gr[i] = sentiment_gr[i].astype(float)\n    sentiment_gr = sentiment_gr.groupby([\'PetID\']).agg(aggregates)\n    sentiment_gr.columns = pd.Index([\'{}_{}_{}\'.format(\n                prefix, c[0], c[1].upper()) for c in sentiment_gr.columns.tolist()])\n    sentiment_gr = sentiment_gr.reset_index()\n    \n    return sentiment_gr, metadata_gr, metadata_desc, sentiment_desc\n\n\ndef breed_features(df, _labels_breed):\n    breed_main = df[[\'Breed1\']].merge(_labels_breed, how=\'left\', left_on=\'Breed1\', right_on=\'BreedID\', suffixes=(\'\', \'_main_breed\'))\n    breed_main = breed_main.iloc[:, 2:]\n    breed_main = breed_main.add_prefix(\'main_breed_\')\n    \n    breed_second = df[[\'Breed2\']].merge(_labels_breed, how=\'left\', left_on=\'Breed2\', right_on=\'BreedID\', suffixes=(\'\', \'_second_breed\'))\n    breed_second = breed_second.iloc[:, 2:]\n    breed_second = breed_second.add_prefix(\'second_breed_\')\n    \n    return breed_main, breed_second\n\n\ndef impact_coding(data, feature, target=\'y\'):\n    \'\'\'\n    In this implementation we get the values and the dictionary as two different steps.\n    This is just because initially we were ignoring the dictionary as a result variable.\n    \n    In this implementation the KFolds use shuffling. If you want reproducibility the cv \n    could be moved to a parameter.\n    \'\'\'\n    n_folds = 20\n    n_inner_folds = 10\n    impact_coded = pd.Series()\n    \n    oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)\n    kf = KFold(n_splits=n_folds, shuffle=True)\n    oof_mean_cv = pd.DataFrame()\n    split = 0\n    for infold, oof in kf.split(data[feature]):\n            impact_coded_cv = pd.Series()\n            kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)\n            inner_split = 0\n            inner_oof_mean_cv = pd.DataFrame()\n            oof_default_inner_mean = data.iloc[infold][target].mean()\n            for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):\n                # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)\n                oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()\n                impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(\n                            lambda x: oof_mean[x[feature]]\n                                      if x[feature] in oof_mean.index\n                                      else oof_default_inner_mean\n                            , axis=1))\n\n                # Also populate mapping (this has all group -> mean for all inner CV folds)\n                inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how=\'outer\')\n                inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)\n                inner_split += 1\n\n            # Also populate mapping\n            oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how=\'outer\')\n            oof_mean_cv.fillna(value=oof_default_mean, inplace=True)\n            split += 1\n            \n            impact_coded = impact_coded.append(data.iloc[oof].apply(\n                            lambda x: inner_oof_mean_cv.loc[x[feature]].mean()\n                                      if x[feature] in inner_oof_mean_cv.index\n                                      else oof_default_mean\n                            , axis=1))\n\n    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean  \n    \n    \ndef frequency_encoding(df, col_name):\n    new_name = "{}_counts".format(col_name)\n    new_col_name = "{}_freq".format(col_name)\n    grouped = df.groupby(col_name).size().reset_index(name=new_name)\n    df = df.merge(grouped, how = "left", on = col_name)\n    df[new_col_name] = df[new_name]/df[new_name].count()\n    del df[new_name]\n    return df\n    \n\n# FROM: https://www.kaggle.com/myltykritik/simple-lgbm-image-features\n\n# The following 3 functions have been taken from Ben Hamner\'s github repository\n# https://github.com/benhamner/Metrics\ndef confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):\n    """\n    Returns the confusion matrix between rater\'s ratings\n    """\n    assert(len(rater_a) == len(rater_b))\n    if min_rating is None:\n        min_rating = min(rater_a + rater_b)\n    if max_rating is None:\n        max_rating = max(rater_a + rater_b)\n    num_ratings = int(max_rating - min_rating + 1)\n    conf_mat = [[0 for i in range(num_ratings)]\n                for j in range(num_ratings)]\n    for a, b in zip(rater_a, rater_b):\n        conf_mat[a - min_rating][b - min_rating] += 1\n    return conf_mat\n\n\ndef histogram(ratings, min_rating=None, max_rating=None):\n    """\n    Returns the counts of each type of rating that a rater made\n    """\n    if min_rating is None:\n        min_rating = min(ratings)\n    if max_rating is None:\n        max_rating = max(ratings)\n    num_ratings = int(max_rating - min_rating + 1)\n    hist_ratings = [0 for x in range(num_ratings)]\n    for r in ratings:\n        hist_ratings[r - min_rating] += 1\n    return hist_ratings\n\n\ndef quadratic_weighted_kappa(y, y_pred):\n    """\n    Calculates the quadratic weighted kappa\n    axquadratic_weighted_kappa calculates the quadratic weighted kappa\n    value, which is a measure of inter-rater agreement between two raters\n    that provide discrete numeric ratings.  Potential values range from -1\n    (representing complete disagreement) to 1 (representing complete\n    agreement).  A kappa value of 0 is expected if all agreement is due to\n    chance.\n    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b\n    each correspond to a list of integer ratings.  These lists must have the\n    same length.\n    The ratings should be integers, and it is assumed that they contain\n    the complete range of possible ratings.\n    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating\n    is the minimum possible rating, and max_rating is the maximum possible\n    rating\n    """\n    rater_a = y\n    rater_b = y_pred\n    min_rating=None\n    max_rating=None\n    rater_a = np.array(rater_a, dtype=int)\n    rater_b = np.array(rater_b, dtype=int)\n    assert(len(rater_a) == len(rater_b))\n    if min_rating is None:\n        min_rating = min(min(rater_a), min(rater_b))\n    if max_rating is None:\n        max_rating = max(max(rater_a), max(rater_b))\n    conf_mat = confusion_matrix(rater_a, rater_b,\n                                min_rating, max_rating)\n    num_ratings = len(conf_mat)\n    num_scored_items = float(len(rater_a))\n\n    hist_rater_a = histogram(rater_a, min_rating, max_rating)\n    hist_rater_b = histogram(rater_b, min_rating, max_rating)\n\n    numerator = 0.0\n    denominator = 0.0\n\n    for i in range(num_ratings):\n        for j in range(num_ratings):\n            expected_count = (hist_rater_a[i] * hist_rater_b[j]\n                              / num_scored_items)\n            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)\n            numerator += d * conf_mat[i][j] / num_scored_items\n            denominator += d * expected_count / num_scored_items\n\n    return (1.0 - numerator / denominator)\n\nclass OptimizedRounder(object):\n    def __init__(self):\n        self.coef_ = 0\n\n    def _kappa_loss(self, coef, X, y):\n        X_p = np.copy(X)\n        for i, pred in enumerate(X_p):\n            if pred < coef[0]:\n                X_p[i] = 0\n            elif pred >= coef[0] and pred < coef[1]:\n                X_p[i] = 1\n            elif pred >= coef[1] and pred < coef[2]:\n                X_p[i] = 2\n            elif pred >= coef[2] and pred < coef[3]:\n                X_p[i] = 3\n            else:\n                X_p[i] = 4\n\n        ll = quadratic_weighted_kappa(y, X_p)\n        return -ll\n\n    def fit(self, X, y):\n        loss_partial = partial(self._kappa_loss, X=X, y=y)\n        initial_coef = [0.5, 1.5, 2.5, 3.5]\n        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method=\'nelder-mead\')\n\n    def predict(self, X, coef):\n        X_p = np.copy(X)\n        for i, pred in enumerate(X_p):\n            if pred < coef[0]:\n                X_p[i] = 0\n            elif pred >= coef[0] and pred < coef[1]:\n                X_p[i] = 1\n            elif pred >= coef[1] and pred < coef[2]:\n                X_p[i] = 2\n            elif pred >= coef[2] and pred < coef[3]:\n                X_p[i] = 3\n            else:\n                X_p[i] = 4\n        return X_p\n\n    def coefficients(self):\n        return self.coef_[\'x\']\n    \n    \ndef rmse(actual, predicted):\n    return sqrt(mean_squared_error(actual, predicted))\n    \n\ndef train_lightgbm(X_train, X_test, params, n_splits, num_rounds, verbose_eval, early_stop):\n    kfold = StratifiedKFold(n_splits=n_splits, random_state=1337)\n    oof_train = np.zeros((X_train.shape[0]))\n    oof_test = np.zeros((X_test.shape[0], n_splits))\n    \n    i = 0\n    for train_index, valid_index in kfold.split(X_train, X_train[\'AdoptionSpeed\'].values):\n        \n        X_tr = X_train.iloc[train_index, :]\n        X_val = X_train.iloc[valid_index, :]\n        \n        y_tr = X_tr[\'AdoptionSpeed\'].values\n        X_tr = X_tr.drop([\'AdoptionSpeed\'], axis=1)\n        \n        y_val = X_val[\'AdoptionSpeed\'].values\n        X_val = X_val.drop([\'AdoptionSpeed\'], axis=1)\n        \n        print(\'\\ny_tr distribution: {}\'.format(Counter(y_tr)))\n        \n        d_train = lgb.Dataset(X_tr, label=y_tr)\n        d_valid = lgb.Dataset(X_val, label=y_val)\n        watchlist = [d_train, d_valid]\n        \n        print(\'training LGB:\')\n        model = lgb.train(params,\n                          train_set=d_train,\n                          num_boost_round=num_rounds,\n                          valid_sets=watchlist,\n                          verbose_eval=verbose_eval,\n                          early_stopping_rounds=early_stop)\n        \n        val_pred = model.predict(X_val, num_iteration=model.best_iteration)\n        test_pred = model.predict(X_test, num_iteration=model.best_iteration)\n        \n        oof_train[valid_index] = val_pred\n        oof_test[:, i] = test_pred\n        \n        i += 1\n    \n    return oof_train, oof_test\n \n\npet_parser = PetFinderParser() \n  \ndef main():\n    \n    train_pet_ids = train.PetID.unique()\n    test_pet_ids = test.PetID.unique()\n    \n    dfs_train = Parallel(n_jobs=6, verbose=1)(\n    delayed(extract_additional_features)(i, mode=\'train\') for i in train_pet_ids)\n    \n    train_dfs_sentiment = [x[0] for x in dfs_train if isinstance(x[0], pd.DataFrame)]\n    train_dfs_metadata = [x[1] for x in dfs_train if isinstance(x[1], pd.DataFrame)]\n    \n    train_dfs_sentiment = pd.concat(train_dfs_sentiment, ignore_index=True, sort=False)\n    train_dfs_metadata = pd.concat(train_dfs_metadata, ignore_index=True, sort=False)\n    \n    dfs_test = Parallel(n_jobs=6, verbose=1)(\n    delayed(extract_additional_features)(i, mode=\'test\') for i in test_pet_ids)\n    \n    test_dfs_sentiment = [x[0] for x in dfs_test if isinstance(x[0], pd.DataFrame)]\n    test_dfs_metadata = [x[1] for x in dfs_test if isinstance(x[1], pd.DataFrame)]\n    \n    test_dfs_sentiment = pd.concat(test_dfs_sentiment, ignore_index=True, sort=False)\n    test_dfs_metadata = pd.concat(test_dfs_metadata, ignore_index=True, sort=False)\n    \n    train_sentiment_gr, train_metadata_gr, train_metadata_desc, train_sentiment_desc = agg_features(train_dfs_metadata, train_dfs_sentiment) \n    test_sentiment_gr, test_metadata_gr, test_metadata_desc, test_sentiment_desc = agg_features(test_dfs_metadata, test_dfs_sentiment) \n    \n    train_proc = train.copy()\n    for tr in [train_sentiment_gr, train_metadata_gr, train_metadata_desc, train_sentiment_desc]:\n        train_proc = train_proc.merge(tr, how=\'left\', on=\'PetID\')\n    \n    test_proc = test.copy()\n    for ts in [test_sentiment_gr, test_metadata_gr, test_metadata_desc, test_sentiment_desc]:\n        test_proc = test_proc.merge(\n            ts, how=\'left\', on=\'PetID\')\n\n    train_proc = pd.merge(train_proc, train_img_features, on="PetID")\n    test_proc = pd.merge(test_proc, test_img_features, on="PetID")\n    \n    train_breed_main, train_breed_second = breed_features(train_proc, labels_breed)\n    train_proc = pd.concat([train_proc, train_breed_main, train_breed_second], axis=1)\n    \n    test_breed_main, test_breed_second = breed_features(test_proc, labels_breed)\n    test_proc = pd.concat([test_proc, test_breed_main, test_breed_second], axis=1)\n    \n    X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False)\n    column_types = X.dtypes\n\n    int_cols = column_types[column_types == \'int\']\n    float_cols = column_types[column_types == \'float\']\n    cat_cols = column_types[column_types == \'object\']\n    \n    X_temp = X.copy()\n\n    text_columns = [\'Description\', \'metadata_annots_top_desc\', \'sentiment_entities\']\n    categorical_columns = [\'main_breed_BreedName\', \'second_breed_BreedName\']\n\n    to_drop_columns = [\'PetID\', \'Name\', \'RescuerID\']\n    \n    rescuer_count = X.groupby([\'RescuerID\'])[\'PetID\'].count().reset_index()\n    rescuer_count.columns = [\'RescuerID\', \'RescuerID_COUNT\']\n    \n    X_temp = X_temp.merge(rescuer_count, how=\'left\', on=\'RescuerID\')\n    \n    for i in categorical_columns:\n        X_temp.loc[:, i] = pd.factorize(X_temp.loc[:, i])[0]\n        \n    X_text = X_temp[text_columns]\n\n    for i in X_text.columns:\n        X_text.loc[:, i] = X_text.loc[:, i].fillna(\'<MISSING>\')\n        \n    n_components = 5\n    text_features = []\n\n\n    # Generate text features:\n    for i in X_text.columns:\n        \n        # Initialize decomposition methods:\n        print(\'generating features from: {}\'.format(i))\n        svd_ = TruncatedSVD(\n            n_components=n_components, random_state=1337)\n        nmf_ = NMF(\n            n_components=n_components, random_state=1337)\n        \n        tfidf_col = TfidfVectorizer().fit_transform(X_text.loc[:, i].values)\n        svd_col = svd_.fit_transform(tfidf_col)\n        svd_col = pd.DataFrame(svd_col)\n        svd_col = svd_col.add_prefix(\'SVD_{}_\'.format(i))\n        \n        nmf_col = nmf_.fit_transform(tfidf_col)\n        nmf_col = pd.DataFrame(nmf_col)\n        nmf_col = nmf_col.add_prefix(\'NMF_{}_\'.format(i))\n        \n        text_features.append(svd_col)\n        text_features.append(nmf_col)\n    \n        \n    # Combine all extracted features:\n    text_features = pd.concat(text_features, axis=1)\n    \n    # Concatenate with main DF:\n    X_temp = pd.concat([X_temp, text_features], axis=1)\n    \n    # Remove raw text columns:\n    for i in X_text.columns:\n        X_temp = X_temp.drop(i, axis=1)\n    \n    X_temp["name_length"] = X_temp.Name[X_temp.Name.isnull()].map(lambda x: len(str(x)))\n    X_temp["name_length"] = X_temp.Name.map(lambda x: len(str(x)))\n    X_temp = X_temp.drop(to_drop_columns, axis=1)\n    \n    # Split into train and test again:\n    X_train = X_temp.loc[np.isfinite(X_temp.AdoptionSpeed), :]\n    X_test = X_temp.loc[~np.isfinite(X_temp.AdoptionSpeed), :]\n    \n    # Remove missing target column from test:\n    X_test = X_test.drop([\'AdoptionSpeed\'], axis=1)\n    \n    \n    print(\'X_train shape: {}\'.format(X_train.shape))\n    print(\'X_test shape: {}\'.format(X_test.shape))\n    \n    assert X_train.shape[0] == train.shape[0]\n    assert X_test.shape[0] == test.shape[0]\n    \n    \n    # Check if columns between the two DFs are the same:\n    train_cols = X_train.columns.tolist()\n    train_cols.remove(\'AdoptionSpeed\')\n    \n    test_cols = X_test.columns.tolist()\n    \n    np.random.seed(13)\n    \n    categorical_features = ["Type", "Breed1", "Breed2", "Color1" ,"Color2", "Color3", "State"]\n    \n    impact_coding_map = {}\n    for f in categorical_features:\n        print("Impact coding for {}".format(f))\n        X_train["impact_encoded_{}".format(f)], impact_coding_mapping, default_coding = impact_coding(X_train, f, target="AdoptionSpeed")\n        impact_coding_map[f] = (impact_coding_mapping, default_coding)\n        mapping, default_mean = impact_coding_map[f]\n        X_test["impact_encoded_{}".format(f)] = X_test.apply(lambda x: mapping[x[f]] if x[f] in mapping\n                                                         else default_mean, axis=1)\n\n    for cat in categorical_features:\n        X_train = frequency_encoding(X_train, cat)\n        X_test = frequency_encoding(X_test, cat)\n\n    params = {\'application\': \'regression\',\n          \'boosting\': \'gbdt\',\n          \'metric\': \'rmse\',\n          \'num_leaves\': 70,\n          \'max_depth\': 9,\n          \'learning_rate\': 0.01,\n          \'bagging_fraction\': 0.85,\n          \'feature_fraction\': 0.8,\n          \'min_split_gain\': 0.02,\n          \'min_child_samples\': 150,\n          \'min_child_weight\': 0.02,\n          \'lambda_l2\': 0.0475,\n          \'verbosity\': -1,\n          \'data_random_seed\': 17}\n\n    # Additional parameters:\n    early_stop = 500\n    verbose_eval = 100\n    num_rounds = 10000\n    n_splits = 5\n    \n    oof_train, oof_test = train_lightgbm(X_train, X_test, params, n_splits, num_rounds, verbose_eval, early_stop)\n    optR = OptimizedRounder()\n    optR.fit(oof_train, X_train[\'AdoptionSpeed\'].values)\n    coefficients = optR.coefficients()\n    pred_test_y_k = optR.predict(oof_train, coefficients)\n    print("\\nValid Counts = ", Counter(X_train[\'AdoptionSpeed\'].values))\n    print("Predicted Counts = ", Counter(pred_test_y_k))\n    print("Coefficients = ", coefficients)\n    qwk = quadratic_weighted_kappa(X_train[\'AdoptionSpeed\'].values, pred_test_y_k)\n    print("QWK = ", qwk)\n    \n    # Manually adjusted coefficients:\n    coefficients_ = coefficients.copy()\n    \n    coefficients_[0] = 1.645\n    coefficients_[1] = 2.115\n    coefficients_[3] = 2.84\n    \n    train_predictions = optR.predict(oof_train, coefficients_).astype(int)\n    print(\'train pred distribution: {}\'.format(Counter(train_predictions)))\n    \n    test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_)\n    print(\'test pred distribution: {}\'.format(Counter(test_predictions)))\n    \n    # Generate submission:\n    submission = pd.DataFrame({\'PetID\': test[\'PetID\'].values, \'AdoptionSpeed\': test_predictions.astype(np.int32)})\n    return submission\nsubmission3 =  main()\nsubmission3.to_csv(\'submission3.csv\', index=False)')


# ## MODEL4 460 with kfold

# In[ ]:


get_ipython().run_cell_magic('time', '', 'def run_xgb(params, X_train, X_test):\n    n_splits = 10\n    verbose_eval = 1000\n    num_rounds = 60000\n    early_stop = 500\n    if \'RescuerID\' in X_train.columns:\n        X_train = X_train.drop(to_drop_columns, axis=1)\n        X_test = X_test.drop(to_drop_columns, axis=1)\n        \n    if \'Pet_Breed\' in X_train.columns:\n        X_train = X_train.drop([ \'RescuerID_COUNT\'], axis=1)\n        X_test = X_test.drop([ \'RescuerID_COUNT\'], axis=1)\n        \n    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)\n\n    oof_train = np.zeros((X_train.shape[0]))\n    oof_test = np.zeros((X_test.shape[0], n_splits))\n\n    i = 0\n\n    for train_idx, valid_idx in kf.split(X_train, X_train[\'AdoptionSpeed\'].values):\n\n        X_tr = X_train.iloc[train_idx, :]\n        X_val = X_train.iloc[valid_idx, :]\n\n        y_tr = X_tr[\'AdoptionSpeed\'].values\n        X_tr = X_tr.drop([\'AdoptionSpeed\'], axis=1)\n\n        y_val = X_val[\'AdoptionSpeed\'].values\n        X_val = X_val.drop([\'AdoptionSpeed\'], axis=1)\n\n        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)\n        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)\n\n        watchlist = [(d_train, \'train\'), (d_valid, \'valid\')]\n        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,\n                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)\n\n        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)\n        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)\n\n        oof_train[valid_idx] = valid_pred\n        oof_test[:, i] = test_pred\n\n        i += 1\n    return model, oof_train, oof_test\n\n\nmodel, oof_train, oof_test = run_xgb(xgb_params, X_train_non_null, X_test_non_null)\n\noptR = OptimizedRounder()\noptR.fit(oof_train, X_train[\'AdoptionSpeed\'].values)\ncoefficients = optR.coefficients()\nvalid_pred = optR.predict(oof_train, coefficients)\nqwk = quadratic_weighted_kappa(X_train[\'AdoptionSpeed\'].values, valid_pred)\nprint("QWK = ", qwk)\n\ncoefficients_ = coefficients.copy()\ncoefficients_[0] = 1.66\ncoefficients_[1] = 2.13\ncoefficients_[3] = 2.85\ntrain_predictions = optR.predict(oof_train, coefficients_).astype(np.int8)\nprint(f\'train pred distribution: {Counter(train_predictions)}\')\ntest_predictions = optR.predict(oof_test.mean(axis=1), coefficients_).astype(np.int8)\nprint(f\'test pred distribution: {Counter(test_predictions)}\')\n\n\nprint (Counter(train_predictions), Counter(test_predictions))\n\nsubmission4 = pd.DataFrame({\'PetID\': test[\'PetID\'].values, \'AdoptionSpeed\': test_predictions})\nsubmission4.to_csv(\'submission4.csv\', index=False)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'def run_xgb(params, X_train, X_test):\n    n_splits = 10\n    verbose_eval = 1000\n    num_rounds = 60000\n    early_stop = 500\n    if \'RescuerID\' in X_train.columns:\n        X_train = X_train.drop(to_drop_columns, axis=1)\n        X_test = X_test.drop(to_drop_columns, axis=1)\n        \n    if \'Pet_Breed\' in X_train.columns:\n        X_train = X_train.drop([ \'RescuerID_COUNT\'], axis=1)\n        X_test = X_test.drop([ \'RescuerID_COUNT\'], axis=1)\n        \n    kf = KFold(n_splits=n_splits, shuffle=True, random_state=34535)\n\n    oof_train = np.zeros((X_train.shape[0]))\n    oof_test = np.zeros((X_test.shape[0], n_splits))\n\n    i = 0\n\n    for train_idx, valid_idx in kf.split(X_train, X_train[\'AdoptionSpeed\'].values):\n\n        X_tr = X_train.iloc[train_idx, :]\n        X_val = X_train.iloc[valid_idx, :]\n\n        y_tr = X_tr[\'AdoptionSpeed\'].values\n        X_tr = X_tr.drop([\'AdoptionSpeed\'], axis=1)\n\n        y_val = X_val[\'AdoptionSpeed\'].values\n        X_val = X_val.drop([\'AdoptionSpeed\'], axis=1)\n\n        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)\n        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)\n\n        watchlist = [(d_train, \'train\'), (d_valid, \'valid\')]\n        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,\n                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)\n\n        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)\n        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)\n\n        oof_train[valid_idx] = valid_pred\n        oof_test[:, i] = test_pred\n\n        i += 1\n    return model, oof_train, oof_test\n\n\nmodel, oof_train, oof_test = run_xgb(xgb_params, X_train_non_null, X_test_non_null)\n\noptR = OptimizedRounder()\noptR.fit(oof_train, X_train[\'AdoptionSpeed\'].values)\ncoefficients = optR.coefficients()\nvalid_pred = optR.predict(oof_train, coefficients)\nqwk = quadratic_weighted_kappa(X_train[\'AdoptionSpeed\'].values, valid_pred)\nprint("QWK = ", qwk)\n\ncoefficients_ = coefficients.copy()\ncoefficients_[0] = 1.66\ncoefficients_[1] = 2.13\ncoefficients_[3] = 2.85\ntrain_predictions = optR.predict(oof_train, coefficients_).astype(np.int8)\nprint(f\'train pred distribution: {Counter(train_predictions)}\')\ntest_predictions = optR.predict(oof_test.mean(axis=1), coefficients_).astype(np.int8)\nprint(f\'test pred distribution: {Counter(test_predictions)}\')\n\n\nprint (Counter(train_predictions), Counter(test_predictions))\n\nsubmission5 = pd.DataFrame({\'PetID\': test[\'PetID\'].values, \'AdoptionSpeed\': test_predictions})\nsubmission5.to_csv(\'submission5.csv\', index=False)')


# # FINAL BLENDING

# In[ ]:


#subs 1 460 (groupfold), sub2 453(sfold ), sub3 lightgbm(sfold), sub4 xgb460 kfold
submission1["AdoptionSpeed"] = submission1.AdoptionSpeed*0.4 + submission4.AdoptionSpeed*0.1 + submission2.AdoptionSpeed*0.20 + submission3.AdoptionSpeed*0.20+submission5.AdoptionSpeed*0.1
submission1["AdoptionSpeed"] = submission1["AdoptionSpeed"].round().astype(int)
submission1.to_csv('submission.csv', index=False)


# In[ ]:


import datetime
datetime.datetime.now()


# In[ ]:




