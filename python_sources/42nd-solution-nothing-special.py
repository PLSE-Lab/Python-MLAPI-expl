#!/usr/bin/env python
# coding: utf-8

# This result was actually a big surprise. We were at ~260th in the public leaderboard and I was kind of resigned to us having done something critically wrong (we did). Conversely our "A" team from the meetup was sitting pretty at 42nd. Looks like the major trap they fell into was target encoding. 

# The major structure of our code is very similar to other people's. As a general overview we did 4 models: NN, xLearn FFM, LGBM, XGBoost. We didn't have time to individually test the performance on leaderboard of each, but we have QWK CV numbers for each:
# * NN - .416
# * xLearn FFM - .418
# * LGBM - .443
# * XGBoost - ..445
# 
# Total ensemble(Ridge Regression) CV - .46
# 
# Many of our features were the same as everyone else's, but we also had a couple unique ones and did some interesting things for feature selection. 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from math import sqrt
from sklearn.metrics import cohen_kappa_score as kappa_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.metrics import mean_squared_error
import scipy as sp
from skimage import feature
from collections import defaultdict
from collections import Counter
import operator
import cv2
from scipy.stats import itemfreq
from joblib import Parallel, delayed
import json
import string
from sklearn.model_selection import StratifiedKFold, GroupKFold
from PIL import Image as IMG
from PIL import Image
import glob
kappa_scorer = None

import os
os.environ['USER'] = 'root'
os.system('pip install ../input/xlearn/xlearn/xlearn-0.40a1/')

import xlearn as xl
print(os.listdir("../input"))
print(os.listdir("../input/petfinder-adoption-prediction/"))
print(os.listdir("../input/petfinder-adoption-prediction/train"))
print(os.listdir("../input/petfinder-adoption-prediction/test"))


# In[ ]:


train_df = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")
test_df = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")


# In[ ]:


y = train_df["AdoptionSpeed"]


# In[ ]:


labels_breed = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')


# We toyed around with the cats and dog breeds rating files but found the coverage was very poor. We tried to do some fuzzy matching and remapping, but didn't end up completing this to get enough coverage to really be useful. This was just leftover code. 

# In[ ]:


# add features from ratings 
with open('../input/cat-and-dog-breeds-parameters/rating.json', 'r') as f:
    ratings = json.load(f)
cat_ratings = ratings['cat_breeds']
dog_ratings = ratings['dog_breeds']
breed_id = {}


# In[ ]:


pet_attributes = set(list(cat_ratings["Abyssinian"].keys()) + list(dog_ratings["German Shepherd Dog"].keys()))


# In[ ]:


for i in pet_attributes:
    train_df[i] = 0
    test_df[i] = 0


# In[ ]:


cat_ratings = {k.lower(): v for k, v in cat_ratings.items()}
dog_ratings = {k.lower(): v for k, v in dog_ratings.items()}


# In[ ]:


for id,name in zip(labels_breed.BreedID,labels_breed.BreedName):
    breed_id[id] = str(name).lower()
breed_names_1 = [i for i in cat_ratings.keys()]
breed_names_2 = [i for i in dog_ratings.keys()]
for i, id in enumerate(train_df['Breed1']):
    if id in breed_id.keys(): 
        name = str(breed_id[id]).lower() 
        if name in breed_names_1:
            #print(cat_ratings[name])
            for key in cat_ratings[name].keys():
                #print(key)
                train_df.loc[i, key] = cat_ratings[name][key]
        if name in breed_names_2:
            #print(dog_ratings[name])
            for key in dog_ratings[name].keys():
                
                train_df.loc[i, key] = dog_ratings[name][key]
                
                
                
for i, id in enumerate(test_df['Breed1']):
    if id in breed_id.keys(): 
        str(breed_id[id]).lower() 
        if name in breed_names_1:
            #print(cat_ratings[name])
            for key in cat_ratings[name].keys():
                #print(key)
                test_df.loc[i, key] = cat_ratings[name][key]
        if name in breed_names_2:
            #print(dog_ratings[name])
            for key in dog_ratings[name].keys():
                
                test_df.loc[i, key] = dog_ratings[name][key]


# In[ ]:


train_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'))
train_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_metadata/*.json'))
train_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_sentiment/*.json'))

print('num of train images files: {}'.format(len(train_image_files)))
print('num of train metadata files: {}'.format(len(train_metadata_files)))
print('num of train sentiment files: {}'.format(len(train_sentiment_files)))


test_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg'))
test_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_metadata/*.json'))
test_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_sentiment/*.json'))

print('num of test images files: {}'.format(len(test_image_files)))
print('num of test metadata files: {}'.format(len(test_metadata_files)))
print('num of test sentiment files: {}'.format(len(test_sentiment_files)))


# In[ ]:


# Images:
train_df_ids = train_df[['PetID']]
print(train_df_ids.shape)

train_df_imgs = pd.DataFrame(train_image_files)
train_df_imgs.columns = ['image_filename']
train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)
print(len(train_imgs_pets.unique()))

pets_with_images = len(np.intersect1d(train_imgs_pets.unique(), train_df_ids['PetID'].unique()))
print('fraction of pets with images: {:.3f}'.format(pets_with_images / train_df_ids.shape[0]))

# Metadata:
train_df_ids = train_df[['PetID']]
train_df_metadata = pd.DataFrame(train_metadata_files)
train_df_metadata.columns = ['metadata_filename']
train_metadata_pets = train_df_metadata['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
train_df_metadata = train_df_metadata.assign(PetID=train_metadata_pets)
print(len(train_metadata_pets.unique()))

pets_with_metadatas = len(np.intersect1d(train_metadata_pets.unique(), train_df_ids['PetID'].unique()))
print('fraction of pets with metadata: {:.3f}'.format(pets_with_metadatas / train_df_ids.shape[0]))

# Sentiment:
train_df_ids = train_df[['PetID']]
train_df_sentiment = pd.DataFrame(train_sentiment_files)
train_df_sentiment.columns = ['sentiment_filename']
train_sentiment_pets = train_df_sentiment['sentiment_filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
train_df_sentiment = train_df_sentiment.assign(PetID=train_sentiment_pets)
print(len(train_sentiment_pets.unique()))

pets_with_sentiments = len(np.intersect1d(train_sentiment_pets.unique(), train_df_ids['PetID'].unique()))
print('fraction of pets with sentiment: {:.3f}'.format(pets_with_sentiments / train_df_ids.shape[0]))


# In[ ]:


# Images:
test_df_ids = test_df[['PetID']]
print(test_df_ids.shape)

test_df_imgs = pd.DataFrame(test_image_files)
test_df_imgs.columns = ['image_filename']
test_imgs_pets = test_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
test_df_imgs = test_df_imgs.assign(PetID=test_imgs_pets)
print(len(test_imgs_pets.unique()))

pets_with_images = len(np.intersect1d(test_imgs_pets.unique(), test_df_ids['PetID'].unique()))
print('fraction of pets with images: {:.3f}'.format(pets_with_images / test_df_ids.shape[0]))


# Metadata:
test_df_ids = test_df[['PetID']]
test_df_metadata = pd.DataFrame(test_metadata_files)
test_df_metadata.columns = ['metadata_filename']
test_metadata_pets = test_df_metadata['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
test_df_metadata = test_df_metadata.assign(PetID=test_metadata_pets)
print(len(test_metadata_pets.unique()))

pets_with_metadatas = len(np.intersect1d(test_metadata_pets.unique(), test_df_ids['PetID'].unique()))
print('fraction of pets with metadata: {:.3f}'.format(pets_with_metadatas / test_df_ids.shape[0]))



# Sentiment:
test_df_ids = test_df[['PetID']]
test_df_sentiment = pd.DataFrame(test_sentiment_files)
test_df_sentiment.columns = ['sentiment_filename']
test_sentiment_pets = test_df_sentiment['sentiment_filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
test_df_sentiment = test_df_sentiment.assign(PetID=test_sentiment_pets)
print(len(test_sentiment_pets.unique()))

pets_with_sentiments = len(np.intersect1d(test_sentiment_pets.unique(), test_df_ids['PetID'].unique()))
print('fraction of pets with sentiment: {:.3f}'.format(pets_with_sentiments / test_df_ids.shape[0]))


# are distributions the same?
print('images and metadata distributions the same? {}'.format(
    np.all(test_metadata_pets == test_imgs_pets)))


# In[ ]:


class PetFinderParser(object):
    
    def __init__(self, debug=False):
        
        self.debug = debug
        self.sentence_sep = ' '
        
        # Does not have to be extracted because main DF already contains description
        self.extract_sentiment_text = False
        
        
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
        
        return df_sentiment
    
    def parse_metadata_file(self, file):
        """
        Parse metadata file. Output DF with metadata features.
        """
        
        file_keys = list(file.keys())
        
        if 'labelAnnotations' in file_keys:
            file_annots = file['labelAnnotations'][:int(len(file['labelAnnotations']) * 0.3)]
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
    

# Helper function for parallel data processing:
def extract_additional_features(pet_id, mode='train'):
    
    sentiment_filename = '../input/petfinder-adoption-prediction/{}_sentiment/{}.json'.format(mode, pet_id)
    try:
        sentiment_file = pet_parser.open_sentiment_file(sentiment_filename)
        df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)
        df_sentiment['PetID'] = pet_id
    except:
        df_sentiment = []

    dfs_metadata = []
    metadata_filenames = sorted(glob.glob('../input/petfinder-adoption-prediction/{}_metadata/{}*.json'.format(mode, pet_id)))
    if len(metadata_filenames) > 0:
        for f in metadata_filenames:
            metadata_file = pet_parser.open_metadata_file(f)
            df_metadata = pet_parser.parse_metadata_file(metadata_file)
            df_metadata['PetID'] = pet_id
            dfs_metadata.append(df_metadata)
        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)
    dfs = [df_sentiment, dfs_metadata]
    
    return dfs


pet_parser = PetFinderParser()


# We used the same json parser as most other people. We found at the end that many people made some slight alterations that made this process way faster. Time wasn't really an issue for the submission, but it would have helped our prototyping phase if we didnt have to do the 20 minute wait for this everytime. 

# In[ ]:


# Unique IDs from train and test:
debug = False
train_pet_ids = train_df.PetID.unique()
test_pet_ids = test_df.PetID.unique()

if debug:
    train_pet_ids = train_pet_ids[:1000]
    test_pet_ids = test_pet_ids[:500]


# Train set:
# Parallel processing of data:
dfs_train = Parallel(n_jobs=6, verbose=1)(
    delayed(extract_additional_features)(i, mode='train') for i in train_pet_ids)

# Extract processed data and format them as DFs:
train_dfs_sentiment = [x[0] for x in dfs_train if isinstance(x[0], pd.DataFrame)]
train_dfs_metadata = [x[1] for x in dfs_train if isinstance(x[1], pd.DataFrame)]

train_dfs_sentiment = pd.concat(train_dfs_sentiment, ignore_index=True, sort=False)
train_dfs_metadata = pd.concat(train_dfs_metadata, ignore_index=True, sort=False)

print(train_dfs_sentiment.shape, train_dfs_metadata.shape)


# Test set:
# Parallel processing of data:
dfs_test = Parallel(n_jobs=6, verbose=1)(
    delayed(extract_additional_features)(i, mode='test') for i in test_pet_ids)

# Extract processed data and format them as DFs:
test_dfs_sentiment = [x[0] for x in dfs_test if isinstance(x[0], pd.DataFrame)]
test_dfs_metadata = [x[1] for x in dfs_test if isinstance(x[1], pd.DataFrame)]

test_dfs_sentiment = pd.concat(test_dfs_sentiment, ignore_index=True, sort=False)
test_dfs_metadata = pd.concat(test_dfs_metadata, ignore_index=True, sort=False)

print(test_dfs_sentiment.shape, test_dfs_metadata.shape)


# In[1]:


like most other people we did a bunch of aggregates on various features.


# In[ ]:


# Extend aggregates and improve column naming
aggregates = ['mean', 'sum', 'var', 'max', 'min']

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
train_metadata_gr.columns = pd.Index(['{}_{}_{}'.format(
            prefix, c[0], c[1].upper()) for c in train_metadata_gr.columns.tolist()])
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
train_sentiment_gr = train_sentiment_gr.groupby(['PetID']).agg(aggregates)
train_sentiment_gr.columns = pd.Index(['{}_{}_{}'.format(
            prefix, c[0], c[1].upper()) for c in train_sentiment_gr.columns.tolist()])
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
test_metadata_gr.columns = pd.Index(['{}_{}_{}'.format(
            prefix, c[0], c[1].upper()) for c in test_metadata_gr.columns.tolist()])
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
test_sentiment_gr = test_sentiment_gr.groupby(['PetID']).agg(aggregates)
test_sentiment_gr.columns = pd.Index(['{}_{}_{}'.format(
            prefix, c[0], c[1].upper()) for c in test_sentiment_gr.columns.tolist()])
test_sentiment_gr = test_sentiment_gr.reset_index()


# In[ ]:


# Train merges:
train_proc = train_df.copy()
train_proc = train_proc.merge(
    train_sentiment_gr, how='left', on='PetID')
train_proc = train_proc.merge(
    train_metadata_gr, how='left', on='PetID')
train_proc = train_proc.merge(
    train_metadata_desc, how='left', on='PetID')
train_proc = train_proc.merge(
    train_sentiment_desc, how='left', on='PetID')

# Test merges:
test_proc = test_df.copy()
test_proc = test_proc.merge(
    test_sentiment_gr, how='left', on='PetID')
test_proc = test_proc.merge(
    test_metadata_gr, how='left', on='PetID')
test_proc = test_proc.merge(
    test_metadata_desc, how='left', on='PetID')
test_proc = test_proc.merge(
    test_sentiment_desc, how='left', on='PetID')


print(train_proc.shape, test_proc.shape)
# assert train_proc.shape[0] == train_df.shape[0]
# assert test_proc.shape[0] == test_df.shape[0]


# In[ ]:


X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False, axis = 0)
print('NaN structure:\n{}'.format(np.sum(pd.isnull(X))))


# In[ ]:


X_temp = X.copy()
text_columns = ['Description', 'metadata_annots_top_desc', 'sentiment_entities']


# We did RescuerID_Count like most other people and we also found an additional feature, number of unique breeds per rescuerID that gave us a small boost as well. The intuition behind this is that rescuer ID count tells you the quantity of pets, big corporation or individual and the number of unique breeds likely gives you a more clear signal of if they were breeders of one specific dog or not. 
# 
# I think we likely could have created a lot more features around this, but ultimately we just ran out of time and didn't create as many features as most other people. 

# In[ ]:


# Count RescuerID occurrences:
rescuer_count = X.groupby(['RescuerID'])['PetID'].count().reset_index()
rescuer_count.columns = ['RescuerID', 'RescuerID_COUNT']

# Merge as another feature onto main DF:
X_temp = X_temp.merge(rescuer_count, how='left', on='RescuerID')

# Count Unique breed occurrences:
unique_count = X.groupby(['RescuerID'])['Breed1'].nunique().reset_index()
unique_count.columns = ['RescuerID', 'RescuerID_UNIQUE']

# Merge as another feature onto main DF:
X_temp = X_temp.merge(unique_count, how='left', on='RescuerID')


# Subset text features:
X_text = X_temp[text_columns]

for i in X_text.columns:
    X_text.loc[:, i] = X_text.loc[:, i].fillna('<MISSING>')


# We used the same process of tfidf and then svd or other dimensionality reduction tricks, but we decided to do much smaller component count than most other people. This was informed by our feature selection process I will describe in more detail later. 

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import SparsePCA, TruncatedSVD, LatentDirichletAllocation, NMF

n_components = 16
text_features = []


# Generate text features:
for i in X_text.columns:
    
    # Initialize decomposition methods:
    print('generating features from: {}'.format(i))
    svd_ = TruncatedSVD(
        n_components=n_components, random_state=1337)
    
    tfidf_col = TfidfVectorizer().fit_transform(X_text.loc[:, i].values)
    svd_col = svd_.fit_transform(tfidf_col)
    svd_col = pd.DataFrame(svd_col)
    svd_col = svd_col.add_prefix('SVD_{}_'.format(i))
    text_features.append(svd_col)

    
# Combine all extracted features:
text_features = pd.concat(text_features, axis=1)

# Concatenate with main DF:
X_temp = pd.concat([X_temp.reset_index(drop = True), text_features], axis=1)
text_columns = ['metadata_annots_top_desc', 'sentiment_entities']
# Remove raw text columns:
for i in text_columns:
    X_temp = X_temp.drop(i, axis=1)


# In[ ]:


train_df = X_temp.loc[np.isfinite(X_temp.AdoptionSpeed), :]
test_df = X_temp.loc[~np.isfinite(X_temp.AdoptionSpeed), :]

# Remove missing target column from test:
test_df = test_df.drop(['AdoptionSpeed'], axis=1)


# Our lists of categorical features and numerical features. In an ideal world we would have had time to tune each model individually with the best features for each but again we didn't really have enough time to truly hone in on each one indivudally so this is basically our feature set with maybe one or two alterations between models. 

# In[ ]:


cat_cols = ['Breed1','Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'State', 'label_description']


# In[ ]:


num_cols = ['Fee', 'Age', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
 'Sterilized', 'Health', 'Quantity', 'VideoAmt', 'PhotoAmt',
 'Type', 'doc_sent_mag', 'doc_sent_score', 'vertex_x', 'vertex_y',
 'bounding_confidence', 'bounding_importance', 'dominant_blue', 'dominant_green',
 'dominant_red', 'label_score', 'state_gdp', 'state_population', 
 'sentiment_sentiment_magnitude_MEAN', 'sentiment_sentiment_magnitude_SUM',
 'sentiment_sentiment_score_MEAN', 'sentiment_sentiment_score_SUM', 
 'sentiment_sentiment_document_magnitude_MEAN', 'sentiment_sentiment_document_magnitude_SUM',
 'sentiment_sentiment_document_score_MEAN', 'sentiment_sentiment_document_score_SUM',
 'metadata_metadata_annots_score_MEAN', 'metadata_metadata_annots_score_SUM', 'metadata_metadata_color_score_MEAN',
 'metadata_metadata_color_score_SUM', 'metadata_metadata_color_pixelfrac_MEAN', 'metadata_metadata_color_pixelfrac_SUM',
 'metadata_metadata_crop_conf_MEAN', 'metadata_metadata_crop_conf_SUM', 'metadata_metadata_crop_importance_MEAN',
 'metadata_metadata_crop_importance_SUM', 'RescuerID_COUNT', 
 'agg_mean_Age', 'agg_mean_Gender', 'agg_mean_Color1',
 'agg_mean_Color2', 'agg_mean_Color3', 'agg_mean_FurLength', 'agg_mean_Vaccinated', 'agg_mean_Dewormed',
 'agg_mean_Sterilized', 'agg_mean_Health', 'agg_mean_Quantity', 'agg_mean_Fee', 'agg_mean_PhotoAmt',
 'agg_mean_VideoAmt', 'agg_std_Age', 'agg_std_Gender', 'agg_std_Color1', 'agg_std_Color2', 
 'agg_std_Color3', 'agg_std_FurLength', 'agg_std_Vaccinated', 'agg_std_Dewormed', 'agg_std_Sterilized',
 'agg_std_Health', 'agg_std_Quantity', 'agg_std_Fee', 'agg_std_PhotoAmt', 'agg_std_VideoAmt', 'name_len', 
'average_word_length', 'num_words', 'desc_char_length', 'RescuerID_UNIQUE'] 


# In[ ]:


text_cols = ['Description']


# ## Handling categorical columns

# In[ ]:


import json
train_id = train_df['PetID']
test_id = test_df['PetID']
doc_sent_mag = []
doc_sent_score = []
nf_count = 0
for pet in train_id:
    try:
        with open('../input/petfinder-adoption-prediction/train_sentiment/' + pet + '.json', 'r', encoding = "utf-8") as f:
            sentiment = json.load(f)
        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
        doc_sent_score.append(sentiment['documentSentiment']['score'])
    except:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)

train_df.loc[:, 'doc_sent_mag'] = doc_sent_mag
train_df.loc[:, 'doc_sent_score'] = doc_sent_score

doc_sent_mag = []
doc_sent_score = []
nf_count = 0
for pet in test_id:
    try:
        with open('../input/petfinder-adoption-prediction/test_sentiment/' + pet + '.json', 'r', encoding = "utf-8") as f:
            sentiment = json.load(f)
        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
        doc_sent_score.append(sentiment['documentSentiment']['score'])
    except:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)

test_df.loc[:, 'doc_sent_mag'] = doc_sent_mag
test_df.loc[:, 'doc_sent_score'] = doc_sent_score
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
        with open('../input/petfinder-adoption-prediction/train_metadata/' + pet + '-1.json', 'r', encoding = "utf-8") as f:
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
train_df.loc[:, 'vertex_x'] = vertex_xs
train_df.loc[:, 'vertex_y'] = vertex_ys
train_df.loc[:, 'bounding_confidence'] = bounding_confidences
train_df.loc[:, 'bounding_importance'] = bounding_importance_fracs
train_df.loc[:, 'dominant_blue'] = dominant_blues
train_df.loc[:, 'dominant_green'] = dominant_greens
train_df.loc[:, 'dominant_red'] = dominant_reds
train_df.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
train_df.loc[:, 'dominant_score'] = dominant_scores
train_df.loc[:, 'label_description'] = label_descriptions
train_df.loc[:, 'label_score'] = label_scores


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
        with open('../input/petfinder-adoption-prediction/test_metadata/' + pet + '-1.json', 'r', encoding = "utf-8") as f:
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
test_df.loc[:, 'vertex_x'] = vertex_xs
test_df.loc[:, 'vertex_y'] = vertex_ys
test_df.loc[:, 'bounding_confidence'] = bounding_confidences
test_df.loc[:, 'bounding_importance'] = bounding_importance_fracs
test_df.loc[:, 'dominant_blue'] = dominant_blues
test_df.loc[:, 'dominant_green'] = dominant_greens
test_df.loc[:, 'dominant_red'] = dominant_reds
test_df.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
test_df.loc[:, 'dominant_score'] = dominant_scores
test_df.loc[:, 'label_description'] = label_descriptions
test_df.loc[:, 'label_score'] = label_scores


# In[ ]:


print(train_df.shape, test_df.shape)
from sklearn.preprocessing import LabelEncoder


# ## Handling text columns

# In[ ]:


import tensorflow


# In[ ]:


from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


print('getting embeddings')
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in tqdm(open('../input/fasttext-english-word-vectors-including-subwords/wiki-news-300d-1M-subword.vec', encoding= "utf-8")))


# In[ ]:


num_words = 200000
maxlen = 200
embed_size = 300


# In[ ]:


test_df = test_df.reset_index(drop = True)


# Had to reload things and reorganize some stuff. This was mostly because at the beginning we would use some preprocessed input data instead of recomputing things everytime. Because when we made our precomputed input file I forgot we had tokenized inplace I needed to load in the descriptions again. This section is more correcting errors than doing anything deliberate and intelligent. 

# In[ ]:


train_df.loc[:, "Description"] = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")["Description"]
test_df.loc[:, "Description"] = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")["Description"]


# In[ ]:


train_df['Description'] = train_df['Description'].astype(str).fillna('no text')
test_df['Description'] = test_df['Description'].astype(str).fillna('no text')
train_df['Description1'] = train_df['Description'].astype(str).fillna('no text')
test_df['Description1'] = test_df['Description'].astype(str).fillna('no text')


# In[ ]:


train_df['Description1'] = train_df['Description1'].astype(str).fillna('no text')
test_df['Description1'] = test_df['Description1'].astype(str).fillna('no text')


# In[ ]:


print("Fitting tokenizer...")
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_df['Description'].values.tolist() + test_df['Description'].values.tolist())


# In[ ]:


train_df['Description'] = tokenizer.texts_to_sequences(train_df['Description'])
test_df['Description'] = tokenizer.texts_to_sequences(test_df['Description'])


# Some simple text features we added in. Didnt seem to make any real difference but werent hurting anything so left them in. 

# In[ ]:



avg_word_length = []
desc_length = []
number_words = []
for desc in train_df["Description1"]:
    desc_length.append(len(desc))
    words = desc.split()
    number_words.append(len(words))
    word_len = 0
    for word in words:
        word_len += len(word)
    avg_word_length.append(word_len / len(words))
train_df["average_word_length"] = avg_word_length
train_df["num_words"] = number_words
train_df["desc_char_length"] = desc_length


# In[ ]:


avg_word_length = []
desc_length = []
number_words = []
for desc in test_df["Description1"]:
    desc_length.append(len(desc))
    words = desc.split()
    number_words.append(len(words))
    word_len = 0
    for word in words:
        word_len += len(word)
    avg_word_length.append(word_len / len(words))
test_df["average_word_length"] = avg_word_length
test_df["num_words"] = number_words
test_df["desc_char_length"] = desc_length


# Name length feature. Also didnt seem to make a measurable difference but was left in

# In[ ]:


name_len = []
for name in train_df["Name"]:
    try:
        name_len.append(len(name))
    except:
        name_len.append(-1)
train_df["name_len"] = name_len


# In[ ]:


name_len = []
for name in test_df["Name"]:
    try:
        name_len.append(len(name))
    except:
        name_len.append(-1)
test_df["name_len"] = name_len


# Used these state gdp and state population inputs like most other people. Wasn't crucial but did seem to get some use in the LGBM feature importance charts

# In[ ]:


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

train_df["state_gdp"] = train_df.State.map(state_gdp)
train_df["state_population"] = train_df.State.map(state_population)
test_df["state_gdp"] = test_df.State.map(state_gdp)
test_df["state_population"] = test_df.State.map(state_population)


# In[ ]:


word_index = tokenizer.word_index
nb_words = min(num_words, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= nb_words: continue
    try:
        embedding_vector = embeddings_index[word]
    except:
        embedding_vector = None
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# Here's one of the main sections I think we got a bump over other people. For our NN embedding layer we need them to be the correct size for the number of possible categories. The way that Christof set things up in his original kernel a lot of people branched from he didnt fit on both the train and test options and if i'm not mistaken he didn't relabel encode the categoricals. This meant that in the NN it would look up a embedding for breed 274, but the embedding layer would only be size 125 because it didnt have all possible options. This caused major NN instability. The below code takes inventory of all possible categories and then condenses them down to indexes counting up 1, 2, 3, 4, etc. to match with the size of the embedding layer. This was kind of found by sheer luck. I increased the embedding size by plus 50 and it seemed to alleviate some, but not all of the stability issues. That shouldnt have made any performance difference so I inspected that area a bit more. Prior to that I was focused on maybe lowering the learning rate or clipping gradients because I couldnt figure out what was going wrong. 

# In[ ]:


for col in cat_cols:
    le = LabelEncoder()
    le.fit(train_df[col].tolist() + test_df[col].tolist())
    train_df[col] = le.transform(train_df[col].tolist())
    test_df[col] = le.transform(test_df[col].tolist())
embed_sizes = [len(set(list(train_df[col].unique()) + list(test_df[col].unique()))) + 1 for col in cat_cols]


# In[ ]:


agg_list = ["Age", "Gender", "FurLength", "Vaccinated", "Dewormed", "Sterilized",
 "Health", "Quantity", "Fee", "PhotoAmt", "VideoAmt", "Color1", "Color2", "Color3", "vertex_y", "label_score"]
train_df[agg_list]


# Here we did a bunch of aggregations on the rescuerID. So here we knew that we couldn't use rescuerID as a feature to learn from as there was no overlap between the train and test set, but we also knew it was a very predictive feature so we tried to instead get proxy understandings about the rescuerID's. What kind of pets did they typically have, how old were they, do they do a good job of uploading a lot of photos. agg_mean_PhotoAmt was one of our most important features consistently. More important so than even PhotoAmt on it's own. 

# In[ ]:


for col in agg_list:
    means = train_df.groupby("RescuerID")[col].mean()
    train_df["agg_mean_" + col] = train_df["RescuerID"].map(means)
    means = test_df.groupby("RescuerID")[col].mean()
    test_df["agg_mean_" + col] = test_df["RescuerID"].map(means)
    num_cols.append("agg_mean_" + col)
for col in agg_list:
    stds = train_df.groupby("RescuerID")[col].std()
    stds = stds.fillna(stds.mean())
    train_df["agg_std_" + col] = train_df["RescuerID"].map(stds)
    stds = test_df.groupby("RescuerID")[col].std()
    test_df["agg_std_" + col] = test_df["RescuerID"].map(stds)
    num_cols.append("agg_std_" + col)


# In[ ]:


from tqdm import tqdm
from copy import deepcopy
train_df_copy = deepcopy(train_df)
test_df_copy = deepcopy(test_df)


# In[ ]:


# train_df = train_df_copy
# test_df = test_df_copy


# Here we did the same image through pretrained network trick but again we did a slight twist on it. Instead of doing the averaging step down to 256 or 128 we used SVD again in order to more efficiently pack that same information down to 32 components. This was again informed by the feature selection process we used. 

# In[ ]:


img_size = 256
batch_size = 16
pet_ids = train_df['PetID'].values
n_batches = len(pet_ids) // batch_size + (len(pet_ids) % batch_size != 0)
from keras.applications.densenet import preprocess_input, DenseNet121
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

from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
inp = Input((256,256,3))
backbone = DenseNet121(input_tensor = inp, include_top = False, weights = None)
backbone.load_weights("../input/densenet-keras/DenseNet-BC-121-32-no-top.h5")
print("weights loaded succesfully")
x = backbone.output
x = GlobalAveragePooling2D()(x)
x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
# x = AveragePooling1D(4)(x)
out = Lambda(lambda x: x[:,:,0])(x)

m = Model(inp,out)
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
train_feats = pd.DataFrame.from_dict(features, orient='index').values
n_components = 32
svd = TruncatedSVD(n_components=n_components)
svd.fit(train_feats)
train_feats = svd.transform(train_feats)
train_feats = pd.DataFrame(train_feats, columns=['img_svd1_{}'.format(i) for i in range(n_components)])
train_feats.to_csv('train_img_features.csv')
train_df = pd.concat([train_df, train_feats], axis = 1)
pet_ids = test_df['PetID'].values
n_batches = len(pet_ids) // batch_size + (len(pet_ids) % batch_size != 0)
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
test_feats = pd.DataFrame.from_dict(features, orient='index').values
test_feats = svd.transform(test_feats)
test_feats = pd.DataFrame(test_feats, columns=['img_svd1_{}'.format(i) for i in range(n_components)])
test_feats.to_csv('test_img_features.csv')
test_df = pd.concat((test_df.reset_index(drop = True), test_feats), axis=1)
print(train_df.shape, test_df.shape)


# In[ ]:


print(test_df.shape, test_feats.shape)


# In[ ]:


from keras.models import load_model
catvsdog = load_model("../input/kerascatvsdog/best.hd5")
img_size = 299
batch_size = 16
pet_ids = train_df['PetID'].values
n_batches = len(pet_ids) // batch_size + (len(pet_ids) % batch_size != 0)
from keras.applications.densenet import preprocess_input, DenseNet121
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
backbone = catvsdog
out = backbone.layers[-2].output


m = Model(backbone.input, out)
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
train_feats = pd.DataFrame.from_dict(features, orient='index').values
n_components = 32
svd = TruncatedSVD(n_components=n_components)
svd.fit(train_feats)
train_feats = svd.transform(train_feats)
train_feats = pd.DataFrame(train_feats, columns=['img_svd2_{}'.format(i) for i in range(n_components)])
train_feats.to_csv('train_img_features.csv')
train_df = pd.concat([train_df, train_feats], axis = 1)
pet_ids = test_df['PetID'].values
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
test_feats = pd.DataFrame.from_dict(features, orient='index').values
test_feats = svd.transform(test_feats)
test_feats = pd.DataFrame(test_feats, columns=['img_svd2_{}'.format(i) for i in range(n_components)])
test_feats.to_csv('test_img_features.csv')
test_df = pd.concat((test_df.reset_index(drop = True), test_feats), axis=1)
print(train_df.shape, test_df.shape)


# In[ ]:


n_components = 32
train_desc = train_df.Description1.fillna("none").values
test_desc = test_df.Description1.fillna("none").values
print(test_desc.shape)
tfv = TfidfVectorizer(min_df=50,  max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        )
    
# Fit TFIDF
tfv.fit(list(train_desc))
X =  tfv.transform(train_desc)
X_test = tfv.transform(test_desc)

svd = TruncatedSVD(n_components=n_components)
svd.fit(X)
X = svd.transform(X)
X = pd.DataFrame(X, columns=['svd1_{}'.format(i) for i in range(n_components)])
train_df = pd.concat((train_df, X), axis=1)
X_test = svd.transform(X_test)
X_test = pd.DataFrame(X_test, columns=['svd1_{}'.format(i) for i in range(n_components)])
test_df = pd.concat((test_df.reset_index(drop = True), X_test), axis=1)
print(test_df.shape)


# In[ ]:


#helper for resetting
# train_df = train_df.drop(['svd2_{}'.format(i) for i in range(n_components)], axis = 1)
# test_df = test_df.drop(['svd2_{}'.format(i) for i in range(n_components)], axis = 1)


# We utilized image features like many others but these werent a huge factor. 

# In[ ]:


from PIL import Image
split_char = '/'
train_df_ids = train_df[['PetID']]
test_df_ids = test_df[['PetID']]

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

train_df = train_df.merge(agg_train_imgs, how='left', on='PetID')
test_df = test_df.merge(agg_test_imgs, how='left', on='PetID')


# In[ ]:


for col in agg_train_imgs.columns[1:]:
    num_cols.append(col)


# Scaling is very important for the NN but not for any of the other methods.

# In[ ]:


#num_cols = [x for x in keep_features if x not in cat_cols]
print('scaling num_cols')
for col in num_cols:
#     if col in no_scale_cols:
#         continue
    print('scaling {}'.format(col))
    try:
        col_mean = train_df[col].mean()
        train_df[col].fillna(col_mean, inplace=True)
        test_df[col].fillna(col_mean, inplace=True)
        scaler = StandardScaler()
        train_df[col] = scaler.fit_transform(train_df[col].values.reshape(-1, 1))
        test_df[col] = scaler.transform(test_df[col].values.reshape(-1, 1))
    except:
        print("*********")


# In[ ]:




from keras import backend as K
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
def get_input_features(df):
    nn_nums= [x for x in list(num_cols) #if x not in list(no_scale_cols)
             ]
    X = {'description':pad_sequences(df['Description'], maxlen=maxlen)}
    X['numerical'] = np.array(df[nn_nums])
    X['bow_inputs'] = np.array(df[list(["svd1_" + str(i) for i in range(n_components)] + 
                                       ["SVD_metadata_annots_top_desc_" + str(i) for i in range(16)] +
                                       ["SVD_sentiment_entities_" + str(i) for i in range(16)]
                                     )])
    X['img_inputs'] = np.array(df[list(["img_svd1_" + str(i) for i in range(n_components)] + ["img_svd2_" + str(i) for i in range(n_components)]
                                     )])
    for cat in cat_cols:
        X[cat] = np.array(df[cat])
    return X


# ## Define NN Model

# We toyed with directly optimizing qwk loss for the NN, but this didn't seem to be better so we stuck with treating it as a regression problem and using rmse. 

# In[ ]:


import tensorflow as tf

def kappa_loss(y_pred, y_true, y_pow=2, eps=1e-10, N=5, bsize=256, name='kappa'):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            bsize: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""

    with tf.name_scope(name):
        y_true = tf.to_float(y_true)
        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)
    
        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))
    
        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)
    
        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)
    
        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                              tf.to_float(bsize))
    
        return nom / (denom + eps)


# Here is the definition of our NN model. It isn't anything too fancy, just embedding layers for the categoricals, dense layers for the numericals, img and text SVD features and then CNN over the text. 

# In[ ]:


from keras.layers import Input, Embedding, Concatenate, Flatten, Dense, Dropout, BatchNormalization, CuDNNLSTM, SpatialDropout1D
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D, MaxPool1D, concatenate
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import  Adam
import keras.backend as k
def make_model(softmax = False):
    k.clear_session()

    categorical_inputs = []
    for cat in cat_cols:
        categorical_inputs.append(Input(shape=[1], name=cat))

    categorical_embeddings = []
    for i, cat in enumerate(cat_cols):
        categorical_embeddings.append(
            Embedding(embed_sizes[i], 10, name = cat + "_embed")(categorical_inputs[i]))

    categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])
    categorical_logits = Dropout(.5)(categorical_logits)
    categorical_logits = Dense(50, activation = 'relu')(categorical_logits)

    numerical_inputs = Input(shape=[X_train["numerical"].shape[1]], name = 'numerical')
    numerical_logits = Dropout(.2)(numerical_inputs)
    numerical_logits = Dense(50, activation = 'relu')(numerical_logits)
    numerical_logits = Dense(50, activation = 'relu')(numerical_logits)
    
    img_inputs = Input(shape = [n_components * 2], name = "img_inputs")
    img_logits = Dropout(.2)(img_inputs)
    
    bow_inputs = Input(shape = [n_components * 2], name = "bow_inputs")
    bow_logits = Dropout(.2)(bow_inputs)
    

    text_inp = Input(shape=[maxlen], name='description')
    text_embed = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(text_inp)
    emb_desc = SpatialDropout1D(.6)(text_embed)
    filter_sizes=[1,2,3,4]
    convs = []
    for filter_size in filter_sizes:
        conv = Conv1D(8, kernel_size=(filter_size), 
                        kernel_initializer="normal", activation="relu")(emb_desc)
        convs.append(MaxPool1D(pool_size=(maxlen-filter_size+1))(conv))
    text_logits = concatenate(convs)
    avg_pool = GlobalAveragePooling1D()(text_logits)
    max_pool = GlobalMaxPooling1D()(text_logits)
    text_logits = Concatenate()([avg_pool, max_pool])     
        

    x = Concatenate()([
        bow_logits,
        categorical_logits, 
        text_logits, 
        numerical_logits,
        img_logits
    ])
    x = Dense(300, activation = 'relu')(x)
    x = Dropout(.2)(x)
    x = Dense(200, activation = 'relu')(x)
    x = Dropout(.2)(x)
    x = Dense(100, activation = 'relu')(x)
    
    if softmax == True:
        out = Dense(5, activation = 'softmax')(x)
    else:
        out = Dense(1, activation = 'sigmoid')(x)
    

    model = Model(inputs=[text_inp] + categorical_inputs + [numerical_inputs] + [bow_inputs] + [img_inputs],outputs=out)
    if softmax == True:
        loss = kappa_loss
    else:
        loss = root_mean_squared_error
    model.compile(optimizer=Adam(lr = 0.0005), loss = loss)
    return model


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


# We utilized the "Golden" rounder which seemed to give us a very small fractional gain consistently. Only disadvantage was it was much slower. Likely would have been better to use the faster methods while prototyping but we werent constrained too tightly on this competition. 

# In[ ]:


# put numerical value to one of bins
def to_bins(x, borders):
    for i in range(len(borders)):
        if x <= borders[i]:
            return i
    return len(borders)

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _loss(self, coef, X, y, idx):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        ll = -quadratic_weighted_kappa(y, X_p)
        return ll

    def fit(self, X, y):
        coef = [1.5, 2.0, 2.5, 3.0]
        golden1 = 0.618
        golden2 = 1 - golden1
        ab_start = [(1, 2), (1.5, 2.5), (2, 3), (2.5, 3.5)]
        for it1 in range(10):
            for idx in range(4):
                # golden section search
                a, b = ab_start[idx]
                # calc losses
                coef[idx] = a
                la = self._loss(coef, X, y, idx)
                coef[idx] = b
                lb = self._loss(coef, X, y, idx)
                for it in range(20):
                    # choose value
                    if la > lb:
                        a = b - (b - a) * golden1
                        coef[idx] = a
                        la = self._loss(coef, X, y, idx)
                    else:
                        b = b - (b - a) * golden2
                        coef[idx] = b
                        lb = self._loss(coef, X, y, idx)
        self.coef_ = {'x': coef}

    def predict(self, X, coef):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        return X_p

    def coefficients(self):
        return self.coef_['x']


# In[ ]:


rescuerID = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")["RescuerID"]


# In[ ]:


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# Like many other people discovered, because the rescuerID was unique between the train and test set we chose to use groupKFold which allowed us to have unique rescuerID's between our train and validation sets. We also chose higher K than most people in hopes this would give us more stable results. We used 10 for the NN and 20 for the others. This may or may not have been the way to go as it makes your validation set very small. At the same time when training so many models you have to assume across all of them it will find some useful combinations of the data. 

# One other interesting thing we did was use a sigmoid activation function. Some others used a linear function, but I wanted out predictions to only be bounded to the possible labels. In order to do this I divided the adoption speed by 4. This condensed the range down to between 0-1 and allowed for sigmoid acitvation function at the end. I don't think this was important to our solution, it seems many teams were able to get similar results with linear activations, but sigmoid seemed to converge more quickly for me. 

# In[ ]:


num_folds = 10
softmax = False
kf = GroupKFold(n_splits=num_folds)
y = train_df["AdoptionSpeed"]
fold_splits = kf.split(train_df, y, rescuerID)
oof = np.zeros((y.shape))
test_preds = np.zeros((test_df.shape[0]))
if softmax == True:
    test_preds = np.zeros((test_df.shape[0], 5))
    oof = np.zeros(shape = (y.shape[0], 5))
for i, (dev_index, val_index) in enumerate(fold_splits):
    tr_df = train_df.iloc[dev_index, :]
    val_df = train_df.iloc[val_index, :]
    X_train = get_input_features(tr_df)
    X_valid = get_input_features(val_df)
    X_test = get_input_features(test_df)
    model = make_model(softmax = softmax)
    ckpt = ModelCheckpoint("model" + str(i) + ".h5", monitor='val_loss', save_best_only = True, verbose = False)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, mode='auto', verbose = False)
    if softmax == True:
        y_train = np.zeros(shape = (tr_df.shape[0], 5))
        y_valid = np.zeros(shape = (val_df.shape[0], 5))
        for j, l in enumerate(tr_df['AdoptionSpeed'].values):
            j = int(j)
            l = int(l)
            y_train[j,l] = 1
        for j, l in enumerate(val_df['AdoptionSpeed'].values):
            j = int(j)
            l = int(l)
            y_valid[j,l] = 1
    else:
        y_train = tr_df['AdoptionSpeed'].values / 4
        y_valid = val_df['AdoptionSpeed'].values / 4

    

    hist = model.fit(X_train, y_train, validation_data = (X_valid,y_valid), batch_size = 500, epochs = 35, verbose = 2, callbacks = [ckpt, rlr])
    model.load_weights("model" + str(i) + ".h5")
    if softmax == True:
        val_preds = model.predict(X_valid)
        oof[val_index] = val_preds
        test_pred = model.predict(X_test)
        test_preds += (test_pred/num_folds)
    else:
        val_preds = model.predict(X_valid)[:, 0] * 4
        y_valid = y_valid * 4
        optR = OptimizedRounder()
        optR.fit(val_preds, y_valid)
        coefficients = optR.coefficients()
        val_pred_rounded = optR.predict(val_preds, coefficients)
        oof[val_index] = val_preds
        test_pred = model.predict(X_test)[:, 0] * 4
        test_preds += (test_pred/num_folds)
if softmax == True:
    oof_rounded = np.argmax(oof, axis = 1)
    test_rounded = np.argmax(test_preds, axis = 1)
else:
    optR = OptimizedRounder()
    optR.fit(oof, y)
    coefficients = optR.coefficients()
    oof_rounded = optR.predict(oof, coefficients)
    test_rounded = optR.predict(test_preds, coefficients)


# In[ ]:


print(num_cols)
print(quadratic_weighted_kappa(y, oof_rounded))
oof_nn = oof
test_preds_nn = test_preds


# In[ ]:


from copy import deepcopy
X_temp = deepcopy(pd.concat([train_df, test_df], axis = 0))


# Now we can start looking at the FFM setup. While FFM's are pretty good at dealing with categoricals, theyre not very useful with numericals on their own. So to deal with this we used the KBinsDiscretizer which put the numerical variables into their own groupings. This seemed to have a strong impact on CV. 

# In[ ]:


from sklearn.preprocessing import KBinsDiscretizer
n_bins = 5
for c in num_cols:
    if X_temp[c].nunique() < 5:
        continue
    vals = X_temp[c].values.reshape(-1, 1)
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal')
    est.fit(vals)
    vals = est.transform(vals).reshape(1, -1)[0]
    X_temp[c] = vals


# In[ ]:


train_df_ffm = X_temp.iloc[:train_df.shape[0], :]
test_df_ffm = X_temp.iloc[train_df.shape[0]:, :]


# Honestly really not a fan of the tools that prefer you to  write these things to specific file formats. Have worked with this one and vowpal wabbit now. Similar boilerplate from the public kernels. Someone from our group tried the scikit-learn stype api and it seemed to function much slower so we conceded defeat and did the file writing process. 

# In[ ]:


field_features = defaultdict()
global max_val
max_val = 1
global max_val
line = []
label = []
dtypes = {}
for feature in num_cols:
#     if "agg_std" in feature:
#         #print(feature)
#         continue
#     if "RescuerID_UNIQUE" in feature:
#         print(feature)
#         continue
    dtypes[feature] = "float16"
for feature in cat_cols:
    dtypes[feature] = "category"
    
from collections import defaultdict
import math

dont_use = ['RescuerID', 'AdoptionSpeed', 'Description', 'PetID']
too_many_vals = [""]

categories = [k for k, v in dtypes.items() if k not in dont_use]
categories_index = dict(zip(categories, range(len(categories))))

def generate_ffm_file(index, train = True):
    global max_val
    ffeatures = []
    if train == True:
        path = 'train.libffm'
    else:
        path = 'valid.libffm'
    with open(path, 'w') as the_file:
        for t, (index, row) in enumerate(train_df_ffm.iloc[index, :].iterrows()):
            if t % 5000 == 0:
                print(t, len(field_features), max_val)
                #print(line)
            label = [str(int(row['AdoptionSpeed']))]
            ffeatures = []

            for field in categories:
                if field == 'AdoptionSpeed':
                    continue
                feature = row[field]
                if feature == '':
                    feature = "unk"
                if field not in num_cols:
                    ff = field + '_____' + str(feature)
                else:
                    if feature == "unk" or float(feature) == -1:
                        ff = field + '_____' + str(0)
                    else:
                        if field in too_many_vals:
                            ff = field + '_____' + str(int(round(math.log(1 + float(feature)))))
                        else:
                            ff = field + '_____' + str(feature)
                if ff not in field_features:
                    if len(field_features) == 0:
                        field_features[ff] = 1
                        max_val += 1
                    else:
                        field_features[ff] = max_val + 1
                        max_val += 1

                fnum = field_features[ff]
                ffeatures.append('{}:{}:1'.format(categories_index[field], str(fnum)))

            line = label + ffeatures
            the_file.write('{}\n'.format(' '.join(line)))


# In[ ]:


with open('test.libffm', 'w') as the_file:
    global max_val
    ffeatures = []
    for t, (index, row) in tqdm(enumerate(test_df_ffm.iterrows())):
        if t % 3000 == 0:
            print(t, len(field_features), max_val)
            #print(line)
        label = [str(int(0))]
        ffeatures = []

        for field in categories:
            if field == 'AdoptionSpeed':
                continue
            feature = row[field]
            if feature == '':
                feature = "unk"
            if field not in num_cols:
                ff = field + '_____' + str(feature)
            else:
                if feature == "unk" or float(feature) == -1:
                    ff = field + '_____' + str(0)
                else:
                    if field in too_many_vals:
                        ff = field + '_____' + str(int(round(math.log(1 + float(feature)))))
                    else:
                        ff = field + '_____' + str(feature)
            if ff not in field_features:
                if len(field_features) == 0:
                    field_features[ff] = 1
                    max_val += 1
                else:
                    field_features[ff] = max_val + 1
                    max_val += 1

            fnum = field_features[ff]
            ffeatures.append('{}:{}:1'.format(categories_index[field], str(fnum)))

        line = label + ffeatures
        the_file.write('{}\n'.format(' '.join(line)))


# In[ ]:


n_splits = 20
kfold = GroupKFold(n_splits=n_splits)
ffm_oof_train = np.zeros((train_df_ffm.shape[0]))
ffm_oof_test = np.zeros((test_df_ffm.shape[0], n_splits))

i = 0
for train_index, valid_index in kfold.split(train_df_ffm, y.values, rescuerID):
    print('\nsplit ', i)
    generate_ffm_file(train_index)
    generate_ffm_file(valid_index, train = False)
    # create ffm model
    ffm_model = xl.create_ffm()
    # set training
    ffm_model.setTrain("train.libffm")
    ffm_model.setValidate("valid.libffm")
    # define params
    param = {'task':'reg', 'lr':.2,
             'lambda':0.0002, 'metric':'rmse', 'epoch' : 100, 'k': 4}
    # train the model
    ffm_model.fit(param, "./model.out")
    # # set the valid data
    ffm_model.setTest("valid.libffm")

    ffm_model.predict("./model.out", "output_valid.txt")
    valid_ffm = pd.read_csv('output_valid.txt', header=None)[0].values
    optR = OptimizedRounder()
    optR.fit(valid_ffm, y[valid_index])
    coefficients = optR.coefficients()
    valid_ffm_rounded = optR.predict(valid_ffm, coefficients)
    print(quadratic_weighted_kappa(valid_ffm_rounded, y[valid_index]))
    ffm_oof_train[valid_index] = valid_ffm
    
    ffm_model.setTest("test.libffm")
    ffm_model.predict("./model.out", "output_test.txt")
    test_ffm = pd.read_csv('output_test.txt', header=None)[0].values
    ffm_oof_test[:, i] = test_ffm
    print(test_ffm)
    i += 1
    
ffm_oof_test = ffm_oof_test.mean(axis=1)


# In[ ]:


print(ffm_oof_test.mean())
print(ffm_oof_train.mean())


# In[ ]:


optR = OptimizedRounder()
optR.fit(ffm_oof_train, y)
coefficients = optR.coefficients()
train_ffm_rounded = optR.predict(ffm_oof_train, coefficients)
print(quadratic_weighted_kappa(train_ffm_rounded, y))


# In[ ]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
# from sklearn.preprocessing import LabelEncoder

def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# In[ ]:


from copy import deepcopy
train = deepcopy(train_df)
test = deepcopy(test_df)


# In[ ]:


train.loc[:, cat_cols] = train[cat_cols].astype('category')
test.loc[:, cat_cols] = test[cat_cols].astype('category')
print(train.shape)
print(test.shape)
train.head()


# In[ ]:


train.head()


# In[ ]:


# num_cols = num_cols1
# bad_cols = [
#            'Color3',
#            'Vaccinated',
#            'Dewormed',
#            'num_words_upper',
#            'Health',
#            'bounding_importance',
#            'Type',
#            'VideoAmt',
#            'bounding_confidence']
# num_cols = [col for col in num_cols if col not in bad_cols]


# In[ ]:


train.drop(set(['Name', 'RescuerID', 'Description', 'Description1', 'PetID', 'AdoptionSpeed']), axis=1, inplace=True)
test.drop(set(['Name', 'RescuerID', 'Description', 'Description1', 'PetID']), axis=1, inplace=True)


# In[ ]:


# train.drop(list(bad_cols), axis=1, inplace=True)
# test.drop(list(bad_cols), axis=1, inplace=True)


# In[ ]:


# train.to_csv("train_lgb.csv")
# test.to_csv("test_lgb.csv")


# In[ ]:





# In[ ]:


# train["null_importance"] = np.random.normal(size = train.shape[0])
# test["null_importance"] = np.random.normal(size = test.shape[0])


# Our LGBM model really isn't anything too interesting but what was interesting was how we used it for feature selection and to inform our decisions. Most people have seen the feature importance plots. We built on top of these in order to drop features. We injected a random noise feature and found that many features ended up below this random variable. We dropped any features that were below this threshold (unless we were confident they would be useful/harmless to leave in) This allowed us to drastically cut down our number of useless features, speed up our processing time and also increase our generalization capability. There was a point where we had ~600 features we would plug in and after null importance filtering we would only have ~100. 
# 
# This informed our design of the component counts to use as well. We reduced the number of components until all of the components would consistently end up above the null feature importance. In this way we ensured that the information density from each was better than random noise. Not sure if this is definitively the best way to do that but it seemed to make logical sense at the time. 

# In[ ]:


def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model', num_iters = 1, num_folds = 19, ):
    cv_scores = []
    qwk_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0], num_folds))
    all_coefficients = np.zeros((num_folds * num_iters, 4))
    feature_importance_df = pd.DataFrame()
    i = 1
    for j in range(num_iters):
        kf = GroupKFold(n_splits=num_folds)
        fold_splits = kf.split(train, target, rescuerID)

        for dev_index, val_index in fold_splits:
            print('Started ' + label + ' fold ' + str(i) + '/5')
            if isinstance(train, pd.DataFrame):
                dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
                dev_y, val_y = target[dev_index], target[val_index]
            else:
                dev_X, val_X = train[dev_index], train[val_index]
                dev_y, val_y = target[dev_index], target[val_index]

            params2 = params.copy()
            pred_val_y, pred_test_y, importances, coefficients, qwk = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
            pred_full_test += (pred_test_y / num_iters)
            pred_train[val_index] += (pred_val_y/num_iters)
            all_coefficients[i-1, :] = coefficients
            if eval_fn is not None:
                cv_score = eval_fn(val_y, pred_val_y)
                cv_scores.append(cv_score)
                qwk_scores.append(qwk)
                print(label + ' cv score {}: RMSE {} QWK {}'.format(i, cv_score, qwk))
            fold_importance_df = pd.DataFrame()
            fold_importance_df['feature'] = train.columns.values
            fold_importance_df['importance'] = importances
            fold_importance_df['fold'] = i
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)        
            i += 1
        pred_full_test = (pred_full_test / num_folds)
        print('{} cv RMSE scores : {}'.format(label, cv_scores))
        print('{} cv mean RMSE score : {}'.format(label, np.mean(cv_scores)))
        print('{} cv std RMSE score : {}'.format(label, np.mean(cv_scores)))
        print('{} cv QWK scores : {}'.format(label, qwk_scores))
        print('{} cv mean QWK score : {}'.format(label, np.mean(qwk_scores)))
        print('{} cv std QWK score : {}'.format(label, np.std(qwk_scores)))
        results = {'label': label,
                   'train': pred_train, 'test': pred_full_test,
                    'cv': cv_scores, 'qwk': qwk_scores,
                   'importance': feature_importance_df,
                   'coefficients': all_coefficients}
    return results


params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 70,
          'max_depth': 7,
          'learning_rate': 0.02,
          'bagging_fraction': 0.85,
          'feature_fraction': 0.8,
          'min_split_gain': 0.02,
          'min_child_samples': 150,
          'min_child_weight': 0.2,
          'lambda_l2': 0.05,
          'verbosity': -1,
          'data_random_seed': 24,
          'early_stop': 100,
          'verbose_eval': 100,
          'num_rounds': 10000}

def runLGB(train_X, train_y, test_X, test_y, test_X2, params):
    print('Prep LGB')
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    print('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)
    print('Predict 1/2')
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    optR = OptimizedRounder()
    optR.fit(pred_test_y, test_y)
    coefficients = optR.coefficients()
    pred_test_y_k = optR.predict(pred_test_y, coefficients)
    print("Valid Counts = ", Counter(test_y))
    print("Predicted Counts = ", Counter(pred_test_y_k))
    print("Coefficients = ", coefficients)
    qwk = quadratic_weighted_kappa(test_y, pred_test_y_k)
    print("QWK = ", qwk)
    print('Predict 2/2')
    pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), model.feature_importance("gain"), coefficients, qwk

# results = run_cv_model(train, test, y, runLGB, params, rmse, 'lgb')


# In[ ]:


no_drop = ['Color1', 'Color2', 'Color3', 'Dewormed', 'Fee', 'FurLength',
       'Gender', 'Health', 'MaturitySize','PhotoAmt', 'Quantity','Sterilized', 'Type', 'Vaccinated',
           #'label_description',
           'state_gdp', 'state_population',]


# In[ ]:


num_cols


# The features we eneded up utilizing after many runs determining which of our features were successful and useful

# In[ ]:


keep_features = (['Age', 'Breed1', 'Breed2', 'Color1', 'Color2', 'Color3', 'Fee', 'Quantity',
        'MaturitySize','FurLength','Vaccinated','Dewormed', 'Sterilized','Health','Quantity','Fee',
        'VideoAmt','PhotoAmt', 'state_gdp','state_population', 'doc_sent_mag', 'doc_sent_score', 'vertex_x', 'vertex_y', 
        'bounding_confidence', 'bounding_importance','dominant_blue', 'dominant_green', 'dominant_red', 'dominant_pixel_frac',
        'dominant_score', 'label_description', 'label_score',
       'RescuerID_COUNT','State', 'Sterilized', 'agg_mean_Age', 'agg_mean_Color1',
       'agg_mean_Color2', 'agg_mean_Color3', 'agg_mean_Dewormed',
       'agg_mean_Fee', 'agg_mean_FurLength', 'agg_mean_Gender',
       'agg_mean_PhotoAmt', 'agg_mean_Quantity', 'agg_mean_Sterilized',
       'agg_mean_Vaccinated', 'agg_mean_VideoAmt', 'agg_std_Age',
       'agg_std_Color1', 'agg_std_Color2', 'agg_std_Color3',
       'agg_std_Dewormed', 'agg_std_Fee', 'agg_std_FurLength',
       'agg_std_Gender', 'agg_std_PhotoAmt', 'agg_std_Quantity',
       'agg_std_Sterilized', 'agg_std_Vaccinated', 'label_score',
       'metadata_metadata_annots_score_MAX',
       'metadata_metadata_annots_score_MEAN',
       'metadata_metadata_annots_score_MIN',
       'metadata_metadata_annots_score_SUM',
       'metadata_metadata_annots_score_VAR',
       'metadata_metadata_color_pixelfrac_MIN',
       'metadata_metadata_color_pixelfrac_SUM',
       'metadata_metadata_color_pixelfrac_VAR',
       'metadata_metadata_color_score_SUM',
       'metadata_metadata_color_score_VAR', 'vertex_x', 'vertex_y','RescuerID_UNIQUE'] 
       +["svd1_" + str(i) for i in range(n_components)] 
       +["img_svd1_" + str(i) for i in range(n_components)] 
       +["img_svd2_" + str(i) for i in range(n_components)]
#        +["SVD_metadata_annots_top_desc_" + str(i) for i in range(16)]
#        +["SVD_sentiment_entities_" + str(i) for i in range(16)]
        )


# In[ ]:


results = run_cv_model(train[keep_features], test[keep_features], y, runLGB, params, rmse, 'lgb')


# In[ ]:


len(keep_features)


# In[ ]:


imports = results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
imports.sort_values('importance', ascending=False)


# Xgboost. Again, nothing exceptionally interesting here. Did some minor hyperparameter tuning. One thing we considered though was that we are doing hyperparameter tuning on models individually. It would likely be more useful in the future to tune parameters dependent on performance in an ensemble. Ie. optimizing the NN. then we could optimize the parameters of xLearn when ensembled with the NN. In theory this might steer us away from strong individual model performance but diverse models that ensemble better. Just a thought though. Not necessarily something we have implemented here.

# In[ ]:


import xgboost as xgb
xgb_params = {
    'eval_metric': 'rmse',
    'seed': 1337,
    'eta': 0.0123,
    'subsample': 1.0,
    'colsample_bytree': 0.6,
    'min_child_weight': 1.0,
    'gamma' : .5,
    'max_depth' : 5,
    'tree_method': 'gpu_hist',
    'device': 'gpu',
    'silent': 1,
}


# In[ ]:


def run_xgb(params, X_train, X_test):
    n_splits = 20
    verbose_eval = 1000
    num_rounds = 30000
    early_stop = 500

    kf = GroupKFold(n_splits=n_splits)

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))

    i = 0

    for train_idx, valid_idx in kf.split(X_train, y,  rescuerID):

        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        y_tr = y.iloc[train_idx]
        y_val = y.iloc[valid_idx]
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


final_feats = keep_features


# Had to remap categoricals to dummy variables since XGBoost cant handle these out of the box like LGBM does. 

# In[ ]:


X_temp = pd.concat([train, test], ignore_index=True, sort=False)
for i in cat_cols:
    X_temp = pd.concat([pd.get_dummies(X_temp.loc[:, i], prefix = str(i)), X_temp], axis = 1)
    final_feats = final_feats + [i + "_"+ str(x) for x in range(X_temp.loc[:, i].nunique())]
train = X_temp.loc[:train.shape[0]-1, :]
test = X_temp.loc[train.shape[0]:, :]


# In[ ]:


train[cat_cols] = train[cat_cols].astype(int)
test[cat_cols] = test[cat_cols].astype(int)


# In[ ]:


model, oof_train, oof_test = run_xgb(xgb_params, train[train[final_feats].columns.unique()], test[test[final_feats].columns.unique()])


# In[ ]:


optR = OptimizedRounder()
optR.fit(oof_train, y)
coefficients = optR.coefficients()
oof_xgb = optR.predict(oof_train, coefficients)
qwk = quadratic_weighted_kappa(y, oof_xgb)
print("QWK = ", qwk)


# In[ ]:


test_preds_xgb = optR.predict(oof_test.mean(axis=1), coefficients).astype(np.int8)


# In[ ]:


optR = OptimizedRounder()
coefficients_ = np.mean(results['coefficients'], axis=0)
print(coefficients_)
train_predictions = [r[0] for r in results['train']]
oof_lgb = np.array(train_predictions)
train_predictions = optR.predict(train_predictions, coefficients_).astype(int)
Counter(train_predictions)


# In[ ]:


test_predictions = [r[0] for r in results['test']]
test_preds_lgb = np.array(test_predictions)
test_predictions = optR.predict(test_predictions, coefficients_).astype(int)


# In[ ]:


print(oof_nn.shape)
print(oof_lgb.shape)
print(oof_xgb.shape)
print(test_preds_nn.shape)
print(test_preds_lgb.shape)
print(test_preds_xgb.shape)


# In[ ]:


print((test_preds_lgb).mean())
print(oof_lgb.mean())


# In[ ]:


#lgb only
optR = OptimizedRounder()
optR.fit(oof_lgb, y)
coefficients = optR.coefficients()
print(coefficients)
oof_rounded = optR.predict(oof_lgb, coefficients)
print(quadratic_weighted_kappa(y, oof_rounded))
test_rounded_lgb = optR.predict(test_preds_lgb, coefficients)


# In[ ]:


#nn only
if softmax == False:
    optR = OptimizedRounder()
    optR.fit(oof_nn, y)
    coefficients = optR.coefficients()
    print(coefficients)
    oof_rounded = optR.predict(oof_nn, coefficients)
    print(quadratic_weighted_kappa(y, oof_rounded))
    test_rounded_nn = optR.predict(test_preds_nn, coefficients)
if softmax == True:
    oof_rounded = np.argmax(oof_nn, axis = 1)
    test_rounded_nn = np.argmax(test_preds_nn, axis = 1)
    print(quadratic_weighted_kappa(y, oof_rounded))


# In[ ]:


oof_models = [oof_nn, oof_lgb
                 , oof_train, ffm_oof_train
                ]
test_models = [test_preds_nn, test_preds_lgb
                                     , oof_test.mean(axis = 1), ffm_oof_test
                                    ]


# In order to ensemble our models we did Ridge Regression of the outputs of the various models. We turned fit intercept off and in this way the regression was simply finding the best set of weights to combine the the models. One thing to note is that this is optimizing for loss, rather than qwk. One thing we explored was doing some iterative hill climbing approach so we could more directly optimize for QWK, but we didnt pursue it. Ultimately I think that would have just led to overfitting as now we are directly training on the labels of the data without any validation structure in place. 
# 
# One thing to note with the implementation we used is I renormalized the weights. Out of the box Ridge regression might choose to give a total weighting of 1.2 rather than 1.0. I renormalized these weights to sum to 1. This likely isnt truly valid because the regression is trying to move our distribution to match the train distribution more closely. By renormalizing it we are shifting it away from fitting the distribution better, but it seems we got away with it. Would be interesting to compare the results vs non-normalized. 

# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge
lr = Ridge(fit_intercept=False)
lr.fit(np.array(oof_models).T, y)
print(lr.coef_)
lr.coef_ = lr.coef_ * 1/(sum(lr.coef_))
print(lr.coef_)
oof_lr = lr.predict(np.array(oof_models).T)
test_preds_lr = lr.predict(np.array(test_models).T)
#lr of nn and lgb and xgb
optR = OptimizedRounder()
optR.fit(oof_lr, y)
coefficients = optR.coefficients()
print(coefficients)
oof_rounded = optR.predict(oof_lr, coefficients)
print(quadratic_weighted_kappa(y, oof_rounded))
test_rounded_lr = optR.predict(test_preds_lr, coefficients)


# In[ ]:


submission_df = pd.read_csv("../input/petfinder-adoption-prediction/test/sample_submission.csv")


# In[ ]:


submission_df["AdoptionSpeed"] = test_rounded_lr.astype(int)
submission_df.to_csv("submission.csv", index=False)


# In[ ]:


submission_df['AdoptionSpeed'].mean()


# In[ ]:


submission_df['AdoptionSpeed'].value_counts(normalize = False)


# In[ ]:


print(test_preds_nn.mean(), test_preds_lgb.mean(), oof_test.mean(), ffm_oof_test.mean())
print(oof_nn.mean(), oof_lgb.mean(), oof_train.mean(), ffm_oof_train.mean())


# In[ ]:




