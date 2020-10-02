#!/usr/bin/env python
# coding: utf-8

# # PetFinder.my Adoption Prediction
# Animal adoption rates are strongly correlated to the metadata associated with their online profiles, such as descriptive text and photo characteristics. As one example, PetFinder is currently experimenting with a simple AI tool called the Cuteness Meter, which ranks how cute a pet is based on qualities present in their photos.
# 
# In this competition you will be developing algorithms to predict the adoptability of pets - specifically, how quickly is a pet adopted? If successful, they will be adapted into AI tools that will guide shelters and rescuers around the world on improving their pet profiles' appeal, reducing animal suffering and euthanization.
# 
# ## For HUTU

# In[ ]:


from IPython.display import Image
import os
Image("../input/mypics/IMG_20190104_103954-01.jpeg", width = 500, height = 500)
# print(os.listdir("../input/hututu/IMG_20190104_103954.jpg"))


# ## Acknowledgements: Many functions and ideas are forked from these great kernels and many many kernels therein. Thanks for sharing
# https://www.kaggle.com/artgor/exploration-of-data-step-by-step
# 
# https://www.kaggle.com/christofhenkel/extract-image-features-from-pretrained-nn
# 
# https://www.kaggle.com/skooch/petfinder-simple-lgbm-baseline

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
import json
import gc
import glob
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from joblib import Parallel, delayed
from tqdm import tqdm, tqdm_notebook

import scipy as sp

from collections import Counter
from functools import partial
from math import sqrt

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix

import time
import datetime
import os
print(os.listdir("../input/petfinder-adoption-prediction"))

# Any results you write to the current directory are saved as output.


# # Data Overview
# Here we use a pretrained model and transfer learning to try and identify the different types of proteins present in the image.
# 
# refer to https://www.kaggle.com/artgor/exploration-of-data-step-by-step

# In[ ]:


breeds_df = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
colors_df = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')
states_df = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')

train_df = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
test_df = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
target_df = train_df[['PetID','AdoptionSpeed']]
def label(ID,target_df=target_df):
    return target_df['AdoptionSpeed'].loc[target_df['PetID']==ID].values[0]


# In[ ]:


print("train:",len(train_df))
print("test:",len(test_df))


# In[ ]:


train_df.head(2)


# In[ ]:


train_df.info()


# ## Let's have a quick look at the  descriptions

# In[ ]:


from wordcloud import WordCloud
fig, ax = plt.subplots(figsize = (12, 8))
text_cat = ' '.join(train_df['Description'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white',
                      width=1200, height=1000).generate(text_cat)
plt.imshow(wordcloud)
plt.title('Top words in description');
plt.axis("off");


# ## Target: Adoption speed
# 
# * 0 - Pet was adopted on the same day as it was listed.
# * 1 - Pet was adopted between 1 and 7 days (1st week) after being listed.
# * 2 - Pet was adopted between 8 and 30 days (1st month) after being listed.
# * 3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed.
# * 4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days). 

# In[ ]:


train_df['AdoptionSpeed'].value_counts().sort_index().plot('barh', color='teal');
plt.title('Adoption speed classes counts');


# ## Metric: Quadratic_weighted_kappa

# In[ ]:


from sklearn.metrics import cohen_kappa_score
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


# ## Parse Data from Sentiment and Metadata json files
# 
# ## Image Metadata
# We have run the images through Google's Vision API, providing analysis on Face Annotation, Label Annotation, Text Annotation and Image Properties. You may optionally utilize this supplementary information for your image analysis.
# 
# ## Sentiment Data
# We have run each pet profile's description through Google's Natural Language API, providing analysis on sentiment and key entities. You may optionally utilize this supplementary information for your pet description analysis. There are some descriptions that the API could not analyze. As such, there are fewer sentiment files than there are rows in the dataset.

# In[ ]:


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
train_pet_ids = train_df.PetID.unique()
test_pet_ids = test_df.PetID.unique()

if debug:
    train_pet_ids = train_pet_ids[:1000]
    test_pet_ids = test_pet_ids[:500]


dfs_train = Parallel(n_jobs=6, verbose=1)(
    delayed(extract_additional_features)(i, mode='train') for i in train_pet_ids)

train_dfs_sentiment = [x[0] for x in dfs_train if isinstance(x[0], pd.DataFrame)]
train_dfs_metadata = [x[1] for x in dfs_train if isinstance(x[1], pd.DataFrame)]

train_dfs_sentiment = pd.concat(train_dfs_sentiment, ignore_index=True, sort=False)
train_dfs_metadata = pd.concat(train_dfs_metadata, ignore_index=True, sort=False)

print(train_dfs_sentiment.shape, train_dfs_metadata.shape)


dfs_test = Parallel(n_jobs=6, verbose=1)(
    delayed(extract_additional_features)(i, mode='test') for i in test_pet_ids)

test_dfs_sentiment = [x[0] for x in dfs_test if isinstance(x[0], pd.DataFrame)]
test_dfs_metadata = [x[1] for x in dfs_test if isinstance(x[1], pd.DataFrame)]

test_dfs_sentiment = pd.concat(test_dfs_sentiment, ignore_index=True, sort=False)
test_dfs_metadata = pd.concat(test_dfs_metadata, ignore_index=True, sort=False)

print(test_dfs_sentiment.shape, test_dfs_metadata.shape)


# ## Group extracted features by PetID: one PetID may have several images and many  entries extracted from the json files

# In[ ]:


aggregates = ['sum', 'mean', 'var']
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


# In[ ]:


train_sentiment_gr.head(2)


# In[ ]:


train_metadata_desc.head(2)


# ### Merge sentiment and metadata 

# In[ ]:


# Train merges:
train_1 = train_df.copy()
train_1 = train_1.merge(
    train_sentiment_gr, how='left', on='PetID')
train_1 = train_1.merge(
    train_metadata_gr, how='left', on='PetID')
train_1 = train_1.merge(
    train_metadata_desc, how='left', on='PetID')
train_1 = train_1.merge(
    train_sentiment_desc, how='left', on='PetID')

# Test merges:
test_1 = test_df.copy()
test_1 = test_1.merge(
    test_sentiment_gr, how='left', on='PetID')
test_1 = test_1.merge(
    test_metadata_gr, how='left', on='PetID')
test_1 = test_1.merge(
    test_metadata_desc, how='left', on='PetID')
test_1 = test_1.merge(
    test_sentiment_desc, how='left', on='PetID')

print(train_1.shape, test_1.shape)
assert train_1.shape[0] == train_df.shape[0]
assert test_1.shape[0] == test_df.shape[0]


# In[ ]:


train_1.head(2)


# ## A bit more information(maybe):
# 
# I also attach the language and the overall score and magnitude plus its max and min to the dataframe.

# In[ ]:


sentiment_dict = {}
for filename in os.listdir('../input/petfinder-adoption-prediction/train_sentiment/'):
    with open('../input/petfinder-adoption-prediction/train_sentiment/' + filename, 'r') as f:
        sentiment = json.load(f);
        max_emotion = max(sentiment['sentences'], key=lambda sen : sen['sentiment']['magnitude']*sen['sentiment']['score'])
        min_emotion = min(sentiment['sentences'], key=lambda sen : sen['sentiment']['magnitude']*sen['sentiment']['score'])
        pet_id = filename.split('.')[0]
        sentiment_dict[pet_id] = {}
        sentiment_dict[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']
        sentiment_dict[pet_id]['score'] = sentiment['documentSentiment']['score']
        sentiment_dict[pet_id]['language'] = sentiment['language']
        sentiment_dict[pet_id]['max_score'] = max_emotion['sentiment']['score']
        sentiment_dict[pet_id]['min_score'] = min_emotion['sentiment']['score']
        sentiment_dict[pet_id]['max_magnitude'] = max_emotion['sentiment']['magnitude']
        sentiment_dict[pet_id]['min_magnitude'] = min_emotion['sentiment']['magnitude']
    
for filename in os.listdir('../input/petfinder-adoption-prediction/test_sentiment/'):
    with open('../input/petfinder-adoption-prediction/test_sentiment/' + filename, 'r') as f:
        sentiment = json.load(f);
        max_emotion = max(sentiment['sentences'], key=lambda sen : sen['sentiment']['magnitude']*sen['sentiment']['score'])
        min_emotion = min(sentiment['sentences'], key=lambda sen : sen['sentiment']['magnitude']*sen['sentiment']['score'])
        pet_id = filename.split('.')[0]
        sentiment_dict[pet_id] = {}
        sentiment_dict[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']
        sentiment_dict[pet_id]['score'] = sentiment['documentSentiment']['score']
        sentiment_dict[pet_id]['language'] = sentiment['language']
        sentiment_dict[pet_id]['max_score'] = max_emotion['sentiment']['score']
        sentiment_dict[pet_id]['min_score'] = min_emotion['sentiment']['score']
        sentiment_dict[pet_id]['max_magnitude'] = max_emotion['sentiment']['magnitude']
        sentiment_dict[pet_id]['min_magnitude'] = min_emotion['sentiment']['magnitude']
        

    


# In[ ]:


train_1['language'] = train_1['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else None)
train_1['magnitude'] = train_1['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
train_1['score'] = train_1['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)
train_1['max_score'] = train_1['PetID'].apply(lambda x: sentiment_dict[x]['max_score'] if x in sentiment_dict else 0)
train_1['min_score'] = train_1['PetID'].apply(lambda x: sentiment_dict[x]['min_score'] if x in sentiment_dict else 0)
train_1['max_magnitude'] = train_1['PetID'].apply(lambda x: sentiment_dict[x]['max_magnitude'] if x in sentiment_dict else 0)
train_1['min_magnitude'] = train_1['PetID'].apply(lambda x: sentiment_dict[x]['min_magnitude'] if x in sentiment_dict else 0)

test_1['language'] = test_1['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else None)
test_1['magnitude'] = test_1['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
test_1['score'] = test_1['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)
test_1['max_score'] = test_1['PetID'].apply(lambda x: sentiment_dict[x]['max_score'] if x in sentiment_dict else 0)
test_1['min_score'] = test_1['PetID'].apply(lambda x: sentiment_dict[x]['min_score'] if x in sentiment_dict else 0)
test_1['max_magnitude'] = test_1['PetID'].apply(lambda x: sentiment_dict[x]['max_magnitude'] if x in sentiment_dict else 0)
test_1['min_magnitude'] = test_1['PetID'].apply(lambda x: sentiment_dict[x]['min_magnitude'] if x in sentiment_dict else 0)


# ## We use TfidfVectorizer function to transform the text columns into numerical vector.

# In[ ]:


text_columns = ['Description', 'metadata_annots_top_desc', 'sentiment_entities']
for col in text_columns:
    train_1[col] = train_1[col].fillna('None')
    test_1[col] = test_1[col].fillna('None')
train_1['Length_Description'] = train_1['Description'].map(len)
train_1['Length_metadata_annots_top_desc'] = train_1['metadata_annots_top_desc'].map(len)
train_1['Lengths_sentiment_entities'] = train_1['sentiment_entities'].map(len)
test_1['Length_Description'] = test_1['Description'].map(len)
test_1['Length_metadata_annots_top_desc'] = test_1['metadata_annots_top_desc'].map(len)
test_1['Lengths_sentiment_entities'] = test_1['sentiment_entities'].map(len)


# In[ ]:


#train
n_components = 32
text_features = []

# Generate text features:
for txtcol in text_columns:
    
    # Initialize decomposition methods:
    print(f'generating features from: {txtcol}')
    tfv = TfidfVectorizer(min_df=2,  max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)
    svd_ = TruncatedSVD(
        n_components=n_components, random_state=1337)
    
    tfidf_col = tfv.fit_transform(train_1[txtcol].values)
    
    svd_col = svd_.fit_transform(tfidf_col)
    svd_col = pd.DataFrame(svd_col)
    svd_col = svd_col.add_prefix('TFIDF_{}_'.format(txtcol))
    
    text_features.append(svd_col)
    
text_features = pd.concat(text_features, axis=1)

train_2 = pd.concat([train_1, text_features], axis=1)

for txtcol in text_columns:
    train_2.drop(txtcol, axis = 1, inplace = True) 
    
    
#test
n_components = 32
text_features = []

# Generate text features:
for txtcol in text_columns:
    
    # Initialize decomposition methods:
    print(f'generating features from: {txtcol}')
    tfv = TfidfVectorizer(min_df=2,  max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)
    svd_ = TruncatedSVD(
        n_components=n_components, random_state=1337)
    
    tfidf_col = tfv.fit_transform(test_1[txtcol].values)
    
    svd_col = svd_.fit_transform(tfidf_col)
    svd_col = pd.DataFrame(svd_col)
    svd_col = svd_col.add_prefix('TFIDF_{}_'.format(txtcol))
    
    text_features.append(svd_col)
    
text_features = pd.concat(text_features, axis=1)

test_2 = pd.concat([test_1, text_features], axis=1)

for txtcol in text_columns:
    test_2.drop(txtcol, axis = 1, inplace = True) 


# In[ ]:


train_2.head(2)


# ### Treat "Name": there are different expressions such as "None" or "No Name Yet"  in the Name column all meaning that the pet has no name. I map such names to 0 and to length otherwise

# In[ ]:


def Namelength(x):
    x = str(x)
    if 'No' in x.split() or 'None' in x.split():
        return 0
    else:
        return len(x)


# In[ ]:


train_2['Lengths_Name'] = train_2['Name'].map(Namelength)
train_2.drop('Name', axis = 1, inplace = True)
test_2['Lengths_Name'] = test_2['Name'].map(Namelength)
test_2.drop('Name', axis = 1, inplace = True)


# In[ ]:


train_2.head(2)


# ## Add image_size features

# In[ ]:


split_char = '/'
from PIL import Image
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

# agg_imgs = pd.concat([agg_train_imgs, agg_test_imgs], axis=0).reset_index(drop=True)
train_3 = train_2.merge(agg_train_imgs, how='left', on='PetID')
test_3 = test_2.merge(agg_test_imgs, how='left', on='PetID')


# ## Use pretrained models to extract more feature

# In[ ]:


import cv2
import os
from keras.applications.densenet import preprocess_input, DenseNet121


# In[ ]:


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

def load_image(path, pet_id, pic_num):
    image = cv2.imread(f'{path}{pet_id}-{pic_num}.jpg')
    new_image = resize_to_square(image)
    new_image = preprocess_input(new_image)
    return new_image


# In[ ]:


img_size = 256
batch_size = 32


# In[ ]:


from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Flatten, Lambda, AveragePooling2D, MaxPooling2D
import keras.backend as K
inp = Input((256,256,3))
backbone = DenseNet121(input_tensor = inp, 
                       weights="../input/densenet-keras/DenseNet-BC-121-32-no-top.h5",
                       include_top = False)
x = backbone.output
x = MaxPooling2D(pool_size=4, strides=4)(x)
x = AveragePooling2D(2)(x)
out = Flatten()(x)

m = Model(inp,out)


# In[ ]:


m.summary()


# In[ ]:


def gen_image_feature(df, batch_size = 32, img_size = 256, mode = 'train',pic_num = '1'):
    pet_ids = df['PetID'].values
    n_batches = len(pet_ids) // batch_size + 1

    features = {}
    for b in tqdm(range(n_batches)):
        start = b*batch_size
        end = (b+1)*batch_size
        batch_pets = pet_ids[start:end]
        batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
        for i,pet_id in enumerate(batch_pets):
            try:
                batch_images[i] = load_image("../input/petfinder-adoption-prediction/{mode}_images/", pet_id, pic_num)
            except:
                pass
        batch_preds = m.predict(batch_images)
        for i,pet_id in enumerate(batch_pets):
            features[pet_id] = batch_preds[i]
    return features


# In[ ]:


train_imagefeature1 = gen_image_feature(train_df, batch_size = 32, img_size = 256, mode = 'train',pic_num = '1')
train_imagefeature2 = gen_image_feature(train_df, batch_size = 32, img_size = 256, mode = 'train',pic_num = '2')
train_imagefeature3 = gen_image_feature(train_df, batch_size = 32, img_size = 256, mode = 'train',pic_num = '3')
test_imagefeature1 = gen_image_feature(test_df, batch_size = 32, img_size = 256, mode = 'test',pic_num = '1')
test_imagefeature2 = gen_image_feature(test_df, batch_size = 32, img_size = 256, mode = 'test',pic_num = '2')
test_imagefeature3 = gen_image_feature(test_df, batch_size = 32, img_size = 256, mode = 'test',pic_num = '3')


# In[ ]:


train_feats1 = pd.DataFrame.from_dict(train_imagefeature1, orient='index')
train_feats2 = pd.DataFrame.from_dict(train_imagefeature2, orient='index')
train_feats3 = pd.DataFrame.from_dict(train_imagefeature3, orient='index')
train_feats = pd.concat([train_feats1,train_feats2,train_feats3],axis = 1)
train_feats.columns = [f'pic_{i}' for i in range(train_feats.shape[1])]
test_feats1 = pd.DataFrame.from_dict(test_imagefeature1, orient='index')
test_feats2 = pd.DataFrame.from_dict(test_imagefeature2, orient='index')
test_feats3 = pd.DataFrame.from_dict(test_imagefeature3, orient='index')
test_feats = pd.concat([test_feats1,test_feats2,test_feats3], axis = 1)
test_feats.columns = [f'pic_{i}' for i in range(test_feats.shape[1])]
train_feats = train_feats.reset_index()
train_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)

test_feats = test_feats.reset_index()
test_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)


# ## We use TruncatedSVD to reduce the dimentsion

# In[ ]:


n_components = 32
svd_ = TruncatedSVD(n_components=n_components, random_state=2017)

# features_df = pd.concat([train_feats, test_feats], axis=0)
train_features = train_feats[[f'pic_{i}' for i in range(256*3)]].values

svd_col = svd_.fit_transform(train_features)
svd_col = pd.DataFrame(svd_col)
svd_col = svd_col.add_prefix('IMG_SVD_')

train_img_features = pd.concat([train_df['PetID'], svd_col], axis=1)

n_components = 32
svd_ = TruncatedSVD(n_components=n_components, random_state=2017)

# features_df = pd.concat([train_feats, test_feats], axis=0)
test_features = test_feats[[f'pic_{i}' for i in range(256*3)]].values

svd_col = svd_.fit_transform(test_features)
svd_col = pd.DataFrame(svd_col)
svd_col = svd_col.add_prefix('IMG_SVD_')

test_img_features = pd.concat([test_df['PetID'], svd_col], axis=1)


# In[ ]:


train_4 = train_3.merge(train_img_features, how='left', on='PetID')
test_4 = test_3.merge(test_img_features, how='left', on='PetID')


# ## Combine train and test together to do category-to-num transformation and analysis

# In[ ]:


alldata = pd.concat([train_4, test_4], ignore_index=True, sort=False)


# In[ ]:


#num to catergory
col_num2cat = [ 'Breed1', 'Breed2', 'Color1', 'Color2', 'Color3','State']
alldata[col_num2cat] = alldata[col_num2cat].astype(str)


# In[ ]:


#category cols
cat_cols = col_num2cat + ['language','RescuerID']
#fill nan in language using mode of the same state group
alldata['language'] = alldata.groupby('State')['language'].transform(lambda x: x.fillna(x.mode()[0]))
alldata.fillna(0)


# In[ ]:


def cat2count(df,var):
    count = df.groupby([var])['PetID'].count().reset_index()
    count.columns = [var, var+'_COUNT']
    return count


# In[ ]:


for var in cat_cols:
    count = cat2count(alldata,var)
    alldata = alldata.merge(count, how='left', on=var)


# In[ ]:


objects = []
for i in alldata.columns:
    if alldata[i].dtype == object:
        objects.append(i)
# features.update(features[objects].fillna('None'))
# drop column PetID
alldata.drop(objects, axis = 1, inplace = True)


# In[ ]:


numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []
for i in alldata.columns:
    if alldata[i].dtype in numeric_dtypes:
        numerics.append(i)
# numerics
# features.update(features[numerics].fillna(0))


# In[ ]:


from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


# In[ ]:


skew_features = alldata[numerics].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index
skew_features


# In[ ]:


# for i in skew_index:
#     alldata[i] = boxcox1p(alldata[i], boxcox_normmax(alldata[i] + 1))


# ## split back train and test

# In[ ]:


X_train = alldata.loc[np.isfinite(alldata.AdoptionSpeed), :]
X_test = alldata.loc[~np.isfinite(alldata.AdoptionSpeed), :]
label = X_train['AdoptionSpeed']

X_test = X_test.drop(['AdoptionSpeed'], axis=1)

assert X_train.shape[0] == train_df.shape[0]
assert X_test.shape[0] == test_df.shape[0]


# ## LGBM Model

# In[ ]:


import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

param = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 15,
          'max_depth': 10,
          'learning_rate': 0.0075,
          'bagging_fraction': 0.8,
          'feature_fraction': 0.7,
          'min_split_gain': 0.02,
          'min_child_samples': 55,
          'min_child_weight': 0.02,
          'lambda_l2': 0.022,
          'verbosity': 1,
          'data_random_seed': 2017,
#           'early_stop': 500,
#           'verbose_eval': 5000,
          'num_rounds': 1000000
}


# In[ ]:


def run_lgb(params, X_train, X_test, random_state):
    n_splits = 8
    

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0]))
    feature_importance_df = pd.DataFrame()
    
    

    for fold_,(train_idx, valid_idx) in enumerate(kf.split(X_train, X_train['AdoptionSpeed'].values)):
        print("Fold {}".format(fold_))
        
        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)
        
        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val)

        
        clf = lgb.train(param, trn_data,  valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1000)
        
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = X_val.columns
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        valid_pred = clf.predict(X_val, num_iteration=clf.best_iteration)
        test_pred = clf.predict(X_test, num_iteration=clf.best_iteration) 

        oof_train[valid_idx] = valid_pred
        oof_test += test_pred/n_splits

        
    return clf, oof_train, oof_test, feature_importance_df


# In[ ]:


n_run = 7
prediction = np.zeros((X_test.shape[0]))
train_pred = np.zeros((X_train.shape[0]))
for i in range(n_run):
    model, oof_train, oof_test, feature_importance_df = run_lgb(param, X_train, X_test, i*2019)
    prediction += oof_test/n_run
    train_pred += oof_train/n_run


# In[ ]:


cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:150].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,28))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('Features importance (averaged/folds)')
plt.tight_layout()
plt.savefig('FI.png')


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


# In[ ]:


optR = OptimizedRounder()
optR.fit(oof_train, X_train['AdoptionSpeed'].values)
coefficients = optR.coefficients()
valid_pred = optR.predict(train_pred, coefficients)
qwk = kappa(X_train['AdoptionSpeed'].values, valid_pred)
print("train kappa = ", qwk)


# In[ ]:


train_predictions = optR.predict(train_pred, coefficients).astype(np.int8)
test_predictions = optR.predict(prediction, coefficients).astype(np.int8)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(X_train['AdoptionSpeed'].values, train_predictions)


# In[ ]:


submission = pd.DataFrame({'PetID': test_df['PetID'].values, 'AdoptionSpeed': test_predictions})
submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:




