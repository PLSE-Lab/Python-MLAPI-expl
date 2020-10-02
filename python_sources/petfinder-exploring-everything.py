#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
tqdm.pandas()
import gc
gc.collect()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv('../input/train/train.csv')
df_breed = pd.read_csv('../input/breed_labels.csv')
df_color = pd.read_csv('../input/color_labels.csv')
df_state = pd.read_csv('../input/state_labels.csv')
df_test = pd.read_csv('../input/test/test.csv')


# ## EDA - Train Data
# ### Data Fields
# PetID - Unique hash ID of pet profile
# 
# AdoptionSpeed - Categorical speed of adoption. Lower is faster. This is the value to predict. See below section for more info.
# 
# Type - Type of animal (1 = Dog, 2 = Cat)
# 
# Name - Name of pet (Empty if not named)
# 
# Age - Age of pet when listed, in months
# 
# Breed1 - Primary breed of pet (Refer to BreedLabels dictionary)
# 
# Breed2 - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)
# 
# Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
# 
# Color1 - Color 1 of pet (Refer to ColorLabels dictionary)
# 
# Color2 - Color 2 of pet (Refer to ColorLabels dictionary)
# 
# Color3 - Color 3 of pet (Refer to ColorLabels dictionary)
# 
# MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
# 
# FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
# 
# Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
# 
# Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
# 
# Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
# 
# Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
# 
# Quantity - Number of pets represented in profile
# 
# Fee - Adoption fee (0 = Free)
# 
# State - State location in Malaysia (Refer to StateLabels dictionary)
# 
# RescuerID - Unique hash ID of rescuer
# 
# VideoAmt - Total uploaded videos for this pet
# 
# PhotoAmt - Total uploaded photos for this pet
# 
# Description - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.
# ## AdoptionSpeed
# Contestants are required to predict this value. The value is determined by how quickly, if at all, a pet is adopted. The values are determined in the following way: 
# 
# 0 - Pet was adopted on the same day as it was listed. 
# 
# 1 - Pet was adopted between 1 and 7 days (1st week) after being listed. 
# 
# 2 - Pet was adopted between 8 and 30 days (1st month) after being listed. 
# 
# 3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed. 
# 
# 4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days).
# 
# 

# In[ ]:


df_train.head()


# In[ ]:


df_breed.head()


# In[ ]:


df_color.head()


# In[ ]:


df_state.head()


# In[ ]:


sns.countplot(df_train.AdoptionSpeed)


# **0 Class is little less(which is obvious) but others are almost on similar level**

# In[ ]:


sns.countplot(df_train.Type)


# In[ ]:


# Checking Age Distribution
f, ax = plt.subplots(figsize=(21, 6))
sns.distplot(df_train.Age, ax=ax)


# In[ ]:


# Checking Age,Fee and AdoptionnSpeed correlation
sns.heatmap(df_train[['Age', 'Fee', 'AdoptionSpeed']].corr(), annot=True)


# In[ ]:


df_train.Age.describe()


# In[ ]:


# Since Age is skewed, We can try logarithmic transformation
sns.distplot(np.log(df_train.Age + 0.5))


# **Fee has almost zero coorrelation with Age**

# In[ ]:


# Checking put distribution of Breed1 and Breed2
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18,10))
sns.distplot(df_train.Breed1, ax=ax[0])
sns.distplot(df_train.Breed2, ax=ax[1])


# In[ ]:


df_train.Name = df_train.Name.fillna('')
df_test.Name = df_test.Name.fillna('')


# In[ ]:


df_train['Name'] = df_train['Name'].replace('No Name Yet', '')
df_test['Name'] = df_test['Name'].replace('No Name Yet', '')


# In[ ]:


df_train['name_len'] = df_train.Name.str.len()
df_test['name_len'] = df_test.Name.str.len()


# In[ ]:


df_train.name_len.head()


# In[ ]:


sns.distplot(df_train.name_len)


# In[ ]:


sns.heatmap(df_train[['name_len', 'AdoptionSpeed']].corr(), annot=True)


# In[ ]:


sns.distplot(np.log(df_train.name_len + 1 - df_train.name_len.min()))


# .**Breed1 and Breed2 also has concentrated distribution**

# In[ ]:


# Gender distribution
f, ax = plt.subplots(figsize=(21, 6))
sns.countplot(df_train.Gender, ax=ax)


# In[ ]:


# Quantity Distribution
f, ax = plt.subplots(figsize=(21, 6))
sns.distplot(df_train.Quantity, ax=ax)


# In[ ]:


f, ax = plt.subplots(figsize=(21, 6))
quant_gender1 = df_train[df_train['Gender'] == 1]
quant_gender2 = df_train[df_train['Gender'] == 2]
quant_gender3= df_train[df_train['Gender'] == 3]
sns.distplot(quant_gender1.Quantity, ax=ax , hist=False, rug=True)
sns.distplot(quant_gender2.Quantity, ax=ax,  hist=False, rug=True)
sns.distplot(quant_gender3.Quantity, ax=ax,  hist=False, rug=True)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(21, 6))
sns.countplot('Quantity',data=df_train,hue='Gender', ax=ax)


# **Quanity of gender 1 and 3 is much larger than the second one**

# In[ ]:


# Color distribution
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(18,10))
sns.countplot(df_train.Color1, ax=ax[0])
sns.countplot(df_train.Color2, ax=ax[1])
sns.countplot(df_train.Color3, ax=ax[2])


# In[ ]:


# Maturity Size Distribution
f, ax = plt.subplots(figsize=(21, 6))
sns.countplot(df_train.MaturitySize, ax=ax)


# In[ ]:


# Furlength
f, ax = plt.subplots(figsize=(21, 6))
sns.countplot(df_train.FurLength, ax=ax)


# In[ ]:


# Vaccination
f, ax = plt.subplots(figsize=(21, 6))
sns.distplot(df_train.Vaccinated, ax=ax)


# In[ ]:


fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(18,10))
sns.distplot(df_train.Sterilized, ax=ax[0])
sns.distplot(df_train.Health, ax=ax[1])
sns.distplot(df_train.Dewormed, ax=ax[2])
sns.distplot(df_train.Fee, ax=ax[3])


# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
sns.countplot(df_train.State, ax=ax)


# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
sns.distplot(df_train.Description.fillna('').str.len(), ax=ax)


# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18,10))
sns.distplot(df_train.PhotoAmt, ax=ax[0])
sns.distplot(df_train.VideoAmt, ax=ax[1])


# In[ ]:


f, ax = plt.subplots(figsize=(24, 18))
sns.heatmap(df_train.corr(), annot=True, ax=ax)


# **Vaccianted, Dewormed and Sterilized are positively correlated**
# 
# **Gender and Quantity are positively correlated**
# 
# **Breed and age are negatively correlated**
# 
# **Adoption speed is not correlated to fee**

# ## EDA-Sentiment Analysis

# In[ ]:


import json


# In[ ]:


train_sentiment_path = '../input/train_sentiment/'
test_sentiment_path = '../input/test_sentiment/'
train_meta_path = '../input/train_metadata/'
test_meta_path = '../input/test_metadata/'


# In[ ]:


def get_sentiment(pet_id, json_dir):
    try:
        with open(json_dir + pet_id + '.json') as f:
            data = json.load(f)
        return pd.Series((data['documentSentiment']['magnitude'], data['documentSentiment']['score']))
    except FileNotFoundError:
        return pd.Series((np.nan, np.nan))


# In[ ]:


df_train[['desc_magnitude', 'desc_score']] = df_train['PetID'].progress_apply(lambda x: get_sentiment(x, train_sentiment_path))
df_test[['desc_magnitude', 'desc_score']] = df_test['PetID'].progress_apply(lambda x: get_sentiment(x, test_sentiment_path))


# In[ ]:


df_train.head()


# In[ ]:


sns.heatmap(df_train[['desc_magnitude', 'desc_score', 'AdoptionSpeed']].corr(), annot=True)


# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18,10))
sns.distplot(df_train.desc_magnitude.dropna(), ax=ax[0])
sns.distplot(df_train.desc_score.dropna(), ax=ax[1])


# In[ ]:


df_train.desc_magnitude.count() / df_train.shape[0]


# In[ ]:


sns.distplot(np.log(df_train.desc_magnitude.dropna() + 0.5))


# In[ ]:


sns.heatmap(np.corrcoef(df_train.Description.fillna('').str.len(), df_train.AdoptionSpeed), annot=True)


# ## Checking Out Images meta

# In[ ]:


target = df_train['AdoptionSpeed']
train_id = df_train['PetID']
test_id = df_test['PetID']
df_train.drop(['AdoptionSpeed', 'PetID'], axis=1, inplace=True)
df_test.drop(['PetID'], axis=1, inplace=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', "vertex_xs = []\nvertex_ys = []\nbounding_confidences = []\nbounding_importance_fracs = []\ndominant_blues = []\ndominant_greens = []\ndominant_reds = []\ndominant_pixel_fracs = []\ndominant_scores = []\nlabel_descriptions = []\nlabel_scores = []\nnf_count = 0\nnl_count = 0\nfor pet in train_id:\n    try:\n        with open('../input/train_metadata/' + pet + '-1.json', 'r') as f:\n            data = json.load(f)\n        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']\n        vertex_xs.append(vertex_x)\n        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']\n        vertex_ys.append(vertex_y)\n        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']\n        bounding_confidences.append(bounding_confidence)\n        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)\n        bounding_importance_fracs.append(bounding_importance_frac)\n        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']\n        dominant_blues.append(dominant_blue)\n        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']\n        dominant_greens.append(dominant_green)\n        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']\n        dominant_reds.append(dominant_red)\n        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']\n        dominant_pixel_fracs.append(dominant_pixel_frac)\n        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']\n        dominant_scores.append(dominant_score)\n        if data.get('labelAnnotations'):\n            label_description = data['labelAnnotations'][0]['description']\n            label_descriptions.append(label_description)\n            label_score = data['labelAnnotations'][0]['score']\n            label_scores.append(label_score)\n        else:\n            nl_count += 1\n            label_descriptions.append('nothing')\n            label_scores.append(-1)\n    except FileNotFoundError:\n        nf_count += 1\n        vertex_xs.append(-1)\n        vertex_ys.append(-1)\n        bounding_confidences.append(-1)\n        bounding_importance_fracs.append(-1)\n        dominant_blues.append(-1)\n        dominant_greens.append(-1)\n        dominant_reds.append(-1)\n        dominant_pixel_fracs.append(-1)\n        dominant_scores.append(-1)\n        label_descriptions.append('nothing')\n        label_scores.append(-1)\n\nprint(nf_count)\nprint(nl_count)\ndf_train.loc[:, 'vertex_x'] = vertex_xs\ndf_train.loc[:, 'vertex_y'] = vertex_ys\ndf_train.loc[:, 'bounding_confidence'] = bounding_confidences\ndf_train.loc[:, 'bounding_importance'] = bounding_importance_fracs\ndf_train.loc[:, 'dominant_blue'] = dominant_blues\ndf_train.loc[:, 'dominant_green'] = dominant_greens\ndf_train.loc[:, 'dominant_red'] = dominant_reds\ndf_train.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs\ndf_train.loc[:, 'dominant_score'] = dominant_scores\ndf_train.loc[:, 'label_description'] = label_descriptions\ndf_train.loc[:, 'label_score'] = label_scores\n\n\nvertex_xs = []\nvertex_ys = []\nbounding_confidences = []\nbounding_importance_fracs = []\ndominant_blues = []\ndominant_greens = []\ndominant_reds = []\ndominant_pixel_fracs = []\ndominant_scores = []\nlabel_descriptions = []\nlabel_scores = []\nnf_count = 0\nnl_count = 0\nfor pet in test_id:\n    try:\n        with open('../input/test_metadata/' + pet + '-1.json', 'r') as f:\n            data = json.load(f)\n        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']\n        vertex_xs.append(vertex_x)\n        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']\n        vertex_ys.append(vertex_y)\n        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']\n        bounding_confidences.append(bounding_confidence)\n        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)\n        bounding_importance_fracs.append(bounding_importance_frac)\n        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']\n        dominant_blues.append(dominant_blue)\n        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']\n        dominant_greens.append(dominant_green)\n        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']\n        dominant_reds.append(dominant_red)\n        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']\n        dominant_pixel_fracs.append(dominant_pixel_frac)\n        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']\n        dominant_scores.append(dominant_score)\n        if data.get('labelAnnotations'):\n            label_description = data['labelAnnotations'][0]['description']\n            label_descriptions.append(label_description)\n            label_score = data['labelAnnotations'][0]['score']\n            label_scores.append(label_score)\n        else:\n            nl_count += 1\n            label_descriptions.append('nothing')\n            label_scores.append(-1)\n    except FileNotFoundError:\n        nf_count += 1\n        vertex_xs.append(-1)\n        vertex_ys.append(-1)\n        bounding_confidences.append(-1)\n        bounding_importance_fracs.append(-1)\n        dominant_blues.append(-1)\n        dominant_greens.append(-1)\n        dominant_reds.append(-1)\n        dominant_pixel_fracs.append(-1)\n        dominant_scores.append(-1)\n        label_descriptions.append('nothing')\n        label_scores.append(-1)\n\nprint(nf_count)\ndf_test.loc[:, 'vertex_x'] = vertex_xs\ndf_test.loc[:, 'vertex_y'] = vertex_ys\ndf_test.loc[:, 'bounding_confidence'] = bounding_confidences\ndf_test.loc[:, 'bounding_importance'] = bounding_importance_fracs\ndf_test.loc[:, 'dominant_blue'] = dominant_blues\ndf_test.loc[:, 'dominant_green'] = dominant_greens\ndf_test.loc[:, 'dominant_red'] = dominant_reds\ndf_test.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs\ndf_test.loc[:, 'dominant_score'] = dominant_scores\ndf_test.loc[:, 'label_description'] = label_descriptions\ndf_test.loc[:, 'label_score'] = label_scores")


# In[ ]:


image_meta_col = ['vertex_x', 'vertex_y', 'bounding_confidence', 'bounding_importance', 'dominant_blue', 'dominant_green', 'dominant_red', 'dominant_pixel_frac', 'dominant_score', 'label_score']


# In[ ]:


f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(df_train[image_meta_col].corr(), annot=True)


# In[ ]:


fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(18,10))
sns.distplot(df_train.vertex_x, ax=ax[0])
sns.distplot(df_train.vertex_y, ax=ax[1])
sns.distplot(np.log(df_train.vertex_x + 1 - df_train.vertex_x.min()), ax=ax[2])
sns.distplot(np.log(df_train.vertex_y + 1 - df_train.vertex_x.min()), ax=ax[3])


# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18,10))
sns.distplot(df_train.vertex_x + df_train.vertex_y, ax=ax[0])
sns.distplot(np.log(df_train.vertex_x + 1 - df_train.vertex_x.min()) +np.log(df_train.vertex_y + 1 - df_train.vertex_x.min()) , ax=ax[1])


# In[ ]:


sns.distplot(df_train.bounding_importance)


# In[ ]:


sns.distplot(df_train.bounding_confidence)


# In[ ]:


sns.distplot((df_train.bounding_confidence + df_train.bounding_importance) / 2)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(df_train.dominant_red, ax=ax, color='red', hist=False)
sns.distplot(df_train.dominant_green, ax=ax, color='green',  hist=False)
sns.distplot(df_train.dominant_blue, ax=ax, color='blue', hist=False)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(np.log(df_train.dominant_red), ax=ax, color='red', hist=False)
sns.distplot(np.log(df_train.dominant_green), ax=ax, color='green',  hist=False)
sns.distplot(np.log(df_train.dominant_blue), ax=ax, color='blue', hist=False)


# In[ ]:


sns.distplot(np.log((df_train.dominant_blue + df_train.dominant_green + df_train.dominant_red ) / 3 + 3))


# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(df_train.dominant_pixel_frac, ax=ax, color='red', hist=False)
sns.distplot(df_train.dominant_score, ax=ax, color='green',  hist=False)
sns.distplot(df_train.label_score, ax=ax, color='blue', hist=False)


# In[ ]:


sns.distplot((df_train.dominant_pixel_frac + df_train.dominant_score + df_train.label_score) / 3)


# In[ ]:


sns.heatmap(np.corrcoef((df_train.bounding_confidence + df_train.bounding_importance) / 2, np.log((df_train.dominant_pixel_frac + df_train.dominant_score + df_train.label_score) / 3 + 3)), annot=True)


# ## Data cleaning, filling and transformmations

# In[ ]:


df_train.isna().sum()


# In[ ]:


def log_transform(feature, df_train, df_test):
    min_feature = min(df_train[feature].min(), df_test[feature].min())
    df_train[feature] = np.log(df_train[feature] + 1 - min_feature)
    df_test[feature] = np.log(df_test[feature] + 1 - min_feature)
    return df_train, df_test


# In[ ]:


df_train, df_test = log_transform('vertex_x', df_train, df_test)


# In[ ]:


df_train, df_test = log_transform('vertex_y', df_train, df_test)


# In[ ]:


df_train['bounding_agg'] = (df_train.bounding_confidence + df_train.bounding_importance) / 2
df_test['bounding_agg'] = (df_test.bounding_confidence + df_test.bounding_importance) / 2


# In[ ]:


df_train['dominant_color'] = (df_train.dominant_blue + df_train.dominant_green + df_train.dominant_red ) / 3
df_test['dominant_color'] = (df_test.dominant_blue + df_test.dominant_green + df_test.dominant_red ) / 3


# In[ ]:


df_train, df_test = log_transform('dominant_color', df_train, df_test)


# In[ ]:


df_train['dominant_frac_agg'] = (df_train.dominant_pixel_frac + df_train.dominant_score + df_train.label_score) / 3
df_test['dominant_frac_agg'] = (df_test.dominant_pixel_frac + df_test.dominant_score + df_test.label_score) / 3


# In[ ]:


df_train.info()


# In[ ]:


df_train.drop(['Name', 'Description', 'RescuerID', 'bounding_confidence', 'bounding_importance', 'dominant_blue', 'dominant_green', 'dominant_red', 'dominant_pixel_frac', 'dominant_score', 'label_score', 'label_description'], axis=1, inplace=True)
df_test.drop(['Name', 'Description', 'RescuerID', 'bounding_confidence', 'bounding_importance', 'dominant_blue', 'dominant_green', 'dominant_red', 'dominant_pixel_frac', 'dominant_score', 'label_score', 'label_description'], axis=1, inplace=True)


# In[ ]:


df_train.info()


# In[ ]:


magnitude_std = df_train.desc_magnitude.std()
magnitude_mean = df_train.desc_magnitude.mean()
score_std = df_train.desc_score.std()
score_mean = df_train.desc_score.mean()
df_train['desc_magnitude'].fillna(np.random.normal(magnitude_mean, magnitude_std), inplace=True)
df_train['desc_score'].fillna( np.random.normal(score_mean, score_std), inplace=True)
df_test['desc_magnitude'].fillna(np.random.normal(magnitude_mean, magnitude_std), inplace=True)
df_test['desc_score'].fillna(np.random.normal(score_mean, score_std), inplace=True)


# In[ ]:


category_columns = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State']
numerical_columns = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'name_len', 'desc_magnitude', 'desc_score', 'bounding_agg', 'dominant_color', 'dominant_frac_agg']


# In[ ]:


df_train[category_columns] = df_train[category_columns].astype('category')
df_test[category_columns] = df_test[category_columns].astype('category')


# In[ ]:


min_age = min(df_train.Age.min(), df_test.Age.min())
df_train.Age = np.log(df_train.Age + 1 - min_age)
df_test.Age = np.log(df_test.Age + 1 - min_age)


# In[ ]:


min_magn = min(df_train.desc_magnitude.min(), df_test.desc_magnitude.min())
df_train.desc_magnitude = np.log(df_train.desc_magnitude + 1 - min_magn)
df_test.desc_magnitude = np.log(df_test.desc_magnitude + 1 - min_magn)


# In[ ]:


df_train.name_len = np.log(df_train.name_len + 1)
df_test.name_len = np.log(df_test.name_len + 1)


# In[ ]:


df_train.info()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(df_train, target, train_size=0.8, random_state=1234)


# ## Model Training

# In[ ]:


import lightgbm as lgbm


# In[ ]:


params_lgbm = {'num_leaves': 38,
         'min_data_in_leaf': 146, 
         'objective':'multiclass',
         'num_class': 5,
         'max_depth': 4,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9980062052116254,
         "bagging_freq": 1,
         "bagging_fraction": 0.844212672233457,
         "bagging_seed": 11,
         "metric": 'multi_logloss',
         "lambda_l1": 0.12757257166471625,
         "random_state": 133,
         "verbosity": -1
              }


# In[ ]:


lgbm_train = lgbm.Dataset(X_train, y_train, categorical_feature=category_columns)
lgbm_valid = lgbm.Dataset(X_val, y_val, categorical_feature=category_columns)


# In[ ]:


model_lgbm = lgbm.train(params_lgbm, lgbm_train, 10000, valid_sets=[lgbm_valid],  verbose_eval= 500, categorical_feature=category_columns, early_stopping_rounds = 200)


# In[ ]:


(np.argmax(model_lgbm.predict(X_val), axis=1) == y_val).sum() / y_val.shape[0]


# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
features = X_train.columns
importances = model_lgbm.feature_importance()
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier as Rf
from sklearn.ensemble import GradientBoostingClassifier as Gb


# In[ ]:


model_rf = Rf()


# In[ ]:


model_rf.fit(X_train,y_train)


# In[ ]:


model_rf.score(X_train,y_train)


# In[ ]:


model_rf.score(X_val,y_val)


# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
features = X_train.columns
importances = model_rf.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:


model_gb = Gb()


# In[ ]:


model_gb.fit(X_train,y_train)


# In[ ]:


model_gb.score(X_train,y_train)


# In[ ]:


model_gb.score(X_val,y_val)


# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
features = X_train.columns
importances = model_gb.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:


# Checking result of two best models
val_lgbm = model_lgbm.predict(X_val)
val_gb = model_gb.predict_log_proba(X_val)


# In[ ]:


val_mixed = (val_gb + val_lgbm) / 2


# In[ ]:





# In[ ]:


(np.argmax(val_mixed, axis=1) == y_val).sum() / y_val.shape[0]


# ## Submission Time

# In[ ]:


test_lgbm = np.argmax(model_lgbm.predict(df_test), axis=1)


# In[ ]:


test_id = pd.DataFrame(test_id)


# In[ ]:


submission = test_id.join(pd.DataFrame(test_lgbm, columns=['AdoptionSpeed']))


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




