#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import data visualization
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# # Pivot EDA might hint why cutmix works.
# 
# As you will see each root-consonant-vowel combination has roughly 150 images, so full classes are well balanced. However, (root-consonant), (root-vowel) and (consonant-vowel) are constructed in a peculiar way. For isntance, less popular consonants are mostly mixed with more popular vowels and the other way around. I am not sure if this have been done on purpose, but this is a good hint for any model to overfit on combinations given in the training set. 
# 
# For instance, consonant_3 is exclusively paired to vowel_0, so it is likely to use vowel_0 presence as a feature for consonant_3. Apparently cutmix works well because it may "construct" combinations that are not present in the training set. 
# 
# A crossvalidation/augumentation strategy must focus on preventing the model from overfitting on combinations given in the train set, becasue test set is likely to conatin significantly different combinations. (which goes inline with the purpose of the competition i.e. "focusing the problem on the grapheme components rather than on recognizing whole graphemes should make it possible to assemble a Bengali OCR system without handwriting samples for all 10,000 graphemes.")

# In[ ]:


# setup the input data folder
DATA_PATH = '../input/bengaliai-cv19/'


# In[ ]:


# load the dataframes with labels
train_labels = pd.read_csv(DATA_PATH + 'train.csv')
test_labels = pd.read_csv(DATA_PATH + 'test.csv')
class_map = pd.read_csv(DATA_PATH + 'class_map.csv')
sample_submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')


# In[ ]:


# count the number of images for each grapheme_root
root_img = train_labels.groupby(by=['grapheme_root']).count().reset_index()[['grapheme_root', 'image_id']]
root_img.sort_values(by=['image_id'], ascending=False, inplace=True)
root_img.rename(columns={'grapheme_root':'root','image_id': 'root_count'}, inplace=True)

# count the number of images for each grapheme_root
vowel_img = train_labels.groupby(by=['vowel_diacritic']).count().reset_index()[['vowel_diacritic', 'image_id']]
vowel_img.sort_values(by=['image_id'], ascending=False, inplace=True)
vowel_img.rename(columns={'image_id': 'vowel_count'}, inplace=True)

# count the number of images for each grapheme_root
consonant_img = train_labels.groupby(by=['consonant_diacritic']).count().reset_index()[['consonant_diacritic', 'image_id']]
consonant_img.sort_values(by=['image_id'], ascending=False, inplace=True)
consonant_img.rename(columns={'image_id': 'consonant_count'}, inplace=True)


# ## Pivots
# ### Root - consonants
# As you can see least populated columns have roughly 150 images each.
# A bit more populated columns have around 300 images, for example columns 82, 37, 19, and 34

# In[ ]:


root_consonant = pd.pivot_table(train_labels, values='image_id', index=['consonant_diacritic'],
                                    columns=['grapheme_root'], aggfunc='count')

root_consonant = root_consonant.reindex(consonant_img.index, columns=root_img.index).fillna(0)
root_consonant.iloc[:,120:150]


# ### Root - vowels
# If we look at columns 82,37,19,34 again, we note that they have about 300 images each because they have two image sets for two vowel diacritics. Hence, every root-vowel-consonant set has roughly 150 images.

# In[ ]:


root_vowel = pd.pivot_table(train_labels, values='image_id', index=['vowel_diacritic'],
                                    columns=['grapheme_root'], aggfunc='count')

root_vowel = root_vowel.reindex(vowel_img.index, columns=root_img.index).fillna(0)
root_vowel.iloc[:,120:150]


# ### 150 images per combination
# Every combination has roughly 150 images, so 'full' classes are well balanced.

# In[ ]:


train_labels['image'] = train_labels['grapheme_root'].astype(str)        + '_' + train_labels['vowel_diacritic'].astype(str)        + '_' + train_labels['consonant_diacritic'].astype(str)

image = train_labels.groupby(by=['image']).count().reset_index()[['image', 'image_id']]
image.sort_values(by=['image_id'], ascending=False, inplace=True)
image.rename(columns={'image_id': 'image_count'}, inplace=True)
image['image_count'].describe()


# ## Consonant - Vowels:
# As you can see the matrix is diogonal, so less popular consonants are mostly mixed with more popular vowels and the other way around. I am not sure if this have been done on purpose, but this is a good hint for model to overfit on training set. For instance, consonant_3 never only appears with vowel_0.
# 
# A crossvalidation/augumentation strategy must focus on preventing the model from overfitting on given combinations. Apparently cutmix works well because it "constructs" additional combinations.

# In[ ]:


consonant_vowel = pd.pivot_table(train_labels, values='image_id', index=['consonant_diacritic'],
                                    columns=['vowel_diacritic'], aggfunc='count')

consonant_vowel = consonant_vowel.reindex(consonant_img.index, columns=vowel_img.index).fillna(0)
consonant_vowel


# One example of validation strategy might be using combinations on the diogonal for validation. Or random subsets of combinations from full combination set.
