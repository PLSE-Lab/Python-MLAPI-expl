#!/usr/bin/env python
# coding: utf-8

# ![](http://s3.amazonaws.com/criterion-production/editorial_content_posts/hero/5619-/VnYalPcKGVUK5e7coXKtrwCYYFxR1g_original.jpg)

# # Little addition to Hero items guide
# 
# *Even a walking dead man has a gun*

# This kernel is an attempt to move a little further in exploration of [Hero items](https://www.kaggle.com/grazder/hero-items-guide). Big chunk of code how to extract feature is the same as in that Guide. It's not shown here, i just loaded 'pickle' files from my kernel.

# In[ ]:


from pathlib import Path
import os
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, ShuffleSplit 

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')
plt.rc('axes', titlesize=18)     
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=16)  
plt.rc('ytick', labelsize=16)

import eli5
from IPython.display import display_html

PATH_TO_DATA = Path('../input/mlcourse-dota2-win-prediction')
SEED = 17


# In[ ]:


y_train = pd.read_csv(PATH_TO_DATA / 'train_targets.csv', index_col='match_id_hash')['radiant_win']
y_train = y_train.map({True: 1, False: 0})


# In[ ]:


import pickle as pkl
train_df = pd.read_pickle('../input/dota-2-extra-feat/items_train.pkl')
test_df = pd.read_pickle('../input/dota-2-extra-feat/items_test.pkl')


# Extracted data looks like this - a list of items for every player:

# In[ ]:


train_df.head()


# Now we make something different: create lists of items for teams and vectorize them. But first lets analyze - are there any unique items which are present only in train or test dataset.

# In[ ]:


# lists for players columns of teams
radiant_items = [f'r{i}_items' for i in range(1, 6)]
dire_items = [f'd{i}_items' for i in range(1, 6)]


# In[ ]:


# making list of strings of items for every team, where a string relates to a particular game
r_temp = train_df[radiant_items].apply(lambda row: ' '.join([' '.join(i) for i in row]), axis=1).tolist()
d_temp = train_df[dire_items].apply(lambda row: ' '.join([' '.join(i) for i in row]), axis=1).tolist()
r_temp_test = test_df[radiant_items].apply(lambda row: ' '.join([' '.join(i) for i in row]), axis=1).tolist()
d_temp_test = test_df[dire_items].apply(lambda row: ' '.join([' '.join(i) for i in row]), axis=1).tolist()


# In[ ]:


# function to see difference between sets of unique items
def symm_diff(list1, list2):
    unique1 = set(y for l in [x.split() for x in list1] for y in l)
    unique2 = set(y for l in [x.split() for x in list2] for y in l)
    print(len(unique1), len(unique2))
    
    return unique1 ^ unique2


# In[ ]:


# difference between teams in train
symm_diff(r_temp, d_temp)


# In[ ]:


# difference between teams in test
symm_diff(r_temp_test, d_temp_test)


# I found interesting items like **'river_painter'** and **'courier'**. First one has several names like **'river_painter3'**, **'river_painter6'** etc. Actually, this is a cosmetic item, useless for fighting. Is it significant if someone in team use this for fun? I guess just drop them both, these are rare items in train and test. Also get rid of recipe prefix (presuming it's potentially an item if created by cheats or bought for upgrade), as recipe items are rare enough too.

# In[ ]:


# normalization of items text
import re
rx = r'{0}[0-9]'.format('river_painter')
r_items = [re.sub(rx,'river_painter', x.replace('recipe_','')) for x in r_temp]
d_items = [re.sub(rx,'river_painter', x.replace('recipe_','')) for x in d_temp]
r_items_test = [re.sub(rx,'river_painter', x.replace('recipe_','')) for x in r_temp_test]
d_items_test = [re.sub(rx,'river_painter', x.replace('recipe_','')) for x in d_temp_test]


# Lets look what we've got:

# In[ ]:


# difference between teams in train
symm_diff(r_items, d_items)


# In[ ]:


# difference between teams in test
symm_diff(r_items_test, d_items_test)


# In[ ]:


# difference between radiants train vs test
symm_diff(r_items_test, r_items)


# In[ ]:


# difference between dires train vs test
symm_diff(d_items_test, d_items)


# I used modified approach instead of simple dummy encoding. Difference of radiant and dire matricies allows to reduce the feature space and, afaik, is good for linear models like logistic regression. Also we can see what items bring most advantage independently what team has it (f.e. upgraded or hard earned ones).

# In[ ]:


get_ipython().run_cell_magic('time', '', '#making occuerence matrix of items for every team\nvectorizer = CountVectorizer()\nr = vectorizer.fit_transform(r_items).toarray()\nd = vectorizer.transform(d_items).toarray()\nr_test = vectorizer.transform(r_items_test).toarray()\nd_test = vectorizer.transform(d_items_test).toarray()')


# In[ ]:


r.shape, d.shape, r_test.shape, d_test.shape


# So, we have 163 items, some of them (consumables) we will drop later. Barplots showing frequent and rare items are below.

# In[ ]:


# frequence of item calculated dividing by length of datasets
items = pd.DataFrame(np.vstack((np.sum(r, axis=0) / len(r), np.sum(d, axis=0) / len(r))).T, 
                     index=vectorizer.get_feature_names(), 
                     columns=['radiant', 'dire']).sort_values(by=['radiant'], ascending=False)
items_test = pd.DataFrame(np.vstack((np.sum(r_test, axis=0) / len(r_test), np.sum(d_test, axis=0) / len(r_test))).T, 
                     index=vectorizer.get_feature_names(), 
                     columns=['radiant', 'dire']).sort_values(by=['radiant'], ascending=False)


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 15))
tidy = items.iloc[:30, :].reset_index().rename(columns={'index': 'item'}).melt(id_vars='item').rename(columns=str.title)
sns.barplot(x='Value', y='Item', hue='Variable', data=tidy, ax=axes[0])
axes[0].set_title('Top 30 frequent items in train')
tidy_test = items_test.iloc[:30, :].reset_index().rename(columns={'index': 'item'}).melt(id_vars='item').rename(columns=str.title)
sns.barplot(x='Value', y='Item', hue='Variable', data=tidy_test, ax=axes[1])
axes[1].set_title('Top 30 frequent items in test')
  
#plt.tick_params(labelsize=14)
sns.despine(fig)
fig.tight_layout()


# Distribution of top frequent items for train and test datasets seems very close.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 15))
tidy = items.iloc[-30:, :].reset_index().rename(columns={'index': 'item'}).melt(id_vars='item').rename(columns=str.title)
sns.barplot(x='Value', y='Item', hue='Variable', data=tidy, ax=axes[0])
axes[0].set_title('Top 30 rare items in train')
axes[0].legend(loc='lower right')
tidy_test = items_test.iloc[-30:, :].reset_index().rename(columns={'index': 'item'}).melt(id_vars='item').rename(columns=str.title)
sns.barplot(x='Value', y='Item', hue='Variable', data=tidy_test, ax=axes[1])
axes[1].set_title('Top 30 rare items in test')
axes[1].legend(loc='lower right')
sns.despine(fig)
fig.tight_layout()


# In[ ]:


#making a dataframe from difference of matricies
count_vect_df = pd.DataFrame(r - d, columns=vectorizer.get_feature_names())
#count_vect_df_test = pd.DataFrame(r_test - d_test, columns=vectorizer.get_feature_names())


# That's how item feature looks like now:

# In[ ]:


count_vect_df.tail()


# In[ ]:


# consumables to be removed from data (minor impact on advantage of any team expected)
consumables = ['tango', 'tpscroll','bottle', 'flask', 'enchanted_mango', 'courier', 
               'clarity', 'faerie_fire', 'ward_observer', 'ward_sentry', 'river_painter']
count_vect_df.drop(columns=consumables, inplace=True)
#count_vect_df_test.drop(columns=consumables, inplace=True)


# We've got space of 152 features.

# In[ ]:


count_vect_df.shape


# Now let's check performance of logistic regression just on hero items feature.

# In[ ]:


def evaluate(df):
    train_df_part, valid_df, y_train_part, y_valid =         train_test_split(df, y_train, test_size=0.25, random_state=SEED)
    logreg = LogisticRegression(C=1, solver='liblinear', random_state=SEED)
    c_values = np.logspace(-2, 1.7, 20)
    shf = ShuffleSplit(n_splits=5, test_size=0.25, random_state=SEED)
    loggrid = GridSearchCV(estimator=logreg, 
                           param_grid={'C': c_values,
                                       'penalty': ['l1', 'l2'] 
                                      },
                           scoring='roc_auc', n_jobs=4, cv=shf, verbose=1)
    loggrid.fit(train_df_part, y_train_part)
    print(loggrid.best_score_, loggrid.best_params_)
    final_model = loggrid.best_estimator_
    final_model.fit(train_df_part, y_train_part)
    valid_pred = final_model.predict_proba(valid_df)[:, 1]
    score = roc_auc_score(y_valid, valid_pred)
    print('Score on validation set:', score)
    final_model.fit(df, y_train)
    res = cross_val_score(final_model, df, y_train, scoring='roc_auc', cv=shf, n_jobs=4, verbose=1)
    print('Scores on folds:', res)
    print('Standard deviation:',np.std(res))
    print('Mean score on whole set:',np.mean(res))
    features = df.columns.tolist()
    display_html(eli5.show_weights(estimator=final_model, 
                  feature_names=features, top=80))
    return final_model


# In[ ]:


classifier = evaluate(count_vect_df)


# Hero items feature only, with logistic regression trained, could give Public score more than 0.79  :)
# 
# We can expect that feature importance would be different with tree-based algorithms. To use this feature for Catboost, you need to call it catecorical.

# In[ ]:


from catboost import CatBoostClassifier, Pool

X_train_part, X_valid, y_train_part, y_valid = train_test_split(count_vect_df, 
                                                                y_train, 
                                                                test_size=0.25, 
                                                                random_state=SEED)

cat_feat_idx=count_vect_df.columns.tolist()

train_data = Pool(data=X_train_part, label=y_train_part, cat_features=cat_feat_idx)
valid_data = Pool(data=X_valid, label=y_valid, cat_features=cat_feat_idx)


params = {'loss_function':'Logloss', 
          'eval_metric':'AUC',
          'early_stopping_rounds': 200,
          'verbose': 200,
          'random_seed': SEED
         }

cbclf = CatBoostClassifier(**params) 
cbclf.fit(train_data,
          eval_set=valid_data, 
          use_best_model=True, 
          plot=True );


# In[ ]:


feature_importance_df = cbclf.get_feature_importance(prettified=True) 
plt.figure(figsize=(10, 20)) 
sns.barplot(x="Importances", y="Feature Id", data=feature_importance_df.iloc[0:80, :]) 
plt.title('CatBoost DOTA 2 top-80 items importances:');


# As we can see, some powerfull artifacts (ultimate_scepter, aegis, manta, radiance, sanga_and_yasha etc.) are very important features from Catboost's view. 
# 
# Thank you for attention.
