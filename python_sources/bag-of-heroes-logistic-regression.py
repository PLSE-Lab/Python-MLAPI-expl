#!/usr/bin/env python
# coding: utf-8

# <img src='https://steamuserimages-a.akamaihd.net/ugc/1047377062453152758/48CD2809864A478C11592F098A59F8B76C2A2D14/'>
# 
# # <center> Dota 2: Bag of Heroes
# 
# In this Kernel I show how to create a simple "bag of heroes", i.e. a sparse matrix (or dataframe) where each column represents a hero (his ID). The value of a cell (i, j) is zero/non-zero if the hero at j-th column wasn't/was present in the i-th match. For the non-zero values representing hero presense, I use `1` for the Radiant team hero and `-1` for the Dire team hero (distinguishing between teams helps!).
# 
# Note that hero duplicates are not allowed in a match, so in a 5v5 match there are 10 unique heroes out of 115 existing heroes (keep in mind that hero IDs are `1, ... 23, 25, ..., 114, 119, 120`).
# 
# Synergy and anti-synergy of heroes can be also considered, i.e. combinations of heroes within each team and between teams respectively. However I found that considering hero combinations doesn't help to improve the LB score -- perhaps because our dataset is too small. Also, as a side effect, it greatly increases the dimensionality of the feature space (e.g. there are $\frac{115!}{3! \cdot 112!} = 246905$ combinations of 3 heroes in the set of 115 heroes...).
# 
# I use Logistic Regression to make predictions based (merely) on the bag-of-heroes and compare some of the encoding approaches. The best score on LB that I got with the bag-of-heroes is slightly above 0.6 which is better than the score of 0.5 obtained without encoding the hero IDs at all or with one-hot encoding them for each player.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import time
import datetime
import pytz
from itertools import combinations
from scipy.sparse import hstack, csr_matrix
import os

# Sklearn stuff
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


PATH_TO_DATA = '../input/'
SEED = 17


# # Load data

# In[ ]:


# Train dataset
df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_features.csv'), 
                                    index_col='match_id_hash')
df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_targets.csv'), 
                                   index_col='match_id_hash')

y_train = df_train_targets['radiant_win'].map({True: 1, False: 0})

# Test dataset
df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), 
                                   index_col='match_id_hash')


# In[ ]:


df_train_features.head()


# Combine train and test datasets and create heroes dataframe

# In[ ]:


df_full_features = pd.concat([df_train_features, df_test_features])

# Index to split the training and test data sets
idx_split = df_train_features.shape[0]

heroes_df = df_full_features[[f'{t}{i}_hero_id' for t in ['r', 'd'] for i in range(1, 6)]]


# In[ ]:


heroes_df.head()


# In[ ]:


heroes_df.shape


# Note that our train dataset is more or less balanced, so assuming that the test dataset has the same class distribution, we expect the baseline score to be close to 0.5.

# In[ ]:


sns.countplot(x=y_train, palette="Set3");


# # Logistic Regression
# In the following I will use Logistic Regression for different bag-of-heroes representations and compare the CV scores. Let's create a helper function

# In[ ]:


def logit_cv(X_heroes_train, y_train, cv=5, random_state=SEED):
    logit = LogisticRegression(random_state=SEED, solver='liblinear')

    c_values = np.logspace(-2, 1, 20)

    logit_grid_searcher = GridSearchCV(estimator=logit, param_grid={'C': c_values},
                                       scoring='roc_auc',return_train_score=False, cv=cv,
                                       n_jobs=-1, verbose=0)

    logit_grid_searcher.fit(X_heroes_train, y_train)
    
    cv_scores = []
    for i in range(logit_grid_searcher.n_splits_):
        cv_scores.append(logit_grid_searcher.cv_results_[f'split{i}_test_score'][logit_grid_searcher.best_index_])
    print(f'CV scores: {cv_scores}')
    print(f'Mean: {np.mean(cv_scores)}, std: {np.std(cv_scores)}\n')
    
    return logit_grid_searcher.best_estimator_, np.array(cv_scores) 


# I also will use stratified cross-validation with 5 folds

# In[ ]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


# # One-hot encoding each player
# Let's first try to one-hot encode heroes of each player (this is the approach I also used for Neural Network classifier, see the [Kernel](https://www.kaggle.com/kuzand/dota-2-winner-prediction-multilayer-nn-pytorch)). 

# In[ ]:


heroes_df_ohe = heroes_df.copy()
for t in ['r', 'd']:
    for i in range(1, 6):
        heroes_df_ohe = pd.get_dummies(heroes_df_ohe, columns = [f'{t}{i}_hero_id'])
        
heroes_df_ohe.head()

X_heroes_train = heroes_df_ohe[:idx_split]
X_heroes_test  = heroes_df_ohe[idx_split:]


# In[ ]:


print(f'Number of features: {heroes_df_ohe.shape[1]}')


# Let's compute the logit cv scores

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nlogit_0, cv_scores_0 = logit_cv(X_heroes_train, y_train, cv=skf, random_state=SEED)')


# With this naive one-hot-encoding we already get a score above the dummy baseline score of 0.5.
# Let's see if we can improve it with the bag-of-heroes representation.

# # Bag of heroes
# Let's write a function which will allow us to easily create different bag-of-heroes representations. It's a little bit hacky, but it works well (however I don't recomend using it for `N > 3` if you don't have enough memory). 

# In[ ]:


def bag_of_heroes(df, N=1, r_val=1, d_val=-1, r_d_val=0, return_as='csr'):
    '''
    Bag of Heroes. Returns a csr matrix (+ list of feature names) or dataframe where each column represents
    a hero (ID) and each row represents a match.
    
    The value of a cell (i, j) in the returned matrix is:
        cell[i, j] = 0, if the hero or combination of heroes of the j-th column is not present in the i-th match
        cell[i, j] = r_val, if the hero (N = 1) or combination of heroes (N > 1, synergy) of the j-th column is within the Radiant team,
        cell[i, j] = d_val, if the hero (N = 1) or combination of heroes (N > 1, synergy) of the j-th column is within the Dire team,
        cell[i, j] = r_d_val, if the combination of heroes of the j-th column is between the Radiant and Dire teams (N>1, anti-synergy).
    
    Parameters:
    -----------
        df: dataframe with hero IDs, with columns ['r1_hero_id', ..., 'r5_hero_id', 'd1_hero_id', ..., 'd5_hero_id']
        N: integer 1 <= N <= 10, for N heroes combinations
        return_as: 'csr' for scipy csr sparse matrix, 'df' for pandas dataframe
    '''
    if N < 1 or N > df.shape[1]:
        raise Exception(f'The number N of hero-combinations should be 1 <= N <= {df.shape[1]}')
        
    # Convert the integer IDs to strings of the form id{x}{x}{x}
    df = df.astype(str).applymap(lambda x: 'id' + '0'*(3 - len(x)) + x)
    
    # Create a list of all hero IDs present in df
    hero_ids = np.unique(df).tolist()

    # Break df into teams Radiant (r) and Dire (d)
    df_r = df[[col for col in df.columns if col[0] == 'r']]
    df_d = df[[col for col in df.columns if col[0] == 'd']]
    
    # Create a list of all the hero IDs in df, df_r and df_d respectively
    f = lambda x: ' '.join(['_'.join(c) for c in combinations(sorted(x), N)])
    
    df_list = df.apply(f, axis=1).tolist()
    df_list.append(' '.join(['_'.join(c) for c in combinations(hero_ids, N)]))

    df_r_list = df_r.apply(f, axis=1).tolist()
    df_r_list.append(' '.join(['_'.join(c) for c in combinations(hero_ids, N)]))
    
    df_d_list = df_d.apply(f, axis=1).tolist()
    df_d_list.append(' '.join(['_'.join(c) for c in combinations(hero_ids, N)]))
    
    # Create countvectorizers
    vectorizer = CountVectorizer()
    vectorizer_r = CountVectorizer()
    vectorizer_d = CountVectorizer()
    
    X = vectorizer.fit_transform(df_list)[:-1]
    X_r = vectorizer_r.fit_transform(df_r_list)[:-1]
    X_d = vectorizer_d.fit_transform(df_d_list)[:-1]
    X_r_d = (X - (X_r + X_d))  
    X = (r_val * X_r + d_val * X_d + r_d_val * X_r_d)
    
    feature_names = vectorizer.get_feature_names()
    
    if return_as == 'csr':
        return X, feature_names
    elif return_as == 'df':
        return pd.DataFrame(X.toarray(), columns=feature_names, index=df.index).to_sparse(0) 


# Let's consider a toy example of three 2v2 matches and 5 heroes to demonstrate the concept

# In[ ]:


df = pd.DataFrame({'r1_hero_id': [1, 5, 2], 'r2_hero_id': [3, 4, 1],
                   'd1_hero_id': [5, 3, 3], 'd2_hero_id': [4, 2, 5]},
                 index=['match_1', 'match_2', 'match_3'])

df


# In[ ]:


# bag_of_heroes(df, N=1, r_val=1, d_val=-1, return_as='csr')[0]
bag_of_heroes(df, N=1, r_val=1, d_val=-1, return_as='df')


# We can also include pairs of heroes. For the sake of clarity the value `2` is used for the bpairs of heroes from opposing teams

# In[ ]:


pd.concat([bag_of_heroes(df, N=1, r_val=1, d_val=-1, return_as='df'),
           bag_of_heroes(df, N=2, r_val=1, d_val=-1, r_d_val=2, return_as='df')], axis=1)


# ## Individual heroes
# Now let's use our real data and start with `N=1`, i.e. bag of individual heroes

# In[ ]:


boh = bag_of_heroes(heroes_df, N=1, r_val=1, d_val=-1, return_as='csr')[0]

X_heroes_train = boh[:idx_split]
X_heroes_test  = boh[idx_split:]


# In[ ]:


print(f'Number of features: {boh.shape[1]}') 


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nlogit_1, cv_scores_1 = logit_cv(X_heroes_train, y_train, cv=skf, random_state=SEED)')


# In[ ]:


cv_scores_1 > cv_scores_0


# ## Synergy
# Let's take into account the hero pairings within each team (two-hero synergy). Assuming that the abilities of individual heroes are more important than the heroes synergy, smaller values `r_val`, `d_val` for `N=2` than for `N=1` are used. But these value choices are quite arbitrary and can be tuned.

# In[ ]:


boh = hstack([bag_of_heroes(heroes_df, N=1, r_val=1, d_val=-1, r_d_val=0, return_as='csr')[0],
              bag_of_heroes(heroes_df, N=2, r_val=0.2, d_val=-0.2, r_d_val=0, return_as='csr')[0]], format='csr')

X_heroes_train = boh[:idx_split]
X_heroes_test  = boh[idx_split:]


# In[ ]:


print(f'Number of features: {boh.shape[1]}')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nlogit_2, cv_scores_2 = logit_cv(X_heroes_train, y_train, cv=skf, random_state=SEED)')


# In[ ]:


cv_scores_2 > cv_scores_1


# ## Anti-synergy

# In[ ]:


boh = hstack([bag_of_heroes(heroes_df, N=1, r_val=1, d_val=-1, r_d_val=0, return_as='csr')[0],
              bag_of_heroes(heroes_df, N=2, r_val=0.2, d_val=-0.2, r_d_val=0.1, return_as='csr')[0]], format='csr')

X_heroes_train = boh[:idx_split]
X_heroes_test  = boh[idx_split:]


# In[ ]:


print(f'Number of features: {boh.shape[1]}') 


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nlogit_3, cv_scores_3 = logit_cv(X_heroes_train, y_train, cv=skf, random_state=SEED)')


# In[ ]:


cv_scores_3 > cv_scores_2


# # Conclusion
# Using bag-of-heroes for hero IDs with Logistic regression gives ROC AUC score around 0.6 which is quite good compared to score obtained from raw IDs or one-hot encoded IDs for each player. Unfortunately considering synergy and anti-synergy of heroes didn't help improving the score on the LB (although the CV scores slightly improved). My guess is that the dataset is too small for the synergy and anti-synergy of heroes to be seen by the model as a signal. Here are the results:

# | |  CV_mean  | CV_std | LB |
# | :--- | :--- | :--- | 
# |**1**	|0.611003	|0.003721	|0.60127|
# |**2**	|0.611519	|0.004123	|0.60062|
# |**3**	|0.611673	|0.004046	|0.60020|

# There are possibly better ways to deal with the hero IDs. Maybe a more clever bag-of-heroes encoding scheme or a different representation such as mapping hero IDs to hero win rates. Any suggestions are welcome!
