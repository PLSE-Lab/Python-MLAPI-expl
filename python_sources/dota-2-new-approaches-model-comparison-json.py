#!/usr/bin/env python
# coding: utf-8

#  # Dota 2 Winner Prediction: Tips&Tricks, Approaches, Features, Models and etc.
#  ![](https://cdn-st1.rtr-vesti.ru/vh/pictures/hdr/164/936/0.jpg)
#  
#  
#  In this kernel we will calmly walk through some of the approaches and ideas that maybe useful for the competition. Don't even need to fasten your seatbelts, it's okay :)
#  
#  **We'll cover the following:**
#  - Base Estimator Setting
#  - Teammembers Substraction And Combining Approach
#  - Approach To The Categorial Vars
#  - Model Comparison (Logistic Regression, Random Forest, CatBoost Classifier, FF NN with Keras)
#  - *Special Gift*

# At the very first step we need to import all necessary frameworks.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
PATH_TO_DATA = '../input/'


# Loading the data that we have from organizers.
# 
# P.S. Many data scientists on the internet claim checking datasets' shape to be a good habit, I also like it, so sorry if it makes you furious.

# In[ ]:


#Importing initial training dataset with targets and test dataset

train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_features.csv'), 
                                    index_col='match_id_hash')
train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 
                                            'train_targets.csv'), 
                                   index_col='match_id_hash')
test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 
                                            'test_features.csv'), 
                                   index_col='match_id_hash')
print(train_df.shape, train_targets.shape, test_df.shape)


# Display the head of the training targets.

# In[ ]:


train_targets.head()


# To start accounting from somewhere we need to settle a base estimator's score for the raw data we have. We'll use a Random Forest model without tuning, just to see, whether our approaches are good, bad or maybe require further analysis.

# In[ ]:


get_ipython().run_cell_magic('time', '', "y_train = train_targets.radiant_win #extract the target variable\n\n#Now make a train-test split, we'll see that results on the holdout set correlate with CV results.\n\nX_train_part, X_valid, y_train_part, y_valid = train_test_split(train_df, y_train, test_size = 0.3, random_state=0) #fixing random_state\n\n#Settling a CV scheme.\ncv = ShuffleSplit(n_splits=5, random_state=1) #using a shuffle split for CV \n\n#Implement RF with just 100 estimators not to wait too long.\nrf = RandomForestClassifier(n_estimators=100, random_state=1)\nrf.fit(X_train_part, y_train_part)\n\n#Count CV scoring and houldout scoring: \nholdout_score = roc_auc_score(y_valid, rf.predict_proba(X_valid)[:,1])\ncv_score = cross_val_score(rf, train_df, y_train, cv=cv, scoring = 'roc_auc') ")


# In[ ]:


#Let's look at the results.
print('CV scores: ', cv_score)
print('CV mean: ', cv_score.mean())
print('Holdout score: ', holdout_score)


# P.S. Notice that results of CV and holdout are really close :)
# 
# And now let's look at the training set to understand the way to move.

#  # Teammembers Substraction And Combining Approach

# In[ ]:


train_df.head()


# We see great number of features probably divided by teammembers and they don't really mean anything to me (as fat as I'm a sports simulators fan). Let's try to find some description on the internet. 
# We can [find](https://www.quora.com/What-is-DOTA-What-are-its-rules) that this is a team game where they have to tear apart some buildings or something like that.
# 
# So let's take such an approach: find the difference between each teammember's qualities and combine them to team ones as in [this](https://www.kaggle.com/daemonis/combine-hero-features-into-team-ones-basic) beautiful kernel.
# 
# Beware, we won't combine *hero ids* (for now) and *firstblood* claimed columns because the first ones are definitely categorial and the second ones are binary.
# 
# Also let's concatenate training and test set, it won't spoil the thing greatly but it'll be more convenient to work.

# In[ ]:


idx_split = train_df.shape[0]
full_df = pd.concat((train_df, test_df))

print(train_df.shape, test_df.shape, full_df.shape)


# Extracting needed columns names.

# In[ ]:


cols = [] 
for i in full_df.columns[5:29]: #list of columns for r1 player
    if i[3:] != 'hero_id' and i[3:] != 'firstblood_claimed':
        cols.append(i[3:]) #drop r1_
print(cols)


# In[ ]:


def substract_numeric_features (df, feature_suffixes):
    col_names=[]
    df_out = df.copy()
    for feat_suff in feature_suffixes:
        for index in range(1,6):
            df_out[f'{index}_{feat_suff}_substract'] = df[f'r{index}_{feat_suff}'] - df[f'd{index}_{feat_suff}'] # e.g. r1_kills - d1_kills
            col_names.append(f'd{index}_{feat_suff}')
            col_names.append(f'r{index}_{feat_suff}')
    df_out.drop(columns = col_names, inplace=True)
    return df_out

#Run the function
full_df_mod = substract_numeric_features(full_df, cols)
full_df_mod.head()


# While combimimg features we might choose almost any aggregating operation that we think might be useful.

# In[ ]:


def combine_sub_features (df_out, feature_suffixes):
    for feat_suff in feature_suffixes:
            player_col_names = [f'{i}_{feat_suff}_substract' for i in range(1,6)] # e.g. 1_gold_substract
            
            df_out[f'{feat_suff}_max_substract'] = df_out[player_col_names].max(axis=1) # e.g. gold_max_substract
            
            df_out[f'{feat_suff}_min_substract'] = df_out[player_col_names].min(axis=1) # e.g. gold_min_substract
            
            df_out[f'{feat_suff}_sum_substract'] = df_out[player_col_names].sum(axis=1) # e.g. gold_sum_substract

            
            df_out.drop(columns=player_col_names, inplace=True) # remove teammembers' substract features from the dataset
    return df_out

#Run the function. Suffixes remain the same
full_df_mod = combine_sub_features(full_df_mod, cols)
full_df_mod.head()


# Now let's see what we've got.

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Remember we need to use only training part of the full set\nX_train_part_1, X_valid_1, y_train_part_1, y_valid_1 = train_test_split(full_df_mod[:idx_split], y_train, test_size = 0.3, random_state=0) #fixing random_state\n\nrf = RandomForestClassifier(n_estimators=100, random_state=1)\nrf.fit(X_train_part_1, y_train_part_1)\n\n#Count CV scoring and houldout scoring: \nholdout_score_1 = roc_auc_score(y_valid_1, rf.predict_proba(X_valid_1)[:,1])\ncv_score_1 = cross_val_score(rf, full_df_mod[:idx_split], y_train, cv=cv, scoring = 'roc_auc') ")


# In[ ]:


#New results.
print('CV scores: ', cv_score_1)
print('CV mean: ', cv_score_1.mean())
print('CV std:', cv_score_1.std())
print('Holdout score: ', holdout_score_1)
print('Better results on CV: ', cv_score_1>cv_score)


# Wonderful! We've made less features to compute, got better results and actually we haven't even dealed with categorial and binary features! (Once again we see a correlation between CV and holdout scores :)) 

# # Approach To The Categorial Vars

# [](http://)From now let's consider hero ids. You can check that there are 120 unique values (1-120). I've looked at a number of DOTA 2 heroes [pictures](https://cyberpowerpc.files.wordpress.com/2016/03/dota2-heroes-view-on-pc-gaming-console.png?w=700&h=391) and found that there're generally 3 categories of heroes with almost the same number of heroes in them.
# 
# So the next approach is totally made out of a hope.
# 
# For heroes we find their hero types (hoping there's something in it). Then we also make and kind of an invariant sum for the team (using logarithm). These features will be treated as categorial and we can just get dummies.
# 
# The same is done for hero ids but we gonna treat them as numerical because there're too many combinations (I understand that there is no meaning but still).

# In[ ]:


def herotype_approach(df):
    r_heroes = ['r%s_hero_id' %i for i in range(1,6)] # e.g. r1_hero_id...
    d_heroes = ['d%s_hero_id' %i for i in range(1,6)] # e.g. d1_hero_id...
    r_herotypes = ['r%s_hero_type' %i for i in range(1,6)] # e.g. r1_hero_type...
    d_herotypes = ['d%s_hero_type' %i for i in range(1,6)] # e.g. d1_hero_type...

    df['r_hero_invar_sum'] = np.log(df[r_heroes]).sum(axis=1) #sum of logs of hero ids for the team r
    df['d_hero_invar_sum'] = np.log(df[d_heroes]).sum(axis=1) #sum of logs of hero ids for the team d
    df['hero_invar_sum_diff'] = df['r_hero_invar_sum'] - df['d_hero_invar_sum'] #their difference (don't try to find the meaning)
    
    df[r_herotypes] = df[r_heroes].apply(lambda x: (x//40)+1) #hero types like 1,2,3 supposing there's about equal number of heroes of each type
    df[d_herotypes] = df[d_heroes].apply(lambda x: (x//40)+1)
    
    df['r_invar_herotype_sum'] = np.log(df[r_herotypes]).sum(axis=1).astype(str) # findning an invariant sum to treat as categorial
    df['d_invar_herotype_sum'] = np.log(df[d_herotypes]).sum(axis=1).astype(str)
    
    return df

full_df_mod = herotype_approach(full_df_mod)


# We'll also use an approach from [this](https://www.kaggle.com/utapyngo/dota-2-how-to-make-use-of-hero-ids) wonderful kernel, I really like that idea!

# In[ ]:


def hero_approach(df):
    for team in 'r', 'd':
        players = [f'{team}{i}' for i in range(1, 6)]
        hero_columns = [f'{player}_hero_id' for player in players]

        d = pd.get_dummies(df[hero_columns[0]])
        for c in hero_columns[1:]:
            d += pd.get_dummies(df[c])
        df = pd.concat([df, d.add_prefix(f'{team}_hero_')], axis=1)
        df.drop(columns=hero_columns, inplace=True)
    return df

full_df_mod = hero_approach(full_df_mod)


# Getting dummies to a separate dataframe so that we can further use raw *full_df_mod* for CatBoost.

# In[ ]:


r_firstblood = ['r%s_firstblood_claimed' %i for i in range(1,6)] 
d_firstblood = ['d%s_firstblood_claimed' %i for i in range(1,6)] 
r_herotypes = ['r%s_hero_type' %i for i in range(1,6)]
d_herotypes = ['d%s_hero_type' %i for i in range(1,6)]

full_df_dum = pd.get_dummies(full_df_mod, columns = ['r_invar_herotype_sum', 'd_invar_herotype_sum'] + r_firstblood + d_firstblood)
full_df_dum.head()


# And finally let's see what we've got.

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Remember we need to use only training part of the full set\nX_train_part_2, X_valid_2, y_train_part_2, y_valid_2 = train_test_split(full_df_dum[:idx_split], y_train, test_size = 0.3, random_state=0) #fixing random_state\n\nrf = RandomForestClassifier(n_estimators=100, random_state=1)\nrf.fit(X_train_part_2, y_train_part_2)\n\n#Count CV scoring and houldout scoring: \nholdout_score_2 = roc_auc_score(y_valid_2, rf.predict_proba(X_valid_2)[:,1])\ncv_score_2 = cross_val_score(rf, full_df_dum[:idx_split], y_train, cv=cv, scoring = 'roc_auc') ")


# In[ ]:


#New results.
print('CV scores: ', cv_score_2)
print('CV mean: ', cv_score_2.mean())
print('CV std:', cv_score_2.std())
print('Holdout score: ', holdout_score_2)
print('Better results on CV: ', cv_score_2>cv_score_1)


# Alright. I've described a number of approaches and some of them sometimes seem to be misunderstading or illogical but it's up to you evaluate them, decide and move forward!
# 
# What can be useful:
# - Finding different approaches to numerical features
# - Feature selection
# - Chosing the right way to treat categorial vars
# - Researching... and then again researching :)

# # Model Comparison
# 
# Here we'll compare some models' perfomances on the dataset we've created. 
# As it's a binary classification task we'll use some familiar algos like Logistic Regression and Random Forest (we've actually already used), also we'll try boosting with CatBoost (LightGBM is a nightmare for me after Flight Delays competition :D) and implement a FF Neural Net with Keras.
# 
# **Models to compare:**
# - Logistic Regression
# - Random Forest
# - CatBoost
# - FF NN with Keras

# **1. Logistic Regression**
# 
# To implement a Logistic Regression model we just need to scale the data and here we go. We won't tune the models, we'll evaluate them out-of-box.

# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\nfull_df_scaled = scaler.fit_transform(full_df_dum) #scalling the full dataset, that's not correct but it saves time\n\nX_train_part_lr, X_valid_lr, y_train_part_lr, y_valid_lr = train_test_split(full_df_scaled[:idx_split], \n                                                                        y_train, test_size = 0.3, random_state=0) #fixing random_state\nlr = LogisticRegression(random_state=0, solver='liblinear')\nlr.fit(X_train_part_lr, y_train_part_lr)\n\nlr_ho_score =  roc_auc_score(y_valid_lr, lr.predict_proba(X_valid_lr)[:,1])\nlr_cv_score = cross_val_score(lr, full_df_scaled[:idx_split], y_train, cv=cv, scoring = 'roc_auc') \n\ndel full_df_dum")


# In[ ]:


#Logistic regression results.
print('CV scores LR: ', lr_cv_score)
print('CV mean LR: ', lr_cv_score.mean())
print('CV std LR:', lr_cv_score.std())
print('Holdout score LR: ', lr_ho_score)


# **2. Random Forest**
# 
# Recall our RF results.

# In[ ]:


rf_cv_score, rf_ho_score = cv_score_2, holdout_score_2 

print('CV scores RF: ', rf_cv_score)
print('CV mean RF: ', rf_cv_score.mean())
print('CV std RF:', rf_cv_score.std())
print('Holdout score RF: ', rf_ho_score)


# **3. CatBoost Classifier**
# 
# It's famous for working well out-of-box and also can use GPU and can run much faster.
# 
# Also it provides unique algos to work with categorial vars (did you know that 'Cat' in CatBoost refers to Categorial? :) )

# In[ ]:


get_ipython().run_cell_magic('time', '', "from catboost import CatBoostClassifier\n#We'll use full_df_mod without dummies and mark categorial vars\nX_train_part_ctb, X_valid_ctb, y_train_part_ctb, y_valid_ctb = train_test_split(full_df_mod[:idx_split], \n                                                                        y_train, test_size = 0.3, random_state=0) #fixing random_state\ncat_vars = ['r_invar_herotype_sum', 'd_invar_herotype_sum'] + r_firstblood + d_firstblood #all the vars that we got dummies of\n\n#Let it train for 200 iterations not to wait too long\nctb = CatBoostClassifier(iterations = 200, random_state=1, verbose=False, task_type='GPU', eval_metric='AUC', cat_features=cat_vars)\n\n#We'll look at an online validation plot\nctb.fit(X_train_part_ctb, y_train_part_ctb.astype(float), eval_set=(X_valid_ctb, y_valid_ctb.astype(float)), plot=True)\n\nctb_ho_score =  roc_auc_score(y_valid_ctb.astype(float), ctb.predict_proba(X_valid_ctb)[:,1])\nctb_cv_score = cross_val_score(ctb, full_df_mod[:idx_split], y_train.astype(float), cv=cv, scoring = 'roc_auc') ")


# In[ ]:


print('CV scores CTB: ', ctb_cv_score)
print('CV mean CTB: ', ctb_cv_score.mean())
print('CV std CTB:', ctb_cv_score.std())
print('Holdout score CTB: ', ctb_ho_score)


# **4. FF NN witn Keras**
# 
# Let's implement a simple FF NN with the help of Keras.

# In[ ]:


#!pip install keras
#!pip install tensorflow


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

from keras import backend as K
from keras import regularizers
from keras import optimizers

#Defining ROC AUC in Keras for evaluation.
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# Now let's build a simple model with just one hidden layer.

# In[ ]:


# to find a number of input dimensions
full_df_scaled.shape


# In[ ]:


def model_function():
    model = Sequential()
    model.add(Dense(50, input_dim = 396, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])
    return model


# We use 1 hidden layer with 50 neurons and RELU, also with use Batch Normalization layer and output layer has 1 neuron with a sigmoid.

# In[ ]:


keras_net = KerasClassifier(build_fn=model_function, epochs=10, batch_size=32, verbose=1) #it's an sklearn wrapper for model
keras_net.fit(X_train_part_lr, y_train_part_lr.astype(float)) #we'll use scaled training part like for LR
nn_ho_score =  roc_auc_score(y_valid_lr.astype(float), keras_net.predict_proba(X_valid_lr)[:,1])


# In[ ]:


keras_net = KerasClassifier(build_fn=model_function, epochs=10, batch_size=32, verbose=False) #turn off the verbose

nn_cv_score = cross_val_score(keras_net, full_df_scaled[:idx_split], y_train.astype(float), cv=cv, scoring = 'roc_auc') 


# In[ ]:


print('CV scores nn: ', nn_cv_score)
print('CV mean nn: ', nn_cv_score.mean())
print('CV std nn:', nn_cv_score.std())
print('Holdout score nn: ', nn_ho_score)


# As you see it's hard to decide which model with parameter tuning will get the best result. 'No Free Lunch' in action. And once again it's up to you to decide which way to move. Still don't think that NNs are bad for this task. We didn't tune anything and learned just a little.

# # Special Gift
# 
# **Raw JSON extraction**

# I'll show a way how to improve Yorko's approach (function) and extract players' data from raw json.

# In[ ]:


import json 

#Collect needed columns names
with open(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')) as fin:
        for i in range(150):
            first_line = fin.readline()
            data_json = json.loads(first_line)
data_json.keys()
key = []
for i in data_json['players'][9].keys():
    if i not in cols: #remember we've settled columns from full dataset
        key.append(i)


# In[ ]:


import collections
from tqdm import tqdm_notebook
def read_matches(matches_file):
    
    MATCHES_COUNT = {
        'test_matches.jsonl': 10000,
        'train_matches.jsonl': 39675,
    }
    _, filename = os.path.split(matches_file)
    total_matches = MATCHES_COUNT.get(filename)
    
    with open(matches_file) as fin:
        for line in tqdm_notebook(fin, total=total_matches):
            yield json.loads(line)

#Extracting function

def extract_features_csv(match, keys):
    row = [
        ('match_id_hash', match['match_id_hash']),
    ]
        
    for slot, player in enumerate(match['players']):
        if slot < 5:
            player_name = 'r%d' % (slot + 1)
        else:
            player_name = 'd%d' % (slot - 4)
# The main idea: if we have int or float or bool - return it, else - return the length of the item
        for field in keys:
            if (type(player[field]) == int) or (type(player[field]) == float) or (type(player[field]) == bool): 
                column_name = '%s_%s' % (player_name, field)
                row.append((column_name, player[field]))
            else:
                column_name = '%s_%s' % (player_name, field)
                row.append((column_name, len(player[field])))
    return collections.OrderedDict(row)


# I won't perform it here just to save the time and kernel memory. But you can use it as a template for further investigations.

# In[ ]:



#df_new_features = []
#for match in read_matches(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')):
#    match_id_hash = match['match_id_hash']
#    features = extract_features_csv(match, key)

#    df_new_features.append(features)
#df_new_features = pd.DataFrame.from_records(df_new_features).set_index('match_id_hash')
#df_new_features.head()


# Since now, I suppose, you have all necessary power :) 
# 
# You're welcome!
# 
# Now come and research... and research :)
# 
# **P.S. I'm not claiming that everything here is a silver bullet (or at least something), I've just shared my own ideas and will be happy to discuss any possible issues in comments.**
# 
# **P.S.S. I'd pleased to get teaming up suggestions. I'm located in Minsk, Belarus, but distant cooperation is not a problem, I suppose :) **
# 
# 
# 
# 
# Sincerely yours
# 
# **slack: Vlad Kisin**
# 

# # Please, be fair and feel free to upvote this kernel if you've found it somehow useful :)
# ![](http://critterbabies.com/wp-content/gallery/kittens/803864926_1375572583.jpg)
