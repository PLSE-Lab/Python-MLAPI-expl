#!/usr/bin/env python
# coding: utf-8

# ## What you will learn?

# Not much, just how to implement LightGBM )
# 
# Motivation to write this kernel was the tabular data structure, which means data is in the form of a table(pandas dataframe). This data structure and the task of binary classifcation is similar to other competition, "Santander Customer Transaction Prediction", where top kernels use LightGBM. Sooo, I tried it here as well)
# 
# PS: no EDA here

# # here, imagine some cool picture about dota2)

# ### Now, let's import all required packages 

# In[1]:


import os #to access files
import pandas as pd #to work with dataframes
import numpy as np #just a tradition
from sklearn.model_selection import StratifiedKFold #for cross-validation
from sklearn.metrics import roc_auc_score #this is we are trying to increase
import matplotlib.pyplot as plt #we will plot something at the end)
import seaborn as sns #same reason
import lightgbm as lgb #the model we gonna use


# ## Let's read the data: train, target and test

# In[2]:


get_ipython().run_cell_magic('time', '', "PATH_TO_DATA = '../input/'\n\ndf_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_features.csv'), index_col='match_id_hash')#.sample(frac=0.01)\ndf_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_targets.csv'), index_col='match_id_hash')#.sample(frac=0.01)\ndf_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), index_col='match_id_hash')#.sample(frac=0.01)")


# ## Lets have a look what are these data look like:

# In[3]:


df_train_features.head(2)


# In[4]:


df_train_targets.head(2)


# In[5]:


df_test_features.head(2)


# I have no idea what these features mean...I prefer FIFA)

# In[6]:


#turn to X and y notations for train data and target
X = df_train_features.values
y = df_train_targets['radiant_win'].values #extract the colomn we need


# In[7]:


#this is to make sure we have "ujson" and "tqdm"
try:
    import ujson as json
except ModuleNotFoundError:
    import json
    print ('Please install ujson to read JSON oblects faster')
    
try:
    from tqdm import tqdm_notebook
except ModuleNotFoundError:
    tqdm_notebook = lambda x: x
    print ('Please install tqdm to track progress with Python loops')


# In[8]:


#a helper function, we will use it in next cell
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


# Now, we define a function which adds some new features:
# 
# PS: all of these are from "how to start" kernel by @yorko

# In[9]:


def add_new_features(df_features, matches_file):
    
    # Process raw data and add new features
    for match in read_matches(matches_file):
        match_id_hash = match['match_id_hash']

        # Counting ruined towers for both teams
        radiant_tower_kills = 0
        dire_tower_kills = 0
        for objective in match['objectives']:
            if objective['type'] == 'CHAT_MESSAGE_TOWER_KILL':
                if objective['team'] == 2:
                    radiant_tower_kills += 1
                if objective['team'] == 3:
                    dire_tower_kills += 1

        # Write new features
        df_features.loc[match_id_hash, 'radiant_tower_kills'] = radiant_tower_kills
        df_features.loc[match_id_hash, 'dire_tower_kills'] = dire_tower_kills
        df_features.loc[match_id_hash, 'diff_tower_kills'] = radiant_tower_kills - dire_tower_kills
        
        # ... here you can add more features ...
        


# In[10]:


get_ipython().run_cell_magic('time', '', "# copy the dataframe with features\ndf_train_features_extended = df_train_features.copy()\ndf_test_features_extended = df_test_features.copy()\n\n# add new features\nadd_new_features(df_train_features_extended, os.path.join(PATH_TO_DATA, 'train_matches.jsonl'))\nadd_new_features(df_test_features_extended, os.path.join(PATH_TO_DATA, 'test_matches.jsonl'))")


# In[11]:


#Just a shorter names for data
newtrain=df_train_features_extended
newtest=df_test_features_extended
target=pd.DataFrame(y)


# In[12]:


#lastly, check the shapes, Andrew Ng approved)
newtrain.shape,target.shape, newtest.shape


# After running the LightGBM model, we will visualize something called "feature importance", which  kind of shows which features and how much they affected the final result. For this reason we need to store feature names:

# In[13]:


features=newtrain.columns


# ## Noow, let's define LightGBM parameters. 
# 
# Personally, I understand only some of these parameters. So, these are some random set up. Maybe it is better to look up the official documentation. Tuning these parameters probably will increase your score.
# 
# Investigation in process...

# In[14]:


param = {
        'bagging_freq': 5,
        'bagging_fraction': 0.5,
        'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': 0.05,
        'learning_rate': 0.01,
        'max_depth': -1,  
        'metric':'auc',
        'min_data_in_leaf': 50,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 10,
        'num_threads': 5,
        'tree_learner': 'serial',
        'objective': 'binary', 
        'verbosity': 1
    }


# # Finally, let's run the model

# In[15]:


get_ipython().run_cell_magic('time', '', '#divide training data into train and validaton folds\nfolds = StratifiedKFold(n_splits=5, shuffle=False, random_state=17)\n\n#placeholder for out-of-fold, i.e. validation scores\noof = np.zeros(len(newtrain))\n\n#for predictions\npredictions = np.zeros(len(newtest))\n\n#and for feature importance\nfeature_importance_df = pd.DataFrame()\n\n#RUN THE LOOP OVER FOLDS\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(newtrain.values, target.values)):\n    \n    X_train, y_train = newtrain.iloc[trn_idx], target.iloc[trn_idx]\n    X_valid, y_valid = newtrain.iloc[val_idx], target.iloc[val_idx]\n    \n    print("Computing Fold {}".format(fold_))\n    trn_data = lgb.Dataset(X_train, label=y_train)\n    val_data = lgb.Dataset(X_valid, label=y_valid)\n\n    \n    num_round = 5000 \n    verbose=1000 \n    stop=500 \n    \n    #TRAIN THE MODEL\n    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=verbose, early_stopping_rounds = stop)\n    \n    #CALCULATE PREDICTION FOR VALIDATION SET\n    oof[val_idx] = clf.predict(newtrain.iloc[val_idx], num_iteration=clf.best_iteration)\n    \n    #FEATURE IMPORTANCE\n    fold_importance_df = pd.DataFrame()\n    fold_importance_df["Feature"] = features\n    fold_importance_df["importance"] = clf.feature_importance()\n    fold_importance_df["fold"] = fold_ + 1\n    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)\n    \n    #CALCULATE PREDICTIONS FOR TEST DATA, using best_iteration on the fold\n    predictions += clf.predict(newtest, num_iteration=clf.best_iteration) / folds.n_splits\n\n#print overall cross-validatino score\nprint("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))')


# ## Feature Importance

# In[16]:


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


# From these feature importance chart, we can see that "x_gold" features play significant role in making the prediction. But still need more investigation of dota2 features...

# ## Prepare submission file

# In[17]:


df_submission = pd.DataFrame({'radiant_win_prob': predictions}, 
                                 index=df_test_features.index)
import datetime
submission_filename = 'submission_{}.csv'.format(
    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
df_submission.to_csv(submission_filename)
print('Submission saved to {}'.format(submission_filename))


# # What's next?
# 
# * try to tune parameters, it will definitely improve your LB score
# * try to come up with good features
# * read other kernels
# * try other models as well

# ### Hopefully, this kernel was usefull. Feel free to fork, comment and upvote!

# # 1'st stage of hyper-param tuning: tuning model complexity

# * https://stackoverflow.com/questions/50686645/grid-search-with-lightgbm-example
# * https://www.kaggle.com/kashnitsky/topic-10-practice-with-logit-rf-and-lightgbm

# In[33]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

gkf = KFold(n_splits=5, shuffle=False, random_state=42).split(X=newtrain.values, y=target.values.ravel())

param_grid = {
    'num_leaves': [10, 31, 127],
    'max_depth': [3, 4, 5, 6, -1],
    #'reg_alpha': [0.1, 0.5],
    #'min_data_in_leaf': [30, 400],
    #'lambda_l1': [0, 1, 1.5],
    #'lambda_l2': [0, 1],
    #'num_boost_round' : [500,1000]
    }

lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective='binary', learning_rate=0.1, metric='auc', verbose=1, n_estimators=500)

gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=gkf)
lgb_model = gsearch.fit(X=newtrain.values, y=target.values.ravel())

print(lgb_model.best_params_, lgb_model.best_score_)


# In[19]:


from lightgbm import LGBMClassifier
lgb_clf = LGBMClassifier(random_state=17, n_estimators=500)


# In[20]:


get_ipython().run_cell_magic('time', '', 'lgb_clf.fit(X_train, y_train)')


# In[21]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_valid, lgb_clf.predict(X_valid))


# In[22]:


param_grid = {'num_leaves': [7, 15, 31, 63], 
              'max_depth': [3, 4, 5, 6, -1]}


# In[23]:


grid_searcher = GridSearchCV(estimator=lgb_clf, param_grid=param_grid, cv=5, verbose=1, n_jobs=4)
grid_searcher.fit(X_train, y_train)


# In[24]:


grid_searcher.best_params_, grid_searcher.best_score_


# In[25]:


roc_auc_score(y_valid, grid_searcher.predict(X_valid))


# # 2'nd stage of hyper-param tuning: convergence:

# In[ ]:


num_iterations = 1500

lgb_clf2 = LGBMClassifier(random_state=17, max_depth=4, num_leaves=15, n_estimators=num_iterations, n_jobs=1)

param_grid2 = {'learning_rate': np.logspace(-3, 0, 10)}
grid_searcher2 = GridSearchCV(estimator=lgb_clf2, param_grid=param_grid2, cv=5, verbose=1, n_jobs=4)
grid_searcher2.fit(X_train, y_train)


# In[ ]:


print(grid_searcher2.best_params_, grid_searcher2.best_score_)
print(roc_auc_score(y_valid, grid_searcher2.predict(X_valid)))


# In[28]:


final_lgb = LGBMClassifier(n_estimators=1500, num_leaves=31,learning_rate=0.046, max_depth=-1, n_jobs=4)


# In[29]:


get_ipython().run_cell_magic('time', '', 'final_lgb.fit(newtrain.values, target.values.ravel())')


# In[30]:


get_ipython().run_cell_magic('time', '', 'lgb_final_pred = final_lgb.predict_proba(newtest.values)')


# In[31]:


predictions = lgb_final_pred[:,1]
predictions.shape


# In[32]:


df_submission = pd.DataFrame({'radiant_win_prob': predictions}, index=df_test_features.index)
import datetime
submission_filename = 'submission_opt_{}.csv'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
df_submission.to_csv(submission_filename)
print('Submission saved to {}'.format(submission_filename))


# In[ ]:




