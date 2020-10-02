#!/usr/bin/env python
# coding: utf-8

# ## In this kernel you will learn how to implement LightGBM + Kfold technique which results in higher score!

# **LightGBM** is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:
# 
# * Faster training speed and higher efficiency.
# * Lower memory usage.
# * Better accuracy.
# * Support of parallel and GPU learning.
# * Capable of handling large-scale data.
# 
# You can read full documentation [here](https://lightgbm.readthedocs.io/en/latest/)

# # here, imagine some cool picture about dota2)

# ### Now, let's import all required packages 

# In[2]:


import os #to access files
import pandas as pd #to work with dataframes
import numpy as np #just a tradition
from sklearn.model_selection import StratifiedKFold #for cross-validation
from sklearn.metrics import roc_auc_score #this is we are trying to increase
import matplotlib.pyplot as plt #we will plot something at the end)
import seaborn as sns #same reason
import lightgbm as lgb #the model we gonna use


# ## Let's read the data: train, target and test

# In[3]:


get_ipython().run_cell_magic('time', '', "PATH_TO_DATA = '../input/'\n\ndf_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, \n                                             'train_features.csv'), \n                                    index_col='match_id_hash')\ndf_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, \n                                            'train_targets.csv'), \n                                   index_col='match_id_hash')\ndf_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), \n                                   index_col='match_id_hash')")


# ## Lets have a look what are these data look like:

# In[4]:


df_train_features.head(2)


# In[5]:


df_train_targets.head(2)


# In[6]:


df_test_features.head(2)


# I have no idea what these features mean...I prefer FIFA)

# In[7]:


#turn to X and y notations for train data and target
X = df_train_features.values
y = df_train_targets['radiant_win'].values #extract the colomn we need


# In[8]:


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


# In[9]:


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

# In[10]:


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
        
        #let's add one more
        df_features.loc[match_id_hash, 'ratio_tower_kills'] = radiant_tower_kills / (0.01+dire_tower_kills)
        # ... here you can add more features ...
        


# In[11]:


get_ipython().run_cell_magic('time', '', "# copy the dataframe with features\ndf_train_features_extended = df_train_features.copy()\ndf_test_features_extended = df_test_features.copy()\n\n# add new features\nadd_new_features(df_train_features_extended, os.path.join(PATH_TO_DATA, 'train_matches.jsonl'))\nadd_new_features(df_test_features_extended, os.path.join(PATH_TO_DATA, 'test_matches.jsonl'))")


# In[12]:


#Just a shorter names for data
newtrain=df_train_features_extended
newtest=df_test_features_extended
target=pd.DataFrame(y)


# In[13]:


#lastly, check the shapes, Andrew Ng approved)
newtrain.shape,target.shape, newtest.shape


# After running the LightGBM model, we will visualize something called "feature importance", which  kind of shows which features and how much they affected the final result. For this reason we need to store feature names:

# In[14]:


features=newtrain.columns


# ## Noow, let's define LightGBM parameters. 
# 
# Personally, I understand only some of these parameters. So, these are some random set up. Maybe it is better to look up the official documentation. Tuning these parameters definitely will increase your score.
# 
# Investigation in process...

# In[15]:


param = {
        'bagging_freq': 5,  #handling overfitting
        'bagging_fraction': 0.5,  #handling overfitting - adding some noise
        'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': 0.05, #handling overfitting
        'learning_rate': 0.01,  #the changes between one auc and a better one gets really small thus a small learning rate performs better
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

# In[16]:


get_ipython().run_cell_magic('time', '', '#divide training data into train and validaton folds\nfolds = StratifiedKFold(n_splits=5, shuffle=False, random_state=17)\n\n#placeholder for out-of-fold, i.e. validation scores\noof = np.zeros(len(newtrain))\n\n#for predictions\npredictions = np.zeros(len(newtest))\n\n#and for feature importance\nfeature_importance_df = pd.DataFrame()\n\n#RUN THE LOOP OVER FOLDS\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(newtrain.values, target.values)):\n    \n    X_train, y_train = newtrain.iloc[trn_idx], target.iloc[trn_idx]\n    X_valid, y_valid = newtrain.iloc[val_idx], target.iloc[val_idx]\n    \n    print("Computing Fold {}".format(fold_))\n    trn_data = lgb.Dataset(X_train, label=y_train)\n    val_data = lgb.Dataset(X_valid, label=y_valid)\n\n    \n    num_round = 5000 \n    verbose=1000 \n    stop=500 \n    \n    #TRAIN THE MODEL\n    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=verbose, early_stopping_rounds = stop)\n    \n    #CALCULATE PREDICTION FOR VALIDATION SET\n    oof[val_idx] = clf.predict(newtrain.iloc[val_idx], num_iteration=clf.best_iteration)\n    \n    #FEATURE IMPORTANCE\n    fold_importance_df = pd.DataFrame()\n    fold_importance_df["Feature"] = features\n    fold_importance_df["importance"] = clf.feature_importance()\n    fold_importance_df["fold"] = fold_ + 1\n    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)\n    \n    #CALCULATE PREDICTIONS FOR TEST DATA, using best_iteration on the fold\n    predictions += clf.predict(newtest, num_iteration=clf.best_iteration) / folds.n_splits\n\n#print overall cross-validatino score\nprint("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))')


# ## Feature Importance

# In[17]:


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


# From these feature importance chart, we can see that some features play significant role in making the prediction than others. Maybe dropping out less affecting features is a good idea. But still need more investigation of dota2 features...

# ## Prepare submission file

# In[18]:


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

# https://www.kaggle.com/sz8416/simple-bayesian-optimization-for-lightgbm

# In[21]:


from bayes_opt import BayesianOptimization


# In[30]:


def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
    params = {'application':'binary','num_iterations':4000, 'learning_rate':0.05, 'early_stopping_round':100, 'metric':'auc'}
    params["num_leaves"] = round(num_leaves)
    params['feature_fraction'] = max(min(feature_fraction, 1), 0)
    params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
    params['max_depth'] = round(max_depth)
    params['lambda_l1'] = max(lambda_l1, 0)
    params['lambda_l2'] = max(lambda_l2, 0)
    params['min_split_gain'] = min_split_gain
    params['min_child_weight'] = min_child_weight
    cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
    return max(cv_result['auc-mean'])


# In[31]:


lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                        'feature_fraction': (0.1, 0.9),
                                        'bagging_fraction': (0.8, 1),
                                        'max_depth': (5, 8.99),
                                        'lambda_l1': (0, 5),
                                        'lambda_l2': (0, 3),
                                        'min_split_gain': (0.001, 0.1),
                                        'min_child_weight': (5, 50)}, random_state=0)


# In[37]:


X = newtrain.values
y = target.values.ravel()


# In[ ]:


def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.05, output_process=False):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False) #categorical_feature = categorical_feats
    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {'application':'binary', 'metric':'auc'} #'num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
        return max(cv_result['auc-mean'])
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 8.99),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    # output optimization process
    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")
    
    # return best parameters
    return lgbBO

opt_params = bayes_parameter_opt_lgb(X, y, init_round=5, opt_round=10, n_folds=3, random_seed=6, n_estimators=100, learning_rate=0.05)


# In[ ]:


#print(opt_params.res['max']['max_params'])


# In[ ]:


print("Final result:", opt_params.max)


# In[ ]:




