#!/usr/bin/env python
# coding: utf-8

# # LGMB with random split for early stopping
# 
# **Edits by Eric Scuccimarra** - This is a fork of  https://www.kaggle.com/mlisovyi/feature-engineering-lighgbm-with-f1-macro, by Misha Losvyo, with a few changes:
#  - Some additional features have been added.
#  - Some features which were previously dropped have been retained.
#  - Some of the code has been reorganized.
#  - Rather than splitting the data once and using the validation data for the LGBM early stopping, I split the data during the training so the entire training set can be trained on. A K-Fold split would be better, I just wasn't sure how to pass that into the parallel processing.
#  
#  Some additional features were taken from: https://www.kaggle.com/kuriyaman1002/reduce-features-140-84-keeping-f1-score, by Kuriyaman.
#  
# **Notes from Original Kernel:** I removed some of the points that I changed.
# 
# This kernel closely follows https://www.kaggle.com/mlisovyi/lighgbm-hyperoptimisation-with-f1-macro, but instead of running hyperparameter optimisation it uses optimal values from that kernel and thus runs faster. 
# 
# Several key points:
# - **This kernel runs training on the heads of housholds only** (after extracting aggregates over households). This follows the announced scoring startegy: *Note that ONLY the heads of household are used in scoring. All household members are included in test + the sample submission, but only heads of households are scored.* (from the data description). However, at the moment it seems that evaluation depends also on non-head household members, see https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#360115. In practise, ful prediction gives ~0.4 PLB score, while replacing all non-head entries with class 1 leads to a drop down to ~0.2 PLB score
# - **It seems to be very important to balance class frequencies.** Without balancing a trained model gives ~0.39 PLB / ~0.43 local test, while adding balancing leads to ~0.42 PLB / 0.47 local test. One can do it by hand, one can achieve it by undersampling. But the simplest (and more powerful compared to undersampling) is to set `class_weight='balanced'` in the LightGBM model constructor in sklearn API.
# - **This kernel uses macro F1 score to early stopping in training**. This is done to align with the scoring strategy.
# - Categoricals are turned into numbers with proper mapping instead of blind label encoding. 
# - **OHE if reversed into label encoding, as it is easier to digest for a tree model.** This trick would be harmful for non-tree models, so be careful.
# - **idhogar is NOT used in training**. The only way it could have any info would be if there is a data leak. We are fighting with poverty here- exploiting leaks will not reduce poverty in any way :)
# - **There are aggregations done within households and new features are hand-crafted**. Note, that there are not so many features that can be aggregated, as most are already quoted on household level.
# - **A voting classifier is used to average over several LightGBM models**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier, RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")


# The following categorical mapping originates from [this kernel](https://www.kaggle.com/mlisovyi/categorical-variables-encoding-function).

# In[ ]:


from sklearn.preprocessing import LabelEncoder

# this only transforms the idhogar field, the other things this function used to do are done elsewhere
def encode_data(df):
    df['idhogar'] = LabelEncoder().fit_transform(df['idhogar'])

# plot feature importance for sklearn decision trees    
def feature_importance(forest, X_train):
    ranked_list = []
    
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]) + " - " + X_train.columns[indices[f]])
        ranked_list.append(X_train.columns[indices[f]])
    
    return ranked_list    


# **There is also feature engineering magic happening here:**

# In[ ]:


def do_features(df):
    feats_div = [('children_fraction', 'r4t1', 'r4t3'), 
                 ('working_man_fraction', 'r4h2', 'r4t3'),
                 ('all_man_fraction', 'r4h3', 'r4t3'),
                 ('human_density', 'tamviv', 'rooms'),
                 ('human_bed_density', 'tamviv', 'bedrooms'),
                 ('rent_per_person', 'v2a1', 'r4t3'),
                 ('rent_per_room', 'v2a1', 'rooms'),
                 ('mobile_density', 'qmobilephone', 'r4t3'),
                 ('tablet_density', 'v18q1', 'r4t3'),
                 ('mobile_adult_density', 'qmobilephone', 'r4t2'),
                 ('tablet_adult_density', 'v18q1', 'r4t2'),
                ]
    
    feats_sub = [('people_not_living', 'tamhog', 'tamviv'),
                 ('people_weird_stat', 'tamhog', 'r4t3')]

    for f_new, f1, f2 in feats_div:
        df['fe_' + f_new] = (df[f1] / df[f2]).astype(np.float32)       
    for f_new, f1, f2 in feats_sub:
        df['fe_' + f_new] = (df[f1] - df[f2]).astype(np.float32)
    
    # aggregation rules over household
    aggs_num = {'age': ['min', 'max', 'mean'],
                'escolari': ['min', 'max', 'mean']
               }
    
    aggs_cat = {'dis': ['mean']}
    for s_ in ['estadocivil', 'parentesco', 'instlevel']:
        for f_ in [f_ for f_ in df.columns if f_.startswith(s_)]:
            aggs_cat[f_] = ['mean', 'count']

    # aggregation over household
    for name_, df_ in [('18', df.query('age >= 18'))]:
        df_agg = df_.groupby('idhogar').agg({**aggs_num, **aggs_cat}).astype(np.float32)
        df_agg.columns = pd.Index(['agg' + name_ + '_' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
        df = df.join(df_agg, how='left', on='idhogar')
        del df_agg

    # Drop id's
    df.drop(['Id'], axis=1, inplace=True)
    
    return df


# In[ ]:


# convert one hot encoded fields to label encoding
def convert_OHE2LE(df):
    tmp_df = df.copy(deep=True)
    for s_ in ['pared', 'piso', 'techo', 'abastagua', 'sanitario', 'energcocinar', 'elimbasu', 
               'epared', 'etecho', 'eviv', 'estadocivil', 'parentesco', 
               'instlevel', 'lugar', 'tipovivi',
               'manual_elec']:
        if 'manual_' not in s_:
            cols_s_ = [f_ for f_ in df.columns if f_.startswith(s_)]
        elif 'elec' in s_:
            cols_s_ = ['public', 'planpri', 'noelec', 'coopele']
        sum_ohe = tmp_df[cols_s_].sum(axis=1).unique()
        #deal with those OHE, where there is a sum over columns == 0
        if 0 in sum_ohe:
            print('The OHE in {} is incomplete. A new column will be added before label encoding'
                  .format(s_))
            # dummy colmn name to be added
            col_dummy = s_+'_dummy'
            # add the column to the dataframe
            tmp_df[col_dummy] = (tmp_df[cols_s_].sum(axis=1) == 0).astype(np.int8)
            # add the name to the list of columns to be label-encoded
            cols_s_.append(col_dummy)
            # proof-check, that now the category is complete
            sum_ohe = tmp_df[cols_s_].sum(axis=1).unique()
            if 0 in sum_ohe:
                 print("The category completion did not work")
        tmp_cat = tmp_df[cols_s_].idxmax(axis=1)
        tmp_df[s_ + '_LE'] = LabelEncoder().fit_transform(tmp_cat).astype(np.int16)
        if 'parentesco1' in cols_s_:
            cols_s_.remove('parentesco1')
        tmp_df.drop(cols_s_, axis=1, inplace=True)
    return tmp_df


# # Read in the data and clean it up

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

test_ids = test.Id


# In[ ]:


def process_df(df_):
    # encode the idhogar
    encode_data(df_)
    
    # create aggregate features
    return do_features(df_)

train = process_df(train)
test = process_df(test)


# Clean up some missing data and convert objects to numeric.

# In[ ]:


# some dependencies are Na, fill those with the square root of the square
train['dependency'] = np.sqrt(train['SQBdependency'])
test['dependency'] = np.sqrt(test['SQBdependency'])

# fill "no"s for education with 0s
train.loc[train['edjefa'] == "no", "edjefa"] = 0
train.loc[train['edjefe'] == "no", "edjefe"] = 0
test.loc[test['edjefa'] == "no", "edjefa"] = 0
test.loc[test['edjefe'] == "no", "edjefe"] = 0

# if education is "yes" and person is head of household, fill with escolari
train.loc[(train['edjefa'] == "yes") & (train['parentesco1'] == 1), "edjefa"] = train.loc[(train['edjefa'] == "yes") & (train['parentesco1'] == 1), "escolari"]
train.loc[(train['edjefe'] == "yes") & (train['parentesco1'] == 1), "edjefe"] = train.loc[(train['edjefe'] == "yes") & (train['parentesco1'] == 1), "escolari"]

test.loc[(test['edjefa'] == "yes") & (test['parentesco1'] == 1), "edjefa"] = test.loc[(test['edjefa'] == "yes") & (test['parentesco1'] == 1), "escolari"]
test.loc[(test['edjefe'] == "yes") & (test['parentesco1'] == 1), "edjefe"] = test.loc[(test['edjefe'] == "yes") & (test['parentesco1'] == 1), "escolari"]

train.loc[train['edjefa'] == "yes", "edjefa"] = 4
train.loc[train['edjefe'] == "yes", "edjefe"] = 4

test.loc[test['edjefa'] == "yes", "edjefa"] = 4
test.loc[test['edjefe'] == "yes", "edjefe"] = 4

train['edjefe'] = train['edjefe'].astype("int")
train['edjefa'] = train['edjefa'].astype("int")
test['edjefe'] = test['edjefe'].astype("int")
test['edjefa'] = test['edjefa'].astype("int")


# create feature with max education of either head of household
train['edjef'] = np.max(train[['edjefa','edjefe']], axis=1)
test['edjef'] = np.max(test[['edjefa','edjefe']], axis=1)

# fill some nas
train['v2a1']=train['v2a1'].fillna(0)
test['v2a1']=test['v2a1'].fillna(0)

test['v18q1']=test['v18q1'].fillna(0)
train['v18q1']=train['v18q1'].fillna(0)

train['rez_esc']=train['rez_esc'].fillna(0)
test['rez_esc']=test['rez_esc'].fillna(0)

train.loc[train.meaneduc.isnull(), "meaneduc"] = 0
train.loc[train.SQBmeaned.isnull(), "SQBmeaned"] = 0

test.loc[test.meaneduc.isnull(), "meaneduc"] = 0
test.loc[test.SQBmeaned.isnull(), "SQBmeaned"] = 0

# fix some inconsistencies in the data - some rows indicate both that the household does and does not have a toilet, 
# if there is no water we'll assume they do not
train.loc[(train.v14a ==  1) & (train.sanitario1 ==  1) & (train.abastaguano == 0), "v14a"] = 0
train.loc[(train.v14a ==  1) & (train.sanitario1 ==  1) & (train.abastaguano == 0), "sanitario1"] = 0

test.loc[(test.v14a ==  1) & (test.sanitario1 ==  1) & (test.abastaguano == 0), "v14a"] = 0
test.loc[(test.v14a ==  1) & (test.sanitario1 ==  1) & (test.abastaguano == 0), "sanitario1"] = 0


# In[ ]:


def train_test_apply_func(train_, test_, func_):
    test_['Target'] = 0
    xx = pd.concat([train_, test_])

    xx_func = func_(xx)
    train_ = xx_func.iloc[:train_.shape[0], :]
    test_  = xx_func.iloc[train_.shape[0]:, :].drop('Target', axis=1)

    del xx, xx_func
    return train_, test_


# In[ ]:


# convert the one hot fields into label encoded
train, test = train_test_apply_func(train, test, convert_OHE2LE)


# # Geo aggregates

# In[ ]:


cols_2_ohe = ['eviv_LE', 'etecho_LE', 'epared_LE', 'elimbasu_LE', 
              'energcocinar_LE', 'sanitario_LE', 'manual_elec_LE',
              'pared_LE']
cols_nums = ['age', 'meaneduc', 'dependency', 
             'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total',
             'bedrooms', 'overcrowding']

def convert_geo2aggs(df_):
    tmp_df = pd.concat([df_[(['lugar_LE', 'idhogar']+cols_nums)],
                        pd.get_dummies(df_[cols_2_ohe], 
                                       columns=cols_2_ohe)],axis=1)

    geo_agg = tmp_df.groupby(['lugar_LE','idhogar']).mean().groupby('lugar_LE').mean().astype(np.float32)
    geo_agg.columns = pd.Index(['geo_' + e for e in geo_agg.columns.tolist()])
    
    del tmp_df
    return df_.join(geo_agg, how='left', on='lugar_LE')

# add some aggregates by geography
train, test = train_test_apply_func(train, test, convert_geo2aggs)


# In[ ]:


# add the number of people over 18 in each household
train['num_over_18'] = 0
train['num_over_18'] = train[train.age >= 18].groupby('idhogar').transform("count")
train['num_over_18'] = train.groupby("idhogar")["num_over_18"].transform("max")
train['num_over_18'] = train['num_over_18'].fillna(0)

test['num_over_18'] = 0
test['num_over_18'] = test[test.age >= 18].groupby('idhogar').transform("count")
test['num_over_18'] = test.groupby("idhogar")["num_over_18"].transform("max")
test['num_over_18'] = test['num_over_18'].fillna(0)

# add some extra features, these were taken from another kernel
def extract_features(df):
    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
    df['rent_to_rooms'] = df['v2a1']/df['rooms']
    df['tamhog_to_rooms'] = df['tamhog']/df['rooms'] # tamhog - size of the household
    df['r4t3_to_tamhog'] = df['r4t3']/df['tamhog'] # r4t3 - Total persons in the household
    df['r4t3_to_rooms'] = df['r4t3']/df['rooms'] # r4t3 - Total persons in the household
    df['v2a1_to_r4t3'] = df['v2a1']/df['r4t3'] # rent to people in household
    df['v2a1_to_r4t3'] = df['v2a1']/(df['r4t3'] - df['r4t1']) # rent to people under age 12
    df['hhsize_to_rooms'] = df['hhsize']/df['rooms'] # rooms per person
    df['rent_to_hhsize'] = df['v2a1']/df['hhsize'] # rent to household size
    df['rent_to_over_18'] = df['v2a1']/df['num_over_18']
    # some households have no one over 18, use the total rent for those
    df.loc[df.num_over_18 == 0, "rent_to_over_18"] = df[df.num_over_18 == 0].v2a1
    df['dependency_yes'] = df['dependency'].apply(lambda x: 1 if x == 'yes' else 0)
    df['dependency_no'] = df['dependency'].apply(lambda x: 1 if x == 'no' else 0)
    
    
extract_features(train)    
extract_features(test)    


# In[ ]:


# drop duplicated columns
needless_cols = ['r4t3', 'tamhog', 'tamviv', 'hhsize', 'v18q', 'v14a', 'agesq',
                 'mobilephone', 'female', ]

instlevel_cols = [s for s in train.columns.tolist() if 'instlevel' in s]

needless_cols.extend(instlevel_cols)

train = train.drop(needless_cols, axis=1)
test = test.drop(needless_cols, axis=1)


# ## Split the data
# 
# We split the data by household to avoid leakage, since rows belonging to the same household usually have the same target. Since we filter the data to only include heads of household this isn't technically necessary, but it provides an easy way to use the entire training data set if we want to do that.
# 
# Note that after splitting the data we overwrite the train data with the entire data set so we can train on all of the data. The split_data function does the same thing without overwriting the data, and is used within the training loop to (hopefully) approximate a K-Fold split. 

# In[ ]:


def split_data(train, y, households, test_percentage=0.20, seed=None):
    np.random.seed(seed=seed)
    train2 = train.copy()
    
    # pick some random households to use for the test data
    cv_hhs = np.random.choice(households, size=int(len(households) * test_percentage), replace=False)
    
    # select households which are in the random selection
    cv_idx = np.isin(households, cv_hhs)
    X_test = train2[cv_idx]
    y_test = y[cv_idx]

    X_train = train2[~cv_idx]
    y_train = y[~cv_idx]
    
    return X_train, y_train, X_test, y_test


# In[ ]:


X = train.query('parentesco1==1')
# X = train.copy()

# pull out and drop the target variable
y = X['Target'] - 1
X = X.drop(['Target'], axis=1)

np.random.seed(seed=1977)

train2 = X.copy()

train_hhs = train2.idhogar

households = train2.idhogar.unique()
cv_hhs = np.random.choice(households, size=int(len(households) * 0.1), replace=False)

cv_idx = np.isin(train2.idhogar, cv_hhs)

X_test = train2[cv_idx]
y_test = y[cv_idx]

X_train = train2[~cv_idx]
y_train = y[~cv_idx]

# train on entire dataset
X_train = train2
y_train = y

train_households = X_train.idhogar


# In[ ]:


# drop some features which aren't used by the LGBM or have very low importance
extra_drop_features = ['geo_manual_elec_LE_2',
 'agg18_estadocivil5_COUNT',
 'agg18_estadocivil7_COUNT',
 'geo_pared_LE_5',
 'agg18_estadocivil6_COUNT',
 'geo_pared_LE_4',
 'agg18_parentesco10_COUNT',
 'num_over_18',
 'agg18_estadocivil1_MEAN',
 'agg18_estadocivil4_COUNT',
 'r4t3_to_tamhog',
 'agg18_estadocivil3_COUNT',
 'geo_pared_LE_3',
 'geo_pared_LE_6',
 'geo_pared_LE_0',
 'geo_pared_LE_7',
 'agg18_parentesco7_COUNT',
 'agg18_parentesco10_MEAN',
 'agg18_parentesco8_MEAN',
 'parentesco1',
 'rez_esc',
 'dependency_yes',
 'hacdor',
 'hacapo',
 'fe_people_weird_stat',
 'geo_hogar_mayor',
 'geo_eviv_LE_0',
 'geo_epared_LE_0',
 'geo_epared_LE_2',
 'geo_elimbasu_LE_4',
 'geo_energcocinar_LE_0',
 'agg18_parentesco9_COUNT',
 'agg18_parentesco8_COUNT',
 'agg18_parentesco11_COUNT',
 'parentesco_LE',
 'agg18_parentesco6_MEAN',
 'agg18_parentesco6_COUNT',
 'agg18_parentesco5_COUNT',
 'agg18_parentesco4_MEAN',
 'agg18_parentesco4_COUNT',
 'geo_energcocinar_LE_1',
 'agg18_parentesco3_COUNT',
 'geo_energcocinar_LE_2',
 'agg18_parentesco2_COUNT',
 'agg18_parentesco1_COUNT',
 'agg18_parentesco12_MEAN',
 'agg18_parentesco12_COUNT',
 'dependency_no'] + ['geo_hogar_nin', 'geo_manual_elec_LE_3']


# In[ ]:


lgb_drop_cols = extra_drop_features + ["idhogar"]
try:
    X_train.drop(lgb_drop_cols, axis=1, inplace=True)
    X_test.drop(lgb_drop_cols, axis=1, inplace=True)
except:
    print("Error dropping columns")


# # Fit a voting classifier
# Define a derived VotingClassifier class to be able to pass `fit_params` for early stopping. Vote based on LGBM models with early stopping based on macro F1 and decaying learning rate.
# 
# The parameters are optimised with a random search in this kernel: https://www.kaggle.com/mlisovyi/lighgbm-hyperoptimisation-with-f1-macro

# In[ ]:


# these parameters have not been altered from when they were originally tuned
# opt_parameters = {'colsample_bytree': 0.93, 'min_child_samples': 56, 'num_leaves': 19, 'subsample': 0.84}
opt_parameters = {'colsample_bytree': 0.88, 'min_child_samples': 90, 'num_leaves': 16, 'subsample': 0.94}

def evaluate_macroF1_lgb(truth, predictions):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', f1, True) 

fit_params={"early_stopping_rounds":300,
            "eval_metric" : evaluate_macroF1_lgb, 
            "eval_set" : [(X_train,y_train), (X_test,y_test)],
            'eval_names': ['train', 'valid'],
            'verbose': False,
            'categorical_feature': 'auto'}

def learning_rate_power_0997(current_iter):
    base_learning_rate = 0.1
    min_learning_rate = 0.02
    lr = base_learning_rate  * np.power(.995, current_iter)
    return max(lr, min_learning_rate)

fit_params['verbose'] = 200
fit_params['callbacks'] = [lgb.reset_parameter(learning_rate=learning_rate_power_0997)]


# In[ ]:


def _parallel_fit_estimator(estimator1, X, y, sample_weight=None, threshold=True, **fit_params):
    estimator = clone(estimator1)
    
    # randomly split the data so we have a test set for early stopping
    X_train, y_train, X_test, y_test = split_data(X, y, households=train_households, seed=None)
    
    # update the fit params with our new split
    fit_params["eval_set"] = [(X_train,y_train), (X_test,y_test)]
    
    # fit the estimator
    if sample_weight is not None:
        if isinstance(estimator1, ExtraTreesClassifier) or isinstance(estimator1, RandomForestClassifier):
            estimator.fit(X_train, y_train)
        else:
            estimator.fit(X_train, y_train, sample_weight=sample_weight, **fit_params)
    else:
        if isinstance(estimator1, ExtraTreesClassifier) or isinstance(estimator1, RandomForestClassifier):
            estimator.fit(X_train, y_train)
        else:
            estimator.fit(X_train, y_train, **fit_params)
    
    if not isinstance(estimator1, ExtraTreesClassifier) and not isinstance(estimator1, RandomForestClassifier):
        best_cv_round = np.argmax(estimator.evals_result_['valid']['macroF1'])
        best_cv = np.max(estimator.evals_result_['valid']['macroF1'])
        best_train = estimator.evals_result_['train']['macroF1'][best_cv_round]
    else:
        best_train = f1_score(y_train, estimator.predict(X_train), average="macro")
        best_cv = f1_score(y_test, estimator.predict(X_test), average="macro")
        print("Train F1:", best_train)
        print("Test F1:", best_cv)
        
    # reject some estimators based on their performance on train and test sets
    if threshold:
        # if the valid score is very high we'll allow a little more leeway with the train scores
        if ((best_cv > 0.39) and (best_train > 0.75)) or ((best_cv > 0.44) and (best_train > 0.65)):
            return estimator

        # else recurse until we get a better one
        else:
            print("Unacceptable!!! Trying again...")
            return _parallel_fit_estimator(estimator1, X, y, sample_weight=sample_weight, **fit_params)
    
    else:
        return estimator
    
class VotingClassifierLGBM(VotingClassifier):
    '''
    This implements the fit method of the VotingClassifier propagating fit_params
    '''
    def fit(self, X, y, sample_weight=None, threshold=True, **fit_params):
        
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')

        if (self.weights is not None and
                len(self.weights) != len(self.estimators)):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d estimators'
                             % (len(self.weights), len(self.estimators)))

        if sample_weight is not None:
            for name, step in self.estimators:
                if not has_fit_parameter(step, 'sample_weight'):
                    raise ValueError('Underlying estimator \'%s\' does not'
                                     ' support sample weights.' % name)
        names, clfs = zip(*self.estimators)
        self._validate_names(names)

        n_isnone = np.sum([clf is None for _, clf in self.estimators])
        if n_isnone == len(self.estimators):
            raise ValueError('All estimators are None. At least one is '
                             'required to be a classifier!')

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self.estimators_ = []

        transformed_y = self.le_.transform(y)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_fit_estimator)(clone(clf), X, transformed_y,
                                                 sample_weight=sample_weight, threshold=threshold, **fit_params)
                for clf in clfs if clf is not None)

        return self


# In[ ]:


clfs = []
for i in range(24):
    clf = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=314+i, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced')
    
    clf.set_params(**opt_parameters)
    clfs.append(('lgbm{}'.format(i), clf))
    
vc = VotingClassifierLGBM(clfs, voting='soft')
del clfs

#Train the final model with learning rate decay
_ = vc.fit(X_train, y_train, **fit_params)

clf_final = vc.estimators_[0]


# In[ ]:


global_score = f1_score(y_test, clf_final.predict(X_test), average='macro')
vc.voting = 'soft'
global_score_soft = f1_score(y_test, vc.predict(X_test), average='macro')
vc.voting = 'hard'
global_score_hard = f1_score(y_test, vc.predict(X_test), average='macro')

print('Validation score of a single LGBM Classifier: {:.4f}'.format(global_score))
print('Validation score of a VotingClassifier on 3 LGBMs with soft voting strategy: {:.4f}'.format(global_score_soft))
print('Validation score of a VotingClassifier on 3 LGBMs with hard voting strategy: {:.4f}'.format(global_score_hard))


# In[ ]:


global_score = f1_score(y_test, clf_final.predict(X_test), average='macro')
vc.voting = 'soft'
global_score_soft = f1_score(y_test, vc.predict(X_test), average='macro')
vc.voting = 'hard'
global_score_hard = f1_score(y_test, vc.predict(X_test), average='macro')

print('Validation score of a single LGBM Classifier: {:.4f}'.format(global_score))
print('Validation score of a VotingClassifier on 3 LGBMs with soft voting strategy: {:.4f}'.format(global_score_soft))
print('Validation score of a VotingClassifier on 3 LGBMs with hard voting strategy: {:.4f}'.format(global_score_hard))


# In[ ]:


ranked_features = feature_importance(clf_final, X_train)


# In[ ]:


extra_drop_features = ranked_features[116:]
extra_drop_features


# # Prepare submission

# In[ ]:


y_subm = pd.DataFrame()
y_subm['Id'] = test_ids


# In[ ]:


y_subm['Target'] = clf_final.predict(test[X_train.columns]) + 1

vc.voting = 'soft'
y_subm_soft = y_subm.copy(deep=True)
y_subm_soft['Target'] = vc.predict(test[X_train.columns]) + 1

vc.voting = 'hard'
y_subm_hard = y_subm.copy(deep=True)
y_subm_hard['Target'] = vc.predict(test[X_train.columns]) + 1


# In[ ]:


from datetime import datetime
now = datetime.now()

sub_file = 'submission_LGB_{:.4f}_{}.csv'.format(global_score, str(now.strftime('%Y-%m-%d-%H-%M')))
sub_file_soft = 'submission_soft_LGB_{:.4f}_{}.csv'.format(global_score_soft, str(now.strftime('%Y-%m-%d-%H-%M')))
sub_file_hard = 'submission_hard_LGB_{:.4f}_{}.csv'.format(global_score_hard, str(now.strftime('%Y-%m-%d-%H-%M')))
sub_file_0forNonHeads = 'submission_0forNonHead_LGB_{:.4f}_{}.csv'.format(global_score, str(now.strftime('%Y-%m-%d-%H-%M')))

y_subm.to_csv(sub_file, index=False)
y_subm_soft.to_csv(sub_file_soft, index=False)
y_subm_hard.to_csv(sub_file_hard, index=False)


# In[ ]:




