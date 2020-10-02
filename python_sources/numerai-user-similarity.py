#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
pd.options.display.max_columns = 99
pd.options.display.max_rows = 100
import datetime as dt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import datetime as dt
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go


# In[ ]:


IGRAPH = True
if IGRAPH:
    from igraph import Graph


# In[ ]:


plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams['font.size'] = 16
start = dt.datetime.now()


# In[ ]:


T = 126
FIRST_STAKE_ROUND = 61
ROUNDS = range(FIRST_STAKE_ROUND, T + 1)
STAKING_ROUNDS = range(FIRST_STAKE_ROUND, T + 1)


# # Read Numerai Leaderboards

# In[ ]:


lb_with_stakes = pd.read_csv('../input/T{}_leaderboards.csv'.format(T))
lb_with_stakes.shape
lb_with_stakes = lb_with_stakes[lb_with_stakes['tournament_name'] == 'bernie']
lb_with_stakes = lb_with_stakes[lb_with_stakes['number'] <= T]
lb_with_stakes['stake.insertedAt'] = pd.to_datetime(lb_with_stakes['stake.insertedAt'], errors='coerce', utc=True)
lb_with_stakes['overfitting'] = lb_with_stakes['liveLogloss'] - lb_with_stakes['validationLogloss']

lb_with_stakes.head()
lb_with_stakes.shape


# # Join User Submission Info
# 

# In[ ]:


submission_info = pd.read_csv('../input/T{}_user_submission_info.csv'.format(T))
submission_info = submission_info[submission_info['tournament_id'] == 1].copy()
submission_info['submission.date'] = pd.to_datetime(submission_info['submission.date'], errors='coerce', utc=True)
submission_info = submission_info[['roundNumber', 'submission.date', 'username']]
submission_info.shape
submission_info.head()
submission_info.dtypes


# In[ ]:


lb_with_stakes = pd.merge(lb_with_stakes, submission_info, how='left',
                          left_on=['number', 'username'],
                          right_on=['roundNumber', 'username'])
lb_with_stakes.head()
lb_with_stakes.shape


# # User Base

# In[ ]:


users_staking = lb_with_stakes.groupby('username').count()[['stake.value']]
users_staking = users_staking[users_staking['stake.value'] >= 0]

users_with_10_rounds = lb_with_stakes.groupby('username').count()[['number']]
users_with_10_rounds = users_with_10_rounds[users_with_10_rounds.number >= 5]

users_after_100_rounds = lb_with_stakes.groupby('username').max()[['number']]
users_after_100_rounds = users_after_100_rounds[users_after_100_rounds.number >= 100]

users_with_resolved_rounds = lb_with_stakes.groupby('username').min()[['number']]
users_with_resolved_rounds = users_with_resolved_rounds[users_with_resolved_rounds.number <= 120]

users = pd.concat([
    users_staking, users_with_10_rounds, users_after_100_rounds, users_with_resolved_rounds
], axis=1, sort=True).dropna().index
len(users)


# # Aggregation

# In[ ]:


user_cnts = lb_with_stakes.groupby('username').count()[['number', 'stake.value']]
user_cnts.columns = ['rounds', 'rounds_staked']
user_median = lb_with_stakes.groupby('username').median()[['liveLogloss', 'validationLogloss', 'overfitting', 'stake.value', 'stake.confidence']]
user_mean = lb_with_stakes.groupby('username').mean()[['liveLogloss', 'validationLogloss', 'overfitting', 'better_than_random']]
user_max = lb_with_stakes.groupby('username').max()[['liveLogloss', 'validationLogloss', 'overfitting', 'stake.value', 'stake.confidence']]
user_min = lb_with_stakes.groupby('username').min()[['liveLogloss', 'validationLogloss', 'overfitting', 'number']]
user_stake_min = lb_with_stakes[['username', 'number', 'stake.value']].dropna()
user_stake_min = user_stake_min.groupby('username').min()[['number']]
user_std = lb_with_stakes.groupby('username').std()[['liveLogloss', 'validationLogloss', 'overfitting']]
user_median.columns = ['live_median', 'validation_median', 'overfitting_median', 'median_stake_value', 'median_stake_conf']
user_mean.columns = ['live_mean', 'validation_mean', 'overfitting_mean', 'better_than_random_mean']
user_max.columns = ['live_max', 'validation_max', 'overfitting_max', 'max_stake_value', 'max_stake_conf']
user_min.columns = ['live_min', 'validation_min', 'overfitting_min', 'round_min']
user_stake_min.columns = ['round_stake_min']
user_std.columns = ['live_std', 'validation_std', 'overfitting_std']
user_stats = pd.concat([user_cnts, user_median, user_mean, user_max, user_min, user_stake_min, user_std], axis=1, sort=True)


# In[ ]:


vll = lb_with_stakes.pivot('username', 'number', 'validationLogloss')
vll.columns = ['vLL{}'.format(r) for r in ROUNDS]
lll = lb_with_stakes.pivot('username', 'number', 'liveLogloss')
lll.columns = ['lLL{}'.format(r) for r in ROUNDS]
sc = lb_with_stakes[lb_with_stakes.number >= FIRST_STAKE_ROUND].pivot('username', 'number', 'stake.confidence')
sc.columns = ['sc{}'.format(r) for r in STAKING_ROUNDS]
sv = lb_with_stakes[lb_with_stakes.number >= FIRST_STAKE_ROUND].pivot('username', 'number', 'stake.value')
sv.columns = ['sv{}'.format(r) for r in STAKING_ROUNDS]
st = lb_with_stakes[lb_with_stakes.number >= FIRST_STAKE_ROUND].pivot('username', 'number', 'stake.insertedAt')
st.columns = ['st{}'.format(r) for r in STAKING_ROUNDS]
submt = lb_with_stakes[lb_with_stakes.number >= FIRST_STAKE_ROUND].pivot('username', 'number', 'submission.date')
submt.columns = ['submt{}'.format(r) for r in ROUNDS]
user_pivot = pd.concat([vll, lll, sc, sv, st, submt], axis=1, sort=True)


# In[ ]:


user_features = pd.concat([user_stats, user_pivot], axis=1)
user_features.shape
def time_since_first_stake(col):
    return - (col.dropna() - col.dropna().max()) / np.timedelta64(10**9, 'ns') / 3600.

for r in STAKING_ROUNDS[::-1]:
    user_features['dt{}'.format(r)] = time_since_first_stake(user_features['st{}'.format(r)])
    user_features['dt2{}'.format(r)] = time_since_first_stake(user_features['submt{}'.format(r)])

staking_user_features = user_features.loc[users].sort_values(by='rounds').reset_index()
staking_user_features.head()

staking_user_features['dt_min'] = staking_user_features[['dt{}'.format(r) for r in STAKING_ROUNDS[::-1]]].min(axis=1)
staking_user_features['dt_mean'] = staking_user_features[['dt{}'.format(r) for r in STAKING_ROUNDS[::-1]]].mean(axis=1)
staking_user_features['dt_max'] = staking_user_features[['dt{}'.format(r) for r in STAKING_ROUNDS[::-1]]].max(axis=1)

staking_user_features['dt2_min'] = staking_user_features[['dt2{}'.format(r) for r in STAKING_ROUNDS[::-1]]].min(axis=1)
staking_user_features['dt2_mean'] = staking_user_features[['dt2{}'.format(r) for r in STAKING_ROUNDS[::-1]]].mean(axis=1)
staking_user_features['dt2_max'] = staking_user_features[['dt2{}'.format(r) for r in STAKING_ROUNDS[::-1]]].max(axis=1)


clean_staking_user_features = staking_user_features[['index', 'better_than_random_mean',
                                                     'rounds', 'rounds_staked', 'round_min', 'round_stake_min',
                                                     'live_median', 'validation_median', 'overfitting_median',
                                                     'median_stake_value', 'median_stake_conf']].copy()
clean_staking_user_features['median_stake_conf'] = clean_staking_user_features['median_stake_conf'].clip(0, 1.1)
clean_staking_user_features = clean_staking_user_features.set_index('index')

clean_staking_user_features = clean_staking_user_features.fillna({
    'round_stake_min': T + 10, 
    'median_stake_value': 0,
    'median_stake_conf': -0.1})
clean_staking_user_features = clean_staking_user_features.fillna(clean_staking_user_features.mean())
user_feature_ranks = clean_staking_user_features.rank() / len(clean_staking_user_features)

user_feature_standardized = clean_staking_user_features.copy()
X = StandardScaler().fit_transform(user_feature_standardized.values)
for i, col in enumerate(user_feature_standardized.columns):
    user_feature_standardized[col] = X[:, i]


# In[ ]:


user_feature_ranks.head()
user_feature_standardized.head()


# In[ ]:


user_feature_standardized.loc['bps'].values


# # Complex features
# 
# * **l1_std_diff**: Aggregated standardized mean feature difference
# *  **l1_rank_diff**: Aggregated rank normalized mean feature difference
# * **round_chi2**: Chi2 Independence test p-value for round submission
# * **staking_round_chi2**: Chi2 Independence test p-value for round staking
# * **corr_cons**: Consistency Correlation
# * **corr_vll**: Validation Logloss correlation
# * **corr_of**: Overfitting correlation
# * **median_stake_time_diff**: Staking time difference
# * **median_submission_time_diff**: Submission time difference

# In[ ]:


user_results = {}
for u, df in tqdm(lb_with_stakes.groupby('username')):
    df = df.set_index('number')
    user_results[u] = df


# In[ ]:


def user_similarity_features(u1, u2):
    row = []
    df1 = user_results[u1]
    df2 = user_results[u2]
    merged_results = pd.merge(df1, df2, left_index=True, right_index=True)
    merged_results['originality.value_x'] = 1 * merged_results['originality.value_x']
    merged_results['originality.value_y'] = 1 * merged_results['originality.value_y']
    max_start_round = max(min(df1.index), min(df2.index))
    row.append(T + 1 - max_start_round)
    
    # standardized user l1 difference
    s1 = user_feature_standardized.loc[u1].values
    s2 = user_feature_standardized.loc[u2].values
    l1_std = np.mean(np.abs(s1 - s2))
    row.append(l1_std)
    
    # rank user l1 difference    
    r1 = user_feature_ranks.loc[u1].values
    r2 = user_feature_ranks.loc[u2].values
    l1_rank = np.mean(np.abs(r1 - r2))
    row.append(l1_rank)
    
    # round match rate
    possible = set(range(max(df1.index[0], df2.index[0]), T + 1))
    i1 = set(df1.index)
    i2 = set(df2.index)
    row.append((len(i1 & i2) + len((possible - i1) & (possible - i2))) / len(possible))

    # staking match rate
    s1 = set(df1[['stake.value']].dropna().index)
    s2 = set(df2[['stake.value']].dropna().index)
    row.append((len(s1 & s2) + len((possible - s1) & (possible - s2))) / len(possible))
    
    # independence test for participation
    observed = np.array([[len((possible - s1) & (possible - s2)), len((possible - s1) & s2)]
                         ,[len(s1 & (possible - s2)), len(s1 & s2)]]) + 1
    chi2, p, dof, ex = chi2_contingency(observed=observed)
    row.append(p)

    observed = np.array([[len((possible - i1) & (possible - i2)), len((possible - i1) & i2)]
                         ,[len(i1 & (possible - i2)), len(i1 & i2)]]) + 1
    chi2, p, dof, ex = chi2_contingency(observed=observed)
    row.append(p)
    
    if len(merged_results):
        # correlations
        for col  in ['consistency', 'validationLogloss', 'liveLogloss', 'overfitting']:
            row.append(merged_results[[col+'_x', col+'_y']].corr().values[0, 1])

        # median stake time diff
        stake_times = merged_results[['stake.insertedAt_x', 'stake.insertedAt_y']].dropna()
        stake_time_diff = (stake_times['stake.insertedAt_x'] - stake_times['stake.insertedAt_y'])
        stake_time_diff = np.abs(stake_time_diff / np.timedelta64(10**9, 'ns') / 3600.)
        row.append(np.clip(stake_time_diff.median(), None, 24))
        
        # median submission time diff
        submission_times = merged_results[['submission.date_x', 'submission.date_y']].dropna()
        submission_time_diff = (submission_times['submission.date_x'] - submission_times['submission.date_y'])
        submission_time_diff = np.abs(submission_time_diff / np.timedelta64(10**9, 'ns') / 3600.)
        row.append(np.clip(submission_time_diff.median(), None, 24))
    else:
        row += [np.nan] * 5
    names = ['round_count', 'l1_std_diff', 'l1_rank_diff', 'round_match', 'staking_round_match',
             'round_chi2', 'staking_round_chi2',
             'corr_cons', 'corr_vll', 'corr_lll', 'corr_of', 'median_stake_time_diff', 'median_submission_time_diff']
    return row, names


# In[ ]:


u1 = 'bps'
u2 = 'bps2'
row, names = user_similarity_features(u1, u2)
pd.DataFrame({'col': names, 'value': row})


# # Annotated user pairs

# In[ ]:


annotated_user_pairs = [
    ['_hal9000', '_hal9001', 'hal__9000'],
    ['acai_forest', 'acai_smoothie', 'acai_sorbet'],
    ['akumei', 'akumei2', 'akumei3'],
    ['alisa', 'beatriz', np.nan],
    ['anna3', 'anna2', 'anna1'],
    ['archangel', 'archangel2', 'archangel3'],
    ['blockchainwolf', 'blockchainwolf2', 'blockchainwolf3'],
    ['bor1', 'bor2', 'bor3'],
    ['bps', 'bps2', 'bps3'],
    ['brain2', 'brain1', np.nan],
    ['chrissly2', 'chrissly31415', 'chrissly3'],
    ['daenris', 'daenris1', 'daenris2'],
    ['dataman', 'dataman_ai', 'dataman_bj'],
    ['ddocw', 'birdy', np.nan],
    ['diacetylmorphine2', 'diacetylmorphine', 'diacetylmorphine3'],
    ['dnum1_2', 'dnum1_1', np.nan],
    ['drvby47', 'drvby21', np.nan],
    ['endure3', 'endure2', 'endure'],
    ['expdes2', 'expdes', 'expdes3'],
    ['giras', 'giras2', 'giras3'],
    ['hb', 'hb_exp', 'hb_scout'],
    ['hodl2', 'hodl1', np.nan],
    ['jakobr', 'jakobator', np.nan],
    ['javibear1', 'javibear', np.nan],
    ['luz2', 'luz', np.nan],
    ['mlt', 'mltrader', np.nan],
    ['mmfine', 'mmfine1', 'washington'],
    ['no_formal_training', 'no_formal_agreement', np.nan],
    ['nolunchamhungry3', 'nolunchamhungry1', np.nan],
    ['nosaai3', 'nosaai2', 'nosaai'],
    ['objectscience_3', 'objectscience_2', 'objectscience'],
    ['optimus2', 'optimus', np.nan],
    ['shimco_x', 'shimco', np.nan],
    ['smirmik3', 'smirmik2', 'smirmik'],
    ['sweatybear3', 'sweatybear', np.nan],
    ['themiconmanweb', 'themiconman', 'themicon'],
    ['tyler333', 'tyler33', 'tyler3'],
    ['uuazed3', 'uuazed2', 'uuazed'],
    ['wiingy', 'wiingy3', np.nan],
    ['wpe', 'wwpe', np.nan],
    ['accountnumber1', 'accountnumber2', 'accountnumber3'],
    ['acendai', 'acendai2', np.nan],
    ['adsp_1', 'adsp_2', np.nan],
    ['baby_one', 'baby_three', 'baby_two'],
    ['bayesfactor', 'bayesfactor2', 'bayesfactor3'],
    ['bukosabino', 'bukosabino2', np.nan],
    ['christophoroa', 'christophorob', 'christophoroc'],
    ['dick', 'dick2', 'dick3'],
    ['dlkl', 'dlkl0', 'dlkl1'],
    ['dustoff', 'dustoff2', 'dustoff3'],
    ['emerita', 'emerita_c', 'emerita_ca'],
    ['epattaro2', 'epattaro3', np.nan],
    ['h3ll0m4rk3t', 'h3ll0mr_r0b0t', 'h3ll0w0rld'],
    ['hephyrion', 'hephyrion_ii', np.nan],
    ['hoogleraar', 'hoogleraarkz', 'hoogleraarnl'],
    ['jfb', 'jfb2', 'jfb3'],
    ['kev1', 'kev2', np.nan],
    ['kristofer', 'kristofer2', np.nan],
    ['lsynb', 'lsyznb', np.nan],
    ['misfire', 'misfire1', 'misfire2'],
    ['nb_one', 'nb_three', 'nb_two'],
    ['ngs5st', 'ngs5st1', 'ngs5st2'],
    ['palmseed', 'palmtree', np.nan],
    ['prediction2nmr', 'prediction2nmr2', 'prediction2nmr3'],
    ['proudofme', 'proudofyou', np.nan],
    ['roadsandwines_blue', 'roadsandwines_green', 'roadsandwines_red'],
    ['rue', 'rue2', 'rue4'],
    ['sal___9000', 'sal__9000', np.nan],
    ['selvin', 'selvin2', np.nan],
    ['shgshgdkhml', 'shgshgdkhml2', np.nan],
    ['silly_bayes', 'silly_bayes_2', np.nan],
    ['sjn', 'sjn2', np.nan],
    ['ssh', 'ssh2', 'ssh3'],
    ['starlord', 'starlord2', np.nan],
    ['steve', 'steve1', 'steve2'],
    ['stpatrick1', 'stpatrick2', 'stpatrick3'],
    ['talos', 'talosblack', 'taloswhite'],
    ['teddy1', 'teddy2', 'teddy3'],
    ['true2', 'true3', np.nan],
    ['wiseai', 'wiserai', np.nan],
    ['wu_tang', 'wu_tang2', 'wu_tang3'],
    ['wander', 'transit', 'traveler'],
    ['yudanshai', 'yudanshaii', 'yudanshaiii']
]
user_pairs = pd.DataFrame(annotated_user_pairs, columns=['a', 'b', 'c'])
user_pairs['index'] = range(len(user_pairs))


# In[ ]:


a = user_pairs[['index', 'a']]
a.columns = ['user_id', 'user_name']
b = user_pairs[['index', 'b']]
b.columns = ['user_id', 'user_name']
c = user_pairs[['index', 'c']].dropna()
c.columns = ['user_id', 'user_name']
abc = pd.concat([a, b, c])
abc['one'] = 1

annotated_user_pairs = abc.merge(abc, on='one')
annotated_user_pairs = annotated_user_pairs[annotated_user_pairs.user_id_x <= annotated_user_pairs.user_id_y]
annotated_user_pairs = annotated_user_pairs[annotated_user_pairs.user_name_x != annotated_user_pairs.user_name_y]
annotated_user_pairs = annotated_user_pairs.drop('one', axis=1)
annotated_user_pairs['same_user'] = 1 * (annotated_user_pairs.user_id_x == annotated_user_pairs.user_id_y)
annotated_user_pairs.index = np.arange(len(annotated_user_pairs))
annotated_user_pairs.shape
annotated_user_pairs.head()
annotated_user_pairs.tail()
annotated_user_pairs.mean()
annotated_user_pairs.count()


# In[ ]:


rows = []
for u1, u2 in tqdm(annotated_user_pairs[['user_name_x', 'user_name_y']].values):
    row, feature_names = user_similarity_features(u1, u2)
    rows.append(row)
features = pd.DataFrame(rows, columns=feature_names)
features = features.round(3)


# In[ ]:


annotated_user_pairs_with_features = pd.concat([annotated_user_pairs, features], axis=1)


# In[ ]:


annotated_user_pairs_with_features.shape
annotated_user_pairs_with_features.mean()
annotated_user_pairs_with_features.count()
annotated_user_pairs_with_features.describe()


# In[ ]:


# Impute missing values
annotated_user_pairs_with_features = annotated_user_pairs_with_features.fillna(annotated_user_pairs_with_features.mean())


# # Train Model for User Similarity

# In[ ]:


selected_feature_names = ['l1_std_diff', 'l1_rank_diff', 'round_chi2', 'staking_round_chi2',
                          'corr_cons', 'corr_vll', 'corr_of', 
                          'median_stake_time_diff', 'median_submission_time_diff']


# In[ ]:


oof = []
for uid in tqdm(user_pairs.index):
    train = annotated_user_pairs_with_features[np.logical_and(annotated_user_pairs_with_features['user_id_x'] != uid,
                                                              annotated_user_pairs_with_features['user_id_y'] != uid)].copy()
    valid = annotated_user_pairs_with_features[np.logical_or(annotated_user_pairs_with_features['user_id_x'] == uid,
                                                             annotated_user_pairs_with_features['user_id_y'] == uid)].copy()
    scaler = StandardScaler()
    Xtr = train[selected_feature_names].values
    Xtr = scaler.fit_transform(Xtr)
    ytr = train.same_user.values
    Xv = scaler.transform(valid[selected_feature_names].values)
    yv = valid.same_user.values
    logreg = LogisticRegression()
    _ = logreg.fit(Xtr, ytr)
    valid['prediction'] = logreg.predict_proba(Xv)[:, 1]
    oof.append(valid)
predicted = pd.concat(oof)

fpr, tpr, thr = metrics.roc_curve(predicted.same_user.values, predicted.prediction.values)
auc = metrics.auc(fpr, tpr)


# In[ ]:


fig, ax = plt.subplots()
plt.plot(fpr, tpr, lw=5, label='ROC')
plt.plot(fpr, thr, lw=5, label='threshold')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc=0)
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('LogReg trained with {} user pairs (AUC ~ {:.3f})'.format(annotated_user_pairs_with_features.shape[0], auc))
plt.show();


# In[ ]:


fig, ax = plt.subplots()
plt.plot(fpr, tpr, lw=5, label='ROC')
plt.plot(fpr, thr, lw=5, label='threshold')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc=0)
plt.grid()
plt.xlim(0, 0.05)
plt.ylim(0, 1)
plt.title('Using 0.5 threshold would give {:.3f} FPR'.format(np.max(fpr[thr > 0.5])))
plt.show();


# # User pair candidates

# In[ ]:


user_candidate_file_name = '../input/T{}_user_candidate_features.csv'.format(T)
RECALCULATE = False
if RECALCULATE  or (not os.path.exists(user_candidate_file_name)):
    rows = []
    for i, u1 in tqdm(enumerate(users)):
        for u2 in users[i+1:]:
            row, feature_names = user_similarity_features(u1, u2)
            rows.append([u1, u2] + row)
    user_candidates = pd.DataFrame(rows, columns=['user_name_x', 'user_name_y'] + feature_names)
    user_candidates = user_candidates.round(3)
    user_candidates.to_csv(user_candidate_file_name, index=False)
else:
    user_candidates = pd.read_csv(user_candidate_file_name)


# In[ ]:


user_candidates.head()
user_candidates.shape
user_candidates.describe()


# In[ ]:


user_candidates = user_candidates.fillna(annotated_user_pairs_with_features.mean())

user_candidates_features = scaler.transform(user_candidates[selected_feature_names].values)
user_candidates['prediction'] = logreg.predict_proba(user_candidates_features)[:, 1]

user_candidates = user_candidates.sort_values(by='prediction', ascending=False)
try:
    user_candidates.to_csv(user_candidate_file_name, index=False)
except Exception as e:
    pass
user_candidates.head()


# # User similarity graph

# In[ ]:


usernames = list(set(user_candidates.user_name_x.unique()) | set(user_candidates.user_name_y.unique()))
user_ids = pd.DataFrame({'user_name': usernames, 'user_id': range(len(usernames))})
user_ids = user_ids.set_index('user_name')
user_candidates = user_candidates.merge(user_ids, left_on='user_name_x', right_index=True)
user_candidates = user_candidates.merge(user_ids, left_on='user_name_y', right_index=True)


# In[ ]:


possible_matches = user_candidates[user_candidates['prediction'] > 0.7].copy()
try:
    possible_matches.to_csv('../input/possible_matches.csv', index=False)
except Exception as e:
    possible_matches.to_csv('possible_matches.csv', index=False)
possible_matches.shape


# In[ ]:


user_coords_file_name = '../input/T{}_user_coords_features.csv'.format(T)
if IGRAPH:
    g = Graph()
    g.add_vertices(len(user_ids))
    g.add_edges(possible_matches[['user_id_x', 'user_id_y']].values)
    g.vs['username'] = user_ids.index
    g.es['weight'] = possible_matches['prediction'].values
    g.summary()

    clusters = g.clusters()
    for sg in clusters.subgraphs():
        if sg.vcount() > 3:
            sg.summary()

    graph_layout = g.layout('fr', weights='weight')
    coords = np.array(graph_layout.coords)
    user_coords = pd.DataFrame(coords, columns=['x', 'y'])
    user_coords.index = user_ids.index
else:
    user_coords = pd.read_csv(user_coords_file_name)
    user_coords = user_coords.set_index('user_name')


# In[ ]:


if IGRAPH:
    degree = pd.DataFrame({'user': g.vs['username'], 'degree': g.vs.degree()})
    degree.sort_values(by='degree', ascending=False)[:30]
    degree[degree.degree > 13].user.values


# In[ ]:


plt.style.use("dark_background")
fig, ax = plt.subplots()
plt.plot(user_coords.x, user_coords.y, 'wo')
for u1, u2 in possible_matches[['user_name_x', 'user_name_y']].values:
    plt.plot([user_coords.loc[u1, 'x'], user_coords.loc[u2, 'x']],
             [user_coords.loc[u1, 'y'], user_coords.loc[u2, 'y']], 'w', alpha=0.5)
plt.xticks([])
plt.yticks([])
fig.savefig('users.png', dpi=300)
plt.show();


# In[ ]:


data = []
for u1, u2 in possible_matches[['user_name_x', 'user_name_y']].values:
    trace = go.Scatter(
        x = [user_coords.loc[u1, 'x'], user_coords.loc[u2, 'x']],
        y = [user_coords.loc[u1, 'y'], user_coords.loc[u2, 'y']],
        mode = 'lines',
        line=dict(color='grey', width=1))
    data.append(trace)
data.append(
    go.Scatter(
        y = user_coords['y'],
        x = user_coords['x'],
        #     mode='markers',
        mode='markers+text',
        marker=dict(sizemode='diameter',sizeref=1, size=10, color='black'),
        text=user_coords.index,
        hoverinfo = 'text',
        textposition=["top center"],
    )
)
layout = go.Layout(
    autosize=True,
    title='User similarity',
    hovermode='closest',
    yaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
    xaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='user_similarity')


# In[ ]:


end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))


# In[ ]:




