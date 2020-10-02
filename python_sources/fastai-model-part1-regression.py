#!/usr/bin/env python
# coding: utf-8

# ### TabularLearner with part 1
# 
# In this notebook data generated from this [kernel](https://www.kaggle.com/keremt/fastai-feature-engineering-part1-6160-feats/) is used during modeling. Data can also be found as a Kaggle [dataset](https://www.kaggle.com/keremt/dsbowl2019-feng-part1). This notebook is part 1 of series of notebooks that will model data from corresponding feature engineering kernels as we keep adding hopefully some creative features :)
# 
# **Important Note:** Feature generation for test data will happen online since private test set is not publicly available for precomputation!
# 
# This notebook will give a LB around: 0.506 (score can vary but it's solely for baseline purposes)
# 
# **To see how features are generated in more detail:** [Feature Engineering Part 1 Notebook](https://www.kaggle.com/keremt/fastai-feature-engineering-part1-6160-feats/)

# ### Imports

# In[ ]:


from fastai.core import *
Path.read_csv = lambda o: pd.read_csv(o)
input_path = Path("/kaggle/input/data-science-bowl-2019")
pd.options.display.max_columns=200
pd.options.display.max_rows=200
input_path.ls()


# ### Read Data

# In[ ]:


train_with_features_part1 = pd.read_csv("../input/dsbowl2019-feng-part1/train_with_features_part1.csv")


# In[ ]:


sample_subdf = (input_path/'sample_submission.csv').read_csv()
# specs_df = (input_path/"specs.csv").read_csv()
# train_labels_df = (input_path/"train_labels.csv").read_csv()
# train_df = (input_path/"train.csv").read_csv()
test_df = (input_path/"test.csv").read_csv()


# In[ ]:


sample_subdf.shape, test_df.shape, train_with_features_part1.shape


# ### Features (part1)
# 
# Basically here we redefine the feature generation code for test.

# In[ ]:


from fastai.tabular import *
import types

stats = ["median","mean","sum","min","max"]
UNIQUE_COL_VALS = pickle.load(open("../input/dsbowl2019-feng-part1/UNIQUE_COL_VALS.pkl", "rb"))


# In[ ]:


for k in UNIQUE_COL_VALS.__dict__.keys():
    print(k, len(UNIQUE_COL_VALS.__dict__[k]))


# In[ ]:


def array_output(f):
    def inner(*args, **kwargs): return array(listify(f(*args, **kwargs))).flatten()
    return inner

feature_funcs = []

@array_output
def time_elapsed_since_hist_begin(df):
    "total time passed until assessment begin"
    return df['timestampElapsed'].max() - df['timestampElapsed'].min()

feature_funcs.append(time_elapsed_since_hist_begin)

@array_output
def time_elapsed_since_each(df, types, dfcol):
    "time since last occurence of each types, if type not seen then time since history begin"
    types = UNIQUE_COL_VALS.__dict__[types]
    last_elapsed = df['timestampElapsed'].max()
    _d = dict(df.iloc[:-1].groupby(dfcol)['timestampElapsed'].max())
    return [last_elapsed - _d[t] if t in _d else time_elapsed_since_hist_begin(df)[0] for t in types]

feature_funcs.append(partial(time_elapsed_since_each, types="media_types", dfcol="type"))
feature_funcs.append(partial(time_elapsed_since_each, types="titles", dfcol="title"))
feature_funcs.append(partial(time_elapsed_since_each, types="event_ids", dfcol="event_id"))
feature_funcs.append(partial(time_elapsed_since_each, types="worlds", dfcol="world"))
feature_funcs.append(partial(time_elapsed_since_each, types="event_codes", dfcol="event_code"))

@array_output
def countfreqhist(df, types, dfcol, freq=False):
    "count or freq of types until assessment begin"
    types = UNIQUE_COL_VALS.__dict__[types]
    _d = dict(df[dfcol].value_counts(normalize=(True if freq else False)))
    return [_d[t] if t in _d else 0 for t in types]

feature_funcs.append(partial(countfreqhist, types="media_types", dfcol="type", freq=False))
feature_funcs.append(partial(countfreqhist, types="media_types", dfcol="type", freq=True))

feature_funcs.append(partial(countfreqhist, types="titles", dfcol="title", freq=False))
feature_funcs.append(partial(countfreqhist, types="titles", dfcol="title", freq=True))

feature_funcs.append(partial(countfreqhist, types="event_ids", dfcol="event_id", freq=False))
feature_funcs.append(partial(countfreqhist, types="event_ids", dfcol="event_id", freq=True))

feature_funcs.append(partial(countfreqhist, types="worlds", dfcol="world", freq=False))
feature_funcs.append(partial(countfreqhist, types="worlds", dfcol="world", freq=True))

feature_funcs.append(partial(countfreqhist, types="event_codes", dfcol="event_code", freq=False))
feature_funcs.append(partial(countfreqhist, types="event_codes", dfcol="event_code", freq=True))

@array_output
def overall_event_count_stats(df):
    "overall event count stats until assessment begin"
    return df['event_count'].agg(stats)
feature_funcs.append(overall_event_count_stats)

@array_output
def event_count_stats_each(df, types, dfcol):
    "event count stats per media types until assessment begin, all zeros if media type missing for user"
    types = UNIQUE_COL_VALS.__dict__[types]
    _stats_df = df.groupby(dfcol)['event_count'].agg(stats)
    _d = dict(zip(_stats_df.reset_index()[dfcol].values, _stats_df.values))
    return [_d[t] if t in _d else np.zeros(len(stats)) for t in types]
feature_funcs.append(partial(event_count_stats_each, types="media_types", dfcol="type"))
feature_funcs.append(partial(event_count_stats_each, types="titles", dfcol="title"))
feature_funcs.append(partial(event_count_stats_each, types="event_ids", dfcol="event_id"))
feature_funcs.append(partial(event_count_stats_each, types="worlds", dfcol="world"))
feature_funcs.append(partial(event_count_stats_each, types="event_codes", dfcol="event_code"))

@array_output
def overall_session_game_time_stats(df):
    "overall session game time stats until assessment begin"
    return df['game_time'].agg(stats)
feature_funcs.append(overall_session_game_time_stats)

@array_output
def session_game_time_stats_each(df, types, dfcol):
    "session game time stats per media types until assessment begin, all zeros if missing for user"
    types = UNIQUE_COL_VALS.__dict__[types]
    _stats_df = df.groupby(dfcol)['game_time'].agg(stats)
    _d = dict(zip(_stats_df.reset_index()[dfcol].values, _stats_df.values))
    return [_d[t] if t in _d else np.zeros(len(stats)) for t in types]
feature_funcs.append(partial(session_game_time_stats_each, types="media_types", dfcol="type"))
feature_funcs.append(partial(session_game_time_stats_each, types="titles", dfcol="title"))
feature_funcs.append(partial(session_game_time_stats_each, types="event_ids", dfcol="event_id"))
feature_funcs.append(partial(session_game_time_stats_each, types="worlds", dfcol="world"))
feature_funcs.append(partial(session_game_time_stats_each, types="event_codes", dfcol="event_code"))

len(feature_funcs)


# ### Test Feature Engineering
# 
# Test set in LB and Private LB is different than what is publicly shared. So feature engineering and inference for test set should be done online.

# In[ ]:


def get_sorted_user_df(df, ins_id):
    "extract sorted data for a given installation id and add datetime features"
    _df = df[df.installation_id == ins_id].sort_values("timestamp").reset_index(drop=True)
    add_datepart(_df, "timestamp", time=True)
    return _df

def get_test_assessment_start_idxs(df): 
    return list(df.sort_values("timestamp")
                  .query("type == 'Assessment' & event_code == 2000")
                  .groupby("installation_id").tail(1).index)

def get_test_feats_row(idx, i):
    "get all faeatures by an installation start idx"
    ins_id = test_df.loc[idx, "installation_id"]
    _df = get_sorted_user_df(test_df, ins_id)
    assessment_row = _df.iloc[-1]
    row_feats = np.concatenate([f(_df) for f in feature_funcs])
    feat_row = pd.Series(row_feats, index=[f"static_feat{i}"for i in range(len(row_feats))])
    row = pd.concat([assessment_row, feat_row])
    return row


# In[ ]:


# Feature Engineering part 1
start_idxs = get_test_assessment_start_idxs(test_df)
res = parallel(get_test_feats_row, start_idxs)
test_with_features_df = pd.concat(res,1).T


# In[ ]:


test_with_features_part1 = test_with_features_df


# In[ ]:


# check to see train and test have same features
num_test_feats = [c for c in test_with_features_df.columns if c.startswith("static")]
num_train_feats = [c for c in train_with_features_part1.columns if c.startswith("static")]
assert num_train_feats == num_test_feats


# ### TabularLearner Model
# 
# Here we use a single validation but in later stages once we finalize features we should use cross-validation. We don't over optimize the model or do any hyperparameter search since the whole purpose is to get a baseline and build on top of it in upcoming parts.

# In[ ]:


from fastai.tabular import *


# In[ ]:


train_with_features_part1.shape, test_with_features_part1.shape


# In[ ]:


# create validation set - split by installation_id
np.random.seed(42)
valid_ids = (np.random.choice(train_with_features_part1.installation_id.unique(),
                              int(len(train_with_features_part1)*0.05)))
valid_idx = (train_with_features_part1[train_with_features_part1.installation_id.isin(valid_ids)].index); valid_idx


# In[ ]:


# get data
cat_names = ['title','world']
cont_names = [c for c in train_with_features_part1.columns if c.startswith("static_")]

procs = [FillMissing, Categorify, Normalize]
data = TabularDataBunch.from_df(path=".", df=train_with_features_part1, dep_var="accuracy", 
                                valid_idx=valid_idx, procs=procs, cat_names=cat_names, cont_names=cont_names)

data.add_test(TabularList.from_df(test_with_features_part1, cat_names=cat_names, cont_names=cont_names));


# In[ ]:


# fit
learner = tabular_learner(data, [256,256], y_range=(0.,1.), ps=0.6)
learner.fit_one_cycle(10, 3e-3)


# ### Check Validation Score
# 
# Again, we don't search for optimal coefficients since main purpose is to create a baseline.

# In[ ]:


from sklearn.metrics import cohen_kappa_score


# In[ ]:


coefs=array([0.25,0.50,0.75])
def soft2hard(o):
    if o < coefs[0]: return 0
    elif o < coefs[1]: return 1
    elif o < coefs[2]: return 2
    else: return 3


# In[ ]:


# get valid preds
preds, targs = learner.get_preds()


# In[ ]:


# get accuracy_group for preds and targs
_preds = array([soft2hard(o.item()) for o in preds])
_targs = array(train_with_features_part1.iloc[valid_idx]['accuracy_group'].values)


# In[ ]:


# see validation score
cohen_kappa_score(_targs, _preds, weights="quadratic")


# ### Submit

# In[ ]:


# get test preds
preds,targs=learner.get_preds(DatasetType.Test)
_preds = array([soft2hard(o.item()) for o in preds])


# In[ ]:


Counter(_preds)


# In[ ]:


# get installation ids for test set
test_ids = test_with_features_part1['installation_id'].values; len(test_ids)


# In[ ]:


# generate installation_id : pred dict
test_preds_dict = dict(zip(test_ids, _preds)); len(test_preds_dict)


# In[ ]:


# create submission
sample_subdf['accuracy_group'] = sample_subdf.installation_id.map(test_preds_dict)
sample_subdf['accuracy_group'] = sample_subdf['accuracy_group'].fillna(3)
sample_subdf.to_csv("submission.csv", index=False)


# ### end
