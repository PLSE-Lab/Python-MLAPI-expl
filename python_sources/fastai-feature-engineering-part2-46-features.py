#!/usr/bin/env python
# coding: utf-8

# ### Feature Engineering with part 2
# 
# In this notebook we implement cross validated target encoding features.
# 
# **To go back to previous modeling notebook:** [Part 1 Modeling Notebook](https://www.kaggle.com/keremt/fastai-model-part1-regression/)
# 
# **To skip and go to next modeling notebook:** [Part 2 Modeling Notebook](https://www.kaggle.com/keremt/fastai-model-part2-regression/)

# ### Imports
# 
# We will use fastai v1

# In[ ]:


from fastai.core import *
Path.read_csv = lambda o: pd.read_csv(o)
input_path = Path("/kaggle/input/data-science-bowl-2019")
pd.options.display.max_columns=200
pd.options.display.max_rows=200
input_path.ls()


# ### Read data

# In[ ]:


sample_subdf = (input_path/'sample_submission.csv').read_csv()
specs_df = (input_path/"specs.csv").read_csv()
train_df = (input_path/"train.csv").read_csv()
train_labels_df = (input_path/"train_labels.csv").read_csv()
test_df = (input_path/"test.csv").read_csv()


# In[ ]:


assert set(train_df.installation_id).intersection(set(test_df.installation_id)) == set()


# Load part 1 data too

# In[ ]:


train_with_features_part1 = pd.read_feather("../input/dsbowl2019-feng-part1/train_with_features_part1.fth")


# In[ ]:


train_with_features_part1.shape, test_df.shape, train_labels_df.shape


# In[ ]:


train_with_features_part1.head()


# In[ ]:


test_df.head()


# In[ ]:


# there shouldn't be any common installation ids between test and train 
assert set(train_df.installation_id).intersection(set(test_df.installation_id)) == set()


# Train (also train labels) and test doesn't have any common installation ids. This means that we can't use past assessment target information for a given user, e.g. we can't use how a child did in his/her previous to predict for future assessment results. 
# 
# Instead we need to create global features from target information, e.g. using categorical target encoding and similar techniques.

# ### Cross Validated Target Encoding
# 
# We will use cv based target encoding for using label information. For more information see really nice documentation here in [h2o.ai](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-munging/target-encoding.html#holdout-type). For these calculations we will use `train_with_features_part1` and calculate stats on different target information by different categorical data. Note that we will not use `event_ids` since it has a 1:1 mapping `titles` adding no extra information, `media_types` since always `Assesment` and `event_codes` since it's always `2000`. 
# 
# **Disclaimer:** Target encoding on the following categories can be not that effective since their cardinality is low.
# 
# Value: num_correct x num_incorrect x accuracy x hist of accuracy group
# 
# By: Title x World 
# 

# In[ ]:


from fastai.tabular import *
import types

stats = ["median","mean","sum","min","max"]
UNIQUE_COL_VALS = pickle.load(open("../input/dsbowl2019-feng-part1/UNIQUE_COL_VALS.pkl", "rb"))
list(UNIQUE_COL_VALS.__dict__.keys())


# In[ ]:


# add accuracy_group unique vals
UNIQUE_COL_VALS.__dict__['accuracy_groups'] = np.unique(train_with_features_part1.accuracy_group)
UNIQUE_COL_VALS.accuracy_groups
pickle.dump(UNIQUE_COL_VALS, open( "UNIQUE_COL_VALS.pkl", "wb" ))


# In[ ]:


def target_encoding_stats_dict(df, by, targetcol):
    "get target encoding stats dict, by:[stats]"
    _stats_df = df.groupby(by)[targetcol].agg(stats)   
    _d = dict(zip(_stats_df.reset_index()[by].values, _stats_df.values))
    return _d


# In[ ]:


def _value_counts(o, freq=False): return dict(pd.value_counts(o, normalize=freq))
def countfreqhist_dict(df, by, targetcol, types, freq=False):
    "count or freq histogram dict for categorical targets"
    types = UNIQUE_COL_VALS.__dict__[types]
    _hist_df = df.groupby(by)[targetcol].agg(partial(_value_counts, freq=freq))
    _d = dict(zip(_hist_df.index, _hist_df.values))
    for k in _d: _d[k] = array([_d[k][t] for t in types]) 
    return _d


# In[ ]:


countfreqhist_dict(train_with_features_part1, "title", "accuracy_group", "accuracy_groups")


# In[ ]:


f1 = partial(target_encoding_stats_dict, by="title", targetcol="num_incorrect")
f2 = partial(target_encoding_stats_dict, by="title", targetcol="num_correct")
f3 = partial(target_encoding_stats_dict, by="title", targetcol="accuracy")
f4 = partial(target_encoding_stats_dict, by="world", targetcol="num_incorrect")
f5 = partial(target_encoding_stats_dict, by="world", targetcol="num_correct")
f6 = partial(target_encoding_stats_dict, by="world", targetcol="accuracy")

f7 = partial(countfreqhist_dict, by="title", targetcol="accuracy_group", types="accuracy_groups",freq=False)
f8 = partial(countfreqhist_dict, by="title", targetcol="accuracy_group", types="accuracy_groups",freq=True)
f9 = partial(countfreqhist_dict, by="world", targetcol="accuracy_group", types="accuracy_groups",freq=False)
f10 = partial(countfreqhist_dict, by="world", targetcol="accuracy_group", types="accuracy_groups",freq=True)


# ### Compute Features for Train

# In[ ]:


from sklearn.model_selection import KFold
# create cross-validated indexes
unique_ins_ids = np.unique(train_with_features_part1.installation_id)
train_val_idxs = KFold(5, random_state=42).split(unique_ins_ids)


# In[ ]:


feature_dfs = [] # collect computed _val_feats_dfs here
for train_idxs, val_idxs  in train_val_idxs:
    # get train and val dfs
    train_ins_ids, val_ins_ids = unique_ins_ids[train_idxs], unique_ins_ids[val_idxs]
    _train_df = train_with_features_part1[train_with_features_part1.installation_id.isin(train_ins_ids)]
    _val_df = train_with_features_part1[train_with_features_part1.installation_id.isin(val_ins_ids)]
    assert (_train_df.shape[0] + _val_df.shape[0]) == train_with_features_part1.shape[0]
    # compute features for val df
    _idxs = _val_df['title'].map(f1(_train_df)).index
    feat1 = np.stack(_val_df['title'].map(f1(_train_df)).values)
    feat2 = np.stack(_val_df['title'].map(f2(_train_df)).values)
    feat3 = np.stack(_val_df['title'].map(f3(_train_df)).values)
    feat4 = np.stack(_val_df['world'].map(f4(_train_df)).values)
    feat5 = np.stack(_val_df['world'].map(f5(_train_df)).values)
    feat6 = np.stack(_val_df['world'].map(f6(_train_df)).values)
    feat7 = np.stack(_val_df['title'].map(f7(_train_df)).values)
    feat8 = np.stack(_val_df['title'].map(f8(_train_df)).values)
    feat9 = np.stack(_val_df['world'].map(f9(_train_df)).values)
    feat10 = np.stack(_val_df['world'].map(f10(_train_df)).values)
    # create dataframe with same index for later merge
    _val_feats = np.hstack([feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10])
    _val_feats_df = pd.DataFrame(_val_feats, index=_idxs)
    _val_feats_df.columns = [f"targenc_feat{i}"for i in range(_val_feats_df.shape[1])]
    feature_dfs.append(_val_feats_df)


# In[ ]:


train_feature_df = pd.concat(feature_dfs, 0)


# In[ ]:


train_with_features_part2 = pd.concat([train_with_features_part1, train_feature_df],1)


# In[ ]:


train_with_features_part2.to_feather("train_with_features_part2.fth")


# ### end
