import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import tqdm
from scipy.stats import kurtosis
import time
import pickle
from collections import defaultdict
from statsmodels.stats.proportion import proportion_confint

tqdm.tqdm.pandas()
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
print(os.listdir("../input"))
plt.style.use('seaborn')
sns.set(font_scale=1)

def save_pickle(file_name, data):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file_name):
    with open(file_name, 'rb') as handle:
        data =  pickle.load(handle)
    return data

n_fold = 10
n_cv = 1
random_state = 42

np.random.seed(random_state)
df_train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
df_test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')

test_public_idx = np.load('../input/list-of-fake-samples-and-public-private-lb-split/public_LB.npy')
test_private_idx = np.load('../input/list-of-fake-samples-and-public-private-lb-split/private_LB.npy')

test_public_idx = list(test_public_idx.tolist()) 
test_private_idx = list(test_private_idx.tolist())
test_real_idx = np.array(test_public_idx+test_private_idx)

df_test = df_test.loc[df_test.index.isin(test_real_idx)]

feat = [f for f in df_train.columns.values if f not in ['ID_code', 'target']]
df_train[feat] = df_train[feat].apply(pd.to_numeric, downcast='float')
df_test[feat] = df_test[feat].apply(pd.to_numeric, downcast='float')
gc.collect()

df = pd.concat((df_train, df_test), axis=0, sort=False)

def calculate_groupby_list(count_list=[1]):
    
    # To-do: select counts to groupby
    
    return count_list

def unique_rank(x):
    tmp = x.rank(method='dense')
    return tmp/tmp.max()

def calculate_per_group(df_in, vcc_th=0.01, count_group_list=None):

    tmp_col = []
    for i in tqdm.tqdm_notebook(range(200), total=200):
        var_col = f'var_{i}'
        count_col = f'var_{i}_value_count'
        rank_per_count_col = f'var_{i}_rank_per_count'
        rank_per_count_col_tmp = rank_per_count_col+'_tmp'
        tmp_col.append(rank_per_count_col_tmp)

        df_in[rank_per_count_col] = 0
        value_count_count = df_in[count_col].value_counts()
        if isinstance(vcc_th, float):
            vcc_list = value_count_count.loc[value_count_count/len(df_in)>=vcc_th].index.tolist()
        elif isinstance(vcc_th, int):
            vcc_list = [vcc_th]
        elif isinstance(vcc_th, list):
            vcc_list = vcc_th
        df_in[rank_per_count_col_tmp] = df_in[count_col]
        df_in.loc[~df_in[count_col].isin(vcc_list), rank_per_count_col_tmp] = 0
        vcc_list += [0]
        for j in vcc_list:
            df_in.loc[df[rank_per_count_col_tmp]==j, rank_per_count_col] = unique_rank(df_in.loc[df_in[rank_per_count_col_tmp]==j, var_col])
        
    df_in.drop(tmp_col, axis=1, inplace=True)
    return df_in

map_dict = {}
for i in tqdm.tqdm_notebook(range(200), total=200):
    # value count, 200 features
    var_col = f'var_{i}'
    count_col = f'var_{i}_value_count'
    map_dict[i] = df[var_col].value_counts().reset_index(drop=False)
    map_dict[i].columns = [var_col, count_col]
    
    # some operations, 600 features
    count_mul_value_col = f'var_{i}_count_mul_value'
    count_div_value_col = f'var_{i}_count_div_value'
    value_div_count_col = f'var_{i}_value_div_count'
    map_dict[i][count_mul_value_col] = map_dict[i][var_col]*map_dict[i][count_col]
    map_dict[i][count_div_value_col] = map_dict[i][var_col]/map_dict[i][count_col]
    
    # rank ratio and zscore per group, 400 features
    tmp_col = f'var_{i}_groupby'
    rank_col = f'var_{i}_rank_per_count'
    zscore_col = f'var_{i}_zscore_per_group'
    groupby_count_list = calculate_groupby_list([1])
    map_dict[i][tmp_col] = map_dict[i][count_col]
    map_dict[i].loc[~map_dict[i][count_col].isin(groupby_count_list), tmp_col] = 0
    map_dict[i][rank_col] = map_dict[i].groupby(tmp_col)[var_col].transform(lambda x: unique_rank(x))
    map_dict[i][zscore_col] = map_dict[i].groupby(tmp_col)[var_col].transform(lambda x: (x-x.mean())/x.std())
    map_dict[i].drop(tmp_col, axis=1, inplace=True)
    
    ### zscore per count, 200 features
    zscore_per_count_col = f'var_{i}_zscore_per_count'
    map_dict[i][zscore_per_count_col] = map_dict[i].groupby(count_col)[var_col].transform(lambda x: (x-x.mean())/x.std())
    
    ### raw value with count==1 removed, 200 features
    var_clean_col = f'var_{i}_count_1_clean'
    map_dict[i][var_clean_col] = map_dict[i][var_col]
    map_dict[i].loc[map_dict[i][count_col]==1, var_clean_col] = -1000
    
    # set value as index
    map_dict[i].set_index(var_col, inplace=True)

def augment_fe(x_train_in, y_train_in, x_valid_in, y_valid_in, x_test_in, 
               pos_aug_ratio=2, augmentation=True):
    '''Augmentation and feature engineering'''
    
    if augmentation:
    
        # Augment only train data
        x = x_train_in.values
        y = y_train_in.values

        xs = [] # positive samples
        xn = [] # negative samples

        # augment positive samples
        pos_aug_ratio = max(int(pos_aug_ratio), 1)
        for i in range(pos_aug_ratio):
            mask = y>0
            x1 = x[mask].copy()
            ids = np.arange(x1.shape[0])
            for c in range(x1.shape[1]):
                np.random.shuffle(ids)
                x1[:,c] = x1[ids,c]
            xs.append(x1)
        xs = np.vstack(xs)
        ys = np.ones(xs.shape[0])

        # augment negative samples
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids,c]
        xn.append(x1)
        xn = np.vstack(xn)
        yn = np.zeros(xn.shape[0])

        x = np.vstack([x,xs,xn])
        y = np.concatenate([y,ys,yn])
        x_train_in = pd.DataFrame(x, columns=x_train_in.columns) # put augmented data back into a DataFrame, but note that #rows is different
    
        y_train_in = pd.Series(y, name=y_train_in.name)

    # FE
    # concat all data into one
    df_all = pd.concat((x_train_in, x_valid_in, x_test_in), axis=0, sort=False)
    for i in tqdm.tqdm_notebook(range(200), total=200):
        for j in map_dict[i].columns:
            df_all[j] = df_all[f'var_{i}'].map(map_dict[i][j])
        
    # separate data
    x_train_out = df_all.iloc[:x_train_in.shape[0], :]
    x_valid_out = df_all.iloc[x_train_in.shape[0]:(x_train_in.shape[0]+x_valid_in.shape[0]), :]
    x_test_out = df_all.iloc[(x_train_in.shape[0]+x_valid_in.shape[0]):, :]
    
    del df_all
    gc.collect()
    
    y_train_out = y_train_in
    y_valid_out = y_valid_in
    
    return x_train_out, y_train_out, x_valid_out, y_valid_out, x_test_out

skf = RepeatedStratifiedKFold(n_splits=n_fold, n_repeats=n_cv, random_state=random_state)
oof = df_train[['ID_code', 'target']]
oof['predict'] = 0
predictions = df_test[['ID_code']]
val_aucs = []
feature_importance_df = pd.DataFrame()

lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state
}

features = [col for col in df_train.columns if col not in ['target', 'ID_code']]
x_test_in = df_test[features]

run_time = []
for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train, df_train['target'])):
    start_time = time.time()
    
    x_train_in, y_train_in = df_train.iloc[trn_idx][features], df_train.iloc[trn_idx]['target']
    x_valid_in, y_valid_in = df_train.iloc[val_idx][features], df_train.iloc[val_idx]['target']

    p_valid, yp = 0, 0
    x_train_out, y_train_out, x_valid_out, y_valid_out, x_test_out = \
        augment_fe(x_train_in, y_train_in, x_valid_in, y_valid_in, x_test_in, 
                   pos_aug_ratio=11, augmentation=False)
    
    trn_data = lgb.Dataset(x_train_out, label=y_train_out)
    val_data = lgb.Dataset(x_valid_out, label=y_valid_out)
    evals_result = {}
    lgb_clf = lgb.train(lgb_params,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000,
                        evals_result=evals_result)
    p_valid += lgb_clf.predict(x_valid_out)
    yp += lgb_clf.predict(x_test_out)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = x_valid_out.columns
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    oof['predict'][val_idx] = p_valid/n_cv
    val_score = roc_auc_score(y_valid_in, p_valid)
    val_aucs.append(val_score)
    
    predictions[f'fold_{fold}'] = yp/n_cv
    
    run_time.append(time.time()-start_time)
    print('='*100)
    print(f'Fold {fold}, AUC={val_score:.6f}, mean AUC={np.mean(val_aucs):.6f}, std AUC={np.std(val_aucs):.6f}. Fold run time {run_time[-1]/60:.3f} min., total run time {np.sum(run_time)/60:.3f} min.')
    print('='*100)
    
    del x_train_out, x_valid_out, x_test_out
    gc.collect()

mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))

cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,120))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances-1.png')
feature_importance_df.to_csv('feature_importance.csv')

# submission
predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
predictions.to_csv('lgb_all_predictions-1.csv', index=False)
sub_df = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
sub_df.loc[sub_df.ID_code.isin(predictions.ID_code), 'target'] = predictions.target.values
sub_df.to_csv("lgb_submission-1.csv", index=False)
oof.to_csv('lgb_oof-1.csv', index=False)