import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from sklearn.externals import joblib
import gc
import os

# download data at https://www.kaggle.com/c/pubg-finish-placement-prediction/data
df = pd.read_csv('../input/train_V2.csv', header=0)
print(df.shape)


def reduce_mem_usage(df):
    """
    for kaggle kernal, reduce memory usage
    iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# add new features
df['total_distance'] = df['swimDistance'] + df['walkDistance'] + df['rideDistance']
df['kill_and_assist'] = df['kills'] + df['assists']
df['healitem'] = df['heals'] + df['boosts']


def feature_engineering(df, is_train=True):
    """
    generate group/match data
    :param df: input data
    :param is_train: is input a training set
    :return:
    """
    test_idx = None
    if is_train:
        print("processing training data")
        # remove outliers
        df = df[df['maxPlace'] > 1]
        df = df[
            (df.killStreaks < 8) & (df.assists < 10) & (df.headshotKills < 11) & (df.DBNOs < 16) & (
                df.damageDealt <= 2000) & (
                df.kills < 21)]
    else:
        print("processing test.csv")
        test_idx = df.Id

    print("remove some columns")
    target = 'winPlacePerc'
    features = list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchType")

    # labels of data, if input data is not a training set, y is None
    y = None

    if is_train:
        print("get labels")
        y = np.array(df.groupby(['matchId', 'groupId'])[target].agg('mean'), dtype=np.float64)
        features.remove(target)

    print("get mean value of features in a group and rank in the match")
    agg = df.groupby(['matchId', 'groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    if is_train:
        df_out = agg.reset_index()[['matchId', 'groupId']]
    else:
        df_out = df[['matchId', 'groupId']]

    # merge data
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])

    print("get max value of features in a group and rank in the match")
    agg = df.groupby(['matchId', 'groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    # merge data
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])

    print("get min value of features in a group and position in the match")
    agg = df.groupby(['matchId', 'groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    # merge data
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])

    print("get group size feature")
    agg = df.groupby(['matchId', 'groupId']).size().reset_index(name='group_size')
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])

    print("get mean value of features in a match")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    print("get match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    df_out = df_out.merge(agg, how='left', on=['matchId'])

    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)

    X = df_out

    feature_names = list(df_out.columns)

    # for Kaggle kernel, delete useless data and call gc.collect, avoid OOM
    del df, df_out, agg, agg_rank

    gc.collect()
    return X, y, feature_names, test_idx


features, label, feature_names, _ = feature_engineering(df)

# add headshot rate and kill streak rate of match
features['headshot_rate'] = features['headshotKills'] / features['kills']
features['headshot_rate'].fillna(0, inplace=True)
features['killStreak_rate'] = features['killStreaks'] / features['kills']
features['killStreak_rate'].fillna(0, inplace=True)
features = reduce_mem_usage(features)


def lightgbm(features, label):
    # transform data to lgb dataset
    train_features, valid_features, train_label, valid_label = train_test_split(features, label)
    train_data = lgb.Dataset(train_features, label=train_label)
    valid_data = lgb.Dataset(valid_features, label=valid_label)
    # lgb param
    # see http://lightgbm.apachecn.org/cn/latest/Parameters.html
    parameters = {
        'min_data_in_leaf': 20, 'feature_fraction': 0.80, 'bagging_fraction': 0.8,
        'boosting_type': 'gbdt', 'learning_rate': 0.05, 'num_leaves': 31,
        'application': 'regression', 'zero_as_missing': False, 'metric': 'mae', 'seed': 2
    }
    model = lgb.train(parameters, train_set=train_data, valid_sets=valid_data, verbose_eval=1000,
                      early_stopping_rounds=100, num_boost_round=10000)
    model.save_model('./lgb_model.txt')
    return model, model.best_score


# use random forest model
def random_forest(features, label):
    train_features, valid_features, train_label, valid_label = train_test_split(features, label)
    if os.path.isfile('./rf_model.joblib'):
        rf_model = joblib.load('./rf_model.joblib')
    else:
        rf_model = RandomForestRegressor(n_jobs=-1, criterion='mae', n_estimators=20)
        rf_model.fit(train_features, train_label)
        joblib.dump(rf_model, './rf_model.joblib')
    rf_valid_result = rf_model.predict(valid_features)
    rf_error = mean_absolute_error(valid_label, rf_valid_result)
    return rf_model, rf_error


# use k-nn model
def knn(features, label):
    train_features, valid_features, train_label, valid_label = train_test_split(features, label)
    if os.path.isfile('./knn_model.joblib'):
        knn_model = joblib.load('./knn_model.joblib')
    else:
        knn_model = KNeighborsRegressor(n_jobs=-1, n_neighbors=10)
        knn_model.fit(train_features, train_label)
        joblib.dump(knn_model, './knn_model.joblib')
    knn_valid_result = knn_model.predict(valid_features)
    knn_error = mean_absolute_error(valid_label, knn_valid_result)
    return knn_model, knn_error


def decision_tree(features, label):
    train_features, valid_features, train_label, valid_label = train_test_split(features, label)
    if os.path.isfile('./dt_model.joblib'):
        dt_model = joblib.load('./dt_model.joblib')
    else:
        dt_model = DecisionTreeRegressor(criterion='mae')
        dt_model.fit(train_features, train_label)
        joblib.dump(dt_model, './dt_model.joblib')
    dt_valid_result = dt_model.predict(valid_features)
    dt_error = mean_absolute_error()
    return dt_model, dt_error


print('start training model')
# model, score = lightgbm(features, label)
model, score = random_forest(features, label)
print(score)
# print(test_df.columns)
del features
gc.collect()

# predict the value of test data and generate output file
test_df = pd.read_csv('../input/test_V2.csv', header=0)

# test_id = test_df['Id']

# add features on test set
test_df['total_distance'] = test_df['swimDistance'] + test_df['walkDistance'] + test_df['rideDistance']
test_df['kill_and_assist'] = test_df['kills'] + test_df['assists']
test_df['healitem'] = test_df['heals'] + test_df['boosts']
# test_df = reduce_mem_usage(test_df)

test_features, _, _, test_id = feature_engineering(test_df, is_train=False)
test_features['headshot_rate'] = test_features['headshotKills'] / test_features['kills']
test_features['headshot_rate'].fillna(0, inplace=True)
test_features['killStreak_rate'] = test_features['killStreaks'] / test_features['kills']
test_features['killStreak_rate'].fillna(0, inplace=True)
test_features = reduce_mem_usage(test_features)
test_predict = model.predict(test_features)
# # print(test_predict)
test_predict = test_predict.reshape((len(test_predict), 1))
test_predict = pd.DataFrame(test_predict, columns=['winPlacePerc'])
# print(test_predict)
result = pd.concat([test_id, test_predict], axis=1)
# print(result)
result.to_csv('./submission.csv', index=False)
