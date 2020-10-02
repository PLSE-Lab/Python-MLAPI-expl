# largely taken from samrat's kernel: https://www.kaggle.com/samratp/aggregates-sumvalues-sumzeros-k-means-pca

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

lgbm_params =  {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    "learning_rate": 0.001,
    "num_leaves": 31,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.5,
    'bagging_freq': 1,
    "max_depth": -1,
    'zero_as_missing':True
    }

max_iter = 1000000



def get_features_vec(df, clusters, scaler1, scaler2):
    agg_col = ['num_zeroes', 'max', 'min', 'sum', 'median', 'var', 'stddev', 'kurt', 'skew', 'mad']
    df = df.replace(0, np.nan)

    df_columns = [i for i in df.columns if i not in ['ID', 'target']]
    df['num_zeroes'] = df[df_columns].count(axis=1)
    df['mean'] = df[df_columns].mean(axis=1)
    df['median'] = df[df_columns].median(axis=1)
    df['sum'] = df[df_columns].sum(axis=1)
    df['max'] = df[df_columns].max(axis=1)
    df['min'] = df[df_columns].min(axis=1)
    df['var'] = df[df_columns].var(axis=1)
    df['stddev'] = df[df_columns].std(axis=1)
    df['kurt'] = df[df_columns].kurt(axis=1)
    df['skew'] = df[df_columns].skew(axis=1)
    df['mad'] = df[df_columns].mad(axis=1)
    df = df.fillna(0)

    if not scaler1 or not scaler2:
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()
        x_df = df[df_columns]
        x_df2 = df[agg_col]

        scaler1.fit(x_df)
        scaler2.fit(x_df2)


    x_df = df[df_columns]
    x_df2 = df[agg_col]
    x_df = scaler1.transform(x_df)
    x_df2 = scaler2.transform(x_df2)


    if not clusters:
        clusters = []

        for i in [10, 25, 50, 100]:
            clf = MiniBatchKMeans(n_clusters=i)
            clf.fit(x_df)
            clusters.append(clf)

            clf2= MiniBatchKMeans(n_clusters=i)
            clf2.fit(x_df2)
            clusters.append(clf2)

    for count, i in enumerate(clusters):
        if count %2 == 0:
            df['cluster_{0}'.format(count)] = i.predict(x_df)
        else:
            df['cluster_{0}'.format(count)] = i.predict(x_df2)

    return df, clusters, scaler1, scaler2


def drop_constant(df_train, df_test):
    colsToRemove = []
    for col in df_train.columns:
        if col != 'ID' and col != 'target' and df_train[col].std() == 0:
            colsToRemove.append(col)

    df_train = df_train.drop(colsToRemove, axis=1)

    df_test = df_test.drop(colsToRemove, axis=1)
    return df_train, df_test

def main():
    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')
    train_df, test_df = drop_constant(train_df, test_df)

    print('preprocessing train')
    train_df, clusters, scaler1, scaler2 = get_features_vec(train_df, None, None, None)
    print('preprocessing test')
    test_df, _, _, _ = get_features_vec(test_df, clusters, scaler1, scaler2)

    y_train = np.log(train_df.target + 1)
    train_df.drop(['target', 'ID'], axis=1, inplace=True)

    k = KFold(n_splits=10, shuffle=True, random_state=1)

    for fold_id, (train_idx, val_idx) in enumerate(k.split(train_df, y_train)):
        X_train_k = np.array(train_df)[train_idx]
        y_train_k = np.array(y_train)[train_idx]
        X_valid_k =  np.array(train_df)[val_idx]
        y_valid_k = np.array(y_train)[val_idx]

        lgtrain = lgb.Dataset(X_train_k, y_train_k)
        lgvalid = lgb.Dataset(X_valid_k, y_valid_k)

        modelstart = time.time()
        model = lgb.train(
            lgbm_params,
            lgtrain,
            num_boost_round=max_iter,
            valid_sets=[lgtrain, lgvalid],
            valid_names=['train', 'valid'],
            early_stopping_rounds=100,
            verbose_eval=1
        )

        test_x = test_df.drop('ID', axis = 1)
        test_pred = model.predict(test_x.values, num_iteration=model.best_iteration)
        Y_target.append(test_pred)

    Y_target = np.array(Y_target)

    test_df['target'] = np.exp(Y_target.mean(axis=0)) - 1
    test_df = test_df[['ID', 'target']]
    test_df.to_csv('res.csv', index=False)


if __name__ == '__main__':
    main()
    # test_df = pd.read_csv('res.csv')
    # test_df = test_df[['ID', 'target']]
    # test_df.to_csv('res.csv', index=False)


