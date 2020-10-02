!pip install tensorflow==2.0.0-alpha0
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.feature_column import numeric_column
from tensorflow.estimator import BoostedTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time

tanh_scalar = MinMaxScaler(feature_range=(-1, 1))

print(tf.__version__)


def load_data(df, target, drop_cols):
    train_y = df[target]
    train_X = df.drop([*drop_cols,target], axis=1)

    return train_test_split(train_X, train_y, test_size=0.1, random_state=42)

def scale_features(df, cols, scalar):
    df[cols] = scalar.fit_transform(df[cols])
    return df

def reduce_mem(df):

    start_mem = round((df.memory_usage().sum() / 1024**2), 2)
    print(f'Memory usage of dataframe is {start_mem}')

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

    end_mem = round((df.memory_usage().sum() / 1024**2), 2)
    print(f'Memory usage after optimization is: {end_mem}')

    saved_mem_perc = 100 * ((start_mem - end_mem) / start_mem)
    print(f'Decreased by {saved_mem_perc}%')

    return df


FEATURE_COLS = [ 'assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'matchDuration', 'maxPlace',
       'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills',
       'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',
       'weaponsAcquired', 'winPoints']

DROP_COLS = ['Id', 'groupId', 'matchId', 'matchType']
TARGET = 'winPlacePerc'


def create_feature_columns(feature_names=FEATURE_COLS):
    feature_columns = []
    for feature in feature_names:
        feature_columns.append(numeric_column(
            key=feature,
            dtype=tf.float32))
    return feature_columns

def input_fn(X, y, batch_size=128, shuffle=False, test=False):
    dataset = Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
        dataset = dataset.shuffle(batch_size)
    dataset = dataset.batch(batch_size)
    return dataset

def build_BoostedTreesRegressor(feature_columns, data_len, batch_size=128):
    params = {
        'n_trees': 50,
        'max_depth': 13,
        'n_batches_per_layer': 1,
        'center_bias': True
    }
    return BoostedTreesRegressor(feature_columns, **params)

train = pd.read_csv('../input/train_V2.csv', nrows=500000)

train_X, valid_X, train_y, valid_y = load_data(train, TARGET, DROP_COLS)
train_X = scale_features(train_X, FEATURE_COLS, tanh_scalar)
valid_X = scale_features(valid_X, FEATURE_COLS, tanh_scalar)
train_X = reduce_mem(train_X)
valid_X = reduce_mem(valid_X)

fc = create_feature_columns()

estimator = build_BoostedTreesRegressor(fc,len(train_X))

train_input_fn = lambda: input_fn(train_X, train_y, shuffle=True)
valid_input_fn = lambda: input_fn(valid_X, valid_y)


for _ in range(10):
    estimator.train(train_input_fn, steps=10)



test = pd.read_csv('../input/test_V2.csv')

labels = test.pop('Id')
test_X = scale_features(test[FEATURE_COLS], FEATURE_COLS, tanh_scalar)
test_X = reduce_mem(test_X)

pred_input_fn = lambda: Dataset.from_tensors(dict(test_X))

pred_dicts = list(estimator.experimental_predict_with_explanations(pred_input_fn))

placements = pd.Series([p['predictions'][0] for p in pred_dicts])

submission = pd.DataFrame({'Id': labels.values, 'winPlacePerc': placements.values})
submission.to_csv('submission.csv', index=False)
