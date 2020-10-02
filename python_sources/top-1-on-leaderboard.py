#I build 3 models: a Gradient Boosting, a CNN+DNN and a seq2seq RNN model. Final model was a weighted average of these models (where each model is stabilized by training multiple times with different random seeds then take the average). Each model separately can stay in top 1% in the final ranking.
# Model 1 gbm
from datetime import date, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import gc

from Utils import *

df_2017, promo_2017, items = load_unstack('1617')

promo_2017 = promo_2017[df_2017[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
df_2017 = df_2017[df_2017[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
promo_2017 = promo_2017.astype('int')
df_test = pd.read_csv("test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                      parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])
item_nbr_test = df_test.index.get_level_values(1)
item_nbr_train = df_2017.index.get_level_values(1)
item_inter = list(set(item_nbr_train).intersection(set(item_nbr_test)))

df_2017 = df_2017.loc[df_2017.index.get_level_values(1).isin(item_inter)]
promo_2017 = promo_2017.loc[promo_2017.index.get_level_values(1).isin(item_inter)]


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(t2017, is_train=True, one_hot=False):
    X = pd.DataFrame({
        "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
        "day_2_2017": get_timespan(df_2017, t2017, 2, 1).values.ravel(),
        "day_3_2017": get_timespan(df_2017, t2017, 3, 1).values.ravel(),
#         "day_4_2017": get_timespan(df_2017, t2017, 4, 1).values.ravel(),
#         "day_5_2017": get_timespan(df_2017, t2017, 5, 1).values.ravel(),
#         "day_6_2017": get_timespan(df_2017, t2017, 6, 1).values.ravel(),
#         "day_7_2017": get_timespan(df_2017, t2017, 7, 1).values.ravel(),
#         "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
#         "std_7_2017": get_timespan(df_2017, t2017, 7, 7).std(axis=1).values,
#         "max_7_2017": get_timespan(df_2017, t2017, 7, 7).max(axis=1).values,
#         "median_7_2017": get_timespan(df_2017, t2017, 7, 7).median(axis=1).values,
#         "median_30_2017": get_timespan(df_2017, t2017, 30, 30).median(axis=1).values,
#         "median_140_2017": get_timespan(df_2017, t2017, 140, 140).median(axis=1).values,
        'promo_3_2017': get_timespan(promo_2017, t2017, 3, 3).sum(axis=1).values,
        "last_year_mean": get_timespan(df_2017, t2017, 365, 16).mean(axis=1).values,
        "last_year_count0": (get_timespan(df_2017, t2017, 365, 16)==0).sum(axis=1).values,
        "last_year_promo": get_timespan(promo_2017, t2017, 365, 16).sum(axis=1).values
    })
    
    for i in [7, 14, 21, 30, 60, 90, 140, 365]:
        X['mean_{}_2017'.format(i)] = get_timespan(df_2017, t2017, i, i).mean(axis=1).values
        X['median_{}_2017'.format(i)] = get_timespan(df_2017, t2017, i, i).mean(axis=1).values
        X['max_{}_2017'.format(i)] = get_timespan(df_2017, t2017, i, i).max(axis=1).values
        X['mean_{}_haspromo_2017'.format(i)] = get_timespan(df_2017, t2017, i, i)[get_timespan(promo_2017, t2017, i, i)==1].mean(axis=1).values
        X['mean_{}_nopromo_2017'.format(i)] = get_timespan(df_2017, t2017, i, i)[get_timespan(promo_2017, t2017, i, i)==0].mean(axis=1).values
        X['count0_{}_2017'.format(i)] = (get_timespan(df_2017, t2017, i, i)==0).sum(axis=1).values
        X['promo_{}_2017'.format(i)] = get_timespan(promo_2017, t2017, i, i).sum(axis=1).values
        item_mean = get_timespan(df_2017, t2017, i, i).mean(axis=1).groupby('item_nbr').mean().to_frame('item_mean')
        X['item_{}_mean'.format(i)] = df_2017.join(item_mean)['item_mean'].values
        item_count0 = (get_timespan(df_2017, t2017, i, i)==0).sum(axis=1).groupby('item_nbr').mean().to_frame('item_count0')
        X['item_{}_count0_mean'.format(i)] = df_2017.join(item_count0)['item_count0'].values
        store_mean = get_timespan(df_2017, t2017, i, i).mean(axis=1).groupby('store_nbr').mean().to_frame('store_mean')
        X['store_{}_mean'.format(i)] = df_2017.join(store_mean)['store_mean'].values
        store_count0 = (get_timespan(df_2017, t2017, i, i)==0).sum(axis=1).groupby('store_nbr').mean().to_frame('store_count0')
        X['store_{}_count0_mean'.format(i)] = df_2017.join(store_count0)['store_count0'].values
        
    for i in range(7):
        X['mean_4_dow{}'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_10_dow{}'.format(i)] = get_timespan(df_2017, t2017, 70-i, 10, freq='7D').mean(axis=1).values
        X['count0_10_dow{}'.format(i)] = (get_timespan(df_2017, t2017, 70-i, 10)==0).sum(axis=1).values
        X['promo_10_dow{}'.format(i)] = get_timespan(promo_2017, t2017, 70-i, 10, freq='7D').sum(axis=1).values
        item_mean = get_timespan(df_2017, t2017, 70-i, 10, freq='7D').mean(axis=1).groupby('item_nbr').mean().to_frame('item_mean')
        X['item_mean_10_dow{}'.format(i)] = df_2017.join(item_mean)['item_mean'].values
        X['mean_20_dow{}'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values
        
    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[t2017 + timedelta(days=i)].values
    
    if one_hot:
        family_dummy = pd.get_dummies(df_2017.join(items)['family'], prefix='family')
        X = pd.concat([X, family_dummy.reset_index(drop=True)], axis=1)
        store_dummy = pd.get_dummies(df_2017.reset_index().store_nbr, prefix='store')
        X = pd.concat([X, store_dummy.reset_index(drop=True)], axis=1)
#         X['family_count'] = df_2017.join(items).groupby('family').count().iloc[:,0].values
#         X['store_count'] = df_2017.reset_index().groupby('family').count().iloc[:,0].values
    else:
        df_items = df_2017.join(items)
        df_stores = df_2017.join(stores)
        X['family'] = df_items['family'].astype('category').cat.codes.values
        X['perish'] = df_items['perishable'].values
        X['item_class'] = df_items['class'].values
        X['store_nbr'] = df_2017.reset_index().store_nbr.values
        X['store_cluster'] = df_stores['cluster'].values
        X['store_type'] = df_stores['type'].astype('category').cat.codes.values
#     X['item_nbr'] = df_2017.reset_index().item_nbr.values
#     X['item_mean'] = df_2017.join(item_mean)['item_mean']
#     X['store_mean'] = df_2017.join(store_mean)['store_mean']

#     store_promo_90_mean = get_timespan(promo_2017, t2017, 90, 90).sum(axis=1).groupby('store_nbr').mean().to_frame('store_promo_90_mean')
#     X['store_promo_90_mean'] = df_2017.join(store_promo_90_mean)['store_promo_90_mean'].values
#     item_promo_90_mean = get_timespan(promo_2017, t2017, 90, 90).sum(axis=1).groupby('item_nbr').mean().to_frame('item_promo_90_mean')
#     X['item_promo_90_mean'] = df_2017.join(item_promo_90_mean)['item_promo_90_mean'].values
    
    if is_train:
        y = df_2017[pd.date_range(t2017, periods=16)].values
        return X, y
    return X


print("Preparing dataset...")
X_l, y_l = [], []
t2017 = date(2017, 7, 5)
n_range = 14
for i in range(n_range):
    print(i, end='..')
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(t2017 - delta)
    X_l.append(X_tmp)
    y_l.append(y_tmp)
    
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
X_val, y_val = prepare_dataset(date(2017, 7, 26))
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)

params = {
    'num_leaves': 31,
    'objective': 'regression',
    'min_data_in_leaf': 300,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'l2',
    'max_bin':128,
    'num_threads': 8
}

print("Training and predicting models...")
MAX_ROUNDS = 700
val_pred = []
test_pred = []
# best_rounds = []
cate_vars = ['family', 'perish', 'store_nbr', 'store_cluster', 'store_type']
w = (X_val["perish"] * 0.25 + 1) / (X_val["perish"] * 0.25 + 1).mean()

for i in range(16):

    print("Step %d" % (i+1))

    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=None)
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        weight=w,
        categorical_feature=cate_vars)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], verbose_eval=100)

    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True)[:15]))
    best_rounds.append(bst.best_iteration or MAX_ROUNDS)

    val_pred.append(bst.predict(X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    test_pred.append(bst.predict(X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))
    gc.collect();

cal_score(y_val, np.array(val_pred).T)

make_submission(df_2017, np.array(test_pred).T)


# Using CNN

import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras import optimizers
import gc

from Utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tf warnings

df, promo_df, items, stores = load_unstack('all')

# data after 2015
df = df[pd.date_range(date(2015,6,1), date(2017,8,15))]
promo_df = promo_df[pd.date_range(date(2015,6,1), date(2017,8,31))]

promo_df = promo_df[df[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
df = df[df[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
promo_df = promo_df.astype('int')

df_test = pd.read_csv("test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                      parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])
item_nbr_test = df_test.index.get_level_values(1)
item_nbr_train = df.index.get_level_values(1)
item_inter = list(set(item_nbr_train).intersection(set(item_nbr_test)))
df = df.loc[df.index.get_level_values(1).isin(item_inter)]
promo_df = promo_df.loc[promo_df.index.get_level_values(1).isin(item_inter)]

df_index = df.index
del item_nbr_test, item_nbr_train, item_inter, df_test; gc.collect()

timesteps = 200

# preparing data
train_data = train_generator(df, promo_df, items, stores, timesteps, date(2017, 7, 5),
                                           n_range=16, day_skip=7, batch_size=2000, aux_as_tensor=False, reshape_output=2)
Xval, Yval = create_dataset(df, promo_df, items, stores, timesteps, date(2017, 7, 26),
                                     aux_as_tensor=False, reshape_output=2)
Xtest, _ = create_dataset(df, promo_df, items, stores, timesteps, date(2017, 8, 16),
                                    aux_as_tensor=False, is_train=False, reshape_output=2)

w = (Xval[7][:, 2] * 0.25 + 1) / (Xval[7][:, 2] * 0.25 + 1).mean() # validation weight: 1.25 if perishable and 1 otherwise per competition rules

del df, promo_df; gc.collect()

print('current no promo 2') # log info

latent_dim = 32

# Define input
# seq input
seq_in = Input(shape=(timesteps, 1))
is0_in = Input(shape=(timesteps, 1))
promo_in = Input(shape=(timesteps+16, 1))
yearAgo_in = Input(shape=(timesteps+16, 1))
quarterAgo_in = Input(shape=(timesteps+16, 1))
item_mean_in = Input(shape=(timesteps, 1))
store_mean_in = Input(shape=(timesteps, 1))
# store_family_mean_in = Input(shape=(timesteps, 1))
weekday_in = Input(shape=(timesteps+16,), dtype='uint8')
weekday_embed_encode = Embedding(7, 4, input_length=timesteps+16)(weekday_in)
# weekday_embed_decode = Embedding(7, 4, input_length=timesteps+16)(weekday_in)
dom_in = Input(shape=(timesteps+16,), dtype='uint8')
dom_embed_encode = Embedding(31, 4, input_length=timesteps+16)(dom_in)
# dom_embed_decode = Embedding(31, 4, input_length=timesteps+16)(dom_in)
# weekday_onehot = Lambda(K.one_hot, arguments={'num_classes': 7}, output_shape=(timesteps+16, 7))(weekday_in)

# aux input
cat_features = Input(shape=(6,))
item_family = Lambda(lambda x: x[:, 0, None])(cat_features)
item_class = Lambda(lambda x: x[:, 1, None])(cat_features)
item_perish = Lambda(lambda x: x[:, 2, None])(cat_features)
store_nbr = Lambda(lambda x: x[:, 3, None])(cat_features)
store_cluster = Lambda(lambda x: x[:, 4, None])(cat_features)
store_type = Lambda(lambda x: x[:, 5, None])(cat_features)

# store_in = Input(shape=(timesteps+16,), dtype='uint8')
family_embed = Embedding(33, 8, input_length=1)(item_family)
# class_embed = Embedding(337, 8, input_length=1)(item_class)
store_embed = Embedding(54, 8, input_length=1)(store_nbr)
cluster_embed = Embedding(17, 3, input_length=1)(store_cluster)
type_embed = Embedding(5, 2, input_length=1)(store_type)

encode_slice = Lambda(lambda x: x[:, :timesteps, :])
# encode_features = concatenate([promo_in, yearAgo_in, quarterAgo_in], axis=2)
# encode_features = encode_slice(encode_features)

x_in = concatenate([seq_in, encode_slice(promo_in), item_mean_in], axis=2)

# Define network
# c0 = TimeDistributed(Dense(4))(x_in)
# # c0 = Conv1D(4, 1, activation='relu')(sequence_in)
c1 = Conv1D(latent_dim, 2, dilation_rate=1, padding='causal', activation='relu')(x_in)
c2 = Conv1D(latent_dim, 2, dilation_rate=2, padding='causal', activation='relu')(c1)
c2 = Conv1D(latent_dim, 2, dilation_rate=4, padding='causal', activation='relu')(c2)
c2 = Conv1D(latent_dim, 2, dilation_rate=8, padding='causal', activation='relu')(c2)
# c2 = Conv1D(latent_dim, 2, dilation_rate=16, padding='causal', activation='relu')(c2)

c4 = concatenate([c1, c2])
# c2 = MaxPooling1D()(c2)

conv_out = Conv1D(8, 1, activation='relu')(c4)
# conv_out = GlobalAveragePooling1D()(c4)
conv_out = Dropout(0.25)(conv_out)
conv_out = Flatten()(conv_out)

decode_slice = Lambda(lambda x: x[:, timesteps:, :])
promo_pred = decode_slice(promo_in)
# qAgo_pred = decode_slice(quarterAgo_in)
# yAgo_pred = decode_slice(yearAgo_in)


# Raw sequence in results overfitting!!!
dnn_out = Dense(512, activation='relu')(Flatten()(seq_in))
dnn_out = Dense(256, activation='relu')(dnn_out)
# dnn_out = BatchNormalization()(dnn_out)
dnn_out = Dropout(0.25)(dnn_out)

x = concatenate([conv_out, dnn_out,
                 Flatten()(promo_pred), Flatten()(family_embed), Flatten()(store_embed), Flatten()(cluster_embed), Flatten()(type_embed), item_perish])
# x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.25)(x)
# x = Dense(256, activation='relu')(x)
# x = BatchNormalization()(x)
# x = concatenate([x, seq_in])
output = Dense(16, activation='relu')(x)

model = Model([seq_in, is0_in, promo_in, yearAgo_in, quarterAgo_in, weekday_in, dom_in, cat_features, item_mean_in, store_mean_in], output)

# rms = optimizers.RMSprop(lr=0.002)
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit_generator(train_data, steps_per_epoch=1000, workers=4, use_multiprocessing=True, epochs=10, verbose=2,
                    validation_data=(Xval, Yval, w))
                    

val_pred = model.predict(Xval)
cal_score(Yval, val_pred)

test_pred = model.predict(Xtest)
make_submission(df_index, test_pred, 'cnn_no-promo2.csv')
# gc.collect()

# model.save('save_models/cnn_model')

# Using seq to seq

import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn import metrics
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras import optimizers
import gc

from Utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tf warnings

timesteps = 365

df, promo_df, items, stores = load_unstack('all')

# data after 2015
df = df[pd.date_range(date(2014,6,1), date(2017,8,15))]
promo_df = promo_df[pd.date_range(date(2014,6,1), date(2017,8,31))]

promo_df = promo_df[df[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
df = df[df[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
promo_df = promo_df.astype('int')

df_test = pd.read_csv("test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                      parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])
item_nbr_test = df_test.index.get_level_values(1)
item_nbr_train = df.index.get_level_values(1)
item_inter = list(set(item_nbr_train).intersection(set(item_nbr_test)))
df = df.loc[df.index.get_level_values(1).isin(item_inter)]
promo_df = promo_df.loc[promo_df.index.get_level_values(1).isin(item_inter)]

df_index = df.index
del item_nbr_test, item_nbr_train, item_inter, df_test; gc.collect()

train_data = train_generator(df, promo_df, items, stores, timesteps, date(2017, 7, 9),
                                           n_range=380, day_skip=1, batch_size=1000, aux_as_tensor=True, reshape_output=2)
Xval, Yval = create_dataset(df, promo_df, items, stores, timesteps, date(2017, 7, 26),
                                     aux_as_tensor=True, reshape_output=2)
Xtest, _ = create_dataset(df, promo_df, items, stores, timesteps, date(2017, 8, 16),
                                    aux_as_tensor=True, is_train=False, reshape_output=2)

w = (Xval[7][:, 0, 2] * 0.25 + 1) / (Xval[7][:, 0, 2] * 0.25 + 1).mean()

del df, promo_df; gc.collect()

# Note
# current best: add item_mean, dim: 50, all as tensor ~ 3500 (~3630 in new cv)
print('1*100, train on private 7, nrange 380 timestep 200, data 1000*1500 \n')

latent_dim = 100

# seq input
seq_in = Input(shape=(timesteps, 1))
is0_in = Input(shape=(timesteps, 1))
promo_in = Input(shape=(timesteps+16, 1))
yearAgo_in = Input(shape=(timesteps+16, 1))
quarterAgo_in = Input(shape=(timesteps+16, 1))
item_mean_in = Input(shape=(timesteps, 1))
store_mean_in = Input(shape=(timesteps, 1))
# store_family_mean_in = Input(shape=(timesteps, 1))
weekday_in = Input(shape=(timesteps+16,), dtype='uint8')
weekday_embed_encode = Embedding(7, 4, input_length=timesteps+16)(weekday_in)
# weekday_embed_decode = Embedding(7, 4, input_length=timesteps+16)(weekday_in)
dom_in = Input(shape=(timesteps+16,), dtype='uint8')
dom_embed_encode = Embedding(31, 4, input_length=timesteps+16)(dom_in)
# dom_embed_decode = Embedding(31, 4, input_length=timesteps+16)(dom_in)
# weekday_onehot = Lambda(K.one_hot, arguments={'num_classes': 7}, output_shape=(timesteps+16, 7))(weekday_in)

# aux input
cat_features = Input(shape=(timesteps+16, 6))
item_family = Lambda(lambda x: x[:, :, 0])(cat_features)
item_class = Lambda(lambda x: x[:, :, 1])(cat_features)
item_perish = Lambda(lambda x: x[:, :, 2])(cat_features)
store_nbr = Lambda(lambda x: x[:, :, 3])(cat_features)
store_cluster = Lambda(lambda x: x[:, :, 4])(cat_features)
store_type = Lambda(lambda x: x[:, :, 5])(cat_features)

# store_in = Input(shape=(timesteps+16,), dtype='uint8')
family_embed = Embedding(33, 8, input_length=timesteps+16)(item_family)
class_embed = Embedding(337, 8, input_length=timesteps+16)(item_class)
store_embed = Embedding(54, 8, input_length=timesteps+16)(store_nbr)
cluster_embed = Embedding(17, 3, input_length=timesteps+16)(store_cluster)
type_embed = Embedding(5, 2, input_length=timesteps+16)(store_type)

# Encoder
encode_slice = Lambda(lambda x: x[:, :timesteps, :])
encode_features = concatenate([promo_in, yearAgo_in, quarterAgo_in, weekday_embed_encode,
                               family_embed, Reshape((timesteps+16,1))(item_perish), store_embed, cluster_embed, type_embed], axis=2)
encode_features = encode_slice(encode_features)

# conv_in = Conv1D(8, 5, padding='same')(concatenate([seq_in, encode_features], axis=2))
# conv_raw = concatenate([seq_in, encode_slice(quarterAgo_in), encode_slice(yearAgo_in), item_mean_in], axis=2)
# conv_in = Conv1D(8, 5, padding='same')(conv_raw)
conv_in = Conv1D(4, 5, padding='same')(seq_in)
# conv_in_deep = Conv1D(2, 2, padding='causal', dilation_rate=1)(seq_in)
# conv_in_deep = Conv1D(2, 2, padding='causal', dilation_rate=2)(conv_in_deep)
# conv_in_deep = Conv1D(2, 2, padding='causal', dilation_rate=4)(conv_in_deep)
# conv_in_deep = Conv1D(2, 2, padding='causal', dilation_rate=8)(conv_in_deep)
# conv_in_quarter = Conv1D(4, 5, padding='same')(encode_slice(quarterAgo_in))
# conv_in_year = Conv1D(4, 5, padding='same')(encode_slice(yearAgo_in))
# conv_in = concatenate([conv_in_seq, conv_in_deep, conv_in_quarter, conv_in_year])

x_encode = concatenate([seq_in, encode_features, conv_in, item_mean_in], axis=2)
                        # store_mean_in, is0_in, store_family_mean_in], axis=2)
# encoder1 = CuDNNGRU(latent_dim, return_state=True, return_sequences=True)
# encoder2 = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
# encoder3 = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
encoder = CuDNNGRU(latent_dim, return_state=True)
print('Input dimension:', x_encode.shape)
_, h= encoder(x_encode)
# s1, h1 = encoder1(x_encode)
# s1 = Dropout(0.25)(s1)
# s2, h2 = encoder2(s1)
# _, h3 = encoder3(s2)

# Connector
h = Dense(latent_dim, activation='tanh')(h)
# h1 = Dense(latent_dim, activation='tanh')(h1)
# h2 = Dense(latent_dim, activation='tanh')(h2)

# Decoder
previous_x = Lambda(lambda x: x[:, -1, :])(seq_in)

decode_slice = Lambda(lambda x: x[:, timesteps:, :])
decode_features = concatenate([promo_in, yearAgo_in, quarterAgo_in, weekday_embed_encode,
                               family_embed, Reshape((timesteps+16,1))(item_perish), store_embed, cluster_embed, type_embed], axis=2)
decode_features = decode_slice(decode_features)

# decode_idx_train = np.tile(np.arange(16), (Xtrain.shape[0], 1))
# decode_idx_val = np.tile(np.arange(16), (Xval.shape[0], 1))
# decode_idx = Input(shape=(16,))
# decode_id_embed = Embedding(16, 4, input_length=16)(decode_idx)
# decode_features = concatenate([decode_features, decode_id_embed])

# aux_features = concatenate([dom_embed_decode, store_embed_decode, family_embed_decode], axis=2)
# aux_features = decode_slice(aux_features)

# decoder1 = CuDNNGRU(latent_dim, return_state=True, return_sequences=True)
# decoder2 = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
# decoder3 = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
decoder = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
# decoder_dense1 = Dense(128, activation='relu')
decoder_dense2 = Dense(1, activation='relu')
# dp = Dropout(0.25)
slice_at_t = Lambda(lambda x: tf.slice(x, [0,i,0], [-1,1,-1]))
for i in range(16):
    previous_x = Reshape((1,1))(previous_x)
    
    features_t = slice_at_t(decode_features)
    # aux_t = slice_at_t(aux_features)

    decode_input = concatenate([previous_x, features_t], axis=2)
    # output_x, h1 = decoder1(decode_input, initial_state=h1)
    # output_x = dp(output_x)
    # output_x, h2 = decoder2(output_x, initial_state=h2)
    # output_x, h3 = decoder3(output_x, initial_state=h3)
    output_x, h = decoder(decode_input, initial_state=h)
    # aux input
    # output_x = concatenate([output_x, aux_t], axis=2)
    # output_x = Flatten()(output_x)
    # decoder_dense1 = Dense(64, activation='relu')
    # output_x = decoder_dense1(output_x)
    # output_x = dp(output_x)
    output_x = decoder_dense2(output_x)

    # gather outputs
    if i == 0: decoder_outputs = output_x
    elif i > 0: decoder_outputs = concatenate([decoder_outputs, output_x])

    previous_x = output_x

model = Model([seq_in, is0_in, promo_in, yearAgo_in, quarterAgo_in, weekday_in, dom_in, cat_features, item_mean_in, store_mean_in], decoder_outputs)

# rms = optimizers.RMSprop(lr=0.002)
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit_generator(train_data, steps_per_epoch=1500, workers=5, use_multiprocessing=True, epochs=18, verbose=2,
                    validation_data=(Xval, Yval, w))

# val_pred = model.predict(Xval)
# cal_score(Yval, val_pred)

test_pred = model.predict(Xtest)
make_submission(df_index, test_pred, 'seq-private_only-7.csv')

# model.save('save_models/seq2seq_model-withput-promo-2')

# Averaging all the model 3 models
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from datetime import date, timedelta
import gc

def load_data():
    # df_train = pd.read_feather('train_after1608_raw')
    df_train = pd.read_csv('train.csv', usecols=[1, 2, 3, 4, 5], dtype={'onpromotion': bool},
                           converters={'unit_sales': lambda u: np.log1p(float(u)) if float(u) > 0 else 0},
                           parse_dates=["date"])
    df_test = pd.read_csv("test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                          parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])

    # subset data
    df_2017 = df_train.loc[df_train.date>=pd.datetime(2016,1,1)]

    # promo
    promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
    promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
    promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
    promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
    promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
    promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
    del promo_2017_test, promo_2017_train

    df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
    df_2017.columns = df_2017.columns.get_level_values(1)

    # items
    items = pd.read_csv("items.csv").set_index("item_nbr")
    stores = pd.read_csv("stores.csv").set_index("store_nbr")
    # items = items.reindex(df_2017.index.get_level_values(1))

    return df_2017, promo_2017, items, stores

def save_unstack(df, promo, filename):
    df_name, promo_name = 'df_' + filename + '_raw', 'promo_' + filename + '_raw'
    df.columns = df.columns.astype('str')
    df.reset_index().to_feather(df_name)
    promo.columns = promo.columns.astype('str')
    promo.reset_index().to_feather(promo_name)

def load_unstack(filename):
    df_name, promo_name = 'df_' + filename + '_raw', 'promo_' + filename + '_raw'
    df_2017 = pd.read_feather(df_name).set_index(['store_nbr','item_nbr'])
    df_2017.columns = pd.to_datetime(df_2017.columns)
    promo_2017 = pd.read_feather(promo_name).set_index(['store_nbr','item_nbr'])
    promo_2017.columns = pd.to_datetime(promo_2017.columns)
    items = pd.read_csv("items.csv").set_index("item_nbr")
    stores = pd.read_csv("stores.csv").set_index("store_nbr")

    return df_2017, promo_2017, items, stores

# Create validation and test data
def create_dataset(df, promo_df, items, stores, timesteps, first_pred_start, is_train=True, aux_as_tensor=False, reshape_output=0):
    encoder = LabelEncoder()
    items_reindex = items.reindex(df.index.get_level_values(1))
    item_family = encoder.fit_transform(items_reindex['family'].values)
    item_class = encoder.fit_transform(items_reindex['class'].values)
    item_perish = items_reindex['perishable'].values

    stores_reindex = stores.reindex(df.index.get_level_values(0))
    store_nbr = df.reset_index().store_nbr.values - 1
    store_cluster = stores_reindex['cluster'].values - 1
    store_type = encoder.fit_transform(stores_reindex['type'].values)

    # item_mean_df = df.groupby('item_nbr').mean().reindex(df.index.get_level_values(1))
    item_group_mean = df.groupby('item_nbr').mean()
    store_group_mean = df.groupby('store_nbr').mean()
    # store_family_group_mean = df.join(items['family']).groupby(['store_nbr', 'family']).transform('mean')
    # store_family_group_mean.index = df.index

    cat_features = np.stack([item_family, item_class, item_perish, store_nbr, store_cluster, store_type], axis=1)

    return create_dataset_part(df, promo_df, cat_features, item_group_mean, store_group_mean, timesteps, first_pred_start, reshape_output, aux_as_tensor, is_train)

def train_generator(df, promo_df, items, stores, timesteps, first_pred_start,
    n_range=1, day_skip=7, is_train=True, batch_size=2000, aux_as_tensor=False, reshape_output=0, first_pred_start_2016=None):
    encoder = LabelEncoder()
    items_reindex = items.reindex(df.index.get_level_values(1))
    item_family = encoder.fit_transform(items_reindex['family'].values)
    item_class = encoder.fit_transform(items_reindex['class'].values)
    item_perish = items_reindex['perishable'].values

    stores_reindex = stores.reindex(df.index.get_level_values(0))
    store_nbr = df.reset_index().store_nbr.values - 1
    store_cluster = stores_reindex['cluster'].values - 1
    store_type = encoder.fit_transform(stores_reindex['type'].values)

    # item_mean_df = df.groupby('item_nbr').mean().reindex(df.index.get_level_values(1))
    item_group_mean = df.groupby('item_nbr').mean()
    store_group_mean = df.groupby('store_nbr').mean()
    # store_family_group_mean = df.join(items['family']).groupby(['store_nbr', 'family']).transform('mean')
    # store_family_group_mean.index = df.index

    cat_features = np.stack([item_family, item_class, item_perish, store_nbr, store_cluster, store_type], axis=1)

    while 1:
        date_part = np.random.permutation(range(n_range))
        if first_pred_start_2016 is not None:
            range_diff = (first_pred_start - first_pred_start_2016).days / day_skip
            date_part = np.concat([date_part, np.random.permutation(range(range_diff, int(n_range/2) + range_diff))])

        for i in date_part:
            keep_idx = np.random.permutation(df.shape[0])[:batch_size]
            df_tmp = df.iloc[keep_idx,:]
            promo_df_tmp = promo_df.iloc[keep_idx,:]
            cat_features_tmp = cat_features[keep_idx]
            # item_mean_tmp = item_mean_df.iloc[keep_idx, :]

            pred_start = first_pred_start - timedelta(days=int(day_skip*i))

            # Generate a batch of random subset data. All data in the same batch are in the same period.
            yield create_dataset_part(df_tmp, promo_df_tmp, cat_features_tmp, item_group_mean, store_group_mean, timesteps, pred_start, reshape_output, aux_as_tensor, True)

            gc.collect()

def create_dataset_part(df, promo_df, cat_features, item_group_mean, store_group_mean, timesteps, pred_start, reshape_output, aux_as_tensor, is_train, weight=False):

    item_mean_df = item_group_mean.reindex(df.index.get_level_values(1))
    store_mean_df = store_group_mean.reindex(df.index.get_level_values(0))
    # store_family_mean_df = store_family_group_mean.reindex(df.index)

    X, y = create_xy_span(df, pred_start, timesteps, is_train)
    is0 = (X==0).astype('uint8')
    promo = promo_df[pd.date_range(pred_start-timedelta(days=timesteps), periods=timesteps+16)].values
    weekday = np.tile([d.weekday() for d in pd.date_range(pred_start-timedelta(days=timesteps), periods=timesteps+16)],
                          (X.shape[0],1))
    dom = np.tile([d.day-1 for d in pd.date_range(pred_start-timedelta(days=timesteps), periods=timesteps+16)],
                          (X.shape[0],1))
    item_mean, _ = create_xy_span(item_mean_df, pred_start, timesteps, False)
    store_mean, _ = create_xy_span(store_mean_df, pred_start, timesteps, False)
    # store_family_mean, _ = create_xy_span(store_family_mean_df, pred_start, timesteps, False)
    # month_tmp = np.tile([d.month-1 for d in pd.date_range(pred_start-timedelta(days=timesteps), periods=timesteps+16)],
    #                       (X_tmp.shape[0],1))
    yearAgo, _ = create_xy_span(df, pred_start-timedelta(days=365), timesteps+16, False)
    quarterAgo, _ = create_xy_span(df, pred_start-timedelta(days=91), timesteps+16, False)

    if reshape_output>0:
        X = X.reshape(-1, timesteps, 1)
    if reshape_output>1:
        is0 = is0.reshape(-1, timesteps, 1)
        promo = promo.reshape(-1, timesteps+16, 1)
        yearAgo = yearAgo.reshape(-1, timesteps+16, 1)
        quarterAgo = quarterAgo.reshape(-1, timesteps+16, 1)
        item_mean = item_mean.reshape(-1, timesteps, 1)
        store_mean = store_mean.reshape(-1, timesteps, 1)
        # store_family_mean = store_family_mean.reshape(-1, timesteps, 1)

    w = (cat_features[:, 2] * 0.25 + 1) / (cat_features[:, 2] * 0.25 + 1).mean()

    cat_features = np.tile(cat_features[:, None, :], (1, timesteps+16, 1)) if aux_as_tensor else cat_features

    # Use when only 6th-16th days (private periods) are in the training output
    # if is_train: y = y[:, 5:]

    if weight: return ([X, is0, promo, yearAgo, quarterAgo, weekday, dom, cat_features, item_mean, store_mean], y, w)
    else: return ([X, is0, promo, yearAgo, quarterAgo, weekday, dom, cat_features, item_mean, store_mean], y)

def create_xy_span(df, pred_start, timesteps, is_train=True, shift_range=0):
    X = df[pd.date_range(pred_start-timedelta(days=timesteps), pred_start-timedelta(days=1))].values
    if is_train: y = df[pd.date_range(pred_start, periods=16)].values
    else: y = None
    return X, y

# Not used in the final model
def random_shift_slice(mat, start_col, timesteps, shift_range):
    shift = np.random.randint(shift_range+1, size=(mat.shape[0],1))
    shift_window = np.tile(shift,(1,timesteps)) + np.tile(np.arange(start_col, start_col+timesteps),(mat.shape[0],1))
    rows = np.arange(mat.shape[0])
    rows = rows[:,None]
    columns = shift_window
    return mat[rows, columns]

# Calculate RMSE scores for all 16 days, first 5 days (fror public LB) and 6th-16th days (for private LB) 
def cal_score(Ytrue, Yfit):
	print([metrics.mean_squared_error(Ytrue, Yfit), 
	metrics.mean_squared_error(Ytrue[:,:5], Yfit[:,:5]),
	metrics.mean_squared_error(Ytrue[:,5:], Yfit[:,5:])])

# Create submission file
def make_submission(df_index, test_pred, filename):
	df_test = pd.read_csv("test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                      parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])
	df_preds = pd.DataFrame(
	    test_pred, index=df_index,
	    columns=pd.date_range("2017-08-16", periods=16)
	).stack().to_frame("unit_sales")
	df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

	submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
	submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
	submission.to_csv(filename, float_format='%.4f', index=None)

# if it is useful for you please vote 