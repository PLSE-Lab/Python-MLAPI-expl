#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##### decorators.py #####

import functools
import time


class DecoratorInputOutputExceptionTime(object):
    def __call__(self, fn):
        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            try:
                print("{} ({} - {})".format(fn.__name__, args, kwargs))
                tic = time.time()
                result = fn(*args, **kwargs)
                toc = time.time()
                print("Result: {} [in {:.4f}s]".format(result, toc-tic))
                return result
            except Exception as ex:
                print("Exception {0}".format(ex))
                raise ex
        return decorated


# In[ ]:


##### myio.py #####
import pathlib
import os
import pandas as pd
import pickle

train_file_path = '/kaggle/input/covid19-global-forecasting-week-4/train.csv'
test_file_path = '/kaggle/input/covid19-global-forecasting-week-4/test.csv'
enrichment_file_path = '/kaggle/input/covid-19-enriched-dataset-week-2/enriched_covid_19_week_2.csv'


def cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['location'] = df.Country_Region + '_' + df.Province_State
    df.location = df.location.astype('category')
    df = df.drop(columns=['Country_Region', 'Province_State'])
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    df.Date = pd.to_datetime(df.Date)
    return df


def read_enrichment_file() -> pd.DataFrame:
    df = pd.read_csv(enrichment_file_path, na_filter=False)
    # Fix the representation of Country_Region colunn
    df['Country_Region'] = df['Country_Region'].apply(lambda val: val.split('_')[0])
    df = cast_columns(df)

    enriched_df = df.sort_values('Date').groupby('location').last().reset_index().dropna()
    enriched_df = enriched_df.drop(columns=["ConfirmedCases", "Fatalities", "Date", 'restrictions', 'quarantine', 'schools'])

    # Fill the missing rows with avg values
    avg_row = enriched_df.mean().to_dict()
    train_df = read_train_file()
    train_df_locations = train_df.groupby('location').last().reset_index().dropna()['location']

    joined_df = pd.merge(train_df_locations, enriched_df['location'], indicator=True, on='location', how='left')
    for _, row in joined_df[joined_df._merge != 'both'].iterrows():
        avg_row['location'] = row.location
        enriched_df = enriched_df.append(pd.Series(avg_row), ignore_index=True)
    assert len(enriched_df) == len(train_df_locations)
    return enriched_df


def read_train_file() -> pd.DataFrame:
    df = pd.read_csv(train_file_path, na_filter=False)
    df = cast_columns(df)
    return df


def read_test_file() -> pd.DataFrame:
    df = pd.read_csv(test_file_path, na_filter=False)
    df = cast_columns(df)
    return df


def get_train_subset() -> pd.DataFrame:
    # before 2020-04-01
    df = read_train_file()
    df = df[df.Date < "2020-04-01"].reset_index(drop=True).copy(deep=True)
    return df


def get_validation_subset() -> pd.DataFrame:
    # 2020-04-01 to 2020-04-15 (included)
    df = read_train_file()
    df = df[(df.Date >= "2020-04-01") & (df.Date <= "2020-04-15")].reset_index(drop=True).copy(deep=True)
    return df


def save_pickle(obj: object, file_path: str) -> None:
    with open(file_path, "wb") as pk:
        pickle.dump(obj, pk)


def load_pickle(file_path: str) -> object:
    # May rise IOError if the file is not there
    with open(file_path, "rb") as pk:
        return pickle.load(pk)


# In[ ]:


##### featurization.py #####

import random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple, Dict


def build_point_feats_label(df: pd.DataFrame, lookback: int) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    assert len(df) == lookback+1
    points = [(row.ConfirmedCases, row.Fatalities,) for _, row in df.iterrows()]
    return points[:-1], points[-1]


def build_point_feats(df: pd.DataFrame) -> List[Tuple[float, float]]:
    return [(row.ConfirmedCases, row.Fatalities,) for _, row in df.iterrows()]


def create_timelines(dataset: pd.DataFrame, labels_from: datetime, lookback: int) -> Dict[str, List]:
    timelines = {}
    for loc in dataset.location.unique():
        sorted_points_for_location = dataset[dataset.location == loc].sort_values(by='Date')
        timelines[loc] = []
        for label_idx in range(len(sorted_points_for_location)-1, lookback+1, -1):
            label_date = sorted_points_for_location.iloc[label_idx].Date
            if label_date < labels_from:
                break
            timeline = sorted_points_for_location.iloc[label_idx-lookback: label_idx+1]
            point = build_point_feats_label(timeline, lookback)

            timelines[loc].append(point)
    return timelines


def create_start_timelines_for_val_test(train_dataset: pd.DataFrame, test_val_dataset: pd.DataFrame,
                                        lookback: int) -> Dict[str, List[Tuple[float, float]]]:
    # 1 starting point for each country
    start_timeline = {}
    for loc in test_val_dataset.location.unique():
        first_date = test_val_dataset[test_val_dataset.location == loc].sort_values(by='Date').iloc[0].Date
        sorted_points_for_location = train_dataset[train_dataset.location == loc].sort_values(by='Date')
        point_feats = build_point_feats(sorted_points_for_location.iloc[-lookback-1:-1])
        start_timeline[loc] = point_feats
    return start_timeline


def create_contextual_features(enrichment_df: pd.DataFrame) -> Dict[str, List]:
    feats = {}
    for _, row in enrichment_df.iterrows():
        feat = row.tolist()
        feats[row.location] = feat[1:]
    return feats


def location_label_encoder_dict(list_of_locations: List[str]) -> Dict[str, int]:
    return {loc: idx for idx, loc in enumerate(list_of_locations)}


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[str(col) + '_sqrt'] = df[col].pow(1./2)
        df[str(col) + '_log2'] = np.log2(df[col].values + 1)
    return df


# In[ ]:


##### modeling.py #####
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Input, Concatenate, Dropout, Reshape, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def create_model(cat_df: pd.DataFrame, ctx_df: pd.DataFrame, tl_df: pd.DataFrame, labels_df: pd.DataFrame) ->        Tuple[keras.Model, StandardScaler, StandardScaler]:

    ctx_ss = StandardScaler()
    tl_ss = StandardScaler()

    X_train_cat = shuffle(cat_df.values, random_state=0)
    X_train_ctx = shuffle(ctx_ss.fit_transform(ctx_df.values), random_state=0)
    X_train_tl = shuffle(tl_ss.fit_transform(tl_df.values), random_state=0)
    Y_train = shuffle(labels_df.values, random_state=0)

    cat_inputs = Input(shape=(X_train_cat.shape[1], ))
    cat_embeddings = Embedding(len(np.unique(X_train_cat)), 32)(cat_inputs)
    modifier_cat_embeddings = Reshape(target_shape=(32,))(cat_embeddings)
    cat_output = Dense(64, activation='relu')(modifier_cat_embeddings)

    ctx_inputs = Input(shape=(X_train_ctx.shape[1], ))
    ctx_dense = Dense(512, activation='relu')(ctx_inputs)

    tl_inputs = Input(shape=(X_train_tl.shape[1],))
    tl_dense = Dense(512, activation='relu')(tl_inputs)

    merge_layer = Concatenate(axis=-1)([cat_output, ctx_dense, tl_dense])
    normalization_layer = BatchNormalization()(merge_layer)
    merge_dropout_layer = Dropout(0.2)(normalization_layer)
    dense_layer = Dense(512, activation='relu')(merge_dropout_layer)
    normalization_layer_2 = BatchNormalization()(dense_layer)
    dropout_layer = Dropout(0.2)(normalization_layer_2)
    embedding_layer = Dense(256, activation='relu')(dropout_layer)

    cases = Dense(1, activation='relu')(embedding_layer)
    fatalities = Dense(1, activation='relu')(embedding_layer)

    model = Model(inputs=[cat_inputs, ctx_inputs, tl_inputs], outputs=[cases, fatalities])
    model.compile(loss=[tf.keras.losses.MeanSquaredLogarithmicError(),
                        tf.keras.losses.MeanSquaredLogarithmicError()],
                  optimizer=Nadam())

    print(model.summary())
    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=1, factor=0.8),
                 EarlyStopping(monitor='val_loss', patience=20),
                 ModelCheckpoint(filepath='/tmp/keras_covid_w4_best_model.h5', monitor='val_loss', save_best_only=True)]

    train_history = model.fit([X_train_cat, X_train_ctx, X_train_tl], [Y_train[:,0], Y_train[:,1]],
                              epochs=500,
                              batch_size=1024,  # with BatchNormalization, should be large enough
                              validation_split=0.1,
                              callbacks=callbacks,
                              shuffle=True,
                              use_multiprocessing=True)

    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('loss per epoch')
    plt.ylabel('overall loss')
    plt.xlabel('# epochs')
    plt.legend(['train', 'val'])
    plt.show()
    return model, ctx_ss, tl_ss

def apply_model(cat_df: pd.DataFrame, ctx_df: pd.DataFrame, tl_df: pd.DataFrame, model: keras.Model,
                ctx_ss: StandardScaler, tl_ss: StandardScaler) -> pd.DataFrame:

    X_cat = cat_df.values
    X_ctx = ctx_ss.transform(ctx_df.values)
    X_tl = tl_ss.transform(tl_df.values)

    Y_pred = model.predict([X_cat, X_ctx, X_tl])
    return pd.DataFrame(list(zip(Y_pred[0].ravel().tolist(), Y_pred[1].ravel().tolist())), columns=['ConfirmedCases', 'Fatalities'])


# In[ ]:


##### main.py #####

import datetime
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
# from myio import get_train_subset, get_validation_subset, read_test_file, read_enrichment_file
# from featurization import (create_timelines, create_start_timelines_for_val_test, create_contextual_features,
#                            location_label_encoder_dict, generate_features)
# from modeling import create_model, apply_model
pd.options.display.max_columns = 20
pd.set_option('max_colwidth', 100)


def create_features_labels(train_tl: Dict[str, list], contextual_feats: Dict[str, list], le: Dict[str, int]) -> List[pd.DataFrame]:
    cat_features, ctx_features, tl_feats, labels = [], [], [], []
    for location, train_label_tl in train_tl.items():
        le_location = le[location]
        ctx_feats = contextual_feats[location]
        for (feat, label) in train_label_tl:
            cat_features.append([le_location])
            ctx_features.append(ctx_feats)
            tl_feats.append([f for ft in feat for f in ft])
            labels.append([label[0], label[1]])

    return [pd.DataFrame.from_records(cat_features),
            generate_features(pd.DataFrame.from_records(ctx_features)),
            generate_features(pd.DataFrame.from_records(tl_feats)),
            pd.DataFrame.from_records(labels)]


def create_val_test_start_features(val_test_tl: Dict[str, list], contextual_feats: Dict[str, list], le: Dict[str, int]) -> List[pd.DataFrame]:
    cat_features, ctx_features, tl_feats = [], [], []
    for location, feat in val_test_tl.items():
        cat_features.append([le[location]])
        ctx_features.append(contextual_feats[location])
        tl_feats.append([f for ft in feat for f in ft])
    return [pd.DataFrame.from_records(cat_features),
            generate_features(pd.DataFrame.from_records(ctx_features)),
            generate_features(pd.DataFrame.from_records(tl_feats))]


def add_new_point_to_tl(val_test_tl: Dict[str, list], prediction: pd.DataFrame) -> Dict[str, list]:
    for idx, location in enumerate(val_test_tl):
        feat = val_test_tl[location]
        new_point = (np.around(prediction.iloc[idx].ConfirmedCases), np.around(prediction.iloc[idx].Fatalities))
        new_feat = feat[1:] + [new_point]
        val_test_tl[location] = new_feat
    return val_test_tl


def msle(y_true: np.array, y_pred: np.array):
    y_pred_clip = y_pred.clip(min=0.0)
    return np.mean((np.log(y_true + 1) - np.log(y_pred_clip + 1)) ** 2)


# In[ ]:


train_dataset = get_train_subset()
validation_dataset = get_validation_subset()
test_dataset = read_test_file()
enrichment_dataset = read_enrichment_file()
LOOKBACK = 28

train_tl = create_timelines(train_dataset, datetime.datetime(2020, 3, 10), LOOKBACK)
val_tl = create_start_timelines_for_val_test(train_dataset, validation_dataset, LOOKBACK)
test_tl = create_start_timelines_for_val_test(train_dataset, test_dataset, LOOKBACK)
contextual_feats = create_contextual_features(enrichment_dataset)


# In[ ]:


location_le = location_label_encoder_dict(sorted(list(train_tl.keys())))

cat_features_train, ctx_features_train, tl_feats_train, labels_train = create_features_labels(train_tl, contextual_feats, location_le)
model, ctx_ss, tl_ss = create_model(cat_features_train, ctx_features_train, tl_feats_train, labels_train)


# In[ ]:



validation_dataset["predict_ConfirmedCases"] = np.nan
validation_dataset["predict_Fatalities"] = np.nan
cat_features_val, ctx_features_val, tl_feats_val = create_val_test_start_features(val_tl, contextual_feats, location_le)
for date in validation_dataset.Date.unique():
    print(date)
    y_pred = apply_model(cat_features_val, ctx_features_val, tl_feats_val, model, ctx_ss, tl_ss)
    val_tl = add_new_point_to_tl(val_tl, y_pred)
    cat_features_val, ctx_features_val, tl_feats_val = create_val_test_start_features(val_tl, contextual_feats, location_le)
    for idx, location in enumerate(validation_dataset.location.unique()):
        validation_dataset.loc[(validation_dataset.Date == date) & (validation_dataset.location == location), 'predict_ConfirmedCases'] = y_pred.iloc[idx].ConfirmedCases
        validation_dataset.loc[(validation_dataset.Date == date) & (validation_dataset.location == location), 'predict_Fatalities'] = y_pred.iloc[idx].Fatalities


val_MSLE_confirmed_cases = msle(validation_dataset.ConfirmedCases.values, validation_dataset.predict_ConfirmedCases.values)
val_MSLE_fatalities = msle(validation_dataset.Fatalities.values, validation_dataset.predict_Fatalities.values)
print('L1={:.3f}, L2={:.3f} -> L1+L2={:.3f}'.format(val_MSLE_confirmed_cases, val_MSLE_fatalities, val_MSLE_confirmed_cases+val_MSLE_fatalities))


# In[ ]:



test_dataset["predict_ConfirmedCases"] = np.nan
test_dataset["predict_Fatalities"] = np.nan
cat_features_test, ctx_features_test, tl_feats_test = create_val_test_start_features(test_tl, contextual_feats, location_le)
for date in test_dataset.Date.unique():
    print(date)
    y_pred = apply_model(cat_features_test, ctx_features_test, tl_feats_test, model, ctx_ss, tl_ss)
    test_tl = add_new_point_to_tl(test_tl, y_pred)
    cat_features_test, ctx_features_test, tl_feats_test = create_val_test_start_features(test_tl, contextual_feats, location_le)
    for idx, location in enumerate(test_dataset.location.unique()):
        test_dataset.loc[(test_dataset.Date == date) & (test_dataset.location == location), 'predict_ConfirmedCases'] = y_pred.iloc[idx].ConfirmedCases
        test_dataset.loc[(test_dataset.Date == date) & (test_dataset.location == location), 'predict_Fatalities'] = y_pred.iloc[idx].Fatalities

submission_df = test_dataset.rename(columns={"predict_ConfirmedCases": "ConfirmedCases", "predict_Fatalities": "Fatalities"})[['ForecastId', 'ConfirmedCases', 'Fatalities']]
submission_df.to_csv('submission.csv', header=True, index=False)


# In[ ]:




