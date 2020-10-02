#!/usr/bin/env python
# coding: utf-8

# #### > Inspired by: https://www.kaggle.com/frlemarchand/covid-19-forecasting-with-an-rnn
# > Requires an extra dataset: https://www.kaggle.com/optimo/covid19-enriched-dataset-week-2
# > 
# 
# ## Preprocessing
# 
# 1. Traininig data is read and put into 2 tables at random: train (80%) and validation (20%).
# 2. For each point, 40 (variable MAX_LOOKBACK) points of the past are collected (ConfirmedCases and Fatalities features). If data is not available, ConfirmedCases and Fatalities are both set to 0.
# 3. At random, between 0 and 14 points (variable CENSOR) of the most recent datapoints are deleted/censored. This is to make the classifier able to predict up to 14 days in the future.
# 4. Of the remanining datapoints, only 24 (variable N_POINTS_TO_KEEP) of them are kept ar random. This is to train the model to deal with missing information of the past.
# 5. Those 14 points are reshaped into an array, containing (days_in_the_past, confirmed_cases and fatalities)
#  * The feature vector looks like this:
#  * `[days_in_the_past_0=22, confirmed_cases_0=0, fatalities_0=0,`
#     `days_in_the_past_1=20, confirmed_cases_1=5, fatalities_1=0,`
#     `...`
#     `days_in_the_past_23=4, confirmed_cases_23=36, fatalities_23=5]`
# 6. The information is then joined with the location (i.e. the union of the features Country_Region and Province_State)
# 7. And the array is finally joined with the information coming from the enrichment table. This table is idexed by location, and provides a set of info connected to the location (age split, smokers percentage,...) see the enrichment dataset above for more info.
# 8. This process is done for all the points in the training set, validation set and testing set.
# 
# 
# ## Modeling (using TF/Keras)
# 
# 1. The only categorical feature is the location. It's label-encoded
# 2. The numerical features are all the others. To each numerical column, is created its sqrt and log2 version.
# 3. Numerical data goes into a StandardScaler, trained on the train dataset, and applied on the training, validation and test.
# 4. The Deep model is the following:
#  * The label-encoded categorical feature goes through a Embedding(out_size=64) and Dense(128)
#  * The numerical features goes into a Dense(1024), BatchNormalization, Dropout(0.2), Dense(256)
#  * The two output layers are Concatenate-d. Then it's applied BatchNormalization, Dropout(0.2), Dense(256) and a Dropout(0.2).
#  * Finally, the two predicted labels are two Dense(1) layers.
#  * All the Dense layers uses a relu activation. All of them.
#  * Optimiser=Adam, batch_size is 2048 (due to the BatchNormalization layers)
#  * Callbacks: ReduceLROnPlateau, EarlyStopping
# 
# ## Postprocessing
# 1. The prediction must be above the historical data (see point 3, preprocessing), so both predictions goes into a max(prediction, max(historical_data))
# 2. Predictions are written as csv
# 

# In[ ]:


# ====== decorators.py ======
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


# ====== io.py ======


import pathlib
import os
import pandas as pd
import pickle

train_file_path = '/kaggle/input/covid19-global-forecasting-week-3/train.csv'
test_file_path = '/kaggle/input/covid19-global-forecasting-week-3/test.csv'
submission_file_path = '/kaggle/input/covid19-global-forecasting-week-3/submission.csv'
my_submission_file_path = 'submission.csv'
enrichment_file_path = '/kaggle/input/covid-19-enriched-dataset-week-2/enriched_covid_19_week_2.csv'


def cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.Province_State = df.Province_State.astype('category')
    df.Country_Region = df.Country_Region.astype('category')
    df.Date = pd.to_datetime(df.Date)
    return df


def read_enrichment_file() -> pd.DataFrame:
    df = pd.read_csv(enrichment_file_path, na_filter=False)
    # Fix the representation of Country_Region colunn
    df['Country_Region'] = df['Country_Region'].apply(lambda val: val.split('_')[0])

    grouped_cols = ['Country_Region', 'Province_State']
    cast_columns(df)
    enriched_df = df.sort_values('Date').groupby(grouped_cols).last().reset_index().dropna()
    enriched_df = enriched_df.drop(columns=["Id", "ConfirmedCases", "Fatalities", "Date"])

    # Fill the missing rows with avg values
    avg_row = enriched_df.mean().to_dict()
    train_df = read_train_file()
    train_df_locations = train_df.groupby(grouped_cols).last().reset_index().dropna()[grouped_cols]

    joined_df = pd.merge(train_df_locations, enriched_df[grouped_cols], indicator=True, on=grouped_cols, how='left')
    for _, row in joined_df[joined_df._merge != 'both'].iterrows():
        avg_row['Country_Region'] = row.Country_Region
        avg_row['Province_State'] = row.Province_State
        enriched_df = enriched_df.append(pd.Series(avg_row), ignore_index=True)
    assert len(enriched_df) == len(train_df_locations)
    return enriched_df


def read_train_file() -> pd.DataFrame:
    df = pd.read_csv(train_file_path, na_filter=False)
    cast_columns(df)
    return df


def read_test_file() -> pd.DataFrame:
    df = pd.read_csv(test_file_path, na_filter=False)
    cast_columns(df)
    return df


def get_train_subset(pc: float) -> pd.DataFrame:
    df = read_train_file().sample(frac=pc).reset_index(drop=True).copy(deep=True)
    return df


def save_pickle(obj: object, file_path: str) -> None:
    with open(file_path, "wb") as pk:
        pickle.dump(obj, pk)


def load_pickle(file_path: str) -> object:
    # May rise IOError if the file is not there
    with open(file_path, "rb") as pk:
        return pickle.load(pk)


def write_submission_csv(submission_df: pd.DataFrame) -> None:
    df = submission_df[['ForecastId', 'ConfirmedCases', 'Fatalities']]
    df.to_csv(my_submission_file_path, header=True, index=False)


# In[ ]:


# ====== featurization.py ======
import random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple


def get_empty_point(point: pd.Series, date: datetime) -> pd.Series:
    new_point = point.copy(deep=True)
    new_point.Date = date
    new_point.Fatalities = 0.0
    new_point.ConfirmedCases = 0.0
    new_point.Id = -1
    return new_point


def get_points_in_the_past(point: pd.Series, max_lookback_days: int, train_points: pd.DataFrame) -> List[pd.Series]:
    history = []
    for i in range(1, max_lookback_days+1):
        date = point.Date - pd.DateOffset(days=i)
        past_point = train_points[(train_points.Province_State == point.Province_State) &                                   (train_points.Country_Region == point.Country_Region) &                                   (train_points.Date == date)]
        if len(past_point) == 1:
            history.append(past_point.iloc[0])
        else:
            history.append(get_empty_point(point, date))

    assert len(history) == max_lookback_days
    return history[::-1]


def randomize_the_points_and_censor_the_last_n(points: List[pd.Series],
                                               n_points_to_keep: int,
                                               censor_days: Optional[int],) -> List[Optional[pd.Series]]:
    if censor_days is None:
        censor_days = random.randint(0, 14)  # up to 30 days of censorship
    points = points[:len(points)-censor_days]
    if len(points) < n_points_to_keep:
        raise Exception('There are not enough points to choose from (after censorship)')

    sampled_points = random.sample(points, n_points_to_keep)
    assert len(sampled_points) == n_points_to_keep
    return sorted(sampled_points, key=lambda el: el.Date)


def to_features(point: pd.Series, history: List[pd.Series], is_test_set: bool) -> Tuple[str, pd.DataFrame, Tuple[float, float]]:
    province_state = point.Province_State
    country_region = point.Country_Region
    date = point.Date
    feats = []
    for p in history:
        point_dict = {
            'days_back': -1 * (p.Date - date).days,
            'confirmed_cases': p.ConfirmedCases,
            'fatalities': p.Fatalities
        }
        feats.append(point_dict)
    df = pd.DataFrame.from_records(feats)
    labels = (None, None) if is_test_set else (point.ConfirmedCases, point.Fatalities)
    return f"{province_state}{country_region}", df, labels


def get_enriched_row(point: pd.Series, enrichment_df: pd.DataFrame) -> pd.Series:
    row = enrichment_df[(enrichment_df.Province_State == point.Province_State) &                         (enrichment_df.Country_Region == point.Country_Region)].iloc[0]
    return row


def to_feat_row(feat: Tuple[str, pd.DataFrame, Optional[Tuple[float, float]]],
                enriched_row: pd.Series, row_id: Optional[int] = None) -> pd.Series:

    loc, df, labels = feat
    feat_row = {'location': loc, 'label_confirmed_cases': labels[0], 'label_fatalities': labels[1]}
    if row_id:
        feat_row['ForecastId'] = row_id

    for idx, row in df.iterrows():
        feat_row[f'days_back_{idx}'] = row.days_back
        feat_row[f'confirmed_cases_{idx}'] = row.confirmed_cases
        feat_row[f'fatalities_{idx}'] = row.fatalities

    for col in enriched_row.to_dict().keys():
        if col in ('Province_State', 'Country_Region'):
            continue
        feat_row[col] = enriched_row[col]

    return pd.Series(feat_row)


def generate_features(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numerical_columns = [col for col in train_df.columns if not col[:5] in ('label', 'locat')]
    for df in [train_df, validation_df, test_df]:
        for col in numerical_columns:
            df[col + '_sqrt'] = df[col].pow(1./2)
            df[col + '_log2'] = np.log2(df[col].values + 1)
    return train_df, validation_df, test_df


# In[ ]:


# ====== modeling.py ======

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Input, Concatenate, Dropout, Reshape, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler



def label_encode_columns(cols: List[str], df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) ->                             Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    les = {}
    for col in cols:
        le = LabelEncoder()
        df_train[col + '_encoded'] = le.fit_transform(df_train[col])
        df_val[col + '_encoded'] = le.transform(df_val[col])
        df_test[col + '_encoded'] = le.transform(df_test[col])
        les[col] = le
    return df_train, df_val, df_test, les


def create_model(cat_encoded_cols: List[str], numerical_cols: List[str], labels: List[str],
                 df_train: pd.DataFrame, df_val: pd.DataFrame, model_path: str) -> Tuple[keras.Model, StandardScaler]:

    X_train_cat = df_train[cat_encoded_cols].values
    X_train_num = df_train[numerical_cols].values
    Y_train = df_train[labels].values

    X_val_cat = df_val[cat_encoded_cols].values
    X_val_num = df_val[numerical_cols].values
    Y_val = df_val[labels].values

    # scaling numerical inputs
    ss = StandardScaler()
    X_train_num = ss.fit_transform(X_train_num)
    X_val_num = ss.transform(X_val_num)

    cat_distinct_values = len(df_train.groupby(cat_encoded_cols))

    cat_inputs = Input(shape=(len(cat_encoded_cols)))
    cat_embeddings = Embedding(cat_distinct_values, 64)(cat_inputs)
    modifier_cat_embeddings = Reshape(target_shape=(64,))(cat_embeddings)
    cat_output = Dense(128, activation='relu')(modifier_cat_embeddings)

    num_inputs = Input(shape=(len(numerical_cols)))
    num_dense = Dense(1024, activation='relu')(num_inputs)
    num_normalized = BatchNormalization()(num_dense)
    num_dropout = Dropout(0.2)(num_normalized)
    num_output = Dense(256, activation='relu')(num_dropout)

    merge_layer = Concatenate(axis=-1)([cat_output, num_output])
    normalization_layer = BatchNormalization()(merge_layer)
    merge_dropout_layer = Dropout(0.2)(normalization_layer)
    dense_layer = Dense(256, activation='relu')(merge_dropout_layer)
    dropout_layer = Dropout(0.2)(dense_layer)

    cases = Dense(1, activation='relu')(dropout_layer)
    fatalities = Dense(1, activation='relu')(dropout_layer)

    model = Model(inputs=[cat_inputs, num_inputs], outputs=[cases, fatalities])
    model.compile(loss=[tf.keras.losses.MeanSquaredLogarithmicError(),
                        tf.keras.losses.MeanSquaredLogarithmicError()],
                  optimizer=Adam())

    print(model.summary())
    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=1, factor=0.8),
                 EarlyStopping(monitor='val_loss', patience=50),
                 ModelCheckpoint(filepath=model_path+'_best_model.h5', monitor='val_loss', save_best_only=True)]

    train_history = model.fit([X_train_cat, X_train_num], [Y_train[:,0], Y_train[:,1]],
                              epochs=500,
                              batch_size=2048,  # with BatchNormalization, should be large enough
                              validation_data=([X_val_cat, X_val_num], [Y_val[:, 0], Y_val[:, 1]]),
                              callbacks=callbacks)
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('loss per epoch')
    plt.ylabel('overall loss')
    plt.xlabel('# epochs')
    plt.legend(['train', 'val'])
    plt.show()
    return model, ss


def predict_test_cases(model: keras.Model, scaler: StandardScaler, df_test: pd.DataFrame, cat_encoded_cols: List[str], numerical_cols: List[str]) -> pd.DataFrame:
    X_test_cat = df_test[cat_encoded_cols].values
    X_test_num = scaler.transform(df_test[numerical_cols].values)
    Y_pred = model.predict([X_test_cat, X_test_num])
    return Y_pred


# In[ ]:


# ====== main.py ======
import os
import pandas as pd
from typing import List, Optional, Tuple
from multiprocessing import Pool
from functools import partial

# from lib.io import get_train_subset, read_test_file, save_pickle, load_pickle, write_submission_csv, read_enrichment_file
# from lib.featurization import (get_points_in_the_past, randomize_the_points_and_censor_the_last_n, to_features,
#                                to_feat_row, generate_features, get_enriched_row)
# from lib.modeling import label_encode_columns, create_model, predict_test_cases

CACHE_PATH = '/tmp/covid_'
MAX_LOOKBACK = 40
N_POINTS_TO_KEEP = 24
CENSOR = None  # random between 1 and 10
CPU_COUNT = os.cpu_count()


def row_to_feat_row(row: pd.Series, train_df: pd.DataFrame, enrichment_df: pd.DataFrame, keep_id_col=False) -> pd.Series:
    history = get_points_in_the_past(row, MAX_LOOKBACK, train_df)
    history = randomize_the_points_and_censor_the_last_n(history, N_POINTS_TO_KEEP, CENSOR)
    enriched_row = get_enriched_row(row, enrichment_df)
    feat = to_features(row, history, keep_id_col)
    feat_row = to_feat_row(feat, enriched_row, row.ForecastId if keep_id_col else None)
    return feat_row


def get_train_validation_sets(pc_train: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = get_train_subset(1.0)
    enrichment_df = read_enrichment_file()
    oversampling_factor = 2
    train_len = int(pc_train * oversampling_factor * len(train_df))
    validation_len = len(train_df) - train_len

    train_rows = [el[1] for el in train_df.iterrows() for _ in range(oversampling_factor)]
    with Pool(CPU_COUNT * 2) as p:
        trains = p.map(partial(row_to_feat_row, train_df=train_df, enrichment_df=enrichment_df), train_rows[:train_len])
    with Pool(CPU_COUNT * 2) as p:
        validations = p.map(partial(row_to_feat_row, train_df=train_df, enrichment_df=enrichment_df), train_rows[-validation_len:])

    return pd.DataFrame(trains), pd.DataFrame(validations)


def get_test_set() -> pd.DataFrame:
    train_df = get_train_subset(1.0)
    test_df = read_test_file()
    enrichment_df = read_enrichment_file()

    test_rows = [el[1] for el in test_df.iterrows()]
    with Pool(CPU_COUNT * 2) as p:
        tests = p.map(partial(row_to_feat_row, train_df=train_df, enrichment_df=enrichment_df, keep_id_col=True), test_rows)

    return pd.DataFrame(tests)


def clip_and_fix_submission(submission_df: pd.DataFrame) -> pd.DataFrame:
    for idx in submission_df.index:
        max_cases = submission_df.loc[idx][f'confirmed_cases_{N_POINTS_TO_KEEP-1}']
        max_fatalities = submission_df.loc[idx][f'fatalities_{N_POINTS_TO_KEEP-1}']
        if submission_df.loc[idx]['ConfirmedCases'] < max_cases:
            submission_df.at[idx, 'ConfirmedCases'] = max_cases
        if submission_df.loc[idx]['Fatalities'] < max_fatalities:
            submission_df.at[idx, 'Fatalities'] = max_fatalities
    return submission_df


# In[ ]:


try:
    train_df, validation_df, test_df = load_pickle(CACHE_PATH + 'datasets.pk')
except IOError:
    print(f'Creating datasets with {CPU_COUNT} CPUs')
    print('Train and validation ...')
    train_df, validation_df = get_train_validation_sets(pc_train=0.8)
    print('Test ...')
    test_df = get_test_set()
    save_pickle([train_df, validation_df, test_df], CACHE_PATH + 'datasets.pk')

train_df.head(5)


# In[ ]:



train_df, validation_df, test_df, label_encoder_dict = label_encode_columns(['location'], train_df, validation_df, test_df)
train_df, validation_df, test_df = generate_features(train_df, validation_df, test_df)

train_df.head(5)


# In[ ]:


categorical_columns = ['location_encoded']
numerical_columns = [col for col in train_df.columns if not col[:5] in ('label', 'locat')]
label_columns = ['label_confirmed_cases', 'label_fatalities']
model, scaler = create_model(cat_encoded_cols=categorical_columns,
                             numerical_cols=numerical_columns,
                             labels=label_columns,
                             df_train=train_df,
                             df_val=validation_df,
                             model_path=CACHE_PATH)
labels_pred = predict_test_cases(model, scaler, test_df, categorical_columns, numerical_columns)
for i in range(len(label_columns)):
    test_df[label_columns[i]] = labels_pred[i]
test_df = test_df.rename(columns={'label_confirmed_cases': 'ConfirmedCases',
                                  'label_fatalities': 'Fatalities'})


# In[ ]:


submission_df = clip_and_fix_submission(test_df)[['ForecastId', 'ConfirmedCases', 'Fatalities']].astype('int')
write_submission_csv(submission_df)


# In[ ]:




