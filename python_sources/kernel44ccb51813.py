# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print(tf.__version__)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

df = pd.read_csv('/kaggle/input/adult-census-income/adult.csv')
df.loc[df['income'] == '>50K', 'income'] = 1
df.loc[df['income'] == '<=50K', 'income'] = 0
df.loc[df['sex'] == 'Male', 'sex'] = 1
df.loc[df['sex'] == 'Female', 'sex'] = 0

columns = df.columns.values
for c in columns:
  df =df[df[c] != '?']

df.drop(['capital.gain', 'capital.loss', 'relationship', 'marital.status', 'workclass', 'native.country', 'race', 'fnlwgt'], axis=1, inplace=True)

train_df, test_df = train_test_split(df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

print('Num of examples in train {}'.format(len(train_df)))
print('Num of examples in test {}'.format(len(test_df)))
print('Num of examples in val {}'.format(len(val_df)))

def df_to_dataset(dataframe, batch_size, shuffle=True):
  dataframe = dataframe.copy()
  target = dataframe.pop('income')
  dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), target))
  if shuffle:
    dataset = dataset.shuffle(len(dataframe))

  return dataset.batch(batch_size)

BATCH_SIZE=64
train_ds = df_to_dataset(train_df, BATCH_SIZE)
test_ds = df_to_dataset(test_df, BATCH_SIZE, shuffle=False)
val_ds = df_to_dataset(val_df, BATCH_SIZE, shuffle=False)




feature_columns = []

for column in ['education.num', 'sex', 'hours.per.week']:
  feature_columns.append(tf.feature_column.numeric_column(column))


age_buckets = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('age'), 
                                                        boundaries=[18, 26, 34, 42, 50, 58, 66, 74, 82, 90])
feature_columns.append(age_buckets)

education_cat = tf.feature_column.categorical_column_with_vocabulary_list('education', 
                                                                          ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'])
occupation_cat = tf.feature_column.categorical_column_with_vocabulary_list('occupation',
                                                                           ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'])

education_age_cross_fet = tf.feature_column.crossed_column([age_buckets, education_cat], hash_bucket_size=1000)
occupation_age_corss_fet = tf.feature_column.crossed_column([age_buckets, occupation_cat], hash_bucket_size=1000)

feature_columns.append(tf.feature_column.indicator_column(education_age_cross_fet))
feature_columns.append(tf.feature_column.indicator_column(occupation_age_corss_fet))



feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(1)
])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

EPOCHS = 500
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping])