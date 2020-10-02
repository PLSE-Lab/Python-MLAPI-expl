# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import tensorflow as tf
from tensorflow import keras

import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import missingno as msno

file = tf.keras.utils
raw_df = pd.read_excel('../input/covid19/dataset.xlsx')

cleaned_df = raw_df.copy()

# dataprep categorical
mask_pos_neg = {'positive': 1, 'negative': 0}
mask_detected = {'detected': 1, 'not_detected': 0}
mask_notdone_absent_present = {'not_done': 0, 'absent': 1, 'present': 2}
mask_normal = {'normal': 1}
mask_urine_color = {'light_yellow': 1, 'yellow': 2, 'citrus_yellow': 3, 'orange': 4}
mask_urine_aspect = {'clear': 1, 'lightly_cloudy': 2, 'cloudy': 3, 'altered_coloring': 4}
mask_realizado = {'Não Realizado': 0}
mask_urine_leuk = {'<1000': 1000}
mask_urine_crys = {'Ausentes': 1, 'Urato Amorfo --+': 0, 'Oxalato de Cálcio +++': 0, 'Oxalato de Cálcio -++': 0, 'Urato Amorfo +++': 0}

cleaned_df = cleaned_df.replace(mask_detected)
cleaned_df = cleaned_df.replace(mask_pos_neg)
cleaned_df = cleaned_df.replace(mask_notdone_absent_present)
cleaned_df = cleaned_df.replace(mask_normal)
cleaned_df = cleaned_df.replace(mask_realizado)
cleaned_df = cleaned_df.replace(mask_urine_leuk)
cleaned_df = cleaned_df.replace(mask_urine_color)
cleaned_df = cleaned_df.replace(mask_urine_aspect)
cleaned_df = cleaned_df.replace(mask_urine_crys)

null_cleaned_df = cleaned_df.isna().mean().round(4) * 100
nulls = null_cleaned_df[null_cleaned_df > 85]

cleaned_df = cleaned_df[[col for col in cleaned_df.columns if col not in nulls]]
cleaned_df = cleaned_df.dropna(how='any')

drop_cols = [
    'Patient ID',
    'Patient addmited to regular ward (1=yes, 0=no)',
    'Patient addmited to semi-intensive unit (1=yes, 0=no)',
    'Patient addmited to intensive care unit (1=yes, 0=no)'
]

cleaned_df.drop(drop_cols, axis=1, inplace=True)

print('Size of the data', cleaned_df.shape)
print(cleaned_df.head())

# Use a utility from sklearn to split and shuffle our dataset.
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('SARS-Cov-2 exam result'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('SARS-Cov-2 exam result'))
test_labels = np.array(test_df.pop('SARS-Cov-2 exam result'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)

print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

def make_model(metrics = METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(
          16, 
          activation='relu',
          input_shape=(train_features.shape[-1],)
      ),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(
          1, 
          activation='sigmoid',
          bias_initializer=output_bias
      ),
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics
  )

  return model

EPOCHS = 10
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True
)

model = make_model()

model.predict(train_features[:10])

results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))

baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_features, val_labels)
)

def plot_metrics(history):
  metrics =  ['loss', 'auc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()
    
plot_metrics(baseline_history)

train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))

baseline_results = model.evaluate(test_features, test_labels,
                                  batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_baseline)