#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from __future__ import print_function
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils


# In[ ]:


# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
print('keras version ', keras.__version__)
# Same labels will be reused throughout the program
LABELS = ['Downstairs',
          'Jogging',
          'Sitting',
          'Standing',
          'Upstairs',
          'Walking']
# The number of steps within one time segment
TIME_PERIODS = 80
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 40


# In[ ]:


def read_data(file_path):

    column_names = ['user-id',
                    'activity',
                    'timestamp',
                    'x-axis',
                    'y-axis',
                    'z-axis']
    df = pd.read_csv(file_path,
                     header=None,
                     names=column_names)
    # Last column has a ";" character which must be removed ...
    df['z-axis'].replace(regex=True,
      inplace=True,
      to_replace=r';',
      value=r'')
    # ... and then this column must be transformed to float explicitly
    df['z-axis'] = df['z-axis'].apply(convert_to_float)
    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    df.dropna(axis=0, how='any', inplace=True)

    return df

def convert_to_float(x):

    try:
        return np.float(x)
    except:
        return np.nan
 
def show_basic_dataframe_info(dataframe):

    # Shape and how many rows and columns
    print('Number of columns in the dataframe: %i' % (dataframe.shape[1]))
    print('Number of rows in the dataframe: %i\n' % (dataframe.shape[0]))


# In[ ]:


# Load data set containing all the data from csv
df = read_data('../input/WISDM_ar_v1.1_raw.txt')


# In[ ]:


# Describe the data
show_basic_dataframe_info(df)
df.tail(20)


# In[ ]:


# Show how many training examples exist for each of the six activities
df['activity'].value_counts().plot(kind='bar',
                                   title='Training Examples by Activity Type')
plt.show()
# Better understand how the recordings are spread across the different
# users who participated in the study
df['user-id'].value_counts().plot(kind='bar',
                                  title='Training Examples by User')
plt.show()


# In[ ]:


def plot_activity(activity, data):

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,
         figsize=(15, 10),
         sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'X-Axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'Y-Axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_axis(ax, x, y, title):

    ax.plot(x, y, 'r')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

for activity in np.unique(df['activity']):
    subset = df[df['activity'] == activity][:180]
    plot_activity(activity, subset)


# In[ ]:


# Define column name of the label vector
LABEL = 'ActivityEncoded'
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
df[LABEL] = le.fit_transform(df['activity'].values.ravel())


# In[ ]:


from scipy.stats import moment

# x, y, z acceleration as features
N_FEATURES = 3

def create_segments_and_labels(df, time_steps, step, label_name):
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    users =[]
    for i in range(0, len(df) - time_steps, step):
        xs = df['x-axis'].values[i: i + time_steps]
        ys = df['y-axis'].values[i: i + time_steps]
        zs = df['z-axis'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)
        user = stats.mode(df['user-id'][i: i + time_steps])[0][0]
        users.append(user-1)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)
    users = np.asarray(users)

    return reshaped_segments, labels, users

x, y, y_users = create_segments_and_labels(df,
                                          TIME_PERIODS,
                                          STEP_DISTANCE,
                                          LABEL)


# In[ ]:


print(LABELS)
print(le.transform(LABELS))


# In[ ]:


from keras.utils.np_utils import to_categorical

Y_one_hot       = to_categorical(y)
Y_one_hot_users = to_categorical(y_users)


# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train_one_hot, y_test_one_hot = train_test_split(x, Y_one_hot, test_size=0.33, random_state=42)


# In[ ]:


import keras
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Conv1D(160, 12, input_shape=(x_train.shape[1],x_train.shape[2]) , activation='relu'))
model.add(layers.Conv1D(128, 10, activation='relu'))
model.add(layers.Conv1D(96, 8, activation='relu'))
model.add(layers.Conv1D(64, 6, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(6, activation='softmax'))

print(model.summary())


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


history = model.fit(x_train, y_train_one_hot, validation_data=(x_test, y_test_one_hot), epochs=150, batch_size=1024)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

y_pred = model.predict_classes(x_test)

y_test = y_test_one_hot.argmax(axis=-1)

# Creates a confusion matrix
cm = confusion_matrix(y_test, y_pred) 

# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index   = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking'], 
                     columns = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking'])

plt.figure(figsize=(10,10))
sns.heatmap(cm_df, annot=True, fmt="d", linewidths=0.5, cmap='Blues', cbar=False, annot_kws={'size':14}, square=True)
plt.title('Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']))


# In[ ]:


from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

# redefine model to output right after the first hidden layer
ixs = [0, 1, 2, 3, 4, 5, 6]
outputs = [model.layers[i].output for i in ixs]
model_view = Model(inputs=model.inputs, outputs=outputs)
# get feature map for first hidden layer
model_view.summary()


# In[ ]:


sample = x_train[3199].reshape(1,TIME_PERIODS, N_FEATURES)
feature_maps = model_view.predict(sample)


# In[ ]:


plt.plot(sample.reshape(TIME_PERIODS, N_FEATURES))
plt.show()

plt.rcParams["axes.grid"] = False
plt.figure(figsize=(20,10))
plt.imshow(sample.reshape(TIME_PERIODS, N_FEATURES).transpose(), cmap='Blues')
plt.show()

for fmap in feature_maps:
    print(fmap.shape)
    if (fmap.ndim==3):
        # plot filter channel in grayscale
        plt.rcParams["axes.grid"] = False
        plt.figure(figsize=(20,10))
        plt.imshow(fmap[0,:, :].transpose(), cmap='Blues')
        plt.tight_layout()
    else:
        plt.rcParams["axes.grid"] = True
        plt.bar(range(fmap.shape[1]), fmap[0,:])
    # show the figure
    plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train_one_hot, y_test_one_hot = train_test_split(x, Y_one_hot_users, test_size=0.33, random_state=42)


# In[ ]:


model = models.Sequential()
model.add(layers.Conv1D(192, 16, input_shape=(x_train.shape[1],x_train.shape[2]) , activation='relu'))
model.add(layers.Conv1D(160, 12, activation='relu'))
model.add(layers.Conv1D(128, 10, activation='relu'))
model.add(layers.Conv1D(96, 10, activation='relu'))
model.add(layers.Conv1D(64, 8, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(36, activation='softmax'))

print(model.summary())


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


history = model.fit(x_train, y_train_one_hot, validation_data=(x_test, y_test_one_hot), epochs=150, batch_size=1024)


# In[ ]:


y_pred = model.predict_classes(x_test)

y_test = y_test_one_hot.argmax(axis=-1)

# Creates a confusion matrix
cm = confusion_matrix(y_test, y_pred) 

# Transform to df for easier plotting
cm_df = pd.DataFrame(cm)

plt.figure(figsize=(20,20))
sns.heatmap(cm_df, annot=True, fmt="d", linewidths=0.5, cmap='Blues', cbar=False, annot_kws={'size':14}, square=True)
plt.title('Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[ ]:


print(classification_report(y_test, y_pred))

