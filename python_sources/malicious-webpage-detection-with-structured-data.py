#!/usr/bin/env python
# coding: utf-8

# # <font style="color:red;">Deep Learning Webpage Classification Model with Structured Data <br/> (Balanced Using Class Weights & Faster Convergence Using Initial Bias)</font>

# ## Basic Initialisation

# In[ ]:


# Installing mandatory libraries
get_ipython().system(' pip install profanity_check ')
get_ipython().system(' pip install tld')


# In[ ]:


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Load the TensorBoard notebook extension.
get_ipython().run_line_magic('load_ext', 'tensorboard')

# Clear any logs from previous runs
get_ipython().system('rm -rf ./logs/ ')

# Common imports
import pandas as pd
import numpy as np
import time
import os
import sklearn

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "C3_Deep Learning Classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "imgs", CHAPTER_ID)

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "imgs", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True, fig_extension="svg", resolution=300):
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension)


# ### Loading Dataset

# In[ ]:


# Verifying pathname of dataset before loading
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename));
        print(os.listdir("../input"))


# In[ ]:


# Load Datasets
def loadDataset(file_name):
    df = pd.read_csv(file_name,engine = 'python')
    return df

df_train = loadDataset("/kaggle/input/dataset-of-malicious-and-benign-webpages/Webpages_Classification_train_data.csv/Webpages_Classification_train_data.csv")
df_test = loadDataset("/kaggle/input/dataset-of-malicious-and-benign-webpages/Webpages_Classification_test_data.csv/Webpages_Classification_test_data.csv")
#Ensuring correct sequence of columns & dropping IP Address
df_train = df_train[['url','url_len','geo_loc','tld','who_is','https','js_len','js_obf_len','label']]
df_test = df_test[['url','url_len','geo_loc','tld','who_is','https','js_len','js_obf_len','label']]


#  ### Vectorizing URL Text Using Profanity Score

# In[ ]:


#vectorising the URL Text
from urllib.parse import urlparse
from tld import get_tld

start_time= time.time()
#Function for cleaning the URL text before vectorization
def clean_url(url):
    url_text=""
    try:
        domain = get_tld(url, as_object=True)
        domain = get_tld(url, as_object=True)
        url_parsed = urlparse(url)
        url_text= url_parsed.netloc.replace(domain.tld," ").replace('www',' ') +" "+ url_parsed.path+" "+url_parsed.params+" "+url_parsed.query+" "+url_parsed.fragment
        url_text = url_text.translate(str.maketrans({'?':' ','\\':' ','.':' ',';':' ','/':' ','\'':' '}))
        url_text.strip(' ')
        url_text.lower()
    except:
        url_text = url_text.translate(str.maketrans({'?':' ','\\':' ','.':' ',';':' ','/':' ','\'':' '}))
        url_text.strip(' ')
    return url_text

df_test['url'] = df_test['url'].map(clean_url)
df_train['url'] = df_train['url'].map(clean_url)
print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))


# In[ ]:


# give profanity score to each URL using the Profanity_Check Library
from profanity_check import predict_prob

start_time= time.time()
#Function for calculating profanity in a dataset column
def predict_profanity(df):
    arr=predict_prob(df['url'].astype(str).to_numpy())
    arr= arr.round(decimals=3)
    df['url'] = pd.DataFrame(data=arr,columns=['url'])
    #df['url']= df_test['url'].astype(float).round(decimals=3) #rounding probability to 3 decimal places
    return df['url']

df_train['url']= predict_profanity(df_train)
df_test['url']= predict_profanity(df_test)

print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))


# In[ ]:


#Looking for NaN, if any
print(df_train['url'].isnull().sum())
print(df_test['url'].isnull().sum())


# In[ ]:


#Rename 'url' to 'url_vect'
df_train.rename(columns = {"url": "url_vect"}, inplace = True)
df_test.rename(columns = {"url": "url_vect"}, inplace = True)
df_train


# In[ ]:


#converting 'label' values to numerical value for classification
import time

start_time= time.time()

df_test['label'].replace(to_replace ="good", value =1, inplace=True)
df_train['label'].replace(to_replace ="good", value =1, inplace=True)
df_test['label'].replace(to_replace ="bad", value =0, inplace=True)
df_train['label'].replace(to_replace ="bad", value =0, inplace=True)

print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))


# In[ ]:


df_test


# In[ ]:


df_train


# ### Analysis of Class Imbalance

# In[ ]:


# No of Classes in Label
df_train['label'].unique()


# In[ ]:


# Class Distribution of Labels
df_train.groupby('label').size()


# In[ ]:


# Analysis of Postives and Negatives in the Dataset
neg, pos = np.bincount(df_train['label'])
total = neg + pos
print ('Total of Samples: %s'% total)
print('Positive: {} ({:.2f}% of total)'.format(pos, 100 * pos / total))
print('Negative: {} ({:.2f}% of total)'.format(neg, 100 * neg / total))


# In[ ]:


#Class Labels shown in Histogram
df_train["label"].hist()
#save_fig("Fig2")


# In[ ]:


# Representation of Labels in the Stack form

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# create dummy variable then group by that
# set the legend to false because we'll fix it later
df_train.assign(dummy = 1).groupby(['dummy','label']).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()
).to_frame().unstack().plot(kind='bar',stacked=True,legend=False, color={'grey','white'}, linewidth=0.50, ec='k')
# or it'll show up as 'dummy' 
plt.xlabel('Benign/Malicious Webpages')
# disable ticks in the x axis
plt.xticks([])
# fix the legend or it'll include the dummy variable
current_handles, _ = plt.gca().get_legend_handles_labels()
reversed_handles = reversed(current_handles)
correct_labels = reversed(['Malicious','Benign'])
plt.legend(reversed_handles,correct_labels)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
#save_fig("Fig3")
plt.show()


# ### Earmarking Validation, Train & Test Sets

# In[ ]:


#Selection lower numbers as of now for fast testing

train= df_train.iloc[:1000000,]
val= df_train.iloc[1000001:,]
test= df_test.iloc[0:,]

print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# #### Tensorflow TF Dataset

# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers


# In[ ]:


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('label')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


# In[ ]:


#Using a Batch Size of 2048 and copying data to tf.data dataset
batch_size = 2048
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


# In[ ]:


# For checking the batch
#example_batch = next(iter(train_ds))[0]
#example_batch


# #### Making the Feature Layer with Feature Columns

# In[ ]:


feature_columns = []

# numeric cols
for header in ['url_vect', 'url_len', 'js_len', 'js_obf_len']:
  feature_columns.append(feature_column.numeric_column(header))

#Categorical Columns
who_is = feature_column.categorical_column_with_vocabulary_list('who_is', ['complete', 'incomplete'])
who_is_one_hot = feature_column.indicator_column(who_is)
https = feature_column.categorical_column_with_vocabulary_list('https', ['yes', 'no'])
https_one_hot = feature_column.indicator_column(https)
feature_columns.append(https_one_hot)
feature_columns.append(who_is_one_hot)

# Hashed Categorical Columns
geo_loc_hashed = feature_column.categorical_column_with_hash_bucket('geo_loc', hash_bucket_size=230)
tld_hashed = feature_column.categorical_column_with_hash_bucket('tld', hash_bucket_size=1200)
geo_loc_indicator = feature_column.indicator_column(geo_loc_hashed)
tld_indicator = feature_column.indicator_column(tld_hashed)
feature_columns.append(tld_indicator)
feature_columns.append(geo_loc_indicator)

#Creating Feature Layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


# ## Tensor Flow Model Building

# ### Setting the Initial Bias

# 
# The correct bias to set can be derived from:
# 
# $$ p_0 = pos/(pos + neg) = 1/(1+e^{-b_0}) $$
# $$ b_0 = -log_e(1/p_0 - 1) $$
# $$ b_0 = log_e(pos/neg)$$

# With this initialization the initial loss should be approximately:
# 
# $$-p_0log(p_0)-(1-p_0)log(1-p_0) = 0.01317$$

# This loss is about 50 times less than a naive initialisation

# In[ ]:


#Using Initial Bias to overcome Class Imbalance
initial_bias = np.log([pos/neg])
initial_bias


# In[ ]:


#Making a Tensorflow Model
from tensorflow import keras

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
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='sigmoid',
        bias_initializer=output_bias),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=metrics)
    return model


# In[ ]:


#Initialize the Model
model = make_model()
model_initial_bias = make_model(output_bias = initial_bias)


# #### Making Class Specific Weights

# In[ ]:


# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


# #### Fitting the Model

# In[ ]:


#Fitting the Model with Class Weights
from datetime import datetime

#Defining Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    verbose=1,
    patience=20,
    mode='max',
    restore_best_weights=True)

# Define the Keras TensorBoard callback.
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

start_time= time.time()
zero_bias_history = model.fit(train_ds, epochs=40,validation_data=val_ds,
          callbacks=[early_stopping,tensorboard_callback],
          class_weight=class_weight
         )
print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))


# In[ ]:


#Fitting the Model with Class Weights & Inital Bias
from datetime import datetime

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    verbose=1,
    patience=20,
    mode='max',
    restore_best_weights=True)

# Define the Keras TensorBoard callback.
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

start_time= time.time()
initial_bias_history = model_initial_bias.fit(train_ds, epochs=40,validation_data=val_ds,
          callbacks=[early_stopping,tensorboard_callback],
          class_weight=class_weight
         )
print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))


# In[ ]:


# The final model Summary
model_initial_bias.summary()


# ### Evaluation of Model & Results

# In[ ]:


#Accuracy & Loss - Class Weights w/o Initial Bias
results = model.evaluate(test_ds)
#print("Accuracy", accuracy)
print("Loss: {0}, Accuracy: {1}".format(results[0],results[5]))


# In[ ]:


#Accuracy & Loss - Class Weights with Initial Bias
start_time= time.time()
results = model_initial_bias.evaluate(test_ds)
#print("Accuracy", accuracy)
print("Loss: {0}, Accuracy: {1}".format(results[0],results[5]))
print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))


# In[ ]:


start_time= time.time()
X_test, y_test = iter(test_ds).next()
X_train, y_train = iter(train_ds).next()
print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))


# In[ ]:


#Confusion Matrix

start_time= time.time()
y_pred=model_initial_bias.predict_classes(X_test)
con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=4)
con_mat_df = pd.DataFrame(con_mat_norm,index = ['Benign','Malicious'], columns = ['Benign','Malicious'])
print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))


# In[ ]:


# Confusion Matrix
con_mat_df


# In[ ]:


# Plotting the Confusion Matrix Using Matploit & Seaborn
import seaborn as sns

start_time= time.time()
figure = plt.figure(figsize=(6,6))
sns.heatmap(con_mat_df,annot=True,cmap=plt.cm.binary,fmt='g',linewidths=0.50,linecolor='black')
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
#save_fig("Fig7")
plt.show()

print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))


# ### Showing the Influence of Initial Bias

# In[ ]:


mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_loss(history, label, n):
    # Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch,  history.history['loss'],
               color=colors[n], label='Train '+label)
    plt.semilogy(history.epoch,  history.history['val_loss'],
          color=colors[n], label='Val '+label,
          linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


# In[ ]:


plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(initial_bias_history, "Initial Bias", 1)
#save_fig("Fig8")


# ### Plotting of Metrics: Loss, AUC, Precision & Recall

# In[ ]:


def plot_metrics(history):
    metrics =  ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()


# In[ ]:


plot_metrics(initial_bias_history)


# #### ROC Plot

# In[ ]:


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False Positive Rate [%]')
    plt.ylabel('True Positive Rate [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


# In[ ]:


train_predictions_baseline = model_initial_bias.predict(X_train, batch_size=2048)
test_predictions_baseline = model_initial_bias.predict(X_test, batch_size=2048)


# In[ ]:


plot_roc("Train", y_train, train_predictions_baseline, color=colors[0])
plot_roc("Test", y_test, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')
#save_fig("Fig9")


# #### Confusion Matrix

# In[ ]:


def plot_cm(labels, predictions, p=0.5):
    cm = tf.math.confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])
    print('Total : ', np.sum(cm[1]))


# In[ ]:



#for name, value in zip(model.metrics_names, results):
#    print(name, ': ', value)
# print()
# plot_cm(y_test, y_pred)


# In[ ]:


tensorboard --logdir logs  --port=8021

