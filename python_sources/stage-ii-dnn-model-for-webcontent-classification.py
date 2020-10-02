#!/usr/bin/env python
# coding: utf-8

# # <font style="color:red;"> <center>Deep Learning Hybrid Classification Model for Web Content <br> (Balanced Using Weights, Initialized with Output Bias) </center></font>

# ## <font color='blue'>Basic Initialisation</font>

# In[ ]:


#Note the code in this Notebook requires tensorflow version 2.1.0rc0
#!pip install grpcio==1.27.2 #This version of grpcio is reqd for tensorflow 2.1
#!pip install tensorflow==2.1.0rc0
#tf.__version__
#!pip install tensorflow-hub


# In[ ]:


# Common imports
import pandas as pd
import numpy as np
import time
import os
import warnings

#Time/CPU Profiling
overall_start_time= time.time()

# Load the TensorBoard notebook extension.
get_ipython().run_line_magic('load_ext', 'tensorboard')

# Clear any logs from previous runs
get_ipython().system('rm -rf ./logs/ ')

# to make this notebook's output stable across runs
np.random.seed(42)

#Disabling Warnings
warnings.filterwarnings('ignore')

# To plot figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# ## <font color='blue'> Loading Dataset </font>

# In[ ]:


#Verifying pathname of dataset before loading - for Kaggle
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
start_time= time.time()
df_train = loadDataset("/kaggle/input/dataset-of-malicious-and-benign-webpages/Webpages_Classification_train_data.csv/Webpages_Classification_train_data.csv")
df_test = loadDataset("/kaggle/input/dataset-of-malicious-and-benign-webpages/Webpages_Classification_test_data.csv/Webpages_Classification_test_data.csv")
#Ensuring correct sequence of columns 
df_train = df_train[['url','content','label']]
df_test = df_test[['url','content','label']]
print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))


#  ## <font color='blue'>Preprocessing the Dataset </font>

# #### Cleaning the Dataset

# In[ ]:


start_time= time.time()
df_test['content'] = df_test['content'].str.lower()
df_test.drop(columns=['url'],inplace=True)
df_test.rename(columns={'content':'text'},inplace=True)
df_train['content'] = df_train['content'].str.lower()
df_train.drop(columns=['url'],inplace=True)
df_train.rename(columns={'content':'text'},inplace=True)
print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))
#Looking for NaN, if any
#print(df_train.isnull().sum())
#print(df_test.isnull().sum())


# In[ ]:


#df_test
#df_train


# #### Converting Label Value to 0,1

# In[ ]:


#converting 'label' to numerical value (0-Malicious,1-Benign)
start_time= time.time()
df_test['label'].replace(to_replace ="good", value =1, inplace=True)
df_train['label'].replace(to_replace ="good", value =1, inplace=True)
df_test['label'].replace(to_replace ="bad", value =0, inplace=True)
df_train['label'].replace(to_replace ="bad", value =0, inplace=True)
print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))


# ## <font color='blue'>Analysis of Class Imbalance </font>

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


# Representation of Labels in the Stack form
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# create dummy variable then group by that
# set the legend to false because we'll fix it later
df_train.assign(dummy = 1).groupby(['dummy','label']).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()).to_frame().unstack().plot(kind='bar',
    stacked=True,legend=False, color={'red','green'})
# or it'll show up as 'dummy' 
plt.xlabel('good/bad Websites')
# disable ticks in the x axis
plt.xticks([])
# fix the legend or it'll include the dummy variable
current_handles, _ = plt.gca().get_legend_handles_labels()
reversed_handles = reversed(current_handles)
correct_labels = reversed(['bad','good'])
plt.legend(reversed_handles,correct_labels)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
#plt.savefig("/img/C4.1/Fig1.png")
plt.show()


# ## <font color='blue'> Earmarking Validation, Train & Test Sets </font>

# In[ ]:


#Segregating Validation Set away from the Training Set
train= df_train.iloc[:1000000,]
val= df_train.iloc[1000001:,]
test= df_test.iloc[:,]

print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# In[ ]:


#Converting the dataframes into X, y numpy arrays 
X_train = train['text'].to_numpy()
y_train = train['label'].astype(int).to_numpy()
X_val = val['text'].to_numpy()
y_val = val['label'].astype(int).to_numpy()
X_test = test['text'].to_numpy()
y_test = test['label'].astype(int).to_numpy()


# ## <font color='blue'> Making the Tensor Flow Deep Learning Model </font>

# ### Using Transfer Learning - making use of Text Embedding Model from Tensorflow Hub

# In[ ]:


# Using Transfer Learning from Tensorflow hub- Universal Text Encoder
import tensorflow_hub as hub
from tensorflow import keras 

start_time= time.time()
# Use the saved ecoder from Stage I
encoder = keras.models.load_model("/kaggle/input/savedmodel/PretrainedTFModel/1")
#encoder = hub.load("/kaggle/input/savedmodel/PretrainedTFModel/1")
print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))
encoder(['Hello World']) #For Testing the Encoder


# ### Making Class Specific Weights (for handling the Imbalance)

# In[ ]:


# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0
class_weight = {0: weight_for_0, 1: weight_for_1}
print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


# ### Setting the Initial Bias

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


# ### Making and Initializing the TensorFlow Model

# In[ ]:


#Making a Tensorflow Model
from tensorflow import keras
import tensorflow as tf
import tensorflow_hub as hub

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc')]
def make_model(metrics = METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
        #First layer is of Universal Text Encoder Deep Averaging Network (DAN) using Transfer Learning
        hub.KerasLayer(encoder, input_shape=[],dtype=tf.string,trainable=True),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid',
        bias_initializer=output_bias),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001),loss='binary_crossentropy',metrics=metrics)
    return model


# In[ ]:


#Initialize the Model
model_zero_bias = make_model()
model_initial_bias = make_model(output_bias = initial_bias)
model_zero_bias.summary() # print model summary with zeor bias
model_initial_bias.summary() # print model summary with initial bias


# ## <font color =blue> Training the Model </font>

# ### Defining Early Stopping and Keras Tensorboard

# In[ ]:


#Fitting the Model with Class Weights
from datetime import datetime

#Defining Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    verbose=1,
    patience=70,
    mode='max',
    restore_best_weights=True)

# Define the Keras TensorBoard callback.
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


# ### Fitting and Training without Initial Bias

# In[ ]:


# Fitting the Model without Inital Bias, but with Class weights
start_time= time.time()
zero_bias_history = model_zero_bias.fit(X_train,y_train,batch_size=2048, epochs=100,validation_data=(X_val, y_val),
          callbacks=[early_stopping,tensorboard_callback],
          class_weight=class_weight)
print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))


# ### Evaluation of Model & Results: With Zero Bias

# In[ ]:


#Accuracy over Test Dataset
results = model_zero_bias.evaluate(X_test,y_test)
print("Loss: {0}, Accuracy: {1}".format(results[0], results[5]))


# ### Fitting and Training with Initial Bias

# In[ ]:


# Fitting the Model without Inital Bias, but with Class weights
start_time= time.time()
initial_bias_history = model_initial_bias.fit(X_train,y_train,batch_size=2048, epochs=100,validation_data=(X_val, y_val),
          callbacks=[early_stopping,tensorboard_callback])
print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))


# ### Evaluation of Model & Results: With Initial Bias

# In[ ]:


#Accuracy over Test Dataset
results = model_initial_bias.evaluate(X_test,y_test)
print("Loss: {0}, Accuracy: {1}".format(results[0], results[5]))


# #### Confusion Matrix: Initial Bias Model

# In[ ]:


#Confusion Matrix
r=results
con_mat_norm= [[r[3]/(r[3]+r[4]),r[2]/(r[2]+r[3])],[r[4]/(r[4]+r[1]),r[1]/(r[1]+r[4])]]
con_mat_df = pd.DataFrame(con_mat_norm,index = ['Malicious','Benign'], columns = ['Malicious','Benign'])
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
#plt.savefig("/img/C4.1/Fig2:ConfusionMatrix_B&W.png")
plt.show()
print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))


# ## <font color=blue> Showing the Influence of Initial Bias </font>

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
#plt.savefig("/img/C4.1/Fig3:Bias Plot.png")


# ## <font color=blue> Plotting of Metrics: Loss, AUC, Precision & Recall </font>

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
#plt.savefig("/img/C4.1/Fig4:Loss, AuC, Precision & Recall.png")


# #### ROC Plot

# In[ ]:


import sklearn
from sklearn import metrics

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
#plt.savefig("/img/C4.1/Fig5:ROC Curve.png")


# ### Display Tensorboard Graphs

# In[ ]:


tensorboard --logdir logs  --port=8050


# ## <font color=blue> Run Time Profiling Statistics of this Notebook </font>

# In[ ]:


# Total Runtime of this Notebook
print("***Total Time taken --- %s mins ---***" % ((time.time() - overall_start_time)/60))


# ### Miscellaneous Maintenance Code: Run this for Selected Variables to Clear RAM Space
# (Note: Run this selectively only if you are Running Short of Memory)

# In[ ]:


#Clearing Additional load of variables: Creating More RAM Space
import gc

#del df_train_good
#del df_train_bad
#del df_trial
#gc.collect()

