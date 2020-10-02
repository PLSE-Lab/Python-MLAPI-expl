#!/usr/bin/env python
# coding: utf-8

# ## Learn Tensorflow 2: An easy way
# 
# ### Objective
# The key objective of this kernel is 
# - To introduce the new programming style for creating a neural network using Keras on TF2
# - One should be capable of starting Deep Learning without going through the conventional way of ML algorithms and landing here eventually. Ref:[Learn ML - ML101, Rank 4500 to ~450 in a day](https://www.kaggle.com/gowrishankarin/learn-ml-ml101-rank-4500-to-450-in-a-day)
# - Instill knowledge on key principles includes
#     - Train, Validation and Test data splitting
#     - A simple mechanism to fill the missing values in the dataset
#     - Bias and Overfit handlers like class weights and initial bias calculation
#     - Handling categorical columns using One Hot Encoding or Embedding principles
#     - Elegant way of creating dataset of tensor slices from pandas dataframes
#     - Build a simple and flat a NN architecture using Keras
#     - Predict the targets via predict functions
#     - Analyse the results using various metrics include accuracy, precision, ROC curve etc
# 
# This is an attempt to make an engineer novice to expert on approach and process of building a neural network for classification problem.

# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split


# ## Pre-Processing
# 
# ### Read Data
# - Read the data from the data source using Pandas package.
# - We have 3 files - train.csv, test.csv and sample_submission.csv
# - Train set has 23 feature columns and 60,000 observations
# - Test set has 23 feature columns and 40,000 observations
# 
# Hint: 
# - Test set volume is huge, there is a chance of imbalance target value. Ensure it is handled to avoid bias.
# - Also note, if the data is imbalanced - Right metrics is AUC(area under the curve) and not accuracy.

# In[ ]:


is_local = False
INPUT_DIR = "/kaggle/input/cat-in-the-dat-ii/"

import tensorflow as tf; 
print(tf.__version__)

if(is_local):
    INPUT_DIR = "../input/"

import os
for dirname, _, filenames in os.walk(INPUT_DIR):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv(INPUT_DIR + "train.csv")
test_df = pd.read_csv(INPUT_DIR + "test.csv")
submission_df = pd.read_csv(INPUT_DIR + "sample_submission.csv")
print("Shape of the train data is ", train_df.shape)
print("Shape of the test data is ", test_df.shape)

train_df.head()


# ### Constants
# Let us initialize the constants 
# 
# - **Embedding Dimensions:** An embedding a low-dimensional space into which a high dimensional vector is represented. 
#     It makes a large space of information into a sparse vectors. 
#     Here we are randomly picking 9 dimensions to represent the sparse vector of certain features. More about embeddings below.
# - **Batch Size:** Batch size tells the network, the number of observations to propagate through the networks. The key aspect of the batch size is 
#     how quickly a model trains/learns.
# - **Epochs:** Epoch is the number of times a learning algorithms is iterated on the entire training data set. ie Count of every observation in the training dataset
#     involved in the learning process of the model. 
# - **Train and Validation Split:** A model is better if it is generalized rather than work well for the given dataset. So the train dataset is split into train and validation data. During the training process, accuracy of the model is measured through validation data metrics rather than train.
# - **Metrics:** We are observing 8 metrics to understand a model. Ref: [Precision, Recall, ROC, AUC - Validation Metrics](https://www.kaggle.com/gowrishankarin/precision-recall-roc-auc-validation-metrics)

# In[ ]:


EMBEDDING_DIMENSIONS=9
BATCH_SIZE = 1024
EPOCHS = 25
TRAIN_VAL_SPLIT_RATIO = 0.3

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]


# ## Understanding the Features
# Understanding the dataset by doing an **Exploratory Data Analysis and Visualization** brings significant insights for solving the problem. Ref: [Interactive EDA using Plotly](https://www.kaggle.com/gowrishankarin/interactive-eda-using-plotly)  
# In our dataset, we have 5 types of features
# - Binary Data - Discrete (1, 0) or (True, False), (Yes, No)
# - Nominal Data - Discrete (Male, Female) or (Eye Colors) or (Hair Color)
# - Ordinal Data - Discrete Sequence (Hot, Hotter, Hottest) or (Scale of 1-10)
# - Cyclic Data - Continuous, Numeric (Weekdays) or (Month Dates)
# - Target Data - Our target data is a binary data

# In[ ]:


COLUMN_TYPES = {
    'id': 'index',
    'bin_0': 'binary', 'bin_1': 'binary', 'bin_2': 'binary', 'bin_3': 'binary', 
    'bin_4': 'binary', 'nom_0': 'categorical', 'nom_1': 'categorical',
    'nom_2': 'categorical', 'nom_3': 'categorical', 'nom_4': 'categorical', 
    'nom_5': 'categorical', 'nom_6': 'categorical', 'nom_7': 'categorical', 
    'nom_8': 'categorical', 'nom_9': 'categorical',
    'ord_0': 'ordinal', 'ord_1': 'ordinal', 'ord_2': 'ordinal', 
    'ord_3': 'ordinal', 'ord_4': 'ordinal', 'ord_5': 'ordinal', 
    'day': 'cyclic', 'month': 'cyclic',
    'target': 'target'
}


# ### Dealing with Missing Data
# The most time consuming aspect of a dataset is dealing with missing data. We are presenting a simple way to address with this
# - All categorical and binary data are filled with a special value NaN
# - Cyclic data is filled with the median value for simplicity
# 
# There are sophisticated imputation mechanisms available in sklearn packages. Ref: [Fancy Impute](https://github.com/iskandr/fancyimpute), [sklearn.impute](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute)

# In[ ]:


def fill_missing_values(dataframe, ignore_cols=['id', 'target']):
    feature_cols = [column for column in dataframe.columns if column not in ignore_cols]
    for a_column in feature_cols:
        typee = COLUMN_TYPES[a_column]
        if(typee == 'binary'):
            dataframe.loc[:, a_column] = dataframe.loc[:, a_column].astype(str).fillna(-9999999)
        elif(typee == 'numeric'):
            pass
        elif(typee == 'categorical'):
            dataframe.loc[:, a_column] = dataframe.loc[:, a_column].astype(str).fillna(-9999999)
        elif(typee == 'ordinal'):
            dataframe.loc[:, a_column] = dataframe.loc[:, a_column].astype(str).fillna(-9999999)
        elif(typee == 'cyclic'):
            median_val = np.median(dataframe[a_column].values)
            if(np.isnan(median_val)):
                median_val = np.median(dataframe[~np.isnan(dataframe[a_column])][a_column].values)
            print(a_column, median_val)
            dataframe.loc[:, a_column] = dataframe.loc[:, a_column].astype(float).fillna(median_val)
            
    return dataframe.copy(deep=True)

train_df = fill_missing_values(train_df, ignore_cols=['id', 'target'])
test_df = fill_missing_values(test_df, ignore_cols=['id'])


# ## Bias and Class Weights
# - Our goal is to avoid overfitting and generalization of the model we create.
# - As a first measure we split the data into train and validation set
# - Further we shall intervene by calculating initial bias and class weights
# - How to find imbalance in the target value
#     - Find the ratio of positive and negatives in the target distribution
#     - In our dataset, we have 18.72 percent are positives and rest are negatives
#     - We assume a similar behavior in the test dataset as well
#     - However there is no guarantee that it will be true
#     
# ### Initial Bias
# Since the dataset is imbalanced, our output layer to reflect the bias. Bias can be calculated as follows
# 
# \begin{equation*}
# p_0 = \frac{pos}{pos + neg} = \frac{1}{1 + b^0} \\
# b_0 = - log_e(\frac{1}{p_0} - 1) \\
# b_0 = log_e(\frac{pos}{neg})
# \end{equation*}
# 
# With this inialization, Initial loss/cost function or **cross entropy**
# 
# <div align="center" style="font-size: 18px">Loss/Cost Function or Cross Entropy</div><br>
# 
# 
# \begin{equation*}
# - p_0 log(p_0) - (1 - p_0) log(1 - p_0)
# \end{equation*}

# In[ ]:


def get_initial_bias(df, col_name='target'):
    neg, pos = np.bincount(df[col_name])
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    initial_bias = np.log([pos/neg])
    
    return initial_bias



initial_bias = get_initial_bias(train_df)


# ### Class Weights
# - The idea is to have the model heavily weight the few positive option available for training. 
# - This is done using "class_weight" param of the model to pay more attention towards under represented class. 
# - Incorporation of class weights is made aware through the cost function
# - The loss/cost function discussed earlier transform as follows
# 
# <div align="center" style="font-size: 18px">Weighted Cost Entropy</div><br>
# 
# \begin{equation*}
# - w_0 p_0 log(p_0) - w1 (1 - p_0) log(1 - p_0)
# \end{equation*}

# In[ ]:


def get_class_weights(df, col_name='target'):
    neg, pos = np.bincount(df[col_name])
    weight_for_0 = (1 / neg) * (neg + pos) / 2.0
    weight_for_1 = (1 / pos) * (neg + pos) / 2.0

    class_weight = {
        0: weight_for_0,
        1: weight_for_1
    }

    print("Class 0: ", weight_for_0, "Weightage")
    print("Class 1: ", weight_for_1, "Weightage")
    
    return class_weight

class_weight = get_class_weights(train_df)


# ## Stratified Split of Train and Validation Data
# To avoid Sampling bias, we shall split the data using Stratification. Stratified split ensure the imbalance ratio is maintained in train and validation dataset.
# 
# What is Sampling Bias?  
# When some members of the population have lower sampling probability than others, a sampling bias occurs. 
# It results in samples favoring a particular group of the population and the model end up with bias.

# In[ ]:


### Stratified Split

from sklearn.model_selection import StratifiedShuffleSplit

def split_train_validation_data(df, col_name='target', stratify=True, test_size=0.3):
    train = None
    val = None
    
    if(stratify):
        

        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=21)
        sss.get_n_splits(df, df.target)

        splits = sss.split(df, df.target) 
        
        indices = []
        for train_index, test_index in splits:
            indices.append({
                'train': train_index,
                'test': test_index
            })

        train = df.iloc[indices[0]['train']]
        val = df.iloc[indices[0]['test']]
        
    else:
        train, val = train_test_split(train_df, test_size=test_size)
    return train, val

train, val = split_train_validation_data(train_df, test_size=TRAIN_VAL_SPLIT_RATIO)


# In[ ]:


get_initial_bias(train)
get_initial_bias(val)


# ## Tensorflow 2 - Feature Columns
# - Tensorflow feature columns are the most awesome capability released recently
# - Feature columns are the bridge between raw data and the neural network operated data
# - What kind of data a NN will operate on - Numbers, mostly floating point numbers
# - How to translate the categorical columns like the few we discussed earlier. eg Color of eye, Gender, All retail shops of a market etc
# - The rich nature of feature columns enable one to transform diverse range of raw format to NN operatable format
# - Naturally, the output of the feature column becomes the input to the model.
# 
# 
# ## One Hot Encoding
# - For Categorical features, transformation of non-number data to number data is the goal
# - Categorical variables are nominal, The process of transforming into binary value is One Hot Encoding
# - The raw data in long format is converted into wide format
# - It is nothing but, binarization of a categorical values
# 
# |Company   	|Rank   	|Price   	|   	|   	|
# |:-:	|:-:	|:-:	|:-:	|:-:	|
# |VW   	|4   	|   	|100   	|   	|
# |Honda   	|2   	|   	|10000   	|   	|
# |Ford   	|3   	|   	|1000   	|   	|
# |Tesla   	|1   	|   	|100000   	|   	|
# 
# <div align="center" style="font-size: 30px">TO</div>
# 
# |Company   	|Rank 1  	|Rank 2   	|Rank 3   	|Rank 4   	|Price   	|
# |:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|
# |VW   	|0   	|0   	|0   	|1   	|100   	|
# |Honda   	|0   	|1   	|0   	|0   	|10000   	|
# |Ford   	|0   	|0   	|1   	|0   	|1000   	|
# |Tesla   	|1   	|0   	|0   	|0   	|100000   	|
# 
# 

# ## Embeddings
# - Another popular scheme used for transformation of categorical variables into numbers is embeddings.
# - This scheme represents discrete values as continuous vectors
# - This process for machine translation yields significan improvement to the model performance
# - In an NN, embeddings are low-dimensional continuous vectors
# - The reduction of dimensionality of a categorical variable and meaningful representation in the transformed space is the goal
# - Dimensionality reduction addresses the high cardinality of the categorical value
# - Embeddings place the similar things closer in the embedding space
# 
# Examples  
# 1. Books on Data Science: Let us say there are 20000 books covering the wide gamut of all data science problems. Actual number of dimension here is 2000. By reducing the dimensonality of the dataset, From 20000 to 200 - We can represent the whole dataset.
# 2. Retail Outlets of a Country: Let us say there are 1.2 million retail outlets present in a country. 1000 odd number can represent the characteristics can represent the attributes of every outlet.
# 
# **Representation of Embedding with 2 vectors**
# <pre><code>
# shops = ["Sainsbury", "Tesco", "Reliance", "Lulu", "Costco"]  
# embeddings = [
#     [ 0.11, 0.52],  
#     [0.32, 0.56], 
#     [-0.56, -0.91], 
#     [-0.21, 0.21]
# ]
# </code></pre>                
# Here we reduced the dimensionality to 2 from 5 to represent the property of a variable

# ### Numeric Columns
# Numeric columns are pretty straight forword where the raw data is already with numeric value. Feature columns of TF2's numeric_column api comes handy to address the problem
# 

# ### Random Decisions
# - If the variation is less than 100 on the categorical columns, One Hot Encoding taken in this code
# - Beyond 100 variation, Embeddings are preferred

# In[ ]:


def handle_feature_columns(df, columns_to_remove=['id', 'target'], all_categorical_as_ohe=True):
    
    def demo(feature_column):
        feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    
    def one_hot_encode(col_name, unique_values):
        from_vocab = tf.feature_column.categorical_column_with_vocabulary_list(
            col_name, unique_values
        )
        ohe = tf.feature_column.indicator_column(from_vocab)
        data.append(ohe)
        demo(ohe)
    
    def embedd(col_name, unique_values):
        from_vocab = tf.feature_column.categorical_column_with_vocabulary_list(
            col_name, unique_values
        )
        embeddings = tf.feature_column.embedding_column(from_vocab, dimension=EMBEDDING_DIMENSIONS)
        data.append(embeddings)
        demo(embeddings)
        
    def numeric(col_name, unique_values):
        from_numeric = tf.feature_column.numeric_column(
            col_name, dtype=tf.float32
        )
        data.append(from_numeric)
        demo(from_numeric)
    
    dataframe = df.copy()
    for pop_col in columns_to_remove:
        dataframe.pop(pop_col)
    data = []
    
    for a_column in dataframe.columns:
        typee = COLUMN_TYPES[a_column]
        nunique = dataframe[a_column].nunique()
        unique_values = dataframe[a_column].unique()
        print('Column :', a_column, nunique, unique_values[:10])                
        if(typee == 'binary'):
            one_hot_encode(a_column, unique_values)
        elif(typee == 'cyclic'):
            numeric(a_column, unique_values)
            
        else:
            if(all_categorical_as_ohe):
                one_hot_encode(a_column, unique_values)
            else:
                if(typee == 'categorical'):
                    if(nunique < 100):
                        one_hot_encode(a_column, unique_values)
                    else:
                        embedd(a_column, unique_values)
                elif(typee == 'ordinal'):
                    embedd(a_column, unique_values)
            
    return data


# In[ ]:


feature_columns = handle_feature_columns(train, all_categorical_as_ohe=False)


# ### Data Preparation
# Tensors are the centrol data types for a Tensorflow NN framework. Crafting a tensors for the feature columns ensures the raw data translation into model acceptable one.
# Simply speaking, a tensor is a multi-dimensional numerical array. To get full picture of a tensor, we have to understand few key words
# - **Rank:** Number of dimension of a tensor is its Rank.
# - **Shape:** Shape of a tenser is its count of rows and columns.
#     - A rank zero tensor is a single number or it is a **scalar**
#     - A rank one tensor is an array of numbers or it is called as **vector**
#     - A rank two tensor is a matrix of numers or it has rows and columns
# - **Tensor Slice:** A tensor slice is a portion of data from the population based the batch size given

# In[ ]:


y_train = train.pop('target')
y_val = val.pop('target')


# In[ ]:


def df_to_dataset(dataframe, y, shuffle=True, batch_size=32, is_test_data=False):
    dataframe = dataframe.copy()
    ds = None
    if(is_test_data):
        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    else:
        
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), y))
        if(shuffle):
            ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

train_ds = df_to_dataset(train, y_train, shuffle=False, batch_size=BATCH_SIZE)
val_ds = df_to_dataset(val, y_val, shuffle=False, batch_size=BATCH_SIZE)
test_ds = df_to_dataset(test_df, None, shuffle=False, batch_size=BATCH_SIZE, is_test_data=True)


# ## Model, Training and Prediction
# Once the dataset is ready, a model is created, trained and evaluated.
# The current model has 6 unique items stacked one after another... {WIP}
# 
# - **Sequential Model:** 
# - **Feature Layer:**
# - **Dense Layer:**
# - **Batch Normalization:**
# - **Dropouts:**
# - **Activation Function - Relu:**
# - **Activation Function - Sigmoid:**
# - **Loss Function: Binary Cross Entropy:**
# - **From Logits:**
# - **Optimizer - Adam:**
# 

# In[ ]:


def create_silly_model_2(feature_layer, initial_bias=None):
    bias = None
    if(initial_bias):
        bias = tf.keras.initializers.Constant(initial_bias)
        
    model = tf.keras.Sequential([
        feature_layer,
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=bias)
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=METRICS
    )
    return model


# ### Fit and Run the Model
# To run the model, we have 14 items to ensure {WIP}
# - **Callbacks:**
# - **Early Stopping:**
# - **Reduce Learning on Plateau**
# - **Accuracy Monitoring:**
# - **Patience:**
# - **Baseline:**
# - **Restore Best Weights:**
# - **History:**

# In[ ]:



def run(
    train_data, val_data, feature_columns, 
    epochs=EPOCHS, es=False, rlr=False, 
    class_weights=None, initial_bias=None
):
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    model = create_silly_model_2(feature_layer, initial_bias)

    callbacks = []
    if(es):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', min_delta=0.00001, patience=5, 
                mode='auto', verbose=1, baseline=None, restore_best_weights=True
            )
        )
    if(rlr):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', factor=0.5, patience=3, 
                min_lr=3e-6, mode='auto', verbose=1
            )
        )

    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=epochs, 
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    return model, history


# In[ ]:


model, history = run(
    train_ds, val_ds, feature_columns, 
    epochs=EPOCHS, es=False, rlr=False, 
    class_weights=None, initial_bias=None
)


# In[ ]:


predictions = model.predict(test_ds)
submit = pd.DataFrame()
submit["id"] = test_df["id"]
submit['target'] = predictions
submit.to_csv('submission_dl_stratify.csv', index=False)


# ## Model Performance and Metrics
# - A model is rated through various metrics and the list we saw in the data preparation part. These metrics gives us an opportunity get insight over the model's performance.
# - The primary goal of building a model is to avoid overfit over the training data. Achieving a high accuracy on train data almost always result in poor performance over the real data
# - That is nothing but the neural network failed to learn the critical patterns hidden deeply inside the dataset.
# - Training for longer time will result in overfit. That is control over the number of epochs.
# 
# In this section we shall plot the following metrics across train and validation datasets
# - Loss
# - AUC, Area Under the Curve
# - Precision
# - Recall
# 
# Further we shall draw the ROC(Receiver Operating Curve) and observe the model performance from the training history.

# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (15, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# In[ ]:


def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(
            history.epoch, 
            history.history[metric], 
            color=colors[0], 
            label='Train'
        )
        plt.plot(
            history.epoch, 
            history.history['val_' + metric], 
            color=colors[0], 
            linestyle="--", 
            label='val'
        )
        plt.title(metric.upper())
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if(metric == 'loss'):
            plt.ylim([0, plt.ylim()[1]])
        elif(metric == 'auc'):
            plt.ylim([0, 1])
        else:
            plt.ylim([0, 1])
        plt.legend()
        
plot_metrics(history)


# ### Observations
# **Loss:**  
#     Loss of validation dataset to go down, inference: overfitting  
# **Area Under the Curve(AUC):**  
#     AUC of validation dataset to be more than train set, inference: overfitting  
# **Precision:**   
#     Precision of Validation and Train to be similar, inference: overfitting  
# **Recall:**   
#     Recall progression validation set to be along or above the train set: overfitting  

# ## ROC - Receiver Operating Characteristics
# For a binary classification problems, a ROC curve illustrates the ability of the model to predict the actual.
# - It is plotted between True Positive Rate(TPR) and False Positive Rate(FPR)
# - TPR is probability of predicting rightly
# - FPR is the probability of a false prediction
# - ROC's goal is to play as a tool to select optimal models

# In[ ]:


train_predictions = model.predict(train_ds)
val_predictions = model.predict(val_ds)


# In[ ]:


from sklearn import metrics

def roc(name, labels, predictions, **kwargs):
    fp, tp, _ = metrics.roc_curve(labels, predictions)
    
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False Positives [%]')
    plt.ylabel('True Positives [%]')
    plt.xlim([-0.5, 110])
    plt.ylim([1, 110])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.legend()


# In[ ]:


roc('Train', y_train, train_predictions, color=colors[0])
roc('Validate', y_val, val_predictions, color=colors[0], linestyle='--')


# ## Intervention 1: Early Stopping and Reduce Learning Rate on Plateau
# Let us try to fix the overfitting problem by
# - Early stop the training and
# - Reduce learning rate when a plateau is encountered

# In[ ]:


model, history = run(
    train_ds, val_ds, feature_columns, 
    epochs=EPOCHS, es=True, rlr=True, 
    class_weights=None, initial_bias=None
)


# ## Plot Metrics: Loss vs AUC vs Precision vs Recall

# In[ ]:


plot_metrics(history)


# ### Observations
# **Loss:**  
#     Loss of validation dataset got better compared to earlier, inference: overfitting is addressed to some extent  
# **Area Under the Curve(AUC):**  
#     AUC of validation dataset is increased compared to earlier, inference: overfitting is addressed to some extent  
# **Precision:**   
#     No significant change to precision, inference: overfitting  
# **Recall:**   
#     Recall of train and validation set are overalapping: overfitting  

# In[ ]:


train_predictions_es_rlr = model.predict(train_ds)
val_predictions_es_rlr = model.predict(val_ds)


# ### ROC comparison
# Let us compare the baseline model results with the model having intervention of Early Stopping and RLR

# In[ ]:



roc('Train Baseline', y_train, train_predictions, color=colors[0])
roc('Validate Baseline', y_val, val_predictions, color=colors[0], linestyle='--')
roc('Train [ES, RLR]', y_train, train_predictions_es_rlr, color=colors[1])
roc('Validate [ES, RLR]', y_val, val_predictions_es_rlr, color=colors[1], linestyle='--')


# ## Intervention 2: Class Weights and Initial Bias
# Let us try to fix further the overfitting problem by
# - Incorporating Class Weights and
# - Initial Bias

# In[ ]:


model, history = run(
    train_ds, val_ds, feature_columns, 
    epochs=EPOCHS, es=True, rlr=True, 
    class_weights=class_weight, initial_bias=initial_bias
)


# In[ ]:


plot_metrics(history)


# ### Observations
# **Loss:**  
#     Loss increased compared to earlier, inference: overfitting, model degraded  
# **Area Under the Curve(AUC):**  
#     AUC of previous and current intervention remain same, inference: no improvement  
# **Precision:**   
#     Precision decreased from earlier, inference: model degraded  
# **Recall:**   
#     Recall increased: model degraded  

# In[ ]:


train_predictions_bias_cws = model.predict(train_ds)
val_predictions_bias_cws = model.predict(val_ds)


# In[ ]:


roc('Train Baseline', y_train, train_predictions, color=colors[0])
roc('Validate Baseline', y_val, val_predictions, color=colors[0], linestyle='--')
roc('Train [ES, RLR]', y_train, train_predictions_es_rlr, color=colors[1])
roc('Validate [ES, RLR]', y_val, val_predictions_es_rlr, color=colors[1], linestyle='--')
roc('Train [ES, RLR, BIAS, IniWts]', y_train, train_predictions_bias_cws, color=colors[2])
roc('Validate [ES, RLR, BIAS, IniWts]', y_val, val_predictions_bias_cws, color=colors[2], linestyle='--')


# ## Inference of Interventions
# Intervention 1 improved the model but intervention 2 did not make any significant change to the model performance. 

# In[ ]:


predictions = model.predict(test_ds)
submit = pd.DataFrame()
submit["id"] = test_df["id"]
submit['target'] = predictions
submit.to_csv('submission_dl_final.csv', index=False)

