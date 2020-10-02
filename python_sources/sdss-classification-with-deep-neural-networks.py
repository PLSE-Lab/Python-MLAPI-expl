#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Founded by the University of Chicago, the [Sloan Digital Sky Survey (SDSS)](http://https://www.sdss.org/) uses a wide-angle optical telescope at the Apache Point Observatory in New Mexico, United States to create multi-colored 3D maps of a third of our sky. The survey includes spectral information on all detected astronomical objects which include stars, galaxies, and quasars. Data is periodically and publically released. This notebook will cover Data Release 16 (DR16).
# 
# This data set contains 100,000 labelled samples and 17 features. However, clever feature engineering allows me to reduce our number of features and yield classification accuracies above 98%. For this exercise, I'll use a deep neural network (DNN) to classify stars, galaxies, and quasars in the SDSS DR16 data set.
# 
# Note: I've also created a random forest model which is more accuracte and faster to train than a DNN in a separate kernel.
# 
# 
# **Approach**
# 1. Data exploration
# 2. Feature Engineering
# 3. Construct and train DNN
# 4. Evaluate DNN
# 
# For code readability, I like to separate my data exploration (step 1) and ML Python scripts (steps 2-4).
# 
# # Data exploration
# 
# Off the bat, I recognized that there are a few features that bear no physical significance whatsoever: objid, specobjid, run, rerun, camcol, and field. I'll omit these features from further discussion.
# 
# Let's generate the following figures:
# * Pie chart of labels
# * Equitorial coordinates of observations
# * Histogram distributions of redshift, plate, modified julien day, and fiber id
# * Histogram distributions of Thuan-Gunn Astronomic Magnitude system

# In[ ]:


"""
Project:    Classification of Sloan Digital Sky Survey (SDSS) Objects
Purpose:    Data Exploration

@author:    Kevin Trinh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pieChart(sdss_df):
    '''Plot a pie chart for label count.'''
    label_counts = sdss_df['class'].value_counts()
    colors = ['skyblue', 'red', 'gold']
    fig1, ax1 = plt.subplots()
    ax1.pie(label_counts, labels=['Galaxy', 'Stars', 'Quasars'],
            autopct='%1.2f%%', startangle=45, colors=colors)
    ax1.axis('equal')
    plt.title('SDSS Object Classes')
    plt.show()

def distribution(sdss_df, axes, feature, row):
    '''Plot the distribution of a space object w.r.t. a given feature.'''
    labels = np.unique(sdss_df['class'])
    colors = ['skyblue', 'gold', 'red']
    for i in range(len(labels)):
        label = labels[i]
        ax = sns.distplot(sdss_df.loc[sdss_df['class']==label, feature], 
                          kde=False, bins=30, ax=axes[row, i], color=colors[i])
        ax.set_title(label)
        if (i == 0):
            ax.set(ylabel='Count')
            
def equitorial(sdss_df, row):
    '''Plot equitorial coordinates of observations.'''
    labels = np.unique(sdss_df['class'])
    colors = ['skyblue', 'gold', 'red']
    label = labels[row]
    sns.lmplot(x='ra', y='dec', data=sdss_df.loc[sdss_df['class']==label],
               hue='class', palette=[colors[row]], scatter_kws={'s': 2}, 
               fit_reg=False, height=4, aspect=2)
    plt.ylabel('dec')
    plt.title('Equitorial coordinates')
    

def main():

    # read in SDSS data
    filepath = '../input/sloan-digital-sky-survey-dr16/Skyserver_12_30_2019 4_49_58 PM.csv'
    sdss_df = pd.read_csv(filepath, encoding='utf-8')

    # define lists of relevant features
    geo = ['ra', 'dec']
    nonugriv = ['redshift', 'plate', 'mjd', 'fiberid']
    ugriv = ['u', 'g', 'r', 'i', 'z']

    # plot pie chart of label count
    pieChart(sdss_df)

    # plot equitorial coordinates of observations
    for row in range(3):
        equitorial(sdss_df, row)
        plt.show()
    
    # plot the distribution of non-geo and non-ugriv features
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 14))
    plt.subplots_adjust(wspace=.4, hspace=.4)
    for row in range(len(nonugriv)):
        feat = nonugriv[row]
        distribution(sdss_df, axes, feat, row)
    plt.show()
        
    # plot the distribution of ugriv features
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 15))
    plt.subplots_adjust(wspace=.4, hspace=.4)
    for row in range(len(ugriv)):
        feat = ugriv[row]
        distribution(sdss_df, axes, feat, row)
    plt.show()

main()


# I noticed a few interesting things about my figures.
# 
# First, my dataset consists of mostly galaxies and stars, though quasars are a sizeable minority (10.58% of samples). A skilled data scientist (which I am not) may factor this in when creating curated batches for ML models to improve model performance.
# 
# Second, the equitorial coordinates of stars are distributed differently than that of galaxies and quasars. This could be due to the Milky Way's shape as a barred spiral galaxy. We are more likely to see stars from our own solar system if we make observations along the plane of our galaxy. These coordinates could be useful in separating stars from other classes.
# 
# Third, redshift appears to be very telling of the observed object's class. The axes of each class are drastically different. This makes sense because redshift results from observing objects travelling away from the observer at fast speeds. Stars travel away from us fast; galaxies travel away from us at faster rates; quasars travel away from us at even faster rates.
# 
# Fourth, the bulk of galaxies observed have low plate numbers and fiber IDs relative to quasars and stars. 
# 
# Fifth, the distributions among classes for the ugriv variables look the same, and differences between classes are present but subtle. It is worth noting that each ugriv feature (i.e. band) is ordinally related to each other, so some additional processing of these features should be done prior to any ML.
# 
# 
# # Machine Learning (Training)
# 
# Note: The below script is broken into two parts: 1) feature engineering, training, and validation, and 2) plots and predictions.
# 
# 
# I shuffled, partitioned, and reformated my data to feed into my DNN. Since I had three different classes, I used one-hot encoding to represent my labels. I scaleed all features and used three principal component analysis (PCA) features to replace the u, g, r, i, and z features.
# 
# Next, I trained my DNN using 50 epochs which was enough for convergence. However, run-time was quite long. Fine-tuning my model eventually led to a validation accuracy of 98.4%.

# In[ ]:


"""
Project:    Classification of Sloan Digital Sky Survey (SDSS) Objects
Phase:      Feature engineering and ML classification

Algorithm: Deep neural network (DNN)

Steps:      1) Import libraries
            2) Read, shuffle, and partition data
            3) Restructure data as inputs for DNN
            4) Feature Engineering
            5) Create and train DNN
            6) Make predictions on validation sets
            7) Fine-tune models for highest performance on validation set
            8) Make predictions on test set
            9) Evaluate model with confusion matrix

@author:    Kevin Trinh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.utils import np_utils
import seaborn as sns


# read in and shuffle SDSS data
filename = '../input/sloan-digital-sky-survey-dr16/Skyserver_12_30_2019 4_49_58 PM.csv'
sdss_df = pd.read_csv(filename, encoding='utf-8')
sdss_df = sdss_df.sample(frac=1)

# drop physically insignificant columns
sdss_df = sdss_df.drop(['objid', 'specobjid', 'run', 'rerun', 'camcol',
                        'field'], axis=1)


# partition SDSS data (60% train, 20% validation, 20% test)
train_count = 60000
val_count = 20000
test_count = 20000

train_df = sdss_df.iloc[:train_count]
validation_df = sdss_df.iloc[train_count:train_count+val_count]
test_df = sdss_df.iloc[-test_count:]


# obtain feature dataframes
X_train = train_df.drop(['class'], axis=1)
X_validation = validation_df.drop(['class'], axis=1)
X_test = test_df.drop(['class'], axis=1)

# one-hot encode labels for DNN
le = LabelEncoder()
le.fit(sdss_df['class'])
encoded_Y = le.transform(sdss_df['class'])
onehot_labels = np_utils.to_categorical(encoded_Y)

y_train = onehot_labels[:train_count]
y_validation = onehot_labels[train_count:train_count+val_count]
y_test = onehot_labels[-test_count:]

# scale features
scaler = StandardScaler()
scaler.fit(X_train) # fit scaler to training data only
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_validation = pd.DataFrame(scaler.transform(X_validation), columns=X_validation.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_validation.columns)

# apply principal component analysis to wavelength intensities
pca = PCA(n_components=3)
dfs = [X_train, X_validation, X_test]
for i in range(len(dfs)):
    df = dfs[i]
    ugriz = pca.fit_transform(df[['u', 'g', 'r', 'i', 'z']])
    df = pd.concat((df, pd.DataFrame(ugriz)), axis=1)
    df.rename({0: 'PCA1', 1: 'PCA2', 2: 'PCA3'}, axis=1, inplace=True)
    df.drop(['u', 'g', 'r', 'i', 'z'], axis=1, inplace=True)
    dfs[i] = df
X_train, X_validation, X_test = dfs

# create a deep neural network model
num_features = X_train.shape[1]
dnn = Sequential()
dnn.add(Dense(9, input_dim=num_features, activation='relu'))
dnn.add(Dropout(0.1))
dnn.add(Dense(9, activation='relu'))
dnn.add(Dropout(0.1))
dnn.add(Dense(9, activation='relu'))
dnn.add(Dropout(0.05))
dnn.add(Dense(6, activation='relu'))
dnn.add(Dropout(0.05))
dnn.add(Dense(6, activation='relu'))
dnn.add(Dense(6, activation='relu'))
dnn.add(Dense(3, activation='softmax', name='output'))

dnn.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['categorical_accuracy'])

# train DNN
my_epochs = 50
history = dnn.fit(X_train, y_train, epochs=my_epochs, batch_size=50,
                    validation_data=(X_validation, y_validation))


# In[ ]:


# plot model loss while training
epochs_arr = np.arange(1, my_epochs + 1, 1)
my_history = history.history
line1 = plt.plot(epochs_arr, my_history['loss'], 'r-', label='training loss')
line2 = plt.plot(epochs_arr, my_history['val_loss'], 'b-', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model loss')
plt.legend()
plt.show()

# plot model accuracy while training
line1 = plt.plot(epochs_arr, my_history['categorical_accuracy'], 'r-', label='training accuracy')
line2 = plt.plot(epochs_arr, my_history['val_categorical_accuracy'], 'b-', label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model accuracy')
plt.legend()
plt.show()

preds = pd.DataFrame(dnn.predict(X_validation))
preds = preds.idxmax(axis=1)
y_validation = y_validation.dot([0,1,2])
model_acc = (preds == y_validation).sum().astype(float) / len(preds) * 100

print('Deep Neural Network')
print('Validation Accuracy: %3.5f' % (model_acc))


# # Machine Learning (Testing)
# 
# I then evaluated my model against the test set and plot a confusion matrix. Unlike my training and validation set, I ran the following code ONCE for good practice so that my test data can be truly considered 'unseen' data.
# 
# My model yielded a test accuracy of 98.8% which is marginally better than my validation performance. Furthermore, my confusion matrix reveals some interesting insights about my model:
# 
# * Galaxies have a precision of 99.4% and recall of 99.2%.
# * Quasars have a precision of 96.6% and recall of 97.7%.
# * Stars have a precision of 99.9% and recall of 98.5%.
# 
# In other words, my model performs much better on galaxies and stars than quasars. This makes sense because quasars are the minority in my 100,000 samples.
# 

# In[ ]:


# make predictions on test set (DO ONLY ONCE)
preds = pd.DataFrame(dnn.predict(X_test))
preds = preds.idxmax(axis=1)
y_test = y_test.dot([0,1,2])
model_acc = (preds == y_test).sum().astype(float) / len(preds) * 100

print('Deep Neural Network')
print('Test Accuracy: %3.5f' % (model_acc))

# plot confusion matrix
labels = np.unique(sdss_df['class'])

ax = plt.subplot(1, 1, 1)
ax.set_aspect(1)
plt.subplots_adjust(wspace = 0.3)
sns.heatmap(confusion_matrix(y_test, preds), annot=True,
                  fmt='d', xticklabels = labels, yticklabels = labels,
                  cbar_kws={'orientation': 'horizontal'})
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Deep Neural Network')
plt.show()


# # Conclusion
# 
# My DNN took quite a while to train, but I was able to reach a fairly strong classification accuracy (98.8%). However, there are other classification algorithms that can achieve better results faster, namely the random forest classifier (I'll link another kernel shortly).
# 
# If you'd like to check out my other Kaggle kernels or learn my about my background, feel free to check out my [website](http://www.kevinttrinh.com/data-science/)!
