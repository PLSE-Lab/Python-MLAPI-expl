#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This data set consists of 8,124 labelled samples (edible or poisonous) with 22 categorical variables. As a greenhorn data scientist, I found this data set to be especially useful for practicing feature engineering with non-numeric data and working with panda dataframes. However, I did get some suspiciously good results (100% accuracy), though other Kernels I've looked at obtained similar results for various ML algorithms. I'd love for a data expert to comment on my results!
# 
# **Goal:** Classify edible mushrooms using a deep neural network.
# 
# 
# # Approach
# 1. Data visualization
# 2. Feature engineering
# 3. Neural network (NN) classification
# 4. Discussion
# 
# 
# # Data visualization
# 
#     Goals
#     - Use dual histograms to compare edible and poisonous mushrooms across all features
#     - Qualitatively guess which features are collinear or heavily skewed
# 
# Let's explore our data with histograms for each feature. Run the script below for the figures.
# 

# In[ ]:


'''
Project:    Mushroom Classification - Data Visualizer
Purpose:    Explore the Mushroom data set prior to ML

@author:    Kevin Trinh
'''

import numpy as np
from numpy.core.defchararray import add
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def histCompare(edf, pdf, feature):
    '''Plot a dual histogram of edible and poisonous mushrooms for a 
    certain feature.
    @param edf --> (pandas dataframe) a dataframe of edible mushrooms
    @param pdf --> (pandas dataframe) a dataframe of poisonous mushrooms
    @param feature --> (string) the name of the feature to be compared
    '''
    e_array = list(edf[feature])
    p_array = list(pdf[feature])
    plt.hist([e_array, p_array], color=['b', 'g'], alpha=0.5)
    plt.xlabel(feature)
    plt.title('Histogram (n = 8124)')
    plt.legend(['Edible', 'Poisonous'])
    plt.show()

# read in .csv data as pandas dataframe
mushroom_df = pd.read_csv('../input/mushroom-classification/mushrooms.csv', encoding='utf-8')

# separate dataframe by class
edible_df = mushroom_df.loc[mushroom_df['class'] == 'e']
poisonous_df = mushroom_df.loc[mushroom_df['class'] == 'p']

# obtain list of features
features = list(mushroom_df)

# generate comparative histograms for each feature
for feat in features:
    histCompare(edible_df, poisonous_df, feat)


# Here are some noteworthy observations about our data:
# 
# * We have a roughly equal number of positive (edible) and negative (poisonious) examples, so we don't need to worry about balancing each training batch with enough of both classes.
# 
# * Without any formal test for correlation, stalk characteristics above and below the ring seem to be collinear. Let's omit the stalk features below the ring.
# 
# * Veil type and veil color are heavily skewed towards one of its categories, and the edibility of mushrooms don't seem to make a difference here. Let's omit these features too.
# 
# * Odor, spore print color, gill color, gill size, stalk surface above/below ring, and bruises seem to have lots of valuable information in determining the edibility of a mushroom.
# 
# Understanding the distribution of our data helps us decide on the best features to use for our NN.
# 
# # Feature engineering
# 
#     Goals
#     - Use dummy variables to represent categorical data containing > 2 categories
#     - Use binary variables to represent features containing 2 categories
#     - Omit collinear, heavily skewed, or otherwise flawed features
#    
# 
# All categorical features (each with m categories) will be handled in one of three ways:
# 
# 1. Converted into dummy variables (m - 1 columns) 
# 2. Converted into a binary variable (1 column)
# 3. Dropped from the pandas dataframe (0 columns)
# 
# We also drop the original feature after creating new artificial features to avoid collinearity. Having redundant features will lead to overfitting.
# 
# The following script will save a .csv file that encodes the mushrooms.csv file in the following way:
# 
# **Dummy Features:** Cap shape, cap surface, cap color, odor, gill color, stalk root, stalk surface above ring, stalk color above ring, stalk color below ring, ring type, spore print color, population, and habitat.
# 
# **Binary Features:** Class, bruises, gill attachment, gill spacing, gill size, stalk shape, and ring number.
# 
# **Omitted Features:** Stalk surface below ring, veil type, veil color.

# In[ ]:


"""
Project:    Mushroom Classification -- Feature Engineering
Purpose:    - Encode categorical data
            - omit redundant and highly skewed features

@author:    Kevin Trinh
"""


import numpy as np
from numpy.core.defchararray import add
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encodeDummy(df, feature):
    '''Encode a given feature into dummy variables, omitting the first
    alphabetically-sorted category. Remove the original feature.
    
    @param df --> (pandas dataframe) dataframe to be modified
    @param feature --> (str) name of feature
    @return df --> (pandas dataframe) modified dataframe
    '''
    labels = np.unique(df[feature])
    labels = add(feature, labels)
    le = LabelEncoder()
    dummy_labels = le.fit_transform(df[feature])
    df[feature] = dummy_labels
    dummy_features = pd.get_dummies(df[feature], drop_first=True)
    df[labels[1:]] = dummy_features
    return df.drop(feature, axis=1)
    

def encodeBinary(df, feature, positive):
    '''Encode a given feature into a binary variable with 'positive' as 1 and
    all other values as 0.
    
    @param df --> (pandas dataframe) dataframe to be modified
    @param feature --> (str) name of feature
    @param positive --> (str) category to be a positive binary
    @return df --> (pandas dataframe) modified dataframe
    '''
    positive_arr = df[feature] == positive
    df.loc[positive_arr, feature] = 1
    df.loc[~positive_arr, feature] = 0
    return df

def encodeOmit(df, feature):
    '''Omit feature from dataframe.
    
    @param df --> (pandas dataframe) dataframe to be modified
    @param feature --> (str) name of feature
    @return df --> (pandas dataframe) modified dataframe
    '''
    return df.drop(feature, axis=1)


# read in .csv data as pandas dataframe
mushroom_df = pd.read_csv('mushrooms.csv', encoding='utf-8')

# select features to encode or omit
my_dummies = ['cap-shape', 'cap-surface', 'cap-color', 'odor', 'gill-color',
              'stalk-root', 'stalk-surface-above-ring', 
              'stalk-color-above-ring', 'ring-type', 'spore-print-color', 
              'population', 'habitat']

my_binaries = [('class', 'e'), ('bruises', 't'), ('gill-attachment', 'f'),
               ('gill-spacing', 'c'), ('gill-size', 'b'), ('stalk-shape', 't'), 
               ('ring-number', 'o')]

my_omissions = ['stalk-surface-below-ring', 'stalk-color-below-ring',
                'veil-type', 'veil-color']


# encode dataframe
for feat in my_dummies:
    mushroom_df = encodeDummy(mushroom_df, feat)
for feat, pos in my_binaries:
    mushroom_df = encodeBinary(mushroom_df, feat, pos)
for feat in my_omissions:
    mushroom_df = encodeOmit(mushroom_df, feat)



mushroom_df.to_csv('mushrooms_encoded.csv')


# # Neural Network
# 
#     Goals
#     - Shuffle and partition data into training, validation, and test sets
#     - Construct and train neural network
#     - After satisfactory performance on the validation set, make predictions ONLY ONCE on the test data
# 
# Now that we have feature engineered our data set, we are ready to do some machine learning.

# In[ ]:


"""
Project:    Mushroom Classification -- Neural Network
Purpose:    Construct a neural network to predict mushroom edibility

            Note: Run mushroomEncoder.py before running mushroomClassifier.py

@author:    Kevin Trinh
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.metrics import binary_crossentropy
import matplotlib.pyplot as plt


# read in and shuffle encoded data
mushroom_df = pd.read_csv('../input/mushrooms-encoded/mushrooms_encoded.csv', encoding='utf-8')
mushroom_df = mushroom_df.drop(mushroom_df.columns[0], axis=1) # omit index column
mushroom_df = mushroom_df.sample(frac=1)

# partition into training (60%), validation (20%), and test set (20%)
samples = mushroom_df.shape[0]
train_count = round(samples * 0.6)
val_count = round(samples * 0.2)
test_count = samples - train_count - val_count

train_df = mushroom_df.iloc[:train_count]
validation_df = mushroom_df.iloc[train_count:train_count + val_count]
test_df = mushroom_df.iloc[-test_count:]

X_train = train_df.drop(['class'], axis=1)
X_validation = validation_df.drop(['class'], axis=1)
X_test = test_df.drop(['class'], axis=1)

y_train = train_df['class']
y_validation = validation_df['class']
y_test = test_df['class']


### Build neural network architecture ###
num_features = mushroom_df.shape[1] - 1

model = Sequential()
model.add(Dense(16, input_dim=num_features, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(12, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid', name='output'))
  
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['binary_accuracy'])

# train NN
my_epochs = 50
history = model.fit(X_train, y_train, epochs=my_epochs, batch_size=20,
                    validation_data=(X_validation, y_validation))

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
line1 = plt.plot(epochs_arr, my_history['binary_accuracy'], 'r-', label='training accuracy')
line2 = plt.plot(epochs_arr, my_history['val_binary_accuracy'], 'b-', label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model accuracy')
plt.legend()
plt.show()


# evaluate the keras model against the test set (DO ONCE)
_, accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy: %.2f' % (accuracy*100))


# # Discussion
# 
# **Results**
# 
# My regularized deep neural network obtained a 100% (or very close to perfect) classification accuracy on my validation and test data while achieving about 97% accuracy on my training set. Oddly, my validation and test accuracy outperformed my training accuracy, contradicting the idea that ML algorithms are optimized for its given training data.
# 
# If I remove my drop out layers (i.e. don't apply regularization to my model), then my training, validation, and test sets all achieve ~100% accuracy. The regularization appears to slow down my models convergence to 100% accuracy for all data sets, but convergence happens fast (i.e. small number of epochs) nonetheless.
#     
# 
# **Interpretation**
# 
# I was pretty skeptical about my 100% classification accuracy on my validation and test set, but after comparing with other kernels, I've noticed that such a high accuracy is not uncommon on the mushroom classification data set given the right ML algorithm. However, there still is the issue of having a lower training accuracy than that of my validation and test set. I have a possible explanation for this weird result, but I would love to see what others think!
# 
# Perhaps the mushroom data set is inherently easy to classify edible mushrooms (i.e. various independent features are strongly correlated with the output label) such that a good ML algorithm is bound to have ~100% accuracy. However, when we introduce drop out layers, we are randomly disabling neurons which makes it artificially harder for our model to learn from our training data. This loss of information puts a cap on how well our model can perform on our training data.
# 
# In general, regularization should decrease performance on our training data to match that of unseen data, but this decrease in training accuracy is only favorable when validation and test accuracy is not extraordinarily high.
# 
# Note: I doubt that I'm overfitting my training data because 1) I'm applying regularization and 2) an overfit model should have a high training accuracy and low validation/test accuracy.
# 
# 
