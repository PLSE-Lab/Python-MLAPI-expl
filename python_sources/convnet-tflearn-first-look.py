#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf #Conv Net
import tflearn
import os
import pickle
from sklearn.model_selection import train_test_split
from skimage import color
from skimage.transform import resize
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# All dependencies needed for this kernel

# In[ ]:


from subprocess import check_output
path = "D:/Ramses/Documents/GPU_TensorFlow/Kaggle/DogBreedClassifier"
os.path.exists(path)


# 1. DATA PREPROCESSING:

# In[ ]:


#CONSTANTS
image_shape = (100, 100)
channels = 3
alpha = 0.01
sample_csv = None
df_train = None
df_test = None
df_train_labels = None


# Just to have an idea of what the submission file looks like
# 
# +Reading in the training data's labels
# 
# Input data files are available in the "../input/" directory.

# In[ ]:


def get_sample_csv():
    sample = pd.read_csv(path + '/sample_submission.csv')
    return sample
def get_training_data():
    labels = pd.read_csv(path + '/labels.csv', index_col=0)
    return labels

df_train_labels = get_training_data()
sample_csv = get_sample_csv()


# tqdm is for the nice-looking progress bar
# 
# Data preprocessing and extraction:
#     * Get the image and its id
#     * Resize the image to be 100 x 100 x 3
#     * Store the image in pandas.DataFrame with id as index and 'image' as column
#     * Store the ids in a separate numpy array to be reused later

# In[ ]:


from tqdm import tqdm
#Getting the train and test data
def image_preprocessing(image):
    #image = color.rgb2gray(image)
    image = resize(image, image_shape, mode='constant')
    return image
    
def get_train_test_df(): #Maybe store in a pickle?
    train, test = pd.DataFrame(dtype=object,columns=['image']), pd.DataFrame(dtype=object, columns=['image'])
    train_ids, test_ids = np.array([], dtype=object), np.array([], dtype=object)
    for i, p in enumerate([path + '/train', path + '/test']):
        for f in tqdm(os.listdir(p)):
            _id = os.path.split(f)[-1]
            _id = str(_id.split('.', 1)[0])
            entry = plt.imread(p + '/' + f, format='jpg')
            entry = image_preprocessing(entry)
            if i is 0: 
                train = train.append({'image': entry}, ignore_index=True)
                train_ids = np.append(train_ids, _id)
            else: 
                test = test.append({'image': entry}, ignore_index=True)
                test_ids = np.append(test_ids, _id)
    return train, test, train_ids, test_ids

df_train, df_test, df_train_ids, df_test_ids = get_train_test_df()
print('Done.')


# Save the data to separate pickle files so that the extraction doesn't have to be done everytime:

# In[ ]:


for name in ['df_train', 'df_test', 'df_train_ids', 'df_test_ids']: 
    with open(path + '/' + name + '.pickle', 'wb') as f:
        pickle.dump(globals()[name], f)


# Load the data if needed:

# In[ ]:


if os.path.exists(path + '/df_train.pickle'):
    for name in ['df_train', 'df_test', 'df_train_ids', 'df_test_ids']: 
        with open(path + '/' + name + '.pickle', 'rb') as f:
            globals()[name] = pickle.load(f)


# A quick check that the data looks the way we want it to:

# In[ ]:


img = df_train.iloc[0]['image']
plt.imshow(img)


# Getting all the labels:
#     * Realizations:
#         * Data set contains too many labels with too little data to train a convnet on
#         * Other ML techniques would be possible at this stage
#         * Gathering more data would be a possibility
#         * Reducing the number of targets to a decent amount is possible
#     * Solution I chose:
#         * Divided the data into 12 targets each containing 10 sub-targets
#         * Division is done alphabetically (There are better ways but they require a deeper insight on the data)
#         * Network is only going to predict one of these 12 general-classes
#         * If I have time, I'll improve it to use a K-Nearest Neighbour algorithm
#           to determine which of the 10 subclasses the input belongs to (or maybe an array of dedicated convnets)

# In[ ]:


df_labels = []
for row in df_train_labels.itertuples():
    if row[-1] not in df_labels:
        df_labels.append(row[-1])
df_labels = np.asarray(df_labels)
df_labels = np.sort(df_labels)
df_labels = np.split(df_labels, 12)


# Quick check to see how many samples the biggest general-class contains

# In[ ]:


from collections import Counter
occurances = np.zeros(12)
values = []
for value in df_train_labels.values:
    values.append(value[0])
for index, label in enumerate(df_labels):
    for value in values:
        if value in label:
            occurances[index] += 1
print(max(occurances))


# Train-test-validation split of the data (I used 80-20 because we already don't have a lot of training data for each class)

# In[ ]:


def get_train_test_split(test_size=0.5):
    index_split = int(df_train.size * (1 - test_size))
    train_input, test_input = np.split(df_train.values, [index_split])
    train_targets, test_targets = np.split(df_train_labels.values, [index_split])
    assert len(test_input) == len(test_targets)
    return train_input, test_input, train_targets, test_targets

network_input, validation_input, network_targets, validation_targets = get_train_test_split(test_size=0.2)


# Converting the targets to be one-hot arrays to facilitate the prediction:

# In[ ]:


#make targets a one-hot array
def get_one_hot_array(array):
    one_hot = []
    for entry in array:
        for i, group in enumerate(df_labels):
            if entry[0] in group:
                index = i
                break
        zero = np.zeros(len(df_labels))
        zero[index] = 1
        one_hot.append(zero)
    return one_hot
network_targets = get_one_hot_array(network_targets)
validation_targets = get_one_hot_array(validation_targets)
network_targets[0]


# Add dimension to network_input so it becomes a 4D array and is compatible with the convnet:

# In[ ]:


def get_correct_dimensions(array):
    inputs = []
    for arr in array:
        inputs.append(arr[0])
    inputs = np.array(inputs)
    return inputs

network_input = get_correct_dimensions(network_input)
validation_input = get_correct_dimensions(validation_input)
print(network_input.shape)
print(validation_input.shape)


# 2 BUILDING THE NETWORK AND TRAINING

# Building the network:
#     * Structure:
#         * Simplest Convnet with dropout
#         * Layers:
#             - 4 Convolutional Layers
#             - 4 Fully Connected
#         * Optimizer:
#             - Adam
#     * Engine:
#         * TensorFlow (through TFLearn)
#         * Why?
#             - Simple to use
#             - Great for a first pass through the data
#             - Very intuitive to read for new comers

# In[ ]:


from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

def build_conv_net(shape, channels=5, n_features=1, learning_rate=0.1):
    inputs = input_data(shape=[None, shape[0], shape[1], channels], name='input')
    network = conv_2d(inputs, 5, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 5, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(inputs, 5, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 5, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, n_features, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network)
    return model


# Creating the model:

# In[ ]:


n_features = len(df_labels)
alpha = 0.001
convnet = build_conv_net(image_shape, channels=channels, n_features=n_features, learning_rate=alpha)


# Training of the model:

# In[ ]:


def train_network(model):
    model.fit({'input': network_input},
              {'targets': network_targets},
              n_epoch=1,
              batch_size=32,
              validation_set=({'input': validation_input}, {'targets': validation_targets}),
              snapshot_step=500,
              show_metric=True,
              run_id='dog_breed_classifier')
    model.save('dogs.model')


# In[ ]:


train_network(convnet)


# We can see the model achieves terrible accuracy but:
#     * Training was only for 1 epoch
#     * Wasn't designed to achieve 100% accuracy
#     * Data set is fairly small for each target
# So results are within expectations

# 3 TESTING THE MODEL AND CREATING SUBMISSION FILE

# Collecting the predictions for the testing data:

# In[ ]:


def test_model(model):
    try: model.load('dogs.model')
    except: print('Model not found.')
    predictions = model.predict(df_test)
    return predictions


# In[ ]:


test_results = test_model(convnet)


# Making the submission file:
#     * Store the results in a pandas.DataFrame with index the id and columns the targets
#     * Store the results in a csv using pandas.to_csv()

# In[ ]:


def make_submission_file():
    df_predictions = pd.DataFrame(index=df_test.values, columns=df_labels)
    for i, _id in enumerate(df_test_ids):
        for j, label in enumerate(df_labels):
            df_predictions.at[label, _id] = test_results[i][j]
    df_predictions.to_csv(path_or_buf='submission_file.csv')
make_submission_file()


# In[ ]:





# In[ ]:




