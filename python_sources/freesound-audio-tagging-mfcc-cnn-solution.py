#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import signal
import random
from sklearn.model_selection import StratifiedKFold
import sklearn
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/freesound-audio-tagging-2019"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Author: Trent J. Bradberry <trentjason@hotmail.com>
# License: BSD 3 clause

import numpy as np

from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples, check_array
from sklearn.utils.multiclass import type_of_target

from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits,     BaseShuffleSplit, _validate_shuffle_split


def IterativeStratification(labels, r, random_state):
    """This function implements the Iterative Stratification algorithm described
    in the following paper:
    Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of
    Multi-Label Data. In: Gunopulos D., Hofmann T., Malerba D., Vazirgiannis M.
    (eds) Machine Learning and Knowledge Discovery in Databases. ECML PKDD
    2011. Lecture Notes in Computer Science, vol 6913. Springer, Berlin,
    Heidelberg.
    """

    n_samples = labels.shape[0]
    test_folds = np.zeros(n_samples, dtype=int)

    # Calculate the desired number of examples at each subset
    c_folds = r * n_samples

    # Calculate the desired number of examples of each label at each subset
    c_folds_labels = np.outer(r, labels.sum(axis=0))

    labels_not_processed_mask = np.ones(n_samples, dtype=bool)

    while np.any(labels_not_processed_mask):
        # Find the label with the fewest (but at least one) remaining examples,
        # breaking ties randomly
        num_labels = labels[labels_not_processed_mask].sum(axis=0)

        # Handle case where only all-zero labels are left by distributing
        # across all folds as evenly as possible (not in original algorithm but
        # mentioned in the text). (By handling this case separately, some
        # code redundancy is introduced; however, this approach allows for
        # decreased execution time when there are a relatively large number
        # of all-zero labels.)
        if num_labels.sum() == 0:
            sample_idxs = np.where(labels_not_processed_mask)[0]

            for sample_idx in sample_idxs:
                fold_idx = np.where(c_folds == c_folds.max())[0]

                if fold_idx.shape[0] > 1:
                    fold_idx = fold_idx[random_state.choice(fold_idx.shape[0])]

                test_folds[sample_idx] = fold_idx
                c_folds[fold_idx] -= 1

            break

        label_idx = np.where(num_labels == num_labels[np.nonzero(num_labels)].min())[0]
        if label_idx.shape[0] > 1:
            label_idx = label_idx[random_state.choice(label_idx.shape[0])]

        sample_idxs = np.where(np.logical_and(labels[:, label_idx].flatten(), labels_not_processed_mask))[0]

        for sample_idx in sample_idxs:
            # Find the subset(s) with the largest number of desired examples
            # for this label, breaking ties by considering the largest number
            # of desired examples, breaking further ties randomly
            label_folds = c_folds_labels[:, label_idx]
            fold_idx = np.where(label_folds == label_folds.max())[0]

            if fold_idx.shape[0] > 1:
                temp_fold_idx = np.where(c_folds[fold_idx] ==
                                         c_folds[fold_idx].max())[0]
                fold_idx = fold_idx[temp_fold_idx]

                if temp_fold_idx.shape[0] > 1:
                    fold_idx = fold_idx[random_state.choice(temp_fold_idx.shape[0])]

            test_folds[sample_idx] = fold_idx
            labels_not_processed_mask[sample_idx] = False

            # Update desired number of examples
            c_folds_labels[fold_idx, labels[sample_idx]] -= 1
            c_folds[fold_idx] -= 1

    return test_folds


class MultilabelStratifiedKFold(_BaseKFold):
    """Multilabel stratified K-Folds cross-validator
    Provides train/test indices to split multilabel data into train/test sets.
    This cross-validation object is a variation of KFold that returns
    stratified folds for multilabel data. The folds are made by preserving
    the percentage of samples for each label.
    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    shuffle : boolean, optional
        Whether to shuffle each stratification of the data before splitting
        into batches.
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Unlike StratifiedKFold that only uses random_state
        when ``shuffle`` == True, this multilabel implementation
        always uses the random_state since the iterative stratification
        algorithm breaks ties randomly.
    Examples
    --------
    >>> from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    >>> import numpy as np
    >>> X = np.array([[1,2], [3,4], [1,2], [3,4], [1,2], [3,4], [1,2], [3,4]])
    >>> y = np.array([[0,0], [0,0], [0,1], [0,1], [1,1], [1,1], [1,0], [1,0]])
    >>> mskf = MultilabelStratifiedKFold(n_splits=2, random_state=0)
    >>> mskf.get_n_splits(X, y)
    2
    >>> print(mskf)  # doctest: +NORMALIZE_WHITESPACE
    MultilabelStratifiedKFold(n_splits=2, random_state=0, shuffle=False)
    >>> for train_index, test_index in mskf.split(X, y):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 3 4 6] TEST: [1 2 5 7]
    TRAIN: [1 2 5 7] TEST: [0 3 4 6]
    Notes
    -----
    Train and test sizes may be slightly different in each fold.
    See also
    --------
    RepeatedMultilabelStratifiedKFold: Repeats Multilabel Stratified K-Fold
    n times.
    """

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(MultilabelStratifiedKFold, self).__init__(n_splits, shuffle, random_state)

    def _make_test_folds(self, X, y):
        y = np.asarray(y, dtype=bool)
        type_of_target_y = type_of_target(y)

        if type_of_target_y != 'multilabel-indicator':
            raise ValueError(
                'Supported target type is: multilabel-indicator. Got {!r} instead.'.format(type_of_target_y))

        num_samples = y.shape[0]

        rng = check_random_state(self.random_state)
        indices = np.arange(num_samples)

        if self.shuffle:
            rng.shuffle(indices)
            y = y[indices]

        r = np.asarray([1 / self.n_splits] * self.n_splits)

        test_folds = IterativeStratification(labels=y, r=r, random_state=rng)

        return test_folds[np.argsort(indices)]

    def _iter_test_masks(self, X=None, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.
        y : array-like, shape (n_samples, n_labels)
            The target variable for supervised learning problems.
            Multilabel stratification is done based on the y labels.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super(MultilabelStratifiedKFold, self).split(X, y, groups)


# In[ ]:


def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
                                                                truth[nonzero_weight_sample_indices, :] > 0, 
                                                                scores[nonzero_weight_sample_indices, :], 
                                                                sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap


# # Models

# In[ ]:


# models
from keras import losses, models
from keras.models import Sequential, Model
from keras.layers import (Input, Conv2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dense,
                          GlobalMaxPooling2D, MaxPooling2D, concatenate, Activation, ZeroPadding2D, Dropout, add, ReLU)
from keras.callbacks import (LearningRateScheduler, ReduceLROnPlateau)
from keras.optimizers import Adam, SGD
from keras.regularizers import l2

def cnn_6(lr, do=0, l2reg=0):
    
    inp = Input(shape=(64,128,1))
    
    x = BatchNormalization()(inp)
    x = ReLU()(x)
    x = Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), kernel_regularizer=l2(l2reg), padding='same')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), kernel_regularizer=l2(l2reg), padding='same')(x)
    
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), kernel_regularizer=l2(l2reg), padding='same')(x)
    
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), kernel_regularizer=l2(l2reg), padding='same')(x)
    
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), kernel_regularizer=l2(l2reg), padding='same')(x)
    
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), kernel_regularizer=l2(l2reg), padding='same')(x)
    
    x = GlobalMaxPooling2D()(x)
    #x = Dense(2048, activation='relu')(x)
    #x = Dropout(do)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(do)(x)
    out = Dense(80, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=out)
    opt = Adam(lr=lr)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[])
    return model


# In[ ]:


# Creates parent-label mappings.
#===========================================================================================================================================
# For splitting with MultilabelStratifiedKFold (iterative stratification):

# 1. 
# 'train_labels_dict': dict, stores actual labels of each parent file, where each label may contain more than one positive class.
# key             : value
# parent_filename : complete_label
# '0c28d31c.wav'  : [0,0,1,0, ..., 1] (1D array of length 80, with num positive classes >= 1)

# 2.
# 'all_files': array, 1D (N), stores names of parent files. For splitting using indices by StratifiedKFold and MultilabelStratifiedKFold.
# [parent_filename1, parent_filename2, ...]
# [ '0006ae4e.wav',   '0019ef41.wav',  ...]
# also used for approximate stratification.

# 3.
# 'y_split_iterative': array, 2D (N, C), stores the complete label of each file in 'all_files' (order must equal to that of all_files)
# [ [0,0,1,...,0] 
#   [0,1,0,...,1]
#       ...
#   [1,0,1,...,0] ]
#===========================================================================================================================================
# For splitting with StratifiedKFold (approximate stratification)

# 4. 
# 'labels_for_splitting': dict, stores a random positive element from each parent file's label. For stratification approximation purposes.
# key              : value
# parent_filename  : single_label
# '0c28d31c.wav_0' : [0,0,1,0, ..., 0] (1D array of length 80, with num positive classes == 1)

# 5. y_split: array, 2D (N, C), stores the label from labels_for_splitting (single-class labels only) of each file in 'all_files' 
# (order must equal to that of all_files)
# [ [0,0,1,...,0] 
#   [0,1,0,...,0]
#       ...
#   [1,0,0,...,0] ]
#===========================================================================================================================================
# ACTUAL LABELS:
# stored in train-Labels_dict

# load train
train = pd.read_csv("../input/freesound-audio-tagging-2019/train_curated.csv") # just change this target to train_noisy if needed
print(train.head(10))
print()

# binarize labels
labels = train['labels'].tolist()
split_labels = [] # create a list of sets
for label in labels:
    split_labels.append(set(label.split(',')))
mlb = MultiLabelBinarizer()
onehot_labels = mlb.fit_transform(split_labels)
print('train set labels shape:', onehot_labels.shape)
print('found', len(mlb.classes_), 'unique classes')
print()

# create dict mapping between filenames and labels
train_labels_dict = {}
for i, filename in enumerate(train['fname'].tolist()):
    train_labels_dict[filename] = onehot_labels[i]

# explore distribution of number of labels
print('distribution:')
n_labels = np.sum(onehot_labels, axis=1)
n_labels = pd.Series(n_labels)
for series_name, series in n_labels.groupby(n_labels):
    print(series_name, 'labels:', len(series))
#===========================================================================================================================================
# LABELS FOR SPLITTING
# stored in labels_for_splitting

# split labels into sets
labels = train['labels'].tolist()
split_labels = [] # create a list of sets
for label in labels:
    split_labels.append(set(label.split(',')))

# create dict mapping between filenames and labels
labels_complete = {}
for i, filename in enumerate(train['fname'].tolist()):
    labels_complete[filename] = split_labels[i]

# pick a random label for every training sample
random.seed(1001)
labels_one = {}
for file, label_set in labels_complete.items():
    labels_one[file] = random.sample(label_set, 1)

# binarized version
single_labels = []
for file, label in labels_one.items():
    single_labels.append(set(label))
single_labels = mlb.transform(single_labels)
labels_for_splitting = {}
for i, file in enumerate(train['fname'].tolist()):
    labels_for_splitting[file] = single_labels[i]
#===========================================================================================================================================
# FILE NAMES LIST
# saved in all_files
# y_split for stratification approximation
all_files = np.empty(len(train), dtype='U12')
y_split = np.zeros((len(train), len(mlb.classes_)))
i = 0
for file, label in labels_for_splitting.items():
    all_files[i] = file
    y_split[i] = label
    i += 1
y_split = np.argmax(y_split, axis=-1)
#===========================================================================================================================================
# y_split_iterative for stratification using iterative approach
y_split_iterative = np.zeros((len(train), len(mlb.classes_)))
i = 0
for file, label in train_labels_dict.items():
    y_split_iterative[i] = label
    i += 1


# In[ ]:


# Stores MFCC arrays of children files.
#============================================================================================================================================
# 1. 
# 'training_data': list, len N, stores each child's (filename, MFCC array, label) of training data.
# [(filename1, MFCC_array1, label1), (filename2, MFCC_array2, label2), ...]
# shuffles the list after constructing it.

# 2.
# 'test_data': list, len N_test, stores each child's (filename, MFCC array) of test data.
# [(filename1, MFCC_array1), (filename2, MFCC_array2), ...]
#============================================================================================================================================
# Config variables and utilities

SAMPLE_DURATION = 3.0
SAMPLE_RATE = 44100
L = int(SAMPLE_RATE * SAMPLE_DURATION) # L = number of samples in a training example before padding

def random_slice(samples):
    begin = np.random.randint(0, len(samples) - L)
    return samples[begin : begin+L]

def pad(samples):
    if len(samples) >= L: 
        return samples
    else: 
        return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))
#===========================================================================================================================================
# TRAINING DATA
# splits each sample into 3-second chunks, and uses all of them for training data

training_data = []
progress = 0
for file_name in os.listdir('../input/freesound-audio-tagging-2019/train_curated/'):
    
    # read data array indicated by 'file_name' and pick a random 1s slice within it
    file_data, _ = librosa.core.load('../input/freesound-audio-tagging-2019/train_curated/' + file_name, sr=SAMPLE_RATE)
    duration = librosa.core.get_duration(file_data, sr=SAMPLE_RATE)
    
    # split array into slices of duration SAMPLE_DURATION
    for i in range(math.ceil(duration/SAMPLE_DURATION)):
        begin = i*L
        file_data_i = file_data[begin:begin+L]
        if len(file_data_i) < L:
            file_data_i = pad(file_data_i)
        
        file_array_i = librosa.feature.mfcc(file_data_i, 
                                            sr=SAMPLE_RATE, 
                                            n_fft=2560, 
                                            hop_length=347*int(SAMPLE_DURATION), 
                                            n_mels=128, 
                                            n_mfcc=64)
        training_data.append((file_name+'_'+str(i), file_array_i, train_labels_dict[file_name]))
    
    # progress tracker
    if progress%100 == 0:
        print('processed', progress, 'files')
    progress += 1
print('train data done')

# shuffle order of data
random.Random(1001).shuffle(training_data)
#===========================================================================================================================================
# do the same for test data: no need to do this for model selection
test = pd.read_csv("../input/freesound-audio-tagging-2019/sample_submission.csv")
test_data = []
progress = 0
for file_name in test['fname'].tolist():
    
    # read data array indicated by 'file_name' and pick a random 1s slice within it
    file_data, _ = librosa.core.load('../input/freesound-audio-tagging-2019/test/' + file_name, sr=SAMPLE_RATE)
    duration = librosa.core.get_duration(file_data, sr=SAMPLE_RATE)
    
    # split array into slices of duration SAMPLE_DURATION
    for i in range(math.ceil(duration/SAMPLE_DURATION)):
        begin = i*L
        file_data_i = file_data[begin:begin+L]
        if len(file_data_i) < L:
            file_data_i = pad(file_data_i)
        
        file_array_i = librosa.feature.mfcc(file_data_i, 
                                            sr=SAMPLE_RATE, 
                                            n_fft=2560, 
                                            hop_length=347*int(SAMPLE_DURATION), 
                                            n_mels=128, 
                                            n_mfcc=64)
        test_data.append((file_name+'_'+str(i), file_array_i))
    
    # progress tracker
    if progress%100 == 0:
        print('processed', progress, 'files')
    progress += 1
print('test data done')


# # Utilities

# In[ ]:


# utilities to get X, y, and arithmetic and geometric averaging.
# consult function descriptions for clarification.
#===========================================================================================================================================
# get train data
def get_train(train_files):
    
    '''
    Constructs the X_train and y_train matrices of [children files], given an array/list of [parent filenames] for training.
    
    Input(s) : 'train_files': an array/list of [parent filenames] for training (len N_train_parents), obtained from StratifiedKFold.
    
    Output(s): 'X_train': the training matrix, shape (N_train_children, MFCC_array.shape[0], MFCC_array.shape[1])
               'y_train': the training labels, shape (N_train_children, C)
    '''
    
    # get the data in one list
    # format: [(file_name, file_array, file_label), ...]
    datalist = []
    for file_name, file_array, file_label in training_data:
        if file_name.split('_')[0] in train_files:
            datalist.append((file_name, file_array, file_label))
        else:
            continue
    
    # shuffle
    random.Random(1001).shuffle(datalist)
    
    # get X and y
    X_train = np.empty((len(datalist), datalist[0][1].shape[0], datalist[0][1].shape[1]))
    y_train = np.empty((len(datalist), len(mlb.classes_)))
    i = 0
    for file_name, file_array, file_label in datalist:
        X_train[i] = file_array
        y_train[i] = file_label
        i += 1

    return X_train, y_train
    
# get val data
def get_val(val_files):
    
    '''
    Constructs the X_val and y_val matrices of [children files], given an array/list of [parent filenames] for validation.
    
    Input(s) : 'val_files': an array/list of [parent filenames] for validation (len N_val_parents), obtained from StratifiedKFold.
    
    Output(s): 'X_val': the validation matrix, shape (N_val_children, MFCC_array.shape[0], MFCC_array.shape[1])
               'y_val': the validation labels, shape (N_val_children, C)
               'filenames_val': 1D array, shape (N_val_children,), stores children filenames for geometric averaging. 
                                Must be in the same order as X_val.
    '''
    
    # get the data in one list
    # format: [(file_name, file_array, file_label), ...]
    datalist = []
    for file_name, file_array, file_label in training_data:
        if file_name.split('_')[0] in val_files:
            datalist.append((file_name, file_array, file_label))
        else:
            continue
    
    # get filenames, X, and y
    filenames_val = np.empty((len(datalist)), dtype=object)
    X_val = np.empty((len(datalist), datalist[0][1].shape[0], datalist[0][1].shape[1]))
    y_val = np.empty((len(datalist), len(mlb.classes_)))
    i = 0
    for file_name, file_array, file_label in datalist:
        filenames_val[i] = file_name
        X_val[i] = file_array
        y_val[i] = file_label
        i += 1

    return filenames_val, X_val, y_val

# get test data
def get_test():
    
    '''
    Constructs the X_train matrix of [children files], given an array/list of [parent filenames] for test data.
    
    Input(s) : None, takes data from 'test_data'
    
    Output(s): 'X_test': the test data matrix, shape (N_test_children, MFCC_array.shape[0], MFCC_array.shape[1])
               'filenames_test': 1D array, shape (N_test_children,), stores children filenames for geometric averaging. 
                                Must be in the same order as X_test.
    '''
    
    # get filenames and X
    filenames_test = np.empty((len(test_data)), dtype=object)
    X_test = np.empty((len(test_data), test_data[0][1].shape[0], test_data[0][1].shape[1]))
    i = 0
    for file_name, file_array in test_data:
        filenames_test[i] = file_name
        X_test[i] = file_array
        i += 1

    return filenames_test, X_test

def arithmetic_average_val_scores(filenames_val, y_pred, y_true):
    
    '''
    Calculates the arithmetic average of the [prediction of a parent file], given a [collection of predictions of children files]
    
    Input(s): 'filenames_val': 1D array, shape (N_val_children,), stores children filenames.
                               Must be in the same order as y_true and y_pred.
                               Output from get_val().
              'y_pred': 2D array, shape (N_val_children, C), stores predictions of each child file.
              'y_true': 2D array, shape (N_val_children, C), stores ground truth labels of each child file.
              
    Output(s): 'y_pred_parents': 2D array, shape (N_val_parents, C), stores predictions of each parent file by averaging from children.
               'y_true_parents': 2D array, shape (N_val_parents, C), stores ground truth labels of each parent file.
    '''
    
    # format:
    # filename: (sum_array, label, num_children)
    sum_scores = {}
    for i, filename in enumerate(filenames_val):
        
        split_filename = filename.split('_')[0]
        
        if split_filename not in sum_scores.keys():
            sum_scores[split_filename] = [y_pred[i], y_true[i], 1]
        else:
            sum_scores[split_filename][0] += y_pred[i]
            sum_scores[split_filename][2] += 1
    
    # average the scores
    y_pred_parents = np.empty((len(sum_scores.keys()), len(mlb.classes_)))
    y_true_parents = np.empty((len(sum_scores.keys()), len(mlb.classes_)))
    i = 0
    for filename, values in sum_scores.items():
        y_pred_parents[i] = values[0]/values[2]
        y_true_parents[i] = values[1]
        i += 1
    
    # return averaged scores
    return y_pred_parents, y_true_parents

def geometric_average_val_scores(filenames_val, y_pred, y_true):
    
    '''
    Calculates the geometric average of the [prediction of a parent file], given a [collection of predictions of children files]
    
    Input(s): 'filenames_val': 1D array, shape (N_val_children,), stores children filenames
                               Must be in the same order as y_true and y_pred.
                               Output from get_val().
              'y_pred': 2D array, shape (N_val_children, C), stores predictions of each child file.
              'y_true': 2D array, shape (N_val_children, C), stores ground truth labels of each child file.
              
    Output(s): 'y_pred_parents': 2D array, shape (N_val_parents, C), stores predictions of each parent file by averaging from children.
               'y_true_parents': 2D array, shape (N_val_parents, C), stores ground truth labels of each parent file.
               'filenames_parents': list, len(N_val_parents), stores the names of the parents in the same order as y_pred_parents
    '''
    
    # format:
    # filename: (sum_array, label, num_children)
    sum_scores = {}
    for i, filename in enumerate(filenames_val):
        
        split_filename = filename.split('_')[0]
        
        if split_filename not in sum_scores.keys():
            sum_scores[split_filename] = [y_pred[i], y_true[i], 1]
        else:
            sum_scores[split_filename][0] *= y_pred[i]
            sum_scores[split_filename][2] += 1
    
    # average the scores
    filenames_parents = []
    y_pred_parents = np.empty((len(sum_scores.keys()), len(mlb.classes_)))
    y_true_parents = np.empty((len(sum_scores.keys()), len(mlb.classes_)))
    i = 0
    for filename, values in sum_scores.items():
        filenames_parents.append(filename)
        y_pred_parents[i] = values[0]**(1.0/values[2])
        y_true_parents[i] = values[1]
        i += 1
    
    # return averaged scores
    return filenames_parents, y_pred_parents, y_true_parents


# # Training, single fold

# In[ ]:


# learning rate scheduler
def step_decay(epoch):
    
    initial_lrate = 4e-4
    drop = 0.5
    epoch_cutoff = 7
    epochs_drop = 3.0
    
    if epoch<epoch_cutoff:
        return initial_lrate
    else:
        return initial_lrate * math.pow(drop, math.floor((1+epoch-epoch_cutoff)/epochs_drop))

# get lists of train and val filenames
train_files = all_files

# get data matrices
X_train, y_train = get_train(train_files)
X_train = np.expand_dims(X_train, axis=3)
filenames_test, X_test = get_test()
X_test = np.expand_dims(X_test, axis=3)

# build, summarize, and fit
model = cnn_6(lr=4e-4, do=0.5, l2reg=0)
model.summary()
lrate = LearningRateScheduler(step_decay, verbose=1)
model_history = model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, callbacks=[lrate])

# evaluate model on training data
y_train_pred = model.predict(X_train, batch_size=128)
train_lwlwrap = calculate_overall_lwlrap_sklearn(y_train, y_train_pred)
print('train lwlwrap score:', train_lwlwrap)

# create predictions for test data
y_test_pred   = model.predict(X_test, batch_size=128) # children format
filnames_test_parents, y_test_pred, _ = geometric_average_val_scores(filenames_test, y_test_pred, y_test_pred)

print('training complete')


# # Create predictions file

# In[ ]:


# construct dataframe from array and insert 'fname' column
predictions = pd.DataFrame(data=y_test_pred) 
predictions.insert(loc=0, column='fname', value=filnames_test_parents) 

# rename columns: ['0', '1', '2', ...] -> ['fname', 'Accelerating_and_revving_and_vroom', 'Accordion', ...]
columns = ['fname']
columns.extend(mlb.classes_.tolist())
predictions.columns = columns

# save as output csv
predictions.to_csv('submission.csv', index=False)

