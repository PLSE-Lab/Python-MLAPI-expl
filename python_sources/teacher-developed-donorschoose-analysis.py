#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Common/Standard packages
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Filepath to main training dataset.
train_file_path = '../input/train.csv'

# Read data and store in DataFrame.
train_data = pd.read_csv(train_file_path, sep=',')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Datasets include:
# * resources.csv - gives id, description, quantity, and price for all items requested in the projects
# * train.csv - inspect this dataset below
# * test.csv - features similar to train.csv, minus the approval outcome (no project_is_approved column)
# * sample_submission.csv - cursory Kaggle example submission
# 

# In[ ]:


train_data.columns


# train.csv contains the following variables:
# id
# teacher_id
# teacher_prefix
# school_state
# project_submitted_datetime
# project_grade_category
# project_subject_categories
# project_subject_subcategories
# project_title
# project_essay_1
# project_essay_2
# project_essay_3
# project_essay_4
# project_resource_summary
# teacher_number_of_previously_posted_projects
# project_is_approved
# 
# INITIAL OBSERVATIONS
# 
# *Domain Knowledge Insights*
# 
# I can't help noticing certain limitations in the data based on personal experience trying to get my own DonorsChoose classroom projects approved. Variables that could be significant contributors, but that are not included (some of which may be more difficult to track):
# 
# **Match Offer projects** - my projects submitted under match offers were approved more reliably and quickly than standard projects.
# 
# **Social Media connections** - most of my donors found my project on social media, so it may be helpful to track what social media accounts teachers have connected to their DonorsChoose accounts and possibly how many friends/followers each account has in all connected accounts, aggregated.
# 
# *Data Insights/Ideas*
# 
# Project essays 1 through 4 are all in one submission, but not every application essay is broken up into 4 paragraphs. Requirements define a word count, not a paragraph number. May want to explore the size of paragraphs, whether or not projects have 4 paragraphs, etc.
# 
# Certain sections begin with uneditable sentence stems. Check to see if these are included as data. Possibly remove this common sentence stem as a possible confounding variable? May be too minimal to matter, but I'm going to say it's worth keeping in mind.

# In[ ]:


# Additional packages for these models as suggested by Kaggle
import tensorflow as tf
from tensorflow.python.data import Dataset
import sklearn.metrics as metrics


# In[ ]:


# Define predictor feature(s); start with a simple example with one feature.
my_feature_name = 'teacher_number_of_previously_posted_projects'
my_feature = train_data[[my_feature_name]]

# Specify the label to predict.
my_target_name = 'project_is_approved'

# Prepare training and validation sets.
N_TRAINING = 160000
N_VALIDATION = 100000

# Choose examples and targets for training.
training_examples = train_data.head(N_TRAINING)[[my_feature_name]].copy()
training_targets = train_data.head(N_TRAINING)[[my_target_name]].copy()

# Choose examples and targets for validation.
validation_examples = train_data.tail(N_VALIDATION)[[my_feature_name]].copy()
validation_targets = train_data.tail(N_VALIDATION)[[my_target_name]].copy()


# Datasets API

# In[ ]:


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified
    if shuffle:
      # Shuffle with a buffer size of 10000
      ds = ds.shuffle(10000)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# Linear Classifier

# In[ ]:


# Learning rate for training.
learning_rate = 0.00001

# Function for constructing feature columns from input features
def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.
  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

# Create a linear classifier object.
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Set a clipping ratio of 5.0
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)  
linear_classifier = tf.estimator.LinearClassifier(
    feature_columns=construct_feature_columns(training_examples),
    optimizer=my_optimizer
)


# Create input functions for training the model, predicting on the prediction data, and predicting on the validation data:

# In[ ]:


batch_size = 10

# Create input function for training
training_input_fn = lambda: my_input_fn(training_examples, 
                                        training_targets[my_target_name],
                                        batch_size=batch_size)

# Create input function for predicting on training data
predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                training_targets[my_target_name],
                                                num_epochs=1, 
                                                shuffle=False)

# Create input function for predicting on validation data
predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                  validation_targets[my_target_name],
                                                  num_epochs=1, 
                                                  shuffle=False)


# Training the model

# In[ ]:


# Train for 200 steps
linear_classifier.train(
  input_fn=training_input_fn,
  steps=200
)

# Compute predictions.    
training_probabilities = linear_classifier.predict(
    input_fn=predict_training_input_fn)
training_probabilities = np.array(
      [item['probabilities'] for item in training_probabilities])
    
validation_probabilities = linear_classifier.predict(
    input_fn=predict_validation_input_fn)
validation_probabilities = np.array(
    [item['probabilities'] for item in validation_probabilities])
    
training_log_loss = metrics.log_loss(
    training_targets, training_probabilities)
validation_log_loss = metrics.log_loss(
    validation_targets, validation_probabilities)
  
# Print the training and validation log loss.
print("Training Loss: %0.2f" % training_log_loss)
print("Validation Loss: %0.2f" % validation_log_loss)

auc = metrics.auc


# Next, let's calculate the AUC (area under the curve), which is the metric this competition uses to assess the accuracy of prediction. This may take a few minutes. When calculation is complete, the training and validation AUC values will be output:

# In[ ]:


training_metrics = linear_classifier.evaluate(input_fn=predict_training_input_fn)
validation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print("AUC on the training set: %0.2f" % training_metrics['auc'])
print("AUC on the validation set: %0.2f" % validation_metrics['auc'])


# We've achieved AUC values of 0.56, which is slightly better than random. This is a good start, but can you improve the model to achieve better results?
# 
# What to Try Next
# A couple ideas for model refinements you can try to see if you can improve model accuracy:
# 
# Try adjusting the learning_rate and steps hyperparameters on the existing model.
# Try adding some text features to the model, such as the content of the project essays (project_essay_1, project_essay_2, project_essay_3, project_essay_4). You may want to try building a vocabulary from these strings; see the Machine Learning Crash Course Intro to Sparse Data and Embeddings exercise for some practice on working with text data and vocabularies.
# 
# HERE ENDS "Getting Started" CODE SNAG
