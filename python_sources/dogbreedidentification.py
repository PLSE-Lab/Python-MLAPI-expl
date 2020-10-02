#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # End to End Dog Breed Multi Class Classification
# This notebook is written in tensorflow 2.0 and tensorhub
# ## 1.Problem
# There are 120 breeds of dog in the data. We have to identify according to their breeds.
# ## 2.Data
# There are training set and a test set of images of dogs. Each image has a filename that is its unique id. The dataset comprises 120 breeds of dogs.
# * train.zip - the training set, you are provided the breed for these dogs
# * test.zip - the test set, you must predict the probability of each breed for each image
# * sample_submission.csv - a sample submission file in the correct format
# * labels.csv - the breeds for the images in the train set
# 
# ## Evaluation
# Predicting the probability of each breed in test data.
# ## Features
# * we are dealing with the unstructure data set.
# * There are about 10000+ images in the training data.
# * There are about 10000+ images in the test data.
# 
# Now Since we have unstructured data we have to work with the library like tensorflow. so lets import the necessary libraries and directly jump into the project

# In[ ]:


# standard imports
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plte
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
import tensorflow_hub as hub


# ## Getting our data ready
# communication with our data and turning them into the tensors.All data must be in a numerical form.
# 
# Lets start by accessing our data and checking the labels.

# In[ ]:


labels_csv = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
labels_csv.head()


# In[ ]:


labels_csv.describe()


# In[ ]:


labels_csv['breed'].value_counts().plot.bar(figsize=(20,12));


# In[ ]:


#median number of image in each class.
labels_csv['breed'].value_counts().median()


# In[ ]:


#viewing any image from the train data.
from IPython.display import Image
Image('/kaggle/input/dog-breed-identification/train/0a0c223352985ec154fd604d7ddceabd.jpg')


# ## Getting Images and their labels.
# Create path names for image ID's.

# In[ ]:


filenames = ['/kaggle/input/dog-breed-identification/train/' + fname + '.jpg' for fname in labels_csv['id']]
filenames[:5]


# Check wheather the the number of files matches number of actual images.

# In[ ]:


import os
if len(os.listdir('/kaggle/input/dog-breed-identification/train/')) == len(filenames):
    print('Number of file matches number of actual images!')
else:
    print('Number of file doesnot matches number of actual images!!')


# In[ ]:


#visualizing images according to their index.
Image(filenames[900])


# In[ ]:


#finding the name of the above displayed dog.
labels_csv['breed'][900]


# ## Turning our data into numbers

# In[ ]:


labels = labels_csv['breed']
labels = np.array(labels)
labels


# chech wheather the number of labels matches the number of filenames.

# In[ ]:


if len(labels) == len(filenames):
    print('Number of labels matches the number of filenames.')
else:
    print('Number of labels doesnot matches the number of filenames')


# Finding the unique labels values

# In[ ]:


unique_breed = np.unique(labels) 
unique_breed


# In[ ]:


#Turn single label into an array of boolean.
print(labels[0])
labels[0] == unique_breed


# In[ ]:


#Turning every label into an array of boolean
boolean_labels = [labels == unique_breed for labels in labels]
boolean_labels[:2]


# In[ ]:


# Turining boolean arrays into integers.
print(labels[0])   #orginal index
print(np.where(unique_breed==labels[0]))    #index where labels occurs.
print(boolean_labels[0].argmax())     #index where label occurs in boolean array
print(boolean_labels[0].astype(int))   #there will be a 1 where sample label occurs


# ## Creating our own validation set.

# In[ ]:


# setup x and y variables.
X = filenames
y = boolean_labels


# First starting with ~1000 images because we have lots of data to train for the very first attempt

# In[ ]:


#set number of images to set for the experiment.
NUM_IMAGES = 1000 #@param {type:"slider",min:1000,max:10000}


# In[ ]:


#let's split our data into train and validation.
from sklearn.model_selection import train_test_split

#spliting into training and validation of total size NUM_IMAGES.

X_train,X_val,y_train,y_val = train_test_split(X[:NUM_IMAGES],
                                                y[:NUM_IMAGES],
                                                test_size=0.2,
                                                random_state=42)
len(X_train),len(X_val),len(y_train),len(y_val)


# In[ ]:


X_train[:5],y_train[:5]


# # Preprocessing Images
# Turning images into tensors
# 
# Let's write a function to preprocess the image. The function will do the following tasks.
# 
# * The function will take an image filepath as input.
# * Use the tensorflow to read the file and save it to the variable.
# * Turn our variable (.jpg) into tensors.
# * Normalize our image(convert color channel from 0-255 to 0-1).|
# * Resize the image.
# * Return the modified variable.

# In[ ]:


# converting images to numpy array

from matplotlib.pyplot import imread
image = imread(filenames[42])
image.shape


# In[ ]:


image


# In[ ]:


#lets conver them into tensor
tf.constant(image)[:2]


# ### Making a function to preprocess the data.

# In[ ]:


# Define image size
IMG_SIZE = 224

def process_image(image_path):
  """
  Takes an image file path and turns it into a Tensor.
  """
  # Read in image file
  image = tf.io.read_file(image_path)
  # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
  image = tf.image.decode_jpeg(image, channels=3)
  # Convert the colour channel values from 0-225 values to 0-1 values
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image to our desired size (224, 244)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  return image


# ### Turning our data into batches
# 
# Why turn our data into batches?
# 
#  We are trying to fit the 10000+ data images. They all might not fit into memory.
# 
# So,that's why we use 32(this is batch size) images at a time. we can change the batch size whenever we need.
# 
# In order to use the tensorflow effective we need to convert the images into tuple tensor which looks like   `(image,labels)`

# In[ ]:


# Create a simple function to return a tuple (image, label)
def get_image_label(image_path, label):
  """
  Takes an image file path name and the associated label,
  processes the image and returns a tuple of (image, label).
  """
  image = process_image(image_path)
  return image, label


# Let's make a function to turn all our data (x,y) into batches

# In[ ]:



# Define the batch size, 32 is a good default
BATCH_SIZE = 32

# Create a function to turn data into batches
def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
  """
  Creates batches of data out of image (x) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
  Also accepts test data as input (no labels).
  """
  # If the data is a test dataset, we probably don't have labels
  if test_data:
    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) # only filepaths
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch
  
  # If the data if a valid dataset, we don't need to shuffle it
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                               tf.constant(y))) # labels
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

  else:
    # If the data is a training dataset, we shuffle it
    print("Creating training data batches...")
    # Turn filepaths and labels into Tensors
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                              tf.constant(y))) # labels
    
    # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
    data = data.shuffle(buffer_size=len(x))

    # Create (image, label) tuples (this also turns the image path into a preprocessed image)
    data = data.map(get_image_label)

    # Turn the data into batches
    data_batch = data.batch(BATCH_SIZE)
  return data_batch


# In[ ]:


# Create training and validation data batches
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)


# In[ ]:


# Check out the different attributes of our data batches
train_data.element_spec, val_data.element_spec


# Look at that! We've got our data in batches, more specifically, they're in Tensor pairs of (images, labels) ready for use on a GPU.
# 
# But having our data in batches can be a bit of a hard concept to understand. Let's build a function which helps us visualize what's going on under the hood.

# ## Visualizing data batches

# In[ ]:


import matplotlib.pyplot as plt

# Create a function for viewing images in a data batch
def show_25_images(images, labels):
  """
  Displays 25 images from a data batch.
  """
  # Setup the figure
  plt.figure(figsize=(10, 10))
  # Loop through 25 (for displaying 25 images)
  for i in range(25):
    # Create subplots (5 rows, 5 columns)
    ax = plt.subplot(5, 5, i+1)
    # Display an image
    plt.imshow(images[i])
    # Add the image label as the title
    plt.title(unique_breed[labels[i].argmax()])
    # Turn gird lines off
    plt.axis("off")


# To make computation efficient, a batch is a tighly wound collection of Tensors.
# 
# So to view data in a batch, we've got to unwind it.
# 
# We can do so by calling the as_numpy_iterator() method on a data batch.
# 
# This will turn our a data batch into something which can be iterated over.
# 
# Passing an iterable to next() will return the next item in the iterator.
# 
# In our case, next will return a batch of 32 images and label pairs.

# In[ ]:


# Visualize training images from the training data batch
train_images, train_labels = next(train_data.as_numpy_iterator())
show_25_images(train_images, train_labels)


# In[ ]:


# Visualize validation images from the validation data batch
val_images, val_labels = next(val_data.as_numpy_iterator())
show_25_images(val_images, val_labels)


# # Creating and training a model.
# Now our data is ready now lets model our data.
# 
# Before we build a model, there are a few things we need to define:
# 
# * The input shape (images, in the form of Tensors) to our model.
# * The output shape (image labels, in the form of Tensors) of our model.
# 

# In[ ]:


#setting up input shape to our model.
# Setting up input shape to the model
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] # batch, height, width, colour channels

# Setting up output shape of the model
OUTPUT_SHAPE = len(unique_breed) # number of unique labels

# Setting up model URL from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"


# Now we've got the inputs, outputs and model we're using ready to go. We can start to put them together
# 
# There are many ways of building a model in TensorFlow but one of the best ways to get started is to use the Keras API.
# 
# Knowing this, let's create a function which:
# 
# * Takes the input shape, output shape and the model we've chosen's URL as parameters.
# * Defines the layers in a Keras model in a sequential fashion (do this first, then this, then that).
# * Compiles the model (says how it should be evaluated and improved).
# * Builds the model (tells it what kind of input shape it'll be getting).
# * Returns the model.

# In[ ]:


# Create a function which builds a Keras model
def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
  print("Building model with:", MODEL_URL)

  # Setup the model layers
  model = tf.keras.Sequential([
    hub.KerasLayer(MODEL_URL), # Layer 1 (input layer)
    tf.keras.layers.Dense(units=OUTPUT_SHAPE, 
                          activation="softmax") # Layer 2 (output layer)
  ])

  # Compile the model
  model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(), # Our model wants to reduce this (how wrong its guesses are)
      optimizer=tf.keras.optimizers.Adam(), # A friend telling our model how to improve its guesses
      metrics=["accuracy"] # We'd like this to go up
  )

  # Build the model
  model.build(INPUT_SHAPE) # Let the model know what kind of inputs it'll be getting
  
  return model


# In[ ]:


# Create a model and check its details
model = create_model()
model.summary()


# ### Creating callbacks
# We've got a model ready to go but before we train it we'll make some callbacks.
# 
# Callbacks are helper functions a model can use during training to do things such as save a models progress, check a models progress or stop training early if a model stops improving.

# In[ ]:


# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


import datetime

# Create a function to build a TensorBoard callback
def create_tensorboard_callback():
  # Create a log directory for storing TensorBoard logs
  logdir = os.path.join("drive/My Drive/Data/logs",
                        # Make it so the logs get tracked whenever we run an experiment
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  return tf.keras.callbacks.TensorBoard(logdir)


# ### Early Stopping Callback
# Early stopping helps prevent overfitting by stopping a model when a certain evaluation metric stops improving. If a model trains for too long, it can do so well at finding patterns in a certain dataset that it's not able to use those patterns on another dataset it hasn't seen before (doesn't generalize).
# 
# It's basically like saying to our model, "keep finding patterns until the quality of those patterns starts to go down."

# In[ ]:


# Create early stopping (once our model stops improving, stop training)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  patience=3) # stops after 3 rounds of no improvements


# ## Training a model (On a subset of data)
# Our first model is only going to be trained on 1000 images. Or trained on 800 images and then validated on 200 images, meaning 1000 images total or about 10% of the total data.
# 
# We do this to make sure everything is working. And if it is, we can step it up later and train on the entire training dataset.
# 
# The final parameter we'll define before training is NUM_EPOCHS (also known as number of epochs).
# 
# NUM_EPOCHS defines how many passes of the data we'd like our model to do. A pass is equivalent to our model trying to find patterns in each dog image and see which patterns relate to each label.
# 
# If NUM_EPOCHS=1, the model will only look at the data once and will probably score badly because it hasn't a chance to correct itself
# 

# In[ ]:


NUM_EPOCHS = 100
# Build a function to train and return a trained model
def train_model():
  """
  Trains a given model and returns the trained version.
  """
  # Create a model
  model = create_model()

  # Create new TensorBoard session everytime we train a model
  tensorboard = create_tensorboard_callback()

  # Fit the model to the data passing it the callbacks we created
  model.fit(x=train_data,
            epochs=NUM_EPOCHS,
            validation_data=val_data,
            validation_freq=1, # check validation metrics every epoch
            callbacks=[tensorboard, early_stopping])
  
  return model


# In[ ]:


# Fit the model to the data
model = train_model()


# It look like our model is overfitting.

# ## Making and evaluating predictions using a trained model

# In[ ]:


# Make predictions on the validation data (not used to train on)
predictions = model.predict(val_data, verbose=1) # verbose shows us how long there is to go
predictions


# In[ ]:


# Check the shape of predictions
predictions.shape


# Making predictions with our model returns an array with a different value for each label.
# 
# In this case, making predictions on the validation data (200 images) returns an array (predictions) of arrays, each containing 120 different values (one for each unique dog breed).
# 
# These different values are the probabilities or the likelihood the model has predicted a certain image being a certain breed of dog. The higher the value, the more likely the model thinks a given image is a specific breed of dog.

# In[ ]:


# First prediction
print(predictions[0])
print(f"Max value (probability of prediction): {np.max(predictions[0])}") # the max probability value predicted by the model
print(f"Sum: {np.sum(predictions[0])}") # because we used softmax activation in our model, this will be close to 1
print(f"Max index: {np.argmax(predictions[0])}") # the index of where the max value in predictions[0] occurs
print(f"Predicted label: {unique_breed[np.argmax(predictions[0])]}") # the predicted label


# In[ ]:


# Turn prediction probabilities into their respective label (easier to understand)
def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label.
  """
  return unique_breed[np.argmax(prediction_probabilities)]

# Get a predicted label based on an array of prediction probabilities
pred_label = get_pred_label(predictions[0])
pred_label


# Now we've got a list of all different predictions our model has made, we'll do the same for the validation images and validation labels.
# 
# The model hasn't trained on the validation data, during the fit() function, it only used the validation data to evaluate itself. So we can use the validation images to visually compare our models predictions with the validation labels.
# 
# Since our validation data (val_data) is in batch form, to get a list of validation images and labels, we'll have to unbatch it (using unbatch()) and then turn it into an iterator using as_numpy_iterator().
# 
# Let's make a small function to do so.

# In[ ]:


# Create a function to unbatch a batched dataset
def unbatchify(data):
  """
  Takes a batched dataset of (image, label) Tensors and returns separate arrays
  of images and labels.
  """
  images = []
  labels = []
  # Loop through unbatched data
  for image, label in data.unbatch().as_numpy_iterator():
    images.append(image)
    labels.append(unique_breed[np.argmax(label)])
  return images, labels

# Unbatchify the validation data
val_images, val_labels = unbatchify(val_data)
val_images[0], val_labels[0]


# Now we've got ways to get:
# 
# * Prediction labels
# * Validation labels (truth labels)
# * Validation images
# Let's make some functions to make these all a bit more visualize.
# 
# More specifically, we want to be able to view an image, its predicted label and its actual label (true label).
# 
# The first function we'll create will:
# 
# * Take an array of prediction probabilities, an array of truth labels, an array of images and an integer.
# * Convert the prediction probabilities to a predicted label.
# * Plot the predicted label, its predicted probability, the truth label and target image on a single plot.

# In[ ]:


def plot_pred(prediction_probabilities, labels, images, n=1):
  """
  View the prediction, ground truth label and image for sample n.
  """
  pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]
  
  # Get the pred label
  pred_label = get_pred_label(pred_prob)
  
  # Plot image & remove ticks
  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])

  # Change the color of the title depending on if the prediction is right or wrong
  if pred_label == true_label:
    color = "green"
  else:
    color = "red"

  plt.title("{} {:2.0f}% ({})".format(pred_label,
                                      np.max(pred_prob)*100,
                                      true_label),
                                      color=color)


# In[ ]:


# View an example prediction, original image and truth label
plot_pred(prediction_probabilities=predictions,
          labels=val_labels,
          images=val_images)


# Since we're working with a multi-class problem (120 different dog breeds), it would also be good to see what other guesses our model is making. More specifically, if our model predicts a certain label with 24% probability, what else did it predict?
# 
# Let's build a function to demonstrate. The function will:
# 
# * Take an input of a prediction probabilities array, a ground truth labels array and an integer.
# * Find the predicted label using get_pred_label().
# * Find the top 10:
#     * Prediction probabilities indexes
#     * Prediction probabilities values
#     * Prediction labels
# * Plot the top 10 prediction probability values and labels, coloring the true label green.

# In[ ]:


def plot_pred_conf(prediction_probabilities, labels, n=1):
  """
  Plots the top 10 highest prediction confidences along with
  the truth label for sample n.
  """
  pred_prob, true_label = prediction_probabilities[n], labels[n]

  # Get the predicted label
  pred_label = get_pred_label(pred_prob)

  # Find the top 10 prediction confidence indexes
  top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
  # Find the top 10 prediction confidence values
  top_10_pred_values = pred_prob[top_10_pred_indexes]
  # Find the top 10 prediction labels
  top_10_pred_labels = unique_breed[top_10_pred_indexes]

  # Setup plot
  top_plot = plt.bar(np.arange(len(top_10_pred_labels)), 
                     top_10_pred_values, 
                     color="grey")
  plt.xticks(np.arange(len(top_10_pred_labels)),
             labels=top_10_pred_labels,
             rotation="vertical")

  # Change color of true label
  if np.isin(true_label, top_10_pred_labels):
    top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
  else:
    pass


# In[ ]:


plot_pred_conf(prediction_probabilities=predictions,
               labels=val_labels,
               n=20)


# In[ ]:


# Let's check a few predictions and their different values
i_multiplier = 0
num_rows = 3
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(5*2*num_cols, 5*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_pred(prediction_probabilities=predictions,
            labels=val_labels,
            images=val_images,
            n=i+i_multiplier)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_pred_conf(prediction_probabilities=predictions,
                labels=val_labels,
                n=i+i_multiplier)
plt.tight_layout(h_pad=1.0)
plt.show()


# ## Saving and reloading a model

# In[ ]:


def save_model(model, suffix=None):
  """
  Saves a given model in a models directory and appends a suffix (str)
  for clarity and reuse.
  """
  # Create model directory with current time
  modeldir = os.path.join("/kaggle/working/drive/My Drive/Data/",
                          datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
  model_path = modeldir + "-" + suffix + ".h5" # save format of model
  print(f"Saving model to: {model_path}...")
  model.save(model_path)
  return model_path


# In[ ]:


def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model


# In[ ]:


# Save our model trained on 1000 images
save_model(model, suffix="1000-images-Adam")


# # Training a model in full dataset
# 
# Now we know our model works on a subset of the data, we can start to move forward with training one on the full data.
# 
# Above, we saved all of the training filepaths to X and all of the training labels to y. Let's check them out.

# In[ ]:


len(X),len(y)


#  We've got over 10,000 images and labels in our training set.
# 
# Before we can train a model on these, we'll have to turn them into a data batch.

# In[ ]:


# Turn full training data in a data batch
full_data = create_data_batches(X, y)


# Our data is in a data batch, all we need now is a model.
# 
# we've got a function for that too! Let's use create_model() to instantiate another model.

# In[ ]:


# Instantiate a new model for training on the full dataset
full_model = create_model()


# Since we've made a new model instance, full_model, we'll need some callbacks too.

# In[ ]:


# Create full model callbacks

# TensorBoard callback
full_model_tensorboard = create_tensorboard_callback()

# Early stopping callback
# Note: No validation set when training on all the data, therefore can't monitor validation accruacy
full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy",
                                                             patience=3)


# Fitting the full model to the full training data with about 10000+ images

# In[ ]:


# Fit the full model to the full training data
full_model.fit(x=full_data,
               epochs=5,
               callbacks=[full_model_tensorboard, 
                          full_model_early_stopping])


# Saving the full trained model to the file followed by the suffix all-images-Adam

# In[ ]:


# Save model to file
save_model(full_model, suffix="all-images-Adam")


# Loading the full saved model using load_model()

# In[ ]:


# Load in the full model
loaded_full_model = load_model('/kaggle/working/drive/My Drive/Data/20200628-03271593314820-all-images-Adam.h5')


# In[ ]:


# Load test image filenames (since we're using os.listdir(), these already have .jpg)
test_path = "/kaggle/input/dog-breed-identification/test/"
test_filenames = [test_path + fname for fname in os.listdir(test_path)]

test_filenames[:10]


# Creating test data batches.

# In[ ]:


# Create test data batch
test_data = create_data_batches(test_filenames, test_data=True)


# Making predictions on test data batch using the loaded full model

# In[ ]:


# Make predictions on test data batch using the loaded full model
test_predictions = loaded_full_model.predict(test_data,
                                             verbose=1)


# Displaying the outcome in the pandas dataframe.

# In[ ]:


# Create pandas DataFrame with empty columns
preds_df = pd.DataFrame(columns=["id"] + list(unique_breed))
preds_df.head()


# In[ ]:


# Append test image ID's to predictions DataFrame
test_path = "/kaggle/input/dog-breed-identification/test/"
preds_df["id"] = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
preds_df.head()


# In[ ]:


# Add the prediction probabilities to each dog breed column
preds_df[list(unique_breed)] = test_predictions
preds_df.head()


# Creating the csv file to submit for the competition.

# In[ ]:


preds_df.to_csv("/kaggle/working/drive/My Drive/Data/MySubmission.csv",
                 index=False)


# So, this is the predicted outcome of the dog breed identification.
# 
