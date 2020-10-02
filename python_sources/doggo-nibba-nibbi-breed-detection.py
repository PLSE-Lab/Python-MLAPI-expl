#!/usr/bin/env python
# coding: utf-8

# # end-to_end_Multi-class-Doggo-Breed-Classification
# this notebook builds an end-to-end multiclass image classifier using TenserFlow 2.0 and TenserFlow Hub

# ## 1. Problem Statement
# Identifying The Breed of a dog given an image of a Dog.

# ## 2. Data
# from Kaggle

# ## 3. Evaluation
# The Evaluation is a file with prediction probabilities for each dog breed of each test image.
# 
# 

# ## 4. Features
# some information about the data:
# * We are dealing with images (Unstructured data) so it's  probably use deep learning / transfer learning .
# * There are 120 breeds of dogs (this means there are 120 different classes)
# * There are around 10k+ images in the training set (these images have labels)
# * There are around 10k+ images in the test set (these images have no labels we'll want to predict them)
# 

# In[ ]:


# Import necessary tools into kaggle
import tensorflow as tf
import tensorflow_hub as hub
print("TF version : ",tf.__version__)
print("TF Hub version : ", hub.__version__)

# Cheak for GPU availability
print("GPU","available (Yes !)" if tf.config.list_physical_devices("GPU") else "Not Available")


# ## Getting our data ready (turning into tensors)
# 
# With all machine learning models, our data has to be in numerical format. So  that's what we'll be doing first. Tuning our images into tensors (numerical representation).
# 
# Let's Start By acessing our data and cheaking out the labels

# In[ ]:


# Cheakout the labels of our data
import pandas as pd
labels_csv = pd.read_csv("../input/dog-breed-identification/labels.csv")
print(labels_csv.describe())
print(labels_csv.head())


# In[ ]:


labels_csv.head()


# In[ ]:


labels_csv["breed"].value_counts().plot.bar(figsize=(20,10));


# In[ ]:


labels_csv["breed"].value_counts().median()


# In[ ]:


# Let's view an image
from IPython.display import Image
Image("../input/dog-breed-identification/train/001513dfcb2ffafc82cccf4d8bbaba97.jpg")


# ### Getting images and labels
# let's get a list of all our images file pathnames

# In[ ]:


labels_csv.head()


# In[ ]:


# Create pathnames from image ID's
filenames = ["../input/dog-breed-identification/train/"+names+".jpg" for names in labels_csv["id"] ]
# Cheak the first 10
filenames[:10]


# In[ ]:


# Cheak weather number of filenames matches number of actual image files
import os
if len(os.listdir("../input/dog-breed-identification/train/"))==len(filenames):
    print("Filenames match equal ammount of files ! Proceed")
else:
    print("filenames do not match actual ammount of files, cheak the target directory.")


# In[ ]:


Image(filenames[9275])


# Since we've now got our training image filepaths in a list, let's prepare our labels 

# In[ ]:


import numpy as np
labels = labels_csv["breed"].to_numpy()
# labels = np.array(labels)  # Does same thing as above
labels , len(labels)


# In[ ]:


# See if number of labels matches the number of filenames
if len(labels) == len(filenames):
    print("Number of labels matches number of filenames !")
else:
    print("Number of labels does not match number of filenames, cheak data directories")


# In[ ]:


# Find the unique label values
unique_breeds = np.unique(labels)
unique_breeds,len(unique_breeds)


# In[ ]:


# Trun a single label into array of booleans
print(labels[0])
labels[0] == unique_breeds


# In[ ]:


# turn every label into a boolean array
boolean_labels = [label == unique_breeds for label in labels]
boolean_labels[:2]


# In[ ]:


# Example : Turning boolean array into integers
print(labels[0]) # original label
print(np.where(unique_breeds==labels[0])) # index where label occurs
print(boolean_labels[0].argmax()) # index where label occurs in boolean array
print(boolean_labels[0].astype(int)) # there will be a 1 where the sample label occurs


# In[ ]:


print(labels[2])
print(boolean_labels[2].astype(int))


# In[ ]:


filenames[:10]


# ## Creating our own validation set 
# Since the dataset from kaggle does'nt come with a validation set , we are going to create our own .

# In[ ]:


# Setup X and Y variables 
X = filenames 
Y = boolean_labels
len(filenames)


# We're going to start off experimenting  with-1000 images and increase as needed

# In[ ]:


# Set number of images to use for experimenting
NUM_IMAGES = 1000 #@param {type:"slider", min:1000 , max:10000 ,step:100 } works with colab


# In[ ]:


# Let's split our data into train and validation sets
from sklearn.model_selection import train_test_split

np.random.seed(42)
# Split them into training and validation of total size Num_Images
X_train,X_valid,Y_train,Y_valid = train_test_split(X[:NUM_IMAGES],Y[:NUM_IMAGES],test_size=0.2,random_state=42)

len(X_train),len(Y_train),len(X_valid),len(Y_valid)


# In[ ]:


X_train[:2] ,Y_train[:2]


# # Preprocessing Images(turning images into Tensors )
# To preprocess our images into Tensors  We are going to write a function which does the few things:
# 1. Take an image filepath as input 
# 2. Use TensorFlow to read the file and save it to a variable "image"
# 3. Turn our "image" (a jpg) into Tensors
# 4. Normalize our image (convert color channel values from 0-255 to 0-1)
# 4. Resize the "image to be a shape of (244, 244)
# 5. Return the modified image 
# 
# Before we do lets see what importing an image looks like 

# In[ ]:


# Convert image into NumPy array
from matplotlib.pyplot import imread
image = imread(filenames[42])
image.shape


# In[ ]:


image


# In[ ]:


image.max(),image.min()


# In[ ]:


# Turn image into Tensor
tf.constant(image)


# In[ ]:


tensor = tf.io.read_file(filenames[26])
tensor


# In[ ]:


tensor = tf.image.decode_jpeg(tensor ,channels=3)
tensor


# In[ ]:


tensor = tf.image.convert_image_dtype(tensor, tf.float32)
tensor


# In[ ]:


# Define image size 
IMG_SIZE = 224

# Create a function for preprocessing images 
def process_image(image_path,img_size=IMG_SIZE):
    """
    Takes an image file path and turns the image into tensors
    """
    
    # Read in a image file 
    image = tf.io.read_file(image_path)
    
    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    image = tf.image.decode_jpeg(image ,channels=3)
    
    # Convert the colour channel values from 0-255 to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Resize the image to our desired value 
    image = tf.image .resize(image, size=[IMG_SIZE, IMG_SIZE])
    
    return image


# ## Turning our data into batches
# 
# Why turn our data into batches ?
# 
# Let's say you're trying to process 10k+ images in one go... they all might not fit into memory 
# 
# So that's why we do about 32 (this is the batch size) images at a time (you can manually adjust the batch size if need be)
# 
# In order to use Tenserflow effectively, we need our data in the form of Tensor tuples which look likes this `(image,label)`

# In[ ]:


# Create a simple function to return a tuple (image,label)
def get_image_label(image_path,label):
    """
    Takes an image file path name and the associated label,
    processes the image and returns a tuple of (image,label)
    """
    image = process_image(image_path)
    return image, label


# In[ ]:


(process_image(X[42]),tf.constant(Y[42]))


# now we've got a way to turn our data into tuples of tensors in the form:`(image, label)` , let's make a function to turn all of our data ('X' and 'Y') into batches !

# In[ ]:


# Define the batch size, 32 is a good start
BATCH_SIZE = 32

# Create a function to turn into a batches 

def create_data_batches(X, Y=None,batch_size=BATCH_SIZE, valid_data=False , test_data=False):
    """
    Creates batches of data out of image (X) and label (Y) pairs.
    Suffles the data if it is training data but does'nt suffle if it is validation data.
    Also accepts test data as input (no labels)
    """
    # If the data is a test dataset, we probably don't have labels
    if test_data:
        print("Creating test data batches... ")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))  # Only filepaths (NO labels)
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch
    
    # If the data is a valid dataset , we don't need to suffle it 
    elif valid_data:
        print("Creating validation data batches... ")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), # filepaths
                                                   tf.constant(Y))) # labels 
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch
    
    # Training dataset
    else: 
        print("Creating training data batches...")
        # Turn filepaths and labels into Tensors
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), # filepaths
                                                   tf.constant(Y))) # labels
        # Shffling pathnames and labels before mapping image processor function is faster than suffling images
        data = data.shuffle(buffer_size=len(X))
        
        # Create (image, label) tuples (this also turns the image path into a preprocessed image)
        data = data.map(get_image_label)
        
        # Turn the training data into batches
        data_batch = data.batch(BATCH_SIZE)
        
    return data_batch


# In[ ]:


# Create training and validation data batches
train_data = create_data_batches(X_train, Y_train)
val_data = create_data_batches(X_valid, Y_valid ,valid_data=True)


# In[ ]:


# Cheakout the different attributes of our data batches
train_data.element_spec ,val_data.element_spec


# ## Visualizing Data Batches
# Our data is now batches, however, these can be a little hard to understand/comprehend, let's visualize the data batches

# In[ ]:


import matplotlib.pyplot as plt

# Create a function for viewing images in a data batch
def show_25_images(images,labels):
    fig = plt.figure(figsize=(10,10))
    for i in range(0,25):
        fig.add_subplot(5,5,i+1)
        plt.imshow(images[i])
        plt.title(unique_breeds[train_labels[i].argmax()])
        plt.axis("off")
    plt.show()


# In[ ]:


train_data


# In[ ]:


train_images, train_labels = next(train_data.as_numpy_iterator())
len(train_images),len(train_labels)

# Now let's visualize the data in a training batch
show_25_images(train_images, train_labels)


# In[ ]:


# noe let's visualize our validation set
valid_images, valid_labels= next(val_data.as_numpy_iterator())
show_25_images(valid_images, valid_labels)


# # Building a model
# **Before we build a modelthere are few things we need to define:**
# 
# * The input shape  (our images shape, in the form of tensor ) to our model.
# * The output shape (images labels, in the form of Tensors) of our model.
# * The URL of the model we want to use.
# 

# In[ ]:


IMG_SIZE


# In[ ]:


# Setup the  input to the model
INPUT_SHAPE = [None, IMG_SIZE,IMG_SIZE,3] # batch,height, Width ,colour channels

# Setup the Output shape of the model
OUTPUT_SHAPE = len(unique_breeds)

# Setup model URL from tensorflow hub
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"


# 
# 
# Now we've got our inputs ,outputs and model ready to go.
# 
# Let's put them together into a Keras deep learning model
# 
# Knowing this ,let's create a function which:
# * Takes the input shape, output shape and the model we've choosen as parameters 
# * Defines the layers in a Keras model in sequential fashion (do this first, then this then that ).
# * Compiles the model (says it should be evaluated and improved).
# * Builds the model (tells the model the input shape it'll be getting.
# * Returns the model
# 
# All of these steps can be found here https://www.tensorflow.org/guide/keras/overview

# In[ ]:


# Create a  function which builds a keras model 
def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE,model_url=MODEL_URL):
    print("building model with : ",MODEL_URL)
    
    # Setup the model layers
    model = tf.keras.Sequential([
        hub.KerasLayer(MODEL_URL), # Layer 1 (input layer)
        tf.keras.layers.Dense(units=OUTPUT_SHAPE,
                             activation= "softmax")  # Layer 2 (output layer)
    ])
    
    # Compile the model 
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )
    
    # Build the model
    model.build(INPUT_SHAPE)
    
    return model
    


# In[ ]:


model = create_model()
model.summary()


# # Creating callbacks
# Callbacks are helper functions a model can use during a training to do such things as save its progress, cheak its progress or stop training early if a model stops improving
# 
# we'll create two callbacks , one for Tensorboard which helps track our models progress and another for early stopping which prevents our model for training too long 

# ### Tensorboard callback
# To setup a TensorBoard callback ,we need to do 3 things :
# 
# 1. Load the TensorBoard notebook extension
# 2. Create a TensorBoard callback which is able to save logs to directory and pass it to our models fit function.
# 3. Vissualize our models training logs with the `%tensorboard` magic function (we'll do this after model training).
# 

# In[ ]:


# Load TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


import datetime

# Create a function to build a TensorBoard callback
def create_tensorboard_callback():
    # Create a log directory for storing TensorBoard logs
    logdir= os.path.join("../working/outputs/logs",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(logdir)


# ### Early Stopping Callback
#  Early Stopping helps our model from overfitting by stopping training if certain evaluation metric stops improving 

# In[ ]:


# Create early stopping callback
early_stopping =tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=3)


# ## Training a model (on subset of data)
# 
# Our first model is only going to train on 1000 images, to make sure everything is working.
# 

# In[ ]:


NUM_EPOCHS = 100 #@param {type:"slider" ,min:10 , max:100}


# In[ ]:


# Cheak to make sure we're still running on the GPU
print("GPU","available (Yes !)" if tf.config.list_physical_devices("GPU") else "Not Available")


# Let's Create a function which trains a model
# 
# * Create a model using `create_model()`
# * Setup the TensorBoard callback using `create_tensorboard_callback()`
# * Call the `fit()` function on out model passing it the train data, validation data,number of epochs to train for (`NUM_EPOCHS`) and the callbacks we'd like to use
# * Return the model
# 

# In[ ]:


# Build a function to train and return a trained model
def train_model():
    """
    Trains a given model and returns the trained version.
    """
    # Create a model
    model = create_model()
    # Create new TensorBoard  sessiion everytime we train a model
    tensorboard = create_tensorboard_callback()
    # Fit the model to the data passing it the callbacks we created
    model.fit(x=train_data,
             epochs=NUM_EPOCHS,
             validation_data= val_data,
             validation_freq= 1,
             callbacks= [tensorboard, early_stopping])
    #Return the fitted model
    return model


# In[ ]:


# Fit the model to the data
model = train_model()


# **Question : ** It looks like our model is overfitting far better on the training dataset than the validation  dataset, what are some ways to prevent model overfitting in deep learning neural networks? 
# 
# **Note : ** Overfitting to begin with is a good thing it means our model is learning!!! 

# # Cheaking the TensorFlow logs
# the tensorflow magic function (`%tensorboard`) will access the logs directory we created earlier and visualize its contents

# In[ ]:



get_ipython().run_line_magic('tensorboard', '--logdir working/outputs/logs')
get_ipython().system('kill 5770')


# ## Making and evaluating predictions using a trained model

# In[ ]:


val_data


# In[ ]:


predictions = model.predict(val_data, verbose= 1)
predictions


# In[ ]:


predictions.shape


# In[ ]:


len(Y_valid)


# In[ ]:


# First prediction
index = 1
print(predictions[index])
print(f"Max value (probablity of prediction): {np.max(predictions[index])}")
print(f"Sum : {np.sum(predictions[index])}")
print(f"Max index : {np.argmax(predictions[index])}")
print(f"Predicted label : {unique_breeds[np.argmax(predictions[index])]}")


# having the above functionality is great but we want to able to do ot at scale
# 
# And it would be even better if we could see the image the prediction is being made on !
# 
# **Note : **Prediction probablities are also known as confidence levels
# 

# In[ ]:


# Turn the prediction probablities into their respctive label (easier to understand)
def get_pred_label(prediction_probabilities):
    """
    turns an array of prediction probablities into labels.
    """
    return unique_breeds[np.argmax(prediction_probabilities)]

# Get a predicted label based on an array of prediction probablities 
pred_label = get_pred_label(predictions[81])
pred_label


# Since now our validation data is still in a batch dataset, we'll have to unbatchify it to make predictions on the validation images and then compare those predictions to the validation labels(truth labels).
# 
# 

# In[ ]:


val_data


# In[ ]:


# create a function to unbatch a batch dataset

def unbatchify(data):
    """
    Takes a batched dataset of (image, label) Tensors and returns separate arrays of images and labels.
    """
    images = []
    labels = []
    # Loop through unbatched data
    for image,label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels.append(unique_breeds[np.argmax(label)])
    return images,labels

# Unbatchify the validation data
val_images ,val_labels = unbatchify(val_data)
val_images[0] , val_labels[0]


# Now we got ways to get :
# 
# * Prediction labels
# * Validation labels (truth labels)
# * Validation images
# 
# Let's make some function to make these all a bit more visaulize
# 
# We'll create a function which :
# * Takes an array of prediction probablities, an array of truth labels and an array of images and integers.
# * Convert the prediction probablities to a predicted label.
# * Plot the predicted label , its predicted probablities truth label and the target image on a single plot.
# 

# In[ ]:


def plot_pred(prediction_probabilities, labels, images ,n=1):
    """
    View the prediction, ground truth and image of the sample n
    """
    pred_prob, true_label,image =prediction_probabilities[n] ,labels[n],images[n]
    
    # Get the pred label
    pred_label = get_pred_label(pred_prob)
    
    # Plot image & remove ticks
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    # Change the color of the title depending upon the prediction is right or wrong 
    if pred_label == true_label:
        color="green"
    else:
        color="red"
    # Change plot title to be predicted, probablity of prediction and truth label
    plt.title("{} {:2.0f}% {}".format(pred_label,
                                     np.max(pred_prob)*100,
                                     true_label),color=color)


# In[ ]:


plot_pred(prediction_probabilities= predictions,
         labels= val_labels,
         images= val_images,
         n=77)


# Now we'll got one function to visualize our models top prediction , let's make another to view our model top 10 predictions 
# 
# This function will:
# * Take an input of prediction probablities array and a ground truth array and an integer
# * Find the prediction using `get_pred_label()`
# * Find the top 10:
#     * Prediction probabilities indexes
#     * Prediction probabilities values 
#     * Prediction labels
# * Plot the top 10 prediction probablity values and labels , colouring the true label green

# In[ ]:


def plot_pred_conf(prediction_probabilities , labels , n):
    """
    Plus the top 10 highest prediction confidence along with the truth label for sample n.
    """
    pred_prob, true_label = prediction_probabilities [n], labels[n]
    
    # Get the predicted label
    pred_label = get_pred_label(pred_prob)
    
    # Find the top 10 prediction confidence indexes
    top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
    # Find the top 10 prediction confidence values
    top_10_pred_values = pred_prob[top_10_pred_indexes]
    # Find the top 10 prediction labels
    top_10_pred_labels = unique_breeds[top_10_pred_indexes]
    
    # Setup plot 
    top_plot = plt.bar(np.arange(len(top_10_pred_labels)),
                      top_10_pred_values,
                      color="grey")
    plt.xticks(np.arange(len(top_10_pred_labels)),
              labels=top_10_pred_labels,
              rotation="vertical")
    
    # Change the colour of true label
    if np.isin(true_label, top_10_pred_labels):
        top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
    else:
        pass


# In[ ]:


plot_pred_conf(prediction_probabilities=predictions,
              labels=val_labels,
              n=96
              )


# Now we've got some function to help us visualize our predictions and evaluate our model, let's cheak out 

# In[ ]:


# Let's cheak out a few predictions and their different values 
i_multiplier = 20
num_rows= 3
num_cols= 2
num_images = num_rows*num_cols
plt.figure(figsize=(10*num_cols,5*num_rows))
for i in range(num_images):
    plt.subplot(num_rows,2*num_cols,2*i+1)
    plot_pred(prediction_probabilities=predictions,
             labels=val_labels,
             images=val_images,
             n=i+i_multiplier)
    plt.subplot(num_rows ,2*num_cols, 2*i+2)
    plot_pred_conf(prediction_probabilities=predictions,
                  labels=val_labels,
                  n=i+i_multiplier)
plt.tight_layout()
plt.show()


# ** Challange ** How would you create a confusion matrix with our models predictions and true labels?
# 

# # Saving and reloading a trained model

# In[ ]:


# Create a function to save a model
def save_model(model, suffix=None):
    """
    Saves a given model in a models directory and appends a suffix (string)
    """
    # Create a model directory pathname with current time
    modeldir = os.path.join("../working/models",datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
    model_path = modeldir + "-" +suffix+".h5" # save format to model
    print(f"Saving model to : {model_path}...")
    model.save(model_path)
    return model_path


# In[ ]:


# Create a function to load a train model 
def load_model(model_path):
    """
    Loads a save model from a specified path 
    """
    print(f"Loading saved model from:  {model_path}")
    model = tf.keras.models.load_model(model_path,
                                      custom_objects={"KerasLayer":hub.KerasLayer})
    return model


# Now we've got functions to save and load a trained model , let's make sure they work !

# In[ ]:


get_ipython().system(' cd ../working')


# In[ ]:


get_ipython().system(' mkdir models')


# In[ ]:


# Save our model trained on 1000 images 
save_model_path=save_model(model, suffix="1000-images-mobilenetv2-Adam")


# In[ ]:


# Load a train model
loaded_1000_image_adam_model = load_model(save_model_path)


# In[ ]:


# Evaluate the loaded model 
loaded_1000_image_adam_model.evaluate(val_data)


# In[ ]:


# Evaluate the pre-saved model
model.evaluate(val_data)


# # Training a big dog model on full dataset

# In[ ]:


len(X) , len(Y)


# In[ ]:


# Create a data set from a full dataset 
full_data = create_data_batches(X,Y)


# In[ ]:


full_data


# In[ ]:


# Create A model for full model
full_model = create_model()


# In[ ]:


# Create full model callbacks
full_model_tensorboard = create_tensorboard_callback()
# No validation set when training on all the data, so we can't monitor validation accuracy 
full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy",patience=3)


# **Note : **Running the cell below will take a little while (maybe up to 30 min fo rthe first epoch ) because the GPU we're using in the runtime has to load all of the images into memory.

# In[ ]:


# Fit the full model to the full data
full_model.fit(x=full_data,
              epochs=NUM_EPOCHS,
              callbacks=[full_model_tensorboard, full_model_early_stopping])


# In[ ]:


save_model(full_model,suffix="full-image-model-mobilenetv2-Adam")


# # Making Predictions on the test dataset
# Since our model has been trained on images in the form of tensors batches , to make predictions on the test data, we'll have to get into the same format.
# 
# Luckily we created `create_data_batches()` earlier whichcan take a list of filenames as input and convert them into Tensor batches
# 
# To make predictions on the test data ,we'll: 
# * Get the test image filenames 
# * Convert the filenames into test data batches using `create_data_batches` and setting the `test_data` parameter to the `true` (since the test data does'nt have labels).
# * Make a predictions array by passing the test batches to the `predict()` method called on our model.
# 

# In[ ]:


import os 
# Load test image filenames
test_path = "../input/dog-breed-identification/test/"
test_filenames = [test_path + fname for fname in os.listdir(test_path)]
test_filenames
ids=[i[:-4] for i in os.listdir(test_path)]
test_filenames


# In[ ]:


# Create test databatch
test_data=create_data_batches(test_filenames,test_data=True)


# In[ ]:


test_data


# **Note : ** calling `predict()` on our full model and passing it test data batch will take a long time to run (above an hour)

# In[ ]:


# make predictions on test data using full model
test_predictions = full_model.predict(test_data, verbose = 1)


# In[ ]:


test_predictions


# In[ ]:


np.savetxt("../working/outputs/preds_array.csv",test_predictions,delimiter=",")


# In[ ]:


submission=pd.DataFrame(test_predictions,columns=unique_breeds)
submission.insert(0,"id",ids)
submission


# In[ ]:


submission.to_csv("../working/outputs/preds_array.csv",index=False)


# # Apply Our Model on our own images

# In[ ]:


# Creating a function to apply model on our own images

def identify_breed(filepath):
    temp = create_data_batches([filepath,],test_data=True)
    result = full_model.predict(temp, verbose = 1)
    result = unique_breeds[result.argmax()]
    return result

xi = '../input/dog-breed-identification/test/1672018bbbc549cc43a14d9129197f08.jpg'
print(identify_breed(xi))


# In[ ]:




