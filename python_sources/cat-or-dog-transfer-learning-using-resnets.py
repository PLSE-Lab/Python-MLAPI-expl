#!/usr/bin/env python
# coding: utf-8

# <h1 style="font-family:verdana;"> <center>Cat or Dog - Using CNN with Transfer Learning</center> </h1>
# <p><center style="color:#159364;font-size:20px"> Using pretrained ResNet-50 </center></p>
# 
# ***

# ![Stay Home](https://i.insider.com/536aa78069bedddb13c60c3a?width=1100&format=jpeg&auto=webp.png)

# <a id = 'content'></a>
# # Content

# 1. [Load Required Packages](#sec1)
# 2. [Read the Data](#sec2)
# 3. [Visualizing Data](#sec3)
#     - [Preparing data for visualization](#subsec1)
#     - [Visualizing Training Set](#subsec2)
#     - [Visualizing Test Set](#subsec3)
# 4. [Preprocessing Data](#sec4)
#     - [Convert data into dataframe](#subsec11)
#     - [Generating mini batches for training](#subsec12)
# 5. [Model](#sec5)
#     - [Prepare the model](#subsec21)
#     - [Train the model](#subsec22)
# 6. [Results](#sec6)
#     - [Visualizing Predictions](#subsec31)
#     - [Prepare Submission File](#subsec32)

# The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.
# For each image in the test set, you should predict a probability that the image is a dog (1 = dog, 0 = cat).  
# We will use image size of (224 x 224), batch size of 64 and 10 epochs.

# <a id = 'sec1'></a>
# # Load Required Packages

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import cv2
from tqdm import tqdm          #Used for the purpose of showing iterations getting loaded in bar kind of form
from random import shuffle
from zipfile import ZipFile
import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from kaggle_datasets import KaggleDatasets
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# weight_loc = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'  
# We'll not be using this path directly because it will cause some dimension error instead we will download weights of imagenet.
# Here is the explanation:
# https://stackoverflow.com/questions/60119041/failed-to-load-keras-resnet50-model-offline-using-weight-file

# <a id = 'sec2'></a>
# # Read the Data

# We will unzip train and test directory using zipfile library.

# In[ ]:


zip_file = glob.glob('../input/dogs-vs-cats-redux-kernels-edition/*.zip')  #return any files with .zip extension
print(zip_file)

#extract file into a temp folder
def extract_zip(file):
    with ZipFile(file,'r') as zip_ref:
        zip_ref.extractall('.')
        
#extract both train and test1 zip
for files in zip_file:
    extract_zip(files)


# In[ ]:


def get_path(train_filenames):
    path = []
    for files in os.listdir(train_filenames):
        path.append(files)
    return path

train_path = get_path('../working/train')
test_path = get_path('../working/test')


# [Back to contents(click here)](#content)

# <a id = 'sec3'></a>
# # Visualizing Data

# <a id = 'subsec1'></a>
# <span style="color:green;font-size:20px"> Preparing data for visualization </span>

# First we will one hot encode the categories(cat and dog) for the purpose of data exploration.

# In[ ]:


#this labelling function is just for the visualization, we'll have separate one for preprocessing.
def label_img(img):
    category = img.split('.')[-3]
    if category == 'cat': return [1,0]
    elif category == 'dog': return [0,1]

#Process the data, here we're converting images into numpy array. This function takes image data, image directory, a boolean as an argument.
def process_data(img_data, data_dir, isTrain=True):
    data_df = []
    for img in tqdm(img_data):
        path = os.path.join(data_dir,img)         #Assigning path to images by concatenating directory and images
        if(isTrain):
            label = label_img(img)            #Calling label_img to assign labels to image present in training directory
        else:
            label = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224,224))
        data_df.append([np.array(img),np.array(label)])          #append image and labels as numpy array in data_df list
    shuffle(data_df)
    return data_df


# In[ ]:


#We will plot the images of dogs and cats and display the assigned label above image
def show_images(data, isTest=False):
    f, ax = plt.subplots(nrows = 5, ncols = 5, figsize = (15,15))
    for i,data in enumerate(data[:25]):              #enumerate helps in keeping track of count of iterations
        img_num = data[1]
        img_data = data[0]
        label = np.argmax(img_num)                    #to get maximum indices of an array
        if label  == 1: 
            str_label='Dog'
        elif label == 0: 
            str_label='Cat'
        if(isTest):
            str_label="None"
        ax[i//5, i%5].imshow(img_data)
        ax[i//5, i%5].axis('off')                 #removing axis for better look
        ax[i//5, i%5].set_title("Label: {}".format(str_label))
    plt.show()


# <a id = 'subsec2'></a>
# <span style="color:green;font-size:20px"> Visualizing Training Set </span>

# In[ ]:


train = process_data(train_path, './train/')
show_images(train)


# <a id = 'subsec3'></a>
# <span style="color:green;font-size:20px"> Visualizing Test Set </span>

# In[ ]:


test = process_data(test_path, './test/', False)
show_images(test,True)


# [Back to contents(click here)](#content)

# <a id = 'sec4'></a>
# # Preprocessing Data

# <a id = 'subsec11'></a>
# <span style="color:green;font-size:20px"> Convert data into dataframe </span>
# 
# In our data formatting, we will generate the respective labels for dogs (1) and cats (0) for our training data. File path will also be collected as a column for our dataframe so that it can be used to load and train our images.  
# 
# file.split('.')[0] means -
# * the particular filename is a String separated by dots.
# * line.split(".")[0] returns the 1st item of the array. (the actual file looks like "cat.134.jpg")

# In[ ]:


#We will use below function to label data
def get_label(directory):
    label = []
    for file in os.listdir(directory):
        if (file.split('.')[0] == 'dog'):
            label.append(str(1))
        elif (file.split('.')[0] == 'cat'):
            label.append(str(0))
    return label

label = get_label('../working/train')

#In case of train_test_split, allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes. Convert the images and labels into dataframe.
df = pd.DataFrame({'filename': train_path, 'label': label})
df.head()


# In[ ]:


#split data into train and valid set
train_df, valid_df = train_test_split(df, test_size = 0.2, stratify = df['label'], random_state = 123)
print(train_df.shape)
print(valid_df.shape)


# <a id = 'subsec12'></a>
# <span style="color:green;font-size:20px"> Generating mini batches for training </span>  
# 
# We'll use ImageDataGenerator for preprocessing. It helps in data augmentation and helps to get better accuracy by making data flow in batches. Mini-Batch Gradient Descent uses a mini-batches of training example for each gradient calculation; which makes convergence faster, and at the same time, utilise the speed up from vectorization.

# Here we'll assign preprocess_input to preprocessing_function argument of ImageDataGenerator because it has better advantage on accuracy.

# In[ ]:


#We'll perform individually on train and validation set.
train_datagen = ImageDataGenerator(rotation_range = 10, zoom_range = 0.1, horizontal_flip = True, fill_mode = 'nearest', 
                                   width_shift_range = 0.1, height_shift_range = 0.1, preprocessing_function = preprocess_input)

#flow_from_dataframe() method will accept dataframe with filenames as x_column and labels as y_column to generate mini-batches
train_gen = train_datagen.flow_from_dataframe(train_df, directory = '../working/train', x_col = 'filename', y_col = 'label', target_size = (224,224),
                                              batch_size = 64, class_mode='binary')

#we do not augment validation data.
valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

valid_gen = valid_datagen.flow_from_dataframe(valid_df, directory = '../working/train', x_col = 'filename', y_col = 'label', target_size = (224,224),
                                              batch_size = 64, class_mode='binary')


# [Back to contents(click here)](#content)

# <a id = 'sec5'></a>
# # Model

# <a id = 'subsec21'></a>
# <span style="color:green;font-size:20px"> Prepare the model </span>  
# 
# We initialize the ResNet-50 model, adding an additional last layer of type Dense, with sigmoid activation function.
# We also set the first layer of the model to be not trainable, becaise ResNet-50 model was already trained.

# In[ ]:


model = Sequential()
model.add(ResNet50(include_top = False, pooling = 'max', weights = 'imagenet'))
model.add(Dense(1, activation = 'sigmoid'))

model.layers[0].trainable = False 


# <span style = 'color:green;font-size:20px'> Model summary </span>  
# 
# We plot the model description. We can see that the ResNet-50 model represent the 1st layer of our model, of type Model.

# In[ ]:


model.summary()


# <span style="color:green;font-size:20px"> Compile the model </span>  
# 
# We compile the model, using a adam optimized, the loss function as binary crossentropy and the metric accuracy.

# In[ ]:


model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'binary_crossentropy')


# <span style = 'color:green;font-size:20px'> Callback </span>
# 
# Best weight will be stored in 'dogcat.weights.best.hdf5'

# In[ ]:


checkpointer = ModelCheckpoint(filepath = 'dogcat.weights.best.hdf5', save_best_only = True, save_weights_only = True)


# <a id = 'subsec22'></a>
# <span style = 'color:green;font-size:20px'> Train the model </span>

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit_generator(train_gen, epochs = 10, validation_data = valid_gen, callbacks = [checkpointer])')


# Plot the loss and accuracy curves.

# In[ ]:


loss = pd.DataFrame(model.history.history)
loss[['loss', 'val_loss']].plot()
loss[['accuracy', 'val_accuracy']].plot()


# [Back to contents(click here)](#content)

# <a id = 'sec6'></a>
# # Results

# In[ ]:


#load the best weights saved using checkpointer
model.load_weights('dogcat.weights.best.hdf5')


# In[ ]:


test_df = pd.DataFrame({'filename': test_path})    #test_path from function get_path


# <a id = 'subsec31'></a>
# <span style = 'color:green;font-size:20px'> Visualizing Predictions </span>

# We will use test dataframe generated above and process the image. We will predict class using 'model.predict' and append the labels in a list.

# In[ ]:


test_sample = test_df.sample(n = 25, random_state = 123)    #Randomly sample 25 images from test directory for visualizing predictions

label_pred = []                         #list for predicted labels
for path in test_sample['filename'].to_numpy():
    full_path = '../working/test/'+path
    x = load_img(full_path, target_size=(224,224))
    img_array = img_to_array(x)
    img_array = np.expand_dims(img_array, axis=0)
    
    out =  model.predict(img_array)    
    out = 'Dog' if float(out) >0.5 else 'Cat'
    label_pred.append(out)


# We will plot 25 random test images with their class using list of predicted labels.

# In[ ]:


test_array = test_sample['filename'].to_numpy()            # convert dataframe to numpy array 

plt.figure(figsize=(15, 15))
for i in range(0, 25):
    plt.subplot(5, 5, i+1)
    img = test_array[i]
    path = '../working/test/' + img
    image = load_img(path, target_size=(256,256))
    
    plt.title('Predicted: {}'.format(label_pred[i]))
    plt.imshow(image)
    plt.axis('off')

plt.tight_layout()


# In[ ]:


#Generate mini batches of test data
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

test_gen = test_datagen.flow_from_dataframe(test_df, directory = '../working/test', batch_size = 64, x_col = 'filename', y_col = None, class_mode = None, shuffle = False,
                                            img_size = (224, 224))


# In[ ]:


#performing predictions
prediction = model.predict_generator(test_gen)
prediction = prediction.clip(min = 0.005, max = 0.995)     #use clip to set the minimum and maximum limit of predicted probabilities.


# <a id = 'subsec32'></a>
# <span style = 'color:green;font-size:20px'> Prepare Submission File </span>

# In[ ]:


submission_df = pd.read_csv('../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')

for i, fname in enumerate(test_path):
    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
    submission_df.at[index-1, 'label'] = prediction[i]
submission_df.to_csv('Cats&DogsSubmission.csv', index=False)


# In[ ]:


submission_df.head()


# <span style = 'color:green;font-size:20px'> Conclusion </span>  
# 
# Using a pretrained model for tensorflow, ResNet-50, with a Dense model with sigmoid activation added on top and training with a reduced set of we were able to obtain quite good model in terms of validation accuracy.
# The model was used to predict the classes of the images from the independent test set and results were submitted to test the accuracy of the prediction with fresh data.

# <center><span style = 'color:green;font-size:22px'> Please UPVOTE if you find the notebook useful, any suggestions are welcome. </span></center>

# [Back to contents(click here)](#content)
