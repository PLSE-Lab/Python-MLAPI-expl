#!/usr/bin/env python
# coding: utf-8

# ## Guess Who!
# 
# This Notebook is a quick bit of fun to try and create an AI playing Guess Who. This notebook shows the method used to train an image classifier on each attribute of the dataset. The final game can be found at https://github.com/JonnyEvans321/Guess_Who_celebrities
#  
# To save time I started with Marcos Alvarado's gender recognition notebook (http://www.kaggle.com/bmarcos/image-recognition-gender-detection-inceptionv3/notebook), thanks Marcos! Note that due to Kaggle's memory limitation, I am using a reduced amount of images to train and validate.
# 
# ## Instructions:
# 1. Pick an attribute from this list to ask our AI: 5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald, Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair, Blurry, Brown_Hair, Bushy_Eyebrows, Chubby, , Double_Chin, Eyeglasses, Goatee, Gray_Hair, Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open, Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, Pointy_Nose, Receding_Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, Wavy_Hair, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie, Young
# 
# 2. Click the 'fast-forward' button and scroll to the bottom of this notebook. Wait for a Console message to say 'TRAINING COMPLETE', then press the 'play' button on the penultimate code cell (section).
# 
# ## Dataset
# 
# For this project we will use the CelebA dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), which is available on Kaggle.
# 
# Description of the CelebA dataset from kaggle (https://www.kaggle.com/jessicali9530/celeba-dataset): 
# 
# ### Overall
# 
# 202,599 number of face images of various celebrities
# 10,177 unique identities, but names of identities are not given
# 40 binary attribute annotations per image
# 5 landmark locations
# 
# ### Data Files
# 
# - <b>img_align_celeba.zip</b>: All the face images, cropped and aligned
# - <b>list_eval_partition.csv</b>: Recommended partitioning of images into training, validation, testing sets. Images 1-162770 are training, 162771-182637 are validation, 182638-202599 are testing
# - <b>list_bbox_celeba.csv</b>: Bounding box information for each image. "x_1" and "y_1" represent the upper left point coordinate of bounding box. "width" and "height" represent the width and height of bounding box
# - <b>list_landmarks_align_celeba.csv</b>: Image landmarks and their respective coordinates. There are 5 landmarks: left eye, right eye, nose, left mouth, right mouth
# - <b>list_attr_celeba.csv</b>: Attribute labels for each image. There are 40 attributes. "1" represents positive while "-1" represents negative
# 
# ---
# 

# ### Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import cv2    
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.optimizers import SGD

from IPython.core.display import display, HTML
from PIL import Image
from io import BytesIO
import base64
import gc

plt.style.use('ggplot')

get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
print(tf.__version__)


# ### Set the variables

# In[ ]:


# set variables 
main_folder = '../input/celeba-dataset/'
images_folder = main_folder + 'img_align_celeba/img_align_celeba/'

TRAINING_SAMPLES = 10000
VALIDATION_SAMPLES = 2000
TEST_SAMPLES = 2000
IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 16
#just one epoch to save computing time, for more accurate results increase this number
NUM_EPOCHS = 1

#what characteristic are we going to train and test for? (note if set to 'all' it'll use all of them)
ATTR='Blond_Hair'


# ## Getting the model ready
# 
# We're going to use Inception v3, TensorfFlow's image recognition model, which was trained for the ImageNet Large Visual Recognition Challenge using the data from 2012.
# 
# 'I can use the model too?? Aw shucks!'

# In[ ]:


# Import InceptionV3 Model
inc_model = InceptionV3(weights='../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top=False,
                        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

#inc_model.summary()


# <h2>Inception-V3 model structure</h2>
# This is the structure of Inception-V3.
# 
# 
# <img src="https://i.imgur.com/kdXUzu1.png" width="1000px"/>
# source: https://hackathonprojects.files.wordpress.com/2016/09/74911-image03.png
# 
# We're not going to include the final 5 layers, because we're transfer learning to our task. These layers will be replaced for the following layers:

# In[ ]:


#Adding custom Layers
x = inc_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)


# <h2>New classification layers</h2>
# Classification layers to be trained with the new model.
# <img src="https://i.imgur.com/rWF7bRY.png" width="800px"/>

# In[ ]:


# create the model 
model_ = Model(inputs=inc_model.input, outputs=predictions)

# Lock initial layers to not be trained
for layer in model_.layers[:52]:
    layer.trainable = False

# compile the model
model_.compile(optimizer=SGD(lr=0.0001, momentum=0.9)
                    , loss='categorical_crossentropy'
                    , metrics=['accuracy'])


# ## Getting data
# 
# We will be using the CelebA Dataset, which has images of 178 x 218 px. 

# ### Load the attributes (characteristics) of every picture
# File: list_attr_celeba.csv

# In[ ]:


# import the data set that include the attribute for each picture
df_attr = pd.read_csv(main_folder + 'list_attr_celeba.csv')
df_attr.set_index('image_id', inplace=True)
df_attr.replace(to_replace=-1, value=0, inplace=True) #replace -1 by 0
df_attr.shape

#if you choose to train on all attributes, set ATTR='all', else write out the desired attributes in a list
if(ATTR=='all'):
    ATTR=list(df_attr.columns)
elif(isinstance(ATTR, str)):
    ATTR=[ATTR]

# List of available attributes
for i, j in enumerate(df_attr.columns):
    print(i, j)


# ## Recommended splits for training, validation and testing
# 
# The recommended partitioning of images into training, validation, testing of the data set is: 
# * 1-162770 are training
# * 162771-182637 are validation
# * 182638-202599 are testing
# 
# The recommended partition is in file <b>list_eval_partition.csv</b>
# 
# However, due to Kaggle's restrictions, we will be using a reduced number of images:
# 
# * Training 20000 images
# * Validation 5000 images
# * Test 5000 Images
# 

# In[ ]:


# Recomended partition
df_partition = pd.read_csv(main_folder + 'list_eval_partition.csv')
df_partition.head()

# display counter by partition
# 0 -> TRAINING
# 1 -> VALIDATION
# 2 -> TEST
df_partition['partition'].value_counts().sort_index()

df_partition.set_index('image_id', inplace=True)


# This degree project explains how imbalanced training data can impact on CNNs models:
# 
# https://www.kth.se/social/files/588617ebf2765401cfcc478c/PHensmanDMasko_dkand15.pdf
# 
# So we will create functions that will help us to create balanced partitions.

# In[ ]:


def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img)/255.
    x = x.reshape((1,) + x.shape)

    return x

def generate_df(df,partition, attr, num_samples):
    '''
    partition
        0 -> train
        1 -> validation
        2 -> test
    '''
    
    #The sample size is at most the number stated above, but at least the size of the smallest class of the dataframe. This results in some uncommon attributes (e.g. sideburns) having to train on very few samples.
    min_class_size=min(len(df[(df['partition'] == partition) & (df[attr] == 0)]),len(df[(df['partition'] == partition) & (df[attr] == 1)]) )
    sample_size=int(num_samples/2)
    if(min_class_size<int(num_samples/2)):
        sample_size=min_class_size
    
    df_ = df[(df['partition'] == partition) & (df[attr] == 0)].sample(sample_size)
    df_ = pd.concat([df_,
                      df[(df['partition'] == partition) 
                                  & (df[attr] == 1)].sample(sample_size)])

    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(images_folder + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        y_ = np_utils.to_categorical(df_[attr],2)
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(images_folder + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target[attr])

    return x_, y_


# ## Training 
# Function to train the model

# In[ ]:


def training(ATTR, train_generator,x_valid,y_valid):
    #https://keras.io/models/sequential/ fit generator
    checkpointer = ModelCheckpoint(filepath='weights.best.inc.'+ATTR+'.hdf5', 
                                   verbose=1, save_best_only=True)

    hist = model_.fit_generator(train_generator
                         , validation_data = (x_valid, y_valid)
                          , steps_per_epoch= TRAINING_SAMPLES/BATCH_SIZE
                          , epochs= NUM_EPOCHS
                          , callbacks=[checkpointer]
                          , verbose=1
                        )
    return hist


# ## Step 3: Pre-processing Images: Data Augmentation
# 
# Data Augmentation allows us generate images with modifications from our dataset. The model will learn from these variations (changing angle, size and position), which will help it to predict new images which could have the same variations in position, size and angle.
# 
# Heres the function to do that:

# In[ ]:


def generator(ATTR,df_partition):    
    # join the partition with the chosen attribute in the same data frame
    df_par_attr = df_partition.join(df_attr[ATTR], how='inner')

    # Create Train dataframes
    x_train, y_train = generate_df(df_par_attr,0, ATTR, TRAINING_SAMPLES)

    # Create Validation dataframes
    x_valid, y_valid = generate_df(df_par_attr,1, ATTR, VALIDATION_SAMPLES)

    # Train - Data Preparation - Data Augmentation with generators
    train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
    )

    train_datagen.fit(x_train)

    train_generator = train_datagen.flow(x_train, y_train,batch_size=BATCH_SIZE)
    
    del x_train, y_train
    
    return train_generator, x_valid, y_valid


# ## Let's get training
# 
# Now that we have all the functions we need, lets train our model(s). This cell takes each attribute, and trains an Inception model to recognise the feature.

# In[ ]:


#for each attribute, run the necessary functions in order to train, and then save an Inception model for the task
for attr in ATTR:
    print('Learning to recognise: ',attr,', which is attribute',ATTR.index(attr)+1,' of ',len(ATTR))
    train_generator, x_valid, y_valid=generator(attr,df_partition)
    #gotta save memory
    gc.collect()
    training(attr, train_generator,x_valid,y_valid)
    print(' ')
    print(' ')
    #gotta save memory
    gc.collect()


# ## Let's see some results
# 
# Now that we have a trained model for each of the attributes we're interested in, it's time to see how the model(s) did. The cell below contains some functions that we're going to use to do that nicely.

# In[ ]:


#function to load the image of the character we want to predict for
def img_to_display(filename):
    # inspired on this kernel:
    # https://www.kaggle.com/stassl/displaying-inline-images-in-pandas-dataframe
    # credits to stassl :)
    
    i = Image.open(filename)
    i.thumbnail((200, 200), Image.LANCZOS)
    
    with BytesIO() as buffer:
        i.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()
    
#use some cool html to print out the prediction in a nice way
def display_result(filename, test_attr, prediction, target):
    '''
    Display the results in HTML
    
    '''
    #convert ai speak to what we want to say is our true value of an attribute
    attribute_target = {0: 'False'
                    , 1: 'True'}
    
    #finds out if the model is going to predict true or false
    attribute = 'True'
    if prediction[1] <= 0.5:
        attribute = 'False'
            
    display_html = '''
    <div style="overflow: auto;  border: 2px solid #D8D8D8;
        padding: 5px; width: 420px;" >
        <img src="data:image/jpeg;base64,{}" style="float: left;" width="200" height="200">
        <div style="padding: 10px 0px 0px 20px; overflow: auto;">
            <h3 style="margin-left: 15px; margin-top: 2px;">Does this person have {}?</h3>
            <p style="margin-left: 15px; margin-top: -0px; font-size: 12px">Prediction: {}</p>
            <p style="margin-left: 15px; margin-top: -16px; font-size: 12px">{} probability.</p>
            <p style="margin-left: 15px; margin-top: -0px; font-size: 12px">Truth: {}</p>
            <p style="margin-left: 15px; margin-top: -16px; font-size: 12px">Filename: {}</p>

        </div>
    </div>
    '''.format(img_to_display(filename)
               , test_attr
               , attribute
               , "{0:.2f}%".format(round(max(prediction)*100,2))
               , attribute_target[target]
               , filename.split('/')[-1]
               
               )

    display(HTML(display_html))

#predict an image
def attribute_prediction(filename):
    '''
    predict the attribute
    
    input:
        filename: str of the file name
        
    return:
        array of the prob of the targets.
    
    '''
    
    im = cv2.imread(filename)
    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (178, 218)).astype(np.float32) / 255.0
    im = np.expand_dims(im, axis =0)
    
    # prediction
    result = model_.predict(im)
    prediction = np.argmax(result)
    
    return result
    


# ### Reminder or the available attributes:

# In[ ]:


#what are the attributes?
for attr in ATTR:
    print(attr)


# ## Let's see the AI in action!
# 
# The user selects which attribute to look at by changing the 'test_attr' variable. Then by running the cell, a prediction of one of the test set images is made. This is compared with the truth to see how it did.

# In[ ]:


#choose which attribute to use
test_attr='Blond_Hair'

#Run this cell to see the AI in action!

# make the dataframe with the chosen attribute
df_par_attr = df_partition.join(df_attr[ATTR], how='inner')

#first, load the relevant model weights
model_.load_weights('weights.best.inc.'+test_attr+'.hdf5')

#select random image from the test set
df_to_test = df_par_attr[(df_par_attr['partition'] == 2)].sample(10)

for index, target in df_to_test.iterrows():
    result = attribute_prediction(images_folder + index)
    
    #display result
    display_result(images_folder + index, test_attr, result[0], target[test_attr])


# Look above for the image classifier outputs.
# 
# Conclusions:
# * Wow, transfer learning really works! I'm sure I'd never get such good results training a model from scratch. The fact it could re-train to work on each attribute in one epoch with ~600 images blows my mind. By just saving the weights of the re-trained layers, I can see how transfer learning could be used in a lot of real-world AI systems, becoming really useful really quickly, while not needing to store large models or datasets.
# * I started thinking about how transfer learning can be used for traninig a futue artificial general intelligence (AGI) system. First thought it, this isn't how our human brains work is it? Whenever we start a different task, we don't alter the weights (biological structure) of our neurons, right? We are just able to do a different task with exactly the same weights in the brain. So for an AGI, wouldn't it be best to train on a great many tasks with the same NN structure? Perhaps with some sort of gating network for the AGI to decide which task it was doing, then use a particular NN (from a system of many) for that task? I'd like to become more informed on transfer learning research for AGI, so any thoughts on this would be welcomed.
