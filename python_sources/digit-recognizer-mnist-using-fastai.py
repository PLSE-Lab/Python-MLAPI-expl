#!/usr/bin/env python
# coding: utf-8

# # Note I am currently working on fastAIs course, this is to help me learn.
# - This is for me to practice and develop my data science skills using the FastAI library.
# - I will be adding many comments as I learn and figure things out.

# <img src="https://media.giphy.com/media/YaJknABE4uFUY/giphy.gif">
# 
# <a id="Techniques"></a>
# ## Brief Description of Techniques used :-
# * ***Pre processed data*** by data wrangling & putting into correct structure
#     * Involved converting csv file into images, done using numpy & panda & self made functions to extract files & labels and convert arrays into images.
# * ***Data visualization*** for inspection.
#     * Utilizing matplotlib.
# * ***Transfer learning*** technique using fastAI utilizing resnet34 CNN model architecture.
#   * Involved unfreezing layers, use LR_find() method to find a good range of learning rate to then slice and use as range when training entire model. LR_find() method is an approach proposed in [a paper by Leslie Smith](http://arxiv.org/abs/1803.09820) that was later implemented into the fastAI library as a LR_find() function for the model.
# * Implemented techniques to ***optimize model*** such as:
#   * Dropout probability - randomly deactivates nodes to reduce overfitting.
#   * Transforming and normalizing the data to ensure efficiency and spatial variance.
#   * Usual data augmentation such as zoom, rotate, lighting etc.
#   
# =============================================================
# =============================================================
# 
# # TABLE OF CONTENTS :-
#    
#    - [Introduction](#Understand)
#    
#    - [Brief Description of Techniques Used](#Techniques)
# 
# - **PRE-PROCESSING**
#     - [Data Wrangling & What is It?](#Data-Wrangling)
#     - [Converting .csv to .jpg](#Reshaping-Array)
#     - [Inspecting the Test Data](#TestData)
# - **CREATING DATA OBJECT**
#     - [Initializing the Data Object](#Initializing-Data-Object)
#     - [Inspecting the Data Object](#Inspecting-Data-Object)
# - **REVIEWING ResNet34 CNN CONCEPTUALLY AND ARCHITECTURALLY**
#     - [Examining the ResNet34 Model Architecture from FastAI - CNN](#Model-Architecture)
# - **CREATING MODEL & TRAINING**
#     - [Initializing the CNN](#CNNinitialize)
#     - [Transfer Learning & What is It?](#TransferLearning)
#     - [Learning Rate Finder Method](#LR-Finder)
# - **PREDICTING & EVALUATION**
#     - [Predicting After Model is Trained](#Prediction)
#     - [Evaluation After Training](#Evaluation)
# - **RE-STRUCTURING PREDICTIONS TO SUBMISSION FILE FORMAT SPECIFIED**
#     - [Creating Submission .csv File](#Submission)
# 

# <a id="Understand"></a>
# # Understanding the problem :-
# The goal of this problem is to be able to take an image of a handwritten digit and determine what that digit is. It is judged on the accuracy. i.e. the error rate.
# 
# ### LOOKING AT THE DATA :-
# ### *train.csv*
# * Greyscale images from zero through nine
# * 42,000 images
# * File contains all necessary information for training the model
# * Each row is one image i.e. an image is a 1D array of 1x784
# * First column of each image is the label. It tells us which digit is shown.
# * Other 784 columns are the pixels for each digit, they should be read like this when in image-square format as sqr(784) = 28
# 
# `000 001 002 003 ... 026 027`  
# `028 029 030 031 ... 054 055`  
# `056 057 058 059 ... 082 083`  
# ` |   |   |   |  ...  |   |  `  
# `728 729 730 731 ... 754 755`  
# `756 757 758 759 ... 782 783`  
# 
# 
# ### *test.csv*
# * greyscale images from zero through nine
# * structure is the same as in train.csv, but there are no labels thus only 784 columns not 785.
# * these 28,000 images are used later to test how good our model is
# 
# ### *sample_submission.csv*
# * show us, how to structure our prediction results to submit them to the competition
# * 28,000 images
# * we need two columns: ImageId and Label
# * the rows don't need to be ordered
# * the submission file should look like this:
# 
# `ImageId, Label`  
# `1, 3`  
# `2, 4`   
# `3, 9`  
# `4, 1`  
# `5, 7`  
# `(27995 more lines)`
# 
# ### What is MNIST?
# MNIST is the perfect dataset to get started learning more about pattern recognition and machine learning. That is why people call it the "Hello World of machine learning". It's a large database of handwritten digits. There are a total of 70.000 grayscale images, each is 28x28 pixels. They show 10 different classes representing the numbers from 0 to 9. The dataset is split into 60.000 images in the training set and 10.000 in the test set. This competition is based on the MNIST dataset. However, the train-test distribution is different. Here, there are 42.000 images in the training set and 28.000 images in the test set. MNIST was published by the godfather of CNNs, Yann LeCun.
# 

# ## Importing fastai libraries:-
# - importing fastai library which runs ontop of PyTorch framework.

# In[ ]:


# the following three lines are suggested by the fast.ai course
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

# hide Kaggle notebook warnings
import warnings
warnings.simplefilter('ignore')

# Importing fastai library
from fastai import *
from fastai.vision import *


# ## **Data processing libraries:-**

# In[ ]:


# to get all files from a directory
import os

# to easier work with paths
from pathlib import Path

# to read and manipulate .csv-files
import pandas as pd

# For linear algebra processes e.g. manipulating arrays.
import numpy as np 


# In[ ]:


INPUT = Path("../input/digit-recognizer")
os.listdir(INPUT)


# This tells me there are 3 .csv files contained within the /digit-recognizer path, i.e. folder.
# 
# * Note : Now "INPUT" path object leads to our data.

# Looking into the test & train csv files. Note: sample_submission is just a structural sample to follow when I do submit the final submission.

# In[ ]:


train_df = pd.read_csv(INPUT/"train.csv")
train_df.head(3)


# In[ ]:


test_df = pd.read_csv(INPUT/"test.csv")
test_df.head(3)


# The files are in the proper structure.

# <a id="Data-Wrangling"></a>
# # Data Wrangling:-
# According to the fastai documentation for computer vision, it only accepts image files. But I am given .csv files so I must perform **Data Wrangling** i.e. converting one data format to another more appropriate and valuable format, which is an image in this particular case.
# 
# * So I have to convert the .csv files which contain the pixel value for each pixel (represented in columns) into images.
# 
# fastai documentation source : https://docs.fast.ai/vision.data.html#ImageDataBunch

# The "ImageDataBunch" class in fastai is the data object that the model uses and it sees folder names as labels, thus the images must be in a folder structure like so:
# 
# `train\0\ arbitraryName.jpg  
#          \ arbitraryName.jpg
#           .....etc
#        \1\ arbitraryName.jpg
#           ....etc
#        \2\ arbitraryName.jpg
#           ...etc`
#           
# Where \0 && \1 && \2 etc. up to 9 are the folders/labels i.e. the class/digit in the training folder. Which is dictated by the first column in the .csv file.
# 
# The test folder would look like this since there are no labels.
#     
# `test\
#       arbitraryName.jpg
#       ....etc`
# 
# 
#      

# ### So, now I will  create these folder structures :-
# 

# P.S. Kaggle input folders are always read-only so create folders outside of input folder.

# In[ ]:


# CREATING PATHS
TRAIN = Path("../train")
TEST = Path("../test")


# In[ ]:


# CREATING TRAINING DIRECTORIES

for index in range(10):
    try:
        os.makedirs(TRAIN/str(index))
    except:
        pass


# In[ ]:


# CONFIRMING LABELS

sorted(os.listdir(TRAIN))


# In[ ]:


# CREATING TEST DIRECTORY

try:
    os.makedirs(TEST)
except:
    pass


# In[ ]:


os.listdir(TEST) 


# In[ ]:


if os.path.isdir(TRAIN):
    print('Train directory has been created')
else:
    print('Train directory creation failed.')

if os.path.isdir(TEST):
    print('Test directory has been created')
else:
    print('Test directory creation failed.')

# I was having problems with the directory objects so this is
# just troubleshooting


# <a id="Reshaping-Array"></a>
# # Reshaping array & converting to image :-
# 
# QUICK NOTES:-
# * Panda to extract label and rows i.e. the flat array images (1x784)
# * Numpy to reshape each flat array (1x784) image row into square image (28x28)
# * PIL library to create .jpg image from the (28x28) numpy array via its .fromarray function.
# * Creating python code - Defining functions for easier processing, and FOR loops to save files into training folder and testing folder. Note that training folder images must be assigned to correct labeled folder so I have to extract the 1st column of training image and save the training images in the appropriate folder based on the label.
# 
# ### Creating a function that recieves a 1x784 array (one image) , and file path. Which then converts vector to 28x28 and puts into its file path.

# In[ ]:


#Numpy already imported as np

# import PIL to create images from arrays
from PIL import Image

def saveDigit(digit, filepath):
    digit = digit.reshape(28,28)
    digit = digit.astype(np.uint8) # uint8 is the image type as 2^8-1 is the highest pixel value possible

    img = Image.fromarray(digit)
    img.save(filepath)


# ***saveDigit(digit, filepath) :-***
# * 'digit' argument is the 1x784 array that gets converted to a 28x28 array of type unint8, which then gets converted to a .jpg image and saved to the 'filepath' argument object.
# * I have to ensure the filepath object that will be given to the function is the appropriate one for the digit object.

# ## Saving training & testing images loops & correct filepath :-
# * Using panda library to extract the label & 1x784 arrays (digit)
# * Creating folder directory using label
# * Creating a filepath directory object appropriate to digit at hand
# * Running the 'digit' and 'filepath' through the saveDigit() function.
# 
# Note: For testing images there are no labels.

# In[ ]:


# Using panda library & its directory objects
# SAVING TRAINING IMAGES
for index, column in train_df.iterrows(): #for all rows in file
    
    label,digit = column[0], column[1:] #label is first column & digit is rest
    digit = digit.values                #i.e. digit is now a 1x784 array, just one row
   
    folder = TRAIN/str(label) #create folder based on label of row
    filename = f"{index}.jpg" #arbitrary index from loop
    filepath = folder/filename #filepath now is the directory of the 'digit' in the correct folder/label.
                                
    # Lastly, using the saveDigit function I created to save & convert the 
    # 1x784 array 'digit' into an image and into the 'filepath' directory,
    # filepath directory
    saveDigit(digit, filepath)


# In[ ]:


# SAVING TESTING IMAGES
for index, digit in test_df.iterrows(): 

   folder = TEST
   filename = f"{index}.jpg"
   filepath = folder/filename
   
   digit = digit.values # .iterrows is row-by-row loop so digit is 1x784
   
   saveDigit(digit, filepath)


# <a id="TestData"></a>
# # Displaying test data using matplotlib
# * I will display the training data shortly, using fastAI's DataBunchObject as it is more convenient. There seems to be no way to visualize the test data appropriately using fastAI.

# In[ ]:


import matplotlib.pyplot as plt

def displayTestingData():
    fig = plt.figure(figsize=(5, 10))
    
    paths = os.listdir(TEST)
    
        
    for i in range(1, 51):
        randomNumber = random.randint(0, len(paths)-1)
        image = Image.open(TEST/paths[randomNumber])
        
        ax = fig.add_subplot(10, 5, i)
        ax.axis("off")
        
        plt.imshow(image, cmap='gray')
    plt.show()


# In[ ]:


print("TESTING DATA SAMPLES")
displayTestingData()


# In[ ]:


# LOOKING AT ONE SINGLE IMAGE UP CLOSE 

image_path = TEST/os.listdir(TEST)[6]
image = Image.open(image_path)
image_array = np.asarray(image)


fig, ax = plt.subplots(figsize=(15, 15))

img = ax.imshow(image_array, cmap='gray')

for x in range(28):
    for y in range(28):
        value = round(image_array[y][x]/255.0, 2)
        color = 'black' if value > 0.5 else 'white'
        ax.annotate(s=value, xy=(x, y), ha='center', va='center', color=color)

plt.axis('off')
plt.show()


# In[ ]:


# Also - A simple function I found from fastai that is able to print quickly.
# but verryyy small -- most likely real size 28x28
testData = ImageList.from_folder(TEST)
testData.open(testData.items[5])


# <a id="Initializing-Data-Object"></a>
# # Initializing ImageDataBunch object i.e. loading data
# **INITIALIZING OUR DATA OBJECT WITH 'ImageDataBunch' :-**
# * 'ImageDataBunch' is an object that represents all the data needed to train a CNN model. It also has factory methods within it that generates the validation and training data (The validation set is a set that model never gets to look at and is taken from its training set, i.e a test/validation set while training, to prevent overfitting)
#     * In fastAI all the data objects will be "DataBunch" objects that will have all the data augmentation and extraction and normalization components in it etc. For images its just "ImageDataBunch"
# * Uses of the ImageDataBunch :-
#     * Is our data object that the model object will use.
#     * Can configure hyperparameters such as validation split percentage, batch count, size of image, number of workers, transformation (i.e. data augmentation). 
#     * Note: The transformation argument takes a "get_transforms()" object as a parameter. This object already has pre-existing settings for augmentation that have proven to be useful such as rotation, zooming, lighting, warping, etc. More can be seen here in the functions documentation. https://github.com/fastai/fastai/blob/master/fastai/vision/transform.py#L308
#     * Normalization also has its pre-existing settings from fastai, but after looking into the document I found that they have a setting specifically for MNIST data titled "mnist_stats" so I will use that as my normalization setting //I attempted another pre-existed setting "imagenet_stats" which proved to do better.
#     
#     
# **np.random.seed( ):-**
# * The "ImageDataBunch" object creates a validation set randomly (using numpy) each time the code block is run.
# * So to maintain reproducibility of the model, you set the 'seed' (i.e. the specific set of data/images) of which the "ImageDataBunch" objects validation set gets randomly chosen from.
# * The integer parameter used is arbitrary, but you just have to use that same number again for reproducibility otherwise the model is using a different validation set every time it runs. 
# 
# 

# In[ ]:


# Specifying transformation object not to flip images for obvious reasons.
tfms = get_transforms(do_flip=False)


# In[ ]:


np.random.seed(9)


# In[ ]:


# More directory troubleshooting
print('test : ',TEST)
print('train: ', TRAIN)
print(type(TEST)) # I kept getting a PosixPath type error 


# In[ ]:


data = ImageDataBunch.from_folder(
    path = ("../train"), # using the actual string location works!
    test = ("../test"),
    valid_pct = 0.15, #15% of training set split into validation
   #valid_pct=0.2
    # bs = 16,
    bs = 256,
    ds_tfms = tfms,
    size = 28,
    num_workers = 0
    )

data.normalize(imagenet_stats)


# <a id="Inspecting-Data-Object"></a>
# ### Inspecting the data object :-

# In[ ]:


data.test_ds.x[0] # showing first element of test data


# In[ ]:


# DISPLAYING TRAINING DATA WITH LABELS !! ITS WORKING

data.show_batch( rows=3, figsize=(7,8) ) #//should display 3 rows of images w the class ontop 


# In[ ]:



len(data.train_ds), len(data.valid_ds), data.c, len(data.test_ds)


# ( # of training images, # of validation images, # of classes, # of test images)

# In[ ]:


imagenet_stats, mnist_stats


# <a id="CNNinitialize"></a>
# # Data is ready - CNN LEARNING TIME:-

# * Regarding my implementation, I will utilize the transfer learning technique on fastAI's ResNet34 CNN model architechture. There is a ResNet18 & a Resnet50 but I will stick to the 34 size.
# * Note that you can build your own architecture from scratch as I've done before using Keras, but that method is more time-consuming as you have to manually input the layers, its neurons, activation functions, feature maps, pooling layers, which all contain little configurations such as padding and feature map sizing etc...This most likely will not end up being as robust or even accurate of a model as the ResNet.

# In[ ]:


doc(cnn_learner)


# In[ ]:


modelCNN = cnn_learner(
    data, 
    base_arch = models.resnet34, 
   # DROPOUT RATE IS DEFAULT SET TO 0.5 i.e 50%
    metrics = accuracy,
    model_dir="/tmp/models", 
    callback_fns=ShowGraph 
    )


# In[ ]:


# Looking at model architecture

modelCNN.model


# <a id="Model-Architecture"></a>
# ## Brief Inspection of Model Architecture
# Looks exactly like Keras.
# - Kernel sizes are the feature detector, 
# - ReLU function to prevent linearity in images, 
# - Also using the common practice of doubling the neurons as more layers are put on,
# - MaxPooling to further highlight the significant parts as it only contains the highest values of the Feature Map, 
# - After a couple layers the stride size of the feature map goes down.
# - Flattens all pooled feature map pixel values into a column (vector)
# - Flattened vector is the input values for the final fully connected ANN layers (they did throw in ANN layers in the middle as well, in my experience I've only added them on top at the end)
#     - The fully connected ANN is integral for training as it adjusts all parameters (weights, attributes, EVEN FEATURE DETECTORS etc.) through the training process, which amplifies the CNNs capability.
# 
# Basically in the form of: 
# Convolution layer ( Input Image -> Feature detector -> Feature map -> ReLU) 
# -> POOLING LAYER ( Feature Map -> Pool Feature Detector -> Pooled Feature map). For every feature map we create a pooled feature map.
# -> Flattened pooled feature map -> ANN -> Train
# 
# All the basic concepts I've learnt, just put into play in a larger complex scale.
# 
# <img src="https://media.giphy.com/media/3oKIPlLZEbEbacWqOc/giphy.gif">

# <a id="TransferLearning"></a>
# # On Transfer Learning:-
# * **It is basically using a pre-trained model annndd..**
#     1. Training the last layers of it so as to not corrupt its foundations as during gradient descent (re-calibration of weights based on errors) the model will configure the weights equally throughout, i.e. weights from layer 1 will be affected as much as layer 30. This is a big no-no as the layers in the model correlate to fundamentals of vision, say layer 1-5 is figuring out shapes and what they mean, then layers 5-10 might be it figuring out that these shapes are different objects and layers 25-30 could be it distinguishing between two similiar faces.
#     2. Then 'unfreezing' all the layers and use the learning rate finder method to figure out a good learning rate sweetspot to train the entire model, BUT this method involves index slicing the learning rate, so that the beginning layers are not affected as much as the later ones. Specifically by examining the validation/loss function and finding where the loss starts to decrease. It is an approach based off [a paper by Leslie Smith](http://arxiv.org/abs/1803.09820) that has been incorporated into the fastAI library as lr_find(). 
# * In this case I am using Resnet34 which has been trained on looking at ~1.5mil pics of things and 1,000 classes/categories using an image set called ImageNet. This makes everything easier, and most importantly MORE EFFECTIVE!!

# In[ ]:


modelCNN.fit_one_cycle(3)


# In[ ]:


modelCNN.save('model34-3')


# In[ ]:


modelCNN.load('model34-3')


# <a id="LRfinder"></a>
# ## With just 3 epochs it achieved a high accuracy. Now lets unfreeze and use learning rate finder method!

# In[ ]:


# UNFREEZING THE LAYERS !

modelCNN.unfreeze()


# In[ ]:


# LR_FIND() METHOD

modelCNN.lr_find()


# In[ ]:


modelCNN.recorder.plot()


# ### FINDING LR RANGE :-
# I can see that the loss starts to go down at a LR of 1e-04ish and proceeds to keep going down until it starts to pick up again at 1e-02, at 1e-03 it seems to be a little steeper so I will try a range of 1e-03 -> 1e-02 , so that the LR starts off at 1e-03 and gradually increases to 1e-02
# 

# In[ ]:


modelCNN.fit_one_cycle( 30, slice(1e-3, 1e-2))


# # 99.6 % accuracy - training time 35 mins ! SUCCESS
# <img src="https://media.giphy.com/media/3o7bu6t4kZRB906jGo/giphy.gif">
# 

# <a id="Evaluation"></a>
# # EVALUATION :-

# In[ ]:


# VIEWING SOME PREDICTIONS FROM TRAINING

modelCNN.show_results(3, figsize= (7,7))


# In[ ]:


# VIEWING TOP 9 IMAGES WITH HIGHEST LOSSES

interp = ClassificationInterpretation.from_learner(modelCNN)


# In[ ]:


interp.plot_top_losses(9, figsize=(7, 7))


# Ya I'm not dissapointed, I would get these wrong as well.

# In[ ]:


# CONFUSION MATRIX

interp.plot_confusion_matrix()


# <a id="Prediction"></a>
# # PREDICTIONS :-

# In[ ]:


# USING .get_preds(DatasetType.Test) WHICH RETURNS EACH CLASS PREDICTION
# FOR object 'y' i.e. the image.

class_score , y = modelCNN.get_preds(DatasetType.Test)


# In[ ]:


probabilities = class_score[0].tolist()

# For loop running prediction percentages for one object.
[f"{index}: {probabilities[index]}" for index in range(len(probabilities))]


# ## We only need the highest chance prediction :-
# ### For this one it would be a '1'.

# In[ ]:


class_score = np.argmax(class_score, axis=1)


# In[ ]:


class_score[0].item()


# <a id="Submission"></a>
# # Creating submission file:-

# In[ ]:


# VIEWING THE CORRECT FORMAT GIVEN TO US

sample_submission =  pd.read_csv(INPUT/"sample_submission.csv")
display(sample_submission.head(2))
display(sample_submission.tail(2))


# In[ ]:


# CREATING MY SUBMISSION FILE

# # # Fixing up ImageID
# remove file extension from filename
ImageId = [os.path.splitext(path)[0] for path in os.listdir(TEST)]
# typecast to int so that file can be sorted by ImageId
ImageId = [int(path) for path in ImageId]
# +1 because index starts at 1 in the submission file
ImageId = [ID+1 for ID in ImageId]


# In[ ]:


# Using class_score object from prediction as labels.
submission  = pd.DataFrame({
    "ImageId": ImageId,
    "Label": class_score
})
# submission.sort_values(by=["ImageId"], inplace = True)
submission.to_csv("submission.csv", index=False)
display(submission.head(3))
display(submission.tail(3))


# **DONT FORGET TO TURN ON GPU BEFORE TRAINING**
# <a id="Techniques"></a>
# ## Techniques used :-
# * ***Pre processed data*** by data wrangling & putting into correct structure
#     * Involved converting csv file into images, done using numpy & panda & self made functions to extract files & labels and convert arrays into images.
# * ***Data visualization*** for inspection.
#     * Utilizing matplotlib.
# * ***Transfer learning*** technique using fastAI utilizing resnet34 CNN model architecture.
#   * Involved unfreezing layers, use LR_find() method to find a good range of learning rate to then slice and use as range when training entire model. LR_find() method is an approach proposed in [a paper by Leslie Smith](http://arxiv.org/abs/1803.09820) that was later implemented into the fastAI library as LR_find().
# * Implemented techniques to ***optimize model*** such as:
#   * Dropout probability - randomly deactivates nodes to reduce overfitting.
#   * Transforming and normalizing the data to ensure efficiency and spatial variance.
#   * Usual data augmentation such as zoom, rotate, lighting etc.
#   
# =============================================================
# =============================================================
# 
# # TABLE OF CONTENTS :-
#    
#    - [Introduction](#Understand)
#    
#    - [Brief Description of Techniques Used](#Techniques)
# 
# - **PRE-PROCESSING**
#     - [Data Wrangling & What is It?](#Data-Wrangling)
#     - [Converting .csv to .jpg](#Reshaping-Array)
#     - [Inspecting the Test Data](#TestData)
# - **CREATING DATA OBJECT**
#     - [Initializing the Data Object](#Initializing-Data-Object)
#     - [Inspecting the Data Object](#Inspecting-Data-Object)
# - **REVIEWING ResNet34 CNN CONCEPTUALLY AND ARCHITECTURALLY**
#     - [Examining the ResNet34 Model Architecture from FastAI - CNN](#Model-Architecture)
# - **CREATING MODEL & TRAINING**
#     - [Initializing the CNN](#CNNinitialize)
#     - [Transfer Learning & What is It?](#TransferLearning)
#     - [Learning Rate Finder Method](#LR-Finder)
# - **PREDICTING & EVALUATION**
#     - [Predicting After Model is Trained](#Prediction)
#     - [Evaluation After Training](#Evaluation)
# - **RE-STRUCTURING PREDICTIONS TO SUBMISSION FILE FORMAT SPECIFIED**
#     - [Creating Submission .csv File](#Submission)
# 
