#!/usr/bin/env python
# coding: utf-8

# # Using convolution neural network to classify fruits
# We will use keras library to build a convolution neural network to classify fruits into different classes. We wil be using kaggle kernel for this analysis. We will also use kaggle's gpu.  In order to do this we go to the settings tab and mark the check box named 'Enable GPU'.  
# As we will be using kaggle kernel for this analysis, the first step would be doing some exploratory data analysis (EDA) and getting data ready for analysis.
# This notebook is intended for performing the EDA and data wrangling and is ***Part 1*** of fruit classification project. This notebook takes users in a step wise manner to getting their data ready for model building.
# We will be working with version 2018.07.01.0 of the 'fruits-360' dataset.

# ## Importing required packages
# Firstly, lets import the packages that we will be using. The particular use of every package is shown in the comments.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np # linear algebra
import pandas as pd # I/O of data
import matplotlib.pyplot as plt # making plots

import os, random, shutil, zlib # directory operations


# ## Navigating different directories: the directory where we will work, the directory where the input data is stored and any directory that we need to create in the directory where we will work.

# We need to specify the location where the data exists. In kaggle the default directory (location where the kernel starts in) is the 'working' directory. We can print this directory's path and it's contents as shown below.

# In[ ]:


# print the path of the current directory
print('Current directory is {}'.format(os.getcwd()))

# print the contents of the current directory
print('Current directory contains the follwoing sub-directories:\n {}'.format(os.listdir())) 


#  If we navigate upstream (to the directory 'kaggle' and print it's contents , we will find a directory named 'input'. This is where the input data exists. The benefit of using kaggle kernel for analyzing kaggle datasets is that we do not need to download this data as it can be accessed directly from the kernel.

# In[ ]:


# print the current directory
print('Kaggle directory contains the following sub-directories:\n {}'.format(os.listdir('../'))) 


# The fruits-360 dataset, which we are using for this analysis, contains 'Test' and 'Training' directories. 

# In[ ]:


print('Input directory contains the following sub-directories:\n {}'.format(os.listdir('../input/fruits-360_dataset/fruits-360')))


# Within each of these are sub-directories containing fruits images. These sub-directories are named after the fruits that they contain so we can use the directory names as the target variables (fruit classes).

# In[ ]:


print('Validation directory contains the following sub-directories:\n {}'.       format(os.listdir('../input/fruits-360_dataset/fruits-360/Test')))


# In[ ]:


print('Training directory contains the following sub-directories:\n {}'.       format(os.listdir('../input/fruits-360_dataset/fruits-360/Training')))


# The first check we need to make is that the directories in the the Validation directory and Training directory are equal in number and have the same names, as we will be using these names as the names of our classes. To do that we will use the assert function as shown below:

# In[ ]:


assert os.listdir('../input/fruits-360_dataset/fruits-360/Test') == os.listdir('../input/fruits-360_dataset/fruits-360/Training')


# As we do not see any error the input data has passed the first check. Now that we know where our data is we can create variables for the paths to these directories.
# But as we can see that we do not have any validation dataset, therefore, we will need to move some images from the 'Test' directory to a new directory. As we only have write access to the 'working' directory, that is where we will create three new directories: 1- for test images, another for training images and the last one for validation images. The 'Training' directory does not need to be created if the kernel is run on an updated datset. The only reason we are creating this directory is because we will save the created directories at the end of the kernel. The reason for saving is that we won't have to run this kernel again when we start building our convolution network. *** Note that the training folder will be copied as is to the working directory***.    
# The following code sets the paths for different directories.

# In[ ]:


# path to validation input directory
validationPathSource = '../input/fruits-360_dataset/fruits-360/Test'
# path to training input directory
trainPathSource = '../input/fruits-360_dataset/fruits-360/Training'

# path to the validation directory to which we will move validation images
validationPathDest = '../working/Validation' 
# path to the test directory to which we will move test images
testPathDest = '../working/Test' 
# path to the test directory to which we will move training images
trainPathDest = '../working/Training'


# Copying images from one directory to another will require alot of directory navigation. To make the workflow easier, it will be helpfull for us if we choose one directory as our reference point. As the paths that we have defined above are relative to the 'working' directory, the 'working' directory is a reasonable choice for a reference point. To make directory navigation simpler, let's first define a function which changes the current directory to the 'working' directory. 

# In[ ]:


def get2working():
    
    """"" This function changes the current directory to the working directory
    regardless of the fact if the current directory is upstream or downstream the
    working directory """""
    
    while True:
        if os.getcwd() == '/kaggle/working': # if we are in the working directory then break
            break
        elif os.getcwd() == '/': # else if we are upstream change it to the working directory
            os.chdir('kaggle/working')
        else:
            os.chdir('..') # else if we are downstream move back a directory untill we are in the working directory


# Another function that will help us for directory navigation is for the creation of folders.  Given a path ending with the directory's name to be created as the input to this function, a folder with that name is created. 

# In[ ]:


def createfolder(pathandname):
    
    """"" Given a path ending with the directory's name to be created as the input to 
    this function, a folder with that name is created """""
    
    get2working() # ensure that the current directory is the working directory
    
    try:
        os.mkdir(pathandname) # make the desired directory
        print('Folder created')
    except FileExistsError:
        print('Folder already exists so command ignored') # ignore if the directory already exits


# Now using the above functions and the paths and directory names already defined, we create the required directories.

# In[ ]:


createfolder(validationPathDest)
createfolder(testPathDest)
createfolder(trainPathDest)


# # Splitting the test data into validation and test set.
# Now we are ready to access the input test data and copy some data from it to the created test folder in our working directory. The contents of this directory will serve as the test data for our model. We will copy the remaining data in the input validation directory to the created validation folder in our working directory. This will serve as the validation data on which we will validate our models

# But, before splitting the data in this way we need to know that no class of fruits (sub-directories in the 'Training' and 'Testing') has zero images. This will also inform us about the number of training and validation images.
# To do this we will create two dictionaries with keys =  class of a fruit and values = the number of fruits in that class. We will then convert these dictionaries to a pandas dataframe and inspect the dataframe. The purpose of creatiing the dataframe is to make the output easily readable. Also we can check for zero or null values in a dataframe easily.
# The following code is used for the achievement of the above described goals.

# In[ ]:


test_dict = {} # empty dictionary to store validation data
train_dict = {} # empty dictionary to store training data
fruit_numbers = {} # empty dictionary to store the above defined 2 dictionaries
get2working() # ensure that the current directory is the working directory

for classes in os.listdir(validationPathSource): # looping over the subdirectories in the validationPath (this can be changed to trainPathSource too as it will make no difference to the result)
    # calculating number of fruits
    test_dict[classes] = len(os.listdir(os.path.join(validationPathSource,classes))) 
    train_dict[classes] = len(os.listdir(os.path.join(trainPathSource,classes)))

fruit_numbers['Test'] = test_dict # assigning val_dict to 'Validation' key
fruit_numbers['Training'] = train_dict # assigning test_dict to 'Training' key

df_fruit_numbers = pd.DataFrame.from_dict(fruit_numbers) # creating a dataframe from fruit_numbers
print(df_fruit_numbers) # visualizing the dataframe

# making sure that no values are null or zero. The following code should not print empty dataframes
print(df_fruit_numbers[(df_fruit_numbers.Training == 0)])
print(df_fruit_numbers[(df_fruit_numbers.Training == np.nan)])
print(df_fruit_numbers[(df_fruit_numbers.Test == 0)])
print(df_fruit_numbers[(df_fruit_numbers.Test == np.nan)])                  


# ### Important insights from the dataframe:    
# No folder is empty. If we had an empty folder we would have made a note of it BUT **wouldn't have deleted it. We would have deleted it after we had split our data. The reason being, the source data can be updated any time. An empty folder today can contain images tomorrow.** So if we delete a folder at this step the same folder can contain images when the data set is updated.        

# Let's save the data frame to a csv file

# In[ ]:


df_fruit_numbers.to_csv('FruitNumbers.csv') 


# Now we can split the validation data by copying 25% of the data to the created test folder and the remaining data to the created validation folder.

# In[ ]:


get2working() # Changing path to working directory
os.chdir(validationPathSource) # Changing path to input training folder
fruitnames = [file for file in os.listdir()]; # Storing names of fruits (sub folders within the training folder) in a list

# Looping over the list of fruit names
for fruit in fruitnames:
    get2working() # Changing path to working directory
    validationpath = os.path.join(validationPathDest,fruit) # Creating path for a specifc fruit for the output validation folder
    testpath = os.path.join(testPathDest,fruit) # Creating path for a specifc fruit for the output test folder
    trainpath = os.path.join(trainPathDest,fruit) # Creating path for a specifc fruit for the output training folder
    
    sourcepath = os.path.join(validationPathSource,fruit) # Creating path for a specifc fruit for the source validation folder
    sourcepathtrain = os.path.join(trainPathSource,fruit) # Creating path for a specifc fruit for the source training folder
    
    os.mkdir(testpath) # Creating a folder for a specific fruit in the test directory
    os.mkdir(validationpath) # Creating a folder for a specific fruit in the validation directory
    os.mkdir(trainpath) # Creating a folder for a specific fruit in the training directory
    
    os.chdir(sourcepath) # Changing path to the source directory
    randomsample = random.sample(os.listdir(),len(os.listdir())) # Sampling random fruit images for a certain fruit
    
    get2working() # Changing path to the working directory
    # Copying the first 25% fruit images from the source folder (randomaly sampled already) and copying them to the test folder 
    for k in range(0,len(randomsample)//4):
        shutil.copy(os.path.join(sourcepath,randomsample[k]),testpath)
    
    # Copying the rest of fruit images from the source folder (randomaly sampled already) and copying them to the validation folder
    for k in range(len(randomsample)//4,len(randomsample)):
        shutil.copy(os.path.join(sourcepath,randomsample[k]),validationpath)
    
    # Copying all images from source training folder to the training folder in the working directory
    os.chdir(sourcepathtrain) # Changing path to the source training directory
    name_images = os.listdir()
    
    get2working() # Changing path to the working directory
    for k in range(0,len(name_images)):
        shutil.copy(os.path.join(sourcepathtrain,name_images[k]),trainpath)


# Now that we have split the data set, we can compress and write it. We can also remove the uncompressed folders.

# In[ ]:


# Compressing output folders to zip files
shutil.make_archive('Validation', 'zip',os.getcwd(), validationPathDest)
shutil.make_archive('Testing', 'zip',os.getcwd(), testPathDest)
shutil.make_archive('Training', 'zip',os.getcwd(), trainPathDest)

# Removing uncompressed output folders
shutil.rmtree(validationPathDest)
shutil.rmtree(testPathDest)
shutil.rmtree(trainPathDest);


# In[ ]:


# making sure that the data has been written
os.listdir()


# This will be the last step of this part of the project. At the end we will have a .csv file containg the number of fruits in the data set and three folder containing training, testing and validation dataset. We have also made sure that no data is missing and gone over the basics of navigating directories when working in a kaggle kernel. In the next part of the project we will explore the data further. We will visualize the number of fruits in different classes in different sets. We will also visualize different fruits to see how the data looks. Lastly, we will also calculate some parameters that will be used when we build the convolution neural network.
