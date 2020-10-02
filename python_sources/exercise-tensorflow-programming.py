#!/usr/bin/env python
# coding: utf-8

# **[Deep Learning Course Home Page](https://www.kaggle.com/learn/deep-learning)**
# 
# ---
# 

# # Intro
# The TV show *Silicon Valley* had an app called "See Food" that promised to identify food. 
# 
# In this notebook, you will write code using and comparing pre-trained models to choose one as an engine for the See Food app.
# 
# You won't go too deep into Keras or TensorFlow details in this particular exercise.  Don't worry. You'll go deeper into model development soon.  For now, you'll make sure you know how to use pre-trained models.
# 
# # Set-Up

# We will run a few steps of environmental set-up before writing your own code. **You don't need to understand the details of this set-up code.** You can just run each code cell until you get to the exercises.
# 
# ### 1) Create Image Paths
# This workspace includes image files you will use to test your models. Run the cell below to store a few filepaths to these images in a variable `img_paths`.

# In[ ]:


import os
from os.path import join


hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/hot_dog'

hot_dog_paths = [join(hot_dog_image_dir,filename) for filename in 
                            ['1000288.jpg',
                             '127117.jpg']]

not_hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/not_hot_dog'
not_hot_dog_paths = [join(not_hot_dog_image_dir, filename) for filename in
                            ['823536.jpg',
                             '99890.jpg']]

img_paths = hot_dog_paths + not_hot_dog_paths


# ### 2) Run an Example Model
# Here is the code you saw in the tutorial. It loads data, loads a pre-trained model, and makes predictions. Run this cell too.

# In[ ]:


from IPython.display import Image, display
from learntools.deep_learning.decode_predictions import decode_predictions
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array


image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)


my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)

most_likely_labels = decode_predictions(preds, top=3)


# ### 3) Visualize Predictions

# In[ ]:


for i, img_path in enumerate(img_paths):
    display(Image(img_path))
    print(most_likely_labels[i])


# ### 4) Set Up Code Checking
# As a last step before writing your own code, run the following cell to enable feedback on your code.

# In[ ]:


# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning.exercise_3 import *
print("Setup Complete")


# # Exercises
# 
# You will write a couple useful functions in the next exercises. Then you will put these functions together to compare the effectiveness of various pretrained models for your hot-dog detection program.
# 
# ### Exercise 1
# 
# We want to distinguish whether an image is a hot dog or not. But our models classify pictures into 1000 different categories. Write a function that takes the models predictions (in the same format as `preds` from the set-up code) and returns a list of `True` and `False` values.
# 
# Some tips:
# - Work iteratively. Figure out one line at a time outsie the function, and print that line's output to make sure it's right. Once you have all the code you need, move it into the function `is_hot_dog`. If you get an error, check that you have copied the right code and haven't left anything out.
# - The raw data we loaded in `img_paths` had two images of hot dogs, followed by two images of other foods. So, if you run your function on `preds`, which represents the output of the model on these images, your function should return `[True, True, False, False]`.
# - You will want to use the `decode_predictions` function that was also used in the code provided above. We provided a line with this in the code cell to get you started.
# 
# 
# 

# In[ ]:


# Experiment with code outside the function, then move it into the function once you think it is right

# the following lines are given as a hint to get you started
decoded = decode_predictions(preds, top=1)
# print(decoded[0][0][1])
# print(type(decoded[0][0][1]))

def is_hot_dog(preds):
    '''
    inputs:
    preds_array:  array of predictions from pre-trained model

    outputs:
    is_hot_dog_list: a list indicating which predictions show hotdog as the most likely label
    '''
    decoded_labels = decode_predictions(preds)
    output = [True if prediction[0][1] == 'hotdog' else False for prediction in decoded_labels]
    return output
q_1.check()


# If you'd like to see a hint or the solution, uncomment the appropriate line below.
# 
# **If you did not get a working solution, copy the solution code into your code cell above and run it. You will need this function for the next step.**

# In[ ]:


# q_1.hint()
# q_1.solution()


# ### Exercise 2: Evaluate Model Accuracy
# 
# You have a model (called `my_model`). Is it good enough to build your app around? 
# 
# Find out by writing a function that calculates a model's accuracy (fraction correct). You will try an alternative model in the next step. So we will put this logic in a reusable function that takes data and the model as arguments, and returns the accuracy.
# 
# Tips:
# 
#  - Use the `is_hot_dog` function from above to help write your function
#  - To save you some scrolling, here is the code from above where we used a TensorFlow model to make predictions:
# 
# ```
# my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
# test_data = read_and_prep_images(img_paths)
# preds = my_model.predict(test_data)
# ```

# In[ ]:


def calc_accuracy(model, paths_to_hotdog_images, paths_to_other_images):
    test_data_hotdog = read_and_prep_images(paths_to_hotdog_images)
    test_data_other = read_and_prep_images(paths_to_other_images)
    predictions_hotdog = model.predict(test_data_hotdog)
    predictions_other = model.predict(test_data_other)
    
    preds_hotdog = is_hot_dog(predictions_hotdog)
    preds_other = is_hot_dog(predictions_other)
    
    correct_count = 0
    for pred in preds_hotdog:
        if pred:
            correct_count += 1
    
    for pred in preds_other:
        if not pred:
            correct_count += 1
    
    return correct_count/(len(preds_hotdog) + len(preds_other))
            
# Code to call calc_accuracy.  my_model, hot_dog_paths and not_hot_dog_paths were created in the setup code
my_model_accuracy = calc_accuracy(my_model, hot_dog_paths, not_hot_dog_paths)
print("Fraction correct in small test set: {}".format(my_model_accuracy))

# checks that your function calc_accuracy works correctly
q_2.check()


# If you'd like a hint or the solution, uncomment the appropriate line below

# In[ ]:


#q_2.hint()
# q_2.solution()


# ### Exercise 3:
# There are other models besides the ResNet model (which we have loaded). For example, an earlier winner of the ImageNet competition is the VGG16 model.  Don't worry about the differences between these models yet. We'll come back to that later. For now, just focus on the mechanics of applying these models to a problem.
# 
# The code used to load a pretrained ResNet50 model was
# 
# ```
# my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
# ```
# 
# The weights for the model are stored at `../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5`.
# 
# In the cell below, create a VGG16 model with the preloaded weights. Then use your `calc_accuracy` function to determine what fraction of images the VGG16 model correctly classifies.  Is it better or worse than the pretrained ResNet model?

# In[ ]:


# import the model
from tensorflow.keras.applications import VGG16


vgg16_model = VGG16(weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
# calculate accuracy on small dataset as a test
vgg16_accuracy = calc_accuracy(vgg16_model,hot_dog_paths, not_hot_dog_paths)

print("Fraction correct in small dataset: {}".format(vgg16_accuracy))
q_3.check()


# Uncomment the appropriate line below if you'd like a hint or the solution

# In[ ]:


#q_3.hint()
#q_3.solution()


# If this model is used for an app that runs on a phone, what factors besides accuracy might you care about? After you've thought about it, keep going below.

# # Keep Going
# You are ready for **[Transfer Learning](https://www.kaggle.com/dansbecker/transfer-learning/)**, which will allow you to apply the same level of power for your custom purposes.
# 

# ---
# **[Deep Learning Course Home Page](https://www.kaggle.com/learn/deep-learning)**
# 
# 

# In[ ]:




