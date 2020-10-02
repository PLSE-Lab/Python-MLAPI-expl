#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')


# ### The goal of the competition is to develop an algorithm to estimate the absolute pose of vehicles (6 degrees of freedom) from a single image in a real-world traffic environment
# 
# **From the problem description:**
# - The primary data is images of cars and related `pose` information. The pose information is formatted as strings, as follows:
# `model type, yaw, pitch, roll, x, y, z`
# 
# - A concrete example with two cars in the photo:
# `5 0.5 0.5 0.5 0.0 0.0 0.0 32 0.25 0.25 0.25 0.5 0.4 0.7`
# 
# - Submissions (per sample_submission.csv) are very similar, with the addition of a confidence score, and the removal of the model type. You are not required to predict the model type of the vehicle in question.
#    
#   `ID, PredictionString`
#   `ID_1d7bc9b31,0.5 0.5 0.5 0.0 0.0 0.0 1.0` indicating that this prediction has a confidence score of 1.0.
# 
# <font color='red'>** So, the goal is to predict one or more sets of **:</font> `yaw, pitch, roll, x, y, z, confidence` <font color='red'>**for each picture**:</font>

# ## Yaw / Pitch / Roll..what ? 
# 
# Let us first try to understand what the 6 degrees of freedom mean. 
# 
# - To completely specify the position of a 3-D object, we need to specify how it is rotated with respect to X/Y/Z axis, in addition to the the position a reference point (say center of the object) in the object. 
# - As illustrated in the figure below, roll/pitch/yaw correspond to rotation of an object around the X/Y/Z axis respectively. [Pic Credit Link](https://devforum.roblox.com/t/take-out-pitch-from-rotation-matrix-while-preserving-yaw-and-roll/95204) 

# ![title](https://camo.githubusercontent.com/2d9fda441f1b838bc7e682ca1f3a4f7ab46c9e53/687474703a2f2f646f632e616c6465626172616e2e636f6d2f322d312f5f696d616765732f726f6c6c50697463685961772e706e67)

# Another good picture for pitch / yaw / roll in the context of the cars is shown below from [this reference](https://carsexplained.wordpress.com/2017/02/21/fundamentals-of-car-science-pitch-and-roll/): 
# 
# ![](https://carsexplained.files.wordpress.com/2017/01/post1-2figure2.jpg)

# A good visualization app for Yaw / Pitch / Roll can be found here: http://www.ctralie.com/Teaching/COMPSCI290/Materials/EulerAnglesViz/

# ## Reading the Files
# 

# In[ ]:


# Read the test and train data sets. 
df_train = pd.read_csv('../input/pku-autonomous-driving/train.csv')
print("Shapes of training dataset:");print(df_train.shape)
print("Training Data Sample");display(df_train.head())


# Since the prediction string may contain more than one car, let us try and convert the data to have one row for each car. So, one image can have multiple rows of data. 
# 
# ## Converting the training predictions to one row per car (multiple rows per image)

# In[ ]:


# Number of cars in each picture can be calculated using number of spaces in prediction string
df_train['NumCars'] = [int((x.count(' ')+1)/7) for x in df_train['PredictionString']]
df_train.head()


# ### Now let us expand the dataframe so that we can separate out all the 6 degrees of freedom for the positions. 

# In[ ]:


image_id_expanded = [item for item, count in 
                     zip(df_train['ImageId'], df_train['NumCars']) for i in range(count)]
prediction_strings_expanded = df_train['PredictionString'].str.split(' ',expand = True).values.reshape(-1,7).astype(float)
prediction_strings_expanded = prediction_strings_expanded[~np.isnan(prediction_strings_expanded).all(axis=1)]
df_train_expanded = pd.DataFrame(
    {
        'ImageId': image_id_expanded,
        'model_type': prediction_strings_expanded[:,0].astype(int),
        'yaw':prediction_strings_expanded[:,1],
        'pitch':prediction_strings_expanded[:,2],
        'roll':prediction_strings_expanded[:,3],
        'x':prediction_strings_expanded[:,4],
        'y':prediction_strings_expanded[:,5],
        'z':prediction_strings_expanded[:,6]
    })
print("Shapes of exapanded training dataset:");print(df_train_expanded.shape)
print("Expanded Training Data Sample");display(df_train_expanded.head(10))


# ### Let us take a look at the training data column distributions and any possible correlations. 
# 
# The following observations can be made from the plot below:
# 
# 1. Position related columns (yaw / pitch / roll / x / y / z) are not correlated with each other as expected. 
# 2. X / Y / Z values are most concentrated near 0 (close to the camera), while there are a few points that are very far from the camera. 
# 3. Roll (rotation around X) and Pitch (roation around Y) are distributed between $\pi$ and -$\pi$ (Complete 360 degree rotations allowed around X and Y). 
# 4. Yaw (rotation around Z) seem to be more contrained, which makes sense as cars cannot be completely flipped up-side down on the road side :)

# As illustrated in the figure below

# In[ ]:


df_train_expanded.describe()


# In[ ]:


sns.pairplot(df_train_expanded)


# ## Exploring the images
# 
# Let us start looking at the images, starting with the first image (ID_8a6e65317). From above tables, it appears it has 5 cars that are of interest in it.

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
plt.imshow(plt.imread('../input/pku-autonomous-driving/train_images/ID_8a6e65317.jpg'))


# From the above plot, we can see that there is one car facing our car / camer (white car on the left side), two cars to the right that are perpendicular to the direction of the road, and one car (red) ahead that is in the same direction as our car. Let us see what the training data says.  

# In[ ]:


df_train_expanded.head(6)


# ### Let us take a look at a second picture, that also has a mask associated with it to mark out the cars that are very far away. 
# 
# From the picture below, we can see 3 cars, which matches the count in the tabe below extracted for that image id (the bus, the car far ahead in our lane, and the another car far away on the opposite side of the road). 
# 

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
plt.imshow(plt.imread('../input/pku-autonomous-driving/train_images/ID_7c4a3e0aa.jpg'))

# mask with a transparaent overlay
plt.imshow(plt.imread('../input/pku-autonomous-driving/train_masks/ID_7c4a3e0aa.jpg'), alpha = 0.5)
plt.show()


# In[ ]:


df_train_expanded[df_train_expanded['ImageId'] == 'ID_7c4a3e0aa']


# ### Guessing the X/Y/Z directions from the pictures
# It should be possibel to guess the X/Y/Z directions from the data. 
# 
# My guess at this point is : 
# 
# - X direction approximately corresponds to the direction perpendicular to the road from left to right. Left side of the camera is negative X and right side of the camera is likely positive X. 
# - Z direction seems to correspond to the direction of the road
# - Y direction seems to correspond to the vertical direction (perpendicular to the plane of the road)

# ### Exploring more cases with only one car is tagged in the picture

# In[ ]:


df_train[df_train['NumCars'] == 1]


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
plt.imshow(plt.imread('../input/pku-autonomous-driving/train_images/ID_6dd7f07a5.jpg'))

# mask with a transparaent overlay
plt.imshow(plt.imread('../input/pku-autonomous-driving/train_masks/ID_6dd7f07a5.jpg'), alpha = 0.4)
plt.show()


# In[ ]:


df_train_expanded[df_train_expanded['ImageId'] == 'ID_6dd7f07a5']


# ## Blind-guess submission

# #### Just for kicks let us do a blind guess submission, and see where it takes us. 
# 
# Here is the logic for the submission:
# 
# * From the previous analysis, it looks like: 
#     * median amount of cars in each picture is about 11, 
#     * pitch is taking values that are either -pi or 0 or + pi, and 
#     * roll is taking mostly -pi and +pi. 
# 
# * So, we generate a string that has most probable combinations of pitch and roll (3x2 = 6 combinations), and assign median values for all the other variables like yaw, x, y, z and assign a randomly chosen confidence factor of 0.8
# 
# * With this we generate a submission file. 
# 
# * <font color='red'>** Since no one seems to have submitted any valid solution at all, this blind guess is currently leading the leader board (as of 10/24/2019) !! Yaay !!**</font>

# In[ ]:


pitch_vals  = [-3.14, 0, 3.14 ]
roll_vals = [-3.14, 3.14]

blind_guess_string = ''
for r in itertools.product(pitch_vals, roll_vals): 
    blind_guess_string = blind_guess_string + (str(df_train_expanded.yaw.median()) + ' '+ 
                                                         str(r[0]) + ' '+ str(r[1]) +  ' '+ 
                                                         str(df_train_expanded.x.median()) + ' '+ 
                                                         str(df_train_expanded.y.median()) + ' '+ 
                                                         str(df_train_expanded.z.median()) + ' 0.8 '
                                                        )
print(blind_guess_string)


# In[ ]:


df_submission = pd.read_csv('../input/pku-autonomous-driving/sample_submission.csv')
df_submission['PredictionString'] = blind_guess_string
df_submission.to_csv('blind_guess_submission.csv',index=False)

