#!/usr/bin/env python
# coding: utf-8

# **Version 7**<br>
# Added some simple code that shows how to draw a bounding box.
# 
# **Version 4**<br>
# Added the Synthetic Cell Images dataset. This notebook explains some basic principles that are related to the notebooks in that dataset. It helps to have all notebooks in the same place.

# <hr>

# ## Introduction

# In this kernel we will generate artificial microscope cell images that can be used to then train a machine learning model.
# 
# These are the basic steps:
# 
# 1. Choose a real image
# 2. Crop out a section of the background.
# 3. Crop out three different cells.
# 4. Crop out three examples of noise or artifacts.
# 5. Resize the background crop to 1200 x 1600
# 6. Randomly add a random number of cells and noise to this background, at random locations. The cells and noise will be augmented by adding a random amount of 90 degree counterclockwise rotation. The background will also be randomly rotated.
# 7. Save the newly created artificial image as a png file.
# 

# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np
import os

import cv2

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from skimage.io import imread, imshow
from skimage.transform import resize

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


os.listdir('../input')


# ## Create a dataframe

# In[ ]:


path_malaria = '../input/malaria-bounding-boxes/malaria/malaria/training.json'

df_malaria = pd.read_json(path_malaria)



# get the number of cells in each image
def get_num_cells(x):
    
    num_cells = len(x)
    
    return num_cells

df_malaria['num_cells'] = df_malaria['objects'].apply(get_num_cells)


# drop the objects column
df_malaria = df_malaria.drop('objects', axis=1)

# create new columns

df_malaria['image_id'] = 0
df_malaria['pathname'] = 0
df_malaria['r'] = 0
df_malaria['c'] = 0
df_malaria['channels'] = 0

# df_malaria

for i in range(0,len(df_malaria)):
    
    # get the entry from the 'image' col
    img_dict = df_malaria.loc[i, 'image']
    
    df_malaria.loc[i, 'image_id'] = img_dict['checksum']
    df_malaria.loc[i, 'pathname'] = img_dict['pathname']
    df_malaria.loc[i, 'r'] = img_dict['shape']['r']
    df_malaria.loc[i, 'c'] = img_dict['shape']['c']
    df_malaria.loc[i, 'channels'] = img_dict['shape']['channels']
    
df_malaria = df_malaria.drop('image', axis=1)


# In[ ]:


df_malaria.head()


# <hr>

# ## Some Basic Techniques

# Let's start by reviewing some basic techniques. This will help you understand the code later.

# ### 1. Cropping
# 
# Imagine that we have a matrix. 
# - x and y are the coordinates of a point in the matrix. 
# - h and w are the height and width. 
# 
# Take note that in images the origin (0,0) is located in the top left corner and not in the bottom left corner as in most graphs.

# **This is an example image.**

# In[ ]:


# set the path
pathname = '/images/13099edb-35d9-438f-b093-2cf2ebf9d255.png'
path = '../input/malaria-bounding-boxes/malaria/malaria/' + pathname

# read the image
image = imread(path)

plt.imshow(image)


# In[ ]:





# **To crop out a cell from the image the code is as follows:**

# In[ ]:



y=550
x=720
h=110
w=130

# use numpy slicing to execute the crop
cell_1 = image[y:y+h, x:x+w]

plt.imshow(cell_1)


# In[ ]:





# **This is how to draw a bounding box**

# In[ ]:


# read the image
pathname = '/images/13099edb-35d9-438f-b093-2cf2ebf9d255.png'
path = '../input/malaria-bounding-boxes/malaria/malaria/' + pathname
image1 = imread(path)

# Coordinates of the bounding box
y1=550
x1=720
h=110
w=130

# set the line thickness
stroke = 5
# set the pixel color
color = 0

y2 = y1 + h
x2 = x1 + w

# Draw each line of the box.
# To do this we simply change the pixel color in the image.
image1[y1:y1 + stroke, x1:x2] = color
image1[y2:y2 + stroke, x1:x2] = color
image1[y1:y2, x1:x1 + stroke] = color
image1[y1:y2, x2:x2 + stroke] = color


plt.imshow(image1)


# ### 2. Combining two images

# This is how to combine two images.

# In[ ]:


# image_1

pathname = '/images/89b276cc-cc8f-4378-a877-e01aff333373.png'
path = '../input/malaria-bounding-boxes/malaria/malaria/' + pathname
# read the image
image_1 = imread(path)

# image_2

pathname = '/images/bbf687b5-c6f9-4821-b2e5-a25df1acba47.png'
path = '../input/malaria-bounding-boxes/malaria/malaria/' + pathname
# read the image
image_2 = imread(path)


# Combine the images

new_image = np.minimum(image_1, image_2)


# In[ ]:


# set up the canvas for the subplots
plt.figure(figsize=(30,30))
plt.axis('Off')

# Our subplot will contain 1 rows and 2 columns
# plt.subplot(nrows, ncols, plot_number)

# image_1
plt.subplot(1,3,1)
plt.imshow(image_1)
plt.title('image_1', fontsize=30)
plt.axis('off')


# image_2
plt.subplot(1,3,2)
plt.imshow(image_2)
plt.title('image_2', fontsize=30)
plt.axis('off')

# combined image
plt.subplot(1,3,3)
plt.imshow(new_image)
plt.title('combined image', fontsize=30)
plt.axis('off')


plt.show()


# > Take note that we used **np.minimum()** above. Sometimes depending on the background/foreground  pixel values you may need to use **np.maximum()**.

# ### 3. Numpy transformations

# These are some examples of simple image augmentaions that can be done in numpy.

# In[ ]:


# Transformations

# rotate 90 degrees, counterclockwise
# k specfies how many times to rotate
img_rotate = np.rot90(image_1, k=1)

# flip horizontally
img_horiz = np.fliplr(image_1)

# flip vertically
img_vert = np.flipud(image_1)


# In[ ]:


# set up the canvas for the subplots
plt.figure(figsize=(30,30))
plt.axis('Off')

# Our subplot will contain 1 rows and 4 columns
# plt.subplot(nrows, ncols, plot_number)

# image_1
plt.subplot(1,4,1)
plt.imshow(image_1)
plt.title('original image', fontsize=30)
plt.axis('off')

# image_1
plt.subplot(1,4,2)
plt.imshow(img_rotate)
plt.title('rotated', fontsize=30)
plt.axis('off')


# image_2
plt.subplot(1,4,3)
plt.imshow(img_horiz)
plt.title('flipped horizontally', fontsize=30)
plt.axis('off')

# combined image
plt.subplot(1,4,4)
plt.imshow(img_vert)
plt.title('flipped vertically', fontsize=30)
plt.axis('off')


plt.show()


# ### 4. Generating a random integer

# This generates a random integer between the two specfied numbers. The left number is inclusive. The right is exclusive. This means that here our random numbers will be either 0, 1 or 2.

# In[ ]:


# generate a random integer

num = np.random.randint(0,3)

num


# ### 5. Image Thresholding and Reversing Values
# 
# This is included for future reference. It's not required to create the artificial cell images.

# In[ ]:


# read the image
pathname = '/images/89b276cc-cc8f-4378-a877-e01aff333373.png'
path = '../input/malaria-bounding-boxes/malaria/malaria/' + pathname
# read the image
image = imread(path)

# choose one channel
ch_0 = image[:, :, 0]

# Threshold the values.
# You can try different threshold values. Here we use 175.
thresh_value = 125
thresh_image = (ch_0 >= thresh_value).astype(np.uint8)


# Reverse the values of the thresholded image
# We want background 0 and cell 255. Therefore we reverse the values.
thresh_rev = 255 - thresh_image


# In[ ]:


# set up the canvas for the subplots
plt.figure(figsize=(30,30))
plt.axis('Off')

# Our subplot will contain 1 rows and 3 columns
# plt.subplot(nrows, ncols, plot_number)

# image_1
plt.subplot(1,3,1)
plt.imshow(image)
plt.title('original image', fontsize=30)
plt.axis('off')


# image_2
plt.subplot(1,3,2)
plt.imshow(thresh_image)
plt.title('thresholded image', fontsize=30)
plt.axis('off')

# image_3
plt.subplot(1,3,3)
plt.imshow(thresh_rev)
plt.title('values reversed', fontsize=30)
plt.axis('off')

plt.show()


# ### 6. How to adjust transparency

# This is also included for future reference. It's not required to create the artificial cell images.
# 
# To be able to control transparency we need to add a fourth channel to the image. This would make it an RGBA image (4 channels). An RGB image has only 3 channels. 
# 
# The 4th channel controls opacity. 255 will make a pixel fully opaque, 0 will make a pixel fully transparent. A value in between will make the pixel partly transparent. Therefore, we will create a fourth channel that we will append to the cell image. 
# 
# Here let's make the background transparent and leave the cells opaque.

# In[ ]:


# read the image
pathname = '/images/89b276cc-cc8f-4378-a877-e01aff333373.png'
path = '../input/malaria-bounding-boxes/malaria/malaria/' + pathname
# read the image
image = imread(path)

# choose one channel
ch_0 = image[:, :, 0]

# threshold the values
thresh = (ch_0 >= 125).astype(np.uint8)

# Change the values from 0 and 1 to 0 and 255.
# Here background is 255 and cell is 0.
thresh_opaque = thresh * 255

# We want background 0 and cell 255. Therefore we reverse the values.
thresh_opaque = 255 - thresh_opaque

# add the new channel to the cell so it has 4 channels
image_rgba = np.dstack((image, thresh_opaque))

plt.imshow(image_rgba)

plt.show()


# <hr>

# ## Creating Artificial Cell Images

# ### Create the folder structure

# These are the folders where we will store the image files.

# In[ ]:


# Create a new directory
base_dir = 'base_dir'
os.mkdir(base_dir)


#[CREATE FOLDERS INSIDE THE BASE DIRECTORY]

# now we create 3 folders inside 'base_dir':

    # bground_dir

    # cell_dir

    # noise_dir


# bground_dir
bground_dir = os.path.join(base_dir, 'bground_dir')
os.mkdir(bground_dir)

# cell_dir
cell_dir = os.path.join(base_dir, 'cell_dir')
os.mkdir(cell_dir)

# noise_dir
noise_dir = os.path.join(base_dir, 'noise_dir')
os.mkdir(noise_dir)


# In[ ]:


# check that the folders have been created

os.listdir('base_dir')


# ## Step 1: Choose an Original Image

# This is the image that we will use to create the artificial images.

# In[ ]:


# set the path
pathname = '/images/13099edb-35d9-438f-b093-2cf2ebf9d255.png'
path = '../input/malaria-bounding-boxes/malaria/malaria/' + pathname
# read the image
image = imread(path)

plt.imshow(image)

print(image.shape)


# ## Step 2: Crop out a background, cells and noise

# ### Background

# In[ ]:


# background

fname = 'bground_1.png'

y = 900
x = 600
h = 300
w = 500

bground = image[y:y+h, x:x+w].copy()

plt.imshow(bground)

path = 'base_dir/bground_dir/' + fname
cv2.imwrite(path, bground)


# ### Noise - Create 3 noise images

# In[ ]:


# noise_1

fname = 'noise_1.png'

y=400
x=125
h=70
w=60

noise_1 = image[y:y+h, x:x+w].copy()

plt.imshow(noise_1)

# save the image at the destination as a png file
path = 'base_dir/noise_dir/' + fname
cv2.imwrite(path, noise_1)

print(noise_1.shape)


# In[ ]:


# noise_2

fname = 'noise_2.png'

y=470
x=125
h=60
w=60

noise_2 = image[y:y+h, x:x+w].copy()

plt.imshow(noise_2)

# save the image at the destination as a png file
path = 'base_dir/noise_dir/' + fname
cv2.imwrite(path, noise_2)

print(noise_2.shape)


# In[ ]:


# noise_2

fname = 'noise_3.png'

y=1055
x=1480
h=50
w=60

noise_3 = image[y:y+h, x:x+w].copy()

plt.imshow(noise_3)

# save the image at the destination as a png file
path = 'base_dir/noise_dir/' + fname
cv2.imwrite(path, noise_3)

print(noise_3.shape)


# ### Cell - Create 3 Cell Images

# In[ ]:


# cell_1

fname = 'cell_1.png'

y=550
x=720
h=110
w=130

cell_1 = image[y:y+h, x:x+w]

plt.imshow(cell_1)

# save the image at the destination as a png file
path = 'base_dir/cell_dir/' + fname
cv2.imwrite(path, cell_1)

print(cell_1.shape)


# In[ ]:


# cell_2 

fname = 'cell_2.png'

y=990
x=375
h=115
w=150

cell_2 = image[y:y+h, x:x+w]

plt.imshow(cell_2)

# save the image at the destination as a png file
path = 'base_dir/cell_dir/' + fname
cv2.imwrite(path, cell_2)

print(cell_2.shape)


# In[ ]:


# cell_3

fname = 'cell_3.png'

y=185
x=950
h=110
w=150

cell_3 = image[y:y+h, x:x+w]

plt.imshow(cell_3)

# save the image at the destination as a png file
path = 'base_dir/cell_dir/' + fname
cv2.imwrite(path, cell_3)

print(cell_3.shape)


# In[ ]:





# ## Step 3: Generate the Artificial Images

# In[ ]:


# ==============================

num_images_wanted = 10

min_cells_on_image = 1
max_cells_on_image = 100

# set max x and y to prevent cells from extending outside the bachground image i.e.
# if the get place too close to the edge.
max_x = 1400
max_y = 1000

# ==============================

# Create a new directory

new_images_dir = 'new_images_dir'
os.mkdir(new_images_dir)


# In[ ]:



for i in range(0, num_images_wanted):
    
    
    # randomly choose the number of cells to put in the image
    num_cells_on_image = np.random.randint(min_cells_on_image, max_cells_on_image+1)
    
    # Name the image.
    # The number of cells is included in the file name.
    image_name = 'image_' + str(i) + '_'  + str(num_cells_on_image) + '_.png'
    
    
    # =========================
    # 1. Create the background
    # =========================

    path = 'base_dir/bground_dir/bground_1.png'

    # read the image
    bground_comb = cv2.imread(path)

    # add random rotation to the background
    num_k = np.random.randint(0,3)
    bground_comb = np.rot90(bground_comb, k=num_k)

    # resize the background to match what we want
    bground_comb = cv2.resize(bground_comb, (1600, 1200))
    
    
    # ===============================
    # 2. Add cells to the background
    # ===============================
    
    for j in range(0, num_cells_on_image):
        
        path = 'base_dir/bground_dir/bground_1.png'

        # read the image
        bground = cv2.imread(path)
        # add rotation to the background
        bground = np.rot90(bground, k=num_k)
        # resize the background to match what we want
        bground = cv2.resize(bground, (1600, 1200))
        
        
        # randomly choose a type of cell to add to the image
        cell_type = np.random.randint(1,3+1)
        

        if cell_type == 1:
            
            # cell_1 path
            cell_1 = cv2.imread('base_dir/cell_dir/cell_1.png')

            # add a random rotation to the cell
            cell_1 = np.rot90(cell_1, k=np.random.randint(0,3))
            
            # get the shape after rotation
            shape = cell_1.shape
            
            # get a random x-coord
            y=np.random.randint(0, max_y)
            # get a random y-coord
            x=np.random.randint(0, max_x)
            # set the width and height
            h=shape[0]
            w=shape[1]
            
            # add the cell to the background
            bground[y:y+h, x:x+w] = 0
            bground[y:y+h, x:x+w] = cell_1


        if cell_type == 2:
            
            cell_2 = cv2.imread('base_dir/cell_dir/cell_2.png')

            # add a random rotation to the cell
            cell_2 = np.rot90(cell_2, k=np.random.randint(0,3))

            shape = cell_2.shape

            y=np.random.randint(0,max_y)
            x=np.random.randint(0,max_x)
            h=shape[0]
            w=shape[1]

            bground[y:y+h, x:x+w] = 0
            bground[y:y+h, x:x+w] = cell_2


        if cell_type == 3:
            
            # cell_3

            cell_3 = cv2.imread('base_dir/cell_dir/cell_3.png')

            # add a random rotation to the cell
            cell_3 = np.rot90(cell_3, k=np.random.randint(0,3))

            shape = cell_3.shape

            y=np.random.randint(0,max_y)
            x=np.random.randint(0,max_x)
            h=shape[0]
            w=shape[1]

            bground[y:y+h, x:x+w] = 0
            bground[y:y+h, x:x+w] = cell_3
        
        bground_comb = np.minimum(bground_comb, bground)
    
    
    # =============================================
    # 3. Add noise and artifacts to the background
    # =============================================
    
    # We will only add 3 noise items to each image
    for k in range(0, 3):
        
        path = 'base_dir/bground_dir/bground_1.png'

        # read the image
        bground = cv2.imread(path)
        # add rotation to the background
        bground = np.rot90(bground, k=num_k)
        # resize the background to match what we want
        bground = cv2.resize(bground, (1600, 1200))
        
        
        # randomly choose a type of cell to add to the image
        noise_type = np.random.randint(1,3+1)
        

        if noise_type == 1:
            
            # cell_1 path
            noise_1 = cv2.imread('base_dir/noise_dir/noise_1.png')

            # add a random rotation to the cell
            noise_1 = np.rot90(noise_1, k=np.random.randint(0,3))
            
            # get the shape after rotation
            shape = noise_1.shape
            
            # get a random x-coord
            y=np.random.randint(0, max_y)
            # get a random y-coord
            x=np.random.randint(0, max_x)
            # set the width and height
            h=shape[0]
            w=shape[1]
            
            # add the cell to the background
            bground[y:y+h, x:x+w] = 0
            bground[y:y+h, x:x+w] = noise_1


        if noise_type == 2:
            
            noise_2 = cv2.imread('base_dir/noise_dir/noise_2.png')

            # add a random rotation to the cell
            noise_2 = np.rot90(noise_2, k=np.random.randint(0,3))

            shape = noise_2.shape

            y=np.random.randint(0,max_y)
            x=np.random.randint(0,max_x)
            h=shape[0]
            w=shape[1]

            bground[y:y+h, x:x+w] = 0
            bground[y:y+h, x:x+w] = noise_2


        if noise_type == 3:
            
            # noise_3

            noise_3 = cv2.imread('base_dir/noise_dir/noise_3.png')

            # add a random rotation to the cell
            noise_3 = np.rot90(noise_3, k=np.random.randint(0,3))

            shape = noise_3.shape

            y=np.random.randint(0,max_y)
            x=np.random.randint(0,max_x)
            h=shape[0]
            w=shape[1]

            bground[y:y+h, x:x+w] = 0
            bground[y:y+h, x:x+w] = noise_3
        
        bground_comb = np.minimum(bground_comb, bground)
    
    
    
    # ===============================
    # 3. Save the image
    # ===============================

    path = 'new_images_dir/' + image_name
    cv2.imwrite(path, bground_comb)
    
print('Num imgaes created: ', num_images_wanted)


# In[ ]:


# check that the artificial images exist

os.listdir('new_images_dir')


# ### Display the artificial images

# In[ ]:


image_list = os.listdir('new_images_dir')


# set up the canvas for the subplots
plt.figure(figsize=(10,10))
plt.axis('Off')

# Our subplot will contain 3 rows and 3 columns
# plt.subplot(nrows, ncols, plot_number)


# == row 1 ==

# image
plt.subplot(3,3,1)

fname = image_list[1]
path = 'new_images_dir/' + fname
name = fname.split('_')
num_cells = 'Num Cells: ' + str(name[2])

image = cv2.imread(path)
plt.imshow(image)
plt.title(num_cells, fontsize=14)
plt.axis('off')

# image
plt.subplot(3,3,2)

fname = image_list[2]
path = 'new_images_dir/' + fname
name = fname.split('_')
num_cells = 'Num Cells: ' + str(name[2])

image = cv2.imread(path)
plt.imshow(image)
plt.title(num_cells, fontsize=14)
plt.axis('off')


# image
plt.subplot(3,3,3)

fname = image_list[3]
path = 'new_images_dir/' + fname
name = fname.split('_')
num_cells = 'Num Cells: ' + str(name[2])

image = cv2.imread(path)
plt.imshow(image)
plt.title(num_cells, fontsize=14)
plt.axis('off')

# == row 2 ==

# image
plt.subplot(3,3,4)

fname = image_list[4]
path = 'new_images_dir/' + fname
name = fname.split('_')
num_cells = 'Num Cells: ' + str(name[2])

image = cv2.imread(path)
plt.imshow(image)
plt.title(num_cells, fontsize=14)
plt.axis('off')

# image
plt.subplot(3,3,5)

fname = image_list[5]
path = 'new_images_dir/' + fname
name = fname.split('_')
num_cells = 'Num Cells: ' + str(name[2])

image = cv2.imread(path)
plt.imshow(image)
plt.title(num_cells, fontsize=14)
plt.axis('off')

# image
plt.subplot(3,3,6)

fname = image_list[6]
path = 'new_images_dir/' + fname
name = fname.split('_')
num_cells = 'Num Cells: ' + str(name[2])

image = cv2.imread(path)
plt.imshow(image)
plt.title(num_cells, fontsize=14)
plt.axis('off')

# == row 3 ==

# image
plt.subplot(3,3,7)

fname = image_list[7]
path = 'new_images_dir/' + fname
name = fname.split('_')
num_cells = 'Num Cells: ' + str(name[2])

image = cv2.imread(path)
plt.imshow(image)
plt.title(num_cells, fontsize=14)
plt.axis('off')

# image
plt.subplot(3,3,8)

fname = image_list[8]
path = 'new_images_dir/' + fname
name = fname.split('_')
num_cells = 'Num Cells: ' + str(name[2])

image = cv2.imread(path)
plt.imshow(image)
plt.title(num_cells, fontsize=14)
plt.axis('off')


# image
plt.subplot(3,3,9)

fname = image_list[9]
path = 'new_images_dir/' + fname
name = fname.split('_')
num_cells = 'Num Cells: ' + str(name[2])

image = cv2.imread(path)
plt.imshow(image)
plt.title(num_cells, fontsize=14)
plt.axis('off')


plt.tight_layout()
plt.show()


# In[ ]:





# > **Being able to generate artificial data is cool. But when combined with a python generator this ability becomes a super-power.**

# <hr>

# ## Using an Infinite Data Generator

# Let's say we wanted to train a model where X_train is the image and y_train is the number of cells on that image. In the above setup we are storing files in folders. We will eventually run out of space - both disk space and RAM. But if we use a python generator to feed data into the model then we will never run out of space.
# 
# 
# 
# 

# ### What is a generator?
# 
# This is a simple example of a generator. Unlike a normal function, the ouput from a generator does not stay in memory permanently. Each time a generator is run, the previous output is overwritten in memory. Therefore, it can be used to handle large amounts of image data when only a limited amount of memory is available.

# In[ ]:


def my_generator():

    while True: 
        
        for i in range(0,4):
            
            my_number = i

            yield my_number
        
        
# initialize the generator
infinity_gen = my_generator()


# Each time you run the cell below you will see an ouput. The generator runs 4 times because the range is set as range(0,4). On the fifth run i becomes 0 again, i.e. the generator goes back to the beginning.

# In[ ]:


# Run the generator
out_put = next(infinity_gen)

out_put


# In the above example the generator outputs a variable called my_number, but this output could be anything including a matrix. Let's build a generator that creates one artificial image and outputs it along with the number of cells that are on that image.

# In[ ]:


# Set up the generator

def image_generator():
    
    while True:
        
        min_cells_on_image = 1
        max_cells_on_image = 100

        # set max x and y to prevent cells from extending outside the bachground image i.e.
        # if the get place too close to the edge.
        max_x = 1400
        max_y = 1000
        
        for i in range(0,1000):
            
            # randomly choose the number of cells to put in the image
            num_cells_on_image = np.random.randint(min_cells_on_image, max_cells_on_image+1)


            # =========================
            # 1. Create the background
            # =========================

            path = 'base_dir/bground_dir/bground_1.png'

            # read the image
            bground_comb = cv2.imread(path)

            # add random rotation to the background
            num_k = np.random.randint(0,3)
            bground_comb = np.rot90(bground_comb, k=num_k)

            # resize the background to match what we want
            bground_comb = cv2.resize(bground_comb, (1600, 1200))


            # ===============================
            # 2. Add cells to the background
            # ===============================

            for j in range(0, num_cells_on_image):

                path = 'base_dir/bground_dir/bground_1.png'

                # read the image
                bground = cv2.imread(path)
                # add rotation to the background
                bground = np.rot90(bground, k=num_k)
                # resize the background to match what we want
                bground = cv2.resize(bground, (1600, 1200))


                # randomly choose a type of cell to add to the image
                cell_type = np.random.randint(1,3+1)


                if cell_type == 1:

                    # cell_1 path
                    cell_1 = cv2.imread('base_dir/cell_dir/cell_1.png')

                    # add a random rotation to the cell
                    cell_1 = np.rot90(cell_1, k=np.random.randint(0,3))

                    # get the shape after rotation
                    shape = cell_1.shape

                    # get a random x-coord
                    y=np.random.randint(0, max_y)
                    # get a random y-coord
                    x=np.random.randint(0, max_x)
                    # set the width and height
                    h=shape[0]
                    w=shape[1]

                    # add the cell to the background
                    bground[y:y+h, x:x+w] = 0
                    bground[y:y+h, x:x+w] = cell_1


                if cell_type == 2:

                    cell_2 = cv2.imread('base_dir/cell_dir/cell_2.png')

                    # add a random rotation to the cell
                    cell_2 = np.rot90(cell_2, k=np.random.randint(0,3))

                    shape = cell_2.shape

                    y=np.random.randint(0,max_y)
                    x=np.random.randint(0,max_x)
                    h=shape[0]
                    w=shape[1]

                    bground[y:y+h, x:x+w] = 0
                    bground[y:y+h, x:x+w] = cell_2


                if cell_type == 3:

                    # cell_3

                    cell_3 = cv2.imread('base_dir/cell_dir/cell_3.png')

                    # add a random rotation to the cell
                    cell_3 = np.rot90(cell_3, k=np.random.randint(0,3))

                    shape = cell_3.shape

                    y=np.random.randint(0,max_y)
                    x=np.random.randint(0,max_x)
                    h=shape[0]
                    w=shape[1]

                    bground[y:y+h, x:x+w] = 0
                    bground[y:y+h, x:x+w] = cell_3

                bground_comb = np.minimum(bground_comb, bground)


            # =============================================
            # 3. Add noise and artifacts to the background
            # =============================================

            # We will only add 3 noise items to each image
            for k in range(0, 3):

                path = 'base_dir/bground_dir/bground_1.png'

                # read the image
                bground = cv2.imread(path)
                # add rotation to the background
                bground = np.rot90(bground, k=num_k)
                # resize the background to match what we want
                bground = cv2.resize(bground, (1600, 1200))


                # randomly choose a type of cell to add to the image
                noise_type = np.random.randint(1,3+1)


                if noise_type == 1:

                    # cell_1 path
                    noise_1 = cv2.imread('base_dir/noise_dir/noise_1.png')

                    # add a random rotation to the cell
                    noise_1 = np.rot90(noise_1, k=np.random.randint(0,3))

                    # get the shape after rotation
                    shape = noise_1.shape

                    # get a random x-coord
                    y=np.random.randint(0, max_y)
                    # get a random y-coord
                    x=np.random.randint(0, max_x)
                    # set the width and height
                    h=shape[0]
                    w=shape[1]

                    # add the cell to the background
                    bground[y:y+h, x:x+w] = 0
                    bground[y:y+h, x:x+w] = noise_1


                if noise_type == 2:

                    noise_2 = cv2.imread('base_dir/noise_dir/noise_2.png')

                    # add a random rotation to the cell
                    noise_2 = np.rot90(noise_2, k=np.random.randint(0,3))

                    shape = noise_2.shape

                    y=np.random.randint(0,max_y)
                    x=np.random.randint(0,max_x)
                    h=shape[0]
                    w=shape[1]

                    bground[y:y+h, x:x+w] = 0
                    bground[y:y+h, x:x+w] = noise_2


                if noise_type == 3:


                    noise_3 = cv2.imread('base_dir/noise_dir/noise_3.png')

                    # add a random rotation to the cell
                    noise_3 = np.rot90(noise_3, k=np.random.randint(0,3))

                    shape = noise_3.shape

                    y=np.random.randint(0,max_y)
                    x=np.random.randint(0,max_x)
                    h=shape[0]
                    w=shape[1]

                    bground[y:y+h, x:x+w] = 0
                    bground[y:y+h, x:x+w] = noise_3

                bground_comb = np.minimum(bground_comb, bground)

            yield (bground_comb, num_cells_on_image)
            
        
        


# In[ ]:


# initialize the generator
image_gen = image_generator()


# In[ ]:


# Each time you run this cell you will see a different image.

# use tuple unpacking
image, num_cells = next(image_gen)

print('Num cells:', num_cells)

plt.imshow(image)

plt.show()


# This generator could be used to feed data into a Keras model. This kernel explains how to do that:<br>
# https://www.kaggle.com/vbookshelf/python-generators-to-reduce-ram-usage-part-2

# In[ ]:





# ## Helpful Resources

# - Image processing with scikit-image<br>
# https://www.kaggle.com/ksaaskil/image-processing-with-scikit-image
# 
# - Image manipulation and processing using Numpy and Scipy<br>
# http://scipy-lectures.org/advanced/image_processing/
# 
# - Python Generators to reduce RAM usage<br>
# https://www.kaggle.com/vbookshelf/python-generators-to-reduce-ram-usage-part-2
# 
# - Simple Cell Segmentation with Keras and U-Net<br>
# https://www.kaggle.com/vbookshelf/simple-cell-segmentation-with-keras-and-u-net

# ## Conclusion

# This was just a simple example of how to create artificial images. Once you understand the workflow you can modify the code to include:
# 
# - more backgrounds
# - more cell types
# - more kinds of noise and artifacts
# - more image augmentation
# 
# One thing you'll notice is that the cells on the artificial images do not extend beyond the image boundary. But the cells on the real images do extend beyound the edge, and are cut off at the edges. You could modify the way the artificial images are created to include this. You could also set up the generator to ouput batches of images.
# 
# Thank you for reading.
