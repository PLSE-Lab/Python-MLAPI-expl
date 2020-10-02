#!/usr/bin/env python
# coding: utf-8

# # Import the necessary libraries and load the data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob #deal with paths
import os #deal with paths
import cv2 #deal with images
from skimage.io import imread #read images from files
import matplotlib.pyplot as plt #make plots
import seaborn as sb #pretty plots :P 
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the data and explore a few images
# 
# This step was done using [Kevin's notebook](https://www.kaggle.com/kmader/histology-mnist-loading-and-processing-data) methods.

# In[ ]:


base_tile_dir = '../input/kather_texture_2016_image_tiles_5000/'
df = pd.DataFrame({'path': glob(os.path.join(base_tile_dir, '*', '*', '*.tif'))}) 
df['cell_type'] = df['path'].map(lambda x: os.path.basename(os.path.dirname(x)).split('_')[1])
df['cell_type_idx'] = df['path'].map(lambda x: os.path.basename(os.path.dirname(x)).split('_')[0])
df['image_name'] = df['path'].map(lambda x: os.path.basename(x).split('_Row')[0])
df['image_row'] = df['path'].map(lambda x: os.path.basename(x).split('_Row')[1].split('_')[1])
df['image_col'] = df['path'].map(lambda x: os.path.basename(x).split('_Row')[1].split('_')[3].split('.')[0])
df['image'] = df['path'].map(imread)
df.sample(5)


# Now all our images were read and are displayed as numpy arrays. For those unfamiliar with the concept, each pixel in the image is represented by a 3-sized array, where each element represents the intensity of [red, green, blue] colors, ranging from 0 to 255. Our dataframe is already prepared for us to classify an `image` into one of the `cell_type`. Let's load an image for each cell_type to see how they look like. 

# In[ ]:


from random import randint

def get_first_cell_images(df):
    #get unique cell types
    unique_cell_type = df.cell_type.unique()
    cell_images = []
    for cell in unique_cell_type:
        #get the first row containing an image of cell_type == cell
        first_img_idx= np.where(df.cell_type == cell) 
        #list containing one image of each type
        cell_images.append((df[df.cell_type == cell].loc[first_img_idx[0][0],'image'], cell))
    return cell_images

images = get_first_cell_images(df)

#create the subplots
fig, m_axs = plt.subplots(1, len(images), figsize = (20, 2))
#show the images and label them
for ii, c_ax in enumerate(m_axs):
    c_ax.imshow(images[ii][0])
    c_ax.set_title(images[ii][1])


# We can see here the 7 existing cell types in the dataset. From what I can see, stroma and complex cells are very similar (I'm no hematologist). and perhaps our predictor will be more confused when looking at these two types.
# 
# Here I'll be using Keras (tensorflow backend) to create a Convolutional Neural Network (CNN) that I hope will be capable of discriminating these cell types well. There are lots of tutorial on the web in this topic, such as [this one](http://cs231n.github.io/convolutional-networks/) and [this one](https://medium.freecodecamp.org/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050), so I won't go deep into the theory behind it here. Below I'll just  point out a few relevant aspects about CNNs so we can follow up the reasoning and build a basic one.
# ![Conv](http://adeshpande3.github.io/assets/Cover.png)
# *[Image source](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/)*
# 
# Basically, to build a simple CNN we use three kind of layers: convolutional, pooling and fully-connected. The convolutional layer will apply a filter (also called a kernel) to map the input image into another kind of image in the new layer, as depicted in the image below:
# ![Conv2](https://cdn-images-1.medium.com/max/800/1*EuSjHyyDRPAQUdKCKLTgIQ.png)
# *[Image source](https://medium.freecodecamp.org/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050)*
# 
# Here, the numbers in the filter are the weights our algorithm will try to predict in order to get the best results possible. You may notice that all the nine elements from the input image were mapped into a single one in the new layer. This operation of multiplying the filter elements by a few elements of an image is what give the name to this kind of network - roughly saying, this is what a convolution is. For a more mathematical definition, you can check [Ian Goodfellow's book - chapter 9](https://www.deeplearningbook.org/contents/convnets.html).
# 
# In the second layer type we do what is called 'pooling', which can be commonly seen following every one or two convolutional layers. This is done to reduce dimensionality, so the number of parameters used to train the network is reduced. As a direct effect, the training time and overfitting issues are reduced as well. Below we can see a maxpooling layer. This kind of pooling takes the highest number in a given kernel size (the size of the colored squares on the input image below) and outputs it to the next layer.
# 
# ![maxpool](https://cdn-images-1.medium.com/max/800/1*vbfPq-HvBCkAcZhiSTZybg.png)
# *[Image source](https://medium.freecodecamp.org/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050)*
# 
# After using a given number of convolutional + pooling layers, we can feed the output into a fully-connected layer, which is the usual layer in a [Multilayer Perceptron (MLP)](https://machinelearningmastery.com/neural-networks-crash-course/). 
# 
# For our problem is a multiclass classification, we'll use [softmax](https://www.quora.com/What-is-the-intuition-behind-SoftMax-function) as the final activation function, that will give the probability of the image belonging to each of the classes (cell types).
# 
# First, let's create a 4-rank tensor (see it as a 4-dimensional array). I tried to feed the Keras model directly with `df.images` but I couldn't reshape it into a 4-d input. If anyone has an idea how to do it, I'd be glad to get feedback :) 
# 
# So, the functions below just get the path to the image and load it into the shape (150, 150,3) using the `image` package from `keras.preprocessing`.

# In[ ]:


from keras.preprocessing import image                  
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True     
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(150,150))
    # convert PIL.Image.Image type to 3D tensor with shape (150, 150, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 150, 150, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

images_path = list(df.path)

images_tensors = paths_to_tensor(images_path)

images_tensors.shape


# In[ ]:


df.cell_type.unique()


# Now I will use `LabelBinarizer` to make our outputs either 0 or 1. For instance, our outputs can be ['STROMA', 'DEBRIS', 'ADIPOSE', 'MUCOSA', 'EMPTY', 'TUMOR', 'LYMPHO', 'COMPLEX']. So let's say a cell is 'MUCOSA'. Then, after using LabelBinarizer, the label would be [0,0,0,1,0,0,0,0]. When we run our model, it will give probabilities like [0, 0.2, 0.1,0.7, 0.0.0.0], so the cell could be identified as mucosa with 70% of certainty. 
# 
# I also split the dataset into training, validation and test tensors.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils


train_fraction = 0.8

encoder = LabelBinarizer()
y = encoder.fit_transform(df.cell_type)
x = images_tensors

train_tensors, test_tensors, train_targets, test_targets =    train_test_split(x, y, train_size = train_fraction, random_state = 42)

val_size = int(0.5*len(test_tensors))

val_tensors = test_tensors[:val_size]
val_targets = test_targets[:val_size]
test_tensors = test_tensors[val_size:]
test_targets = test_targets[val_size:]


# Now I'll run the CNN itself. The architecture will be 3 Conv2D -> Dropout -> Maxpooling 4 times, followed by a global maxpooling. At the end, I run the outputs through a fully connected layer to get the probabilities. 

# In[ ]:


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from tensorflow import set_random_seed

set_random_seed(42)

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 8)
checkpointer = ModelCheckpoint(filepath='weights.hdf5', 
                               verbose=1, save_best_only=True)
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (150, 150, 3)))
model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = 3)) 

model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu')) 
model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu')) 
model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = 3)) 

model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = 3))

model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = 3))
model.add(GlobalMaxPooling2D())
model.add(Dense(8, activation = 'softmax'))

model.summary()


# In[ ]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 50
model.fit(train_tensors, train_targets, 
          validation_data=(val_tensors, val_targets),
          epochs=epochs, batch_size=20, verbose=1, callbacks = [early_stopping, checkpointer])


# In[ ]:


model.load_weights('weights.hdf5')

cell_predictions =  [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

test_accuracy = 100*np.sum(np.array(cell_predictions)==np.argmax(test_targets, axis=1))/len(cell_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# The test accuracy is about 77%-83% and validation loss ranging from 0.7 to 0.5 depending on weight's initialization. Not bad at all for a pretty straightforward model trying to identify 7 types of cells.  I tried to use a simpler network, such as using 1 Conv2D->Dropout->Maxpooling, but the model wasn't able to learn at all. I've made some tests using higher and lower dropouts, but 0.3 was the optimal value as far as I could find out. The objective now is to make a more robust architecture to improve the accuracy (my goal is to at least get 85%). 
# 
# I hope this small notebook is helpful to people trying to learn about CNN and how to make them in Keras using this wonderful dataset!
