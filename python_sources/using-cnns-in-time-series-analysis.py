#!/usr/bin/env python
# coding: utf-8

# Hey everyone! I've seen that people have started using NNs for a while now, but I didn't see people using CNNs. Indeed, it is pretty unusual, at least to me, to use a kind of network that performs so well on image data to treat time series. However, I'm completely in love with this kind of NN and I started looking for ways to use it here and I found [this paper](https://arxiv.org/pdf/1710.00886.pdf). There are several issues that we have to deal with along the preprocessing and evaluation steps but I believe that this could be a good starting point for those who want to try using CNNs. I prepared this notebook as an introduction in the topic and necessary preprocessing in case someone wants to investigate it deeper. IMHO, the greatest advantage of this method - and other DL techniques - is that we don't need worry about feature extraction.
# 
# First of all, we'll need to load the data. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 #deal with images
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm
np.random.seed(42)

training = pd.read_csv('../input/training_set.csv')
meta_training = pd.read_csv("../input/training_set_metadata.csv")
merged = training.merge(meta_training, on = "object_id")


# The basic idea described in the paper is that we can transform a 1-dimensional time series into a 2-dimensional image by calculating its recurrences, that is, revealing in which points the series returns to a previous value.  This is done by using the equation:
# 
# $$ R_{i,j} = \theta(\epsilon - || s_i - s_j ||)$$
# 
# where a $s$ are the states (time series values), ||.|| is the norm  and $\epsilon$ is a threshold distance and $\theta$ is the Heaviside (step) function. 
# 
# Below are the functions I used to create a dictionary of such R matrices, where each key is a different object. You'll notice that I preferred to use a slightly different approach by using a sigmoid rather than Heaviside step function. I did so because since we are dealing with several different objects, it turns out that each one has a different "optimal" $\epsilon$ in order to create a proper image. So for simplicity I opted to use sigmoid because it doesn't need tunning for each image.
# 
# After each R matrix for each object and each passband is made, there is another issue that we need to worry about.  Since each object has 6 passbands, we need a way to put all of those images - one for each passband -  into a single image. My solution was to stack them up , in some kind of resemblance to a RGB image that has 3 channels. Here we will have 6 channels instead. The problem is: it is not guaranteed that each passband has the same number of observations, resulting in images of different sizes. If we want to stack them, they need to have the same shape. To solve that, I cropped all of them to the minimum size possible for that object - and here is where we begin to lose information. For instance, if an object has R matrices of shapes (30x30), (50x50) and (48x48), I'd shape them all to (30x30) so they can be stacked. This is done by `crop_obj_plots` function.

# In[ ]:


###recurrent plot

def sigmoid(x):
    '''
    Returns the sigmoid of a value
    '''
    return 1/(1+np.exp(-x))

def R_matrix(signal, eps):
    '''
    Given a time series (signal) and an epsilon,
    return the Recurrent Plot matrix
    '''
    R = np.zeros((signal.shape[0], signal.shape[0]))
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i][j] = np.heaviside((eps - abs(signal[i] - signal[j])),1)
    return R

#using sigmoid rather than heaviside
#because in this dataset the epsilon parameter needs to
#change from object to object and therefore should be learned as well
def R_matrix_modified(signal):
    '''
    Given a time series (signal) and an epsilon,
    return the modified Recurrent Plot matrix
    using sigmoid rather than heaviside
    '''
    R = np.zeros((signal.shape[0], signal.shape[0]))
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i][j] = sigmoid((abs(signal[i] - signal[j])))
    return R

def create_objects_dict(merged_dataset):
    '''
    Input: dataset containing both training data and metadata
    Creates a dictionary using each object as keys and
    one R matrix for each passband in that object
    '''
    objects = {}
    for obj in tqdm(np.unique(merged_dataset.object_id)):
        R_passbands = []
        for passband in np.unique(merged_dataset.passband):
            obj_flux = merged_dataset[(merged_dataset.object_id == obj) & (merged_dataset.passband == passband)].flux.values
            R_passbands.append(R_matrix_modified(obj_flux))
        objects[obj] = (np.asarray(R_passbands), max(merged_dataset[merged_dataset.object_id == obj].target))
    return objects

def get_minmax_shapes(obj_R_matrices):
    '''
    Given an R matrix, get the min and max width 
    to be used to crop and let all images from a given
    object be of the same size so they can be concatenated
    '''
    min_length = 0
    max_length = 0
    for passband in np.unique(merged.passband):
        if passband == 0:
            length = len(obj_R_matrices[passband])
            min_length = length
            max_length = length
        else:
            length = len(obj_R_matrices[passband])
            min_length = min(min_length, length)
            max_length = max(max_length, length)
    return (min_length, max_length)

def crop_obj_plots(objects):
    '''
    Accepts a dictionary where each key is a different object
    and each value is a tuple - one slot with a list of R matrices and 
    the other with the target value (object class)
    '''
    for obj in tqdm(objects.keys()):
        min_len, max_len = get_minmax_shapes(objects[obj][0])
        for passband in np.unique(merged.passband):
            objects[obj][0][passband] = objects[obj][0][passband][:min_len, :min_len]
    return objects

objects = create_objects_dict(merged)
cropped_objects = crop_obj_plots(objects)


# Now each object has a 6-channel image. They all look like the image below:

# In[ ]:


cropped_objects[730][0][3].shape
plt.imshow(cropped_objects[730][0][0])


# Now let's the distribution of shapes across all images.

# In[ ]:


from collections import Counter

shapes = []
for key in tqdm(cropped_objects.keys()):
    shapes.append(cropped_objects[key][0][0].shape[0])
plt.hist(shapes, bins = 50)
Counter(shapes)


# Now we found the second problem and the key source of information loss: we have lots of different shapes, most around 10-12. The most complete images (longer observations) are just a small portion of the dataset. In order to properly work with the dataset, I will try to get most of them into the same shape. From my experience, the very small images are not very good predictors, so I'll just use the larger ones. Now I'll put all images with shapes (50,50) or larger  into shape (57,57). 

# In[ ]:


import math
cropped_2 = np.copy(cropped_objects).item()
for key in tqdm(cropped_2.keys()):
    shape = cropped_2[key][0][0].shape[0]
    if shape < 11:
        for passband in np.unique(merged.passband):
            #how much we will increase the border
            increaseBorder = abs(shape-11)/2
            cropped_2[key][0][passband] = cv2.copyMakeBorder(src = cropped_2[key][0][passband],
                                                             top = math.ceil(increaseBorder), 
                                                             left = math.ceil(increaseBorder),
                                                             bottom = round(increaseBorder),
                                                             right = round(increaseBorder),
                                                             borderType = cv2.BORDER_REFLECT)
    elif shape>11 and shape < 25:
        for passband in np.unique(merged.passband):
            cropped_2[key][0][passband] = cropped_2[key][0][passband][:-(shape-11), :-(shape-11)]
            
    
    elif shape >= 50 and shape < 57:
        for passband in np.unique(merged.passband):
            increaseBorder57 = abs(shape-57)/2
            cropped_2[key][0][passband] = cv2.copyMakeBorder(src = cropped_2[key][0][passband],
                                                             top = math.ceil(increaseBorder57), 
                                                             left = math.ceil(increaseBorder57),
                                                             bottom = round(increaseBorder57),
                                                             right = round(increaseBorder57),
                                                             borderType = cv2.BORDER_REFLECT)
    else:
        continue


# Now let's check how the shapes look like.

# In[ ]:


shapes = []
for key in tqdm(cropped_2.keys()):
    shapes.append(cropped_2[key][0][0].shape[0])
plt.hist(shapes, bins = 50)
Counter(shapes)


# Now we can stack all the images with the same shape to form our input.

# In[ ]:


objects = list(cropped_2.keys())
input_images = list()
labels = list()
for key in tqdm(objects):
    if cropped_2[key][0][0].shape[0] == 57:
        img = np.stack((cropped_2[key][0][0],
                        cropped_2[key][0][1],
                        cropped_2[key][0][2],
                        cropped_2[key][0][3],
                        cropped_2[key][0][4]), axis = -1)  
                                           
        input_images.append(np.expand_dims(img, axis = 0))
        labels.append(cropped_2[key][1])                                                        
input_images = np.vstack(input_images)     
input_images.shape


# We have lots of images with the same shape and therefore we can use them in our CNN. Here I'll just use images with shape 57x57x5.  
# 
# The following cell is to split between train, validation and test samples and also binarize the labels. 

# In[ ]:


#LabelBinarizer and train-test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils


train_fraction = 0.8

encoder = LabelBinarizer()
y = encoder.fit_transform(labels)
x = input_images

train_tensors, test_tensors, train_targets, test_targets =    train_test_split(x, y, train_size = train_fraction, random_state = 42)

val_size = int(0.5*len(test_tensors))

val_tensors = test_tensors[:val_size]
val_targets = test_targets[:val_size]
test_tensors = test_tensors[val_size:]
test_targets = test_targets[val_size:]


# We then train the model. The input_shape needs to be (None, None, 5) rather than (57, 57, 5) - that we know will be our training input shapes - because we want to test the model in images of different sizes as well (remember the huge amount of images of size 11x11?). If we don't set input_shape's width and height to None, our convolutional layers will throw an error. 

# In[ ]:


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense, LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from tensorflow import set_random_seed

set_random_seed(42)

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
checkpointer = ModelCheckpoint(filepath='weights.hdf5', 
                               verbose=1, save_best_only=True)
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 4, padding = 'same', activation = 'relu', input_shape = (None, None,5)))
model.add(Conv2D(filters = 16, kernel_size = 4, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 16, kernel_size = 4, padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = 2)) 

model.add(Conv2D(filters = 32, kernel_size = 4, padding = 'same', activation = 'relu')) 
model.add(Conv2D(filters = 32, kernel_size = 4, padding = 'same', activation = 'relu')) 
model.add(Conv2D(filters = 32, kernel_size = 4, padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = 2)) 

model.add(Conv2D(filters = 64, kernel_size = 4, padding = 'same', activation = 'relu')) 
model.add(Conv2D(filters = 64, kernel_size = 4, padding = 'same', activation = 'relu')) 
model.add(Conv2D(filters = 64, kernel_size = 4, padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(GlobalMaxPooling2D()) 

model.add(Dense(256, activation = 'linear'))
model.add(Dense(128, activation = 'linear'))
model.add(Dense(14, activation = 'softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 100
model.fit(train_tensors, train_targets, 
          validation_data=(val_tensors, val_targets),
          epochs=epochs, batch_size=80, verbose=1, callbacks = [early_stopping, checkpointer])


# In[ ]:


model.load_weights('weights.hdf5')

cell_predictions =  [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

test_accuracy = 100*np.sum(np.array(cell_predictions)==np.argmax(test_targets, axis=1))/len(cell_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# Note that I'm not considering any weights for any particular class here. The loss function that gets the best results so far is categorical crossentropy. By doing so, I'm getting an accuracy of around 60% in the classification of the larger images (objects that have observations for a longer period of time). Making the predictions on the images of shape (11x11), we got:

# In[ ]:


#manipulations to get our images into the proper dimensions (done before with shape 57)
objects = list(cropped_2.keys())
small_labels = list()
small_input_images = list()
for key in tqdm(objects):
    if cropped_2[key][0][0].shape[0] == 11:
        img = np.stack((cropped_2[key][0][0],
                        cropped_2[key][0][1],
                        cropped_2[key][0][2],
                        cropped_2[key][0][3],
                        cropped_2[key][0][4]), axis = -1)  
                                           
        small_input_images.append(np.expand_dims(img, axis = 0))
        small_labels.append(cropped_2[key][1])                                                        
small_input_images = np.vstack(small_input_images)     
small_input_images.shape

#predictions
cell_predictions =  [np.argmax(model.predict(np.expand_dims(small_image, axis=0))) for small_image in small_input_images]

small_encoder = LabelBinarizer()
small_labels_encoded = encoder.fit_transform(small_labels)
test_accuracy = 100*np.sum(np.array(cell_predictions)==np.argmax(small_labels_encoded, axis=1))/len(cell_predictions)
print('Accuracy in small images: %.4f%%' % test_accuracy)


# The model's accuracy for a much smaller test set is awfully low, around 13%-15%. There are at least two possible ways out of this problem (I'm not sure if any would actually work. Just some hypothesis here):
# - Train the model using just the smaller images and try to generalize to the larger ones
# - Train the model using both large and small images
# 
# I'll leave this open for more experienced data scientists to discuss if they want to :) Meanwhile I'll keep working on how to solve this generalization issue and update here if I have an Eureka moment.
