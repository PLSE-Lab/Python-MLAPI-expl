#!/usr/bin/env python
# coding: utf-8

# For this project, I originally created a kernel that trained a CNN model (I originally used the LeNet architecture) on the data to try and predict the bounding box measurements as 5 features of the pixel data in each DICOM image. However, I came across two major problems that I could not seem to fix:One was that the training of the model quickly drained all the run time memory and threw an ResourceExhaustedError. Additionally, the a patient with pneumonia does not always have just one bounding box - a patient can have several different opacities which reflect pneumonia, and a new DICOM image with the same patientId and different bounding box measurements will be in the files for each opacity. To try and start again with something that could solve these problems, I looked at the most popular kernel, located at https://www.kaggle.com/jonnedtc/cnn-segmentation-connected-components/ for ideas. The approach of this kernel, which I followed conceptually, was to aggregate the different images of each patient into one, and combine the bounding boxes into one mask for each patient. A mask would be an equal size array of zeros, with 1's instead for items we choose to keep. The idea is that certain operations between the mask and image,such as AND, will annihilate values we don't want to consider. Then, we can compare the model predictions on normalized data to the mask for each patient.

# In[2]:


import os
import pydicom
import numpy as np
import pandas as pd
from skimage import measure
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras


# After importing all the packages we need, the first step we can take is to analyze the data.

# In[3]:


train_labels = pd.read_csv('../input/stage_2_train_labels.csv')
print(train_labels.head())
print(pd.DataFrame(os.listdir('../input/stage_2_train_images/')).head())


# So the columns of the labels are the patientId, followed by the location and dimensions of the bounding box, followed by the Target(a 1 or 0, where 1 indicates a patient with pneumonia.) The files are named after the patientId, so we can use patientIds to retrieve them. To build our masks, the source notebook has an interesting approach. A simple way to do this would be to create a mask for each patient and use the bounding boxes to build it, but unfortunately, the data is far too large to hold in memory all at once. Luckily, the Sequence class of keras contains generators that can modify for our specific purpose. We can create masks for our images as we load them into the model, but before that we can create a dictionary of all the coordinates to mask for each patient, which will fit into memory easily. 

# In[4]:


pneum_locs = {}
for x in range(0,len(train_labels)):
    row = train_labels.iloc[x]
    patientId = row[0]
    loc = row[1:3]

    if row[4] == '1': #patient has pneumonia
        location = [int(float(i))for i in loc] #data in labels is in string form, must convert
        if(patientId in pneum_locs): #patient already in dictionary
            pneum_locs[patientId].append(loc)
        else:
            pneum_locs[patientId] = [loc]


# So now, pneum_locs contains, for each patient with pneumonia, the location and dimensions of the opacity bounding boxes. Now is the time to build our generator. The docs for keras.utils.Sequence(located at https://keras.io/utils/) mandate that every subclass must override the methods for retrieval and the data length of each batch. We will also override initialization, since we want to set a mode for when we use our generator to predict on test data, and set an image size to resize to (making the image smaller improves runtime). 

# In[40]:


class datagenerator(keras.utils.Sequence):
    def __init__(self,patientIds, p_locations = None, batch_size = 32,image_size = 256):
        self.patientIds = patientIds
        self.p_locations = p_locations
        self.batch_size = batch_size
        self.image_size = image_size
      
    def __load__(self,pID):#loads a file for retrieval 
        image = pydicom.read_file('../input/stage_2_train_images/%s.dcm' % pID).pixel_array
        mask = np.zeros(image.shape)
        image = resize(image,(self.image_size,self.image_size),mode = 'reflect')
        mask = resize(mask, (self.image_size, self.image_size), mode='reflect') > 0.5
        if pID in self.p_locations :
            for loc in self.p_locations[patID]:
                x,y,w,h = loc
                mask[y:y+h,x:x+w] = 1
        image = np.expand_dims(image,-1) #X = image
        mask = np.expand_dims(mask,-1) #Y = mask

        return image,mask
    def __getitem__(self,index):#mandatory inheritance
        pIDs = self.patientIds[index*self.batch_size:(index + 1)*self.batch_size]
        images,masks = zip(*[self.__load__(patientId) for patientId in pIDs])
        images = np.array(images)
        masks = np.array(masks)
        return images,masks
    def __len__(self): #mandatory inheritance
        return int(len(self.patientIds)/self.batch_size)

        


# The source notebook created a model that used creates a CNN using two methods: One that creates a layer to downsample the data, and one that creates a residual block to feed directly to the output as a way to avoid losing information through too many backpropagations. I used the same model, but removed some of the layers to improve runtime.

# In[6]:


def create_downsample(channels, inputs):
    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.LeakyReLU(0)(x)#LeakyReLU with alpha = 0 is identical to ReLU
    x = keras.layers.Conv2D(channels, 1, padding='same')(x)
    x = keras.layers.MaxPool2D(2)(x)
    return x

def create_resblock(channels, inputs):
    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same')(x)

    return keras.layers.add([x, inputs])

def create_network(input_size, channels, n_blocks=2, depth=4): #creates a residual block layer
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = keras.layers.Conv2D(channels, 3, padding='same')(inputs)
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    # output
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    outputs = keras.layers.UpSampling2D(2**depth)(x) ##upsample data to counteract the first downsample
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# The source notebook provided a useful function for calculating the intersection-over-union loss. I used it as the loss for my model, but not the other two methods provided as they caused errors and their purpose was unclear. For the model fitting, the system gave me an error when using batch sizes greater than 1 and I could not find out why, so I stuck with that as it did not seem to affect training time too much. 

# In[9]:


def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score




model = create_network(input_size=256, channels=32, n_blocks=2, depth=4)
model.compile(optimizer='adam',
              loss=iou_loss,
              metrics=['accuracy'])
val_size = 3000 #about 10% of data used for validation

train_IDs = train_labels['patientId'][val_size:]
val_IDs= train_labels['patientId'][:val_size]
validgen = datagenerator(val_IDs,pneum_locs,batch_size = 1,image_size = 256)
traingen = datagenerator(train_IDs, pneum_locs, batch_size=1, image_size=256)
hist = model.fit_generator(traingen,epochs = 3,validation_data = validgen, workers = 4,use_multiprocessing = True)


# Finally, we test our model against the test data. I decided not to use another generator here, as we can just feed the model "batches" of 1 image at a time and finish in reasonable time as we are processing 3000 images as opposed to more than 30,000 five times. 

# In[39]:



test_IDs = pd.DataFrame(os.listdir('../input/stage_2_test_images/'))
for i in range (0, len(test_IDs)):
    test_IDs[0][i]=test_IDs[0][i].split('.')[0]
test_IDs = test_IDs[0]

submission = {}
for pID in test_IDs :
    image = pydicom.read_file('../input/stage_2_test_images/%s.dcm' % pID).pixel_array
    image = resize(image,(256,256),mode = 'reflect')
    image = np.expand_dims(image,-1)
    images = np.zeros((1,256,256,1))
    images[0] = image
    pred = model.predict(images)
    predict = resize(np.squeeze(pred),(1024,1024), mode = 'reflect')
    compute = predict[:,:] >0.5 #transforms values to 1s and 0s
    compute = measure.label(compute)
    predString = ''
    for region in measure.regionprops(compute):
        y,x,y2,x2 = region.bbox
        confidence = np.mean(predict[y:y2,x:x2])
        predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(y2-y) + ' ' + str(x2 - x) + ' '
    submission[pID]= predString
    if(len(submission) >= len(test_IDs)): #loop exit control
        break

submit = pd.DataFrame.from_dict(submission,orient ='index')
print("%s predictions recorded." % len(submit))
submit.index.names = ['patientId']
submit.columns = ['PredictionString']
submit.to_csv('submission.csv')


# In[ ]:




