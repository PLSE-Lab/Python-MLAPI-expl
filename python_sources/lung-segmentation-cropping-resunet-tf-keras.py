#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# In the last Kernel [(Finding Pneumo: EDA and UNet Starter Code)](https://www.kaggle.com/ekhtiar/finding-pneumo-eda-and-unet-starter-code), we did explored the data and trained an UNet model that wasn't performing. In this Kernel, we create a ResNet model, which manages to identify some cases of Pneumothorax.
# 
# **In this Kernel we will not take advantage of any leakage because leakage isn't going to save an Pneumothorax patient.** My goal is identify approach and model that hopefully generalizes well to this and other datasets. For this purpose, I will document the journey of training our ResUNet in the summary section of this Kernel.

# In[ ]:


# Basic imports for the entire Kernel
import numpy as np
import pandas as pd
# imports for loading data
import pydicom
from glob import glob
from tqdm import tqdm
# import mask function
import sys
sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')
from mask_functions import rle2mask, mask2rle
# plotting function
from matplotlib import pyplot as plt


# In[ ]:


# load rles
rles_df = pd.read_csv('../input/siim-train-test/siim/train-rle.csv')
# the second column has a space at the start, so manually giving column name
rles_df.columns = ['ImageId', 'EncodedPixels']


# In[ ]:


def dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=True):
    """Parse DICOM dataset and returns a dictonary with relevant fields.

    Args:
        dicom_data (dicom): chest x-ray data in dicom format.
        file_path (str): file path of the dicom data.
        rles_df (pandas.core.frame.DataFrame): Pandas dataframe of the RLE.
        encoded_pixels (bool): if True we will search for annotation.
        
    Returns:
        dict: contains metadata of relevant fields.
    """
    
    data = {}
    
    # Parse fields with meaningful information
    data['patient_name'] = dicom_data.PatientName
    data['patient_id'] = dicom_data.PatientID
    data['patient_age'] = int(dicom_data.PatientAge)
    data['patient_sex'] = dicom_data.PatientSex
    data['pixel_spacing'] = dicom_data.PixelSpacing
    data['file_path'] = file_path
    data['id'] = dicom_data.SOPInstanceUID
    
    # look for annotation if enabled (train set)
    if encoded_pixels:
        encoded_pixels_list = rles_df[rles_df['ImageId']==dicom_data.SOPInstanceUID]['EncodedPixels'].values
       
        pneumothorax = False
        for encoded_pixels in encoded_pixels_list:
            if encoded_pixels != ' -1':
                pneumothorax = True
        
        # get meaningful information (for train set)
        data['encoded_pixels_list'] = encoded_pixels_list
        data['has_pneumothorax'] = pneumothorax
        data['encoded_pixels_count'] = len(encoded_pixels_list)
        
    return data


# In[ ]:


# create a list of all the files
train_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-train/*/*/*.dcm'))
# parse train DICOM dataset
train_metadata_df = pd.DataFrame()
train_metadata_list = []
for file_path in tqdm(train_fns):
    dicom_data = pydicom.dcmread(file_path)
    train_metadata = dicom_to_dict(dicom_data, file_path, rles_df)
    train_metadata_list.append(train_metadata)
train_metadata_df = pd.DataFrame(train_metadata_list)


# In[ ]:


# create a list of all the files
test_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-test/*/*/*.dcm'))
# parse test DICOM dataset
test_metadata_df = pd.DataFrame()
test_metadata_list = []
for file_path in tqdm(test_fns):
    dicom_data = pydicom.dcmread(file_path)
    test_metadata = dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=False)
    test_metadata_list.append(test_metadata)
test_metadata_df = pd.DataFrame(test_metadata_list)


# ## Lung Segmentation
# In my last EDA, and in the discussion section of this competiton, [Dr. Konya](https://www.kaggle.com/sandorkonya) made some awesome contributions. Something very important and obvious he mentioned is since pneumothorax would only appear on the lungs as far as the use cases for this competition is concerend, we should try to leverage pre-existing models for doing lung segmentation in this competition.
# 
# I found some amazing works on Github, one of which is from the imlab-uiip for lung segmentation. They have also uploaded a pre-trained model on the Github repo. In this section, we use this pre-trained model to create masks for the data in this competition. We also conver the mask with RLE function provided by this competition to make it easy for us to use in future cases.
# 
# I ran the model once, and the accuracy of the lung segmentation wasn't perfect. However, this model is still probably good enough to automatically crop the unwanted part of our image. In the discussion [Lung segmentation Dataset of the training images by a radiologist](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/100864) [Dr. Konya](https://www.kaggle.com/sandorkonya) points to an annotation he did for the competition to crop out only the important part for us. Using this model we will try to automate this and get the bounding box for our segmentation. In this way, we may be able to remove a lot of noise from our data and get better accuracy of our model.

# In[ ]:


import tensorflow as tf
import cv2
from skimage import morphology, io, color, exposure, img_as_float, transform


# In[ ]:


model_dir = '../input/lung-segmentation-for-siimacr-pneumothorax/trained_model.hdf5'
lung_seg_model = tf.keras.models.load_model(model_dir, custom_objects=None, compile=True)


# In[ ]:


def get_lung_seg_tensor(file_path, batch_size, seg_size, n_channels):
    
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((batch_size, seg_size, seg_size, n_channels))

        # Process Image
        pixel_array = pydicom.read_file(file_path).pixel_array
        image_resized = cv2.resize(pixel_array, (seg_size, seg_size))
        image_resized = exposure.equalize_hist(image_resized)
        image_resized = np.array(image_resized, dtype=np.float64)
        image_resized -= image_resized.mean()
        image_resized /= image_resized.std()
        # Store Image
        X[0,] = np.expand_dims(image_resized, axis=2)

        return X


# In[ ]:


def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


# In[ ]:


def bounding_box(img):
    # return max and min of a mask 
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


# In[ ]:


def get_lung_seg_rle(metadata_df, seg_size):

    processed_images = []

    for id, row in metadata_df.iterrows():
        # get image in the 4d tensor
        img = get_lung_seg_tensor(row['file_path'],1,seg_size,1)
        # get segmented mask
        seg_mask = lung_seg_model.predict(img).reshape((seg_size,seg_size))
        # only take above .5
        seg_mask = seg_mask > 0.5
        # remove small region
        seg_mask = remove_small_regions(seg_mask, 0.02 * np.prod(seg_size))
        processed_img = {}
        processed_img['id'] = row['id']
        processed_img['lung_mask'] = mask2rle(seg_mask*255, seg_size, seg_size)
        processed_img['rmin'], processed_img['rmax'], processed_img['cmin'], processed_img['cmax'] = bounding_box(seg_mask)
        processed_images.append(processed_img)
    
    return pd.DataFrame(processed_images)


# In[ ]:


seg_size = 256


# I have created the [Lung Segmentation For SIIM-ACR Pneumothorax
# ](https://www.kaggle.com/ekhtiar/lung-segmentation-for-siimacr-pneumothorax) dataset of the pre-trained model and also the lung segmentation mask pre-processed to make everyone's life easier.

# In[ ]:


#try:
#    train_lung_mask_df = pd.read_csv('../input/lung-segmentation-for-siimacr-pneumothorax/train_lung_mask.csv')
#except FileNotFoundError:
train_lung_mask_df = get_lung_seg_rle(train_metadata_df, seg_size)
train_lung_mask_df.to_csv('./train_lung_mask.csv', index=False)


# In[ ]:


#try:
#    test_lung_mask_df = pd.read_csv('../input/lung-segmentation-for-siimacr-pneumothorax/test_lung_mask.csv')
#except FileNotFoundError:
test_lung_mask_df = get_lung_seg_rle(test_metadata_df, seg_size)
test_lung_mask_df.to_csv('./test_lung_mask.csv', index=False)


# #### proof of concept
# lets check out our segmentation and visually inspect if it is working

# In[ ]:


def plot_lung_seg(file_path, mask_encoded_list, lung_mask, rmin, rmax, cmin, cmax):
    
    pixel_array = pydicom.dcmread(file_path).pixel_array
    
    # use the masking function to decode RLE
    mask_decoded_list = [rle2mask(mask_encoded, 1024, 1024).T for mask_encoded in mask_encoded_list]
    lung_mask_decoded = cv2.resize(rle2mask(lung_mask, 256, 256), (1024,1024))
    rmin, rmax, cmin, cmax =  rmin * 4, rmax * 4, cmin * 4, cmax * 4 
    
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(20,10))
    
    
    ax[0].imshow(pixel_array, cmap=plt.cm.bone)
    ax[0].imshow(lung_mask_decoded, alpha=0.3, cmap="Blues")
    ax[0].set_title('Xray with Lung Mask')
    
    ax[1].imshow(pixel_array[rmin:rmax+1,cmin:cmax+1], cmap=plt.cm.bone)
    ax[1].set_title('Cropped Xray')
   
    ax[2].imshow(lung_mask_decoded, cmap='Blues')
    for mask_decoded in mask_decoded_list:
        ax[2].imshow(mask_decoded, alpha=0.3, cmap="Reds")
    ax[2].set_title('Lung Mask with Pneumothorax')


# In[ ]:


train_lm_metadata_df = pd.concat([train_metadata_df, train_lung_mask_df.drop('id',axis=1)], axis=1)
test_lm_metadata_df = pd.concat([test_metadata_df, test_lung_mask_df.drop('id',axis=1)], axis=1)


# In[ ]:


for i, r in train_lm_metadata_df[train_lm_metadata_df['has_pneumothorax']==True][:10].iterrows():
    file_path = r['file_path']
    encoded_pixels_list = r['encoded_pixels_list']
    lung_mask = r['lung_mask']
    rmin = r['rmin'] 
    rmax = r['rmax']
    cmin = r['cmin']
    cmax = r['cmax']
    
    plot_lung_seg(file_path, encoded_pixels_list, lung_mask, rmin, rmax, cmin, cmax)


# You can see our model gets it right more often than not. Although our lung segmentation is far from perfect, the crop version of the image looks promising! In the next kernel, I will try to make a comparision and see how close we got to the hand made annotation of 1K images from Dr. Konya.

# ## ResUNet
# 
# In this section we will use ResUNet instead of UNet to predict pneumothorax. The original paper that proposes this CNN architecture is [ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data](https://arxiv.org/abs/1904.00592). If you read the paper you will get more details about the network. However for reference, I am attaching the architecture of the model below: 
# 
# ![ResUNet Architecture](https://raw.githubusercontent.com/nikhilroxtomar/Deep-Residual-Unet/master/images/arch.png)

# In[ ]:


# defining configuration parameters
img_size = 512 # image resize size
batch_size = 8
# batch size for training unet
k_size = 3 # kernel size 3x3
val_size = .20 # split of training set between train and validation set
no_pneumo_drop = 0 # dropping some data to balance the class a little bit better


# In[ ]:


# imports for building the network
from tensorflow import reduce_sum
from tensorflow.keras.backend import pow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, Add, Flatten
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
import cv2


# #### Data Generator
# 
# To push the data to our model, we will create a custom data generator. A generator lets us load data progressively, instead of loading it all into memory at once. A custom generator allows us to also fit in more customization during the time of loading the data. As the model is being procssed in the GPU, we can use a custom generator to pre-process images via a generator. At this time, we can also take advantage multiple processors to parallelize our pre-processing.

# In[ ]:


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_path_list, labels, batch_size=32, 
                 img_size=256, channels=1, shuffle=True):
        self.file_path_list = file_path_list
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.channels = channels
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'denotes the number of batches per epoch'
        return int(np.floor(len(self.file_path_list)) / self.batch_size)
    
    def __getitem__(self, index):
        'generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # get list of IDs
        file_path_list_temp = [self.file_path_list[k] for k in indexes]
        # generate data
        X, y = self.__data_generation(file_path_list_temp)
        # return data 
        return X, y
    
    def on_epoch_end(self):
        'update ended after each epoch'
        self.indexes = np.arange(len(self.file_path_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, file_path_list_temp):
        'generate data containing batch_size samples'
        X = np.empty((self.batch_size, self.img_size, self.img_size, self.channels))
        y = np.empty((self.batch_size, self.img_size, self.img_size, self.channels))
        
        for idx, file_path in enumerate(file_path_list_temp):
            
            id = file_path.split('/')[-1][:-4]
            rle = self.labels.get(id)
            image = pydicom.read_file(file_path).pixel_array
            image_resized = cv2.resize(image, (self.img_size, self.img_size))
            image_resized = np.array(image_resized, dtype=np.float64)
            
            X[idx,] = np.expand_dims(image_resized, axis=2)
            
            # if there is no mask create empty mask
            # notice we are starting of with 1024 because we need to use the rle2mask function
            if rle is None:
                mask = np.zeros((1024, 1024))
            else:
                if len(rle) == 1:
                    mask = rle2mask(rle[0], 1024, 1024).T
                else: 
                    mask = np.zeros((1024, 1024))
                    for r in rle:
                        mask =  mask + rle2mask(r, 1024, 1024).T
                        
            mask_resized = cv2.resize(mask, (self.img_size, self.img_size))
            y[idx,] = np.expand_dims(mask_resized, axis=2)
            
        # normalize 
        X = X / 255
        y = (y > 0).astype(int)
            
        return X, y


# In[ ]:


masks = {}
for index, row in train_metadata_df[train_metadata_df['has_pneumothorax']==1].iterrows():
    masks[row['id']] = list(row['encoded_pixels_list'])


# In[ ]:


bad_data = train_metadata_df[train_metadata_df['encoded_pixels_count']==0].index
new_train_metadata_df = train_metadata_df.drop(bad_data)


# In[ ]:


drop_data = new_train_metadata_df[new_train_metadata_df['has_pneumothorax'] == False].sample(no_pneumo_drop).index
new_train_metadata_df = new_train_metadata_df.drop(drop_data)


# In[ ]:


# split the training data into train and validation set (stratified)
X_train, X_val, y_train, y_val = train_test_split(new_train_metadata_df.index, new_train_metadata_df['has_pneumothorax'].values, test_size=val_size, random_state=42)
X_train, X_val = new_train_metadata_df.loc[X_train]['file_path'].values, new_train_metadata_df.loc[X_val]['file_path'].values


# In[ ]:


params = {'img_size': img_size,
          'batch_size': batch_size,
          'channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(X_train, masks, **params)
validation_generator = DataGenerator(X_val, masks, **params)


# We can verify that our generator class is working and is passing the right data visually in the following way.

# In[ ]:


x, y = training_generator.__getitem__(0)
print(x.shape, y.shape)


# In[ ]:


fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(x[6].reshape(img_size, img_size), cmap=plt.cm.bone)
ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(y[6], (img_size, img_size)), cmap="gray")


# ### ResUNet TensorFlow Keras Implementation
# 
# Lets build the ResUNet model in this section. Actually, I found a nice implementation of ResUNet on [Github](https://github.com/nikhilroxtomar/Deep-Residual-Unet), which I am using for this section of the Kernel.

# In[ ]:


def bn_act(x, act=True):
    'batch normalization layer with an optinal activation layer'
    x = tf.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tf.keras.layers.Activation('relu')(x)
    return x


# In[ ]:


def conv_block(x, filters, kernel_size=3, padding='same', strides=1):
    'convolutional layer which always uses the batch normalization layer'
    conv = bn_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv


# In[ ]:


def stem(x, filters, kernel_size=3, padding='same', strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size, padding, strides)
    shortcut = Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([conv, shortcut])
    return output


# In[ ]:


def residual_block(x, filters, kernel_size=3, padding='same', strides=1):
    res = conv_block(x, filters, k_size, padding, strides)
    res = conv_block(res, filters, k_size, padding, 1)
    shortcut = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([shortcut, res])
    return output


# In[ ]:


def upsample_concat_block(x, xskip):
    u = UpSampling2D((2,2))(x)
    c = Concatenate()([u, xskip])
    return c


# In[ ]:


def ResUNet(img_size):
    f = [16, 32, 64, 128, 256, 512, 1024, 2048] * 32
    inputs = Input((img_size, img_size, 1))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    e6 = residual_block(e5, f[5], strides=2)
    e7 = residual_block(e6, f[6], strides=2)
    
    ## Bridge
    b0 = conv_block(e7, f[6], strides=1)
    b1 = conv_block(b0, f[6], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e6)
    d1 = residual_block(u1, f[6])
    
    u2 = upsample_concat_block(d1, e5)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e4)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e3)
    d4 = residual_block(u4, f[1])
    
    u5 = upsample_concat_block(d4, e2)
    d5 = residual_block(u5, f[1])
    
    u6 = upsample_concat_block(d5, e1)
    d6 = residual_block(u6, f[1])
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d6)
    model = tf.keras.models.Model(inputs, outputs)
    return model


# In[ ]:


def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = Flatten()(y_true)
    y_pred_f = Flatten()(y_pred)
    intersection = reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (reduce_sum(y_true_f) + reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


# In[ ]:


model = ResUNet(img_size)
adam = tf.keras.optimizers.Adam(lr = 0.01, epsilon = 0.1)
model.compile(optimizer=adam, loss=bce_dice_loss, metrics=[dsc])
#model.summary() # print out the architecture of our network


# In[ ]:


# load a pre trained model here if you wish
# model.load_weights('../input/resunet-e200-s256/ResUNet.h5')


# In[ ]:


# running more epoch to see if we can get better results
history = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=10, verbose=1)


# In[ ]:


model.save('./ResUNet.h5')


# #### Plotting Model Training History
# 
# In this section we will use the simple non-flashy matplotlib to plot out the performance of our model.

# In[ ]:


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.plot(history.history['dsc'])
plt.plot(history.history['val_dsc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


# #### Checking out our model
# 
# We can visually inspect how our model is doing for our model in the following way.

# In[ ]:


def plot_train(img, mask, pred):
    
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15,5))
    
    ax[0].imshow(img, cmap=plt.cm.bone)
    ax[0].set_title('Chest X-Ray')
    
    ax[1].imshow(mask, cmap=plt.cm.bone)
    ax[1].set_title('Mask')
    
    ax[2].imshow(pred, cmap=plt.cm.bone)
    ax[2].set_title('Pred Mask')
    
    plt.show()


# In[ ]:


# lets loop over the predictions and print some good-ish results
count = 0
for i in range(0,50):
    if count <= 50:
        x, y = validation_generator.__getitem__(i)
        predictions = model.predict(x)
        for idx, val in enumerate(x):
            #if y[idx].sum() > 0 and count <= 15: 
                img = np.reshape(x[idx]* 255, (img_size, img_size))
                mask = np.reshape(y[idx]* 255, (img_size, img_size))
                pred = np.reshape(predictions[idx], (img_size, img_size))
                pred = pred > 0.5
                pred = pred * 255
                plot_train(img, mask, pred)
                count += 1


# #### Making Predictions
# 
# In this section, we predict using our model and create a submission without taking advantage of the leak.

# In[ ]:


def get_test_tensor(file_path, batch_size, img_size, channels):
    
        X = np.empty((batch_size, img_size, img_size, channels))

        # Store sample
        pixel_array = pydicom.read_file(file_path).pixel_array
        image_resized = cv2.resize(pixel_array, (img_size, img_size))
        image_resized = np.array(image_resized, dtype=np.float64)
        image_resized -= image_resized.mean()
        image_resized /= image_resized.std()
        X[0,] = np.expand_dims(image_resized, axis=2)

        return X


# In this competition we are very likely to make a lot of tiny small predictions. If the region is very small generally it is a good policy to take them out. The function below will help us do this.

# In[ ]:


from skimage import morphology

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


# Lets use the model we have just built up to make prediction and identify pneumothorax.

# In[ ]:


submission = []

for i, row in test_metadata_df.iterrows():

    test_img = get_test_tensor(test_metadata_df['file_path'][i],1,img_size,1)
    
    pred_mask = model.predict(test_img).reshape((img_size,img_size))
    prediction = {}
    prediction['ImageId'] = str(test_metadata_df['id'][i])
    pred_mask = cv2.resize(pred_mask.astype('float32'), (1024, 1024))
    pred_mask = (pred_mask > .5).astype(int)
    pred_mask = remove_small_regions(pred_mask, 0.02 * np.prod(1024))
    
    if pred_mask.sum() < 1:
        prediction['EncodedPixels']=  -1
    else:
        prediction['EncodedPixels'] = mask2rle(pred_mask.T * 255, 1024, 1024)
        
    submission.append(prediction)


# In[ ]:


submission_df = pd.DataFrame(submission)
submission_df = submission_df[['ImageId','EncodedPixels']]
# check out some predictions and see if it looks good
submission_df[ submission_df['EncodedPixels'] != -1].head()


# In[ ]:


submission_df.to_csv('./submission.csv', index=False)

