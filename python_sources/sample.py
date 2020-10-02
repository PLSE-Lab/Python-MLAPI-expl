#!/usr/bin/env python
# coding: utf-8

# ## Load modules

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import glob
import numpy as np 
import matplotlib.pyplot as plt

import pydicom as dicom
import nibabel as nib


# ### Set path

# In[2]:


train_image_folder = "../input/train-images/image/"
train_label_folder = "../input/train-labels/label/"

sample_list = os.listdir(train_image_folder)[: 10]


# In[3]:


sample_list


# In[4]:


def load_dicom_volume(src_dir, suffix='*.dcm'):
    """Load DICOM volume and get meta data.
    """

    # Read dicom files from the source directory
    # Sort the dicom slices in their respective order by slice location
    dicom_scans = [dicom.read_file(sp)                    for sp in glob.glob(os.path.join(src_dir, suffix))]
    # dicom_scans.sort(key=lambda s: float(s.SliceLocation))
    dicom_scans.sort(key=lambda s: float(s[(0x0020, 0x0032)][2]))

    # Convert to int16, should be possible as values should always be low enough
    # Volume image is in z, y, x order
    volume_image = np.stack([ds.pixel_array                              for ds in dicom_scans]).astype(np.int16)

    # Get data info
    # spacing = list(dicom_scans[0].PixelSpacing) + [dicom_scans[0].SliceThickness]
    # spacing = list(map(float, spacing))
    # patient_position = list(map(float, dicom_scans[0].ImagePositionPatient))
    # info_dict = {"PatientPosition" : patient_position, 'Spacing': spacing}
    
    return volume_image

def load_label(label_fpath):
    label_data = nib.load(label_fpath)
    label_array = label_data.get_fdata()
    return np.transpose(label_array, axes=(2, 1, 0))


# In[5]:


volume_image = load_dicom_volume(os.path.join(train_image_folder, sample_list[0]))
label_array = load_label(os.path.join(train_label_folder, sample_list[0] + '.nii.gz'))


# In[6]:


_slice = 66

plt.imshow(volume_image[_slice, :, :], cmap='gray')
plt.show()
plt.imshow(label_array[_slice, :, :], cmap='gray')
plt.show()


# In[7]:


volume_image_list = [load_dicom_volume(os.path.join(train_image_folder, sample_name)) for sample_name in sample_list]
label_array_list = [load_label(os.path.join(train_label_folder, sample_name + '.nii.gz')) for sample_name in sample_list]


# In[8]:


train_image = np.vstack([volume_image for volume_image in volume_image_list]).astype(np.float)
train_image = train_image.reshape(train_image.shape + (1,))

train_label = np.vstack([label_array for label_array in label_array_list]).astype(np.float)
train_label = train_label.reshape(train_label.shape + (1,))
train_label = train_label[: train_image.shape[0]]


# In[ ]:





# ## Try to apply UNet 

# In[9]:


from keras.models import Model
from keras import layers as klayers
from keras.optimizers import Adam
from keras import backend as K

# Make sure keras running on GPU
K.tensorflow_backend._get_available_gpus()


# ### Check GPU

# In[10]:


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[ ]:





# ## Loss function

# In[11]:


def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


# ## Model architecture

# In[12]:


def unet(pretrained_weights=None, input_size=[512, 512, 1], depth=3, init_filter=8, 
         filter_size=3, padding='same', pool_size=[2, 2], strides=[2, 2]):
    
    inputs = klayers.Input(input_size)
    
    current_layer = inputs
    encoding_layers = []
    
    # Encoder path
    for d in range(depth + 1):
        num_filters = init_filter * 2 ** d
        
        conv = klayers.Conv2D(num_filters, filter_size, padding=padding, kernel_initializer='he_normal')(current_layer)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        conv = klayers.Conv2D(num_filters * 2, filter_size, padding=padding, kernel_initializer='he_normal')(conv)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        encoding_layers.append(conv)
    
        pool = klayers.MaxPooling2D(pool_size=pool_size)(conv)
        
        if d == depth:
            # Bridge
            current_layer = conv
        else:
            current_layer = pool

        
    # Decoder path
    for d in range(depth, 0, -1):
        num_filters = init_filter * 2 ** d
        up = klayers.Deconvolution2D(num_filters * 2, pool_size, strides=strides)(current_layer)

        crop_layer = encoding_layers[d - 1]
        # Calculate two layers shape
        up_shape = np.array(up._keras_shape[1:-1])
        conv_shape = np.array(crop_layer._keras_shape[1:-1])

        # Calculate crop size of left and right
        crop_left = (conv_shape - up_shape) // 2

        crop_right = (conv_shape - up_shape) // 2 + (conv_shape - up_shape) % 2
        crop_sizes = tuple(zip(crop_left, crop_right))

        crop = klayers.Cropping2D(cropping=crop_sizes)(crop_layer)

        # Concatenate
        up = klayers.Concatenate(axis=-1)([crop, up])
        conv = klayers.Conv2D(num_filters, filter_size, padding=padding, kernel_initializer='he_normal')(up)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        conv = klayers.Conv2D(num_filters, filter_size, padding=padding, kernel_initializer='he_normal')(conv)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        
        current_layer = conv
    
    
    outputs = klayers.Conv2D(1, 1, padding=padding, kernel_initializer='he_normal')(current_layer)
    outputs = klayers.Activation('sigmoid')(outputs)
    model = Model(inputs=inputs, outputs=outputs)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


# In[13]:


model = unet(depth=3)
model.compile(optimizer=Adam(lr=1e-4), loss=dice_coefficient_loss, metrics=[dice_coefficient, 'accuracy'])
print(model.summary())


# In[14]:


history = model.fit(train_image, train_label,
                    batch_size=4,
                    epochs=5,
                    verbose=1,
                    validation_split=0.1)


# In[ ]:




