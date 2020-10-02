#!/usr/bin/env python
# coding: utf-8

# Here we show how to visualize and examine the 3D Nifti images and lung masks in python

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc
import numpy as np
from glob import glob
from skimage.io import imread
BASE_IMG_PATH=os.path.join('..','input')
get_ipython().run_line_magic('matplotlib', 'inline')
try:
    import nibabel as nib #for loading data from NIfTI images
except:
    raise ImportError('Install NIBABEL')


# In[ ]:


glob(os.path.join(BASE_IMG_PATH,'3d_images','*'))


# In[ ]:


# show some of the files
all_images=glob(os.path.join(BASE_IMG_PATH,'3d_images','IMG_*')) 
all_masks = [x.replace('IMG_', 'MASK_') for x in all_images]
print(len(all_images),' matching files found:',all_images[0], all_masks[0])
print(all_images)


# !mkdir /kaggle/output
# !ls /kaggle
# !cp /kaggle/input/3d_images/IMG_0031.nii.gz /kaggle/output
# !gunzip /kaggle/output/IMG_0031.nii.gz
# !ls /kaggle/output
# !ls ../input/3d_images/

# from keras.preprocessing.image import ImageDataGenerator
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
# test_datagen = ImageDataGenerator(rescale=1./255)
# 
# train_generator = train_datagen.flow_from_directory(
#         '../input/3d_images',
#         target_size=(150, 150),
#         batch_size=32,
#         class_mode='binary')
# #model.fit_generator(        train_generator,        steps_per_epoch=2000,        epochs=50)

# In[ ]:


all_masks_img =[]
for mask in all_masks:
    all_masks_img.append(nib.load(mask).get_data())
#print(all_masks_img)          
all_img_img =[]
for img in all_images:
    all_img_img.append(nib.load(img).get_data())
for y in all_images:
    
img = nib.load(all_images[0]).get_data()
img1 = nib.load(all_images[1]).get_data()
img2 = nib.load(all_images[2]).get_data()
img3 = nib.load(all_images[3]).get_data()
#print(img)


# The images of the dataset are indeed grayscale images with a dimension of 512 x 512 so before we feed the data into the model it is very important to preprocess it. You'll first convert each 512 x 512 image into a matrix of size 512 x 512 x 1, which you can feed into the network:

# In[ ]:


print(img.shape)
print(img1.shape)
print(img2.shape)
print(img3.shape)


# In[ ]:


def get_range_value(img_value):
    return (img_value.shape[0]//27, img_value.shape[1]//27, img_value.shape[2]//27)


# In[ ]:


for a in range(4):
    my_range = get_range_value(img)


# In[ ]:


my_range = get_range_value(img)
data = np.zeros((my_range[0],my_range[1],my_range[2],1))
print(data.shape)
print(my_range)

for i in range(my_range[0]):
    for j in range(my_range[1]):
        for k in range(my_range[2]):
            data[i:,j:,k:,0] = img[i:(i+1)*27,j:(j+1)*27:,k:(k+1)*27]


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 

# In[ ]:





# In[ ]:


img_reshape = img.reshape(-1, 512,512,1)
print(img_reshape.shape)


# Next, rescale the data with using max-min normalisation technique:

# In[ ]:


m = np.max(img_reshape)
mi = np.min(img_reshape)


# In[ ]:


m, mi


# In[ ]:


img_reshape = (img_reshape - mi) / (m - mi)


# Let's verify the minimum and maximum value of the data which should be 0.0 and 1.0 after rescaling it!

# In[ ]:


np.min(img_reshape), np.max(img_reshape)


# In[ ]:


temp = np.zeros([117,515, 515,1])


# In[ ]:


temp[:,3:,3:,:] = img_reshape


# In[ ]:


img_reshape = temp


# In[ ]:


from sklearn.model_selection import train_test_split
train_X,valid_X,train_ground,valid_ground = train_test_split(img_reshape,
                                                             img_reshape,)


# In[ ]:


print(img_reshape.shape)


# In[ ]:


plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(train_X[0], (512,512))
plt.imshow(curr_img, cmap='gray')

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(valid_X[0], (512,512))
plt.imshow(curr_img, cmap='gray')


# In[ ]:


batch_size = 128
epochs = 50
inChannel = 1


# Show mask!

# In[ ]:


fig, (ax1) = plt.subplots(1,4,figsize = (12, 6))
for i in range(4):   
    ax1[i].imshow(all_masks_img[i][all_masks_img[i].shape[0]//2])
    ax1[i].set_title('Mask')


# In[ ]:


from skimage.util import montage
fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
ax1.imshow(montage(all_img_img[0]), cmap ='bone')
fig.savefig('ct_scan.png')


# Data Genarate 

# Work with  model

# In[ ]:


117//27


# from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
# from keras.layers import Dropout, Input, BatchNormalization
# from sklearn.metrics import confusion_matrix, accuracy_score
# from plotly.offline import iplot, init_notebook_mode
# from keras.losses import categorical_crossentropy
# from keras.optimizers import Adadelta
# import plotly.graph_objs as go
# from matplotlib.pyplot import cm
# from keras.models import Model
# import numpy as np
# import keras
# import h5py

# In[ ]:


from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
from plotly.offline import iplot, init_notebook_mode
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
import plotly.graph_objs as go
from matplotlib.pyplot import cm
from keras.models import Model
import numpy as np
## input layer
input_layer = Input((27, 27, 27, 3))
## convolutional layers
conv_layer1 = Conv3D(filters=64, kernel_size=(5, 5, 5), activation='relu')(input_layer)
## add max pooling to obtain the most imformatic features
pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer1)
conv_layer2 = Conv3D(filters=64, kernel_size=(5, 5, 5), activation='relu')(pooling_layer1)
conv_layer3 = Conv3D(filters=64, kernel_size=(5, 5, 5), activation='relu')(conv_layer2)
## create an MLP architecture with dense layers : 4096 -> 512 -> 10
## add dropouts to avoid overfitting / perform regularization
dense_layer1 = Dense(1350, activation='relu')(conv_layer3)
output_layer = Dense(160, activation='softmax')(dense_layer1)
## define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)
print(model.summary())


# In[ ]:


for i in range(0,117,27):
    modelll = Model(train_Xtrain_X[i], model_CNN(train_X[i]))


# In[ ]:


modelll = Model(input_img, model_CNN(input_img))


# In[ ]:


modelll.summary()


# In[ ]:


modelll.compile(loss='mean_squared_error', optimizer = RMSprop())


# In[ ]:


model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.1), metrics=['acc'])


# In[ ]:


#model.fit(all_img_img,all_img_img, batch_size=8, epochs=50, validation_split=0.2)

