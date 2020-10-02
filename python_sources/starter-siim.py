#!/usr/bin/env python
# coding: utf-8

# <h1>Let's Get started</h1>
# 

# **What is given?**     
# Train images and test images which we have to download using Healthcare API. And train_rle.csv which have image id and corresponding rle. Beware all train image don't have rle.

# Importing the required libraries.

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import os
import pydicom
from tqdm import tqdm_notebook
from keras import backend as K
from sklearn.model_selection import train_test_split
import gc
from skimage.transform import resize
import PIL
from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout


# In[ ]:


img_size = 256


# In[ ]:


tr = pd.read_csv('../input/siim-train-test/siim/train-rle.csv')


# In[ ]:


tr.head()


# In[ ]:


#These are the functions provided by kaggle to convert a mask to rle and vice-versa.
import numpy as np

def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor: 
                if currentColor >= 127:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    
    mask= np.zeros(width* height)
    if rle == ' -1' or rle == '-1':
        return mask.reshape(width,height)
    array = np.asarray([int(x) for x in rle.split()])
    
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


# In[ ]:


plt.imshow(rle2mask(tr[' EncodedPixels'][5],1024,1024))
plt.show()


# <h4>Converting are rle into masks</h4>

# In[ ]:


def get_mask(encode,width,height):
    if encode == [] or encode == ' -1':
        return rle2mask(' -1',width,height)
    else:
        return rle2mask(encode[0],width,height)       


# <h4>Gettting image and corresponding mask</h4>
# I am using only 2000 train images due to time and memory contraint. Moreover I am checking everything works fine.

# In[ ]:


def image_n_encode(train_images_names,encode_df):
    train_imgs = [] 
    train_encode = []
    c = 0
    for f in tqdm_notebook(train_images_names):
        if c >= 2000:
            break
        try:
            img = pydicom.read_file(f).pixel_array
            c += 1
            encode = list(encode_df.loc[encode_df['ImageId'] == '.'.join(f.split('/')[-1].split('.')[:-1]),
                               ' EncodedPixels'].values)
            
            encode = get_mask(encode,img.shape[1],img.shape[0])
            encode = resize(encode,(img_size,img_size))
            train_encode.append(encode)
            img = resize(img,(img_size,img_size))
            train_imgs.append(img)
        except pydicom.errors.InvalidDicomError:
            print('come here')
        
    return train_imgs,train_encode
        
        
        


# Thanks to @seesee for uploading [this dataset.](https://www.kaggle.com/seesee/siim-train-test)

# In[ ]:


#getting path of all the train and test images
from glob import glob
train_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-train/*/*/*.dcm'))
test_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-test/*/*/*.dcm'))

print(len(train_fns))
print(len(test_fns))


# In[ ]:


images,mask_e = image_n_encode(train_fns,tr)


# In[ ]:


print(len(images),len(mask_e))


# In[ ]:


plt.imshow(images[102],cmap = 'gray')
plt.show()
plt.imshow(mask_e[102])
plt.show()


# In[ ]:


#Evaluation metric
#ref https://www.kaggle.com/jesperdramsch/intro-chest-xray-dicom-viz-u-nets-full-data
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# <h3>Building our model</h3>

# In[ ]:


def build_model(input_layer, start_neurons):
    #ref: https://www.kaggle.com/phoenigs/u-net-dropout-augmentation-stratification
    # 128 -> 64
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    # 64 -> 32
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    # 32 -> 16
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    # 16 -> 8
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

    # 8 -> 16
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

    #uconv1 = Dropout(0.5)(uconv1)
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer

input_layer = Input((img_size, img_size, 1))
output_layer = build_model(input_layer, 16)


# In[ ]:


model = Model(input_layer, output_layer)


# In[ ]:


model_checkpoint = ModelCheckpoint("./unet_best1.model", 
                                   mode = 'max', save_best_only=True, verbose=1)


# In[ ]:


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[dice_coef])


# In[ ]:


model.summary()


# In[ ]:


model.fit(np.array(images).reshape(-1,img_size,img_size,1),np.array(mask_e).reshape(-1,img_size,img_size,1),validation_split = 0.1,
          epochs = 1,batch_size = 16,
         callbacks = [model_checkpoint])


# In[ ]:


del images,mask_e


# In[ ]:


gc.collect()


# <h4> Predicting the mask and converting it into rle </h4>

# def test_images_pred(test_fns):
#     pred_rle = []
#     ids = []
#     for f in tqdm_notebook(test_fns):
#         img = pydicom.read_file(f).pixel_array
#         img = resize(img,(img_size,img_size))
#         img = model.predict(img.reshape(1,img_size,img_size,1))
#         
#         img = img.reshape(img_size,img_size)
#         ids.append('.'.join(f.split('/')[-1].split('.')[:-1]))
#         #img = PIL.Image.fromarray(((img.T*255).astype(np.uint8)).resize(1024,1024))
#         img = PIL.Image.fromarray((img.T*255).astype(np.uint8)).resize((1024,1024))
#         img = np.asarray(img)
#         #print(img)
#         pred_rle.append(mask2rle(img,1024,1024))
#     return pred_rle,ids
#         

# preds,ids = test_images_pred(test_fns)

# In[ ]:


print(preds[0])


# In[ ]:


#print(preds[10])
print(len(preds),len(ids))


# In[ ]:


submission = pd.DataFrame({'ImageId':ids,'EncodedPixels':preds})


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv',index = False)


# In[ ]:


from IPython.display import HTML
html = "<a href = unet_best1.model>d</a>"
HTML(html)


# <h4>Work in progress. I will update it.</h4>
# <h3>Any suggestion. Let me know in comments.</h3>
# 
