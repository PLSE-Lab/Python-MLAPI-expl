#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from glob import glob
import os
print(os.listdir("../input/data"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_data = pd.read_csv('../input/data/Data_Entry_2017.csv')
df_annotations = pd.read_csv('../input/data/BBox_List_2017.csv')


# In[ ]:


df_data.head()


# In[ ]:


print(df_annotations.shape)
df_annotations.drop(['Unnamed: 6','Unnamed: 7','Unnamed: 8'], axis = 1, inplace= True)
df_annotations.head()


# In[ ]:


new_df = pd.merge(df_annotations, df_data , on= 'Image Index')


# In[ ]:


new_df.shape


# In[ ]:


all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('..', 'input/data', 'images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', new_df.shape[0])
new_df['path'] = new_df['Image Index'].map(all_image_paths.get)
#all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x: int(x[:-1]))
new_df.sample(3)


# In[ ]:


Original_annotations = new_df[['Bbox [x','y','w','h]']]
Original_annotations.head()


# In[ ]:


#We create a function, which reads an image, resizes it to 128 x128 dimensions and returns it.
import cv2
def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    return img

from tqdm import tqdm 
train_img = []
for img_path in tqdm(new_df['path'].values):
    train_img.append(read_img( img_path))
    


# In[ ]:


X = np.array(train_img, np.float32) / 255  


# In[ ]:


IMAGE_SIZE = 128
new_df['x0'] = new_df['Bbox [x'] *  IMAGE_SIZE /1024 #new_df['OriginalImage[Width']
new_df['y0'] = new_df['y'] *  IMAGE_SIZE / 1024 #new_df['Height]']
new_df['w0'] = new_df['w'] *  IMAGE_SIZE /1024 #new_df['OriginalImage[Width']
new_df['h0'] = new_df['h]'] *  IMAGE_SIZE /1024#new_df['Height]']
print(new_df['x0'][10])
print(new_df['y0'][10])
print(new_df['h0'][10])
print(new_df['w0'][10])


# In[ ]:


Y = new_df[['x0','y0','h0','w0']]


# In[ ]:


import matplotlib.pyplot as plt

import matplotlib.patches as patches

k=10
#img=train_images[k]
img = X[k]
x0 = new_df['x0'][k]
y0 = new_df['y0'][k]
w = new_df['h0'][k]
h = new_df['w0'][k]


# Locations of the bounding box bottom-left coordinates (x0,y0) and the width and height
print('\n x0 = {}'.format(new_df['x0'][k]))
print('\n y0 = {}'.format(new_df['y0'][k]))
print('\n w = {}'.format(new_df['h0'][k]))
print('\n h = {}'.format(new_df['w0'][k]))

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(img)

# Create a Rectangle patch
# x1-x0 is the width of the bounding box
# y1-y0 is the height of the bounding box
rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

ax.plot(x0, y0, 'o', color='b')  # top-left corner

ax.plot(x0+w, y0+h, '*', color='c' ) # bottom-right corner

plt.show()


# In[ ]:





# In[ ]:


# Width hyper parameter for MobileNet (0.25, 0.5, 0.75, 1.0). 
# Higher width means more accurate but slower
ALPHA = 1.0 

# MobileNet takes images of size 128*128*3 
IMAGE_SIZE = 128 

# Number of epochs
EPOCHS = 10  

# Depends on your GPU/CPU/RAM
BATCH_SIZE = 32 
from keras import Model

from keras.applications.mobilenet import MobileNet, preprocess_input
model = MobileNet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, alpha=ALPHA ,
                  weights = '../input/mobilenet-1-0-128-tf-no-toph5/mobilenet_1_0_128_tf_no_top.h5') 

# To freeze layers, except the new top layer, of course, which will be added below
for layer in model.layers:
    layer.trainable = False


# In[ ]:


from keras.layers import Conv2D , Reshape
# Add new top layer which is a conv layer of the same size as the previous layer so that only 4 coords of BBox can be output
x = model.layers[-1].output

# kernel size should be 3 for img size 96, 4 for img size 128, 5 for img size 160 etc.
x = Conv2D(4, kernel_size=4, name="coords")(x)

# These are the 4 predicted coordinates of one BBox
x = Reshape((4,))(x) 

model = Model(inputs=model.input, outputs=x)
model.summary()


# In[ ]:



import tensorflow as tf
from keras.utils import Sequence
from keras.backend import epsilon
def loss(gt,pred):
    intersections = 0
    unions = 0
    diff_width = np.minimum(gt[:,0] + gt[:,2], pred[:,0] + pred[:,2]) - np.maximum(gt[:,0], pred[:,0])
    diff_height = np.minimum(gt[:,1] + gt[:,3], pred[:,1] + pred[:,3]) - np.maximum(gt[:,1], pred[:,1])
    intersection = diff_width * diff_height
    
    # Compute union
    area_gt = gt[:,2] * gt[:,3]
    area_pred = pred[:,2] * pred[:,3]
    union = area_gt + area_pred - intersection

#     Compute intersection and union over multiple boxes
    for j, _ in enumerate(union):
        if union[j] > 0 and intersection[j] > 0 and union[j] >= intersection[j]:
            intersections += intersection[j]
            unions += union[j]

    # Compute IOU. Use epsilon to prevent division by zero
    iou = np.round(intersections / (unions + epsilon()), 4)
    iou = iou.astype(np.float32)
    return iou

def IoU(y_true, y_pred):
    iou = tf.py_func(loss, [y_true, y_pred], tf.float32)
    return iou


# In[ ]:


# Regression loss is MSE
model.compile(optimizer='Adam', loss='mse', metrics=[IoU]) 


# Checkpoint best validation model
#checkpoint = ModelCheckpoint("MobileNetmodel-{val_iou:.2f}.h5", verbose=1, save_best_only=True, save_weights_only=True, mode="max", period=1) 
# Stop early, if the validation error deteriorates
#stop = EarlyStopping(monitor="val_iou", patience=PATIENCE, mode="max") 
# Reduce learning rate if Validation IOU does not improve
#reduce_lr = ReduceLROnPlateau(monitor="val_iou", factor=0.2, patience=10, min_lr=1e-7, verbose=1, mode="max")
#callback=[checkpoint, stop, reduce_lr]
#history = model.fit(x=train_images, y=train_gt, validation_data=val_data, epochs=10, batch_size = 32, verbose=1, callbacks=callback)


history = model.fit(x= X , y=Y, epochs= 150, batch_size = 132, verbose=1)


# In[ ]:


model_json = model.to_json()
with open("multi_disease_model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("multi_disease_model_weight.h5")
print("Saved model to disk")


# In[ ]:


import matplotlib.pyplot as plt
results=history
plt.figure(figsize=(8, 8))
plt.title("Learning Curve")
plt.plot(results.history["loss"], label="loss")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();


# In[ ]:


# Pick a test image, run model, show image, and show predicted bounding box overlaid on the image
import cv2
filename = new_df['path'][1]
unscaled = cv2.imread(filename) # Original image for display


# In[ ]:


image_height, image_width, _ = unscaled.shape

# Rescale image to run the network
#image = cv2.resize(unscaled, (IMAGE_SIZE, IMAGE_SIZE)) 
image = cv2.resize(X[0], (IMAGE_SIZE, IMAGE_SIZE)) 
#feat_scaled = preprocess_input(np.array(image, dtype=np.float32))


# In[ ]:


# Predict the BBox
region = model.predict(x=np.array([image]))[0]
region


# In[ ]:


# Scale the BBox
# x0,y0 is the scaled co-ordinates of the top-left corner of the bounding box
x0 = float(region[0] * image_width / IMAGE_SIZE)   
y0 = float(region[1] * image_height / IMAGE_SIZE)

# x1,y1 is the scaled co-ordinates of the bottom-right corner of the bounding box
w0 = float(region[2] * image_width / IMAGE_SIZE)
h0 = float((region[3]) * image_height / IMAGE_SIZE)
print(x0,y0,w0,h0)


# In[ ]:



import matplotlib.pyplot as plt
import matplotlib.patches as patches


# In[ ]:


# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(X[0])

# Create a Rectangle patch
# x1-x0 is the width of the bounding box
# y1-y0 is the height of the bounding box
#rect = patches.Rectangle((x0, y0), w0, h0, linewidth=2, edgecolor='r', facecolor='none')

# Add the patch to the Axes
#ax.add_patch(rect)

# Image coordinates - top-left of the image is (0,0)

#ax.plot(x0, y0, 'o', color='b') # top-left of the bounding box
#ax.plot(x0+w0, y0+h0, '*', color='c') # bottom-right of the bounding-box

#plt.show()
plt.savefig('testplot.png')


# In[ ]:


new_df.head()


# In[ ]:


new_df['Finding Label'].unique()


# In[ ]:


sample_df = new_df.head(100)


# In[ ]:


for i in range(sample_df.shape[0]):
    org_x0 = Y.iloc[i][0]
    org_y0 = Y.iloc[i][1]
    org_h0 = Y.iloc[i][2]
    org_w0 = Y.iloc[i][3]
    ###########################################
    image = cv2.resize(X[i], (IMAGE_SIZE, IMAGE_SIZE)) 
    region = model.predict(x=np.array([image]))[0]
    x0 =region[0]  
    y0 = region[1]
    h0 = region[2]
    w0 = region[3]
    #####################################################
        # Display the image
    fig,ax = plt.subplots(1)
    ax.imshow(X[i])

    # Create a Rectangle patch
    # x1-x0 is the width of the bounding box
    # y1-y0 is the height of the bounding box
    rect_pred = patches.Rectangle((x0, y0), w0, h0, linewidth=2, edgecolor='r', facecolor='none')
    rect_org = patches.Rectangle((org_x0, org_y0), org_w0, org_h0, linewidth=2, edgecolor='b', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect_pred)
    ax.add_patch(rect_org)
    # Image coordinates - top-left of the image is (0,0)

    ax.plot(x0, y0, 'o', color='b') # top-left of the bounding box
    ax.plot(x0+w0, y0+h0, '*', color='c') # bottom-right of the bounding-box
    ax.set_title(sample_df['Finding Label'][i])
    plt.show()
    print(Y.iloc[i])
    print(region)
    fig.savefig('prediction'+sample_df['Image Index'][i])


# In[ ]:




