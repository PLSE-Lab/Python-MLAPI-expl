#!/usr/bin/env python
# coding: utf-8

# **In this kernel we will use the dataset in a special way to get better performance and lower loss without any data augmentation.**
# 
# * I have read in one of the forums that the dataset is actually collected by merging 2 datasets together, the first one contains 7000+ samples with 8 features (4 keypoints) for each image, the second one contains 2000+ images that actually belongs to the first dataset but with 30 features (15 keypoints).
# 
# * Now the question is, how to handle this sneaky dataset to get better results and lower loss (without data augmentation).
# 
# * you can try 3 approaches for this one:
#  1. Drop any sample that doesn't contain the full 15 key points, in this approach you simply ignore the first dataset, you will get an even smaller dataset with 2140 samples, eventually after training and submitting, you will get almost 3.0 loss.
#  
#  2. Fill any missing point with the previous available one, in this approach you will end up with 7000+ samples, but most of the features are filled and not accurate, surprisingly this approach will get almost 2.4 loss which is better than the first one, a reasonable explanation for this result is providing the model with 5000 more samples with 4 accurate keypoints and 11 inaccurate filled keypoints lower the loss a bit.
#  
#  3. Enhance the 1st approach by using the ignored dataset (1st dataset) to train a separate model to predict only 4 key points. Why would we do that?, Obviously this model (four-keypoints model) will produce more accurate predictions for those specific key points as the training set contains 7000 samples with accurate labels rather than only 2000 samples (notice that those 4 keypoints are just subset of the 15 keypoints). In this case, we have 2 models, fifteen-keypoints model which produces 30-dim vector for each sample, and four_keypoints model 8-dim vector for each sample (which produces more accurate values for certain four key points), then you should replace the predictions of the four-keypoints model with the corresponding predictions of the fifteen-keypoints model. This approach will lower loss to almost 2.1, this simply because we got more accurate predictions for 8 features.
#  
# * I think with alittle bit of data augmentation after splitting the dataset you can get more decent loss, also i encourage you to take a look at Ole Gee's solution which got 1.28 loss and achieved the 1st place.
# 
# * The code is simple, concise and fully-commented. Feel free to ask for help or more info or more explanation in the comments, i will be more than happy to help.
# 
# * Finally if this kernel helps you somehow, kindly don't forget to leave a little upvote up there.
# 
# * Hope you enjoy.

# In[ ]:


import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Set some directories.
train_zip_path = '/kaggle/input/facial-keypoints-detection/training.zip'
test_zip_path = '/kaggle/input/facial-keypoints-detection/test.zip'
Id_table_path = '/kaggle/input/facial-keypoints-detection/IdLookupTable.csv'
sample_sub_path = '/kaggle/input/facial-keypoints-detection/SampleSubmission.csv'
extracted_files_path = '/kaggle/working'


# In[ ]:


#Unzip train csv file to 'extracted_files_path'.
import zipfile
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_files_path)
#Unzip test csv file to 'extracted_files_path'.
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_files_path)


# In[ ]:


#Read train csv file.
train_csv = pd.read_csv(extracted_files_path + '/training.csv')
#Read test csv file.
test_csv = pd.read_csv(extracted_files_path + '/test.csv')
#Read IdLookUpTable csv file.
looktable_csv = pd.read_csv(Id_table_path)


# In[ ]:


train_csv.info()


# As we can see, left_eye, right_eye, nose_tip and mouse_center_bottom_lip features are available for almost all images, rest of them are only available for 2000+ images.

# In[ ]:


feature_8 = ['left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x','right_eye_center_y',
            'nose_tip_x', 'nose_tip_y',
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y', 'Image']
#Create 2 different datasets.
train_8_csv = train_csv[feature_8].dropna().reset_index()
train_30_csv = train_csv.dropna().reset_index()


# In[ ]:


#7000 samples, 8 features.
train_8_csv.info()


# In[ ]:


#2410 samples, 30 features.
train_30_csv.info()


# In[ ]:


def str_to_array(pd_series):
    '''
    pd_series: a pandas series, contains img pixels as strings,
    each element is a long str (length = 96*96 = 9216),
    contains pixel values. eg:('29 34 122 244 12 ....').
    
    1- Convert str of pixel values to 2d array.
    2- Stack all arrays into one 3d array.
    
    returns 3d numpy array of shape of (num of images, 96, 96, 1).
    '''
    data_size = len(pd_series)
    #intialize output 3d array as numpy zeros array.
    X = np.zeros(shape=(data_size,96,96,1), dtype=np.float32)
    for i in tqdm(range(data_size)):
        img_str = pd_series[i]
        img_list = img_str.split(' ')
        img_array = np.array(img_list, dtype=np.float32)
        img_array = img_array.reshape(96,96,1)
        X[i] = img_array
    return X


# In[ ]:


#Wrap train data and labels into numpy arrays.
X_train_30 = str_to_array(train_30_csv['Image'])
labels_30 =  train_30_csv.drop(['index','Image'], axis=1)
y_train_30 = labels_30.to_numpy(dtype=np.float32)
print('X_train with 30 feature shape: ', X_train_30.shape)
print('y_train with 30 feature shape: ', y_train_30.shape)


# In[ ]:


#Wrap test data and labels into numpy arrays.
X_train_8 = str_to_array(train_8_csv['Image'])
labels_8 =  train_8_csv.drop(['index','Image'], axis=1)
y_train_8 = labels_8.to_numpy(dtype=np.float32)
print('X_train with 8 feature shape: ', X_train_8.shape)
print('y_train with 8 feature shape: ', y_train_8.shape)


# In[ ]:


def plot_face_pts(img, pts):
    plt.imshow(img[:,:,0], cmap='gray')
    for i in range(1,31,2):
        plt.plot(pts[i-1], pts[i], 'b.')


# In[ ]:


#Display samples of the dataset.
fig = plt.figure(figsize=(10, 7))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(12):
    ax = fig.add_subplot(3, 4, i + 1, xticks=[], yticks=[])
    plot_face_pts(X_train_30[i], y_train_30[i])

plt.show()


# In[ ]:


def create_model(output_n = 30):
    '''
    Create and compile a model with custom output layer.
    '''
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=[96,96,1]),

        keras.layers.Conv2D(filters=32, kernel_size=[5,5], padding='same', use_bias=False),
        keras.layers.LeakyReLU(alpha = .1),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=32, kernel_size=[5,5], padding='same', use_bias=False),
        keras.layers.LeakyReLU(alpha = .1),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=[2,2]),

        keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding='same', use_bias=False),
        keras.layers.LeakyReLU(alpha = .1),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding='same', use_bias=False),
        keras.layers.LeakyReLU(alpha = .1),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=[2,2]),

        keras.layers.Conv2D(filters=128, kernel_size=[3,3], padding='same', use_bias=False),
        keras.layers.LeakyReLU(alpha = .1),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=128, kernel_size=[3,3], padding='same', use_bias=False),
        keras.layers.LeakyReLU(alpha = .1),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=[2,2]),

        keras.layers.Conv2D(filters=256, kernel_size=[3,3], padding='same', use_bias=False),
        keras.layers.LeakyReLU(alpha = .1),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=[3,3], padding='same', use_bias=False),
        keras.layers.LeakyReLU(alpha = .1),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=[2,2]),

        keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding='same', use_bias=False),
        keras.layers.LeakyReLU(alpha = .1),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding='same', use_bias=False),
        keras.layers.LeakyReLU(alpha = .1),
        keras.layers.BatchNormalization(),    

        keras.layers.Flatten(),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dropout(.1),
        keras.layers.Dense(units=output_n),
    ])

    model.compile(optimizer = 'adam' , loss = "mean_squared_error", metrics=["mae"])
    return model


# In[ ]:


#Prepare 2 models to handle 2 different datasets.
model_30 = create_model(output_n=30)
model_8 = create_model(output_n=8)


# In[ ]:


#Prepare callbacks
LR_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=10, factor=.4, min_lr=.00001)
EarlyStop_callback = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)


# In[ ]:


#Train the model with 30 features.
history = model_30.fit(X_train_30, y_train_30, validation_split=.1, batch_size=64, epochs=100, callbacks=[LR_callback,EarlyStop_callback])


# In[ ]:


# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['mae'], color='b', label="Training mae")
ax[1].plot(history.history['val_mae'], color='r',label="Validation mae")
legend = ax[1].legend(loc='best', shadow=True)


# In[ ]:


#Train the model with 8 features.
history = model_8.fit(X_train_8, y_train_8, validation_split=.1, batch_size=64, epochs=100, callbacks=[LR_callback,EarlyStop_callback])


# In[ ]:


# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['mae'], color='b', label="Training mae")
ax[1].plot(history.history['val_mae'], color='r',label="Validation mae")
legend = ax[1].legend(loc='best', shadow=True)


# In[ ]:


#Wrap test images into 3d array.
X_test = str_to_array(test_csv['Image'])
print('X_test shape: ', X_test.shape)


# In[ ]:


#Pridect points for each image using 2 different model.
y_hat_30 = model_30.predict(X_test) 
y_hat_8 = model_8.predict(X_test)
print('Predictions shape', y_hat_30.shape)
print('Predictions shape', y_hat_8.shape)


# Merge 2 predictions arrya into one by replacing each column in y_hat_8 with the corresponding column in y_hat_30.

# In[ ]:


feature_8_ind = [0, 1, 2, 3, 20, 21, 28, 29]
#Merge 2 prediction from y_hat_30 and y_hat_8.
for i in range(8):
    print('Copy "{}" feature column from y_hat_8 --> y_hat_30'.format(feature_8[i]))
    y_hat_30[:,feature_8_ind[i]] = y_hat_8[:,i]


# In[ ]:


#Display samples of the dataset.
fig = plt.figure(figsize=(10, 7))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i, f in enumerate(range(39,45)):
    ax = fig.add_subplot(2, 3, i + 1, xticks=[], yticks=[])
    plot_face_pts(X_test[f], y_hat_30[f])

plt.show()


# After merging predictions, Kaggle grader doesn't require all of them, so we gotta filter them out by a look_up table provided by kaggle.

# In[ ]:


#All required features in order.
required_features = list(looktable_csv['FeatureName'])
#All images nmber in order.
imageID = list(looktable_csv['ImageId']-1)
#Generate Directory to map feature name 'Str' into int from 0 to 29.
feature_to_num = dict(zip(required_features[0:30], range(30)))


# In[ ]:


#Generate list of required features encoded into ints.
feature_ind = []
for f in required_features:
    feature_ind.append(feature_to_num[f])


# In[ ]:


#Pick only the required predictions from y_hat_30 (filteration).
required_pred = []
for x,y in zip(imageID,feature_ind):
    required_pred.append(y_hat_30[x, y])


# In[ ]:


#Submit
rowid = looktable_csv['RowId']
loc30 = pd.Series(required_pred,name = 'Location')
submission = pd.concat([rowid,loc30],axis = 1)
submission.to_csv('Merged_Predictions.csv',index = False)

