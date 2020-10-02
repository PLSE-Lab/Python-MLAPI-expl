#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('jupyter nbconvert --execute --to markdown __notebook_source__.ipynb')


# In[ ]:


get_ipython().system('pip freeze')


# In[ ]:


import json
import math
import os
import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy
import tensorflow as tf
from tqdm import tqdm
IMG_SIZE = 224

get_ipython().run_line_magic('matplotlib', 'inline')


# # Set random seed for reproducibility.

# In[ ]:




np.random.seed(2019)
tf.random.set_random_seed(2019)


# # Loading & Exploration

# In[ ]:


train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
print(train_df.shape)
print(test_df.shape)
train_df.head()


# In[ ]:


train_df['diagnosis'].hist()
train_df['diagnosis'].value_counts()


# # Displaying some Sample Images

# In[ ]:


def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(4*columns, 3*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'id_code']
        image_id = df.loc[i,'diagnosis']
        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(train_df)


# # Image Preprocessing using ben's idea
# 
# 
# We will resize the images to 224x224, then create a single numpy array to hold the data.

# In[ ]:


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
    
    
def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image


# In[ ]:


def preprocess_image(image_path, desired_size=IMG_SIZE):
    im = load_ben_color(image_path,sigmaX = 30)
    return im


# # Converting Preprocessed images to Numpy arrays 

# In[ ]:


N = train_df.shape[0]
x_train = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(train_df['id_code'])):
    x_train[i, :, :, :] = preprocess_image(
        f'../input/aptos2019-blindness-detection/train_images/{image_id}.png'
    )


# In[ ]:


N = test_df.shape[0]
x_test = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(test_df['id_code'])):
    x_test[i, :, :, :] = preprocess_image(
        f'../input/aptos2019-blindness-detection/test_images/{image_id}.png'
    )


# # Shapes of data

# In[ ]:


y_train = pd.get_dummies(train_df['diagnosis']).values

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)


# # Splitting into train and validations

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, 
    test_size=0.15, 
    random_state=2019
)


# # Data Augmentation 

# In[ ]:


BATCH_SIZE = 32

def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.15,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )

# Using original generator
data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2019)
# Using Mixup
#mixup_generator = MixupGenerator(x_train, y_train, batch_size=BATCH_SIZE, alpha=0.2, datagen=create_datagen())()


# In[ ]:


plt.imshow(x_train[5], interpolation='nearest')


# # Model: DenseNet-121

# In[ ]:


densenet = DenseNet121(
    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(IMG_SIZE,IMG_SIZE,3)
)


# # Building the model

# In[ ]:


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))
    
    model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=0.00005),
    metrics=['accuracy']
        )
    
    return model


# In[ ]:


model = build_model()
model.summary()


# # Training & Evaluation

# In[ ]:


#appa_metrics = Metrics()

history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=15,
    validation_data=(x_val, y_val),
    
)


# # Saving the model as CNN.model

# In[ ]:


model_json = model.to_json()
with open("model_json", "w") as json_file :
    json_file.write(model_json)


# In[ ]:


model.save_weights("model.h5")
print("model saved to disk")
model.save('CNNmodel.h5')


# In[ ]:


import joblib
joblib.dump(model,'JOBmodel.sav')


# In[ ]:


Jmodel = joblib.load('JOBmodel.sav')


# In[ ]:


from IPython.display import FileLink
FileLink(r'CNNmodel.h5')


# # Plotting accuracy curves of both train & validation

# In[ ]:


print(history.history.keys())
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')


# In[ ]:


loadedModel = pickle.load(open(filename,'rb'))


# In[ ]:


sample_test_images = []
for i in range(10):
    img_path = test_df.loc[i,'id_code']
    img = preprocess_image(f'../input/aptos2019-blindness-detection/test_images/{img_path}.png')
    sample_test_images.append(img)
    
    


# # Predicting a single image

# In[ ]:


image_path=f'../input/aptos2019-blindness-detection/test_images/0299d97f31f7.png'
img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
plt.imshow(img)
img = preprocess_image(image_path)
img = np.expand_dims(img, axis=0)
result=loadedModel.predict_classes(img)
plt.title(label = result)
plt.show()
print("the diabetic retinopathy level of this eye is {}".format(result))


# In[ ]:




