#!/usr/bin/env python
# coding: utf-8

# In this kernel we will explore the concept of Test Time Augmentation (TTA) and will run an experiment on **Dogs vs. Cats** competition. Compare the results with and without TTA.

# ## What is Test Time Augmentation ?

# TTA is simply to apply different transformations to **test** image like: rotations, flipping and translations. Then feed these different transformed images to the trained model and average the results to get more confident answer. For example, the image shown below applies two transformations (Left-Right flipping and Contrast change) together with the original image. All of these images are passed to the same model and the results are averaged.

# ![pipeline](https://preview.ibb.co/kH61v0/pipeline.png)

# [Edafa](https://github.com/andrewekhalel/edafa) is a ready-to-use package for TTA which can be used directly as we will show in the example

# ## A proof of concept on Dogs vs. Cats competition

# In[ ]:


# important dependencies
from sklearn.model_selection import train_test_split
import keras
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from keras.preprocessing import image
from keras.layers import Flatten,Dropout,Dense
from keras.optimizers import Adam,SGD
from keras import Model
import numpy as np
import random
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Let's explore the dataset!

# In[ ]:


TRAIN_DIR = '../input/train/train/'
print ('This is a sample of the dataset file names')
print( os.listdir(TRAIN_DIR)[:5])


# In[ ]:


sample = plt.imread(os.path.join(TRAIN_DIR,random.choice(os.listdir(TRAIN_DIR))))
print ('Visualize a sample of the image')
print ('Image shape:',sample.shape)
plt.imshow(sample)


# Split data into train and validation sets (70% and 30% respectively)

# In[ ]:


f_train, f_valid = train_test_split(os.listdir(TRAIN_DIR), test_size=0.7, random_state=42)


# In[ ]:


# Network input size
PATCH_DIM = 32


# ### Build data generator that reads batch by batch from disk when needed

# In[ ]:


# src: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, files, batch_size=32, dim=(224,224), n_channels=3,n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.files = files
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = np.random.choice(len(self.files), self.batch_size)
        
        # Find files of IDs
        batch_files = self.files[indexes]

        # Generate data
        X, y = self.__data_generation(batch_files)

        return X, y

    def __data_generation(self, batch_files):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, f in enumerate(batch_files):
            # Store sample
            img_path = os.path.join(TRAIN_DIR,f)
            img = image.load_img(img_path, target_size=self.dim)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            x = np.squeeze(x)
            X[i,:,:,:] = x

            # Store class
            if 'dog' in f:
                y[i]=1
            else:
                y[i]=0
                
        return X, y


# In[ ]:


training_generator = DataGenerator(np.array(f_train),dim=(PATCH_DIM,PATCH_DIM))


# ### Build and train the model
# Our model is reusing VGG16 architecture without the fully connected layers. So we used the weights from imagenet and add our head as shown

# In[ ]:


initial_model = VGG16(weights="imagenet", include_top=False ,input_shape = (PATCH_DIM,PATCH_DIM,3))
last = initial_model.output

x = Flatten()(last)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.4)(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(initial_model.input, preds)
model.compile(Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])


# Now we train the model

# In[ ]:


model.fit_generator(generator=training_generator,
                    use_multiprocessing=True,
                    epochs=4,
                    workers=8)


# ### Read validation images and evaluate the model

# In[ ]:


X_val = np.empty((len(f_valid), PATCH_DIM, PATCH_DIM ,3))
y_val = np.empty((len(f_valid)), dtype=int)

for i, f in enumerate(f_valid):
    # Store sample
    img_path = os.path.join(TRAIN_DIR,f)
    img = image.load_img(img_path, target_size=(PATCH_DIM,PATCH_DIM))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = np.squeeze(x)
    X_val[i,:,:,:] = x

    # Store class
    if 'dog' in f:
        y_val[i]=1
    else:
        y_val[i]=0


# In[ ]:


y_pred = model.predict(X_val)


# In[ ]:


y_pred = [(y[0]>=0.5).astype(np.uint8) for y in y_pred]


# In[ ]:


print('Accuracy without TTA:',np.mean((y_val==y_pred)))


# ### Now we use edafa (TTA package)

# Step 1: Import the predictor suitable for your problem (`ClassPredictor` for Classification and `SegPredictor` for Segmentation)

# In[ ]:


from edafa import ClassPredictor


# Step 2: Inherit predictor class and implement the main function `predict_patches(self,patches)`

# In[ ]:


class myPredictor(ClassPredictor):
    def __init__(self,model,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = model

    def predict_patches(self,patches):
        return self.model.predict(patches)


# Step 3: Instantiate your class with configuration and whatever parameters needed

# In[ ]:


# use orignal image and flipped Left-Right images
# use arithmetic mean for averaging
conf = '{"augs":["NO",                "FLIP_LR"],        "mean":"ARITH"}'


# In[ ]:


p = myPredictor(model,conf)


# Step 4: Predict images

# In[ ]:


y_pred_aug = p.predict_images(X_val)


# In[ ]:


y_pred_aug = [(y[0]>=0.5).astype(np.uint8) for y in y_pred_aug ]


# In[ ]:


print('Accuracy with TTA:',np.mean((y_val==y_pred_aug)))


# ## Conclusion
# We can see that the accuracy improved using the exact same model. Thanks to TTA!
