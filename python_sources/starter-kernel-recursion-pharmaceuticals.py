#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import imageio
from tensorflow import keras
from tqdm import tqdm
import glob
import os
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# **Welcome to Recursion Cellular Image Classification competition**.
# <br>This starter kernel will guide you though our data and show you how to train a basic model.

# ## Loading the metadata

# ![](http://)Let's start by reading train.csv:

# In[ ]:


BASE_DIR = '../input'


# In[ ]:


df_train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
df_train.head()


# In[ ]:


df_train.nunique()


# * As you can see, our train data is composed of 36515 samples from 33 different experiments
# * In each experiments we had 4 different plates, each plate contained 384 wells, not all of them are in the dataset
# * Each well has **2 samples/sites**, both marked with a single siRNA

# We want the starter kernel to run fast, hence, we sample only 5000 items from the dataset. When Training your model, we recommend using all samples.

# ## Pixel metadata

# 1. We also provided statistics on all the images. This information will allow you to normalize the data, for example by reducing the mean and deviding by the standard deviation.

# In[ ]:


df_pixel_stats = pd.read_csv(os.path.join(BASE_DIR, 'pixel_stats.csv')).set_index(['id_code','site', 'channel'])
df_pixel_stats.head()


# ## Converting format

# On Kaggle Kernels we provided you with extracted png images, each representing a layer. If you try to load it to memory, you might encounter some IO bollteneck. This script will convert the pngs to numpy array, each representing a single sample. in the GCS extended dataset you will be able to find tfrecords files, which are a fast way to load data to Tensorflow models.

# In[ ]:


OUTPUT_DIR = '../output'


# In[ ]:


DATA_PATH_FORMAT = os.path.join(BASE_DIR, 'train/{experiment}/Plate{plate}/{well}_s{sample}_w{channel}.png')

def transform_image(sample_data, pixel_data):
    x=[]
    for channel in [1,2,3,4,5,6]:
        impath = DATA_PATH_FORMAT.format(experiment=sample.experiment,
                                        plate=sample_data.plate,
                                        well=sample_data.well,
                                        sample=1,# For demo only, we use sample=1, you can use also sample=2
                                        channel=channel)
        # normalize the channel
        img = np.array(imageio.imread(impath)).astype(np.float64)
        img -= pixel_data.loc[channel]['mean']
        img /= pixel_data.loc[channel]['std']
        img *= 255 # To keep MSB
        
        x.append(img)

    return np.stack(x).T.astype(np.byte)


# In[ ]:


get_ipython().system('mkdir -p {OUTPUT_DIR}/np_arrays/train/')


# In[ ]:


for _, sample in tqdm(df_train.iterrows(), total=len(df_train)):
    pixel_data = df_pixel_stats.loc[sample.id_code, 1, :].reset_index().set_index('channel')
    x = transform_image(sample, pixel_data)
    np.save(os.path.join(OUTPUT_DIR, 'np_arrays/train/{sample_id}.npy').format(sample_id=sample.id_code), x)


# ## Controls

# In each experiment, the same 30 siRNAs appear on every plate as positive controls. In addition, there is one well per plate with untreated cells as a negative control. It has the same schema as [train/test].csv, plus a well_type field denoting the type of control.
# 

# ## Train a simple model

# In[ ]:


# base sample : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, df, mode='train', batch_size=32, dim=(512,512), n_channels=6,  shuffle=True):
        self.df = df
        self.dim = dim
        self.batch_size = batch_size
        self.mode = mode
        if mode == 'train':
            self.labels = self.df['sirna'].tolist()
            self.n_classes = self.df['sirna'].nunique()
        self.list_IDs = self.df.index.tolist()
        self.n_channels = n_channels
    
        self.shuffle = shuffle
        if mode == 'train':
            self.npy_data_format =  os.path.join(OUTPUT_DIR,'np_arrays/train/{sample_id}.npy')
        elif mode == 'test':
            self.npy_data_format = os.path.join(OUTPUT_DIR,'np_arrays/test/{sample_id}.npy')
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Store sample
            sample_data = self.df.loc[ID]
            X[i,] = np.load(self.npy_data_format.format(sample_id=sample_data.id_code))                                            .astype(np.float32) / 255.0
            if self.mode == 'train':
            # Store class
                y[i] = sample_data.sirna
            else:
                y[i] = 0
          
        if self.mode == 'train':
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        else:
            return X, y


# In[ ]:


# Generators
training_generator = DataGenerator(df=df_train, shuffle=True)


# In[ ]:


n_classes = df_train['sirna'].nunique()

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same',
                 input_shape=(512, 512, 6), activation='relu'),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(n_classes, activation='softmax')
])


# In[ ]:


model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


# Train model on dataset
model.fit_generator(epochs=1, 
                    generator=training_generator,
                    use_multiprocessing=True,
                    workers=2)


# ## Predict 

# In[ ]:


# Clean train data
get_ipython().system('rm -rf {OUTPUT_DIR}/np_arrays/train/')
get_ipython().system('mkdir -p {OUTPUT_DIR}/np_arrays/test/')


# In[ ]:


df_test = pd.read_csv( os.path.join(BASE_DIR, 'test.csv'))


# In[ ]:


DATA_PATH_FORMAT = os.path.join(BASE_DIR, 'test/{experiment}/Plate{plate}/{well}_s{sample}_w{channel}.png')

for _, sample in tqdm(df_test.iterrows(), total=len(df_test)):
    pixel_data = df_pixel_stats.loc[sample.id_code, 1, :].reset_index().set_index('channel')
    x = transform_image(sample, pixel_data)
    np.save(os.path.join(OUTPUT_DIR, 'np_arrays/test/{sample_id}.npy').format(sample_id=sample.id_code), x)


# In[ ]:


test_generator = DataGenerator(df=df_test, mode='test', shuffle=False)


# In[ ]:


predictions = model.predict_generator(test_generator, steps=1000, verbose=1)


# In[ ]:


get_ipython().system('rm -rf {OUTPUT_DIR}/np_arrays/test/')


# In[ ]:


classes = predictions.argmax(axis=1)

