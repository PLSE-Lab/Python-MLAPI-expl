#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from imgaug import augmenters as iaa
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from glob import glob
from tqdm import tqdm
from sklearn.utils import shuffle
from itertools import chain
from collections import Counter

import warnings
warnings.filterwarnings("ignore")
print(os.listdir('../input'))


# In[ ]:


# From https://www.proteinatlas.org/download/subcellular_location.tsv.zip
subcellular_location = pd.read_csv('../input/external-data-for-protein-atlas/subcellular_location.tsv', sep="\t", index_col = None)

# Get urls
urls = []
for name in subcellular_location[['Gene', 'Gene name']].values:  
    name = '-'.join(name)
    url = ('https://www.proteinatlas.org/'+name+'/antibody#ICC')
    urls.append(url)
    
def get_html(url):
    response = requests.get(url)
    return response.text
    
def load_img(url):
    html = get_html(url)
    soup = BeautifulSoup(html, 'lxml')
    links = []
    for a in soup.findAll('a', {'class':'colorbox'}, href=True):
        if '_selected' in  a['href']:
            links.append(''.join(('https://www.proteinatlas.org'+a['href']).split('_medium')))
    i = 0
    for link in set(links):
        try:
            name = url.split('/')[-2]        
            response = requests.get(link)
            img = Image.open(BytesIO(response.content))
            if np.array(img)[:,:,0].mean()<70:
                img.save('external_data/'+name+'_'+str(i)+'.png')
                i+=1
        except:
            pass


# In[ ]:


subcellular_location.head()


# In[ ]:


# Pages with images
urls[:10]


# In[ ]:


label_names = {
    0:  "Nucleoplasm", 
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center" ,  
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",
    8:  "Peroxisomes",
    9:  "Endosomes",
    10:  "Lysosomes",
    11:  "Intermediate filaments",   
    12:  "Actin filaments",
    13:  "Focal adhesion sites",   
    14:  "Microtubules",
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",
    18:  "Microtubule organizing center",  
    19:  "Centrosome",
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions", 
    23:  "Mitochondria",
    24:  "Aggresome",
    25:  "Cytosol",
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}

all_label_names = {
    0:  "Nucleoplasm", 
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center" ,  
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",
    8:  "Peroxisomes",
    9:  "Endosomes",
    10:  "Lysosomes",
    11:  "Intermediate filaments",   
    12:  "Actin filaments",
    13:  "Focal adhesion sites",   
    14:  "Microtubules",
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",
    18:  "Microtubule organizing center",  
    19:  "Centrosome",
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions", 
    23:  "Mitochondria",
    24:  "Aggresome",
    25:  "Cytosol",
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings",
    # new classes          
    28: "Vesicles",
    29: "Nucleus",
    30: "Midbody",
    31: "Cell Junctions",
    32: "Midbody ring",
    33: "Cleavage furrow"
}


# In[ ]:


all_names = []
for j in tqdm(range(len(subcellular_location))):
    names = np.array(subcellular_location[['Enhanced', 'Supported', 'Approved', 'Uncertain']].values[j])
    names = [name for name in names if str(name) != 'nan']
    split_names = []
    for i in range(len(names)):
        split_names = split_names + (names[i].split(';'))
    all_names.append(split_names)
subcellular_location['names'] = all_names


# In[ ]:


img = Image.open(glob('../input/external-data-for-protein-atlas/external_data/*')[1])
print(img.size)
img


# In[ ]:


# Only old names
data_list = []
for i in tqdm(range(len(subcellular_location))):
    im_name  = subcellular_location['Gene'].values[i]+'-'+subcellular_location['Gene name'].values[i]
    for im in glob('../input/external-data-for-protein-atlas/external_data/'+im_name+'*'):
        labels = []
        for name in subcellular_location['names'].values[i]:
            try:
                if name == 'Rods & Rings': name = "Rods & rings"
                labels.append(list(label_names.values()).index(name))          
            except:
                pass
        if len(labels)>0:
            data_list.append([im.split('/')[-1].split('.png')[0], subcellular_location['names'].values[i], labels])


# In[ ]:


data = pd.DataFrame(data_list, columns = ['Id', 'Names', 'Target'])
data.head()


# In[ ]:


len(data)


# In[ ]:


SIZE = 604
epochs = 16
batch_size = 16


# In[ ]:


# Load dataset info
path_to_train = '../input/external-data-for-protein-atlas/external_data/'

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target']): 
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)
train_dataset_info[0]['path']


# In[ ]:


# Create generator
class data_generator:
    
    def create_train(dataset_info, batch_size, shape, augument=True):
        while True:
            dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = dataset_info[start:end]
                batch_labels = np.zeros((len(X_train_batch), 28))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(
                        X_train_batch[i]['path'], shape)  
                    if augument:
                        image = data_generator.augment(image)
                    batch_images.append(image/255.)
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                try:
                    yield np.array(batch_images, np.float32), batch_labels
                except:
                    pass
                
    def load_image(path, shape):
        image_rgb = Image.open(path+'.png')
        image_rgba = image_rgb.convert('RGBA')
        image = cv2.resize(np.array(image_rgba), (shape[0], shape[1]))
        return image

    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)
        image_aug = augment_img.augment_image(image)
        return image_aug


# In[ ]:


from sklearn.model_selection import train_test_split

# Split data into train, valid
indexes = np.arange(train_dataset_info.shape[0])
np.random.shuffle(indexes)
train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=21)

# Create train and valid data generetors
train_generator = data_generator.create_train(
    train_dataset_info[train_indexes], batch_size, (SIZE, SIZE, 4), augument=True)
validation_generator = data_generator.create_train(
    train_dataset_info[valid_indexes], 32, (SIZE, SIZE, 4), augument=False)


# In[ ]:


import keras
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.applications.xception import Xception
from keras import metrics
from keras.optimizers import Adam 

# Model    
def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    
    ### For External
    base_model = Xception(include_top=False,
                   weights='imagenet', 
                   input_shape=(299, 299, 3),
                   pooling='avg')
    x = BatchNormalization()(input_tensor)
    x = Conv2D(32, kernel_size=(3,3), activation='relu')(x) 
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=(3,3), activation='relu')(x) 
    x = BatchNormalization()(x)
    x = Conv2D(3, kernel_size=(3,3), activation='relu')(x) 
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = base_model(x) 
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model


# In[ ]:


# Train external and save weight, then create new model for real data and load external weight.

model = create_model(
    input_shape=(SIZE, SIZE, 4), 
    n_out=28)

# Train all layers
for layer in model.layers:
    layer.trainable = True
    
model.summary()
    
model.compile(loss='binary_crossentropy',
            optimizer=Adam(lr=1e-4),
            metrics=['accuracy'])


# In[ ]:


from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='external_weights.h5', verbose=1, save_best_only=True)
model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
    validation_data=validation_generator,
    validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
    epochs=epochs, verbose=1, callbacks=[checkpointer])


# In[ ]:




