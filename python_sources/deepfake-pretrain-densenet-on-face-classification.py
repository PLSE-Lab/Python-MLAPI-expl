#!/usr/bin/env python
# coding: utf-8

# # About this kernel
# 
# Since I noticed a lot of competitors is struggling to break the baseline submission (set everything to 0.51), I decided to create this kernel with the purpose of creating a pretrained model that you can easily finetune.
# 
# It uses a simple DenseNet-121, and is trained on around 200k real faces, and 200k GAN-generated faces. The fake faces are kindly [provided by Bojan in this discussion](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/121173), and were downsampled to 224 x 224, whereas the real faces come from the CelebA dataset, provided by Jessica [here](https://www.kaggle.com/jessicali9530/celeba-dataset), this time upsampled to 224 x 224.
# 
# Obviously, the danger of using this model is that both datasets are very "different", so it is very likely the model is not learning facial features as much as underlying pixels distributions; so please use this with caution!

# In[ ]:


import numpy as np
import cv2
import pandas as pd
from tqdm.notebook import tqdm 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os


# In[ ]:


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# # Preprocessing

# ## Create fake filepaths dataframe

# In[ ]:


original_fake_paths = []

for dirname, _, filenames in tqdm(os.walk('/kaggle/input/1-million-fake-faces/')):
    for filename in filenames:
        original_fake_paths.append([os.path.join(dirname, filename), filename])


# ## First downsize all the images

# In[ ]:


save_dir = '/kaggle/tmp/fake/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# In[ ]:


fake_paths = [save_dir + filename for _, filename in original_fake_paths]


# In[ ]:


for path, filename in tqdm(original_fake_paths):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    cv2.imwrite(os.path.join(save_dir, filename), img)


# In[ ]:


train_fake_paths, test_fake_paths = train_test_split(fake_paths, test_size=20000, random_state=2019)

fake_train_df = pd.DataFrame(train_fake_paths, columns=['filename'])
fake_train_df['class'] = 'FAKE'

fake_test_df = pd.DataFrame(test_fake_paths, columns=['filename'])
fake_test_df['class'] = 'FAKE'


# ## Create real file paths dataframe

# In[ ]:


real_dir = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/'
eval_partition = pd.read_csv('/kaggle/input/celeba-dataset/list_eval_partition.csv')

eval_partition['filename'] = eval_partition.image_id.apply(lambda st: real_dir + st)
eval_partition['class'] = 'REAL'


# In[ ]:


real_train_df = eval_partition.query('partition in [0, 1]')[['filename', 'class']]
real_test_df = eval_partition.query('partition == 2')[['filename', 'class']]


# ## Combine both real and fake for dataframe

# In[ ]:


train_df = pd.concat([real_train_df, fake_train_df])
test_df = pd.concat([real_test_df, fake_test_df])


# # Generator

# In[ ]:


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_dataframe(
    train_df,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_dataframe(
    train_df,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    subset='validation'
)


# In[ ]:


datagen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary'
)


# # Modelling

# ## Load and freeze DenseNet

# In[ ]:


densenet = DenseNet121(
    weights='/kaggle/input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)

for layer in densenet.layers:
    layer.trainable = False


# ## Build Model

# In[ ]:


def build_model(densenet):
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.0005),
        metrics=['accuracy']
    )
    
    return model


# In[ ]:


model = build_model(densenet)
model.summary()


# ## Training Phase 1 - Only train top layers

# In[ ]:


checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

train_history_step1 = model.fit_generator(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[checkpoint],
    epochs=7
)


# ## Training Phase 2 - Unfreeze and train all

# In[ ]:


model.load_weights('model.h5')
for layer in model.layers:
    layer.trainable = True

train_history_step2 = model.fit_generator(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[checkpoint],
    epochs=3
)


# ## Eval

# In[ ]:


pd.DataFrame(train_history_step1.history).to_csv('history1.csv')
pd.DataFrame(train_history_step2.history).to_csv('history2.csv')

