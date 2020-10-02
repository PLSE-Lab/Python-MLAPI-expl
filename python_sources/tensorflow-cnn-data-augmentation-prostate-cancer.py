#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction + Set-up
# 
# TensorFlow is a powerful tool to develop any machine learning pipeline, and today we will go over how to load Image+CSV combined datasets, how to use Keras preprocessing layers for image augmentation, and how to use pre-trained models for image classification.
# 
# Skeleton code for the DataGenerator Sequence subclass is credited to [Xie29's NB](https://www.kaggle.com/xiejialun/panda-tiles-training-on-tensorflow-0-7-cv).
# 
# Run the following cell to import the necessary packages. We will be using the GPU accelerator to efficiently train our model. Remember to change the accelerator on the right to GPU. We won't be using a TPU for this notebook because data generators are not safe to run on multiple replicas. If a TPU is not used, change the `TPU_used` variable to `False`.

# In[ ]:


get_ipython().system('pip install tensorflow==2.2.0  --quiet')
get_ipython().system('pip install tf-nightly --quiet')


# In[ ]:


import os
import PIL
import time
import math
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img

SEED = 1337
print('Tensorflow version : {}'.format(tf.__version__))

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)
    
print(tf.__version__)


# # 2. Data loading
# 
# The first step is to load in our data. The original PANDA dataset contains large images and masks that specify which area of the mask led to the ISUP grade (determines the severity of the cancer). Since the original images contain a lot of white space and extraneous data that is not necessary for our model, we will be using tiles to condense the images. Basically, the tiles are small sections of the masked areas, and these tiles can be concatenated together so the only the masked sections of the original image remains.

# In[ ]:


MAIN_DIR = '../input/prostate-cancer-grade-assessment'
TRAIN_IMG_DIR = '../input/panda-tiles/train'
TRAIN_MASKS_DIR = '../input/panda-tiles/masks'
train_csv = pd.read_csv(os.path.join(MAIN_DIR, 'train.csv'))


# Some of the images could not be converted to tiles because the masks were too small or the image was too noisy. We need to take these images out of our DataFrame so that we do not run into a `FileNotFoundError`.

# In[ ]:


valid_images = tf.io.gfile.glob(TRAIN_IMG_DIR + '/*_0.png')
img_ids = train_csv['image_id']


# In[ ]:


for img_id in img_ids:
    file_name = TRAIN_IMG_DIR + '/' + img_id + '_0.png'
    if file_name not in valid_images:
        train_csv = train_csv[train_csv['image_id'] != img_id]
        
radboud_csv = train_csv[train_csv['data_provider'] == 'radboud']
karolinska_csv = train_csv[train_csv['data_provider'] != 'radboud']


# We want both our training dataset and our validation dataset to contain images from both the Karolinska Institute and Radboud University Medical Center data providers. The following cell will split the each datafram into a 80:20 training:validation split.

# In[ ]:


r_train, r_test = train_test_split(
    radboud_csv,
    test_size=0.2, random_state=SEED
)

k_train, k_test = train_test_split(
    karolinska_csv,
    test_size=0.2, random_state=SEED
)


# Concatenate the dataframes from the two different providers and we have our training dataset and our validation dataset.

# In[ ]:


train_df = pd.concat([r_train, k_train])
valid_df = pd.concat([r_test, k_test])

print(train_df.shape)
print(valid_df.shape)


# Generally, it is better practice to specify constant variables than it is to hard-code numbers. This way, changing parameters is more efficient and complete. Specfiy some constants below.

# In[ ]:


IMG_DIM = (1536, 128)
CLASSES_NUM = 6
BATCH_SIZE = 32
EPOCHS = 100
N=12

LEARNING_RATE = 1e-4
FOLDED_NUM_TRAIN_IMAGES = train_df.shape[0]
FOLDED_NUM_VALID_IMAGES = valid_df.shape[0]
STEPS_PER_EPOCH = FOLDED_NUM_TRAIN_IMAGES // BATCH_SIZE
VALIDATION_STEPS = FOLDED_NUM_VALID_IMAGES // BATCH_SIZE


# The `tf.keras.utils.Sequence` is a base object to fit a dataset. Since our dataset is stored both as images and as a csv, we will have to write a DataGenerator that is a subclass of the Sequence class. The DataGenerator will concatenate all the tiles from each original image into a newer image of just the masked areas. It will also get the label from the ISUP grade column and convert it to a one-hot encoding. One-hot encoding is necessary because the ISUP grade is not a continuous datatype but a categorical datatype.

# In[ ]:


class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,
                 image_shape,
                 batch_size, 
                 df,
                 img_dir,
                 mask_dir,
                 is_training=True
                 ):
        
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.df = df
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_training = is_training
        self.indices = range(df.shape[0])
        
    def __len__(self):
        return self.df.shape[0] // self.batch_size
    
    def on_epoch_start(self):
        if self.is_training:
            np.random.shuffle(self.indices)
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size : (index+1) * self.batch_size]
        image_ids = self.df['image_id'].iloc[batch_indices].values
        batch_images = [self.__getimages__(image_id) for image_id in image_ids]
        batch_labels = [self.df[self.df['image_id'] == image_id]['isup_grade'].values[0] for image_id in image_ids]
        batch_labels = tf.one_hot(batch_labels, CLASSES_NUM)
        
        return np.squeeze(np.stack(batch_images).reshape(-1, 1536, 128, 3)), np.stack(batch_labels)
        
    def __getimages__(self, image_id):
        fnames = [image_id+'_'+str(i)+'.png' for i in range(N)]
        images = []
        for fn in fnames:
            img = np.array(PIL.Image.open(os.path.join(self.img_dir, fn)).convert('RGB'))[:, :, ::-1]
            images.append(img)
        result = np.stack(images).reshape(1, 1536, 128, 3) / 255.0
        return result


# We will use the DataGenerator to create a generator for our training dataset and for our validation dataset. At each iteration of the generator, the generator will return a batch of images.

# In[ ]:


train_generator = DataGenerator(image_shape=IMG_DIM,
                                batch_size=BATCH_SIZE,
                                df=train_df,
                                img_dir=TRAIN_IMG_DIR,
                                mask_dir=TRAIN_MASKS_DIR)

valid_generator = DataGenerator(image_shape=IMG_DIM,
                                batch_size=BATCH_SIZE,
                                df=valid_df,
                                img_dir=TRAIN_IMG_DIR,
                                mask_dir=TRAIN_MASKS_DIR)


# # 3. Visualize our input data
# 
# Run the following cell to define the method to visualize our input data. This method displays the new images and their corresponding ISUP grade.

# In[ ]:


def show_tiles(image_batch, label_batch):
    plt.figure(figsize=(20,20))
    for n in range(10):
        ax = plt.subplot(1,10,n+1)
        plt.imshow(image_batch[n])
        decoded = np.argmax(label_batch[n])
        plt.title(decoded)
        plt.axis("off")


# In[ ]:


image_batch, label_batch = next(iter(train_generator))


# The following 12 tiles were from a single image but has been converted to 12 tiles to reduce white space. We see that only the sections that led to the ISUP grade has been preserved.

# In[ ]:


show_tiles(image_batch, label_batch)


# # 4. Build our model + Data augmentation
# 
# We will be utilizing the Xception pre-trained model to classify our data. The PANDA competition scores submissions using the quadratic weighted kappa. The TensorFlow add-on API contains the Cohen Kappa loss and metric functions. Since we want to use the newest version of TensorFlow through tf-nightly to utilize the pretrained EfficientNet model, we will refrain from using the TFA API as it has not been moved over yet to the tf-nightly version. However, feel free to create your own Cohen Kappa Metric and Loss class using the TensorFlow API.

# Data augmentation is helpful when dealing with image data as it prevents overfitting. Data augmentation introduces artificial but realistic variance in our images so that our model can learn from more features. Keras has recently implemented `keras.layers.preprocessing` that allows the model to streamline the data augmentation process.

# Since the base model has already been trained with imagenet weights, we do not want to weights to change, so the base mode must not be trainable. However, the number of classes that our model has differs from the original model. Therefore, we do not want to include the top layers because we will add our own Dense layer that has the same number of nodes as our output class.

# In[ ]:


def make_model():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.15, seed=SEED),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.experimental.preprocessing.RandomFlip("vertical", seed=SEED),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1, seed=SEED)
    ])
    
    base_model = tf.keras.applications.VGG16(input_shape=(*IMG_DIM, 3),
                                             include_top=False,
                                             weights='imagenet')
    
    base_model.trainable = True
    
    model = tf.keras.Sequential([
        data_augmentation,
        
        base_model,
        
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(CLASSES_NUM, activation='softmax'),
    ])
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss='categorical_crossentropy',
                  metrics=tf.keras.metrics.AUC(name='auc'))
    
    return model


# Let's build our model!

# In[ ]:


with strategy.scope():
    model = make_model()


# # 5. Training the model
# 
# And now let's train it! Learning rate is a very important hyperparameter, and it can be difficult to choose the "right" one. A learning rate that it too high will prevent the model from converging, but one that is too low will be far too slow. We will utilize multiple callbacks, using the `tf.keras` API to make sure that we are using an ideal learning rate and to prevent the model from overfitting. We can also save our model so that we do not have to retrain it next time.

# In[ ]:


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(0.01, 20)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("panda_model.h5",
                                                    save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                     restore_best_weights=True)


# In[ ]:


history = model.fit(
    train_generator, epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler]
)


# # 6. Predict results
# 
# For this competition, the test dataset is not available to us. But I wish you all the best of luck, and hopefully this NB served as a helpful tutorial to help you get started.

# In[ ]:




