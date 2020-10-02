#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# Hi gyus, from the public kernels, @lafoss shared some great notebooks about training tiles images on fast.ai/pytorch. <br />
# In most of the previous competitions, the Pytorch/fast.ai become more and more popular than tensorflow. <br />
# We can noticed that there are very few public notebooks were implemented in tensorflow. <br />
# So I implemented the same idea as @lafoss's in tensorflow framework and shared it in this kernel. <br />
# 
# Since it seems impossible to change the batch_size during training in tensorflow. (at least from I known) <br />
# So I splited the original model to two separated models and used customized training loop to get the same effect as @lafoss shared. <br />
# But I believe there must be some better ways to implement this idea in tensorflow. <br />
# Anyway, if you are also a tensorflow user, feel free to use this kernel as your baseline kernel. <br />
# I believe with some fine tunes, this kernel can get 0.7+ LB quite easy. <br />
# Good luck!
# 
# And thanks @lafoss again for the great ideas!
# 

# In[ ]:


get_ipython().system('pip install git+https://github.com/qubvel/classification_models.git')


# In[ ]:


import os
import PIL
import time
import math
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as albu
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from classification_models.tfkeras import Classifiers
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import cohen_kappa_score, make_scorer

SEED = 2020
warnings.filterwarnings('ignore')
print('Tensorflow version : {}'.format(tf.__version__))


# In[ ]:


MAIN_DIR = '../input/prostate-cancer-grade-assessment'
TRAIN_IMG_DIR = '../input/panda-tiles/train'
TRAIN_MASKS_DIR = '../input/panda-tiles/masks'
train_csv = pd.read_csv(os.path.join(MAIN_DIR, 'train.csv'))
noise_csv = pd.read_csv('../input/noisy-csv/suspicious_test_cases.csv')

for image_id in noise_csv['image_id'].values:
    train_csv = train_csv[train_csv['image_id'] != image_id]

radboud_csv = train_csv[train_csv['data_provider'] == 'radboud']
karolinska_csv = train_csv[train_csv['data_provider'] != 'radboud']


# In[ ]:


splits = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
splits = list(splits.split(radboud_csv, radboud_csv.isup_grade))
fold_splits = np.zeros(len(radboud_csv)).astype(np.int)
for i in range(5): 
    fold_splits[splits[i][1]]=i
radboud_csv['fold'] = fold_splits
radboud_csv.head(5)

splits = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
splits = list(splits.split(karolinska_csv, karolinska_csv.isup_grade))
fold_splits = np.zeros(len(karolinska_csv)).astype(np.int)
for i in range(5): 
    fold_splits[splits[i][1]]=i
karolinska_csv['fold'] = fold_splits
karolinska_csv.head(5)

train_csv = pd.concat([radboud_csv, karolinska_csv])
train_csv.shape


# In[ ]:


TRAIN_FLAG=False
TRAIN_FOLD = 0


# In[ ]:


train_df = train_csv[train_csv['fold'] != TRAIN_FOLD]
valid_df = train_csv[train_csv['fold'] == TRAIN_FOLD]

print(train_df.shape)
print(valid_df.shape)


# In[ ]:


IMG_DIM = (128, 128)
CLASSES_NUM = 6
BATCH_SIZE = 32
EPOCHS = 15
N=12

LEARNING_RATE = 1e-4
FOLDED_NUM_TRAIN_IMAGES = train_df.shape[0]
FOLDED_NUM_VALID_IMAGES = valid_df.shape[0]
STEPS_PER_EPOCH = FOLDED_NUM_TRAIN_IMAGES // BATCH_SIZE
VALIDATION_STEPS = FOLDED_NUM_VALID_IMAGES // BATCH_SIZE
PRETRAIN_PATH1 = '../input/tiles-pretrain/stage1.h5'
PRETRAIN_PATH2 = '../input/tiles-pretrain/stage2.h5'


# In[ ]:


print('*'*20)
print('Notebook info')
print('Training data : {}'.format(FOLDED_NUM_TRAIN_IMAGES))
print('Validing data : {}'.format(FOLDED_NUM_VALID_IMAGES))
print('Categorical classes : {}'.format(CLASSES_NUM))
print('Training image size : {}'.format(IMG_DIM))
print('Training epochs : {}'.format(EPOCHS))
print('*'*20)


# In[ ]:


class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,
                 image_shape,
                 batch_size, 
                 df,
                 img_dir,
                 mask_dir,
                 augmentation,
                 is_training=True
                 ):
        
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.df = df
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.aug = augmentation
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
        
        return np.stack(batch_images).reshape(-1,128, 128, 3), np.stack(batch_labels)
        
        
    def __getimages__(self, image_id):
        fnames = [image_id+'_'+str(i)+'.png' for i in range(N)]
        images = []
        for fn in fnames:
            img = np.array(PIL.Image.open(os.path.join(self.img_dir, fn)).convert('RGB'))[:, :, ::-1]
            if self.aug:
                images.append(self.aug(image=img)['image'])
            else:
                images.append(img)     
        return np.stack(images) / 255.0


# In[ ]:


train_augumentation = albu.Compose([
                            albu.OneOf([
                                albu.RandomBrightness(limit=0.15),
                                albu.RandomContrast(limit=0.3),
                                albu.RandomGamma(),
                            ], p=0.25),
                            albu.HorizontalFlip(p=0.4),
                            albu.VerticalFlip(p=0.4),
                            albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.3)
])


# In[ ]:


train_generator = DataGenerator(image_shape=IMG_DIM,
                                batch_size=BATCH_SIZE,
                                df=train_df,
                                img_dir=TRAIN_IMG_DIR,
                                mask_dir=TRAIN_MASKS_DIR,
                                augmentation=train_augumentation)

valid_generator = DataGenerator(image_shape=IMG_DIM,
                                batch_size=BATCH_SIZE,
                                df=valid_df,
                                img_dir=TRAIN_IMG_DIR,
                                mask_dir=TRAIN_MASKS_DIR,
                                augmentation=None)


# In[ ]:


def build_model():
    resnet18, _ = Classifiers.get('resnet18')
    stage1_model = resnet18(input_shape=(*IMG_DIM, 3), weights='imagenet', include_top=False)
    
    input_layer = tf.keras.layers.Input(shape=(4, 4, stage1_model.output_shape[-1]))
    x = tf.keras.layers.GlobalAveragePooling2D()(input_layer)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(units=512, kernel_initializer='he_normal', activation='relu')(x)
    cls_head = tf.keras.layers.Dense(units=CLASSES_NUM, activation='softmax')(x)
    
    stage2_model = tf.keras.models.Model(inputs=[input_layer], outputs=[cls_head])
    return stage1_model, stage2_model, stage1_model.output_shape[-1]


# In[ ]:


stage1_model, stage2_model, stage1_channels = build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fn = lambda a,b: tf.nn.compute_average_loss(tf.keras.losses.sparse_categorical_crossentropy(a,b), global_batch_size=BATCH_SIZE)
train_loss = tf.keras.metrics.Sum()
valid_loss = tf.keras.metrics.Sum()
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


# In[ ]:


if PRETRAIN_PATH1:
    print('load stage 1 model pretrain weights..')
    stage1_model.load_weights(PRETRAIN_PATH1)
    
if PRETRAIN_PATH2:
    print('load stage 2 model pretrain weights..')
    stage2_model.load_weights(PRETRAIN_PATH2)


# In[ ]:


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as stage1_tape, tf.GradientTape() as stage2_tape:
        stage1_output = stage1_model(images, training=True)
        stage1_output = tf.reshape(stage1_output, (-1, N, 4, 4, stage1_channels))
        stage1_output = tf.transpose(stage1_output, (0, 2, 1, 3, 4))
        stage1_output = tf.reshape(stage1_output, (-1, 4, 4*N, stage1_channels))
        stage2_output = stage2_model(stage1_output, training=True)
        loss = loss_fn(labels, stage2_output)
        
    stage1_grads = stage1_tape.gradient(loss, stage1_model.trainable_variables)
    stage2_grads = stage2_tape.gradient(loss, stage2_model.trainable_variables)
    
    optimizer.apply_gradients(zip(stage1_grads, stage1_model.trainable_variables))
    optimizer.apply_gradients(zip(stage2_grads, stage2_model.trainable_variables))
    
    train_loss.update_state(loss)
    train_accuracy.update_state(labels, stage2_output)
    
def valid_step(images, labels):
    stage1_output = stage1_model(images, training=False)
    stage1_output = tf.reshape(stage1_output, (-1, N, 4, 4, stage1_channels))
    stage1_output = tf.transpose(stage1_output, (0, 2, 1, 3, 4))
    stage1_output = tf.reshape(stage1_output, (-1, 4, 4*N, stage1_channels))
    stage2_output = stage2_model(stage1_output, training=False)
    
    loss = loss_fn(labels, stage2_output)
    valid_loss.update_state(loss)
    valid_accuracy.update_state(labels, stage2_output)
    
    return stage2_output.numpy()

def inference_step(images):
    stage1_output = stage1_model(images, training=False)
    stage1_output = tf.reshape(stage1_output, (-1, N, 4, 4, stage1_channels))
    stage1_output = tf.transpose(stage1_output, (0, 2, 1, 3, 4))
    stage1_output = tf.reshape(stage1_output, (-1, 4, 4*N, stage1_channels))
    stage2_output = stage2_model(stage1_output, training=False)
    
    return stage2_output.numpy()    


# In[ ]:


best_valid_qwk = 0
history = {
    'train_loss' : [],
    'valid_loss' : [],
    'train_accuracy' : [],
    'valid_accuracy' : [],
    'qwk' : []
}


# In[ ]:


if TRAIN_FLAG:

    print("Steps per epoch:", STEPS_PER_EPOCH, "Valid steps per epoch:", VALIDATION_STEPS)
    epoch = 0
    for epoch in range(EPOCHS):
        start_time = time.time()
        #model training
        for step in range(train_generator.__len__()):
            images, labels = train_generator.__getitem__(step)
            train_step(images, labels)
            if step % 10 == 0:
                print('=', end='', flush=True)

        print('|', end='', flush=True)
        #model validation
        predictions = []
        groundtruths = []
        for v_step in range(valid_generator.__len__()):
            images, labels = valid_generator.__getitem__(v_step)
            valid_preds = valid_step(images, labels)
            valid_preds = np.argmax(valid_preds, axis=-1)
            groundtruths += list(labels)
            predictions += list(valid_preds)
            if v_step % 10 == 0:
                print('=', end='', flush=True)

        qwk = cohen_kappa_score(groundtruths, predictions, labels=None, weights= 'quadratic', sample_weight=None)

        history['train_loss'].append(train_loss.result().numpy() / STEPS_PER_EPOCH)
        history['valid_loss'].append(valid_loss.result().numpy() / VALIDATION_STEPS)
        history['train_accuracy'].append(train_accuracy.result().numpy())
        history['valid_accuracy'].append(valid_accuracy.result().numpy())
        history['qwk'].append(qwk)

        print('\nEPOCH {:d}/{:d}'.format(epoch+1, EPOCHS))
        print('loss: {:0.4f}'.format(history['train_loss'][-1]),'val_loss: {:0.4f}'.format(history['valid_loss'][-1]))
        print('accuracy : {}'.format(history['train_accuracy'][-1]), 'val_accuracy : {}'.format(history['valid_accuracy'][-1]))
        print('validation qwk : {}'.format(qwk))

        # set up next epoch
        valid_loss.reset_states()
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_accuracy.reset_states()


        if history['qwk'][-1] > best_valid_qwk:
            print('Validation qwk improve from {} to {}, save model checkpoint'.format(best_valid_qwk, history['qwk'][-1]))
            stage1_model.save('stage1.h5')
            stage2_model.save('stage2.h5')
            best_valid_qwk = history['qwk'][-1]

        print('Spending time : {}...'.format(time.time()-start_time))


# In[ ]:


def plot_training_history(history):
    
    plt.figure(figsize=(12,12))
    plt.plot(np.arange(0,len(history['train_loss']),1), history['train_loss'], label='train_loss', color='g')
    plt.plot(np.arange(0,len(history['valid_loss']),1), history['valid_loss'], label='validation_loss', color='r')
    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
if TRAIN_FLAG:
    plot_training_history(history)
    stage1_model.save('stage1_finalcheckpoint.h5')
    stage2_model.save('stage2_finalcheckpoint.h5')


# In[ ]:


def inference():
    
    ground_truth = []
    prediction = []
    
    for step in range(valid_generator.__len__()):
        images, labels = valid_generator.__getitem__(step)
        preds = inference_step(images)
        if step % 10 == 0:
            print('=', end='', flush=True)
            
        preds = np.argmax(preds, axis=-1)
        
        for y_true, y_pred in zip(labels, preds):
            ground_truth.append(y_true)
            prediction.append(y_pred)
        
    qwk = cohen_kappa_score(ground_truth, prediction, labels=None, weights= 'quadratic', sample_weight=None)
    print('\nValid QWK : {}'.format(qwk))


# In[ ]:


if TRAIN_FLAG:
    stage1_model.load_weights('stage1.h5')
    stage2_model.load_weights('stage2.h5')
inference()

