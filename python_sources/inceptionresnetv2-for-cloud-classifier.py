#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -U keras-applications')


# In[ ]:


import keras_applications
import keras
import tensorflow as tf


# In[ ]:



from keras_applications.resnext import ResNeXt101


# this kernel was forked from here : https://www.kaggle.com/samusram/cloud-classifier-for-post-processing?scriptVersionId=20265194
# 
# highest  accuracy of that kernel was 0.629 which is same as this one : https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools?scriptVersionId=20202006 of Andrew Lukyanenko (  @artgor) 
# thanks a lot to them for their public kernels,i recommend everyone playing with those kernels because those kernels has got solid baseline. my special thanks goes to Andrew Lukyanenko (  @artgor) ,i learnt plenty from his works 
# 
# Here in this kernel i got 0.63 after trying InceptionResNetV2 so i decided to share it with you guys,please leave your precious comment below,this is just a starting,i will write my own notebook when time permits but i would like to spend more time on @artgor's kernel to observe if i can improve his work little bit,i recommend you the same.
# 
# whats new here?
# 1. InceptionResNetV2 instead of densenet121
# 2. image height and width 229x229 instead of 224
# 3. version 1 has got 0.63 public lb accuracy ,so please upvote this notebok and obviously main author's notebook if you find these helpful
# 4.  in version 3 i will try to leave everything as it was before but 5 epochs for initial tuning and 10 epochs for fine tuning

# **Version 6**
# - replacing this notebook's submission.csv with my trained models submission.csv file(got 0.658 public lb score) which comes from version 13 of this notebook : https://www.kaggle.com/mobassir/keras-efficientnetb2-for-classifying-cloud
# 
# - using 0.90 as both precision and recall threshold
# - Rotate(limit=20)
# 

# # version 7
# - using 0.94 as both precision and recall threshold
# - 17% for test set
# - lower learning rate
# - train for less epoches

# <font color="red">i hope this kernel is helpful and some upvotes are highly appreciated!</font>

# # version 9
# 
# trying resnext50 model with image size 224x224

# # version 10(last version)
# 
# trying resnext101 model with image size 224x224

# # Intro
# In this notebook I'd create a classifier to distinguish types of cloud formations. Using this classifier I'd check if it improves currently the best LB score from the great [public notebook by Andrew](https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools?scriptVersionId=20202006). 

# # Plan
# 1. [Libraries](#Libraries)
# 2. [Data Generators](#Data-Generators)
#   * [One-hot encoding classes](#One-hot-encoding-classes)
#   * [Stratified split into train/val](#Stratified-split-into-train/val)
#   * [Generator class](#Generator-class)
# 3. [PR-AUC-based Callback](#PR-AUC-based-Callback)
# 4. [Classifier](#Classifier)
#   * [Defining a model](#Defining-a-model)
#   * [Initial tuning of the added fully-connected layer](#Initial-tuning-of-the-added-fully-connected-layer)
#   * [Fine-tuning the whole model](#Fine-tuning-the-whole-model)
#   * [Visualizing train and val PR AUC](#Visualizing-train-and-val-PR-AUC)
# 5. [Selecting postprocessing thresholds](#Selecting-postprocessing-thresholds)
# 6. [Post-processing Andrew's submission](#Post-processing-Andrew's-submission)
# 7. [Future work](#Future-work)

# # Libraries

# In[ ]:


import os, glob
import random
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import pandas as pd
import multiprocessing
from copy import deepcopy
from sklearn.metrics import precision_recall_curve, auc
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence
from albumentations import Compose, VerticalFlip, HorizontalFlip, Rotate, GridDistortion,CenterCrop
import matplotlib.pyplot as plt
from IPython.display import Image
from tqdm import tqdm_notebook as tqdm
from numpy.random import seed
seed(10)
from tensorflow import set_random_seed
set_random_seed(10)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


test_imgs_folder = '../input/understanding_cloud_organization/test_images/'
train_imgs_folder = '../input/understanding_cloud_organization/train_images/'
num_cores = multiprocessing.cpu_count()


# ## One-hot encoding classes

# In[ ]:


train_df = pd.read_csv('../input/understanding_cloud_organization/train.csv')
train_df.head()


# In[ ]:


train_df = train_df[~train_df['EncodedPixels'].isnull()]
train_df['Image'] = train_df['Image_Label'].map(lambda x: x.split('_')[0])
train_df['Class'] = train_df['Image_Label'].map(lambda x: x.split('_')[1])
classes = train_df['Class'].unique()
train_df = train_df.groupby('Image')['Class'].agg(set).reset_index()
for class_name in classes:
    train_df[class_name] = train_df['Class'].map(lambda x: 1 if class_name in x else 0)
train_df.head()


# In[ ]:


# dictionary for fast access to ohe vectors
img_2_ohe_vector = {img:vec for img, vec in zip(train_df['Image'], train_df.iloc[:, 2:].values)}


# ## Stratified split into train/val

# In[ ]:


train_imgs, val_imgs = train_test_split(train_df['Image'].values, 
                                        test_size=0.1, 
                                        stratify=train_df['Class'].map(lambda x: str(sorted(list(x)))), # sorting present classes in lexicographical order, just to be sure
                                        random_state=43)


# ## Generator class

# In[ ]:


class DataGenenerator(Sequence):
    def __init__(self, images_list=None, folder_imgs=train_imgs_folder, 
                 batch_size=32, shuffle=True, augmentation=None,
                 resized_height=224, resized_width=224, num_channels=3):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        if images_list is None:
            self.images_list = os.listdir(folder_imgs)
        else:
            self.images_list = deepcopy(images_list)
        self.folder_imgs = folder_imgs
        self.len = len(self.images_list) // self.batch_size
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.num_channels = num_channels
        self.num_classes = 4
        self.is_test = not 'train' in folder_imgs
        if not shuffle and not self.is_test:
            self.labels = [img_2_ohe_vector[img] for img in self.images_list[:self.len*self.batch_size]]

    def __len__(self):
        return self.len
    
    def on_epoch_start(self):
        if self.shuffle:
            random.shuffle(self.images_list)

    def __getitem__(self, idx):
        current_batch = self.images_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        X = np.empty((self.batch_size, self.resized_height, self.resized_width, self.num_channels))
        y = np.empty((self.batch_size, self.num_classes))

        for i, image_name in enumerate(current_batch):
            path = os.path.join(self.folder_imgs, image_name)
            img = cv2.resize(cv2.imread(path), (self.resized_height, self.resized_width)).astype(np.float32)
            if not self.augmentation is None:
                augmented = self.augmentation(image=img)
                img = augmented['image']
            X[i, :, :, :] = img/255.0
            if not self.is_test:
                y[i, :] = img_2_ohe_vector[image_name]
        return X, y

    def get_labels(self):
        if self.shuffle:
            images_current = self.images_list[:self.len*self.batch_size]
            labels = [img_2_ohe_vector[img] for img in images_current]
        else:
            labels = self.labels
        return np.array(labels)


# In[ ]:


albumentations_train = Compose([
    VerticalFlip(), HorizontalFlip(), Rotate(limit=20), GridDistortion()
], p=1)


# Generator instances

# In[ ]:


data_generator_train = DataGenenerator(train_imgs, augmentation=albumentations_train)
data_generator_train_eval = DataGenenerator(train_imgs, shuffle=False)
data_generator_val = DataGenenerator(val_imgs, shuffle=False)


# # PR-AUC-based Callback

# The callback would be used:
# 1. to estimate AUC under precision recall curve for each class,
# 2. to early stop after 5 epochs of no improvement in mean PR AUC,
# 3. save a model with the best PR AUC in validation,
# 4. to reduce learning rate on PR AUC plateau.

# In[ ]:


class PrAucCallback(Callback):
    def __init__(self, data_generator, num_workers=num_cores, 
                 early_stopping_patience=3, 
                 plateau_patience=3, reduction_rate=0.5,
                 stage='train', checkpoints_path='checkpoints/'):
        super(Callback, self).__init__()
        self.data_generator = data_generator
        self.num_workers = num_workers
        self.class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']
        self.history = [[] for _ in range(len(self.class_names) + 1)] # to store per each class and also mean PR AUC
        self.early_stopping_patience = early_stopping_patience
        self.plateau_patience = plateau_patience
        self.reduction_rate = reduction_rate
        self.stage = stage
        self.best_pr_auc = -float('inf')
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
        self.checkpoints_path = checkpoints_path
        
    def compute_pr_auc(self, y_true, y_pred):
        pr_auc_mean = 0
        print(f"\n{'#'*30}\n")
        for class_i in range(len(self.class_names)):
            precision, recall, _ = precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])
            pr_auc = auc(recall, precision)
            pr_auc_mean += pr_auc/len(self.class_names)
            print(f"PR AUC {self.class_names[class_i]}, {self.stage}: {pr_auc:.3f}\n")
            self.history[class_i].append(pr_auc)        
        print(f"\n{'#'*20}\n PR AUC mean, {self.stage}: {pr_auc_mean:.3f}\n{'#'*20}\n")
        self.history[-1].append(pr_auc_mean)
        return pr_auc_mean
              
    def is_patience_lost(self, patience):
        if len(self.history[-1]) > patience:
            best_performance = max(self.history[-1][-(patience + 1):-1])
            return best_performance == self.history[-1][-(patience + 1)] and best_performance >= self.history[-1][-1]    
              
    def early_stopping_check(self, pr_auc_mean):
        if self.is_patience_lost(self.early_stopping_patience):
            self.model.stop_training = True    
              
    def model_checkpoint(self, pr_auc_mean, epoch):
        if pr_auc_mean > self.best_pr_auc:
            # remove previous checkpoints to save space
            for checkpoint in glob.glob(os.path.join(self.checkpoints_path, 'classifier_epoch_*')):
                os.remove(checkpoint)
        self.best_pr_auc = pr_auc_mean
        self.model.save(os.path.join(self.checkpoints_path, f'classifier_epoch_{epoch}_val_pr_auc_{pr_auc_mean}.h5'))              
        print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")
              
    def reduce_lr_on_plateau(self):
        if self.is_patience_lost(self.plateau_patience):
            new_lr = float(keras.backend.get_value(self.model.optimizer.lr)) * self.reduction_rate
            keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"\n{'#'*20}\nReduced learning rate to {new_lr}.\n{'#'*20}\n")
        
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_generator(self.data_generator, workers=self.num_workers)
        y_true = self.data_generator.get_labels()
        # estimate AUC under precision recall curve for each class
        pr_auc_mean = self.compute_pr_auc(y_true, y_pred)
              
        if self.stage == 'val':
            # early stop after early_stopping_patience=4 epochs of no improvement in mean PR AUC
            self.early_stopping_check(pr_auc_mean)

            # save a model with the best PR AUC in validation
            self.model_checkpoint(pr_auc_mean, epoch)

            # reduce learning rate on PR AUC plateau
            self.reduce_lr_on_plateau()            
        
    def get_pr_auc_history(self):
        return self.history


# Callback instances

# In[ ]:


train_metric_callback = PrAucCallback(data_generator_train_eval)
val_callback = PrAucCallback(data_generator_val, stage='val')


# # Classifier

# ## Defining a model

# > Let's us InceptionResNetV2 pretrained on ImageNet.

# In[ ]:


from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
def get_model():
    #base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
    base_model = model = ResNeXt101(..., backend=tf.keras.backend, layers=tf.keras.layers, weights = 'imagenet', models=tf.keras.models, utils=tf.keras.utils)
    x = base_model.output
    y_pred = Dense(4, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=y_pred)

model = get_model()


# > ## Initial tuning of the added fully-connected layer

# In[ ]:


for base_layer in model.layers[:-1]:
    base_layer.trainable = False
    
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy')
history_0 = model.fit_generator(generator=data_generator_train,
                              validation_data=data_generator_val,
                              epochs=15,
                              callbacks=[train_metric_callback, val_callback],
                              workers=num_cores,
                              verbose=1
                             )


# ## Fine-tuning the whole model

# After unfreezing all the layers I set a less aggressive initial learning rate and train until early stopping (or 100 epochs max).

# In[ ]:


for base_layer in model.layers[:-1]:
    base_layer.trainable = True
    
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy')
history_1 = model.fit_generator(generator=data_generator_train,
                              validation_data=data_generator_val,
                              epochs=20,
                              callbacks=[train_metric_callback, val_callback],
                              workers=num_cores,
                              verbose=1,
                              initial_epoch=1
                             )


# ## Visualizing train and val PR AUC

# In[ ]:


def plot_with_dots(ax, np_array):
    ax.scatter(list(range(1, len(np_array) + 1)), np_array, s=50)
    ax.plot(list(range(1, len(np_array) + 1)), np_array)


# In[ ]:


pr_auc_history_train = train_metric_callback.get_pr_auc_history()
pr_auc_history_val = val_callback.get_pr_auc_history()

plt.figure(figsize=(10, 7))
plot_with_dots(plt, pr_auc_history_train[-1])
plot_with_dots(plt, pr_auc_history_val[-1])

plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Mean PR AUC', fontsize=15)
plt.legend(['Train', 'Val'])
plt.title('Training and Validation PR AUC', fontsize=20)
plt.savefig('pr_auc_hist.png')


# In[ ]:


plt.figure(figsize=(10, 7))
plot_with_dots(plt, history_0.history['loss']+history_1.history['loss'])
plot_with_dots(plt, history_0.history['val_loss']+history_1.history['val_loss'])

plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Binary Crossentropy', fontsize=15)
plt.legend(['Train', 'Val'])
plt.title('Training and Validation Loss', fontsize=20)
plt.savefig('loss_hist.png')


# I left the model to train longer on my local GPU. I then upload the best model and plots from the model training.

# In[ ]:


model = load_model('../input/clouds-classifier-files/classifier_epoch_45_val_pr_auc_0.8344173287108075.h5')


# In[ ]:


Image("../input/clouds-classifier-files/loss_hist.png")


# In[ ]:


Image("../input/clouds-classifier-files/pr_auc_hist.png")


# As a side note, it was interesting to see an overfitting pattern when not using augmentation (learning rate was higher). 

# In[ ]:


Image("../input/clouds-classifier-files/training_hist_no_aug.png")


# # Selecting postprocessing thresholds

# In[ ]:


class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']
def get_threshold_for_recall(y_true, y_pred, class_i, recall_threshold=0.94, precision_threshold=0.95, plot=False):
    precision, recall, thresholds = precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])
    i = len(thresholds) - 1
    best_recall_threshold = None
    while best_recall_threshold is None:
        next_threshold = thresholds[i]
        next_recall = recall[i]
        if next_recall >= recall_threshold:
            best_recall_threshold = next_threshold
        i -= 1
        
    # consice, even though unnecessary passing through all the values
    best_precision_threshold = [thres for prec, thres in zip(precision, thresholds) if prec >= precision_threshold][0]
    
    if plot:
        plt.figure(figsize=(10, 7))
        plt.step(recall, precision, color='r', alpha=0.3, where='post')
        plt.fill_between(recall, precision, alpha=0.3, color='r')
        plt.axhline(y=precision[i + 1])
        recall_for_prec_thres = [rec for rec, thres in zip(recall, thresholds) 
                                 if thres == best_precision_threshold][0]
        plt.axvline(x=recall_for_prec_thres, color='g')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(['PR curve', 
                    f'Precision {precision[i + 1]: .2f} corresponding to selected recall threshold',
                    f'Recall {recall_for_prec_thres: .2f} corresponding to selected precision threshold'])
        plt.title(f'Precision-Recall curve for Class {class_names[class_i]}')
    return best_recall_threshold, best_precision_threshold

y_pred = model.predict_generator(data_generator_val, workers=num_cores)
y_true = data_generator_val.get_labels()
recall_thresholds = dict()
precision_thresholds = dict()
for i, class_name in tqdm(enumerate(class_names)):
    recall_thresholds[class_name], precision_thresholds[class_name] = get_threshold_for_recall(y_true, y_pred, i, plot=True)


# # Post-processing Andrew's submission

# Predicting cloud classes for test.

# In[ ]:


data_generator_test = DataGenenerator(folder_imgs=test_imgs_folder, shuffle=False)
y_pred_test = model.predict_generator(data_generator_test, workers=num_cores)


# Estimating set of images without masks.

# In[ ]:


image_labels_empty = set()
for i, (img, predictions) in enumerate(zip(os.listdir(test_imgs_folder), y_pred_test)):
    for class_i, class_name in enumerate(class_names):
        if predictions[class_i] < recall_thresholds[class_name]:
            image_labels_empty.add(f'{img}_{class_name}')


# In[ ]:


images_labels_full = set()
for i, (img, predictions) in enumerate(zip(os.listdir(test_imgs_folder), y_pred_test)):
    for class_i, class_name in enumerate(class_names):
        if predictions[class_i] > precision_thresholds[class_name]:
            images_labels_full.add(f'{img}_{class_name}')


# My predictions from this kernel : https://www.kaggle.com/mobassir/keras-efficientnetb2-for-classifying-cloud

# In[ ]:


submission = pd.read_csv('../input/cloudefficientnetb2/submission.csv')
submission.head()


# In[ ]:


image_labels_andrew_nonempty = submission.loc[~submission['EncodedPixels'].isnull(), 'Image_Label'].values
image_labels_andrew_empty = submission.loc[submission['EncodedPixels'].isnull(), 'Image_Label'].values


#  I remove the min_size parameter from the post_process function, so that more mask are generated in the submission. I'd add additional masks for the Image_Label pairs predicted by the classifier.

# In[ ]:


images_to_refill = images_labels_full.intersection(set(image_labels_andrew_empty))
submission_no_minsize = pd.read_csv('../input/clouds-classifier-files/submission_no_minsize.csv')
masks_no_minsize_refillers = submission_no_minsize.loc[submission_no_minsize['Image_Label'].isin(images_to_refill) &
                                                       ~submission_no_minsize['EncodedPixels'].isnull()]
print(f'{len(masks_no_minsize_refillers)} masks would be added.')


# In[ ]:


print(f'{len(image_labels_empty.intersection(set(image_labels_andrew_nonempty)))} masks would be removed')


# In[ ]:


#removing masks
submission.loc[submission['Image_Label'].isin(image_labels_empty), 'EncodedPixels'] = np.nan
# adding masks
img_labels_to_refill = set(masks_no_minsize_refillers['Image_Label'].values)
submission = submission[~submission['Image_Label'].isin(img_labels_to_refill)]
submission = pd.concat((submission, masks_no_minsize_refillers))
submission.to_csv('submission.csv', index=None)


# # Future work
# 1. estimate distribution of classes in test set using the classifier. Then, if necessary and doable, modify val set accordingly,
# 2. use the classifier with explainability technique [Gradient-weighted Class Activation Mapping](http://gradcam.cloudcv.org/) to generate a baseline,
# 3. improve the classifier,
# 4. use the classifier as backbone for UNet-like solution.
