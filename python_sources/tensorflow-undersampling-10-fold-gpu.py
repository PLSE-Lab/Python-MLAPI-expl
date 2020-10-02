#!/usr/bin/env python
# coding: utf-8

# ## Melanoma Detection:
# Generally in any medical image diagnosis Machine Learning problems, the number of positive labelled data will be less compared to negative labelled data since the number of people suffering from the disease will be less compared to number of people tested.It is no different in our current dataset.
# 
# The number of images corresponding to benign tumours is 98% which leads to huge Class Imbalance Problem.
# 
# There are various techniques for handling Class Imbalance.The one used is this kernel is ***UnderSampling***.
# UnderSampling in simple terms can be thought of as reducing the number of data points corresponding to the class which has significantly more data points in a class imbalance scenario
# 
# ![](http://)
# 
# 
# 

# In[ ]:


# Installing Necessary Packages

get_ipython().system('pip install efficientnet')
get_ipython().system('pip install sweetviz')


# In[ ]:


import os
import albumentations
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn import model_selection
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import efficientnet.tfkeras as efn 
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import (ModelCheckpoint, LearningRateScheduler,
                                        EarlyStopping, ReduceLROnPlateau, CSVLogger)
import math
import sweetviz as sv


# In[ ]:


# Create a dataframe out of train csv file
df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")


# ## Powerful Visualizations using sweetviz library

# Recently came across a Python Library called "sweetviz" which helps for basic EDA in just two lines of code!!!!!!
# It generates a html report with interactive visualizations.

# In[ ]:


my_report = sv.analyze(df, target_feat = 'target')
my_report.show_html("report.html") # Default arguments will generate to "SWEETVIZ_REPORT.html"


# Just Hover over the feature in the html report, a graph would be shown in the extreme right [Need to open the notebook]

# In[ ]:


# from IPython.display import IFrame

# IFrame(src='report.html', width=2000, height=1000)


# In[ ]:


df.head()


# In[ ]:


df['benign_malignant'].value_counts()
df['target'].value_counts()


# In[ ]:


sns.countplot(x = 'target', data = df)
sns.countplot(x = 'benign_malignant', data = df)


# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(x = 'anatom_site_general_challenge', hue = 'target', data = df)


# ## UnderSampling

# On looking into the dataset it can be noted that for the same person[Patient ID] and for the same region of the body[anatom_site_general_challenge] , there are multiple Images. Only one image per person per anatomy region only is used, the rest all are dropped for benign cases.The malignant datapoints are not touched.

# In[ ]:


#Filtering benign and Malignant datapoints

df_malignant = df[df['target'] == 1]
df_benign = df[df['target'] == 0]

df_malignant = df_malignant.sample(frac=1).reset_index(drop=True)
df_benign = df_benign.sample(frac=1).reset_index(drop=True)


# In[ ]:


# Dropping data points for benign cases

df_benign = df_benign.drop_duplicates(subset=['patient_id','anatom_site_general_challenge'], keep = "first")


# In[ ]:


# Concatenating the data frame

df = pd.concat([df_malignant, df_benign]).reset_index(drop = True)
df = df.sample(frac=1).reset_index(drop=True)


# In[ ]:


df.tail()


# We can see from the counts below that the ratio is now close to 90:10 which is much better than 98:2
# Also with stratified KFold technique the same ratio can be maintained in each folds

# In[ ]:


df['target'].value_counts()


# In[ ]:


#Initializing the parameters

batch_size = 16
train_dir = "../input/melanoma-merged-external-data-512x512-jpeg/512x512-dataset-melanoma/512x512-dataset-melanoma/"
IMG_HEIGHT = 512
IMG_WIDTH = 512


# Stratified KFold samples [Inspired from Abhishek Thakur's Kernel]

# In[ ]:


# Stratified KFold samples


df = df.sample(frac=1).reset_index(drop=True)
n_splits = 5
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits)
for fold, (_, val_ind) in enumerate(kf.split(X=df, y=df.target.values)):
    df.loc[val_ind, 'fold'] = fold

df.to_csv("train_fold.csv", index=False)
df = df.sample(frac=1).reset_index(drop=True) # shuffling 


# In[ ]:


def data_augmentor(image):
    
    '''
    Function which perfoms certain random operations and returns the augmented Image
    
    '''
    
    image = tf.keras.preprocessing.image.random_shift(image,0.4,0.4)
    image = tf.keras.preprocessing.image.random_rotation(image,20)
    image = tf.keras.preprocessing.image.random_zoom(image, (0.2,0.5),(0.2,0.5))
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    
    return image
    


# ### Custom Data Generator

# Customer data generator is built which reads images from the disk and yields it.
# Need knowledge on how "Python Generators" work to understand the below function.
# The reason for building Custom Data Generator is that it gives you more flexibility[ Example : Augment only particular class of image/ augment only particular anatomical body images/ Center crop and resize only certain kind of images .. etc]
# 
# Currently all the images are augmented in the same way.. Will expore different options in the coming weeks 

# In[ ]:


def data_generator(image_dir,data,batch_size,num_batches,train = True): 
    '''
    Generator which yields batch of images.
    image_dir : Path to the images
    data : Dataframe of the data
    batch_size : Number of data points in a batch
    num_batches : Number of batches
    train : True/False 
    
    '''
        
    while True:        
        traversed_datapoints = 0
        for batch in range(num_batches):            
            i = 0
            batch_data = tf.Variable(tf.zeros((batch_size,IMG_HEIGHT,IMG_WIDTH,3))) 
            batch_labels = tf.Variable(tf.zeros((batch_size,1),tf.int32)) 
            #print(batch_labels.dtype)
            
            while(i<batch_size):
                
                augmentation = False
                image_path = os.path.join(image_dir, data['image_id'][i + traversed_datapoints] + '.jpg')
                #print(image_path)
                image = cv2.imread(image_path)              
                image = image.astype(np.float32)/255.0
                label = tf.reshape(int(data['target'][i + traversed_datapoints]) ,[1,] ) 
                #print(label.dtype)
                if train:                    
                    augmentation = True
                    if augmentation:
                        if np.random.randn(1)[0] > 0: 
                            #print(batch_labels[i].shape)
                                                       

                            image = data_augmentor(image)
                            batch_data[i,:, :, :] = tf.Variable(image)                           
                            
                            batch_labels[i].assign(label)
                            i = i + 1                       
                        else:                            
                             batch_data[i,:, :, :] = tf.Variable(image)
                             batch_labels[i].assign(label)
                             i = i + 1
                    else:
                       #print(batch_labels[i].shape)
                       #print(int(data['target'][i + traversed_datapoints]))
                       batch_data[i,:, :, :].assign(tf.Variable(image))
                       batch_labels[i].assign(label)                      
                       i = i + 1
                else:
                    batch_data[i,:, :, :].assign(tf.Variable(image)) 
                    i = i + 1                    
            traversed_datapoints = batch_size*(batch+1)
            
            if data.shape[0] - traversed_datapoints < batch_size:
                print("Modified batch size")
                batch_size = data.shape[0] - traversed_datapoints
                print(batch_size)
                                    
            if train:
                yield batch_data.numpy(), batch_labels.numpy()
            else:
                yield batch_data.numpy()

        


# In[ ]:



def fold_generator(fold):
    '''
    Function with takes in fold as an integer and returns the train and validation generators.
    
    '''
       
    train_data = df[df.fold != fold].reset_index(drop=True)
    val_data = df[df.fold == fold].reset_index(drop=True)  
    num_total = train_data.shape[0]
    num_total_val = val_data.shape[0]
    steps_per_epoch = math.ceil(num_total/batch_size)      
    val_steps = int(num_total_val/batch_size)     
    
    train_data_generator = data_generator(train_dir,train_data,batch_size,steps_per_epoch,True)
    val_data_generator = data_generator(train_dir,val_data,batch_size,steps_per_epoch,True)
    

    return train_data_generator, val_data_generator, train_data,val_data, steps_per_epoch, val_steps


# In[ ]:


def scheduler(epoch):
    '''
    Simple Learning rate scheduler which exponentially decays the learning rate in every epoch
    epoch : The current epoch number
    
    '''
        
    if epoch < 1:
        return 0.0001
    else:
        return 0.00001 * tf.math.exp(0.1 * (10 - epoch))


# In[ ]:


def focal_loss(y_true, y_pred):
    gamma = 2.0
    p_t    = (y_true*y_pred) + ((1-y_true)*(1-y_pred))        
    scaling_factor = K.pow((1-p_t), gamma)
    CE_loss = K.binary_crossentropy(y_true, y_pred)
    focal_loss = scaling_factor*CE_loss
    return focal_loss
    
    


# ## EfficientNetB1 Model

# Pretrained EfficientNet B1 Model is used with its top layer removed and adding a custom head

# In[ ]:



def MelnaomaNet(input_dim, base_model):
    '''
    Function with creates a model and return it
    input_dim : Dimensions of the tensor input to the model
    base_model : EfficientNet Model instance
    
    '''
    
    input_tensor = L.Input(input_dim)
    curr_output  = base_model(input_tensor)
    curr_output  = L.GlobalAveragePooling2D()(curr_output)
    oputput      = L.Dense(1,activation='sigmoid')(curr_output)
    model = Model(input_tensor, oputput)
    opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(
    optimizer = opt,
    loss      = tf.keras.losses.binary_crossentropy,
    metrics   = [tf.keras.metrics.AUC()]
    )
    return model


with strategy.scope():
    dim = (512,512)
    efnet = efn.EfficientNetB1(weights='imagenet', include_top = False, 
                       input_shape=(*dim, 3))
    model = MelnaomaNet(input_dim=(*dim,3), base_model = efnet)


# In[ ]:


model.summary()


# ## Training

# Training the model for 10 Strafified KFolds

# In[ ]:


infer = False

for fold in range(n_splits):
    '''
    
    Function which trains for each of the n_split fold and saves the model.
    
    '''
    train_data_generator, val_data_generator, train_data, val_data, steps_per_epoch, val_steps = fold_generator(fold)

    print(train_data['target'].value_counts())
    print(val_data['target'].value_counts())
    
    checkpoint = ModelCheckpoint('../working/Model_fold_'+ str(fold)+'.h5', 
                             monitor='val_loss', 
                             verbose= 1,save_best_only=True, 
                             mode= 'min',save_weights_only=True,save_freq = 'epoch')
    csv_logger = CSVLogger('../working/history.csv')


    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 1)
    callbacks = [checkpoint, csv_logger,lr_schedule,early_stopping]
    
    if not infer:
        train_history = model.fit_generator(
            train_data_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_data_generator,
            validation_steps = val_steps,
            epochs=5,
            callbacks = callbacks

        )
    else:
        pass
        


# ## Prediction

# In[ ]:


root =  '/kaggle/input/resize-jpg-siimisic-melanoma-classification/640x640/'
test_images  = os.path.join(root ,'test/')


# In[ ]:


df_test = pd.DataFrame({
    'image_name': os.listdir(test_images)
})

df_test['image_name'] = df_test['image_name'].str.split('.').str[0]
print(df_test.shape)
df_test.head()


# The final probability is calculated as the average of the probability of each of the 10 fold models

# In[ ]:


# calling test generator

steps_per_epoch = math.ceil(df_test.shape[0]/batch_size)


final_pred = np.zeros((df_test.shape[0],1))
print(final_pred.shape)
for fold in range(n_splits):
    model.load_weights('../input/trainedfinal/Model_fold_'+str(fold)+'.h5')
    test_generator = data_generator(test_images,df_test,batch_size,steps_per_epoch,False)
    pred = model.predict(test_generator,steps=np.ceil(float(len(df_test)) / float(batch_size)), 
                                  verbose=1) 
    
    print(pred.shape)
    final_pred = final_pred + pred


# In[ ]:


final_pred = final_pred/n_splits


# In[ ]:


print(type(final_pred))
print(final_pred.shape)


# Submitting the csv file

# In[ ]:


df_test['target'] = final_pred


# In[ ]:


df_test.to_csv('submission.csv', index=False)
df_test.head()

