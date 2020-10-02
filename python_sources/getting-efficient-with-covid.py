#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Greetings Everyone this is showing how GANs improve accuracy, I am using Efficient Net which is an SOTA Image classifier and Compare the results.

# ## What is GAN?
# A generative adversarial network (GAN) has two parts:
# 
# * The generator learns to generate plausible data. The generated instances become negative training examples for the discriminator.
# * The discriminator learns to distinguish the generator's fake data from real data. The discriminator penalizes the generator for producing implausible results.
# 
# 
# When training begins, the generator produces obviously fake data, and the discriminator quickly learns to tell that it's fake:
# 
# ![](https://developers.google.com/machine-learning/gan/images/bad_gan.svg)
# 
# As training progresses, the generator gets closer to producing output that can fool the discriminator:
# 
# ![](https://developers.google.com/machine-learning/gan/images/ok_gan.svg)
# 
# Finally, if generator training goes well, the discriminator gets worse at telling the difference between real and fake. It starts to classify fake data as real, and its accuracy decreases.
# 
# ![](https://developers.google.com/machine-learning/gan/images/good_gan.svg)
# 
# 
# 

# # Libraries

# In[ ]:



import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))


# In[ ]:


get_ipython().system('pip install -U --pre efficientnet')
import efficientnet.keras as efn


# # Data loading and pipelining
# 

# Let's Load data First

# In[ ]:


PATH='../input/chest_xray/'
training_path=os.path.join(PATH, 'chest_xray/train')
testing_path=os.path.join(PATH, 'chest_xray/test')


# In[ ]:


bs=8
img_size=384
epoch=5


# In[ ]:


train_aug=ImageDataGenerator(rescale=1./255,validation_split=0.2,rotation_range=15,fill_mode='nearest')
test_aug=ImageDataGenerator(rescale=1./255)


# In[ ]:


train_gen=train_aug.flow_from_directory(batch_size=bs,directory=training_path,shuffle=True,target_size=(img_size,img_size),class_mode='categorical')
valid_gen=train_aug.flow_from_directory(batch_size=bs,directory=training_path,shuffle=True,target_size=(img_size,img_size),class_mode='categorical',subset='validation')
test_gen=test_aug.flow_from_directory(batch_size=1,directory=testing_path,shuffle=False,target_size=(img_size,img_size),class_mode='categorical')


# Here is our metrics which would observe during each epoch

# In[ ]:


matrix=["accuracy",
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        #keras.metrics.BinaryAccuracy(name='bin_accuracy'),
        #keras.metrics.Precision(name='precision'),
        #keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]


# # Model

# I am fine tuning Efficient net B0 , where it has many levels. B0 to B7

# In[ ]:


def eff_model():
    base_model=efn.EfficientNetB0(weights='imagenet',include_top=False,pooling='avg',input_shape=(img_size,img_size,3))
    x= tf.keras.layers.Flatten()(base_model.output)
    x =tf.keras.layers.Dense(1024,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x =tf.keras.layers.Dense(512,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x =tf.keras.layers.Dense(256,activation='relu')(x)
    x =tf.keras.layers.Dense(3,activation='softmax')(x)
    
    return tf.keras.Model(base_model.input,x)


def compiler():
    opt=tf.keras.optimizers.Adam(learning_rate=1e-3,amsgrad=True)
    model=eff_model()
    model.compile(optimizer=opt,loss=tf.keras.losses.CategoricalCrossentropy(),metrics=matrix)
    
    return model


# **Here is the model architecture**
# 
# ![](https://gitcdn.xyz/cdn/Tony607/blog_statics/36894ad880dc3e645513efc36cc070c4cd0d3d7c/images/efficientnet/building_blocks.png)

# In[ ]:


model=compiler()


# In[ ]:


chk_path='kaggle/working/weights'
model_chk_callback=tf.keras.callbacks.ModelCheckpoint(
  filepath=chk_path,
  save_weights_only=True,
  monitor='val_accuracy',
  mode='max',
  save_best_only=True,
  verbose=1,
)


# In[ ]:


reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=2,min_lr=1e-7,verbose=1)

early=tf.keras.callbacks.EarlyStopping(monitor='val_accruracy',patience=6)


# # Training Time 
# **I am taking 5 epochs only**

# In[ ]:


STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size

# Already calculated - this would make model to learn more from samples which low in number in our case - Covid
classweights={0: 4.357,
             1:1.462,
             2:0.4792}
#2.23929825 1.58637832 0.52000326
history = model.fit_generator(
    train_gen,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=5,
    validation_data=valid_gen,
    validation_steps=STEP_SIZE_VALID,
    callbacks=[model_chk_callback,reduce_lr,early],
    class_weight=classweights,
    verbose=1
)  


# In[ ]:


from sklearn.metrics import confusion_matrix
y_test =test_gen.classes
nb_samples = len(y_test)
pred = model.predict_generator(test_gen, nb_samples)
pred= np.argmax(pred, axis=1)

print(pred)


matrix = confusion_matrix(y_test, pred)
matrix = matrix.astype('float')
#cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
print(matrix)


class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
print('Sens covid: {0:.3f}, Normal: {1:.3f}, pneumonia : {2:.3f}'.format(class_acc[0],class_acc[1],class_acc[2]))
ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
print('PPV :covid {0:.3f}, Normal {1:.3f}, pneumonia: {2:.3f}'.format(ppvs[0],ppvs[1],ppvs[2]))


# ## Impressive results within 5 epochs only

# # Time to Train GAN images too

# In[ ]:


from distutils.dir_util import copy_tree

fromDir='../input'
toDir= 'temp'

copy_tree(fromDir,toDir)


# In[ ]:


gan_corona='temp/GAN_Images/corona'
train_corona='temp/chest_xray/chest_xray/train/CORONA/'
print (len([name for name in os.listdir(train_corona) if os.path.isfile(os.path.join(train_corona, name))]))


# In[ ]:


import shutil
import glob2 as glob
for filename in glob.glob(os.path.join(gan_corona, '*.*')):
    shutil.copy2(filename, train_corona)


# In[ ]:


training_path='temp/chest_xray/chest_xray/train'
testing_path='temp/chest_xray/chest_xray/test'
train_gen=train_aug.flow_from_directory(batch_size=bs,directory=training_path,shuffle=True,target_size=(img_size,img_size),class_mode='categorical')
valid_gen=train_aug.flow_from_directory(batch_size=bs,directory=training_path,shuffle=True,target_size=(img_size,img_size),class_mode='categorical',subset='validation')
test_gen=test_aug.flow_from_directory(batch_size=1,directory=testing_path,shuffle=False,target_size=(img_size,img_size),class_mode='categorical')


# In[ ]:


model_net=compiler()


# **Here it begins **

# In[ ]:


STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size

# Already calculated - this would make model to learn more from samples which low in number in our case - Covid
classweights={0: 2.23929825,
             1:1.58637832,
             2:0.52000326}


#2.23929825 1.58637832 0.52000326
history = model_net.fit_generator(
    train_gen,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=5,
    validation_data=valid_gen,
    validation_steps=STEP_SIZE_VALID,
    callbacks=[model_chk_callback,reduce_lr,early],
    class_weight=classweights,
    verbose=1
)  


# In[ ]:


from sklearn.metrics import confusion_matrix
y_test =test_gen.classes
nb_samples = len(y_test)
pred = model_net.predict_generator(test_gen, nb_samples)
pred= np.argmax(pred, axis=1)

print(pred)


matrix = confusion_matrix(y_test, pred)
matrix = matrix.astype('float')
#cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
print(matrix)


class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
print('Sens covid: {0:.3f}, Normal: {1:.3f}, pneumonia : {2:.3f}'.format(class_acc[0],class_acc[1],class_acc[2]))
ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
print('PPV :covid {0:.3f}, Normal {1:.3f}, pneumonia: {2:.3f}'.format(ppvs[0],ppvs[1],ppvs[2]))


# # Damn impressive results
# **Almost same in covid but Normal and pneumonia has increased which is an overall better technique , this has proved my point **
# 

# In[ ]:




