#!/usr/bin/env python
# coding: utf-8

# **Install package**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers,models
from keras.preprocessing.image import img_to_array
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import os,shutil
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
from keras.preprocessing.image import image
from keras import layers,models,optimizers
import math


# dataset dir 

# In[ ]:


#base_dir="../input"/
train_images_dir="../input/train_images"
test_images_dir="../input/test_images"
train_df=pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")


# In[ ]:


train_df.diagnosis.value_counts()


# In[ ]:


train_df.hist()


# In[ ]:


conv_base=VGG16(weights='imagenet',
               include_top=False,
               input_shape=(728,728,3))
conv_base.trainable=False

model=models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu',input_dim=22*22*512))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(5,activation='softmax'))


# In[ ]:


batch_size=32
seed=666

train_df.id_code=train_df.id_code.apply(lambda x :x+'.png')
test_df.id_code=test_df.id_code.apply(lambda x :x+'.png')
train_df['diagnosis']=train_df['diagnosis'].astype(str)
x_train,x_val=train_test_split(train_df,test_size=0.2,random_state=seed)
train_datagen=image.ImageDataGenerator(rescale=1./255,
                                       horizontal_flip=True,rotation_range=40,
                                       width_shift_range=0.2,shear_range=0.2,
                                      zoom_range=0.2,fill_mode='nearest')

val_datagen=image.ImageDataGenerator(rescale=1./255)
    
test_datagen=image.ImageDataGenerator(rescale=1./255)


# In[ ]:


train_generator = train_datagen.flow_from_dataframe(
    dataframe=x_train, 
    directory=train_images_dir,
    x_col='id_code',
    y_col='diagnosis',
    target_size=(728,728),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=batch_size,
    seed=seed)
validation_generator = val_datagen.flow_from_dataframe(
    dataframe=x_val, 
    directory=train_images_dir,
    x_col='id_code',
    y_col='diagnosis',
    target_size=(728,728),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False,
    seed=seed)
test_generator=test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_images_dir,
    x_col='id_code',
    y_col=None,
    target_size=(728,728),
    color_mode='rgb',
    class_mode=None,
    batch_size=batch_size,
    shuffle=False,
    seed=seed)


# In[ ]:


model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
             loss='categorical_crossentropy',
             metrics=['acc'])
history=model.fit_generator(train_generator,  
                steps_per_epoch=math.ceil(len(x_train)/batch_size),
                 epochs=20,
                 validation_data=validation_generator,
                 validation_steps=32
                         )


# In[ ]:


acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='training acc')
plt.plot(epochs,val_acc,'r',label='val acc')
plt.title('training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='training loss')
plt.plot(epochs,val_loss,'r',label='val loss')
plt.title('training and validation loss')
plt.legend()
plt.show()


# In[ ]:


step=math.ceil(len(test_df)/batch_size)
test_generator.reset()
test_pre=model.predict_generator(test_generator,steps=step)


# In[ ]:


# predicted_class_indices = np.argmax(test_pre_array, axis=1)
# labels = (train_generator.class_indices)
# label = dict((v,k) for k,v in labels.items())


# because of my this part no work,i found [this kernel](http://www.kaggle.com/bharatsingh213/keras-resnet-test-time-augmentation) to test my predict.

# In[ ]:


from tqdm import tqdm
tta_steps = 10
step=math.ceil(len(test_df)/batch_size)
preds_tta=[]
for i in tqdm(range(tta_steps)):
    test_generator.reset()
    preds = model.predict_generator(generator=test_generator,steps =step)
#     print('Before ', preds.shape)
    preds_tta.append(preds)
#     print(i,  len(preds_tta))


# In[ ]:


preds_tta


# In[ ]:


final_pred = np.mean(preds_tta, axis=0)
predicted_class_indices = np.argmax(final_pred, axis=1)
len(predicted_class_indices)


# In[ ]:


test_df['diagnosis']=predicted_class_indices.astype(int)
test_df.to_csv('submission.csv',index=False)


# In[ ]:


test_df


# In[ ]:


test_df.diagnosis.value_counts()


# when i saw [this kernel](http://www.kaggle.com/c/aptos2019-blindness-detection/discussion/101366#584009),i decided to use MixUp loss training my model.i wonder if its magical.

# In[ ]:


# import torch.nn as nn
# class MixUpSoftmaxLoss(nn.Module):

#     def __init__(self, crit, reduction='mean'):
#         super().__init__()
#         self.crit = crit
#         setattr(self.crit, 'reduction', 'none')
#         self.reduction = reduction

#     def forward(self, output, target):
#         if len(target.size()) == 2:
#             loss1 = self.crit(output, target[:, 0].long())
#             loss2 = self.crit(output, target[:, 1].long())
#             lambda_ = target[:, 2]
#             d = (loss1 * lambda_ + loss2 * (1-lambda_)).mean()
#         else:
#             # This handles the cases without MixUp for backward compatibility
#             d = self.crit(output, target)
#         if self.reduction == 'mean':
#             return d.mean()
#         elif self.reduction == 'sum':
#             return d.sum()
#         return d`


# In[ ]:


import gc
gc.collect()

