#!/usr/bin/env python
# coding: utf-8

# # SetNet Trainer
# Trains an MobileNetV2++ network to recognize [Set](https://www.setgame.com/) cards. The dataset is my own, made with the tools in [this repository](https://github.com/npeirson/SetAI).

# In[ ]:


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Input,Dense,Flatten,GaussianNoise,LeakyReLU
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.image import ImageDataGenerator,array_to_img
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from keras.utils import to_categorical


# In[ ]:


def get_set(path):
    data = pd.DataFrame()
    for dirpath, subdirs, files in os.walk(path):
        for file in files:
            filepath = os.path.abspath(os.path.join(dirpath,file))
            strpath = str(filepath).split('/')
            class_num = strpath[5]
            class_color = strpath[6]
            class_shape = strpath[7]
            class_shade = strpath[8]
            row = pd.Series({'filepath':filepath,
                             'class_color':class_color,
                             'class_num':class_num,
                             'class_shape':class_shape,
                             'class_shade':class_shade})
            data = data.append(row,ignore_index=True)
    return data

data_panda = get_set('../input/set-cards/set_dataset')
data_panda.head()


# ## simple numeric encoding
# - color: R,G,B
# - num: one,two,three
# - shade: empty,partial,full
# - shape: diamond,oval,squiggle

# In[ ]:


mlb = MultiLabelBinarizer(classes=('red','green','blue','one','two','three','empty','partial','full','diamond','oval','squiggle'))
labels = mlb.fit_transform(data_panda[data_panda.columns[:-1]].values)
data_panda = pd.concat([data_panda,pd.DataFrame(labels)],axis=1)
del data_panda['class_color']
del data_panda['class_num']
del data_panda['class_shade']
del data_panda['class_shape']
display(data_panda.head())
data_panda = data_panda.sample(frac=1).reset_index(drop=True)


# ## Play in either alignment
# Whether cards are laid out in portrait or landscape. Probably unnecessary, really.

# In[ ]:


def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width/2, height/2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def check_rotation(temp_img):
    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
    if (temp_img.shape[0] > temp_img.shape[1]):
        temp_img = rotate_image(temp_img,90)
    return temp_img


# ## Data generators

# In[ ]:


datagen = ImageDataGenerator(shear_range=12,
                             rotation_range=24,
                             zoom_range=(0.8,1.2),
                             brightness_range=(0.6,1.2),
                             horizontal_flip=True,
                             vertical_flip=True,
                             validation_split=0.5,
                             preprocessing_function=check_rotation)

train_data = datagen.flow_from_dataframe(data_panda,
                                         x_col='filepath',
                                         y_col=data_panda.columns[1:],
                                         class_mode='other',
                                         target_size=(64,100),
                                         subset='training',
                                         shuffle=False) # bugged, drops a class :/
val_data = datagen.flow_from_dataframe(data_panda,
                                         x_col='filepath',
                                         y_col=data_panda.columns[1:],
                                         class_mode='other',
                                         target_size=(64,100),
                                         subset='validation',
                                         shuffle=False)

display(array_to_img(next(train_data)[0][2])) # just to see it


# In[ ]:


# this function splits the generator's labels into arrays by classification, for softmax heads
class split_outputs:
    def __init__(self,generator):
        self.generator = generator
        
    def __iter__(self):
        return self
        
    def __next__(self):
        feed = next(self.generator)
        images,labels = feed
        labels_color = labels[:,:3]
        labels_num = labels[:,3:6]
        labels_shade = labels[:,6:9]
        labels_shape = labels[:,9:]
        return (images,[labels_color,labels_num,labels_shade,labels_shape])
    
train_split = split_outputs(train_data)
val_split = split_outputs(val_data)


# ## The model

# In[ ]:


K.clear_session()

# callbacks
model_ckpt = ModelCheckpoint('SetNet_0901.h5',save_best_only=True,verbose=True)
reduce_lr = ReduceLROnPlateau(patience=5,verbose=True)
early_stop = EarlyStopping(patience=9,verbose=True)

# base model
input_tensor = GaussianNoise(0.4)(Input(shape=(64,100,3)))
base_model = MobileNetV2(include_top=False,input_tensor=input_tensor,pooling='avg')
x = base_model.layers[-1].output


# In[ ]:


def softmax_head(x,name):
    x = LeakyReLU(0.4)(x)
    x = Dense(666,activation='relu',kernel_regularizer=l2(0.0001))(x)
    x = LeakyReLU(0.3)(x)
    x = Dense(69,activation='relu')(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(3,activation='softmax',name=name)(x)
    return x

color = softmax_head(x,'color')
number = softmax_head(x,'number')
shade = softmax_head(x,'shade')
shape = softmax_head(x,'shape')

model = Model(base_model.input,[color,number,shade,shape])
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=5e-4,clipnorm=5),
              metrics=['accuracy'])
#display(model.summary())


# In[ ]:


history = model.fit_generator(train_split,validation_data=val_split,
                              steps_per_epoch=100,validation_steps=60,
                              callbacks=[model_ckpt,reduce_lr,early_stop],
                              epochs=666,verbose=2)


# ## Visualize training

# In[ ]:


plt.plot(history.history['val_color_acc'])
plt.plot(history.history['val_number_acc'])
plt.plot(history.history['val_shade_acc'])
plt.plot(history.history['val_shape_acc'])
plt.title('Validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Color', 'Number', 'Shade', 'Shape'], loc='lower right')
plt.show()

plt.plot(history.history['val_color_loss'])
plt.plot(history.history['val_number_loss'])
plt.plot(history.history['val_shade_loss'])
plt.plot(history.history['val_shape_loss'])
plt.title('Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Color', 'Number', 'Shade', 'Shape'], loc='upper right')
plt.show()


# ## test predict

# In[ ]:


test_img = cv2.imread('../input/set-cards/set_dataset/three/blue/oval/partial/440.png',1)
test_img = check_rotation(test_img)
result = model.predict(np.expand_dims(cv2.resize(test_img,(100,64)),0))

color = ['blue','green','red'][np.argmax(result[0][:])]
num = ['one','two','three'][np.argmax(result[1][:])]
shade = ['empty','partial','full'][np.argmax(result[2][:])]
shape = ['diamond','oval','squiggle'][np.argmax(result[3][:])]
display(array_to_img(test_img))
print(num,color,shape,shade)


# In[ ]:


from IPython.display import FileLink, FileLinks
FileLink('SetNet_0901.h5')

