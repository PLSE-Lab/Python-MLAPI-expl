#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


np.random.seed(2019)
tf.set_random_seed(2019)


# In[ ]:


os.listdir('../input/')


# In[ ]:


train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
submission = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')


# In[ ]:


train_path = '../input/aptos2019-blindness-detection/train_images/'
test_path = '../input/aptos2019-blindness-detection/test_images/'

train_ids = train['id_code'].values
test_ids = test['id_code'].values

train_paths = []
for train_id in train_ids:
    image = train_id + '.png'
    path = os.path.join(train_path,image)
    train_paths.append(path)
    
train_paths = np.array(train_paths)
train['path'] = train_paths


# In[ ]:


test_paths = []
for test_id in test_ids:
    image = test_id + '.png'
    path = os.path.join(test_path,image)
    test_paths.append(path)
    
test_paths = np.array(test_paths)
test['path'] = test_paths


# In[ ]:


def find_radius(mid_pixels,mid_y_pixels,threshold_x,threshold_y):
    
    start_x = 0
    end_x = mid_pixels.shape[0] - 1
    
    start_y = 0
    end_y = mid_y_pixels.shape[0] - 1
    
    while True:
        if np.sum(mid_pixels[start_x,:])>threshold_x:
            break
        start_x +=1
    while True:
        if np.sum(mid_pixels[end_x,:])>threshold_x:
            break
        end_x -= 1
        
    while True:
        if np.sum(mid_y_pixels[start_y,:])>threshold_y:
            break
        start_y +=1
    while True:
        if np.sum(mid_y_pixels[end_y,:])>threshold_y:
            break
        end_y -= 1
        
    return start_x,end_x,start_y,end_y
    
    
    
def preprocess_image(img):
    mid = img.shape[1]//2
    mid_pixels = img[mid,:]
    mid_y_pixels = img[:,mid]
    threshold_x = np.mean(mid_pixels)
    threshold_y = np.mean(mid_y_pixels)
    startx,endx,starty,endy = find_radius(mid_pixels,mid_y_pixels,threshold_x,threshold_y)
    return cv2.resize(img[starty:endy,startx:endx],(img.shape[0],img.shape[1]))


# In[ ]:


from sklearn.model_selection import train_test_split
train_df,validation_df = train_test_split(train,test_size = 0.2,stratify=train['diagnosis'].values,random_state = 42)
print (len(train_df),len(validation_df))


# In[ ]:


from collections import Counter
def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: round(float(majority)/float(count), 2) for cls, count in counter.items()}

class_weights = get_class_weights(train_df['diagnosis'].values)
class_weights


# In[ ]:


datagen = ImageDataGenerator(
        zoom_range=0.2,
        rescale = 1./255,
        fill_mode = 'constant',
        horizontal_flip = True,
        vertical_flip = True,
        preprocessing_function = preprocess_image
)


# In[ ]:


train_df['id'] = train_df['id_code'].apply(lambda x: str(x)+'.png')
train_df['diagnosis'] = train_df['diagnosis'].apply(lambda x:str(x))

train_generator = datagen.flow_from_dataframe(
dataframe=train_df,
directory="../input/aptos2019-blindness-detection/train_images/",
x_col="id",
y_col="diagnosis",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
color_mode = 'rgb',
target_size=(224,224))


# In[ ]:


validation_df.head()


# In[ ]:


validation_x = []
for path in tqdm(validation_df['path'].values):
    img = cv2.resize(cv2.imread(path),(224,224))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    validation_x.append(img)
    


# In[ ]:


validation_x = np.array(validation_x)
validation_x = validation_x.astype(np.float32)/255.0
print (validation_x.shape)
print (np.amin(validation_x),np.amax(validation_x))


# In[ ]:


from keras.utils import to_categorical
validation_y = to_categorical(validation_df['diagnosis'].values,5)
print (validation_y.shape)
print (validation_y[:5])


# # Model: DenseNet-121

# In[ ]:


densenet = DenseNet121(
    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)


# In[ ]:


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='softmax'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    
    return model


# In[ ]:


model = build_model()
model.summary()


# # Training & Evaluation

# In[ ]:


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[0],self.validation_data[1]
                
        y_val = np.argmax(y_val,axis=1)
        
        y_pred = self.model.predict(X_val)
        y_pred = np.argmax(y_pred,axis=1)

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return


# In[ ]:


kappa_metrics = Metrics()

history = model.fit_generator(train_generator,validation_data = (validation_x,validation_y),
                              epochs = 10,steps_per_epoch = len(train_df)/32,callbacks = [kappa_metrics],verbose=1,
                              class_weight = class_weights)


# In[ ]:


history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


# In[ ]:


plt.plot(kappa_metrics.val_kappas)


# In[ ]:


from keras.models import load_model
model = load_model('model.h5')
model.evaluate(validation_x,validation_y)


# In[ ]:



test['id'] = test['id_code'].apply(lambda x: str(x)+'.png')
test.head()


# In[ ]:



test_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,vertical_flip=True)

test_generator=test_datagen.flow_from_dataframe(dataframe=test,
                                                directory = test_path,
                                                x_col="id",
                                                target_size=(224,224),
                                                batch_size=1,
                                                shuffle=False, 
                                                class_mode=None, 
                                                seed=42)


# In[ ]:


tta_steps = 5
preds_tta=[]
for i in tqdm(range(tta_steps)):
    test_generator.reset()
    preds = model.predict_generator(test_generator,steps = test.shape[0])
    preds_tta.append(preds)


# In[ ]:


final_pred = np.mean(preds_tta, axis=0)
predicted_class_indices = np.argmax(final_pred, axis=1)
Counter(predicted_class_indices)


# In[ ]:


submission.head()


# In[ ]:


submission['diagnosis'] = predicted_class_indices
submission.to_csv('submission.csv',index=False)


# In[ ]:


submission.head()


# In[ ]:




