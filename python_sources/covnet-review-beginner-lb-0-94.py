#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install imgaug')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, shutil
import keras
from keras import layers, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Conv2D,MaxPooling2D, Dropout, Dense, BatchNormalization, Flatten
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from keras.optimizers import Nadam

import imgaug as ia
import imgaug.augmenters as iaa

import tensorflow as tf
tf.set_random_seed(0)

import matplotlib.pyplot as plt 
from glob import glob
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(101)

# Any results you write to the current directory are saved as output.


# In[ ]:


os.listdir('../input/')


# In[ ]:


train_dir = '../input/train/'
data = pd.DataFrame({'path': glob(os.path.join(train_dir,'*.tif'))                    
                    })
data.head()


# In[ ]:


data['id'] = data.path.apply(lambda x: str(x).split('/')[3].split('.')[0])
data.head()


# In[ ]:


df = pd.read_csv('../input/train_labels.csv')
data = data.merge(df, on='id')
data.head()


# In[ ]:


training_dir= '../training_dir'
validation_dir= '../validation_dir'
os.mkdir(training_dir)
os.mkdir(validation_dir)


# In[ ]:


df = data
df_0 = df[df.label==0]
df_1 = df[df.label==1]
df_1.head()


# In[ ]:


print(df_1.shape)
df_0.shape


# In[ ]:


from tqdm import tqdm
categories = [0,1]
for category in categories:
    os.mkdir(os.path.join(training_dir,str(category))) #../training_dir/0 or 1
    os.mkdir(os.path.join(validation_dir,str(category)))
    
 # CREATING TRAINING DIRECTORY            
for category in tqdm(categories):
    cdir= os.path.join(training_dir,str(category)) #creates '../1 or 0'
    for sample_count, path in enumerate(df[df.label ==category].path):
        id = path.split('/')[3] #generate destination id_name
        src = path
        dst = os.path.join(cdir,id) #destination
        shutil.copyfile(src,dst)
        if sample_count==70000: break
        


# In[ ]:


#CREATING VALIDATION DIRECTORY
for category in tqdm(categories):
    cdir= os.path.join(validation_dir,str(category)) #creates '../1 or 0'
    for sample_count, path in tqdm(enumerate(df[df.label ==category].path)):
        if sample_count>70000:
            id = path.split('/')[3] #generate destination id_name
            src = path
            dst = os.path.join(cdir,id) #destination
            shutil.copyfile(src,dst)
            if sample_count==89000: break
        else: continue


# In[ ]:


len(os.listdir(os.path.join(validation_dir,'1')))


# In[ ]:


os.listdir(validation_dir)


# In[ ]:


def myaug():
    sometimes = lambda augmenter: iaa.Sometimes(0.5)
    seq = iaa.Sequential([iaa.Fliplr(0.5),
                          iaa.Flipud(0.5),
                          iaa.GaussianBlur(sigma=(0,2.0)),
                          sometimes(iaa.Affine(scale={'x': (0.9,1.1), 'y': (0.9,1.1)},
                                    translate_percent={'x': (-0.9,1.1), 'y': (-0.9,1.1)},
                                   rotate=(20),
                                   shear=(-5,5)
                                              ))], random_order=True)


# In[ ]:


# data_gen = ImageDataGenerator(rotation_range=40,
#                           rescale=1./255, width_shift_range=0.2, 
#                               height_shift_range=0.2, 
#                              shear_range=0.2,
#                              zoom_range=0.2, horizontal_flip=True,
#                              fill_mode='nearest')
data_gen = ImageDataGenerator(preprocessing_function=myaug(), rescale=1./255, width_shift_range=0.2, 
                              height_shift_range=0.2,
                             zoom_range=0.2,
                             fill_mode='nearest')


# In[ ]:


class Generator():
    
    """ ATTRIBUTES
    =========================
    Generator = (directory,batch_size)  
    directory: Input required directory
    baych_size = input batch_size
    
    """
        
    def __init__(self, directory,batch_size):
        self.directory = directory
        self.batch_size = batch_size
        
        if self.batch_size ==1: raise ValueError("Batch_size must be greater than 1")

    def generator(self):
        import cv2
        self.filenames = []
        batch_stop=0 #Initializing to Offset modulus-zero
        batch=[]
        for i, file in enumerate(os.listdir(self.directory)):
            img_path = os.path.join(self.directory,file)
            img_array = cv2.imread(img_path)
            img_array=img_array/255
            self.filenames.append(file)
            batch.append(img_array)
            if i>0: batch_stop= i+1  

            batch_catalyst = batch_stop%(self.batch_size)        
            if batch_catalyst == 0 and batch_stop>0:
                yield np.array(batch)
                batch=[]


# In[ ]:


batch_size=20
train_generator = data_gen.flow_from_directory(training_dir, 
                                               target_size=(96,96),
                                               class_mode='binary',
                                               batch_size=batch_size)
test_gen = ImageDataGenerator(rescale=1./255)

validation_generator_shuffled=test_gen.flow_from_directory(validation_dir,
                                                 target_size=(96,96),
                                                 class_mode='binary',
                                                 batch_size=batch_size)

validation_generator=test_gen.flow_from_directory(validation_dir,
                                                 target_size=(96,96),
                                                 class_mode='binary',
                                                 batch_size=batch_size,
                                                 shuffle=False)


# In[ ]:


pool_size= (2,2)

model = Sequential()
model.add(layers.Conv2D(32,3,input_shape=(96,96,3), activation='relu'))
model.add(layers.Conv2D(32,3,activation='relu'))
model.add(layers.Conv2D(32,3,activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size))
# model.add(Dropout(0.3))

model.add(layers.Conv2D(64,3,activation='relu'))
model.add(layers.Conv2D(64,3,activation='relu'))
model.add(layers.Conv2D(64,3,activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size))
# model.add(Dropout(0.3))

model.add(layers.Conv2D(128,3,activation='relu'))
model.add(layers.Conv2D(128,3,activation='relu'))
model.add(layers.Conv2D(128,3,activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size))
# model.add(Dropout(0.3))

model.add(layers.Conv2D(256,3,activation='relu', padding='same'))
model.add(layers.Conv2D(256,3,activation='relu', padding='same'))
model.add(layers.Conv2D(256,3,activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size))
# model.add(Dropout(0.3))

model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(optimizer=optimizers.Nadam(), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# In[ ]:





# In[ ]:


train_samples = 140000
checkpoint_name= '../working/model.h5'
checkpoint = ModelCheckpoint(checkpoint_name, save_best_only=True)
stop = EarlyStopping(patience=10, verbose=1)
LrReduce = ReduceLROnPlateau(verbose=1, patience=4)
model.fit_generator(train_generator, 
                    steps_per_epoch=train_samples//batch_size,
                   validation_data=validation_generator_shuffled,
                   epochs=25,validation_steps=len(validation_generator.classes)//batch_size,
                   callbacks=[checkpoint,stop, LrReduce])


# In[ ]:


history = model.history.history


# In[ ]:


plt.plot(model.history.epoch, history['acc'],label='training_acc')
plt.plot(model.history.epoch, history['val_acc'],c='green', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.legend()


# In[ ]:


plt.plot(model.history.epoch, history['loss'],label='training_loss')
plt.plot(model.history.epoch, history['val_loss'],c='green', label='Validation Loss')
plt.xlabel('Epochs')
plt.legend()


# In[ ]:


os.listdir('../working')


# In[ ]:


from keras.models import load_model
predictor = load_model(checkpoint_name)


# In[ ]:


validation_generator.filenames[:5]


# In[ ]:


test_dir = '../input/test'
test_batch_size = 2
test_generator = Generator(directory=test_dir, batch_size=test_batch_size)


# In[ ]:


no_of_samples = len(os.listdir(test_dir))
predictions = predictor.predict_generator(test_generator.generator(),
                                          steps=no_of_samples//test_batch_size, verbose=1)


# In[ ]:


df_test= pd.DataFrame()
df_test['id'] = test_generator.filenames
df_test.head()


# In[ ]:


for bb in test_generator:
    plt.imshow(bb[0])
    break


# In[ ]:


predictions


# In[ ]:


results = pd.DataFrame({'label': predictions.reshape(-1,)}, index=range(0,no_of_samples))
results.head()


# In[ ]:


# def ro(x):
#     if x>=0.5: x=1
#     else: x=0
#     return x

dd = results
dd.label = dd.label.apply(round)
dd.label.value_counts()


# In[ ]:


results = pd.DataFrame({'label': predictions.reshape(-1,)}, index=range(0,no_of_samples))
results.head()


# In[ ]:


# df_test.drop('path', axis=1, inplace=True)
df_test.id = df_test.id.apply(lambda x: x.split('.')[0])
df_test.head()


# In[ ]:


submission = pd.concat([df_test,results],axis=1)
# submission2 = pd.concat([df_test,dd],axis=1)
submission.head(10)


# In[ ]:


submission.to_csv('submissions.csv', index=False)
# submission2.to_csv('submissionswhole.csv', index=False)


# In[ ]:


pd.read_csv('submissions.csv').head()


# In[ ]:


# def roundup(x):
#     if x>=0.5: x = 1
#     else: x=0
#     return x


# In[ ]:


# results.label.apply(roundup).value_counts()


# In[ ]:


# val_batch_labels=np.zeros(len(validation_generator.classes))


# In[ ]:


# i = 0
# for b, l in (validation_generator):
#     val_batch_labels[i*batch_size:batch_size*(i+1)] = l
#     i+=1
#     if i==3000:
#         break


# In[ ]:


# print('done')
# pd.DataFrame(val_batch_labels)[0].value_counts()


# In[ ]:



from sklearn.metrics import roc_curve,auc, confusion_matrix, classification_report
val_pred = predictor.predict_generator(validation_generator,
                                      steps=len(validation_generator.classes)//batch_size,
                                      verbose=1)


# In[ ]:


val_pred


# In[ ]:


false_positive_rate,true_positive_rate,threshold = roc_curve(validation_generator.classes,
                                                            val_pred)


# In[ ]:


val_pred_whole = np.where(val_pred>=0.5,1,0)
AUC = auc(false_positive_rate,true_positive_rate)
print(AUC)
print(classification_report(validation_generator.classes,val_pred_whole))


# In[ ]:


dg = pd.DataFrame(val_pred_whole, columns=['label'])
dg.label.value_counts()


# In[ ]:


validation_generator.classes


# In[ ]:


shutil.rmtree(validation_dir)


# In[ ]:


# os.mkdir('../test_dir')


# In[ ]:


# test_path= '../input/test'
# test_dir='../test_dir'
# # for image in os.listdir(test_path):
# for img in os.listdir(test_path):
#     src= os.path.join(test_path,img)
#     dst = os.path.join(test_dir,img)
#     shutil.copyfile(src,dst)
# print('done')
# len(os.listdir(test_dir))
    
    


# In[ ]:


# os.mkdir('../test_images')
# test_image_path = '../test_images'
# shutil.move(test_dir, test_image_path)


# In[ ]:


# test_generatorr = test_gen.flow_from_directory(test_image_path,target_size=(96,96),
#                                               shuffle=False, batch_size=test_batch_size,
#                                               class_mode='binary')


# In[ ]:


# test_image_path= os.path.join(test_image_path,'test_dir')
# os.listdir(test_image_path)[:5]


# In[ ]:


# data = pd.DataFrame({'path': glob(os.path.join(test_image_path,'*.tif'))                    
#                     })
# data.head()

# data['id'] = data.path.apply(lambda x: str(x).split('/')[3].split('.')[0])
# data.head()


# In[ ]:


# new_pred= predictor.predict_generator(test_generatorr, 
#                                       steps=len(test_generatorr.classes)//test_batch_size,
#                                     verbose=1)


# In[ ]:


# results = pd.DataFrame({'label': new_pred.reshape(-1,)}, index=range(0,no_of_samples))
# results.head()


# In[ ]:


# results.label.apply(roundup).value_counts()


# In[ ]:




