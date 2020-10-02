#!/usr/bin/env python
# coding: utf-8

# # Detecting Pnemonia using Xray images [Keras] 
# 
# This is a CNN model for binary classification of x-ray images using Keras. It detects if a person is having Pnemonia or not. 
# 
# This model is ideally for beginners who are getting started with CNN, Keras. Experienced data scientist can also help im improving the accurac of model and extend its usability to multiclass classification.
# 
# 
# This model is trained over [ dataset ](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset) provided by [Praveen](https://www.kaggle.com/praveengovi) which contains collection of Chest X Ray of Healthy vs Pneumonia (Corona) affected patients infected patients along with few other categories such as SARS (Severe Acute Respiratory Syndrome ) ,Streptococcus & ARDS (Acute Respiratory Distress Syndrome)
# 
# Distribution of no of images belonging to different category are as follow:
# ![distribution of no of images belonging to different category](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F378285%2Fcfdeda929ebe5e6254590538601c0ef6%2FChest_XRay_dataset_labels.png?generation=1584770009221937&alt=media)
# 

# In[ ]:


import shutil
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[ ]:





# In[ ]:


# Directory 
directory = "/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/" 

output = "/kaggle/working/"+'data/'


# In[ ]:


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
#         for f in files:
#             print('{}{}'.format(subindent, f))


# In[ ]:


print("Directory structure of Dataset : ")
list_files("/kaggle/input/")


# In[ ]:


print("Xray image of Normal person")
img=mpimg.imread(directory+'train/IM-0343-0001.jpeg')
imgplot = plt.imshow(img)
plt.show()
print("Xray image of infected person")
img=mpimg.imread(directory+'train/person1019_virus_1708.jpeg')
imgplot = plt.imshow(img)
plt.show()


# ### Metadata
# additional details of each x-ray is defined in Chest_xray_Corona_Metadata.csv file.
# 
# It contains labels of Normal/Pnemonia, Train/Test and high order cause for Pnemonia.
# 
# We need to saperate all images in 4 different directory as
# <pre>
# train/
#     Normal/
#     Pnemonia/
# test/
#     Normal/
#     Pnemonia/
# </pre>

# In[ ]:


import pandas as pd
filename = "../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv"
df = pd.read_csv(filename)
df.head()


# In[ ]:


# Filtering images belonging to Training set and test set

train = df[df['Dataset_type']=='TRAIN' ] 
test = df[df['Dataset_type']=='TEST'] 


# In[ ]:





# In[ ]:


# Creating folders to store images saperately
try:
    os.mkdir(output)

    os.mkdir(output+'/train')
    os.mkdir(output+'/test')

    os.mkdir(output+'train/Normal')
    os.mkdir(output+'train/Pnemonia')

    os.mkdir(output+'test/Normal')
    os.mkdir(output+'test/Pnemonia')
except Exception as e :
    print(e)


# In[ ]:


print("Directory structure of Custom output : ")
list_files(output)


# In[ ]:


# Saperating data into 4 different categories 

trainNormal = train[train['Label']=="Normal"]
trainPne = train[train['Label']=="Pnemonia"]

testNormal = test[test['Label']=="Normal"]
testPne = test[test['Label']=="Pnemonia"]


# In[ ]:


print("Training Normal samples : ", trainNormal.shape)
print("Training Pnemonia samples : ", trainPne.shape)
print("Test Normal samples : ", testNormal.shape)
print("Test Pnemonia samples : ", testPne.shape)


# ### copying images to respective folders

# In[ ]:



for index,row in testPne.iterrows():
    dest = output+'test/'+'Pnemonia/'+row['X_ray_image_name']
    src = directory+'test/'+row['X_ray_image_name']
    shutil.copyfile(src, dest)


# In[ ]:



for index,row in testNormal.iterrows():
    dest = output+'test/'+'Normal/'+row['X_ray_image_name']
    src = directory+'test/'+row['X_ray_image_name']
    shutil.copyfile(src, dest)


# In[ ]:



for index,row in trainNormal.iterrows():
    dest = output+'train/'+'Normal/'+row['X_ray_image_name']
    src = directory+'train/'+row['X_ray_image_name']
    shutil.copyfile(src, dest)


# In[ ]:



for index,row in trainPne.iterrows():
    dest = output+'train/'+'Pnemonia/'+row['X_ray_image_name']
    src = directory+'train/'+row['X_ray_image_name']
    shutil.copyfile(src, dest)


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest'
                                  )
test_datagen = ImageDataGenerator(rescale=1./255)


# In[ ]:


train_dir=output+'train'
test_dir=output+'test'


# In[ ]:





# In[ ]:




# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


BATCH_SIZE = 16 * strategy.num_replicas_in_sync # this is 8 on TPU v3-8, it is 1 on CPU and GPU

train_gen = train_datagen.flow_from_directory(train_dir,
                                             target_size=(150, 150),
                                             batch_size=BATCH_SIZE,
                                             class_mode='binary'
                                             )
test_gen = test_datagen.flow_from_directory(test_dir,
                                             target_size=(150, 150),
                                             batch_size=BATCH_SIZE,
                                             class_mode='binary'
                                             )


# In[ ]:


from keras import layers
from keras import models
from keras import optimizers


# In[ ]:





# In[ ]:



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'
                      ])


# In[ ]:


model.summary()


# In[ ]:




history = model.fit_generator(
    train_gen,
    steps_per_epoch=260,
    epochs=20,
    validation_data=test_gen,
    validation_steps=30)


# In[ ]:


try:
    os.mkdir('/kaggle/working/model')
    model.save('/kaggle/working/model/P.h5')
except Exception as e:
    os.mkdir('/kaggle/working/model')
    print(e)


# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:


from keras.models import load_model
model = load_model('/kaggle/working/model/P.h5')
model.summary() 


# In[ ]:




