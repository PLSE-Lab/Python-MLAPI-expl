#!/usr/bin/env python
# coding: utf-8

# This is my first attempt to participiate in Kaggle. I am sure I have made mistakes.
# It would be nice if you read and let me know my stupidities:D

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json,cv2
from glob import glob as gb
from matplotlib import pyplot as plt
import tensorflow as tf


# In[ ]:


#global parameters
image_size = (224,224)
num_classes = 212
batch_size = 64
num_samples = 174367
stp_per_epoch = num_samples//batch_size
val_num_sample = 43592
val_stp = val_num_sample//batch_size
currupted_files = ['883572ba-21bc-11ea-a13a-137349068a90.jpg',
                  '8792549a-21bc-11ea-a13a-137349068a90.jpg',
                  '99136aa6-21bc-11ea-a13a-137349068a90.jpg',
                  '87022118-21bc-11ea-a13a-137349068a90.jpg',
                  '8f17b296-21bc-11ea-a13a-137349068a90.jpg',
                  '896c1198-21bc-11ea-a13a-137349068a90.jpg']

# for i in enumerate(currupted_files):
#     currupted_files[i[0]] = '../input/iwildcam-2020-fgvc7/train/'+i[1]
    
accelerator = None


# In[ ]:


if accelerator == 'tpu':
    # detect and init the TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    # instantiate a distribution strategy
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


# **Versions**
# 
# 0. Loading and preprocessing data
#     1. making a generator
#         1. using keras image utiles, like the document
#         2. building class dataframe from JSON file
#         3. flow from dataframe
#     2. preprocessing as simple as possible
#         1. making the sizes equal(!)
# 1. Building the model
#     1. transfer learning
#         1. resnet50
#             1. v1
#             2. v2
#         2. resnet101
#             1. v1
#             2. v2
#         3. resnet152
#             1. v1
#             2. v2
#         4. squeeznet
#         5. alexnet
#         6. inception
#         7. VGG
#         8. ZFNet
#      2. developing my own model
#         1. Create a 50 layer CNN/each layer 500 Units to see how it will work...
#         2. start for reducing

# <h2>V0.1: making a generator</h2>

# In[ ]:


#0.11loading the generators
gen_t = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
gen_v = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


# In[ ]:


#0.12building the dataframe
with open('../input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json') as j_data:
    data = json.load(j_data)
labels_dataframe = pd.DataFrame.from_dict(data['annotations'])
# labels_dataframe.head()


# In[ ]:


# '''Just a check:
# 1- make 10 random numbers
# 2- get the image corresponding to this 10 numbers in the glob list of images
# 3- show them with their cat ID in the DF
# '''
images = gb('../input/iwildcam-2020-fgvc7/train/*.jpg')
random_numbers = np.random.randint(0,len(images),3)

selected_images = [images[i] for i in random_numbers]

for i in selected_images:
    array = cv2.imread(i)
    category = labels_dataframe.loc[labels_dataframe['image_id'] == i[35:-4]]['category_id'].iloc[0]
    cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    plt.imshow(array)
    plt.title(category)
    plt.show()


# In[ ]:


#0.13flow from dataframe

#making the dataframe
labels_dataframe = labels_dataframe.drop(['id','count'],
                                        axis = 1)
#suffle the dataframe
labels_dataframe = labels_dataframe.sample(frac=1)

#split the dataframe into validation and training dataset

#finding the number of all the samples
n_total = len(labels_dataframe.index)
n_train = round(0.8 * n_total)
n_valid = n_total - n_train

#defining the dataframes
df_train = labels_dataframe.head(n_train)
df_valid = labels_dataframe.tail(n_valid)

#changing category IDs to str
df_train,df_valid = df_train.astype('str'),df_valid.astype('str')

#adding '.jpg' to all image names
df_train['image_id'] = df_train['image_id']+'.jpg'
df_valid['image_id'] = df_valid['image_id']+'.jpg'
for currupted_image_id in currupted_files:
    df_train = df_train[df_train.image_id != currupted_image_id]
    df_valid = df_valid[df_valid.image_id != currupted_image_id]

#show the dataframes
df_train.head()


# In[ ]:


df_valid.head()


# In[ ]:


#get all the classes
class_generator_df = pd.concat([df_valid,df_train])

classes = list(set(class_generator_df['category_id']))
num_classes = len(classes)
classes[:20]


# In[ ]:


#making the generator

train_generator = gen_t.flow_from_dataframe(df_train,
                                            x_col='image_id',
                                            y_col='category_id',
                                            target_size=image_size,
                                            class_mode='categorical',
                                            classes = classes,
                                           directory = i[:35])

valid_generator = gen_v.flow_from_dataframe(df_valid,
                                            x_col='image_id',
                                            y_col='category_id',
                                            target_size=image_size,
                                            class_mode='categorical',
                                            classes = classes,
                                           directory = i[:35])


# Found 174367 validated image filenames belonging to 212 classes.
# Found 43592 validated image filenames belonging to 45 classes.

# <h2>V0.12: preprocessing</h2>
# <p>
# Already done.</p>

# <h2>V1:Build the model</h2>

# <h3>V1.1:transfer learning</h3>

# <h4>V1.11:resnet50</h4>

# <h5>V1.111:resnet50v1</h5>

# In[ ]:


if accelerator == 'tpu':
    # instantiating the model in the strategy scope creates the model on the TPU
    with tpu_strategy.scope():
        model = tf.keras.models.Sequential()
        model.add(tf.keras.applications.ResNet50(include_top = False, 
                                              pooling = 'avg', 
                                              weights = 'imagenet'))
        model.add(tf.keras.layers.Dense(num_classes,
                                     activation='softmax'))
        model.layers[0].trainable = False
        model.compile(optimizer = 'adam',
                     loss = 'categorical_crossentropy',
                     metrics=['accuracy'])
        
else:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.applications.ResNet50(include_top = False, 
                                          pooling = 'avg', 
                                          weights = 'imagenet'))
    model.add(tf.keras.layers.Dense(num_classes,
                                 activation='softmax'))
    model.layers[0].trainable = False
    model.compile(optimizer = 'adam',
                 loss = 'categorical_crossentropy',
                 metrics=['accuracy'])


        
model.summary()


# In[ ]:


#overfit on 1 sample
ov_data,ov_label = next(train_generator)
ov_history = model.fit(ov_data,ov_label,
                       epochs=50,
                      verbose = 0)
pd.DataFrame.from_dict(ov_history.history).plot()


# In[ ]:


#callbacks
csvlogger = tf.keras.callbacks.CSVLogger('./v1_1_1_log.csv',append = True)
chk_point = tf.keras.callbacks.ModelCheckpoint('./v1_1_1_ckp.h5',save_best_only=True)


# In[ ]:



from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

history_resnet50v1 = model.fit_generator(train_generator,
                                       steps_per_epoch=stp_per_epoch,
                                       validation_data=valid_generator,
                                       validation_steps=val_stp,
                                       epochs=5,
                                       callbacks=[csvlogger,chk_point],
                                       verbose=1)


# In[ ]:




