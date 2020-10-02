#!/usr/bin/env python
# coding: utf-8

# # Goal
# The goal is to build upon the shoulders of giants, in this case, Kevin Mader. I am attempting to see if I can use his model and make some adjustments to improve the score of the model.
# 
# ## What I did
# * sampling more images
# * increasing image size from 128 x 128 to 256 x 256
# * Removing the flipping horizontal in the image generation (theory is that these are not truly symmetrical and false information may be generated, however I understand that issues could occur in either lung)
# * replaved MobileNet with using the InceptionResNetV2 - the deepest of the layers available in the Keras pre-trained set - which changed the number of parameters from 3 Million to 55 Million and take much longer
# * Turned on the GPU to increase the spead of the training model. The GPU greatly increased the model training speed but seemed to make the non model training portions run slightly slower.
# * Change the goal to simply improve accuracy. Only tracking one number rather than two helped me focus my effort
# 
# ## Things that didn't work well
# * Making a more complex network by adding dense and dropout layers
# * Adjusting dropout amounts to (0.8) and (0.2)
# * Changing loss function from binary_crossentropy to category_crossentropy - loss just hovered around 1.7
# * Changing to relu activtation
# 
# ## Summary and Feedback
# 
# 
# Please review below and let me know your thoughts on how to improve this model or the code.
# 
# 

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[3]:


all_xray_df = pd.read_csv('../input/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('..', 'input', 'images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
#all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x: int(x[:-1]))
all_xray_df.sample(3)


# # Preprocessing Labels
# Here we take the labels and make them into a more clear format. The primary step is to see the distribution of findings and then to convert them to simple binary labels

# In[4]:


label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation = 90)


# In[5]:


all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
from itertools import chain
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
all_xray_df.sample(3)


# ### Clean categories
# Since we have too many categories, we can prune a few out by taking the ones with only a few examples

# In[6]:


# keep at least 1000 cases
MIN_CASES = 1000
all_labels = [c_label for c_label in all_labels if all_xray_df[c_label].sum()>MIN_CASES]
print('Clean Labels ({})'.format(len(all_labels)), 
      [(c_label,int(all_xray_df[c_label].sum())) for c_label in all_labels])


# In[ ]:


# Not sampling and using the full dataset

# since the dataset is very unbiased, we can resample it to be a more reasonable collection
# weight is 0.1 + number of findings
#sample_weights = all_xray_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2
#sample_weights /= sample_weights.sum()
#all_xray_df = all_xray_df.sample(40000, weights=sample_weights)

#label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
#fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
#ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
#ax1.set_xticks(np.arange(len(label_counts))+0.5)
#_ = ax1.set_xticklabels(label_counts.index, rotation = 90)


# In[7]:


label_counts = 100*np.mean(all_xray_df[all_labels].values,0)
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
ax1.set_xticklabels(all_labels, rotation = 90)
ax1.set_title('Adjusted Frequency of Diseases in Patient Group')
_ = ax1.set_ylabel('Frequency (%)')


# # Prepare Training Data
# Here we split the data into training and validation sets and create a single vector (disease_vec) with the 0/1 outputs for the disease status (what the model will try and predict)

# In[8]:


all_xray_df['disease_vec'] = all_xray_df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])


# In[9]:


from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(all_xray_df, 
                                   test_size = 0.25, 
                                   random_state = 2018,
                                   stratify = all_xray_df['Finding Labels'].map(lambda x: x[:4]))
print('train', train_df.shape[0], 'validation', valid_df.shape[0])


# # Create Data Generators
# Here we make the data generators for loading and randomly transforming images

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (128, 128)
core_idg = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = False, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)


# In[12]:


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


# In[ ]:


train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = 32)

valid_gen = flow_from_dataframe(core_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = 256) # we can use much larger batches for evaluation
# used a fixed dataset for evaluating the algorithm
test_X, test_Y = next(flow_from_dataframe(core_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = 1024)) # one big batch


# In[ ]:


t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -1.5, vmax = 1.5)
    c_ax.set_title(', '.join([n_class for n_class, n_score in zip(all_labels, c_y) 
                             if n_score>0.5]))
    c_ax.axis('off')


# In[ ]:


from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPool2D, Dense, Dropout, Flatten, Conv2D
from keras.models import Sequential
base_mobilenet_model = MobileNet(input_shape =  t_x.shape[1:], 
                                 include_top = False, weights = None)
multi_disease_model = Sequential()
multi_disease_model.add(base_mobilenet_model)
multi_disease_model.add(GlobalAveragePooling2D())
multi_disease_model.add(Dropout(0.5))
#multi_disease_model.add(Conv2D(1024, kernel_size=(2,2), activation='sigmoid',input_shape =  t_x.shape[1:]))
multi_disease_model.add(Dense(512))
multi_disease_model.add(Dropout(0.5))
#multi_disease_model.add(GlobalAveragePooling2D())
#multi_disease_model.add(Dropout(0.6))

#multi_disease_model.add(Dense(1024))
multi_disease_model.add(Dense(len(all_labels), activation = 'sigmoid'))
multi_disease_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                           metrics = ['categorical_accuracy'])
multi_disease_model.summary()


# In[14]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('xray_class')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=3)
callbacks_list = [checkpoint, early]


# # First Round
# Here we do a first round of training to get a few initial low hanging fruit results

# In[ ]:


history = multi_disease_model.fit_generator(train_gen, 
                                  steps_per_epoch=100,
                                  validation_data = (test_X, test_Y), 
                                  epochs = 1, 
                                  callbacks = callbacks_list)


# # Check Output
# Here we see how many positive examples we have of each category

# In[ ]:


score = multi_disease_model.evaluate(test_X, test_Y, verbose=0) 


# In[ ]:


print("Accuracy after one epoch: " + str(round(score[1]*100,2)) + "%")


# In[ ]:


pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = True)


# # ROC Curves
# While a very oversimplified metric, we can show the ROC curve for each metric

# In[ ]:


from sklearn.metrics import roc_curve, auc
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('barely_trained_net.png')


# # Continued Training
# Now we do a much longer training process to see how the results improve

# In[ ]:


steps_per_epoch = 100,
validation_data =  (test_X, test_Y), 
epochs = 5, 
callbacks = callbacks_list)


# In[ ]:


print(history_multiple_epochs.history.keys())
#str(history.history['loss',])
#plt.subplot(211)  
print(history_multiple_epochs.history)
print(callbacks_list)
plt.plot(history_multiple_epochs.history['val_loss'])  
plt.plot(history_multiple_epochs.history['loss'])  
plt.plot(history_multiple_epochs.history['val_binary_accuracy'])  
plt.plot(history_multiple_epochs.history['binary_accuracy'])  
plt.plot(history_multiple_epochs.history['val_mean_absolute_error'])  
plt.plot(history_multiple_epochs.history['mean_absolute_error'])  
#plt.title('model accuracy')  
#plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left') 


# In[ ]:


# load the best weights
multi_disease_model.load_weights(weight_path)


# In[ ]:


pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = True)


# In[ ]:


# look at how often the algorithm predicts certain diagnoses 
for c_label, p_count, t_count in zip(all_labels, 
                                     100*np.mean(pred_Y,0), 
                                     100*np.mean(test_Y,0)):
    print('%s: Dx: %2.2f%%, PDx: %2.2f%%' % (c_label, t_count, p_count))


# In[ ]:


from sklearn.metrics import roc_curve, auc
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('trained_net.png')


# # Improvement
# 
# With a larger set of data and only five epochs, AOC is improving.
# 
# What if we change the base model from MobileNet to InceptionResNetV2?

# In[ ]:


from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPool2D, Dense, Dropout, Flatten, Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (256, 256)
core_idg = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = False, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)

train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = 32)

valid_gen = flow_from_dataframe(core_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = 256) # we can use much larger batches for evaluation
# used a fixed dataset for evaluating the algorithm
test_X, test_Y = next(flow_from_dataframe(core_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = 1024)) # one big batch

t_x, t_y = next(train_gen)

#base_inception_model = MobileNet(input_shape =  t_x.shape[1:], include_top = False, weights = None)
base_inception_model = InceptionResNetV2(input_shape =  t_x.shape[1:], include_top = False, weights = None)
multi_disease_inception_model = Sequential()
multi_disease_inception_model.add(base_inception_model)
multi_disease_inception_model.add(GlobalAveragePooling2D())
multi_disease_inception_model.add(Dropout(0.2))
#multi_disease_model.add(Conv2D(1024, kernel_size=(2,2), activation='sigmoid',input_shape =  t_x.shape[1:]))
multi_disease_inception_model.add(Dense(512))
multi_disease_inception_model.add(Dropout(0.2))
#multi_disease_model.add(GlobalAveragePooling2D())
#multi_disease_model.add(Dropout(0.6))

#multi_disease_model.add(Dense(1024))
multi_disease_inception_model.add(Dense(len(all_labels), activation = 'sigmoid'))
multi_disease_inception_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['accuracy'])
multi_disease_inception_model.summary()

#from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('xray_class')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=3)
callbacks_list = [checkpoint, early]


# In[ ]:


inception_history = multi_disease_inception_model.fit_generator(train_gen, 
                                  steps_per_epoch=100,
                                  validation_data = (test_X, test_Y), 
                                  epochs = 5, 
                                  callbacks = callbacks_list)


# In[23]:


print(inception_history.history.keys())
#str(history.history['loss',])
plt.subplot(211)  
#print(inception_history.history)
#print(callbacks_list)
plt.plot(inception_history.history['val_loss'])  
plt.plot(inception_history.history['loss'])  
#plt.plot(inception_history.history['binary_accuracy'])  
#plt.title('model accuracy')  
#plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left') 
plt.show()
#plt.subplot(221)
plt.plot(inception_history.history['val_acc'])  
plt.plot(inception_history.history['acc'])  
#plt.plot(inception_history.history['binary_accuracy'])  
#plt.title('model accuracy')  
#plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left') 


# In[24]:


inception_score = multi_disease_inception_model.evaluate(test_X, test_Y, verbose=0) 
print("Accuracy for inception model after epochs: " + str(round(inception_score[1]*100,2)) + "%")


# In[ ]:




