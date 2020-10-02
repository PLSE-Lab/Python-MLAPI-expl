#!/usr/bin/env python
# coding: utf-8

# # FROM REFERENCEE
# 
# **Catatan penting pada versi ini:**
# 1. Preprocess diganti dari *keras.applications.mobilenet.preprocess_input* menjadi *keras.applications.inception_v3.preprocess_input*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import keras
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd 

df_data = pd.read_csv('../input/metadata/metadata.csv')
df_data.head()


# In[ ]:


df_data['dx'].unique()


# In[ ]:


import matplotlib.pyplot as plt
#exp = pd.Series.to_frame(df1.groupby('dx').sex.value_counts())
df_data['dx'].value_counts().plot.bar(rot=0)
plt.title('Number of images for different dx type')
plt.xlabel('dx')
plt.ylabel('Counts')
plt.grid(axis='y')


# # 1. Create several more columns for the dataframe 'df'

# 1. Create 'num_images' to record the number of images belonging to the same 'lesion_id'
# 2. Create 'dx_id' convert the 'dx' to integer label
# 3. Create 'image_path' to store the path to access the image
# 4. Create 'images' to store the resized image as arrays

# In[ ]:


# Memberi informasi berapa banyak citra yang dikaitkan dengan setiap lesion_id
df = df_data.groupby('lesion_id').count()

# Memfilter lesion_id yang hanya memiliki satu citra yang terkait dengannya
df = df[df['image_id'] == 1]

df.reset_index(inplace=True)

df_data.head()


# In[ ]:


# identifikasi lesion_id yg mempunyai duplikat citra atau tidak.

def identify_duplicates(x):
    
    unique_list = list(df['lesion_id'])
    
    if x in unique_list:
        return 'no_duplicates'
    else:
        return 'has_duplicates'
    
# buat kolom baru yang merupakan salinan dari kolom lesi _id
df_data['duplicates'] = df_data['lesion_id']
# terapkan fungsi ke kolom baru ini
df_data['duplicates'] = df_data['duplicates'].apply(identify_duplicates)

df_data.head(50)


# In[ ]:


df_data['duplicates'].value_counts()


# In[ ]:


# filter citra yang tidak memiliki duplikat
df = df_data[df_data['duplicates'] == 'no_duplicates']

print('Citra yang tidak memiliki duplikat berjumlah')
df.shape


# In[ ]:


# df yang telah dipastikan tidak memiliki duplikat displit kemudian dijadikan set val (validasi)
y = df['dx']

import tensorflow
from sklearn.model_selection import train_test_split
_, df_val = train_test_split(df, test_size=0.17, random_state=101, stratify=y)

#train_size -> If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split
#randostate -> If int, random_state is the seed used by the random number generator
#stratif -> If not None, data is split in a stratified fashion

print('Jumlah citra sebagai validasi')
df_val.shape


# In[ ]:


#Membuat set train yg tidak termasuk images yg ada di set val

#Fungsi ini mengidentifikasi apakah gambar adalah bagian dari set train atau set val
def identify_val_rows(x):
    # create a list of all the lesion_id's in the val set
    val_list = list(df_val['image_id'])
    
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'
# buat kolom baru yang merupakan salinan dari kolom image_id
df_data['train_or_val'] = df_data['image_id']
# terapkan fungsi ke kolom baru ini
df_data['train_or_val'] = df_data['train_or_val'].apply(identify_val_rows)
# filter baris set train
df_train = df_data[df_data['train_or_val'] == 'train']

print('Jumlah citra yang akan dijadikan set train:')
print(len(df_train))
print('Jumlah citra yang akan dijadikan set validasi:')
print(len(df_val))


# In[ ]:


print('Jumlah citra yang tiap class yang akan dijadikan set train sebelum augmanted')
print(df_train['dx'].value_counts())


# In[ ]:


print('Jumlah citra yang tiap class yang akan dijadikan set validas')
print(df_val['dx'].value_counts())


# In[ ]:


# cek berapa banyak image di set train setiap class 
print('Jumlah data citra setelah dilakukan Augmanted')
print(len(os.listdir('../input/basedir/base_dir/base_dir/train_dir/nv')))
print(len(os.listdir('../input/basedir/base_dir/base_dir/train_dir/mel')))
print(len(os.listdir('../input/basedir/base_dir/base_dir/train_dir/bkl')))
print(len(os.listdir('../input/basedir/base_dir/base_dir/train_dir/bcc')))
print(len(os.listdir('../input/basedir/base_dir/base_dir/train_dir/akiec')))
print(len(os.listdir('../input/basedir/base_dir/base_dir/train_dir/vasc')))
print(len(os.listdir('../input/basedir/base_dir/base_dir/train_dir/df')))


# In[ ]:


train_path = '../input/basedir/base_dir/base_dir/train_dir'
valid_path = '../input/basedir/base_dir/base_dir/val_dir'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 10
val_batch_size = 10
image_size = 224

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)


# In[ ]:


print(train_steps)
print(val_steps)


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    preprocessing_function= \
    keras.applications.inception_v3.preprocess_input)

train_batches = datagen.flow_from_directory(train_path,
                                            target_size=(image_size,image_size),
                                            batch_size=train_batch_size)

valid_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size,image_size),
                                            batch_size=val_batch_size)

# Note: shuffle=False causes the test dataset to not be shuffled
test_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size,image_size),
                                            batch_size=1,
                                            shuffle=False)


# # 5. Build CNN model -- InceptionV3

# In[ ]:


import keras
from keras.applications import InceptionV3
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

input_shape = (224, 224, 3)

num_labels = 7

base_model = InceptionV3(include_top=False, input_shape=(224, 224, 3),pooling = 'avg', weights = '../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
model = Sequential()
model.add(base_model)
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu",kernel_regularizer=regularizers.l2(0.02)))
model.add(Dropout(0.5))
model.add(Dense(7, activation = 'softmax',kernel_regularizer=regularizers.l2(0.02)))

for layer in base_model.layers:
    layer.trainable = True

#for layer in base_model.layers[-30:]:
 #   layer.trainable = True

#model.add(ResNet50(include_top = False, pooling = 'max', weights = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'))

model.summary()


# In[ ]:


print(valid_batches.class_indices)


# In[ ]:


class_weights={
    0: 1.0, # akiec
    1: 1.0, # bcc
    2: 1.0, # bkl
    3: 1.0, # df
    4: 3.0, # mel # Try to make the model more sensitive to Melanoma.
    5: 1.0, # nv
    6: 1.0, # vasc
}


# In[ ]:


from keras.optimizers import Adam
optimizer = Adam (lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=5e-7, amsgrad=False)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


# Fit the model

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
class CustomModelCheckPoint(keras.callbacks.Callback):
    def __init__(self,**kargs):
        super(CustomModelCheckPoint,self).__init__(**kargs)
        self.epoch_accuracy = {} # loss at given epoch
        self.epoch_loss = {} # accuracy at given epoch
        def on_epoch_begin(self,epoch, logs={}):
            # Things done on beginning of epoch. 
            return

        def on_epoch_end(self, epoch, logs={}):
            # things done on end of the epoch
            self.epoch_accuracy[epoch] = logs.get("acc")
            self.epoch_loss[epoch] = logs.get("loss")
            self.model.save_weights("name-of-model-%d.h5" %epoch)
            
checkpoint = CustomModelCheckPoint()
cb_early_stopper = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
cb_checkpointer = ModelCheckpoint(filepath = '../working/best.h5', monitor = 'val_loss', save_best_only = True, mode = 'auto')
callbacks_list = [cb_checkpointer, cb_early_stopper]


# In[ ]:


epochs = 30
trainhistory = model.fit_generator(train_batches, validation_steps=val_steps,class_weight=class_weights,
                              epochs = epochs, validation_data = valid_batches,
                              verbose = 1, steps_per_epoch=train_steps,
                                       callbacks=callbacks_list)


# # 6. Plot the accuracy and loss of both training and validation dataset

# In[ ]:


import matplotlib.pyplot as plt
acc = trainhistory.history['acc']
val_acc = trainhistory.history['val_acc']
loss = trainhistory.history['loss']
val_loss = trainhistory.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, '', label='Training loss')
plt.plot(epochs, val_loss, '', label='Validation loss')
plt.title('InceptionV3 -- Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, '', label='Training accuracy')
plt.plot(epochs, val_acc, '', label='Validation accuracy')
plt.title('InceptionV3 -- Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


# In[ ]:




model.load_weights("../working/best.h5")
test_loss, test_acc =model.evaluate_generator(test_batches, steps=len(df_val), verbose=1)
print("test_accuracy = %f  ;  test_loss = %f" % (test_acc, test_loss))


# In[ ]:


# Source: Scikit Learn website
# http://scikit-learn.org/stable/auto_examples/
# model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-
# selection-plot-confusion-matrix-py

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[ ]:


test_labels = test_batches.classes


# In[ ]:


test_labels


# In[ ]:


test_labels.shape


# In[ ]:


predictions = model.predict_generator(test_batches, steps=len(df_val), verbose=1)


# In[ ]:


# argmax returns the index of the max value in a row
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))


# In[ ]:


# Define the labels of the class indices. These need to match the 
# order shown above.
cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


# In[ ]:


# Get the index of the class with the highest probability score
y_pred = np.argmax(predictions, axis=1)

# Get the labels of the test images.
y_true = test_batches.classes


# In[ ]:


from sklearn.metrics import classification_report

# Generate a classification report
report = classification_report(y_true, y_pred, target_names=cm_plot_labels)

print(report)

