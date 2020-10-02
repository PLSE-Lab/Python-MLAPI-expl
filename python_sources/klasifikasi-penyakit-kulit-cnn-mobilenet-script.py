# Sell 1
import keras

import pandas as pd
import numpy as np
from numpy.random import seed
seed(101)
#import keras
#from keras import backend as K

from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import os

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt


os.listdir('../input')

#Sell 2
df_data = pd.read_csv('../input/metadata/metadata.csv')

print('Metadata dataset')
df_data.head()

#Sell 3
# Class yang terdapat pada dataset
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
df_data['lesion']= df_data.dx.map(lesion_type_dict)

#Sell 4
print('Kelas yang terdapat pada dataset awal')
print(df_data['dx'].unique())
print(df_data.lesion.value_counts())

#Sell 5
import matplotlib.pyplot as plt

df_data['dx'].value_counts().plot.bar(rot=0)
plt.title('Jumlah citra untuk tipe class penyakit yang berbeda')
plt.xlabel('Class Penyakit')
plt.ylabel('Jumlah')
plt.grid(axis='y')

#Sell 6
# Memberi informasi berapa banyak citra yang dikaitkan dengan setiap lesion_id
df = df_data.groupby('lesion_id').count()

# Memfilter lesion_id yang hanya memiliki satu citra yang terkait dengannya
df = df[df['image_id'] == 1]

df.reset_index(inplace=True)

df.head()

#Sell 7
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

#Sell 8
df_data['duplicates'].value_counts()

#Sell 9
# filter citra yang tidak memiliki duplikat
df = df_data[df_data['duplicates'] == 'no_duplicates']

print('Citra yang tidak memiliki duplikat berjumlah')
df.shape

#Sell 10
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

#Sell 11
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

#Sell 12
print('Jumlah citra yang tiap class yang akan dijadikan set train sebelum augmanted')
print(df_train['dx'].value_counts())

#Sell 13
print('Jumlah citra yang tiap class yang akan dijadikan set validas')
print(df_val['dx'].value_counts())

#Sell 14
# cek berapa banyak image di set train setiap class 
print('Jumlah data citra setelah dilakukan Augmanted')
print(len(os.listdir('../input/basedir/base_dir/base_dir/train_dir/nv')))
print(len(os.listdir('../input/basedir/base_dir/base_dir/train_dir/mel')))
print(len(os.listdir('../input/basedir/base_dir/base_dir/train_dir/bkl')))
print(len(os.listdir('../input/basedir/base_dir/base_dir/train_dir/bcc')))
print(len(os.listdir('../input/basedir/base_dir/base_dir/train_dir/akiec')))
print(len(os.listdir('../input/basedir/base_dir/base_dir/train_dir/vasc')))
print(len(os.listdir('../input/basedir/base_dir/base_dir/train_dir/df')))

#Sell 15
#Mulai membuat model

train_path = '../input/basedir/base_dir/base_dir/train_dir'
valid_path = '../input/basedir/base_dir/base_dir/val_dir'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 10
val_batch_size = 10
image_size = 224

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

#Sell 16
datagen = ImageDataGenerator(
    preprocessing_function= \
    keras.applications.mobilenet.preprocess_input)

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
                                            
#Sell 17
datagen = ImageDataGenerator(
    preprocessing_function= \
    keras.applications.mobilenet.preprocess_input)

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
                            
#Sell 18
#Model MobileNet
mobile = keras.applications.mobilenet.MobileNet()

#Sell 19
mobile.summary()

#Sell 20
type(mobile.layers)

#Sell 21
# layer yang dimiliki MobileNet
print('Jumlah Layer pada pretained MobileNet')
len(mobile.layers)

#Sell 22
# MEMBUAT MODEL ARCHITECTURE

# Exclude 5 layer terakhir dari model di atas.
# Mencakup semua layer sampai layer global_average_pooling2d_1
x = mobile.layers[-6].output

# Membuat layer Dense baru untuk prediksi
# 7 corresponds sesuai dengan classs yg dimiliki
x = Dropout(0.25)(x)
predictions = Dense(7, activation='softmax')(x)

# input = mobile.input memilih layer input 
# output = prediksi mengacu pada Dense layer yang dibuat di atas.

model = Model(inputs=mobile.input, outputs=predictions)

#Sell 23
model.summary()

#Sell 24
# layer yang dimiliki model
len(model.layers)

#Sell 25
# memilih berapa banyak layer yang sebenarnya ingin di-train

# Di sini kita mem-freez weight semua lapisan kecuali 23 lapisan terakhir dalam model baru
# 23 lapisan terakhir dari model akan dilatih.

for layer in model.layers[:-23]:
    layer.trainable = False
    
#Sell 26
#TRAIN MODEL
# Define Top2 and Top3 Accuracy

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)
    
#Sell 27
model.compile(Adam(lr=0.01), loss='categorical_crossentropy', 
              metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])
              
#Sell 28
# Mendapatkan labels yang terkait dengan setiap indeks
print(valid_batches.class_indices)

#Sell 29
# Tambahkan bobot untuk mencoba membuat model lebih sensitif terhadap melanoma

class_weights={
    0: 1.0, # akiec
    1: 1.0, # bcc
    2: 1.0, # bkl
    3: 1.0, # df
    4: 3.0, # mel # Try to make the model more sensitive to Melanoma.
    5: 1.0, # nv
    6: 1.0, # vasc
}

#Sell 30
filepath = "model04-v5.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=5, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

history = model.fit_generator(train_batches, steps_per_epoch=train_steps, 
                              class_weight=class_weights,
                    validation_data=valid_batches,
                    validation_steps=val_steps,
                    epochs=30, verbose=1,
                   callbacks=callbacks_list)
                   
#Sell 31
# get the metric names so we can use evaulate_generator
model.metrics_names

#Sell 32
# Here the the last epoch will be used.

val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = \
model.evaluate_generator(test_batches, 
                        steps=len(df_val))

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
print('val_top_2_acc:', val_top_2_acc)
print('val_top_3_acc:', val_top_3_acc)

#Sell 33
# Here the best epoch will be used.

model.load_weights('model04-v5.h5')

val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = \
model.evaluate_generator(test_batches, 
                        steps=len(df_val))

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
print('val_top_2_acc:', val_top_2_acc)
print('val_top_3_acc:', val_top_3_acc)

#Sell 34
import matplotlib.pyplot as plt
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
train_top2_acc = history.history['top_2_accuracy']
val_top2_acc = history.history['val_top_2_accuracy']
train_top3_acc = history.history['top_3_accuracy']
val_top3_acc = history.history['val_top_3_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, '', label='Training loss')
plt.plot(epochs, val_loss, '', label='Validation loss')
plt.title('MobileNet -- Training dan Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, '', label='Training cat accuracy')
plt.plot(epochs, val_acc, '', label='Validation cat accuracy')
plt.title('MobileNet -- Training dan Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, train_top2_acc, '', label='Training top2 acc')
plt.plot(epochs, val_top2_acc, '', label='Validation top2 acc')
plt.title('TOP 2 - Training dan Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, train_top3_acc, '', label='Training top3 acc')
plt.plot(epochs, val_top3_acc, '', label='Validation top3 acc')
plt.title('TOP 3 - Training dan Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
