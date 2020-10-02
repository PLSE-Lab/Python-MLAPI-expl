#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import defaultdict
from glob import glob
from random import choice, sample
import cv2
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D, Flatten
from keras.models import Model
from keras.layers import BatchNormalization
from keras.preprocessing import image
from keras.optimizers import Adam
import h5py
from keras.layers import LeakyReLU
from keras import regularizers
import gc
import psutil


# In[ ]:


train_file_path = "../input/train_relationships.csv"
train_folders_path = "../input/train/"
val_famillies = "F09"


# In[ ]:


all_images = glob(train_folders_path + "*/*/*.jpg")
train_images = [x for x in all_images if val_famillies not in x]
val_images = [x for x in all_images if val_famillies in x]


# In[ ]:


train_person_to_images_map = defaultdict(list)
ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

for x in train_images:
    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

val_person_to_images_map = defaultdict(list)

for x in val_images:
    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)


# In[ ]:


relationships = pd.read_csv(train_file_path)
relationships = list(zip(relationships.p1.values, relationships.p2.values))
relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]


# In[ ]:


train = [x for x in relationships if val_famillies not in x[0]]
val = [x for x in relationships if val_famillies in x[0]]


# In[ ]:


def read_img(path):
    img = image.load_img(path, target_size=(197, 197))
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)


# In[ ]:


def gen(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 2)
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(ppl)
            p2 = choice(ppl)

            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1 = np.array([read_img(x) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_img(x) for x in X2])

        yield [X1, X2], labels


# In[ ]:


def baseline_model():
    input_1 = Input(shape=(197, 197, 3))
    input_2 = Input(shape=(197, 197, 3))

    base_model = VGGFace(model='resnet50', include_top=False)

    for x in base_model.layers[:-3]:
        x.trainable = True

    x1 = base_model(input_1)
    x2 = base_model(input_2)

    x1 = Concatenate(axis=-1)([GlobalAvgPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalAvgPool2D()(x2), GlobalAvgPool2D()(x2)])

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    
    x1_ = Multiply()([x1, x1])
    x2_ = Multiply()([x2, x2])
    x4  = Subtract()([x1_, x2_])    
    x   = Concatenate(axis=-1)([x4, x3])
    x   = Dense(256, activation="relu",kernel_regularizer=regularizers.l2(0.01))(x)
    x   = BatchNormalization()(x)
    x   = Dense(256, activation="relu",kernel_regularizer=regularizers.l2(0.01))(x)
    x   = Dropout(0.01)(x) 
    x   = Dense(128, activation="relu",kernel_regularizer=regularizers.l2(0.01))(x)
    x   = BatchNormalization()(x)
    x   = Dense(128, activation="relu",kernel_regularizer=regularizers.l2(0.01))(x)
    x   = Dropout(0.01)(x)    
    x   = Dense(100, activation="relu",kernel_regularizer=regularizers.l2(0.01))(x)    
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)
    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))
    model.summary()

    return model


# In[ ]:


get_ipython().system('pip install git+https://github.com/rcmalli/keras-vggface.git')
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace


# In[ ]:


print("available RAM:", psutil.virtual_memory())
gc.collect()
print("available RAM:", psutil.virtual_memory())


# In[ ]:



file_path = "vgg_face.h5"

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, 
                          patience=15, verbose=0, mode='auto')
reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.2, 
                                      patience=10, verbose=1)

callbacks_list = [checkpoint,early_stop]

curr_model = baseline_model()
curr_model_hist=curr_model.fit_generator(gen(train, train_person_to_images_map, batch_size=16), 
                            use_multiprocessing=True,
                    validation_data=gen(val, val_person_to_images_map, batch_size=16), 
                            epochs=50, 
                            verbose=1,workers = 4, 
                            callbacks=callbacks_list,
                            steps_per_epoch=200,
                            validation_steps=100)


# In[ ]:


import matplotlib.pyplot as plt
def plot_accuracy(y):
    if(y == True):
        plt.plot(curr_model_hist.history['acc'])
        plt.plot(curr_model_hist.history['val_acc'])
        plt.legend(['train', 'test'], loc='lower right')
        plt.title('accuracy plot - train vs test')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.show()
    else:
        pass
    return

def plot_loss(y):
    if(y == True):
        plt.plot(curr_model_hist.history['loss'])
        plt.plot(curr_model_hist.history['val_loss'])
        plt.legend(['training loss', 'validation loss'], loc = 'upper right')
        plt.title('loss plot - training vs vaidation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
    else:
        pass
    return


plot_accuracy(True)
plot_loss(True)


# In[ ]:


print("available RAM:\n", psutil.virtual_memory())
gc.collect()
print("available RAM:\n", psutil.virtual_memory())


# In[ ]:


test_path = "../input/test/"


# In[ ]:


def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


from tqdm import tqdm

submission = pd.read_csv('../input/sample_submission.csv')

predictions = []

for batch in tqdm(chunker(submission.img_pair.values)):
    X1 = [x.split("-")[0] for x in batch]
    X1 = np.array([read_img(test_path + x) for x in X1])

    X2 = [x.split("-")[1] for x in batch]
    X2 = np.array([read_img(test_path + x) for x in X2])

    pred = curr_model.predict([X1, X2]).ravel().tolist()
    predictions += pred

submission['is_related'] = predictions

submission.to_csv("vgg_face.csv", index=False)

