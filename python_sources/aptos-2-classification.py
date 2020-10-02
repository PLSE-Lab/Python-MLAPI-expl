#!/usr/bin/env python
# coding: utf-8

# # APTOS - Classification #

# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import cv2
import gc

from keras import backend as K
from keras import losses
from keras.applications.resnet50 import ResNet50, preprocess_input
#from keras.applications.densenet import DenseNet121, preprocess_input
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, Dropout, Conv2D, BatchNormalization, Input, Flatten, LeakyReLU
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.preprocessing import image

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split


# ## Constants ##

# In[ ]:


SEED = 575
IMAGE_SIZE = 256
NUM_CLASSES = 5
BATCH_SIZE = 16
INITIAL_EPOCHS = 5
MIDDLE_EPOCHS = 10
FINAL_EPOCHS = 30
INITIAL_LR = 5e-3
MIDDLE_LR = 5e-4
FINAL_LR = 1e-5
NUM_SAMPLES_2015 = 1100
NUM_SAMPLES_2019 = 1600
TEST_SIZE = 0.3
QUEUE_SIZE = 300
WORKERS = 3

train_directory_2015 = '../input/aptos-converted-2015/train_images/'
train_labels_2015 = '../input/aptos-converted-2015/train.csv'

train_directory_2019 = '../input/aptos-converted/training_aptos/'
train_labels_2019 = '../input/aptos-converted/train.csv'

test_image_directory = '../input/aptos2019-blindness-detection/test_images/'
test_data_file = '../input/aptos2019-blindness-detection/test.csv'

weights_file = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
#weights_file = '../input/densenet-keras/DenseNet-BC-121-32-no-top.h5'
best_weight_file = 'aptos_best_weights.h5'
submission_file = 'submission.csv'


# In[ ]:


#os.listdir('../input/aptos-converted/training_aptos')


# ## Load data files ##

# In[ ]:


df_train_2015 = pd.read_csv(train_labels_2015).sample(frac=1)
df_train_2019 = pd.read_csv(train_labels_2019).sample(frac=1)

df_train_2015['image'] = df_train_2015['image'].apply(lambda i : "{}.jpeg".format(i))
df_train_2019['id_code'] = df_train_2019['id_code'].apply(lambda i : "{}.png".format(i))

df_train_2015.columns = ['image', 'diagnosis']
df_train_2019.columns = ['image', 'diagnosis']

df_train_2015['diagnosis'] = df_train_2015['diagnosis'].astype('str')
df_train_2019['diagnosis'] = df_train_2019['diagnosis'].astype('str')


# ## Class balancing ##

# In[ ]:


df_train_2015_0 = df_train_2015[df_train_2015['diagnosis'] == '0'].sample(NUM_SAMPLES_2015, replace=False, random_state=SEED)
df_train_2015_1 = df_train_2015[df_train_2015['diagnosis'] == '1'].sample(NUM_SAMPLES_2015, replace=False, random_state=SEED)
df_train_2015_2 = df_train_2015[df_train_2015['diagnosis'] == '2'].sample(NUM_SAMPLES_2015, replace=False, random_state=SEED)
df_train_2015_3 = df_train_2015[df_train_2015['diagnosis'] == '3'].sample(NUM_SAMPLES_2015, replace=True, random_state=SEED)
df_train_2015_4 = df_train_2015[df_train_2015['diagnosis'] == '4'].sample(NUM_SAMPLES_2015, replace=True, random_state=SEED)
df_train_2015 = pd.concat([df_train_2015_0, df_train_2015_1, df_train_2015_2, df_train_2015_3, df_train_2015_4])
df_train_2015 = df_train_2015.sample(frac=1)


# In[ ]:


df_train_2019_0 = df_train_2019[df_train_2019['diagnosis'] == '0'].sample(NUM_SAMPLES_2019, replace=False, random_state=SEED)
df_train_2019_1 = df_train_2019[df_train_2019['diagnosis'] == '1'].sample(NUM_SAMPLES_2019, replace=True, random_state=SEED)
df_train_2019_2 = df_train_2019[df_train_2019['diagnosis'] == '2'].sample(NUM_SAMPLES_2019, replace=True, random_state=SEED)
df_train_2019_3 = df_train_2019[df_train_2019['diagnosis'] == '3'].sample(NUM_SAMPLES_2019, replace=True, random_state=SEED)
df_train_2019_4 = df_train_2019[df_train_2019['diagnosis'] == '4'].sample(NUM_SAMPLES_2019, replace=True, random_state=SEED)
df_train_2019 = pd.concat([df_train_2019_0, df_train_2019_1, df_train_2019_2, df_train_2019_3, df_train_2019_4])
df_train_2019 = df_train_2019.sample(frac=1)


# ## LR Scheduler ##

# In[ ]:


def lr_scheduler(epoch):
    step = (FINAL_LR - 1e-6) / FINAL_EPOCHS
    lr = FINAL_LR - (epoch * step)
    print("Reducing learning rate to {}".format(lr))
    return lr


# ## Kappa callback ##

# In[ ]:


class Kappa(Callback):
    def __init__(self, val_data, val_stop_patience=5):
        super(Callback, self).__init__()
        self.validation_data = val_data
        self.val_kappas = []
        self.val_stop_patience = val_stop_patience

    def on_epoch_end(self, epoch, logs={}):
        print('Calculating Kappa score...')
        y_pred = [] #np.empty(self.validation_data.n, dtype=np.uint8)
        y_val = [] #np.empty(self.validation_data.n, dtype=np.uint8)
        w_size = 0
        for idx in range(self.validation_data.n // self.validation_data.batch_size):
            x, y = self.validation_data.next()
            y_res = np.argmax(y, axis=1)
            y_pres = np.argmax(model.predict(x), axis=1)
            for res in y_pres:
                y_pred.append(res)
            for res in y_res:
                y_val.append(res)

        print("y_val(50)=", y_val[:50])
        print("y_pred(50)=", y_pred[:50])
        
        _val_kappa = cohen_kappa_score(
            np.array(y_val),
            np.array(y_pred), 
            weights='quadratic')

        self.val_kappas.append(_val_kappa)
        print("val_kappa:", np.round(_val_kappa, 3))

        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save_weights(best_weight_file)
            self.val_stop_p_count = 0
        else:
            # stop the training
            self.val_stop_p_count = self.val_stop_p_count + 1;
            if(self.val_stop_p_count > self.val_stop_patience):
                print("Epoch %05d: early stopping " % epoch)
                self.model.stop_training = True


# ## Ordinal loss function ##

# In[ ]:


def ordinal_loss(y_true, y_pred):
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)


# ## Image processing function ##

# In[ ]:


def process_image(img):
    img = preprocess_input(img)
    return img


# ## Model definition ##

# In[ ]:


def design_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(2, 2), input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=1000, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=1000, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


# In[ ]:


def build_model():
    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)
    #base_model = DenseNet121(include_top=False, weights=None, input_tensor=input_tensor)
    base_model.load_weights(weights_file)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    output_tensor = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model


# ## Run the training ##

# In[ ]:


model = design_model()


# #### Initial training on top classifier only, 2015 data ####

# In[ ]:


train_generator = ImageDataGenerator(preprocessing_function=process_image, horizontal_flip=True, vertical_flip=True, zoom_range=0.1, rotation_range=10)
#valid_generator = ImageDataGenerator(preprocessing_function=process_image)
train_flow = train_generator.flow_from_dataframe(directory=train_directory_2015, dataframe=df_train_2015, x_col="image", y_col="diagnosis", class_mode="categorical", drop_duplicates=False, batch_size=BATCH_SIZE, seed=SEED)
#valid_flow = valid_generator.flow_from_dataframe(directory=train_directory_2019, dataframe=df_train_2019, x_col="image", y_col="diagnosis", class_mode="categorical", drop_duplicates=True, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, seed=SEED)

#for layer in model.layers[:-3]:
#    layer.trainable = False
#for layer in model.layers[-3:]:
#    layer.trainable = True

model.compile(optimizer=Adam(INITIAL_LR), loss=ordinal_loss, metrics=['accuracy'])

model.fit_generator(
    generator=train_flow,
    epochs=INITIAL_EPOCHS,
    steps_per_epoch = train_flow.n // train_flow.batch_size,
    verbose=1,
#    validation_data=valid_flow,
#    validation_steps = valid_flow.n // valid_flow.batch_size,
    max_queue_size = QUEUE_SIZE,
    workers = WORKERS
    )


# #### Training on full model, 2015 data ####

# In[ ]:


train_generator = ImageDataGenerator(preprocessing_function=process_image, horizontal_flip=True, vertical_flip=True, zoom_range=0.1, rotation_range=10)
#valid_generator = ImageDataGenerator(preprocessing_function=process_image)
train_flow = train_generator.flow_from_dataframe(directory=train_directory_2015, dataframe=df_train_2015, x_col="image", y_col="diagnosis", class_mode="categorical", drop_duplicates=False, batch_size=BATCH_SIZE, seed=SEED)
#valid_flow = valid_generator.flow_from_dataframe(directory=train_directory_2019, dataframe=df_train_2019, x_col="image", y_col="diagnosis", class_mode="categorical", drop_duplicates=True, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, seed=SEED)

#lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6, verbose=1)
#callbacks = []

for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(MIDDLE_LR), loss=ordinal_loss, metrics=['accuracy'])

model.fit_generator(
    generator=train_flow,
    epochs=MIDDLE_EPOCHS,
    steps_per_epoch = train_flow.n // train_flow.batch_size,
    verbose=1,
#    validation_data=valid_flow,
#    validation_steps = valid_flow.n // valid_flow.batch_size,
    max_queue_size = QUEUE_SIZE,
    workers = WORKERS
#    callbacks = callbacks
    )


# #### Training on full model, 2019 data ####

# In[ ]:


train_generator = ImageDataGenerator(preprocessing_function=process_image, horizontal_flip=True, vertical_flip=True, zoom_range=0.1, rotation_range=10, validation_split=TEST_SIZE)
train_flow = train_generator.flow_from_dataframe(directory=train_directory_2019, dataframe=df_train_2019, x_col="image", y_col="diagnosis", class_mode="categorical", drop_duplicates=False, batch_size=BATCH_SIZE, seed=SEED, subset="training")
valid_flow = train_generator.flow_from_dataframe(directory=train_directory_2019, dataframe=df_train_2019, x_col="image", y_col="diagnosis", class_mode="categorical", drop_duplicates=True, batch_size=BATCH_SIZE, seed=SEED, subset="validation")

lr = LearningRateScheduler(lr_scheduler)
callbacks = [lr]

model.fit_generator(
    generator=train_flow,
    epochs=FINAL_EPOCHS,
    steps_per_epoch = train_flow.n // train_flow.batch_size,
    verbose=1,
#    validation_data=valid_flow,
#    validation_steps = valid_flow.n // valid_flow.batch_size,
    max_queue_size = QUEUE_SIZE,
    workers = WORKERS,
    callbacks = callbacks
    )


# #### Final training on 2019 data with kappa score ####

# In[ ]:


kappa = Kappa(valid_flow)
callbacks = [kappa]

model.fit_generator(
    generator=train_flow,
    epochs=FINAL_EPOCHS,
    steps_per_epoch = train_flow.n // train_flow.batch_size,
    verbose=1,
#    validation_data=valid_flow,
#    validation_steps = valid_flow.n // valid_flow.batch_size,
    max_queue_size = QUEUE_SIZE,
    workers = WORKERS,
    callbacks = callbacks
    )


# In[ ]:


kappa = Kappa(valid_flow)
callbacks = [kappa]

model.fit_generator(
    generator=train_flow,
    epochs=FINAL_EPOCHS * 40,
    initial_epoch = FINAL_EPOCHS + 1,
    steps_per_epoch = train_flow.n // train_flow.batch_size,
    verbose=1,
#    validation_data=valid_flow,
#    validation_steps = valid_flow.n // valid_flow.batch_size,
    max_queue_size = QUEUE_SIZE,
    workers = WORKERS,
    callbacks = callbacks
    )


# # Download link #

# In[ ]:


from IPython.display import FileLink
FileLink(best_weight_file)


# # Run prediction #

# In[ ]:


test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


# In[ ]:


df_test = pd.read_csv(test_data_file)
df_test['filename'] = df_test['id_code'].apply(lambda i : "{}.png".format(i))


# In[ ]:


test_flow = test_generator.flow_from_dataframe(directory=test_image_directory, dataframe=df_test, x_col='filename', batch_size=BATCH_SIZE, max_queue_size=128, class_mode=None, target_size=(IMAGE_SIZE, IMAGE_SIZE))


# In[ ]:


y_pred = model.predict_generator(
                    generator = test_flow,
                    steps = (test_flow.n // test_flow.batch_size) + 1,
                    verbose=1,
                    workers=WORKERS,
                    max_queue_size=QUEUE_SIZE
                )


# In[ ]:


df_test['diagnosis'] = np.argmax(y_pred, axis=-1).astype('uint8')


# ### Display results ###

# In[ ]:


df_test.groupby('diagnosis').count()

