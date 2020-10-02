#!/usr/bin/env python
# coding: utf-8

# based on [General base model stratifiedkfold ensemble w/test](https://www.kaggle.com/meditech101/general-base-model-stratifiedkfold-ensemble-w-test) kernel <br/>
# is originated from [bronze medal[3rd ML Month] Xception, StratifiedKFold, Ensemble](https://www.kaggle.com/janged/3rd-ml-month-xception-stratifiedkfold-ensemble)
# 
# F1 score, Cutout Augmentation, and Test Time Augmentation are applied.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import warnings
import seaborn as sns
import matplotlib.pylab as plt
import PIL
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score
from keras import backend as K
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

warnings.filterwarnings('ignore')
K.image_data_format()


# In[ ]:


BATCH_SIZE = 32
EPOCHS = 50
k_folds = 2
TTA_STEPS = 5
PATIENCE = 6
SEED = 2019
BASE_MODEL = Xception
IMAGE_SIZE = 299


# In[ ]:


DATA_PATH = '../input'

TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')
TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')

df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))

model_path = './'
if not os.path.exists(model_path):
    os.mkdir(model_path)


# In[ ]:


def crop_boxing_img(img_name, margin=0, size=(IMAGE_SIZE,IMAGE_SIZE)):
    if img_name.split('_')[0] == 'train':
        PATH = TRAIN_IMG_PATH
        data = df_train
    else:
        PATH = TEST_IMG_PATH
        data = df_test

    img = PIL.Image.open(os.path.join(PATH, img_name))
    pos = data.loc[data["img_file"] == img_name, ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)

    width, height = img.size
    x1 = max(0, pos[0] - margin)
    y1 = max(0, pos[1] - margin)
    x2 = min(pos[2] + margin, width)
    y2 = min(pos[3] + margin, height)

    return img.crop((x1, y1, x2, y2)).resize(size)


# In[ ]:


get_ipython().run_cell_magic('time', '', "TRAIN_CROPPED_PATH = '../cropped_train'\nTEST_CROPPED_PATH = '../cropped_test'\n\nif (os.path.isdir(TRAIN_CROPPED_PATH) == False):\n    os.mkdir(TRAIN_CROPPED_PATH)\n\nif (os.path.isdir(TEST_CROPPED_PATH) == False):\n    os.mkdir(TEST_CROPPED_PATH)\n\nfor i, row in df_train.iterrows():\n    cropped = crop_boxing_img(row['img_file'])\n    cropped.save(os.path.join(TRAIN_CROPPED_PATH, row['img_file']))\n\nfor i, row in df_test.iterrows():\n    cropped = crop_boxing_img(row['img_file'])\n    cropped.save(os.path.join(TEST_CROPPED_PATH, row['img_file']))")


# In[ ]:


df_train['class'] = df_train['class'].astype('str')
df_train = df_train[['img_file', 'class']]
df_test = df_test[['img_file']]


# In[ ]:


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[ ]:


def get_callback(model_name, patient):
    ES = EarlyStopping(
        monitor='val_f1_m', 
        patience=patient, 
        mode='max', 
        verbose=1)
    RR = ReduceLROnPlateau(
        monitor = 'val_f1_m', 
        factor = 0.5, 
        patience = patient / 2, 
        min_lr=0.000001, 
        verbose=1, 
        mode='max')
    MC = ModelCheckpoint(
        filepath=model_name, 
        monitor='val_f1_m', 
        verbose=1, 
        save_best_only=True, 
        mode='max')

    return [ES, RR, MC]


# In[ ]:


def get_model(model_name, iamge_size):
    base_model = model_name(weights='imagenet', input_shape=(iamge_size,iamge_size,3), include_top=False)
    #base_model.trainable = False
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(2048, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.Dropout(0.15))
 
    model.add(layers.Dense(196, activation='softmax', kernel_initializer='lecun_normal'))
    #model.summary()

    optimizer = optimizers.Nadam(lr=0.0003)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', f1_m, precision_m, recall_m])

    return model


# In[ ]:


#ref: https://github.com/yu4u/cutout-random-erasing/blob/master/cifar10_resnet.py
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


# In[ ]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    #featurewise_center= True,  # set input mean to 0 over the dataset
    #samplewise_center=True,  # set each sample mean to 0
    #featurewise_std_normalization= True,  # divide inputs by std of the dataset
    #samplewise_std_normalization=True,  # divide each input by its std
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.3,
    shear_range=0.5,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest',
    preprocessing_function = get_random_eraser(v_l=0, v_h=255),
    )

valid_datagen = ImageDataGenerator(
    rescale=1./255,
    #featurewise_center= True,  # set input mean to 0 over the dataset
    #samplewise_center=True,  # set each sample mean to 0
    #featurewise_std_normalization= True,  # divide inputs by std of the dataset
    #samplewise_std_normalization=True  # divide each input by its std
    )
test_datagen = ImageDataGenerator(
    rescale=1./255,
    #featurewise_center= True,  # set input mean to 0 over the dataset
    #samplewise_center=True,  # set each sample mean to 0
    #featurewise_std_normalization= True,  # divide inputs by std of the dataset
    #samplewise_std_normalization=True,  # divide each input by its std
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.3,
    shear_range=0.5,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest',
    preprocessing_function = get_random_eraser(v_l=0, v_h=255),
    )


# In[ ]:


skf = StratifiedKFold(n_splits=k_folds, random_state=SEED)
#skf = KFold(n_splits=k_folds, random_state=SEED)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'j = 1\nmodel_names = []\nfor (train_index, valid_index) in skf.split(\n    df_train[\'img_file\'], \n    df_train[\'class\']):\n\n    traindf = df_train\n    validdf = df_train.iloc[valid_index, :].reset_index()\n\n    print("=========================================")\n    print("====== K Fold Validation step => %d/%d =======" % (j,k_folds))\n    print("=========================================")\n    \n    train_generator = train_datagen.flow_from_dataframe(\n        dataframe=traindf,\n        directory=TRAIN_CROPPED_PATH,\n        x_col=\'img_file\',\n        y_col=\'class\',\n        target_size= (IMAGE_SIZE, IMAGE_SIZE),\n        color_mode=\'rgb\',\n        class_mode=\'categorical\',\n        batch_size=BATCH_SIZE,\n        seed=SEED,\n        shuffle=True\n        )\n    \n    valid_generator = valid_datagen.flow_from_dataframe(\n        dataframe=validdf,\n        directory=TRAIN_CROPPED_PATH,\n        x_col=\'img_file\',\n        y_col=\'class\',\n        target_size= (IMAGE_SIZE, IMAGE_SIZE),\n        color_mode=\'rgb\',\n        class_mode=\'categorical\',\n        batch_size=BATCH_SIZE,\n        seed=SEED,\n        shuffle=True\n        )\n    \n    model_name = model_path + str(j) + \'_\' + \'Xception\' + \'.hdf5\'\n    model_names.append(model_name)\n    \n    model = get_model(BASE_MODEL, IMAGE_SIZE)\n    \n    try:\n        model.load_weights(model_name)\n    except:\n        pass\n        \n    history = model.fit_generator(\n        train_generator,\n        steps_per_epoch=len(traindf.index) / BATCH_SIZE,\n        epochs=EPOCHS,\n        validation_data=valid_generator,\n        validation_steps=len(validdf.index) / BATCH_SIZE,\n        verbose=1,\n        shuffle=False,\n        callbacks = get_callback(model_name, PATIENCE)\n        )\n        \n    j+=1')


# In[ ]:


print(history.history.keys())  


# In[ ]:


plt.figure(figsize=(7, 7), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(history.history['acc']) 
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()


# In[ ]:


plt.figure(figsize=(7, 7), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()


# In[ ]:


plt.figure(figsize=(7, 7), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(history.history['f1_m'])  
plt.plot(history.history['val_f1_m'])  
plt.title('model f1_score')  
plt.ylabel('f1_score')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()


# In[ ]:


test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=TEST_CROPPED_PATH,
    x_col='img_file',
    y_col=None,
    target_size= (IMAGE_SIZE, IMAGE_SIZE),
    color_mode='rgb',
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# In[ ]:


prediction = []
for i, name in enumerate(model_names):
    model = get_model(BASE_MODEL, IMAGE_SIZE)
    model.load_weights(name)
    
    preds = []
    for j in range(TTA_STEPS):
        test_generator.reset()
        pred = model.predict_generator(
            generator=test_generator,
            steps = len(df_test)/BATCH_SIZE,
            verbose=1
        )
        preds.append(pred)
    pred_tta = np.mean(preds, axis=0)
    prediction.append(pred_tta)

y_pred = np.mean(prediction, axis=0)


# In[ ]:


preds_class_indices=np.argmax(y_pred, axis=1)


# In[ ]:


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
final_pred = [labels[k] for k in preds_class_indices]


# In[ ]:


submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
submission["class"] = final_pred
submission.to_csv("submission.csv", index=False)
submission.head()


# In[ ]:



