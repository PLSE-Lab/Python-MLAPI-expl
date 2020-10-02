#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# In[ ]:


IMAGE_DIR = '/kaggle/input/plant-pathology-2020-fgvc7/images/'
train_path = '/kaggle/input/plant-pathology-2020-fgvc7/train.csv'
test_path = '/kaggle/input/plant-pathology-2020-fgvc7/test.csv'
IMAGE_W = 150
IMAGE_H = 150


# In[ ]:


def preprocess(df):
    for index, img in enumerate(df.image_id):
        img = img+'.jpg'
        df.image_id[index]=img


# In[ ]:


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
preprocess(train_df)
preprocess(test_df)
train_df.head()


# In[ ]:


def train_val_split(df, val_ratio=0.2):
   val_rows = (np.random.rand(int(val_ratio*df.shape[0]))*df.shape[0]).astype(int)
   val_df = df.iloc[val_rows]
   df.drop(val_rows, axis=0, inplace=True)
   val_df = val_df.reset_index().drop(['index'], axis=1)
   df = df.reset_index().drop(['index'], axis=1)
   int_dict = {'healthy':float, 'multiple_diseases':float, 'rust':float, 'scab':float}
   df = df.astype(int_dict) 
   val_df = val_df.astype(int_dict)
   return df, val_df


# In[ ]:


train, val = train_val_split(train_df)


# In[ ]:


train.head()


# In[ ]:


val.head()


# In[ ]:


labels = list(train.columns[1:])


# In[ ]:


def print_class_freq(df, labels=labels):
    for col in labels:
        print(f'{col}: {sum(df[col])}')


# In[ ]:


print_class_freq(train)


# In[ ]:


print_class_freq(val)


# In[ ]:


from tensorflow.keras.preprocessing.image import load_img, img_to_array


# In[ ]:


def SMOTE_Data(train_df=train):
    from imblearn.over_sampling import SMOTE
    X_train = []
    for img in train_df['image_id']:
        loaded_img = load_img(os.path.join(IMAGE_DIR, img), target_size=(IMAGE_W, IMAGE_H))
        img_arr = img_to_array(loaded_img)
        X_train.append(img_arr)
        
    print(np.array(X_train).shape)  
    y_train = train_df.drop('image_id', axis=1, inplace=False)
    print(y_train.head())
    y_train = np.array(y_train.values)
    X_train = np.array(X_train)
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train.reshape((-1, IMAGE_W * IMAGE_H * 3)), y_train)
    X_train.reshape(-1, IMAGE_W, IMAGE_H, 3)
    return X_train, y_train


# In[ ]:


X_train, y_train = SMOTE_Data()
print(X_train.shape)
print(y_train.shape)


# In[ ]:


X_train = X_train.reshape(-1, IMAGE_W, IMAGE_H, 3)
print(X_train.shape)


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D


# In[ ]:


tst_path = '/kaggle/input/test-image-leaf/Test_img.jpg'
img = load_img(tst_path, target_size=(150, 150, 3))
plt.imshow(img)


# In[ ]:


def preprocess_1_image(img, train_df=train, image_dir=IMAGE_DIR, x_col='image_id', y_cols=labels, sample_size=100, batch_size=32, seed=42, target_w = IMAGE_W, target_h = IMAGE_H):
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=IMAGE_DIR, 
        x_col="image_id", 
        y_col=y_cols, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]
    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)
    test_img = image_generator.flow(img)
    return test_img


# In[ ]:


img_array = img_to_array(img)
img_array = img_array.reshape(1, 150, 150, 3)
test_img = preprocess_1_image(img_array)


# In[ ]:


def get_valid_generator(test_df=test_df, valid_df=val, train_df=train, image_dir=IMAGE_DIR, x_col='image_id', y_cols=labels, sample_size=100, batch_size=32, seed=42, target_w = IMAGE_W, target_h = IMAGE_H):

    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=IMAGE_DIR, 
        x_col="image_id", 
        y_col=y_cols, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            class_mode=None,
            batch_size=1,
            shuffle=False,
            target_size=(target_w,target_h))
    return valid_generator, test_generator


# In[ ]:


def get_train_generator(X=X_train, y=y_train, image_dir=IMAGE_DIR, x_col='image_id', y_cols=labels, shuffle=False, batch_size=32, seed=42, target_w = IMAGE_W, target_h = IMAGE_H):
    
    
    print("getting train generator...") 
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow(
            X,y,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed)
    
    return generator


# In[ ]:


train_generator = get_train_generator()
valid_generator, test_generator = get_valid_generator()


# In[ ]:


X, y = train_generator.__getitem__(0)
print(X.shape)
print(y.shape)
plt.imshow(X[31])


# In[ ]:


def compute_class_freqs(labels):

    N = labels.shape[0]
    
    positive_frequencies = np.mean(labels, axis=0)
    negative_frequencies = 1 - positive_frequencies

    return positive_frequencies, negative_frequencies


# In[ ]:


freq_pos, freq_neg = compute_class_freqs(y_train)

pos_weights = freq_neg
neg_weights = freq_pos


# In[ ]:


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    
    def weighted_loss(y_true, y_pred):
        
        # initialize loss to zero
        loss = 0.0
        

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
            loss += K.mean(-(pos_weights[i]*y_true[:, i]*K.log(y_pred[:, i]+epsilon)
                             + neg_weights[i]*(1-y_true[:, i])*K.log((1-y_pred[:, i])+epsilon)))
        return loss
    
    return weighted_loss


# In[ ]:


def inception_model():
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)

    x = base_model.output

    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # and a logistic layer
    predictions = Dense(len(labels), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print('Model has compiled')
    return model


# In[ ]:


def dense_model():
    base_model = DenseNet121(weights='imagenet', include_top=False)

    x = base_model.output

    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # and a logistic layer
    predictions = Dense(len(labels), activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights), metrics=['accuracy'])
    print('Model has compiled')
    return model


# In[ ]:


def make_model(labels=labels):
    model = Sequential()
    model.add(ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_W, IMAGE_H, 3)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(len(labels), activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print('Model has compiled')
    return model


# In[ ]:


model = inception_model()
model.summary()


# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
LR_reduce=ReduceLROnPlateau(monitor='val_accuracy',
                            factor=.5,
                            patience=7,
                            min_lr=.00001,
                            verbose=1)

ES_monitor=EarlyStopping(monitor='val_loss',
                          patience=20)
checkpoint_filepath = '/kaggle/working'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    verbose=1,
    save_best_only=False)


# In[ ]:


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
history = model.fit(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID, epochs=150, callbacks=[ES_monitor, LR_reduce, model_checkpoint_callback])


# In[ ]:


model.save_weights('plant.h5')


# In[ ]:


test_pred = model.predict_generator(test_img)


# In[ ]:


test_pred


# In[ ]:


test_pred[0][np.argmax(test_pred[0])]


# In[ ]:


print(f'{labels[np.argmax(test_pred[0])]} with {test_pred[0][np.argmax(test_pred[0])]} confidence')


# In[ ]:


predicted_vals = model.predict_generator(valid_generator, steps = len(valid_generator))


# In[ ]:


import matplotlib.pyplot as plt
#copied from Coursera util package
from keras.preprocessing import image
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity
import cv2

def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals


# In[ ]:


auc_rocs = get_roc_curve(labels, predicted_vals, valid_generator)


# In[ ]:


predicted_vals.shape


# In[ ]:


valid_generator.labels


# In[ ]:


test_preds = model.predict_generator(test_generator, steps = len(test_generator))


# In[ ]:


test_preds.shape


# In[ ]:


test_preds[:, 0]


# In[ ]:


test = pd.read_csv(test_path)
test.head()


# In[ ]:


test['healthy'] = test_preds[:, 0]
test['multiple_diseases'] = test_preds[:, 1]
test['rust'] = test_preds[:, 2]
test['scab'] = test_preds[:, 3]
test.head()


# In[ ]:


test.to_csv('submission3.csv',index=False)


# In[ ]:


test.head()


# In[ ]:


from IPython.display import FileLink
FileLink(r'submission3.csv')


# In[ ]:




