#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import gc
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Activation, BatchNormalization, Dropout, LSTM, ConvLSTM2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input,Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation, LSTM, ConvLSTM2D, Lambda, Reshape, BatchNormalization, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard,TerminateOnNaN, LearningRateScheduler
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.layers import Lambda, Reshape, DepthwiseConv2D, ZeroPadding2D, Add, MaxPooling2D,Activation, Flatten, Conv2D, Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import shutil
import cv2
import os
from datetime import datetime
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.list_physical_devices('GPU')
# tf.debugging.set_log_device_placement(True)


# In[ ]:


dataset_path = './dataset'
log_path = './logs'


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'rm -rf dataset\nmkdir -p logs\nmkdir -p dataset/train\nmkdir -p dataset/val\nmkdir -p dataset/test')


# In[ ]:


# !rm -rf dataset
# !rm -rf logs


# In[ ]:


# dataframe containing the filenames of the images (e.g., GUID filenames) and the classes
df = pd.read_csv('../input/covid19-pneumonia-normal-chest-xraypa-dataset/metadata.csv')
df_y = df['class']
df_x = df['directory']

# df_x = df
# del df_x['class']


# In[ ]:


df_x


# In[ ]:


df_y


# In[ ]:


df_y.value_counts()


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y, stratify=df_y, test_size=0.20, random_state=7)


# In[ ]:


Y_train.value_counts()


# In[ ]:


Y_test.value_counts()


# In[ ]:


test = pd.concat([X_test, Y_test], axis = 1)
test.head()


# In[ ]:


# used to copy files according to each fold
def copy_images(df, directory):
    
    # input and output directory
    input_path = "../input/covid19-pneumonia-normal-chest-xray-pa-dataset"     
    output_path = "dataset/" + directory

    # remove all files from previous fold
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # create folder for files from this fold
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # create subfolders for each class
    classs=['normal','covid','pneumonia']
    for c in classs:
        if not os.path.exists(output_path + '/' + c):
            os.makedirs(output_path + '/' + c)
        
    # copy files
    for i, row in df.iterrows():
        path_from = "{}/{}".format(input_path, row['directory'])
        path_to = "{}/{}".format(output_path, row['directory'])
        shutil.copy(path_from, path_to)
#         print(path_to)
        


# In[ ]:


copy_images(test, 'test')


# In[ ]:


get_ipython().system('ls ./dataset/test')


# In[ ]:


len(os.listdir('./dataset/test/covid'))


# In[ ]:


len(os.listdir('./dataset/test/normal'))


# In[ ]:


len(os.listdir('./dataset/test/pneumonia'))


# In[ ]:


def get_model():
    
    # Input layer
    img_dims=224
    inputs = Input(shape=(img_dims, img_dims, 3))

    # 1st conv block
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', trainable=False, padding='same')(inputs)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', trainable=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # 2nd conv block
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', trainable=False, padding='same')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', trainable=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # 3rd conv block
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', trainable=False, padding='same')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', trainable=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # 4th conv block
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', trainable=False, padding='same')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', trainable=False, padding='same')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', trainable=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # 5th conv block
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', trainable=False, padding='same')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', trainable=False, padding='same')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', trainable=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
#     x = AveragePooling2D(pool_size=(4, 4))(x)

#     # LSTM layer
#     x = Reshape((49, 512))(x)
#     x = ((LSTM(512, activation="relu", return_sequences=True, trainable=False)))(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
    
    # FC layer
    x = Flatten(name="flatten")(x)
    x = Dense(units=64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Output layer
    output = Dense(units=3, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=output)
    opt = RMSprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    
    return model


# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[ ]:


test_generator = test_datagen.flow_from_directory(
    directory=r"./dataset/test/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)


# In[ ]:


from math import floor
N_FOLDS = 5
EPOCHS = 50
INIT_LR = 3e-4
T_BS = 16
V_BS = 16
decay_rate = 0.95
decay_step = 1

skf = StratifiedKFold(n_splits = 5, random_state = 7, shuffle = True)
log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [ModelCheckpoint(filepath='best_cnn_model.h5', monitor='val_loss',mode='min',verbose=1,save_best_only=True,save_weights_only=True),
             LearningRateScheduler(lambda epoch : INIT_LR * pow(decay_rate, floor(epoch / decay_step))), tensorboard_callback]


# In[ ]:


# CREATE CALLBACKS
# checkpoint = tf.keras.callbacks.ModelCheckpoint(
#                         filepath='best_cnn_model.h5',
#                         monitor='val_accuracy',
#                         verbose=1, 
#                         save_best_only=True, 
#                         mode='max')

# best_weights = ModelCheckpoint('best_cnn_model.h5', verbose=1, monitor='val_accuracy', save_best_only=True, mode='auto')
# callbacks_list = [checkpoint]


# In[ ]:


class_to_label_map = {2 : 'pneumonia', 1 : 'covid', 0 : 'normal'}


# In[ ]:


submission_prediction = []
submission_predictions = []

for epoch, (train_index, val_index) in enumerate(skf.split(X_train, Y_train)):
    x_train, x_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train, y_val = Y_train.iloc[train_index], Y_train.iloc[val_index]
    
    print(y_train.value_counts())
    print(y_val.value_counts())

    train = pd.concat([x_train, y_train], axis=1)
    val = pd.concat([x_val, y_val], axis = 1)
    
    # copy the images according to the fold
    copy_images(train, 'train')
    copy_images(val, 'val')
    
    print('Running fold '+ str(epoch+1))
    
    # CREATE MODEL
    model = get_model()
    model.summary()
    
    # Load Model Weights
    if epoch != 0:
        model.load_weights('best_cnn_model.h5') 
    
    train_generator = train_datagen.flow_from_directory(
        directory=r"./dataset/train/",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=16,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    valid_generator = val_datagen.flow_from_directory(
        directory=r"./dataset/val/",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=16,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    
    history = model.fit_generator(
                generator=train_generator,
                steps_per_epoch=train_generator.n//train_generator.batch_size,
                validation_data=valid_generator,
                validation_steps=valid_generator.n//valid_generator.batch_size,
                epochs=EPOCHS,
                callbacks=callbacks
    )
    
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = 'history.csv'
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax = ax.ravel()
    for i, met in enumerate(['accuracy', 'loss']):
        ax[i].plot(history.history[met])
        ax[i].plot(history.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('number of epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['train', 'val'])
    plt.savefig('accuracy_performance_graph'+'_'+str(epoch)+'.png')
    
    test_generator.reset()
    predY=model.predict_generator(
            test_generator,
            steps=test_generator.n//test_generator.batch_size,
            verbose=1)
    submission_prediction.append(predY)

    if epoch >= 1:
        submission_predictions.append(predY)
        
    testY = test_generator.classes
    confusion__matrix=confusion_matrix(testY, np.argmax(predY, axis = -1))
    cr=(classification_report(testY, np.argmax(predY, axis = -1), target_names=class_to_label_map, output_dict=True))
    print (cr)
    print(confusion__matrix)
    
    cm_df = pd.DataFrame(confusion__matrix)
    cr_df = pd.DataFrame(cr)
    with open(hist_csv_file, mode='a') as f:
        hist_df.to_csv(f)
        cm_df.to_csv(f)
        cr_df.to_csv(f)
    
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title("Model accuracy & loss")
    plt.ylabel("accuracy & loss")
    plt.xlabel("number of epochs")
    plt.legend(["train_accuracy", "val_accuracy", "train_loss", "val_loss"], loc="upper left")
    plt.savefig('accuracy_loss_performance'+'_'+str(epoch)+'.png')
    
    model.save('final_cnn_model.h5') 
    
    del history
    del model
    gc.collect()


# In[ ]:


len(os.listdir('./dataset/train/normal'))


# In[ ]:


len(os.listdir('./dataset/val/normal'))


# In[ ]:


predY = np.average(submission_predictions, axis = 0, weights = [2**i for i in range(len(submission_predictions))])


# In[ ]:


testY = test_generator.classes


# In[ ]:


roc_auc_score(testY, predY, multi_class='ovo')


# In[ ]:


roc_auc_score(testY, predY, multi_class='ovr')


# In[ ]:


import seaborn as sns
def plot_multiclass_roc(y_test, y_score, n_classes, figsize=(17, 6)):

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for %s' % (roc_auc[i], class_to_label_map[i]))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()

plot_multiclass_roc(testY, predY, n_classes=3, figsize=(16, 10))


# In[ ]:


cm_mat = confusion_matrix(testY, np.argmax(predY, axis = -1))


# In[ ]:


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Greens')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize = 'larger')

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize = 'larger')
        plt.yticks(tick_marks, target_names, fontsize = 'larger')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize = 'larger')
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize = 'larger')


    plt.tight_layout()
    plt.ylabel('True label', fontsize = 'larger')
    plt.xlabel('Predicted label', fontsize = 'larger')
#     plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
plot_confusion_matrix(cm_mat, 
                      normalize = False,
                      target_names = ['Normal', 'COVID-19', 'Pneumonia'],
                      title        = "Confusion Matrix")


# In[ ]:


print(classification_report(testY, np.argmax(predY, axis = -1), target_names = ['normal', 'covid', 'pneumonia']))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


predY = np.average(submission_prediction, axis = 0, weights = [2**i for i in range(len(submission_prediction))])
predY


# In[ ]:


cm_mat = confusion_matrix(testY, np.argmax(predY, axis = -1))
cm_mat


# In[ ]:


print(classification_report(testY, np.argmax(predY, axis = -1), target_names = ['normal', 'covid', 'pneumonia']))


# In[ ]:


get_ipython().system('rm -rf dataset')
get_ipython().system('rm -rf logs')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# import os
# for dirname, _, filenames in os.walk('../input/covid19-pneumonia-normal-chest-xray-pa-dataset'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# In[ ]:


#     #create subfolders for each class
#     for c in set(list(df['class'])):
#         if not os.path.exists(destination_directory + '/' + c):
#             os.makedirs(destination_directory + '/' + c)

#     # copy files for this fold from a directory holding all the files
#     for i, row in df.iterrows():
#         try:
#             path_from = "{}/{}".format(input_path, row['directory'])
            
#             path_to = "{}/{}".format(output_directory, row['directory'])
            
#             print(path_to)
            
            
# #             imagePath = os.path.sep.join(['../input/data/images_001', row['directory']])

#             # move from folder keeping all files to training, test, or validation folder (the "directory" argument)
#             shutil.copy(path_from, path_to)
#         except Exception:
#             print("Error when copying {}: {}".format(row['directory'], Exception))

