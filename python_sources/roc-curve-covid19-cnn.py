#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


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


# Recreate the exact same model, including its weights and the optimizer
base_model = tf.keras.models.load_model('/kaggle/input/covid19-46/final_lstm_model.h5')

# Show the model architecture
base_model.summary()


# In[ ]:


dataset_path = './dataset'
log_path = './logs'


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'rm -rf dataset\nmkdir -p logs\nmkdir -p dataset/train\nmkdir -p dataset/val\nmkdir -p dataset/test')


# In[ ]:


df = pd.read_csv('../input/covid19-pneumonia-normal-chest-xraypa-dataset/metadata.csv')
df_y = df['class']
df_x = df['directory']


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y, stratify=df_y, test_size=0.20, random_state=7)


# In[ ]:


Y_train.value_counts()


# In[ ]:


Y_test.value_counts()


# In[ ]:


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


# In[ ]:


test = pd.concat([X_test, Y_test], axis = 1)
test.head()


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


predY = base_model.predict_generator(
            test_generator,
            steps=test_generator.n//test_generator.batch_size,
            verbose=1)


# In[ ]:


testY = test_generator.classes


# In[ ]:


class_to_label_map = {2 : 'Pneumonia', 1 : 'COVID-19', 0 : 'Normal'}


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
    hist_csv_file = 'history.csv'
#     plt.figure(figsize=(16, 10))
    lw = 2

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

#     fpr_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in fpr.items() ]))
#     tpr_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in tpr.items() ]))
#     roc_auc_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in roc_auc.items() ]))
#     with open(hist_csv_file, mode='a') as f:
#         fpr_df.to_csv(f)
#         tpr_df.to_csv(f)
#         roc_auc_df.to_csv(f)
        
    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.4f) for %s' % (roc_auc[i], class_to_label_map[i]))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()
    

plot_multiclass_roc(testY, predY, n_classes=3, figsize=(16, 10))


# In[ ]:


import seaborn as sns
def plot_multiclass_roc(y_test, y_score, n_classes, figsize=(17, 6)):

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    hist_csv_file = 'history.csv'
    plt.figure(figsize=(10, 10))
    lw = 3

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

#     fpr_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in fpr.items() ]))
#     tpr_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in tpr.items() ]))
#     roc_auc_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in roc_auc.items() ]))
#     with open(hist_csv_file, mode='a') as f:
#         fpr_df.to_csv(f)
#         tpr_df.to_csv(f)
#         roc_auc_df.to_csv(f)
            
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                 ''.format(class_to_label_map[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
#     plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

plot_multiclass_roc(testY, predY, n_classes=3, figsize=(16, 10))


# In[ ]:


import seaborn as sns
def plot_multiclass_roc(y_test, y_score, n_classes, figsize=(17, 6)):

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    hist_csv_file = 'history.csv'
    plt.figure(figsize=(10, 10))
    lw = 2

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

#     fpr_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in fpr.items() ]))
#     tpr_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in tpr.items() ]))
#     roc_auc_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in roc_auc.items() ]))
#     with open(hist_csv_file, mode='a') as f:
#         fpr_df.to_csv(f)
#         tpr_df.to_csv(f)
#         roc_auc_df.to_csv(f)
            
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                 ''.format(class_to_label_map[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
#     plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
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




