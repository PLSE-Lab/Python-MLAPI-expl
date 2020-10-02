#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[ ]:


get_ipython().system('pip install imutils')
get_ipython().system('pip install image-classifiers==1.0.0b1')


# In[ ]:


# import the necessary packages
import tensorflow as tf
import gc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Activation, BatchNormalization, Dropout, LSTM, ConvLSTM2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input,Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation, LSTM, ConvLSTM2D, Lambda, Reshape, BatchNormalization, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard,TerminateOnNaN, LearningRateScheduler
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.layers import Lambda, Reshape, DepthwiseConv2D, ZeroPadding2D, Add, MaxPooling2D,Activation, Flatten, Conv2D, Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import shutil
import cv2
import os
from classification_models.tfkeras import Classifiers
from datetime import datetime
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


dataset_path = './dataset'
log_path = './logs'


# ## Build Dataset

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'rm -rf dataset\nmkdir -p dataset/covid\nmkdir -p dataset/normal\nmkdir -p dataset/pneumonia\nmkdir -p logs')


# ### Covid xray dataset

# In[ ]:


covid_dataset_path='../input/covid19-update-datasets/covid-chestxray-dataset-master'


# In[ ]:


len(os.listdir('../input/covid19-update-datasets/covid-chestxray-dataset-master/images'))


# In[ ]:


csvPath = os.path.sep.join([covid_dataset_path, "metadata.csv"])
df = pd.read_csv(csvPath)

for (i, row) in df.iterrows():
    # if (1) the current case is not COVID-19 or (2) this is not
    # a 'PA' view, then ignore the row
    if row["finding"] != "COVID-19" or row["view"] != "PA":
        continue

    imagePath = os.path.sep.join([covid_dataset_path, "images", row["filename"]])

    if not os.path.exists(imagePath):
        continue

    filename = row["filename"].split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/covid", filename])

    shutil.copy2(imagePath, outputPath)


# In[ ]:


len(os.listdir('../working/dataset/covid'))


# In[ ]:


# !ls ../input/covid19-radiopaedia/covid_radiology


# In[ ]:


covid_dataset_path2 = '../input/covid19-radiopaedia/covid_radiology'
outputPath = '../working/dataset/covid'
src_files = os.listdir(covid_dataset_path2)
for file_name in src_files:
    full_file_name = os.path.join(covid_dataset_path2, file_name)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, outputPath)


# In[ ]:


len(os.listdir('../working/dataset/covid'))


# In[ ]:


samples = 248


# ### Build normal xray dataset

# In[ ]:


pneumonia_dataset_path ='../input/chest-xray-pneumonia/chest_xray'


# In[ ]:


basePath = os.path.sep.join([pneumonia_dataset_path, "train", "NORMAL"])
imagePaths = list(paths.list_images(basePath))

# randomly sample the image paths
random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:samples]

for (i, imagePath) in enumerate(imagePaths):
    filename = imagePath.split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/normal", filename])

    shutil.copy2(imagePath, outputPath)


# In[ ]:


len(os.listdir('../working/dataset/normal'))


# ### Build pneumonia xray dataset

# In[ ]:


basePath = os.path.sep.join([pneumonia_dataset_path, "train", "PNEUMONIA"])
imagePaths = list(paths.list_images(basePath))

# randomly sample the image paths
random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:samples]

for (i, imagePath) in enumerate(imagePaths):
    filename = imagePath.split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/pneumonia", filename])

    shutil.copy2(imagePath, outputPath)


# In[ ]:


len(os.listdir('../working/dataset/pneumonia'))


# ## Plot x-rays

# In[ ]:


def ceildiv(a, b):
    return -(-a // b)

def plots_from_files(imspaths, figsize=(10,5), rows=1, titles=None, maintitle=None):
    f = plt.figure(figsize=(40,10))
    if maintitle is not None: plt.suptitle(maintitle, fontsize=20)
    for i in range(len(imspaths)):
        sp = f.add_subplot(rows, ceildiv(len(imspaths), rows), i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=26)
        img = plt.imread(imspaths[i])
        plt.imshow(img, cmap = 'gray')


# In[ ]:


normal_images = list(paths.list_images(f"{dataset_path}/normal"))
covid_images = list(paths.list_images(f"{dataset_path}/covid"))
covid_images = list(paths.list_images(f"{dataset_path}/pneumonia"))


# In[ ]:


plots_from_files(normal_images, rows=8, maintitle="Normal X-ray images")


# In[ ]:


plots_from_files(covid_images, rows=8, maintitle="Covid-19 X-ray images")


# In[ ]:


plots_from_files(covid_images, rows=8, maintitle="pneumonia X-ray images")


# ## Data preprocessing

# In[ ]:


class_to_label_map = {'pneumonia' : 2, 'covid' : 1, 'normal' : 0}


# In[ ]:


imagePaths = list(paths.list_images(dataset_path))
data = []
labels = []
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)
    data.append(image)
    labels.append(class_to_label_map[label])
    
data = np.array(data) / 255.0
labels = np.array(labels)


# In[ ]:


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.15, stratify=labels, random_state=42)
train_datagen = ImageDataGenerator(
                                   rotation_range=15,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator()


# In[ ]:


trainYSparse = trainY
trainY = to_categorical(trainY)


# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.list_physical_devices('GPU')
# tf.debugging.set_log_device_placement(True)


# ## Model

# In[ ]:


from math import floor
N_FOLDS = 5
EPOCHS = 50
INIT_LR = 3e-4
T_BS = 16
V_BS = 16
decay_rate = 0.95
decay_step = 1

skf = StratifiedKFold(n_splits=N_FOLDS, random_state=1234,)
log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [ModelCheckpoint(filepath='cnn_model_weight.h5', monitor='val_loss',mode='min',verbose=1,save_best_only=True,save_weights_only=True),
             LearningRateScheduler(lambda epoch : INIT_LR * pow(decay_rate, floor(epoch / decay_step))), tensorboard_callback]


# ### Training

# In[ ]:


submission_predictions = []
for epoch, skf_splits in zip(range(0,N_FOLDS),skf.split(trainX,trainYSparse)):

    train_idx = skf_splits[0]
    val_idx = skf_splits[1]
    print(len(train_idx),len(val_idx))
    
    # Create Model..........................................
    
    # Input layer
    baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    for layer in baseModel.layers:
        layer.trainable = False
    x = baseModel.output
    
#     x = AveragePooling2D(pool_size=(3,3), name='avg_pool')(x)

    # LSTM layer
    x = Reshape((49, 2048))(x)
    x = ((LSTM(2048, activation="relu", return_sequences=True, trainable=False)))(x)
#     x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # FC layer
    x = Flatten(name="flatten")(x)
    
    # fc1 layer
    x = Dense(units=4096, activation='relu')(x)
#     x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # fc2 layer
    x = Dense(units=4096, activation='relu')(x)
#     x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    output = Dense(units=3, activation='softmax')(x)

    model = Model(inputs=baseModel.input, outputs=output)
#     opt = SGD(lr=0.01)
    opt = RMSprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=["accuracy"])
#                   metrics=['acc', tf.keras.metrics.AUC()],experimental_run_tf_function=False)
    model.summary()
    
    if epoch != 0:
        model.load_weights('cnn_model_weight.h5') 
    
    history = model.fit(
                train_datagen.flow(trainX[train_idx], trainY[train_idx], batch_size=T_BS),
                steps_per_epoch=len(train_idx) // T_BS,
                epochs=EPOCHS,
                validation_data = val_datagen.flow(trainX[val_idx], trainY[val_idx], batch_size=V_BS),
                validation_steps = len(val_idx) // V_BS,
                callbacks=callbacks)
    
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
    
    predY = model.predict(testX, batch_size=V_BS)
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
    
    if epoch >= 1:
        preds = model.predict(testX, batch_size=V_BS)
        submission_predictions.append(preds)
    
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
    
    model.save('final_ResNet50_lstm_model.h5') 
    
    del history
    del model
    gc.collect()
    


# ### Plot trining metrics

# ### Evaluation

# In[ ]:


predY = np.average(submission_predictions, axis = 0, weights = [2**i for i in range(len(submission_predictions))])


# In[ ]:


roc_auc_score(testY, predY, multi_class='ovo')


# In[ ]:


roc_auc_score(testY, predY, multi_class='ovr')


# In[ ]:


class_to_label_map = {2 : 'pneumonia', 1 : 'covid', 0 : 'normal'}


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


# #### Confusion matrix

# In[ ]:


cm_mat = confusion_matrix(testY, np.argmax(predY, axis = -1))


# In[ ]:


# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import numpy as np
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
        cmap = plt.get_cmap('Blues')

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


get_ipython().system('rm -rf dataset')
get_ipython().system('rm -rf logs')


# In[ ]:




