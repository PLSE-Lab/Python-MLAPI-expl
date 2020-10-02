#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('cd', 'input')
get_ipython().system('cp train_v2.csv /kaggle/working/')
get_ipython().run_line_magic('cd', 'train-jpg')
get_ipython().system('cp -r train-jpg /kaggle/working/')
get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('cd', 'working')
get_ipython().system('ls')
get_ipython().system('mv train_v2.csv train.csv')
get_ipython().system('mv train-jpg images')
get_ipython().system('mkdir graph')
get_ipython().system('mkdir downsample')
get_ipython().system('ls')


# In[ ]:


from os import listdir
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img,img_to_array, ImageDataGenerator,array_to_img
from sklearn.model_selection import train_test_split
from keras import backend
import sys
from matplotlib import pyplot as plt
from keras.models import Sequential,Model,load_model
from keras.layers import Conv2D,Dense,Flatten,BatchNormalization,GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
import gc
import time
import random, shutil
from timeit import default_timer as timer
from numpy import zeros,asarray,savez_compressed,load
from keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
import time
from sklearn.metrics import f1_score as f1score
from sklearn.model_selection import GridSearchCV


# In[ ]:


def create_tag_mapping(mapping_csv):
    labels = set()
    for i in range(len(mapping_csv)):
        tags = mapping_csv[mapping_csv.columns[1]][i].split(' ')
        labels.update(tags)
    labels = list(labels)
    labels.sort()
    
    labels_map = {labels[i]:i for i in range(len(labels))}
    inv_labels_map = {i:labels[i] for i in range(len(labels))}
    return labels_map, inv_labels_map

def create_file_mapping(mapping_csv):
    mapping = dict()
    for i in range(len(mapping_csv)):
        name, tags = mapping_csv[mapping_csv.columns[0]][i], mapping_csv[mapping_csv.columns[1]][i]
        mapping[name] = tags.split(' ')
    return mapping

def one_hot_encode(tags, mapping):
    encoding = np.zeros(len(mapping),dtype='uint8')
    for tag in tags:
        encoding[mapping[tag]] = 1
    return encoding

def load_dataset(path, file_mapping, tag_mapping):
    photos, targets = list(), list()
    if len(listdir(folder)) >= 10000:
        filenames = random.sample(listdir(folder),3000)
        for file in filenames:
            photo = load_img(folder+file, target_size=(224,224))
            photo = img_to_array(photo, dtype='uint8')
            tags = file_mapping[file[:-4]]
            target = one_hot_encode(tags, tag_mapping)
            photos.append(photo)
            targets.append(target)
    else:
        for filename in listdir(folder):
            photo = load_img(path+filename, target_size=(224,224))
            photo = img_to_array(photo, dtype='uint8')
            tags = file_mapping[filename[:-4]]
            target = one_hot_encode(tags, tag_mapping)
            photos.append(photo)
            targets.append(target)
    X = np.asarray(photos, dtype='uint8')
    y = np.asarray(targets, dtype='uint8')
    return X,y


# In[ ]:


filename = 'train.csv'
folder = './images/'
print('====== {} Files Exist ======'.format(len(listdir(folder))))
mapping_csv = pd.read_csv(filename)
tag_mapping,inv_tag_mapping = create_tag_mapping(mapping_csv)
file_mapping = create_file_mapping(mapping_csv)
X,y = load_dataset(folder, file_mapping, tag_mapping)
print(X.shape, y.shape)
np.savez_compressed('numpy_data.npz',X,y)
print('======    {} Images , {} Labels  ======'.format(len(X),y.shape[1]))
print('====== Image,Label Numpy File Saved ======')


# In[ ]:


gc.collect()
input_shape = (224,224,3)
output_shape = len(tag_mapping)


# In[ ]:


class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)
        
def load_dataset():
    data = np.load('numpy_data.npz')
    X,y = data['arr_0'], data['arr_1']
    trainX, testX,trainY,testY = train_test_split(X,y,test_size=0.2,random_state=1)
    return trainX,trainY,testX,testY

def f1_score(y_true, y_pred, beta=1):
    y_pred = backend.clip(y_pred, 0, 1)
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
    p = tp / (tp + fp + backend.epsilon())
    r = tp / (tp + fn + backend.epsilon())
    bb = beta ** 2
    f1_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
    return f1_score

def define_model(in_shape=(224,224,3),out_shape=output_shape,trainable_rate=0,classifier='FC'):
    model = VGG16(include_top=False, input_shape=in_shape)
    for layer in model.layers:
        layer.trainable = False
    if trainable_rate == 20:
        model.get_layer('block5_conv1').trainable = True
        model.get_layer('block5_conv2').trainable = True
        model.get_layer('block5_conv3').trainable = True
        model.get_layer('block5_pool').trainable = True
    elif trainable_rate == 50:
        model.get_layer('block3_conv1').trainable = True
        model.get_layer('block3_conv2').trainable = True
        model.get_layer('block3_conv3').trainable = True
        model.get_layer('block3_pool').trainable = True
        model.get_layer('block4_conv1').trainable = True
        model.get_layer('block4_conv2').trainable = True
        model.get_layer('block4_conv3').trainable = True
        model.get_layer('block4_pool').trainable = True
        model.get_layer('block5_conv1').trainable = True
        model.get_layer('block5_conv2').trainable = True
        model.get_layer('block5_conv3').trainable = True
        model.get_layer('block5_pool').trainable = True
    elif trainable_rate == 100:
        for layer in model.layers:
            layer.trainable = True
   
    if classifier == 'FC':
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
        class2 = BatchNormalization()(class1)
    elif classifier == 'GAP':
        class2 = GlobalAveragePooling2D()(model.layers[-1].output)
    
    output = Dense(out_shape, activation='sigmoid')(class2)
    model = Model(inputs=model.inputs, outputs=output)

    opt = SGD(lr=0.0005, momentum=0.9,decay=1e-6)
    model.compile(optimizer=opt, loss='binary_crossentropy',metrics = [f1_score,'accuracy'])
    return model

def summarize_diagnostics(history,rate,classifier):
    plt.subplot(311)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.subplot(312)
    plt.title('F1-score')
    plt.plot(history.history['f1_score'], color='blue', label='train')
    plt.plot(history.history['val_f1_score'], color='orange', label='test')
    plt.subplot(313)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    filename = sys.argv[0].split('/')[-1]
    plt.savefig('./graph/'+filename + "{}%+{}.png".format(rate,classifier))
    plt.close()
    
def run_test_harness():
    trainX, trainY, testX, testY = load_dataset()
    train_datagen = ImageDataGenerator(preprocessing_function=vgg16_preprocess)
    test_datagen = ImageDataGenerator(preprocessing_function=vgg16_preprocess)
    train_it = train_datagen.flow(trainX,trainY, batch_size=32)
    test_it = test_datagen.flow(testX,testY, batch_size=32)
    
    es = EarlyStopping(monitor='val_loss', patience=20, verbose=1,mode='auto')
    rate_list = [0,20,50,100]
    classifier_list = ['FC','GAP']
    final_report = {}
    for rate in rate_list:
        for classifier in classifier_list:
            cb= TimingCallback()
            backend.clear_session()
            gc.collect()
            model = define_model(trainable_rate=rate,classifier=classifier)
            mc = ModelCheckpoint('./{}{}.h5'.format(rate,classifier), monitor='val_loss',mode = 'min',save_best_only=True)
            history = model.fit_generator(train_it, steps_per_epoch=len(train_it), 
                                     validation_data = test_it, validation_steps=len(test_it), epochs=150,verbose=1,callbacks=[es,mc,cb])
            loss, f1_score,accuracy = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
            print("==========================================")
            print("{}% + {} Model".format(rate,classifier))
            print('> loss=%.6f, F1-score=%.3f, accuracy=%.3f' % (loss, f1_score,accuracy))
            print('> time=%.6f' % (sum(cb.logs)))
            print("==========================================")
            final_report["{}% + {}".format(rate,classifier)] = [round(loss,3),round(accuracy*100,3),round(f1_score,3),int(es.stopped_epoch),int(sum(cb.logs))]
            summarize_diagnostics(history,rate,classifier)
    return final_report


# In[ ]:


final_report = run_test_harness()


# In[ ]:


path = "./graph/"
for i in range(8):
    plt.figure(figsize=(30,30))
    plt.subplot(330+1+i)
    plt.title(listdir(path)[i])
    image = plt.imread(path+listdir(path)[i])
    plt.imshow(image)
plt.show()


# In[ ]:


def feature_extract(X):
    model = VGG16(include_top=False, input_shape=(224,224,3))
    for layer in model.layers:
        layer.trainable = False
    
    return model.predict(X)


# In[ ]:


trainX,trainY,testX,testY = load_dataset()
trainX = vgg16_preprocess(trainX)
testX = vgg16_preprocess(testX)
start_time = time.time()
train_features = feature_extract(trainX)
test_features = feature_extract(testX)
train_features = train_features.reshape(train_features.shape[0],train_features.shape[1]*train_features.shape[2]*train_features.shape[3])
test_features = test_features.reshape(test_features.shape[0],test_features.shape[1]*test_features.shape[2]*test_features.shape[3])
print(train_features.shape,test_features.shape)

subclf = LinearSVC(C=1,class_weight='balanced',verbose=1,random_state=22,max_iter=3000)
clf = MultiOutputClassifier(estimator=subclf)
clf.fit(train_features, trainY)
svm_time=time.time() - start_time
accuracy = clf.score(test_features,testY)
print(accuracy)
yhat_classes = clf.predict(test_features)
f1 = f1score(testY,yhat_classes,average='samples')
print("==========================================")
print("0% + SVM Model")
print('> F1-score=%.4f, accuracy=%.3f' % (f1,accuracy))
print("time :", svm_time)
print("==========================================")
final_report["0% + SVM"] = ['NA',round(accuracy*100,3),round(f1,3),'NA',int(svm_time)]
final_Dataframe = np.transpose(pd.DataFrame(final_report))
columns = ['loss','accuracy','f1_score','epoch','training time']
final_Dataframe.columns = columns
display(final_Dataframe)


# In[ ]:




