#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Removed every class that doesn't have a disease associated with it
# 50 epochs
# 100 pixels images
# Doubled the nodes in each layer
# Removed the validation split because I don't need it anyway (No live-checking)
# Image segmentation with a lot of zooming, rotating and shifting
# Removed the class weights
# Softmax
from keras.applications.mobilenet import MobileNet, preprocess_input
EPOCHS = 25
VERSION = '_v48'


# In[ ]:


import os
import cv2
import numpy as np
import sys # So I can remove the printing limit for the arrays
import gc
import pandas as pd
from blist import blist # A list library that is more efficient in terms of memory than a list and is faster than an np.array
import matplotlib.pyplot as plt 
import keras
from tqdm  import tqdm, tqdm_notebook # A library which prints a progress bar for loops

from keras import regularizers
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation,AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Adam,RMSprop
from keras.models import Sequential, load_model
from keras.callbacks import History
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


from collections import Counter 

from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle 

from sklearn.model_selection import train_test_split


PATH = '../input/new-dataset/plants/Plants/Train'
PATH_TEST = '../input/new-dataset/plants/Plants/Test'
PATH_IRL = '../input/real-life/rl'
NUM_CLASSES=len(os.listdir(PATH))

CATEGORIES = []
COLOR = 'RGB'
if COLOR == 'GRAYSCALE':
    CHANNELS = 1
else:
    CHANNELS = 3
model_name = 'plants'+VERSION+'.model'

IMG_SIZE=224


# In[ ]:


# os.listdir('../input/real-life/rl')


# In[ ]:


def init_categories():
    path = os.listdir(PATH_TEST)
    path.sort()
    temp_cat = []
    for x in path:
            temp_cat.append(x)
    
    temp_cat.sort()
    
    return temp_cat


# In[ ]:


def create_model(_path,_epochs,_batch,retrain):
    

    
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=0.6,
                                       zoom_range = 0.7,
                                       fill_mode = 'constant',
                                       horizontal_flip = True,                                       
                                       width_shift_range = 0.35,
                                       height_shift_range = 0.35)
    train_generator = train_datagen.flow_from_directory(_path,
                                                        target_size=(IMG_SIZE, IMG_SIZE),
                                                        batch_size=_batch,
                                                        class_mode='categorical')
    
#   Reading the CSV's and initializing the list CATEGORIES
    
    counter = Counter(train_generator.classes)
    max_val = float(max(counter.values()))
    class_weights = {class_id: max_val/num_images for class_id,num_images in counter.items()}
#     print(counter)
#     print(max_val)
    np.save('categories'+str(VERSION)+'.npy', train_generator.class_indices)
    if retrain:
        model = load_model(model_name)
        os.remove(model_name)
    else:
        
#         model = Sequential()  

#         model.add(Conv2D(64, (4,4), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)))
#         model.add(Activation('relu'))
#         model.add(Conv2D(64, (4,4), padding='same'))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D((2,2)))
#         model.add(Dropout(0.3))

#         model.add(Conv2D(128, (3,3)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D((2,2)))
#         model.add(Dropout(0.3))

#         model.add(Conv2D(192, (3,3)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D((2,2)))
#         model.add(Dropout(0.4))

#         model.add(Conv2D(256, (2,2)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D((2,2)))
#         model.add(Dropout(0.5))

#         model.add(Flatten())
#         model.add(Dense(512, activation='relu'))
#         model.add(Dropout(0.5))
#         # Output Layer
#         model.add(Dense(NUM_CLASSES))
#         model.add(Activation('softmax'))

        model = MobileNet(weights='imagenet',include_top = False)
        x = model.output #Take the last layer
        x = GlobalAveragePooling2D()(x) #Add a GlobalAvgPooling        
        x = Dense(1024, activation='relu')(x)
        
        out = Dense(NUM_CLASSES, activation='softmax')(x)
        
        model = Model(inputs=model.input, outputs=out)        
        
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        

    history = model.fit_generator(train_generator,
                                  steps_per_epoch = train_generator.n // _batch,
                                  epochs = _epochs,
                                  verbose=1)
    
    model.save(model_name)
    
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['acc'])
    plt.title('Model history')
    plt.ylabel('Loss / Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.savefig('accuracy_plot.png')
    
    return model


# In[ ]:


mod = create_model(PATH,EPOCHS,32,False)


# In[ ]:


def load_testing_data(path,color,_shuffle):
    x = blist([])
    y = blist([])
    main_path=os.listdir(path)
    main_path.sort()
    CATEGORIES = init_categories()
    for category in tqdm_notebook(main_path):  
        k=0
        new_path = os.path.join(path,category)
        cat = CATEGORIES.index(category)
        
        images_path = os.listdir(new_path)
        images_path.sort()
        
        for img in images_path:
            try:                                
                if color == 'RGB':
                    image = cv2.imread(os.path.join(new_path,img))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif color == 'GRAYSCALE':
                    image = cv2.imread(os.path.join(new_path,img),cv2.IMREAD_GRAYSCALE)
                else:
                    image = cv2.imread(os.path.join(new_path,img))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = None
                del img
                
                image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))             
                if path == PATH_TEST:
                    image = image/255.0
                x.append(image)
                y.append(cat)
            
            except Exception as e:
                print("Error: ",e)
      
    if _shuffle:
        x,y = shuffle(x,y)
    return x,y

def find_2nd_max(array,x):
    max = x
    snd_max = 0
    index = 0
    for i in range(len(array)):
        if array[i]>snd_max and array[i]<max:
            snd_max = array[i]
            index = i
    return index

def print_confusion_matrix(y_test,y_pred,categ,acc,file):
    conf = confusion_matrix(y_test,y_pred)
    for i in range(len(conf)):
        total = np.sum(conf[i])
        
        max = conf[i][np.argmax(conf[i])]
        snd_index = find_2nd_max(conf[i],max)
        snd_max = conf[i][snd_index]
        
        max_proc = float("{0:.3f}".format((max/total)*100))
        snd_max_proc = float("{0:.3f}".format((snd_max/total)*100))
        s = ""
        s+="{0} ({1}): {2} ({3}%) ~~ {4}: {5} ({6}%)\n".              format(categ[i],total,max,max_proc,categ[snd_index],snd_max,snd_max_proc)
        file.write(s)
        print(s)


# In[ ]:





# In[ ]:





# In[ ]:


# Creating a dictionary
def predict():
    res = mod.predict(x_test)
    CATEGORIES=init_categories()
    no_correct = 0
    no_wrong = 0
    y_pred = []
    for i in range(len(res)):
        y_pred.append(np.argmax(res[i]))
        if y_test[i] == np.argmax(res[i]):    
            no_correct+=1
        else:
            no_wrong+=1
    acc = float("{0:.3f}".format((no_correct/(no_correct+no_wrong))*100))
    print("Correct: {} from {} ({}%)".format(no_correct,no_correct+no_wrong,acc))

    file = open((VERSION+'accuracy.txt'),'w')
    print_confusion_matrix(y_test, y_pred,CATEGORIES,acc,file)
    file.close()


# In[ ]:


# Testing on the dataset
CATEGORIES=[]
x_test,y_test  = load_testing_data(PATH_TEST,COLOR,True)
x_test = np.asarray(x_test)
x_test = np.reshape(x_test,(-1,IMG_SIZE,IMG_SIZE,CHANNELS))
predict()


# In[ ]:


# Testing on the irl set
CATEGORIES=[]
CATEGORIES = init_categories()

x_test,y_test  = load_testing_data(PATH_IRL,COLOR,True)
x_test = np.asarray(x_test)
x_test = np.reshape(x_test,(-1,IMG_SIZE,IMG_SIZE,CHANNELS))
res = mod.predict(x_test)

l = [0,0]
for i in range(len(res)):
    if y_test[i] == 13: #Grapes
        if np.argmax(res[i]) == 13:
            l[0] = l[0]+1
    elif y_test[i] == 4: #Powdery Mildew (Cherry)
        if np.argmax(res[i]) == 4:
            l[1] = l[1]+1
print(l, " / ", len(res))

