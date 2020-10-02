# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 
import csv
import os
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#"""
# Se define el nombre de la carpeta o directorio a crear
directorio = "/kaggle/working/train"
directorio1 = "/kaggle/working/validation"
directorio2 = "/kaggle/working/train/negative"
directorio3 = "/kaggle/working/train/positive"
directorio4 = "/kaggle/working/validation/negative" 
directorio5 = "/kaggle/working/validation/positive"
try:
    os.mkdir(directorio)
    os.mkdir(directorio1)
    os.mkdir(directorio2)
    os.mkdir(directorio3)
    os.mkdir(directorio4)
    os.mkdir(directorio5)    
except OSError:
    print("La creación del directorio %s falló" % directorio)
    print("La creación el directorio: %s falló" % directorio1)
    print("La creación el directorio: %s falló" % directorio2)
    print("La creación el directorio: %s falló" % directorio3)
    print("La creación el directorio: %s falló" % directorio4)
    print("La creación el directorio: %s falló" % directorio5)    
else:
    print("Se ha creado el directorio: %s " % directorio)
    print("Se ha creado el directorio: %s " % directorio1)
    print("Se ha creado el directorio: %s " % directorio2)
    print("Se ha creado el directorio: %s " % directorio3)
    print("Se ha creado el directorio: %s " % directorio4)
    print("Se ha creado el directorio: %s " % directorio5)
#"""
"""
import os, sys

dirs = os.listdir("/kaggle/working/train/")

for file in dirs:
    print(file)
"""
"""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import pandas as pd
import os
"""
TAM_TN = 7816    #9316 - 1500
TAM_TP = 7816    #9316 - 1500
TAM_VN = 1500
TAM_VP = 1500

Mem_TN = None
Mem_TP = None
Mem_VN = None
Mem_VP = None

START_TN = 0     
START_TP = 0
START_VN = 0
START_VP = 0

with open('/kaggle/input/diabetic-retinopathy-resized/trainLabels.csv') as csvfile:
    reader = csv.DictReader(csvfile)

    #df = pd.DataFrame(reader)
    #df.groupby('level')['image'].nunique().plot(kind='bar')
    #plt.show()
  
    for row in reader:
        #print(row['image'], row['level'])
        
        if row['level'] == '0' and START_TN < TAM_TP:
            Mem_TN = row['image']
            if Mem_TN != Mem_VN:
                print('/kaggle/working/train/negative/'+row['image']+'.jpeg', START_TN)
                imagen = cv2.imread('/kaggle/input/diabetic-retinopathy-resized/resized_train/resized_train/'+row['image']+'.jpeg')
                name = '/kaggle/working/train/negative/'+row['image']+'.jpeg'
                print(type(name))
                cv2.imwrite(name, imagen)
                START_TN = 1 + START_TN
                
            
        if row['level'] == '1' or row['level'] == '2' or row['level'] == '3' or row['level'] == '4':
            if START_TP < TAM_TP:
                Mem_TP = row['image']
                if Mem_TP != Mem_VP:
                    print('/kaggle/working/train/positive/'+row['image']+'.jpeg', START_TP)
                    imagen = cv2.imread('/kaggle/input/diabetic-retinopathy-resized/resized_train/resized_train/'+row['image']+'.jpeg')
                    name = '/kaggle/working/train/positive/'+row['image']+'.jpeg'
                    cv2.imwrite(name, imagen)
                    START_TP = 1 + START_TP            
            
            
            
        if row['level'] == '0' and START_VN < TAM_VN:
            Mem_VN = row['image']
            if Mem_VN != Mem_TN:
                print('/kaggle/working/validation/negative/'+row['image']+'.jpeg', START_VN)
                imagen = cv2.imread('/kaggle/input/diabetic-retinopathy-resized/resized_train/resized_train/'+row['image']+'.jpeg')
                name = '/kaggle/working/validation/negative/'+row['image']+'.jpeg'
                cv2.imwrite(name, imagen)
                START_VN = 1 + START_VN
            
            
            
        if row['level'] == '1' or row['level'] == '2' or row['level'] == '3' or row['level'] == '4':
            if START_VP < TAM_VP:
                Mem_VP = row['image']
                if Mem_VP != Mem_TP:
                    print('/kaggle/working/validation/positive/'+row['image']+'.jpeg', START_VP)
                    imagen = cv2.imread('/kaggle/input/diabetic-retinopathy-resized/resized_train/resized_train/'+row['image']+'.jpeg')
                    name = '/kaggle/working/validation/positive/'+row['image']+'.jpeg'
                    cv2.imwrite(name, imagen)
                    START_VP = 1 + START_VP

"""
import sys
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K
import tensorflow as tf


K.clear_session()



data_train = '/kaggle/working/train'
data_validation = '/kaggle/working/validation'

#PARAMETROS

epocas = 20
altura, longitud = 100,100
batch_size = 32
pasos = 100
pasos_validation = 200
filtrosConv1 = 32
filtrosConv2 = 64
size_filtro1 = (3,3)
size_filtro2 = (2,2)
size_pool = (2,2)
clases = 2
learning_rate = 0.0005

#PRE-PROCESAMIENTO DE IMÁGENES

entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

imagen_train = entrenamiento_datagen.flow_from_directory(
    data_train,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

imagen_validation = validation_datagen.flow_from_directory(
    data_validation,
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

#Crear la red CNN

cnn = Sequential()

cnn.add(Convolution2D(filtrosConv1, size_filtro1, padding='same', input_shape=(altura,longitud,3), activation='relu'))

cnn.add(MaxPooling2D(pool_size=size_pool))

cnn.add(Con1volution2D(filtrosConv2, size_filtro2, padding='same', activation='relu'))

cnn.add(MaxPooling2D(pool_size=size_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5)) #Evitar overfitting, evitar memorizar
cnn.add(Dense(clases, activation='sigmoid'))

cnn.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

cnn.fit(imagen_train,
        steps_per_epoch=pasos,
        epochs=epocas,
        validation_data=imagen_validation,
        validation_steps=pasos_validation)

dir ='/kaggle/working/modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save('/kaggle/working/modelo/modeloDRPrediction1.h5')
cnn.save_weights('/kaggle/working/modelo/pesosDRPrediction1.h5')

