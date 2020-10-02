# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from sklearn import preprocessing
from sklearn.decomposition import PCA
import glob
import math
import csv
import os

#print(os.listdir("../input/ema-crossover/ema-crossover/"))

#############################################################################################################################
# Utility functions
#############################################################################################################################
def loadData(path, subset = -1):        
    allFiles = glob.glob(os.path.join(path, "data_*.csv"))
    if(subset > 0):
        allFiles = allFiles[0:subset]
    data = []
    for file in allFiles:
        print(file)
        with open(file, 'r') as f:
            data.append( [float(x[1]) for x in list(csv.reader(f))] )   
    return np.array(data)

def centerAroundEntry(data):
    print(data.shape)
    # extract the price at 20 min after entry
    labels = data[:,-1]
    # remove the last 20 min of history from our data..
    data = data[:,0:-20]
    # normalise data to the ENTRY point
    for i in range(data.shape[0]):
        labels[i] = (labels[i]/data[i,-1]) - 1.0
        data[i,] = (data[i,]/data[i,-1]) - 1.0
    return (data, labels)
    
def scale(data):
    return preprocessing.scale(data)

def toClasses(labels, num_classes):
    sorted = np.sort(np.array(labels, copy=True))
    bsize = math.floor( len(sorted) / num_classes )
    buckets = []
    for i in range(num_classes):        
        buckets.append(sorted[i*bsize])
    print("buckets: " + str(buckets))
    targets = np.digitize(labels, buckets) - 1
    one_hot_targets = np.eye(num_classes)[targets]
    print(one_hot_targets)
    return one_hot_targets
    
def printLabelDistribution(x):
    unq_rows, count = np.unique(x, axis=0, return_counts=1)
    out = {tuple(i):j for i,j in zip(unq_rows,count)}
    print(out)
    return out

def cacheLoadData(path, num_classes, input_size ):
    cache = "./daytrader_"+str(input_size)+".npy"
    labelsCache = "./daytrader_labels_"+str(input_size)+".npy"
    if( not os.path.isfile(cache) ):
        data = loadData(path)

        (data, labels) = centerAroundEntry(data)
        print(data.shape)
        data_scaled = scale(data)
        labels_classed = toClasses(labels, num_classes)

        printLabelDistribution(labels_classed)

        pca = PCA(n_components=input_size, svd_solver='full')
        data_reduced = pca.fit_transform(data_scaled) 
        np.save(cache, data_reduced)
        np.save(labelsCache, labels_classed)
        
    data = np.load(cache)
    labels_classed = np.load(labelsCache)
    return (data, labels_classed)


#############################################################################################################################
# Model
#############################################################################################################################
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint

# fix random seed for reproducibility
np.random.seed(90210)

num_classes = 5
batch_size = 256
epochs = 250

input_size = 512

savePath = r'./'
path =r'../input/ema-crossover/ema-crossover' # path to data
(data, labels_classed) = cacheLoadData(path, num_classes, input_size)
x_train, x_test, y_train, y_test = train_test_split(data, labels_classed, test_size=0.1)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=input_size))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])

# checkpoint
modelPath= savePath+"mlp-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[checkpoint],
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])    
