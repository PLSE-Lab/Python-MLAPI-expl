import pandas as pd
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
# from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

batch_size = 128
num_classes = 10
epochs = 45
num_models = 19

train = pd.read_csv('../input/train.csv').values  # import full train set values

print("Train Shape: ", train.shape)  # (42000, 785) 42000 Bilder: je 1 Label, 784 Pixel

# Map labels to 10 outputs (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
# train[:, 0] -> Slice, take only first col (0) of each row (:)
trainY = np_utils.to_categorical(train[:, 0].astype('int32'), num_classes)

# Split features
# train[:, 1:] -> Slice, take each row (:), and all values from 1 to end (1:)
trainX = train[:, 1:].astype('float32')
trainX = trainX / 255  # Normalize Data (0-255 tp 0-1)

print("TrainY Shape: ", trainY.shape)  # (42000, 10) 42000 Bilder: je 1 Label mapped auf 10
print("TrainX Shape: ", trainX.shape)  # (42000, 784) 42000 Bilder: je 784 Pixel

rows = 28  # picture size rows
cols = 28  # picture size cols

trainX = trainX.reshape(trainX.shape[0], rows, cols, 1)
print("TrainX Shape: ", trainX.shape)  # (42000, 28, 28, 1) 1 wegen Graustufenbild

# CREATE MORE IMAGES VIA DATA AUGMENTATION
datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.10, 
        shear_range= 0.05,
        width_shift_range=0.1, 
        height_shift_range=0.1)

# input_shape (tuple of integers, does not include the batch axis)
input_shape = (rows, cols, 1)

model = [0]*num_models

for i in range(0,num_models):
    model[i] = Sequential()
    model[i].add(Conv2D(32,
                     data_format='channels_last',
                     kernel_size=(4, 4),
                     activation='relu',
                     input_shape=input_shape))
    model[i].add(Conv2D(128, (4, 4), activation='relu'))
    model[i].add(MaxPooling2D(pool_size=(2, 2)))
    model[i].add(Dropout(0.20))
    model[i].add(Flatten())
    model[i].add(Dense(256, activation='relu'))
    model[i].add(Dropout(0.20))
    model[i].add(Dense(num_classes, activation='softmax'))
    model[i].compile(loss=keras.losses.categorical_crossentropy,
                  optimizer="Adam",
                  metrics=['accuracy'])
    
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
    
    trainX2, valX2, trainY2, valY2 = train_test_split(trainX, trainY, test_size = 0.15)
    model[i].fit_generator(datagen.flow(trainX2,trainY2, batch_size=64),
                         epochs = epochs, steps_per_epoch = trainX2.shape[0]//64,  
                         validation_data = (valX2,valY2), 
                         callbacks=[annealer], 
                         verbose=2)
    
    score = model[i].evaluate(trainX, trainY, verbose=2)
    print('Train accuracy Model', i , ':', score[1])

testX = pd.read_csv('../input/test.csv').values.astype('float32')
testX = testX / 255
testX = testX.reshape(testX.shape[0], rows, cols, 1)

predictions = [0]*(num_models+1)


for i in range(0,num_models):
    predictions[i] = model[i].predict_classes(testX, verbose=2)

def most_common(lst):
    return max(set(lst), key=lst.count)

predictions[num_models] = [0]*len(predictions[0])

for i in range(0,len(predictions[0])):
    single_pred = []
    for j in range(0,num_models):
        single_pred.append(predictions[j][i])
    predictions[num_models][i] = most_common(single_pred)

predictions[num_models]

pd.DataFrame({"ImageId": list(range(1, len(predictions[num_models]) + 1)),
              "Label": predictions[num_models]}
             ).to_csv('submission_45_19.csv', index=False, header=True)
