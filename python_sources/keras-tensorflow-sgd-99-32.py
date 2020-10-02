from keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import np_utils
import os
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras import backend as K
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


def plot_loss_accuracy(history):
    plt.figure()
    epochs = range(len(history.epoch))
    plt.plot(epochs, history.history['acc'], 'r', linewidth=3.0)
    plt.plot(epochs,history.history['val_acc'], 'b', linewidth=3.0)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    
    plt.figure()
    plt.plot(epochs, history.history['loss'], 'r', linewidth=3.0)
    plt.plot(epochs,history.history['val_loss'], 'b', linewidth=3.0)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.legend(['Training Loss', 'Validation Loss'],fontsize=18)
    
    plt.show()

np.random.seed(100)

digit_train =  pd.read_csv("../input/train.csv")
digit_train.shape
#digit_train.info()

X_train = digit_train.iloc[:,1:].values.astype('float32')/255.0
##As we need images to be feeded to the CNN model , we are converting features to images
X_train_images=X_train.reshape(X_train.shape[0],28,28,1)

y_train = np_utils.to_categorical(digit_train["label"])

random_seed = 2
X_train_images, X_val, y_train, Y_val = train_test_split(X_train_images, y_train, test_size = 0.1, random_state=random_seed)

img_width, img_height = 28, 28

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)
    
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape,padding = 'Same'))
model.add(Conv2D(32, (3, 3), activation='relu',padding = 'Same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10,  activation='softmax'))
print(model.summary())

model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train_images)

epochs = 1 ## change this to 20
batchsize = 20
X_train_images.shape
# Fit the model
history = model.fit_generator(datagen.flow(X_train_images,y_train, batch_size=batchsize),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train_images.shape[0] // batchsize)

#history = model.fit(x=X_train_images, y=y_train, verbose=1, epochs=epochs, batch_size=batchsize, validation_split=0.2)
#print(model.get_weights())

historydf = pd.DataFrame(history.history, index=history.epoch)
plot_loss_accuracy(history)

digit_test = pd.read_csv("../input/test.csv")
digit_test.shape
digit_test.info()

X_test = digit_test.values.astype('float32')/255.0
X_test_images=X_test.reshape(X_test.shape[0],28,28,1)

pred = model.predict_classes(X_test_images)
submissions=pd.DataFrame({"ImageId": list(range(1,len(pred)+1)),
                         "Label": pred})
submissions.to_csv("submission.csv", index=False, header=True)

    
