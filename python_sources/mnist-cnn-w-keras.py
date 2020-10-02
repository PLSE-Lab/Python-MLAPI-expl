from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Read data
train = pd.read_csv('../input/train.csv')
labels = train.iloc[:,0].values.astype('int32')
X_train = (train.iloc[:,1:].values).astype('float32')
X_test = (pd.read_csv('../input/test.csv').values).astype('float32')

# convert list of labels to binary class matrix, 1 = [0,1,0,0,0,0,0,0,0,0]
y_train = np_utils.to_categorical(labels) 

# pre-processing: divide by max and substract mean, shape: (42000, 784)
X_train /= 255
X_test /= 255

mean = np.std(X_train)
print(mean)
X_train -= mean
X_test -= mean

# Here's a CNN
model = Sequential() #sequential model
#model.add(Dense(512, activation='sigmoid', input_shape=(784,)))
#model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
#model.add(Dense(10, activation='softmax'))
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

model.add(Conv2D(32, (5,5), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, (5,5), activation='relu'))
# model.output_shape = (None,20,20,32)
model.add(MaxPooling2D((2,2)))
# model.output_shape = (None,10,10,32)
model.add(Conv2D(32, (3,3), activation='relu'))
# model.output_shape = (None, 8, 8, 32)
model.add(Flatten())
# model.output_shape = (None, 2048)
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=10))
model.add(Activation('softmax'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training...")
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2, verbose=2)

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras-cnn.csv")
print("done. :)")
