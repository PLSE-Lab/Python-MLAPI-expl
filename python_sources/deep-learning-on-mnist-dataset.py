import struct
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import backend as K

print(K.image_data_format())

# Thanks @tylerneylon (https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40)
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


X_train = read_idx('../input/mnist-pytorch/train-images-idx3-ubyte')
y_train = read_idx('../input/mnist-pytorch/train-labels-idx1-ubyte')
X_test = read_idx('../input/mnist-pytorch/t10k-images-idx3-ubyte')
y_test = read_idx('../input/mnist-pytorch/t10k-labels-idx1-ubyte')

num_pixels = X_train.shape[1] * X_train.shape[2]

# Reshape the samples
rows = 28
cols = 28
pixel_dimension = 1  # 3 in case of RGB & 1 for gray scale images
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Normalize the pixels between 0 and 1 (Feature Scaling)
X_train = X_train/255
X_test = X_test/255

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model = Sequential()
model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
model.add(Dense(100,activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping_monitor = EarlyStopping(patience=3)

model.fit(X_train,y_train,validation_data=(X_test, y_test), epochs=20, verbose=True, callbacks=[early_stopping_monitor])
scores = model.evaluate(X_test,y_test,verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
