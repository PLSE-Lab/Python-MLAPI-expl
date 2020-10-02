# jjjdk Jwalant Bhatt


# imports
import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# directory path depth image gesture data
data_dir = "../input/depthgestrecog/depthGestRecog/"   # path to depth dataset


# dict that saves class num of corresponding label
label_dict = {}


# get the class num
for label in os.listdir(data_dir):
   for sub_dir in os.listdir(os.path.join(data_dir,label)):
       class_num = int((os.listdir(os.path.join(data_dir,label,sub_dir)))[:][0][5:7])
   label_dict[class_num] = label
   
   
# load data and labels from files
x = []
y = []
for root, _, files in os.walk(data_dir):
    for i in range(len(files)):
        x.append((plt.imread((os.path.join(root,files[i])))))
        y.append((int(files[i][5:7])))

x = np.asarray(x, dtype=np.float32)
y = np.asarray(y)


# convert to one hot encoding
y_ohev = []

for j in y:
    ohev = np.zeros(11)  # 11 classes ( replace by no of classes)
    ohev[j - 1] = 1
    y_ohev.append(ohev)


# split into training and validation
x_train, x_test, y_train, y_test = train_test_split(x, y_ohev, test_size=0.2)


# display few entries
fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 3

for i in range(1, columns * rows + 1):
    gest = ""
    ind = np.random.randint(len(x_train))
    fig.add_subplot(rows, columns, i)
    for key, value in label_dict.items():
        if value == (np.argmax(y_train[ind]) + 1):
            gest = key
    plt.title(gest)
    plt.imshow(x_train[ind])

plt.show()


# pre-processing
x_train = np.array(x_train)
x_train[x_train > 0] = 1
x_test = np.array(x_test)
x_test[x_test > 0] = 1

y_test = np.array(y_test)
y_train = np.array(y_train)


# initialize
batch_size = 128
num_classes = 11
epochs = 2
img_rows = 100
img_cols = 100
# input shape (100, 100, 1)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
    
# keras (mnist) model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))


# plot for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
# plot for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
