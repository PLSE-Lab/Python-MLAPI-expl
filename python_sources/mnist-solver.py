import os
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
trainX = np.array(pd.read_csv(os.path.join("../input","train.csv")))
trainY = trainX[:,0]
trainX = (trainX[:,1:785]).reshape(42000,28,28,1)/255.0
test = np.array(pd.read_csv(os.path.join("../input","test.csv")))
test = test.reshape(28000,28,28,1)/255.0

# for model checking...
dummyTestX = trainX[41000:42000,:,:,:]
dummyTestY = trainY[41000:42000]
dummyTestY = keras.utils.to_categorical(dummyTestY, 10)

trainY = trainY[0:41000]
trainY = keras.utils.to_categorical(trainY, 10)
trainX = trainX[0:41000, : , :, :]

#model Definition
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(28,28,1)))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(filters = 20, kernel_size = 5,  padding='same', data_format='channels_last',  activation='relu')) 
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Dropout(0.3))


model.add(keras.layers.Conv2D(filters = 30, kernel_size = 3,  padding='same',  activation='relu')) 
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(), loss= tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
print(model.summary())
train_x, val_x, train_y, val_y = train_test_split(trainX, trainY, test_size=0.10, random_state=1001)

model.fit(train_x, train_y, epochs=15, batch_size=64, validation_data = (val_x, val_y), verbose=0)

score  = model.evaluate(dummyTestX, dummyTestY, verbose=0)

print('Loss of model on test set = %0.4f' % score[0])
print('Accuracy of model on test set = %0.4f' % score[1])

prediction = model.predict(test)
prediction = np.argmax(prediction, axis=1)
submission = pd.DataFrame(list(range(1,28001)), columns = ['ImageId'])
submission['Label'] = prediction
submission = submission.astype(int)
submission.to_csv("submission.csv",index=None, header=True)