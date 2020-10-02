#this simple keras model get a score of 98.757%

#import libraries
import numpy as np 
import pandas as pd 
import tensorflow as tf
import numpy as np
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#import data
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
test_data=test_data.as_matrix().reshape(28000,28,28)
x_train=train_data.as_matrix()[0:,1:].reshape(42000,28,28)
y_train=pd.get_dummies(train_data["label"])
x_train,test_data=np.expand_dims(x_train,axis=-1),np.expand_dims(test_data,axis=-1)

#creat model
batch_size = 128
num_classes = 10
epochs = 12
input_shape=(28,28,1)
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

#compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#train model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,)

#predict
predictions=model.predict(test_data)
predictions=predictions.tolist()
p=[]
for i in predictions:
  p.append(i.index(max(i)))

#creat result data frame
col_names =  ['ImageId', 'Label']
my_df  = pd.DataFrame(columns = col_names)
for i in range(1,28001):
  my_df.loc[i-1] = [i, p[i-1]]

#export result
my_df.to_csv("/kaggle/working/mnist.csv", encoding='utf-8', index=False)