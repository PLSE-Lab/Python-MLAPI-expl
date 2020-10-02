import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')

#checking for null values
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

X_train = train.loc[:, train.columns != 'label']
y_train = train["label"]

#normalization
X_train = (X_train/255)-0.5
test = (test/255)-0.5

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Label Encoding 
from keras.utils.np_utils import to_categorical 
y_train = to_categorical(y_train, num_classes = 10)

#building the model
model = Sequential()
#layer 1
model.add(Conv2D(64, kernel_size = 3, input_shape = (28, 28, 1,), activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
#layer 2
model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Flatten())
#Fitting in ann
model.add(Dense(256, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing import image
gen = image.ImageDataGenerator()
batches = gen.flow(X_train, y_train, batch_size=64)
history=model.fit_generator(generator=batches, steps_per_epoch=1000, epochs=3)

predictions = model.predict_classes(test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("sub.csv", index=False, header=True)
































