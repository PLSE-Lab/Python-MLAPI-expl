import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical

train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')

x_train = (train.ix[:,1:].values).astype('float32')
y_train = train.ix[:,0].values.astype('int32')                                                   
x_test = test.values.astype('float32')

x_train = x_train / 255.0
x_test = x_test / 255.0

X_train = x_train.reshape(42000, 28, 28, 1)
X_test = x_test.reshape(28000, 28, 28, 1)

y_train = to_categorical(y_train, 10)

X_train, X_eval, Y_train, Y_eval = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)

try: 
    model = load_model('mnist_model.h5')    
except:
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_data=(X_eval, Y_eval), epochs=10, batch_size=64)
    loss, acc = model.evaluate(X_eval, Y_eval, verbose=0)
    print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(loss, acc))

predicted_classes = model.predict_classes(X_test)

submissions = pd.DataFrame({"ImageId": list(range(1,len(predicted_classes) + 1)), "Label": predicted_classes})
submissions.to_csv("mnist.csv", index=False, header=['ImageId', 'Label'])

model.save('mnist_model.h5')



