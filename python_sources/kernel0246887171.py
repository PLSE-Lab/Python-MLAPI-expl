import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

x_train=(train.iloc[:,1:]).values.astype('float32')
y_train = train.iloc[:,0].values.astype('int32')
x_test=test.values.astype('float32')

x_train = x_train/255.0
x_test = x_test/255.0
X_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)


model=tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
model.fit(X_train,y_train,epochs=1)
model.save('model.model')

predicted_classes = model.predict_classes(x_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),
                         "Label": predicted_classes})
submissions.to_csv("result.csv", index=False, header=True)
