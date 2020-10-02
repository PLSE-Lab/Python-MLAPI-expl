import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Flatten,Conv2D, MaxPooling2D



train_ds = pd.read_csv("../input/train.csv")
test_ds = pd.read_csv("../input/test.csv")

y_train = pd.get_dummies(train_ds['label']).as_matrix()
x_train = train_ds
del x_train['label']

x_train = x_train.as_matrix().astype('float32')
x_test = test_ds.as_matrix().astype('float32')
x_train /= 255
x_test /= 255
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 #        Hight,Width,Channels
                 input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))   
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))   
model.add(Flatten(name='flatten'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])

print('Trainning...')
model.fit(x_train, y_train, epochs=5, batch_size=128)

print('Saving trained model...')
model.save('trained_model.h5')
print('Saved trained model')

loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)
print('Loss: ', loss_and_metrics)

# Create submision
id = np.arange(1,28001)
pred_classes = model.predict_classes(x_test)

submission = pd.DataFrame({
    "ImageId": id,
    "Label": pred_classes})

print(submission[0:10])

submission.to_csv('./predictions.csv', index=False)








