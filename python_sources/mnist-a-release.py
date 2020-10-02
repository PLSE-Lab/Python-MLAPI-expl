# changed from keras standard examples
from keras import *
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

(Xtr,Ytr),(Xte,Yte) = datasets.mnist.load_data()
Xtr = Xtr.reshape(60000,28,28,1).astype('float32')
Xtr /= 255.0
Xte = Xte.reshape(10000,28,28,1).astype('float32')
Xte /= 255.0
Ytr = utils.to_categorical(Ytr,10)
Yte = utils.to_categorical(Yte,10)

model = models.Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.1,decay=1e-6),
              metrics=['acc'])
model.fit(Xtr,Ytr,
          batch_size=128,
          epochs=45,
          verbose=2,
          validation_data=(Xte,Yte))

score = model.evaluate(Xte,Yte,verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
