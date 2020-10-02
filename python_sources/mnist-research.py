# changed from keras standard examples
from keras import *
from keras.layers import Dense, Dropout

(Xtr,Ytr),(Xte,Yte) = datasets.mnist.load_data()
Xtr = Xtr.reshape(60000,784).astype('float32')
Xtr /= 255.0
Xte = Xte.reshape(10000,784).astype('float32')
Xte /= 255.0
Ytr = utils.to_categorical(Ytr,10)
Yte = utils.to_categorical(Yte,10)

model = models.Sequential()
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dropout(0.25))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['acc'])
model.fit(Xtr,Ytr,
          batch_size=128,
          epochs=12,
          verbose=2,
          validation_data=(Xte,Yte))

score = model.evaluate(Xte,Yte,verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
