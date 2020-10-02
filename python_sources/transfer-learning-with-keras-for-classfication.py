import keras 
from keras.layers import * 
from keras.models import * 
from keras.preprocessing.image import * 
import keras 
from keras.layers import * 
from keras.models import * 
from keras.preprocessing.image import * 
from keras.optimizers import *

import pandas as pd
import numpy as np

train = pd.read_csv("/content/drive/My Drive/DL/plant/train.csv")
test = pd.read_csv("/content/drive/My Drive/DL/plant/test.csv")
sample =  pd.read_csv("/content/drive/My Drive/DL/plant/sample_submission.csv")

test["image_id"]=test["image_id"]+'.jpg'
train["image_id"]=train["image_id"]+'.jpg'



col = ['healthy', 'multiple_diseases', 'rust', 'scab']


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train[:1200],
        directory='/content/drive/My Drive/DL/plant/images',
        x_col="image_id",
        y_col=col,classes=['healthy', 'multiple_diseases', 'rust', 'scab'],
        target_size=(80, 80),
        batch_size=16,
        class_mode='raw')

validation_generator = train_datagen.flow_from_dataframe(
        dataframe=train[1201:1800],
        directory='/content/drive/My Drive/DL/plant/images',
        x_col="image_id",
        y_col=col,
        target_size=(80, 80),
        batch_size=16,classes=['healthy', 'multiple_diseases', 'rust', 'scab'],
        class_mode='raw')

test_generator = test_datagen.flow_from_dataframe(
        dataframe=test[0:800],
        directory='/content/drive/My Drive/DL/plant/images',
        x_col="image_id",
        target_size=(80, 80),
        batch_size=1,
        class_mode=None)




img = 80

model = keras.applications.vgg16.VGG16(weights = 'imagenet',include_top=False,input_shape=(img,img,3))
for layer in model.layers:
  layer.trainable=True

def addtopmodel(bmodel,num_classes,D=256):
  top_model = bmodel.output
  top_model = Flatten(name='faltten')(top_model)
  top_model = Dense(D,activation ='relu')(top_model)
  top_model= Dropout(0.4)(top_model)
  top_model = Dense(num_classes,activation = 'softmax')(top_model)
  return top_model


num_classes = 4
f  = addtopmodel(model,4)
model1 = Model(inputs=model.input,outputs=f)

model1.summary()

adam = keras.optimizers.Adam(lr=0.0002)
rmprop = keras.optimizers.RMSprop(lr=0.002)
adamax = keras.optimizers.Adamax(lr=0.002)

model1.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model1.fit_generator(
        train_generator,
        steps_per_epoch=120,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=50)


Epoch 1/50
120/120 [==============================] - 95s 793ms/step - loss: 0.1962 - acc: 0.9365 - val_loss: 0.9862 - val_acc: 0.8091
Epoch 2/50
120/120 [==============================] - 89s 742ms/step - loss: 0.1749 - acc: 0.9458 - val_loss: 0.7276 - val_acc: 0.8167
Epoch 3/50
120/120 [==============================] - 89s 743ms/step - loss: 0.1749 - acc: 0.9484 - val_loss: 0.7365 - val_acc: 0.8293
Epoch 4/50
120/120 [==============================] - 90s 754ms/step - loss: 0.1108 - acc: 0.9625 - val_loss: 0.7823 - val_acc: 0.8274
Epoch 5/50
120/120 [==============================] - 90s 746ms/step - loss: 0.1837 - acc: 0.9422 - val_loss: 0.6451 - val_acc: 0.8167
Epoch 6/50
120/120 [==============================] - 90s 752ms/step - loss: 0.1392 - acc: 0.9516 - val_loss: 0.7588 - val_acc: 0.8306
Epoch 7/50
120/120 [==============================] - 91s 755ms/step - loss: 0.1314 - acc: 0.9568 - val_loss: 0.7720 - val_acc: 0.8056
Epoch 8/50
120/120 [==============================] - 90s 752ms/step - loss: 0.0792 - acc: 0.9724 - val_loss: 1.2300 - val_acc: 0.8205
Epoch 9/50
120/120 [==============================] - 92s 763ms/step - loss: 0.1175 - acc: 0.9594 - val_loss: 0.7569 - val_acc: 0.8357
Epoch 10/50
120/120 [==============================] - 92s 763ms/step - loss: 0.0819 - acc: 0.9781 - val_loss: 1.0696 - val_acc: 0.7967
Epoch 11/50
120/120 [==============================] - 89s 741ms/step - loss: 0.1198 - acc: 0.9615 - val_loss: 0.7459 - val_acc: 0.8306
Epoch 12/50
120/120 [==============================] - 89s 742ms/step - loss: 0.0639 - acc: 0.9813 - val_loss: 1.0798 - val_acc: 0.8230
Epoch 13/50
120/120 [==============================] - 89s 740ms/step - loss: 0.1350 - acc: 0.9589 - val_loss: 0.8930 - val_acc: 0.8299
Epoch 14/50
120/120 [==============================] - 88s 736ms/step - loss: 0.0722 - acc: 0.9745 - val_loss: 0.7908 - val_acc: 0.8268
Epoch 15/50
120/120 [==============================] - 88s 736ms/step - loss: 0.0650 - acc: 0.9807 - val_loss: 1.4667 - val_acc: 0.7800
Epoch 16/50
120/120 [==============================] - 89s 739ms/step - loss: 0.0735 - acc: 0.9740 - val_loss: 0.8107 - val_acc: 0.8210
Epoch 17/50
120/120 [==============================] - 89s 741ms/step - loss: 0.0456 - acc: 0.9875 - val_loss: 0.8851 - val_acc: 0.8255
Epoch 18/50
120/120 [==============================] - 88s 737ms/step - loss: 0.0593 - acc: 0.9813 - val_loss: 0.9718 - val_acc: 0.8142
Epoch 19/50
120/120 [==============================] - 88s 737ms/step - loss: 0.0470 - acc: 0.9859 - val_loss: 0.8905 - val_acc: 0.8427
Epoch 20/50
120/120 [==============================] - 88s 737ms/step - loss: 0.0649 - acc: 0.9802 - val_loss: 0.9198 - val_acc: 0.8028
Epoch 21/50
120/120 [==============================] - 88s 735ms/step - loss: 0.0387 - acc: 0.9875 - val_loss: 0.7938 - val_acc: 0.8255
Epoch 22/50
120/120 [==============================] - 88s 734ms/step - loss: 0.0758 - acc: 0.9792 - val_loss: 0.9960 - val_acc: 0.7914
Epoch 23/50
120/120 [==============================] - 89s 743ms/step - loss: 0.0588 - acc: 0.9854 - val_loss: 1.0298 - val_acc: 0.8261
Epoch 24/50
120/120 [==============================] - 88s 736ms/step - loss: 0.0352 - acc: 0.9901 - val_loss: 0.9673 - val_acc: 0.8496
Epoch 25/50
120/120 [==============================] - 88s 733ms/step - loss: 0.0371 - acc: 0.9891 - val_loss: 1.0366 - val_acc: 0.8357
Epoch 26/50
120/120 [==============================] - 89s 740ms/step - loss: 0.0984 - acc: 0.9688 - val_loss: 1.0053 - val_acc: 0.8018
Epoch 27/50
120/120 [==============================] - 89s 739ms/step - loss: 0.0629 - acc: 0.9818 - val_loss: 1.2002 - val_acc: 0.7826
Epoch 28/50
120/120 [==============================] - 88s 732ms/step - loss: 0.0785 - acc: 0.9740 - val_loss: 0.7421 - val_acc: 0.8154
Epoch 29/50
120/120 [==============================] - 89s 741ms/step - loss: 0.0402 - acc: 0.9849 - val_loss: 1.2026 - val_acc: 0.8312
Epoch 30/50
120/120 [==============================] - 89s 744ms/step - loss: 0.0282 - acc: 0.9917 - val_loss: 1.0257 - val_acc: 0.8331
Epoch 31/50
120/120 [==============================] - 89s 743ms/step - loss: 0.0525 - acc: 0.9891 - val_loss: 0.7821 - val_acc: 0.8104
Epoch 32/50
120/120 [==============================] - 88s 736ms/step - loss: 0.0356 - acc: 0.9901 - val_loss: 0.8556 - val_acc: 0.8440
Epoch 33/50
120/120 [==============================] - 89s 738ms/step - loss: 0.0771 - acc: 0.9839 - val_loss: 0.8349 - val_acc: 0.8420
Epoch 34/50
120/120 [==============================] - 89s 742ms/step - loss: 0.0564 - acc: 0.9823 - val_loss: 0.9074 - val_acc: 0.8420
Epoch 35/50
120/120 [==============================] - 88s 732ms/step - loss: 0.0471 - acc: 0.9849 - val_loss: 1.2832 - val_acc: 0.8389
Epoch 36/50
120/120 [==============================] - 88s 730ms/step - loss: 0.0936 - acc: 0.9729 - val_loss: 0.8077 - val_acc: 0.8344
Epoch 37/50
120/120 [==============================] - 88s 734ms/step - loss: 0.0672 - acc: 0.9802 - val_loss: 0.7289 - val_acc: 0.8357
Epoch 38/50
120/120 [==============================] - 88s 732ms/step - loss: 0.0415 - acc: 0.9870 - val_loss: 1.0803 - val_acc: 0.8402
                
                
                
                
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=800)

pred_bool = (pred >0.5)