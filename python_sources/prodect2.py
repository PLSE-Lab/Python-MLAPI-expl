#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.applications.vgg16 import VGG16


# In[ ]:





# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255., validation_split=0.2)

BATCH_SIZE = 16
train_generator = train_datagen.flow_from_directory(
    directory=r"../input/shopee-round-2-product-detection-challenge/train/train/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    directory=r"../input/shopee-round-2-product-detection-challenge/train/train/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    subset='validation'
)


# In[ ]:


# transfer learning
import keras
from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from keras.applications.vgg16 import VGG16
vggmodel = VGG16(weights='imagenet', include_top=True)

for layers in (vggmodel.layers)[:19]:
#     print(layers)
    layers.trainable = False
X= vggmodel.layers[-2].output
predictions = Dense(42, activation="softmax")(X)
model_final = Model(inputs= vggmodel.input, outputs = predictions)

model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
# model_final.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics=["accuracy"])

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# model_final.save_weights("vgg16_1.h5")


# In[ ]:


checkpoint1 = ModelCheckpoint('batch1_vgg16(15).h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
# early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')
# model_final.fit_generator(generator=train_generator, steps_per_epoch= 2, epochs= 100, validation_data=validation_generator, validation_steps=1, callbacks=[checkpoint,early])

model_final.load_weights('../input/prodect2/batch1_vgg16%2814%29.h5')

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
model_final.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID,
                    callbacks=[checkpoint1],
                    epochs=1)


# In[ ]:





# In[ ]:


test_datagen = ImageDataGenerator(rescale=1/255.)

test_generator = test_datagen.flow_from_directory(
    directory=r"../input/shopee-round-2-product-detection-challenge/test/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)


# In[ ]:


# model_final.load_weights('../input/prodect2/vgg16_4.h5')
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model_final.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
import numpy as np
import pandas as pd
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames
filenames2 = [file[5:] for file in filenames]
results=pd.DataFrame({"filename":filenames2,
                      "category":predictions})
results.to_csv("batch8_vgg16_epoch11.csv",index=False)


# In[ ]:


import pandas as pd
x = pd.DataFrame()
x.to_csv('ff.csv')


# In[ ]:


print('heh')jj


# In[ ]:


print('ds')

