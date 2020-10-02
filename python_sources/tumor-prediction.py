import numpy as np 
import pandas as pd 
import os
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import PIL
from PIL import Image

image_size = [224,224]
data_path = 'Data'


conv = VGG16(input_shape= image_size+[3],weights='imagenet',include_top=False)

conv.output


x = conv.output
x = GlobalAveragePooling2D()(x)


x = Dense(1024,activation='relu')(x)
x = Dense(1024,activation='relu')(x)
x = Dense(512, activation='relu')(x)


pred = Dense(2,activation='softmax')(x)


model = Model(inputs = conv.input,outputs=pred)


model.summary()



for layer in conv.layers:
    layer.trainable = False


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.24)

train_generator=train_datagen.flow_from_directory('../input/braintumordata/brain_tumor_dataset', target_size=(224,224), color_mode='rgb', shuffle=True, subset='training', batch_size=202, class_mode='categorical')
val_generator = train_datagen.flow_from_directory('../input/braintumordata/brain_tumor_dataset', target_size=(224,224), color_mode='rgb', shuffle=True, subset='validation', class_mode='categorical')


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


print(train_generator.n)
print(train_generator.batch_size)
print(253//32)


step_size_train=train_generator.n//train_generator.batch_size
r = model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=25, validation_data=val_generator)

#save model

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

from tensorflow.keras.models import load_model

tf.keras.models.save_model(model,'mri_tumor.h5', overwrite=True,
include_optimizer=True)

model.save('mri_tumor.h5')

#load model
model = load_model('mri_tumor.h5')
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory('../input/braintumordata/brain_tumor_dataset', target_size=(224,224), color_mode='rgb', shuffle=True, batch_size=202, class_mode='categorical')



#plot loss

plt.plot(r.history['loss'], label='training loss')
plt.legend(['Training Loss'])
plt.show()
plt.savefig('LossVal_loss')

# plot accuracy

plt.plot(r.history['accuracy'])
plt.title('Model Accuracy')
plt.legend(['Training Accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

model = load_model('mri_tumor.h5')
img_path = ('../input/tumor-test-data/tumortest/Y21.jpg')
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
img_data = preprocess_input(x)

rs = model.predict(img_data)
print(rs)

rs[0][0]

rs[0][1]

if rs[0][0] == 1:
    prediction = 'This image is NOT tumorous.'
else:
    prediction = 'Warning! This image IS tumorous.'

print(prediction)



