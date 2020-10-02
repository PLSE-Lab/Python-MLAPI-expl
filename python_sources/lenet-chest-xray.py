#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/train',
                                                    target_size=(32, 32),
                                                    batch_size=8,
                                                    class_mode='binary')


trainlabels=training_set.classes


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/test',
                                            target_size=(32, 32),
                                            batch_size=8,
                                            class_mode='binary')

testlabels=test_set.classes


# In[ ]:


testpred = ImageDataGenerator(rescale=1./255)
test = testpred.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/val',
                                            target_size=(32, 32),
                                            batch_size=8,
                                            class_mode='binary')

predlabels=test.classes


# In[ ]:


import numpy as np
np.size(testlabels)


# In[ ]:


from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Sequential


# In[ ]:


classifier = Sequential()
classifier.add(Convolution2D(6,(5,5), input_shape=(32,32,3),strides=1, padding='valid', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
classifier.add(Convolution2D(16,(5,5),strides=1, padding='valid',  activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
classifier.add(Convolution2D(120,(5,5),strides=1, padding='valid',  activation='relu'))
classifier.summary()


# In[ ]:


classifier.add(Flatten())
classifier.add(Dense(84, input_shape=(120,)))
#classifier.add(Dense(output_dim=84,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.summary()


# In[ ]:


lenetmodel=classifier.fit(training_set,
                            steps_per_epoch=1000,
                            epochs=10,
                            validation_data=test_set,
                            validation_steps=100,use_multiprocessing=True)


# In[ ]:


predictions=classifier.predict(test)
predictions
for i in range(len(predictions)):
    if predictions[i]>0.50:
        predictions[i]=1
    else:
        predictions[i]=0


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(predlabels, predictions)

