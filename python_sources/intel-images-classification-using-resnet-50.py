#!/usr/bin/env python
# coding: utf-8

# In[ ]:



get_ipython().run_line_magic('reset', '-f')


# In[ ]:


## 1. Call libraries
import numpy as np

# 1.1 Classes for creating models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense

# 1.2 Class for accessing pre-built models
from tensorflow.keras import applications

# 1.3 Class for generating infinite images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1.4 Miscelleneous
import matplotlib.pyplot as plt
import time, os


# In[ ]:


train_data_dir = '..//input//intel-image-classification//seg_train//seg_train'


# In[ ]:


validation_data_dir = '..//input//intel-image-classification//seg_test//seg_test'


# In[ ]:


img_width, img_height = 75,75


# In[ ]:


nb_train_samples, nb_validation_samples = 14000, 3000


# In[ ]:


batch_size = 50


# In[ ]:


bf_filename = '..//working//bottleneck_features_train.npy'


# In[ ]:


val_filename = '..//working//bottleneck_features_validation.npy'


# In[ ]:


datagen_train = ImageDataGenerator(rescale=1. / 255)


# In[ ]:


generator_tr = datagen_train.flow_from_directory(
              directory = train_data_dir,     
              target_size=(img_width, img_height),    
              batch_size=batch_size,                  
              class_mode=None,                        
              shuffle=False                           
                                                    
              )


# In[ ]:


datagen_val = ImageDataGenerator(rescale=1. / 255)


# In[ ]:


generator_val = datagen_val.flow_from_directory(
                                          validation_data_dir,
                                          target_size=(img_width, img_height),
                                          batch_size=batch_size,
                                          class_mode=None,
                                          shuffle=False   
                                                   
                                          )


# In[ ]:


#Buld ResNet50 network model with 'imagenet' weights

model = applications.ResNet50(
	                       include_top=False,
	                       weights='imagenet',
	                       input_shape=(img_width, img_height,3)
	                       )


# In[ ]:


model.summary()


# In[ ]:


start = time.time()


bottleneck_features_train = model.predict_generator(
                                                    generator = generator_tr,
                                                    steps = nb_train_samples // batch_size,
                                                    verbose = 1
                                                    )
end = time.time()


# In[ ]:


print("Time taken: ",(end - start)/60, "minutes")


# In[ ]:


start = time.time()
bottleneck_features_validation = model.predict_generator(
                                                         generator = generator_val,
                                                         steps = nb_validation_samples // batch_size,
                                                         verbose = 1
                                                         )

end = time.time()


# In[ ]:


print("Time taken: ",(end - start)/60, "minutes")


# In[ ]:


if os.path.exists(bf_filename):
    os.system('rm ' + bf_filename)


# In[ ]:



np.save(open(bf_filename, 'wb'), bottleneck_features_train)


# In[ ]:


if os.path.exists(val_filename):
    os.system('rm ' + val_filename)


# In[ ]:


np.save(open(val_filename, 'wb'), bottleneck_features_validation)


# In[ ]:


get_ipython().run_line_magic('reset', '-f')


# In[ ]:


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Softmax
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras import applications
import matplotlib.pyplot as plt
import time, os


# In[ ]:


img_width, img_height = 75,75  


# In[ ]:


nb_train_samples = 14000


# In[ ]:


nb_validation_samples = 3000


# In[ ]:


epochs = 50


# In[ ]:


batch_size = 64


# In[ ]:


num_classes = 6


# In[ ]:


bf_filename = '..//working//bottleneck_features_train.npy'


# In[ ]:


val_filename = '..//working//bottleneck_features_validation.npy'


# In[ ]:


top_model_weights_path = '..//working///bottleneck_fc_model.h5'


# In[ ]:


train_data_features = np.load(open(bf_filename,'rb'))


# In[ ]:


train_data_features.shape 


# In[ ]:


train_labels = np.array([1] * 2300 + [2] * 2300 + [3] * 2300 + [4] * 2300 + [5] * 2300 + [6] * 2300)   


# In[ ]:


train_labels


# In[ ]:


train_labels.shape 


# In[ ]:


x = np.arange(13800)      


# In[ ]:


np.random.shuffle(x)     


# In[ ]:


x


# In[ ]:


train_data_features = train_data_features[x, :,:,:]


# In[ ]:


train_labels = train_labels[x]


# In[ ]:


train_labels.shape


# In[ ]:


train_labels_categ = to_categorical(train_labels , )


# In[ ]:


train_labels_categ.shape


# In[ ]:


train_labels_categ         


# In[ ]:


train_labels_categ = train_labels_categ[:, 1:]


# In[ ]:


train_labels_categ


# In[ ]:


validation_data_features = np.load(open(val_filename,'rb')) 


# In[ ]:


validation_data_features.shape
  


# In[ ]:


validation_labels = np.array([1] * 500 + [2] * 500 + [3] * 500 + [4] * 500 + [5] * 500 + [6] * 500)


# In[ ]:


validation_labels = to_categorical(validation_labels)


# In[ ]:


validation_labels = validation_labels[:,1:]


# In[ ]:


validation_labels.shape


# In[ ]:


validation_labels.shape[1:]


# In[ ]:


model = Sequential()


# In[ ]:


model.add(Flatten(input_shape=train_data_features.shape[1:]))     


# In[ ]:


model.add(Dense(256, activation='relu'))


# In[ ]:


model.add(Dropout(0.5))


# In[ ]:


model.add(Dense(num_classes, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(
              optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy']
              )


# In[ ]:


# Fit model and make predictions on validation dataset

start = time.time()
history = model.fit(train_data_features, train_labels_categ,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(validation_data_features, validation_labels),
                    verbose =1
                   )
end = time.time()


# In[ ]:


print("Time taken: ",(end - start)/60, "minutes")


# In[ ]:


history.history.keys()


# In[ ]:


len(history.history['accuracy'])


# In[ ]:


len(history.history['val_accuracy'])


# In[ ]:



def plot_learning_curve():
    val_acc = history.history['val_accuracy']
    tr_acc=history.history['accuracy']
    epochs = range(1, len(val_acc) +1)
    plt.plot(epochs,val_acc, 'b', label = "Validation accu")
    plt.plot(epochs, tr_acc, 'r', label = "Training accu")
    plt.title("Training and validation accuracy")
    plt.xlabel("epochs-->")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


# In[ ]:


plot_learning_curve()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




