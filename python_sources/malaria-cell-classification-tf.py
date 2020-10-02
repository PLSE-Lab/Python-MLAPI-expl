#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread
import os


# In[ ]:


my_data="../input/files1/Malaria Cells"


# In[ ]:


os.listdir(my_data)


# In[ ]:


train_path=my_data+"/training_set"
test_path=my_data+"/testing_set"


# In[ ]:


os.listdir(train_path)


# In[ ]:


os.listdir(test_path)


# In[ ]:


os.listdir(train_path+'/Parasitized')[0]


# In[ ]:


para_cell=train_path+"/Parasitized"+"/C59P20thinF_IMG_20150803_112802_cell_196.png"


# 

# In[ ]:


para_image=imread(para_cell)
plt.imshow(para_image)


# In[ ]:


para_image.shape


# In[ ]:


os.listdir(train_path+'/Uninfected')[0]


# In[ ]:


normal_cell=train_path+"/Uninfected"+"/C130P91ThinF_IMG_20151004_142951_cell_89.png"


# In[ ]:


normal_img=imread(normal_cell)
plt.imshow(normal_img)


# In[ ]:


normal_img.shape


# In[ ]:


d1=[]
d2=[]
for image_filename in os.listdir(test_path+"/Uninfected"):
    img=imread(test_path+"/Uninfected"+"/"+image_filename)
    w,h,colors=img.shape
    d1.append(w)
    d2.append(h)


# In[ ]:


sns.jointplot(d1,d2)


# In[ ]:


np.mean(d1)


# In[ ]:


np.mean(d2)


# In[ ]:


image_shape=(131,131,3)


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


image_gen=ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.10,
                             height_shift_range=0.10,
                             rescale=1/255,
                             shear_range=0.10,
                             zoom_range=0.10,
                             horizontal_flip=True,
                             fill_mode='nearest'
                              )


# In[ ]:


plt.imshow(image_gen.random_transform(para_image))


# In[ ]:


image_gen.flow_from_directory(train_path)


# In[ ]:


image_gen.flow_from_directory(test_path)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dropout,Flatten,Dense,Conv2D,MaxPooling2D


# In[ ]:


model=Sequential()
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


early_stopping=EarlyStopping(monitor='val_loss')


# In[ ]:


train_image_gen=image_gen.flow_from_directory(train_path,target_size=(131,131),color_mode='rgb',batch_size=16,class_mode='binary',shuffle=True)


# In[ ]:


test_image_gen=image_gen.flow_from_directory(test_path,target_size=(131,131),color_mode='rgb',batch_size=16,class_mode='binary',shuffle=True)


# In[ ]:


train_image_gen.class_indices


# In[ ]:


results=model.fit_generator(train_image_gen,epochs=20,validation_data=test_image_gen,callbacks=[early_stopping])


# In[ ]:


losses=pd.DataFrame(model.history.history)


# In[ ]:


losses[['loss','val_loss']].plot()


# In[ ]:


model.evaluate_generator(test_image_gen)


# In[ ]:


from tensorflow.keras.preprocessing import image


# In[ ]:


pred_probabilities = model.predict_generator(test_image_gen)


# In[ ]:


test_image_gen.classes


# In[ ]:


predictions = pred_probabilities > 0.5


# In[ ]:


predictions


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(test_image_gen.classes,predictions))


# In[ ]:


len(os.listdir(train_path+'/Parasitized'))


# In[ ]:


len(os.listdir(train_path+'/Uninfected'))


# In[ ]:


my_image = image.load_img(normal_cell,target_size=image_shape)


# In[ ]:


my_image = image.img_to_array(my_image)


# In[ ]:


my_image = np.expand_dims(my_image, axis=0)


# In[ ]:


model.predict(my_image)


# In[ ]:


normal_img=imread(normal_cell)
plt.imshow(normal_img)


# In[ ]:


my_image = image.load_img(para_cell,target_size=image_shape)


# In[ ]:


my_image = image.img_to_array(my_image)


# In[ ]:


my_image = np.expand_dims(my_image, axis=0)


# In[ ]:


model.predict(my_image)


# In[ ]:


para_img=imread(para_cell)
plt.imshow(para_img)


# In[ ]:


new_img="../input/malaria-test/31-researchersm.jpg"


# In[ ]:


test= image.load_img(new_img,target_size=image_shape)


# In[ ]:


my_image = image.img_to_array(test)


# In[ ]:


my_image = np.expand_dims(my_image, axis=0)


# In[ ]:


model.predict(my_image)


# In[ ]:


plt.imshow(test_img)


# In[ ]:


model.save('malaria_classifier.h5')


# In[ ]:


import tensorflow as tf
new_model =  tf.keras.models.load_model('malaria_classifier.h5')


# In[ ]:


new_img2="../input/healthy-cell2/healthy_cell2.jpg"


# In[ ]:


test2=image.load_img(new_img2,target_size=image_shape)


# In[ ]:


new_image3 = image.img_to_array(test2)


# In[ ]:


new_img3= np.expand_dims(new_image3, axis=0)


# In[ ]:


new_model.predict(new_img3)


# In[ ]:




