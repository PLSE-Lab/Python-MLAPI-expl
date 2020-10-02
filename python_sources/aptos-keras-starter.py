#!/usr/bin/env python
# coding: utf-8

# Transfer learning from pretrained model (Resnet50) using Keras

# In[ ]:


import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, applications
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import backend as K 


# In[ ]:


train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train_df["id_code"]=train_df["id_code"].apply(lambda x:x+".png")
train_df['diagnosis'] = train_df['diagnosis'].astype(str)
train_df.head()


# In[ ]:


# Example of images 
img_names = train_df['id_code'][:10]

plt.figure(figsize=[15,15])
i = 1
for img_name in img_names:
    img = cv2.imread("../input/aptos2019-blindness-detection/train_images/%s" % img_name)[...,[2, 1, 0]]
    plt.subplot(6, 5, i)
    plt.imshow(img)
    i += 1
plt.show()


# In[ ]:


nb_classes = 5
lbls = list(map(str, range(nb_classes)))
batch_size = 32
img_size = 64
nb_epochs = 5


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain_datagen=ImageDataGenerator(\n    rescale=1./255, \n    validation_split=0.25,\n#     horizontal_flip = True,    \n#     zoom_range = 0.3,\n#     width_shift_range = 0.3,\n#     height_shift_range=0.3\n    )\n\ntrain_generator=train_datagen.flow_from_dataframe(\n    dataframe=train_df,\n    directory="../input/aptos2019-blindness-detection/train_images",\n    x_col="id_code",\n    y_col="diagnosis",\n    batch_size=batch_size,\n    shuffle=True,\n    class_mode="categorical",\n    classes=lbls,\n    target_size=(img_size,img_size),\n    subset=\'training\')\n\nvalid_generator=train_datagen.flow_from_dataframe(\n    dataframe=train_df,\n    directory="../input/aptos2019-blindness-detection/train_images",\n    x_col="id_code",\n    y_col="diagnosis",\n    batch_size=batch_size,\n    shuffle=True,\n    class_mode="categorical", \n    classes=lbls,\n    target_size=(img_size,img_size),\n    subset=\'validation\')')


# In[ ]:


model = applications.ResNet50(weights=None, 
                          include_top=False, 
                          input_shape=(img_size, img_size, 3))
model.load_weights('../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')


# In[ ]:


model.trainable = False


# In[ ]:


#Adding custom layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(nb_classes, activation="softmax")(x)
model_final = Model(input = model.input, output = predictions)

model_final.compile(optimizers.rmsprop(lr=0.001, decay=1e-6),loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


# Callbacks

checkpoint = ModelCheckpoint("model_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model_final.fit_generator(generator=train_generator,                   \n                                    steps_per_epoch=100,\n                                    validation_data=valid_generator,                    \n                                    validation_steps=30,\n                                    epochs=nb_epochs,\n                                    callbacks = [checkpoint, early],\n                                    max_queue_size=16,\n                                    workers=2,\n                                    use_multiprocessing=True,\n                                    verbose=0)')


# In[ ]:


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


# In[ ]:


sam_sub_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
sam_sub_df["id_code"]=sam_sub_df["id_code"].apply(lambda x:x+".png")
print(sam_sub_df.shape)
sam_sub_df.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_datagen = ImageDataGenerator(rescale=1./255)\ntest_generator = test_datagen.flow_from_dataframe(  \n        dataframe=sam_sub_df,\n        directory = "../input/aptos2019-blindness-detection/test_images",    \n        x_col="id_code",\n        target_size = (img_size,img_size),\n        batch_size = 1,\n        shuffle = False,\n        class_mode = None\n        )')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_generator.reset()\npredict=model_final.predict_generator(test_generator, steps = len(test_generator.filenames))')


# In[ ]:


predict.shape


# In[ ]:


filenames=test_generator.filenames
results=pd.DataFrame({"id_code":filenames,
                      "diagnosis":np.argmax(predict,axis=1)})
results['id_code'] = results['id_code'].map(lambda x: str(x)[:-4])
results.to_csv("submission.csv",index=False)


# In[ ]:


results.head()

