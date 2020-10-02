#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import applications, optimizers
from pandas import DataFrame


# In[ ]:


img_size = 224
batch_size = 32
epochs = 3
train_steps_per_epoch = 2000
#just test for double commit
# one more time


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_datagen=ImageDataGenerator(\n#         rotation_range=40,\n#         width_shift_range=0.2,\n#         height_shift_range=0.2,\n#         shear_range=0.2,\n#         zoom_range=0.2,\n#         horizontal_flip=True,\n#         vertical_flip = True\n        )\n\ntrain_generator = train_datagen.flow_from_directory(\n        '../input/detected-plates-2/plates_detected/plates_detected',\n        target_size=(img_size, img_size),\n        batch_size=batch_size,\n        class_mode='binary')\n\ntest_datagen = ImageDataGenerator()\ntest_generator = test_datagen.flow_from_directory(  \n        '../input/platesv2/plates/plates',\n        classes=['test'],\n        target_size = (img_size, img_size),\n        batch_size = 1,\n        shuffle = False,        \n        class_mode = 'binary')    ")


# In[ ]:


base_model = applications.InceptionResNetV2(weights='imagenet', 
                          include_top=False, 
                          input_shape=(img_size, img_size, 3))
base_model.trainable = False
x = base_model.output
x = Flatten()(x)
x = Dense(4096, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)
model = Model(inputs = base_model.input, outputs = predictions)

model.compile(loss='binary_crossentropy', optimizer = optimizers.rmsprop(lr=0.0001, decay=1e-5), metrics=['accuracy'])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit_generator(\n        train_generator,\n        steps_per_epoch=train_steps_per_epoch,\n        epochs=epochs,\n        verbose=1)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "test_generator.reset()\npredict = model.predict_generator(test_generator, steps = len(test_generator.filenames))\nd = {'id': [n for n in range(len(predict))], \n     'label': ['dirty' if x > 0.5 else 'cleaned' for x in predict]}\nresult_df = DataFrame(data=d)\nprint(result_df['label'].value_counts())\nresult_df.to_csv('Br_kaggle_sub.csv', index=False)")


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# create a link to download the dataframe
create_download_link(result_df)

