#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import keras
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


img_x = 224
img_y = 224
bat_siz = 32
num_epok = 32
# In[2]:


data_generator = ImageDataGenerator(
        zoom_range = 0.4,
        vertical_flip  = True,
        horizontal_flip = True,
        rescale=1.0/255.0
        )

model = load_model('../input/aptos-densenet-train-submit/densenet_plus_five')

test_data_labels = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_data_labels['id_code'] = test_data_labels['id_code'] + '.png'


test_generator = data_generator.flow_from_dataframe(dataframe = test_data_labels,
                                                     directory = os.path.join('..', 'input','aptos2019-blindness-detection','test_images'),
        target_size = (img_x, img_y), 
    
        x_col = 'id_code',
        class_mode = None,
        batch_size = bat_siz
        )

predictions = model.predict_generator(test_generator,
                                      steps = test_data_labels.shape[0]/bat_siz)

pred_holder = []
for x in predictions:
    pred_holder.append(np.argmax(x))
    


output_df = pd.DataFrame({'diagnosis':pred_holder, 
                          'id_code':test_data_labels.id_code.str.replace(pat = "\.png", repl = "")})


output_df.to_csv("submission.csv", mode = "w")
output_df.head()

