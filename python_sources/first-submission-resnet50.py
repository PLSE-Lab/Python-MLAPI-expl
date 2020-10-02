#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.misc import imread
from PIL import Image
from keras.models import load_model 
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


with open('../input/config/model.json') as json_file:
        model_config = json.load(json_file)['get_resnet50']
WEIGHT_PATH = '../input/resnet50/ResNet50-12-0.43.hdf5'
TEST_DATA_PATH = '../input/aptos2019-blindness-detection/test_images/'


# In[ ]:


model = load_model(WEIGHT_PATH)


# In[ ]:


# SIZE = 300
submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
predicted = []
for i, name in tqdm(enumerate(submit['id_code'])):
    path = os.path.join('../input/aptos2019-blindness-detection/test_images/', name+'.png')
    image = cv2.imread(path)
    image = cv2.resize(image, (320,240))
    score_predict = model.predict((image[np.newaxis])/255)
    label_predict = np.argmax(score_predict)
    # label_predict = score_predict.astype(int).sum() - 1
    predicted.append(str(label_predict))


# In[ ]:


submit['diagnosis'] = predicted
submit.to_csv('submission.csv', index=False)
submit.head()


# In[ ]:





# In[ ]:





# # Older Code

# In[ ]:


submission_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
submission_df.head()


# In[ ]:


def read_and_preprocess_image(id_code):    
    image = imread(os.path.join(TEST_DATA_PATH, id_code + '.png'))
    image = np.array(Image.fromarray(image).resize(model_config['input_shape'][:-1][::-1])).astype(np.uint8)
    return image


# In[ ]:


id_codes = list(submission_df['id_code'])
x_test = np.array([read_and_preprocess_image(id_code) for id_code in id_codes])


# In[ ]:


test_generator = test_data_gen.flow(x_test)


# In[ ]:


class_labels = ['0', '1', '2', '3', '4']


# In[ ]:


steps_need = test_generator.n//test_generator.batch_size + 1
test_generator.reset() # you need to restart whenever you call the predict_generator.
pred = model.predict_generator(test_generator, steps = steps_need, verbose=1)


# In[ ]:


predicted_class_indices=np.argmax(pred,axis=-1)
predictions = [str(i) for i in predicted_class_indices]
predictions


# In[ ]:


print(np.unique(predictions))
print(len(predictions))


# In[ ]:


submission_df['diagnosis'] = predictions
submission_df.head()


# In[ ]:


submission_df.to_csv('submission.csv', index=False)


# In[ ]:




