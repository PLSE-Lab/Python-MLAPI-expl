#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
from tqdm import tqdm
import json
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')


# # Loading data

# In[ ]:


train_json = '../input/region-proposals-of-crop-weed-dataset/train.json'
test_json = '../input/region-proposals-of-crop-weed-dataset/test.json'
images_path = '../input/crop-and-weed-detection-data-with-bounding-boxes/agri_data/data/'
model_path  = '../input/rcnn-training-part-1-finetuning/RCNN_crop_weed_classification_model.h5'
label_csv = '../input/convert-yolo-labels-to-pascalvoc-format/pascal_voc_format.csv'
negative_ex_path = '../input/rcnn-data-preprocessing-part-2/Train/background/'


# ## Loading pre generated region proposals and Negative examples

# In[ ]:


with open(train_json,'r') as train:
    train_region = json.load(train)


# In[ ]:


with open(test_json,'r') as test:
    test_region = json.load(test)


# In[ ]:


train_images_list = list(train_region.keys())
test_images_list = list(test_region.keys())


# In[ ]:


print(len(train_images_list))
print(len(test_images_list))


# ## Loading object annotation

# In[ ]:


labels = pd.read_csv(label_csv)
labels.head()


# ## Loading pretrained CNN model

# In[ ]:


model = tf.keras.models.load_model(model_path)


# In[ ]:


model.summary()


# ## loading model without last two Fully connected layers

# In[ ]:


model_without_last_2FC = tf.keras.models.Model(model.inputs,model.layers[-5].output)


# In[ ]:


model_without_last_2FC.summary()


# # Extracting features from ground truth labeled images

# When we pass image from model it will return (1,4096) size feature vector

# In[ ]:


train_features = []

test_features = []


for index in tqdm(range(len(labels))):
    id = labels.loc[index,'filename']
    img = cv2.imread(images_path + id)
    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    xmin,ymin,xmax,ymax = int(labels.loc[index,'xmin']) ,int(labels.loc[index,'ymin']),int(labels.loc[index,'xmax']),int(labels.loc[index,'ymax'])

    resized = cv2.resize(rgb_img[ymin:ymax,xmin:xmax,:],(224,224))

    feature_of_img = model_without_last_2FC.predict(resized.reshape(1,224,224,3)/255)
    
    if id in train_images_list:
        
        train_features.append([feature_of_img,labels.loc[index,'class']])
        
    else:
        test_features.append([feature_of_img,labels.loc[index,'class']])
      


# In[ ]:


print(len(train_features))

print(len(test_features))


# # Extracting features from Negative examples

# In[ ]:


for index,img in tqdm(enumerate(os.listdir(negative_ex_path)[:5000])):  #only extracting for 10,000 images
    img = cv2.imread(negative_ex_path + img )
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #images already in (224,224,3)
    feature_of_img = model_without_last_2FC.predict(rgb.reshape(1,224,224,3)/255)
    if index<3500:
        train_features.append([feature_of_img,'background'])
    else:
        test_features.append([feature_of_img,'background'])


# # Preparing data for SVM

# In[ ]:


import random
random.shuffle(train_features)


# In[ ]:


X_train = np.array([x[0] for x in train_features])
X_train = X_train.reshape(-1,4096)


# In[ ]:


X_train.shape


# In[ ]:


y_train = [x[1] for x in train_features]
y_train = np.array(y_train).reshape(-1,1)


# In[ ]:


y_train.shape


# In[ ]:


X_test = np.array([x[0] for x in test_features])
X_test = X_test.reshape(-1,4096)


# In[ ]:


y_test = [x[1] for x in test_features]
y_test = np.array(y_test).reshape(-1,1)


# # SVM Training

# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


# In[ ]:


svm_model_linear = SVC(kernel = 'linear', C = 1,probability=True).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test)


# In[ ]:


accuracy = svm_model_linear.score(X_test, y_test)


# In[ ]:


accuracy


# In[ ]:


cm = confusion_matrix(y_test, svm_predictions) 


# In[ ]:


sns.heatmap(cm,annot=True)


# In[ ]:





# # Check on some images

# In[ ]:


img = cv2.imread(negative_ex_path + os.listdir(negative_ex_path)[45] )
rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(rgb)


# In[ ]:


feature_of_img = model_without_last_2FC.predict(rgb.reshape(1,224,224,3)/255)


# In[ ]:


svm_model_linear.predict(feature_of_img)


# In[ ]:


svm_model_linear.predict_proba(feature_of_img)


# In[ ]:


svm_model_linear.classes_


# In[ ]:


img = cv2.imread(images_path+'agri_0_1024.jpeg')
rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(rgb)


# In[ ]:


resized = cv2.resize(rgb,(224,224))


# In[ ]:


plt.imshow(resized)


# In[ ]:


svm_model_linear.predict_proba(model_without_last_2FC.predict(resized.reshape(1,224,224,3)/255))


# # Saving SVM model

# In[ ]:


import pickle

with open('svm_classifier.pkl','wb') as svm_model:
    pickle.dump(svm_model_linear , svm_model)


# In[ ]:




