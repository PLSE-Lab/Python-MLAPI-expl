#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# In[ ]:


BASE_DATA_FOLDER = "../input"
TRAin_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, "train")


# In[ ]:


def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output


# In[ ]:


images = []
labels = []

for class_folder_name in os.listdir(TRAin_DATA_FOLDER):
    class_folder_path = os.path.join(TRAin_DATA_FOLDER, class_folder_name)
    for image_path in glob(os.path.join(class_folder_path, "*.png")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.resize(image, (150, 150))
        image = segment_plant(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (30,30))
        
        image = image.flatten()
        
        images.append(image)
        labels.append(class_folder_name)
        
images = np.array(images)
labels = np.array(labels)


# In[ ]:


label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}


# In[ ]:


label_ids = [label_to_id_dict[x] for x in labels]


# In[ ]:


images_scaled = StandardScaler().fit_transform(images)


# In[ ]:


tsne = TSNE(n_components=2, init='pca', random_state=0)


# In[ ]:


X_tsne = tsne.fit_transform(images_scaled)


# In[ ]:


plt.figure(figsize=(20,20))
plt.grid()
for label_id in np.unique(label_ids):
    plt.scatter(X_tsne[np.where(label_ids == label_id), 0],
                X_tsne[np.where(label_ids == label_id), 1],
                marker='o',
                color= plt.cm.Set1(label_id / 10.),
                linewidth='1',
                alpha=0.8,
                label=id_to_label_dict[label_id])
plt.legend(loc='best')


# In[ ]:


pca_10 = PCA(n_components=10)
pca_result_10 = pca_10.fit_transform(images_scaled)


# In[ ]:


tsne_pca = TSNE(n_components=2, init='pca', random_state=0)


# In[ ]:


X_pca_tsne = tsne_pca.fit_transform(pca_result_10)


# In[ ]:


plt.figure(figsize=(20,20))
plt.grid()
for label_id in np.unique(label_ids):
    plt.scatter(X_pca_tsne[np.where(label_ids == label_id), 0],
                X_pca_tsne[np.where(label_ids == label_id), 1],
                marker='o',
                color= plt.cm.Set1(label_id / 10.),
                linewidth='1',
                alpha=0.8,
                label=id_to_label_dict[label_id])
plt.legend(loc='best')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




