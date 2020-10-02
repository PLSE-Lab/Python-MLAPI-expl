#!/usr/bin/env python
# coding: utf-8

# # Understanding the data and the data layout

# In[ ]:


import os
from os import listdir


# In[ ]:


base_path = "../input/plantvillage/PlantVillage/"


# In[ ]:


os.listdir(base_path)


# In[ ]:


len(os.listdir(base_path))


# In[ ]:


from glob import glob
imagePatches = glob("../input/plantvillage/PlantVillage/*/*.*", recursive=True)


# In[ ]:


len(imagePatches)


# In[ ]:


imagePatches[0:10]


# In[ ]:


image_path = "../input/plantvillage/PlantVillage/Tomato_Bacterial_spot/3a5a5fef-8a3a-4f70-ab85-eaf5e3ecf6f2___GCREC_Bact.Sp 3449.JPG'"


# In[ ]:


dir_name = os.path.dirname(image_path)


# In[ ]:


len(dir_name.split("/"))


# In[ ]:


dir_name.split("/")[4]


# # Image Processing using fastai

# In[ ]:


from fastai import *
from fastai.vision import *

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve

from math import floor


# In[ ]:


path = Path(base_path)


# In[ ]:


path


# In[ ]:


directory_root = '../input/plantvillage/'


# In[ ]:


image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        
        for disease_folder in plant_disease_folder_list :
            # remove .DS_Store from list
            if disease_folder == ".DS_Store" :
                plant_disease_folder_list.remove(disease_folder)

        for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                
            for single_plant_disease_image in plant_disease_image_list :
                if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)

            for image in plant_disease_image_list[:200]:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(image_directory)
                    label_list.append(plant_disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")


# In[ ]:


image_list[0:10]


# In[ ]:


tfms=get_transforms(flip_vert=True, max_warp=0., max_zoom=0., max_rotate=0.)


# In[ ]:


def get_labels(file_path): 
    dir_name = os.path.dirname(file_path)
    split_dir_name = dir_name.split("/")
    dir_levels = len(split_dir_name)
    label  = split_dir_name[dir_levels - 1]
    return(label)


# In[ ]:


data = ImageDataBunch.from_name_func(path, image_list, label_func=get_labels,  size=96, 
                                     bs=64,num_workers=2,ds_tfms=tfms
                                  ).normalize()


# In[ ]:


data.show_batch(rows=3, figsize=(8,8))


# In[ ]:


learner= cnn_learner(data, models.densenet121, metrics=[accuracy], model_dir='/tmp/models/')


# In[ ]:


learner.lr_find()
learner.recorder.plot()


# In[ ]:


lr=1e-1
learner.fit_one_cycle(1, lr)


# In[ ]:


learner.save('model-1')


# In[ ]:


learner.unfreeze()


# In[ ]:


learner.lr_find()
learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(5,slice(1e-5,1e-3))


# In[ ]:


learner.save('model-2')
learner.load('model-2')


# In[ ]:


learner.fit_one_cycle(80,slice(3e-5,3e-3))


# In[ ]:


learner.recorder.plot_losses()


# In[ ]:


conf= ClassificationInterpretation.from_learner(learner)
conf.plot_confusion_matrix(figsize=(10,8))


# In[ ]:


predictions,labels = learner.get_preds(ds_type=DatasetType.Valid)

predictions = predictions.numpy()
labels = labels.numpy()

predicted_labels = np.argmax(predictions, axis = 1)
print((predicted_labels == labels ).sum().item()/ len(predicted_labels))

