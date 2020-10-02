#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os

from tqdm import tqdm
import PIL
import cv2
from PIL import Image, ImageOps

from keras.models import Sequential, load_model
from keras.layers import (Activation, Dropout, Flatten, Dense, Input, Conv2D, GlobalAveragePooling2D)
from keras.applications.densenet import DenseNet121
import keras
from keras.models import Model

SIZE = 224
NUM_CLASSES = 1108

train_csv = pd.read_csv("../input/recursion-cellular-image-classification/train.csv")
test_csv = pd.read_csv("../input/recursion-cellular-image-classification/test.csv")
sub = pd.read_csv("../input/recursion-cellular-keras-densenet/submission.csv")


# # Train data

# Look at the first 10 sirnas plates assignments across the train set. One can observe that two sirnas that are on the same plate in the first experiment stay on the same plate for all experiments. Moreover, there are only 3 unique rows. 

# In[ ]:


np.stack([train_csv.plate.values[train_csv.sirna == i] for i in range(10)]).transpose()


# The above observation can be easily verified on the whole train data, - there are 4 groups of 277 sirnas in each group which stick together. 
# 
# But is there any order of groups to plates assignment? In general, there are `4*3*2=24` possible combinations of assigning 4 groups to 4 plates. But in the train data only 3 are active, each assignment appearing 22, 7 and 4 times respectively.

# In[ ]:


# you will see the same output here for each sirna number
train_csv.loc[train_csv.sirna==0,'plate'].value_counts()


# Later we will see that the 4th combination, missing from the training data, does in fact appear in the test data. My conlusion here is that Recursion used some kind of rotation of plates only, therefore only 4 combinations.
# 
# Let's calculate which sirna belongs to which plate in every of the 4 assignments:

# In[ ]:


plate_groups = np.zeros((1108,4), int)
for sirna in range(1108):
    grp = train_csv.loc[train_csv.sirna==sirna,:].plate.value_counts().index.values
    assert len(grp) == 3
    plate_groups[sirna,0:3] = grp
    plate_groups[sirna,3] = 10 - grp.sum()
    
plate_groups[:10,:]


# # Test data

# Now let's take a look if we observe the same behavior in the test data. I use the output predictions from the kernel that I mentioned to calculate average probability of each assignment for every experiment.

# In[ ]:


all_test_exp = test_csv.experiment.unique()

group_plate_probs = np.zeros((len(all_test_exp),4))
for idx in range(len(all_test_exp)):
    preds = sub.loc[test_csv.experiment == all_test_exp[idx],'sirna'].values
    pp_mult = np.zeros((len(preds),1108))
    pp_mult[range(len(preds)),preds] = 1
    
    sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
    assert len(pp_mult) == len(sub_test)
    
    for j in range(4):
        mask = np.repeat(plate_groups[np.newaxis, :, j], len(pp_mult), axis=0) ==                np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
        
        group_plate_probs[idx,j] = np.array(pp_mult)[mask].sum()/len(pp_mult)


# Here we go, this is the average probabilities for each test experiment to be in every of the 4 assignments:
# 
# One can see the favorites. 

# In[ ]:


pd.DataFrame(group_plate_probs, index = all_test_exp)


# Let's select the most probable assignment for every test experiment. You may say that some of the selections here are not certain and the probabilities are too close. But we get the same assignments with our much better models, so even this relatively simple model is able to make correct assignments.

# In[ ]:


exp_to_group = group_plate_probs.argmax(1)
print(exp_to_group)


# # Running predictions with the existing DenseNet121 model

# In the code below we load the model, make predictions to get the full probabilites matrix, and set 3 out of 4 plates for every sirna to zero, according to the assignment that we previously selected.

# In[ ]:


from keras.applications.nasnet import NASNetLarge
def create_model(input_shape,n_out):
    input_tensor = Input(shape=input_shape)
    base_model = NASNetLarge(include_top=False,
                   weights='imagenet',
                   input_tensor=input_tensor)
    for layer in base_model.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    final_output = Dense(n_out, activation='softmax', name='final_output')(x)
    model = Model(input_tensor, final_output)
    
    return model


# In[ ]:


model = create_model(input_shape=(SIZE,SIZE,3),n_out=NUM_CLASSES)


# In[ ]:


predicted = []
for i, name in tqdm(enumerate(test_csv['id_code'])):
    path1 = os.path.join('../input/recursion-cellular-image-classification-224-jpg/test/test/', name+'_s1.jpeg')
    image1 = cv2.imread(path1)
    score_predict1 = model.predict((image1[np.newaxis])/255)
    
    path2 = os.path.join('../input/recursion-cellular-image-classification-224-jpg/test/test/', name+'_s2.jpeg')
    image2 = cv2.imread(path2)
    score_predict2 = model.predict((image2[np.newaxis])/255)
    
    predicted.append(0.5*(score_predict1 + score_predict2))
    #predicted.append(score_predict1)


# In[ ]:


predicted = np.stack(predicted).squeeze()


# this is the function that sets 75% of the sirnas to zero according to the selected assignment:

# In[ ]:


def select_plate_group(pp_mult, idx):
    sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
    assert len(pp_mult) == len(sub_test)
    mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis=0) !=            np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
    pp_mult[mask] = 0
    return pp_mult


# In[ ]:


for idx in range(len(all_test_exp)):
    #print('Experiment', idx)
    indices = (test_csv.experiment == all_test_exp[idx])
    
    preds = predicted[indices,:].copy()
    
    preds = select_plate_group(preds, idx)
    sub.loc[indices,'sirna'] = preds.argmax(1)


# In[ ]:


(sub.sirna == pd.read_csv("../input/recursion-cellular-keras-densenet/submission.csv").sirna).mean()


# In[ ]:


sub.to_csv('../working/submission.csv', index=False, columns=['id_code','sirna'])

