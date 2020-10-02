#!/usr/bin/env python
# coding: utf-8

# The purpose of this kernel is time-wise improving of inference for image classification problem. New model should be faster and more memory efficient than EDA (pretrained caffemodel - GoogleNet). Data set contains 3 classes. Train set ~ 2700 images. Test set ~900 images.

# The idea is to implement some sort of 'bag of words' for every image with combination of tiny convolution(caffe) and gradient boosting (catboost). I was inspired by (https://medium.com/bethgelab/neural-networks-seem-to-follow-a-puzzlingly-simple-strategy-to-classify-images-f4229317261f) (russian translation https://habr.com/en/post/443734/)

# Caffe should be installed every time before starting. This is peculatity of Kaggle kernel. 

# In[1]:


get_ipython().system('conda install -y caffe')


# In[2]:


import caffe
import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier, Pool
from pandas import read_csv

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from datetime import datetime


# Here I load first layer of pretrained model for further comparing with optimized one.

# In[3]:


net_1layer = caffe.Net('../input/1conv.prototxt',      # defines the structure of the model
                '../input/_iter_800000.caffemodel',  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
net = caffe.Net('../input/human_car_deploy.prototxt',      # defines the structure of the model
                '../input/_iter_800000.caffemodel',  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


# Here I prepare array with extracted features. Train dataset was splited to train 90% and validation 10% subsets

# In[4]:


train_files  = read_csv('../input/train_list.csv').iloc[:,0]
val_files = read_csv('../input/val_list.csv').iloc[:,0]
test_files = read_csv('../input/test_list.csv').iloc[:,0]
train_labels=read_csv('../input/train_list.csv').iloc[:,1]
val_labels = read_csv('../input/val_list.csv').iloc[:,1]
test_labels = read_csv('../input/test_list.csv').iloc[:,1]


# Here I perform feature extraction with the first pretrained conv layer of caffemodel_800000. Aftewards, I implement some sort of 'bag of words' for found features (summ and mean)

# In[20]:


print('1 layer extraction started')

def feature_extraction(files_list,size,net):
    data_set = np.full((size, 2 * 64), 0, dtype=int)
    for idx, file in enumerate(files_list):
        image = caffe.io.load_image(file)
        image = np.asarray(image).reshape(1, 3, 224, 224) * 256
        image[0, 0] = image[0, 0] - 123
        image[0, 1] = image[0, 1] - 117
        image[0, 2] = image[0, 2] - 104
        np.swapaxes(image, 0, 2)
        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = image
        ### perform classification
        output = np.asanyarray(net_1layer.forward()['conv1/7x7_s2'])
        output_sum = np.sum(output, (2, 3))
        output = np.where(output < 0, output, 0*output)
        output_mean = np.sum(output, (2, 3))
        data_set[idx][0:64] = output_sum.reshape(-1).astype(int)
        data_set[idx][64:] = output_mean.reshape(-1).astype(int)
    return data_set

train_dataset = feature_extraction(train_files, 2318, net_1layer)
val_dataset = feature_extraction(val_files, 389, net_1layer)


# Here I perform catboost training with extracted features. Estimated AUC for classes is quite good ~ 0.9.

# In[21]:


print('catboost data merge started')

cat_features = [0]

train_dataset = Pool(data=train_dataset,
                     label=train_labels,
                     cat_features=cat_features)

eval_dataset = Pool(data=val_dataset,
                    label=val_labels,
                    cat_features=cat_features)

# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=300,
                           learning_rate=1,
                           depth=2,
                           loss_function='MultiClass',
                           custom_metric='AUC')
# Fit model
model.fit(train_dataset, eval_set=eval_dataset, use_best_model=True)
# Get predicted classes
preds_class = model.predict(eval_dataset)
print(preds_class)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(eval_dataset)
# Get predicted RawFormulaVal
preds_raw = model.predict(eval_dataset,
                          prediction_type='RawFormulaVal')
print(model.get_best_score())
model.save_model('1layer_catboost')


# Inference time estimation for the test dataset

# In[7]:


dt_start = datetime.now()

test_dataset = feature_extraction(test_files, 899, net_1layer)

# Get predicted probabilities for each class
preds_proba = model.predict_proba(test_dataset)

dt_end = datetime.now()
execution_time_cat=(dt_end.minute-dt_start.minute)*60 + dt_end.second-dt_start.second  

# custom AUC
val_binarized = label_binarize(test_labels, classes=[0, 1, 2])
print('AUC 0 class '+ str(roc_auc_score(val_binarized[:,0], preds_proba[:,0])))
print('AUC 1 class '+ str(roc_auc_score(val_binarized[:,1], preds_proba[:,1])))
print('AUC 2 class '+ str(roc_auc_score(val_binarized[:,2], preds_proba[:,2])))


# EDA caffemodel800000

# In[8]:


print('Pretrained caffe 800000 inference started...')
dt_start = datetime.now()
data_set = np.full((899, 2 * 64), 0, dtype=int)
for idx, file in enumerate(test_files):
    image = caffe.io.load_image(file)
    image = np.asarray(image).reshape(1, 3, 224, 224) * 256
    image[0, 0] = image[0, 0] - 123
    image[0, 1] = image[0, 1] - 117
    image[0, 2] = image[0, 2] - 104
    np.swapaxes(image, 0, 2)
    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = image
    ### perform classification
    output = np.asanyarray(net.forward())

dt_end = datetime.now()
execution_time=(dt_end.minute-dt_start.minute)*60 + dt_end.second-dt_start.second 


# In[9]:


print('catboost+caffe 900 images inference time '+ str(execution_time_cat) + str(' sec'))
print('Pretrained caffe 800000 900 images inference time '+ str(execution_time) + str(' sec'))

