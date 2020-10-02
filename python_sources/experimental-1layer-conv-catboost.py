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


# Here I convert caffe weights to numpy for experiments

# In[4]:


def shai_net_to_py_readable(prototxt_filename, caffemodel_filename):
  net = caffe.Net(prototxt_filename, caffemodel_filename, caffe.TEST) # read the net + weights
  pynet_ = []
  for li in range(len(net.layers)):  # for each layer in the net
    layer = {}  # store layer's information
    layer['name'] = net._layer_names[li]
    # for each input to the layer (aka "bottom") store its name and shape
    layer['bottoms'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape)
                         for bi in list(net._bottom_ids(li))]
    # for each output of the layer (aka "top") store its name and shape
    layer['tops'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape)
                      for bi in list(net._top_ids(li))]
    layer['type'] = net.layers[li].type  # type of the layer
    # the internal parameters of the layer. not all layers has weights.
    layer['weights'] = [net.layers[li].blobs[bi].data[...]
                        for bi in range(len(net.layers[li].blobs))]
    pynet_.append(layer)
  return pynet_


weights = shai_net_to_py_readable('../input/1conv.prototxt','../input/_iter_800000.caffemodel')[1]['weights'][0]
bias = shai_net_to_py_readable('../input/1conv.prototxt','../input/_iter_800000.caffemodel')[1]['weights'][1]


print(weights.shape)


# Here I prepare array with extracted features. Train dataset was splited to train 90% and validation 10% subsets

# In[5]:


train_files  = read_csv('../input/train_list.csv').iloc[:,0]
val_files = read_csv('../input/val_list.csv').iloc[:,0]
test_files = read_csv('../input/test_list.csv').iloc[:,0]
train_labels=read_csv('../input/train_list.csv').iloc[:,1]
val_labels = read_csv('../input/val_list.csv').iloc[:,1]
test_labels = read_csv('../input/test_list.csv').iloc[:,1]


# Here I define preprocessing function what perform special 'convolution'. In particular, for every conv pixel I summ all pixels of image what suppose to interact to them, and save them. Afterwards, special 'convolution' will multiply every kernel to this small array insteat of whole image to get the same result.

# In[6]:


def A_convolution(image, kernel_size, stride):
    conv_image = np.full((3,kernel_size, kernel_size), 0, dtype=int)
    step = 2 * kernel_size
    image_split_num= int(image.shape[2]/ step)
    # first two zycle perform passing every pixel according kernel size
    for i in range(0,kernel_size): 
        for j in range(0,kernel_size):
            # next cycles select parts of image
            summ_Blue = 0
            summ_Green = 0
            summ_Red = 0
            for k in range(0,image_split_num):
                for m in range(0,image_split_num):
                    im_i = k * step + i
                    im_j = m * step + j
                    summ_Blue = summ_Blue + image[0, 1, im_i, im_j]
                    summ_Green = summ_Green + image[0, 2, im_i, im_j]
                    summ_Red =  summ_Red + image[0, 0, im_i, im_j]
            conv_image[0,i,j] = summ_Blue
            conv_image[1,i,j] = summ_Green
            conv_image[2,i,j] = summ_Red
    
    return conv_image


# In[36]:


def A_convolution_numpy_style(image, kernel_size, stride):
    conv_image = np.full((3,kernel_size, kernel_size), 0, dtype=int)
    step = 3 * kernel_size
    image_split_num= int(image.shape[2]/ step)

    summ_Blue = np.full((kernel_size, kernel_size), 0, dtype=int)
    summ_Green = np.full((kernel_size, kernel_size), 0, dtype=int)
    summ_Red = np.full((kernel_size, kernel_size), 0, dtype=int)
    
    for k in range(0,image_split_num):
        for m in range(0,image_split_num):
            summ_Blue = summ_Blue + image[0, 1, k*step:(k * step) + kernel_size , m * step:(m*step) + kernel_size]
            summ_Green = summ_Green + image[0, 2,  k*step:(k * step) + kernel_size , m * step:(m*step) + kernel_size]
            summ_Red =  summ_Red + image[0, 0,  k*step:(k * step) + kernel_size , m * step:(m*step) + kernel_size]
    conv_image[0] = summ_Blue
    conv_image[1] = summ_Green
    conv_image[2] = summ_Red
    
    return conv_image


# Here I perform experimentall feature extraction for summ and mean values

# In[37]:


print('1 layer exprerimental extraction started')


def feature_extraction(files_list,size,net,weights):
    data_set = np.full((size, 3 * 64), 0, dtype=int)
    #data_set_caffe = np.full((size, 3 * 64), 0, dtype=int)
    for idx, file in enumerate(files_list):
        image = caffe.io.load_image(file)
        image = np.asarray(image).reshape(1, 3, 224, 224) * 256
        image[0, 0] = image[0, 0] - 123
        image[0, 1] = image[0, 1] - 117
        image[0, 2] = image[0, 2] - 104
        np.swapaxes(image, 0, 2)
        output = A_convolution_numpy_style(image, 7,2)
        #output = A_convolution(image, 7,2)

        
        # perform convolution
        for i in range(0,64):
            data_set[idx][i] = np.sum(output[0]*weights[i,1])
            data_set[idx][i+64] =  np.sum(output[1] * weights[i,2])
            data_set[idx][i+128] = np.sum(output[2] * weights[i,0])

    return data_set

train_dataset = feature_extraction(train_files, 2318, net_1layer, weights)
val_dataset = feature_extraction(val_files, 389, net_1layer, weights)


# Here I perform catboost training with extracted features. Estimated AUC for classes is quite good ~ 0.9.

# In[38]:


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

# In[39]:


dt_start = datetime.now()

test_dataset = feature_extraction(test_files, 899, net_1layer, weights)

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

# In[40]:


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


# In[42]:


print('catboost+caffe 900 images inference time '+ str(execution_time_cat) + str(' sec'))
print('Pretrained caffe 800000 900 images inference time '+ str(execution_time) + str(' sec'))

