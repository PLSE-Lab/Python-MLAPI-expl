#!/usr/bin/env python
# coding: utf-8

# https://github.com/fizyr/keras-retinanet
# 
# https://github.com/priya-dwivedi/aerial_pedestrian_detection

# In[ ]:


get_ipython().system('ls snapshots')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from pathlib import Path
# Any results you write to the current directory are saved as output.


# In[ ]:


malaria_dir=Path('../input/malaria/malaria')
train_df = pd.read_json(malaria_dir / 'training.json')
train_df['path'] = train_df['image'].map(lambda x: malaria_dir / x['pathname'][1:])
train_df['image_available'] = train_df['path'].map(lambda x: x.exists())
print(train_df.shape[0], 'images')
train_df = train_df.query('image_available')
print(train_df.shape[0], 'images available')
train_df.sample(5)
# train_df.head(5)


# In[ ]:


train_df.iloc[0].objects


# In[ ]:


list(malaria_dir.iterdir())


# In[ ]:


object_df = pd.DataFrame([dict(image=c_row['path'], **c_item) for _, c_row in train_df.iterrows() for c_item in c_row['objects']])
cat_dict = {v:k for k,v in enumerate(object_df['category'].value_counts().index, 1)}
print(object_df['category'].value_counts())
object_df.sample(3)


# In[ ]:


len(object_df)


# In[ ]:


def get_coordinates(j):
    x1=j.get('minimum').get('r')
    y1=j.get('minimum').get('c')
    x2=j.get('maximum').get('r')
    y2=j.get('maximum').get('c')
    return [x1,y1,x2,y2]

get_coordinates(object_df.loc[0]['bounding_box'])


# In[ ]:


object_df['coordinates']=object_df['bounding_box'].apply(get_coordinates)


# In[ ]:


object_df.head()


# In[ ]:


def coor0_to_col(l):
    return l[0]
def coor1_to_col(l):
    return l[1]
def coor2_to_col(l):
    return l[2]
def coor3_to_col(l):
    return l[3]


# In[ ]:


data=pd.DataFrame()
data['image_path']=object_df['image']
data['x1']=object_df['coordinates'].apply(coor1_to_col)
data['y1']=object_df['coordinates'].apply(coor0_to_col)
data['x2']=object_df['coordinates'].apply(coor3_to_col)
data['y2']=object_df['coordinates'].apply(coor2_to_col)
data['class']=object_df['category']
data.head()


# In[ ]:


labels=data['class']


# In[ ]:


classes_names=set(data['class'])
classes_names


# In[ ]:


for c in classes_names:
    print('number of ',c,' in data: ',len(data[data['class'] == c]))
    print('ratio with respect to red blood cells : ',len(data[data['class'] == c])/len(data[data['class'] == 'red blood cell']),'\n')


# In[ ]:


import seaborn as sns
from sklearn.model_selection import train_test_split

data_train, data_test, _, y_test = train_test_split(data, labels, test_size=0.3, random_state=1)
data_test, data_val, _, _ = train_test_split(data_test, y_test, test_size=0.5, random_state=1)
print(len(data_train),len(data_test))
data_train.reset_index(inplace=True,drop=True)
data_test.reset_index(inplace=True,drop=True)
data_val.reset_index(inplace=True,drop=True)


# In[ ]:


data_train.head()


# In[ ]:


# sanity check for bounding boxes
import random
i=random.randint(1,len(data_train))
img0=data_train.iloc[i]
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
im = np.array(Image.open(img0.image_path), dtype=np.uint8)
fig,ax = plt.subplots(1)
ax.imshow(im)
rect = patches.Rectangle((img0.x1,img0.y1),img0.x2-img0.x1,img0.y2-img0.y1,linewidth=2,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.show()


# In[ ]:


# import random
# indexes=[]
# for i in range(100):
#     x=random.randint(1,len(data_train))
#     indexes.append(x)
# #     print(x)
# df_train_100=data_train.iloc[indexes]
# df_train_100.reset_index(inplace=True,drop=True)
# df_train_100.head()


# In[ ]:


# sns.countplot(df_train_100['class'])


# In[ ]:


sns.countplot(data_train['class'])


# In[ ]:


sns.countplot(data_test['class'])


# In[ ]:


sns.countplot(data_val['class'])


# In[ ]:


# df_train_100.to_csv('annotations_train_100.csv',index=False,header=False)
data_train.to_csv('annots_train.csv',index=False,header=False)
data_test.to_csv('annots_test.csv',index=False,header=False)
data_val.to_csv('annots_valid.csv',index=False,header=False)


# The data consists of two classes of uninfected cells (RBCs and leukocytes) and four classes of infected cells (gametocytes, rings, trophozoites, and schizonts). Annotators were permitted to mark some cells as difficult if not clearly in one of the cell classes.

# In[ ]:


classes=pd.DataFrame()
classes['class_name']=list(classes_names)
classes['id']=list(range(0,7))
classes.head()


# In[ ]:


classes.to_csv('classes.csv',index=False,header=False)


# In[ ]:


get_ipython().system('git clone https://github.com/fizyr/keras-retinanet')


# In[ ]:


get_ipython().run_line_magic('cd', 'keras-retinanet')
get_ipython().system('pip install . --user')


# In[ ]:


get_ipython().system('wget https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5')


# In[ ]:


get_ipython().system('python setup.py build_ext --inplace')
# model = models.load_model('resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')


# In[ ]:


get_ipython().run_line_magic('cd', '../')


# * freeze-backbone: freeze the backbone layers, particularly useful when we use a small dataset, to avoid overfitting
# * random-transform: randomly transform the dataset to get data augmentation
# * weights: initialize the model with a pretrained model (your own model or one released by Fizyr)
# * batch-size: training batch size, higher value gives smoother learning curve
# * steps: number of steps for epochs
# * epochs: number of epochs to train
# * csv: annotations files generated by the script above

# In[ ]:


# !python3 keras-retinanet/keras_retinanet/bin/debug.py --annotations csv annotations_train.csv classes.csv


# In[ ]:


get_ipython().system('ls keras-retinanet')


# In[ ]:


get_ipython().system('python3 keras-retinanet/keras_retinanet/bin/train.py --freeze-backbone --random-transform --weights keras-retinanet/resnet50_coco_best_v2.1.0.h5 --batch-size 8 --steps 500 --epochs 4 csv annots_train.csv classes.csv --val-annotations annots_valid.csv')
#!python3 keras-retinanet/keras_retinanet/bin/train.py --random-transform --weights keras-retinanet/resnet50_coco_best_v2.1.0.h5 --batch-size 8 --steps 500 --epochs 10 csv annots_train.csv classes.csv --val-annotations annots_valid.csv
#  -W ignore


# In[ ]:


get_ipython().system('ls snapshots')


# In[ ]:


get_ipython().system('python3 keras-retinanet/keras_retinanet/bin/evaluate.py csv annots_test.csv classes.csv snapshots/resnet50_csv_04.h5 --convert-model')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

# automatically reload modules when they have changed
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


# In[ ]:


get_ipython().system('ls snapshots')


# In[ ]:


model = models.load_model('snapshots/resnet50_csv_04.h5', backbone_name='resnet50')
model = models.convert_model(model) # converting to inference mode


# In[ ]:


classes_dic={}
i=0
for c in classes['class_name']:
    classes_dic[i]=c
    i+=1
classes_dic


# In[ ]:


# data_test=data_test.reset_index()
# data_test.drop(['index','level_0'],axis=1,inplace=True)
data_test.loc[0].image_path


# In[ ]:


# load image
image = read_image_bgr(data_test.loc[0].image_path)

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)
# plt.imshow(image)
# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
#     if score < 0.5:
#         break
        
    color = label_color(label)
    
    b = box.astype(int)
    draw_box(draw, b, color=color)
    
    caption = "{} {:.3f}".format(classes_dic[label], score)
    draw_caption(draw, b, caption)
    
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()

