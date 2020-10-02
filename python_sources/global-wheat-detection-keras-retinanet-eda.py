#!/usr/bin/env python
# coding: utf-8

# <h1 align=center> Global Wheat Detection with Keras RetinaNet </h1>
# 
# ### This Notebook is for Training Purpose of Keras RetinaNet.
# ### RetinaNet is very slow as compared to F-RCNN so I've kept epochs and steps per epoch small for fast commiting purpose.
# ### I will make another notebook for inference Shortly.
# 
# ### Credits for the EDA goes to [THIS Notebook](https://www.kaggle.com/devvindan/wheat-detection-eda)
# 
# <h3 align=center style=color:red>Upvote If you find this kernel interesting</h3>

# In[ ]:


import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw
from ast import literal_eval
import matplotlib.pyplot as plt
import urllib
from tqdm.notebook import tqdm


# # Installing Keras-RetinaNet 

# In[ ]:


get_ipython().system('git clone https://github.com/fizyr/keras-retinanet.git')


# In[ ]:


get_ipython().run_line_magic('cd', 'keras-retinanet/')

get_ipython().system('pip install .')


# In[ ]:


get_ipython().system('python setup.py build_ext --inplace')


# In[ ]:


import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color


# ## Let's look at the data

# In[ ]:


root = "/kaggle/input/global-wheat-detection/"
train_img = root+"train"
test_img = root+"test"
train_csv = root+"train.csv"
sample_submission = root+"sample_submission.csv"


# In[ ]:


train = pd.read_csv(train_csv)
train.head()


#  Single Image has multiple bbox

# In[ ]:


print(f"Total Bboxes: {train.shape[0]}")


# # EDA

# Let's Check the Dimensions of images

# In[ ]:


train['width'].unique() == train['height'].unique() == [1024]


# In[ ]:


def get_bbox_area(bbox):
    bbox = literal_eval(bbox)
    return bbox[2] * bbox[3]


# In[ ]:


train['bbox_area'] = train['bbox'].apply(get_bbox_area)


# In[ ]:


train['bbox_area'].value_counts().hist(bins=10)


# In[ ]:


unique_images = train['image_id'].unique()
len(unique_images)


# In[ ]:


num_total = len(os.listdir(train_img))
num_annotated = len(unique_images)

print(f"There are {num_annotated} annotated images and {num_total - num_annotated} images without annotations.")


# ### Sources of Data

# In[ ]:


sources = train['source'].unique()
print(f"There are {len(sources)} sources of data: {sources}")


# In[ ]:


train['source'].value_counts()


# Let's look at how many bounding boxes do we have for each image:

# In[ ]:


plt.hist(train['image_id'].value_counts(), bins=10)
plt.show()


# Max number of bounding boxes is 116, whereas min (annotated) number is 1 

# ## Visualizing images

# In[ ]:


def show_images(images, num = 5):
    
    images_to_show = np.random.choice(images, num)

    for image_id in images_to_show:

        image_path = os.path.join(train_img, image_id + ".jpg")
        image = Image.open(image_path)

        # get all bboxes for given image in [xmin, ymin, width, height]
        bboxes = [literal_eval(box) for box in train[train['image_id'] == image_id]['bbox']]

        # visualize them
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:    
            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], width=3)

        plt.figure(figsize = (15,15))
        plt.imshow(image)
        plt.show()


# In[ ]:


show_images(unique_images)


# What can we tell from visualizations:
# 
# * there are plenty of overlappind bounding boxes
# * all photos seem to be taken vertically 
# * all plants are can be rotated differently, there is no single orientation. this means that different flip and roration augmentations should probably help
# * colors of wheet heads are quite different and seem to depend a little bit on the source
# * wheet heads themselves are seen from very different angles of view relevant to the observer

# # Preprocessing Data for Input to RetinaNet

# In[ ]:


bboxs=[ bbox[1:-1].split(', ') for bbox in train['bbox']]
bboxs=[ f"{int(float(bbox[0]))},{int(float(bbox[1]))},{int(float(bbox[0]))+int(float(bbox[2]))},{int(float(bbox[1])) + int(float(bbox[3]))},wheat" for bbox in bboxs]
train['bbox_']=bboxs
train.head()


# In[ ]:


train_df=train[['image_id','bbox_']]
train_df.head()


# In[ ]:


train_df=train_df.sample(frac=1).reset_index(drop=True)
train_df.head()


# ## Preparing Files to be given for training
# 
# ### Annotation file contains all the path of all images and their corresponding bounding boxes
# ### Class file contains the number of classes but in our case it is just 1 (Wheat)

# In[ ]:


with open("annotations.csv","w") as file:
    for idx in range(len(train_df)):
        file.write(train_img+"/"+train_df.iloc[idx,0]+".jpg"+","+train_df.iloc[idx,1]+"\n")
        


# In[ ]:


with open("classes.csv","w") as file:
    file.write("wheat,0")


# ## Downloading the pretrained model

# In[ ]:


PRETRAINED_MODEL = './snapshots/_pretrained_model.h5'

URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'
urllib.request.urlretrieve(URL_MODEL, PRETRAINED_MODEL)

print('Downloaded pretrained model to ' + PRETRAINED_MODEL)


# ### Model Parameters

# In[ ]:


EPOCHS = 1
BATCH_SIZE=8
STEPS = 100 #len(train_df)//BATCH_SIZE #Keeping it small for faster commit
LR=1e-3


# # Training Model

# In[ ]:


get_ipython().system('keras_retinanet/bin/train.py --random-transform --weights {PRETRAINED_MODEL} --lr {LR} --batch-size {BATCH_SIZE} --steps {STEPS} --epochs {EPOCHS} --no-resize csv annotations.csv classes.csv')


# # Loading the trained model

# In[ ]:


get_ipython().system('ls snapshots')


# In[ ]:


model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0])

model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)


# In[ ]:





# # Predictions

# In[ ]:


li=os.listdir(test_img)
li[:5]


# In[ ]:


def predict(image):
    image = preprocess_image(image.copy())
    #image, scale = resize_image(image)

    boxes, scores, labels = model.predict_on_batch(
    np.expand_dims(image, axis=0)
  )

    #boxes /= scale

    return boxes, scores, labels


# In[ ]:


THRES_SCORE = 0.5

def draw_detections(image, boxes, scores, labels):
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < THRES_SCORE:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(image, b, color=color)

        caption = "{:.3f}".format(score)
        draw_caption(image, b, caption)


# In[ ]:


def show_detected_objects(image_name):
    img_path = test_img+'/'+image_name
  
    image = read_image_bgr(img_path)

    boxes, scores, labels = predict(image)
    print(boxes[0,0].shape)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    draw_detections(draw, boxes, scores, labels)
    plt.figure(figsize=(15,10))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


# In[ ]:


for img in li:
    show_detected_objects(img)


# In[ ]:


preds=[]
imgid=[]
for img in tqdm(li,total=len(li)):
    img_path = test_img+'/'+img
    image = read_image_bgr(img_path)
    boxes, scores, labels = predict(image)
    boxes=boxes[0]
    scores=scores[0]
    for idx in range(boxes.shape[0]):
        if scores[idx]>THRES_SCORE:
            box,score=boxes[idx],scores[idx]
            imgid.append(img.split(".")[0])
            preds.append("{} {} {} {} {}".format(score, int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])))
    


# In[ ]:


preds[0]


# In[ ]:


sub={"image_id":imgid, "PredictionString":preds}
sub=pd.DataFrame(sub)
sub.head()


# In[ ]:


sub_=sub.groupby(["image_id"])['PredictionString'].apply(lambda x: ' '.join(x)).reset_index()
sub_


# In[ ]:


samsub=pd.read_csv("/kaggle/input/global-wheat-detection/sample_submission.csv")
samsub.head()


# In[ ]:


for idx,imgid in enumerate(samsub['image_id']):
    samsub.iloc[idx,1]=sub_[sub_['image_id']==imgid].values[0,1]
    
samsub.head()


# In[ ]:


samsub.to_csv('/kaggle/working/submission.csv',index=False)


# In[ ]:




