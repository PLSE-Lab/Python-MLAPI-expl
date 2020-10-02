#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os

df_labels = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")
df_labels.head()


# In[ ]:


pic_ids, bboxes = [], []
for img in os.listdir("/kaggle/input/global-wheat-detection/train"):
    pic_id = img[:-4]
    bbox = df_labels.loc[df_labels['image_id'] == pic_id]["bbox"]
    
    pic_ids.append(pic_id)
    bboxes.append(bbox)


# ### Display boxes on pictures.

# In[ ]:


from ast import literal_eval

def draw_bboxes(img, bbox):
    for box in bbox:
        box = literal_eval(box)
        x, y, w, h = [int(n) for n in box]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    return img


# In[ ]:


import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["figure.figsize"] = 15, 15

for i in range(4):
    pic = pic_ids[i]
    bbox = bboxes[i].values
    
    path = f"/kaggle/input/global-wheat-detection/train/{pic}.jpg"
    pic = cv2.imread(path)
    
    pic = draw_bboxes(pic, bbox)
    
    plt.imshow(pic)
    plt.show()


# In[ ]:


import csv

img_path = "/kaggle/input/global-wheat-detection/train"

with open("img.csv", "w") as f:
    for i in range(len(pic_ids)):
        bbox = bboxes[i].values
        for box in bbox:
            box = literal_eval(box)
            box = [int(b) for b in box]
            try:
                a, b, c, d = box[0], box[1], box[0]+box[2], box[1]+box[3]
                s = f"{img_path}/{pic_ids[i]}.jpg,{a},{b},{c},{d},wheat\n"
    #             s = f"{pic_ids[i]};{b}\n"

            except:
                continue
            f.write(s)


# In[ ]:


sample_df = pd.read_csv("img.csv")
sample_df.head()


# In[ ]:


with open("classes.csv", "w") as f:
    f.write("wheat,0")


# In[ ]:


get_ipython().system(" wget 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'")


# In[ ]:


get_ipython().system('git clone https://github.com/fizyr/keras-retinanet.git')


# In[ ]:


get_ipython().run_line_magic('cd', 'keras-retinanet')
get_ipython().system(' pip install .')


# In[ ]:


get_ipython().system('python setup.py build_ext --inplace')


# ## IMPORTANT!
# You have to downgrade keras to version 2.3.1. It is necessary for the code to work.

# In[ ]:


get_ipython().system(' pip install keras==2.3.1')


# In[ ]:


get_ipython().system('keras_retinanet/bin/train.py --freeze-backbone --random-transform --weights "../resnet50_coco_best_v2.1.0.h5" --lr 1e-3 --batch-size 8 --steps 100 --epochs 5  csv ../img.csv ../classes.csv')


# In[ ]:


get_ipython().run_line_magic('cd', 'snapshots')
os.listdir(os.getcwd())


# In[ ]:


from keras_retinanet.models import load_model
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet import models

model = load_model("resnet50_csv_01.h5", backbone_name='resnet50')
model = models.convert_model(model)


# In[ ]:


from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

def draw_boxes(img_id, img, boxes, scores, labels):
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < 0.5:
            break
        box = [int(b) for b in box]
        
        submission.append([img_id, score, box])
        
        draw_box(img, box, color=label_color(label))
        score = "{:.3f}".format(score)
        draw_caption(img, box, score)


# In[ ]:


test_path = "/kaggle/input/global-wheat-detection/test"
test_imgs  = [img for img in os.listdir(test_path)]


# In[ ]:


get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('mkdir', 'detected')
get_ipython().run_line_magic('cd', 'detected')


# In[ ]:


submission = []

for i in range(len(test_imgs)):
    pic_path = test_path + "/" + test_imgs[i]
    img = cv2.imread(pic_path)
    pic = preprocess_image(img)
    
    box, score, label = model.predict_on_batch(np.expand_dims(pic, axis=0))
    draw_boxes(test_imgs[i], img, box, score, label)
    
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.savefig(test_imgs[i], )


# In[ ]:


df = pd.DataFrame(submission)
df.columns = ["img", "prediction", "bbox"]
df


# In[ ]:


get_ipython().run_line_magic('cd', '../..')


# In[ ]:


df.to_csv("wheat_detected.csv")


# # CONTINUATION [HERE](https://www.kaggle.com/imakaruamikurah/detect-wheat-results)

# In[ ]:




