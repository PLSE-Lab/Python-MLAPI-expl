#!/usr/bin/env python
# coding: utf-8

# # Object Detection Using Google Cloud AutoML Vision
# Tired of testing different neural network architectures and all that hyperparameter tuning? Why not let a machine do it all and just concentrate on putting together the dataset? Dream or reality? Well, Google AutoML claims to do just that. So I decided to give AutoML a spin with the [ArTaxOr dataset](https://www.kaggle.com/mistag/arthropod-taxonomy-orders-object-detection-dataset).  
# There is a nice [tutorial](https://cloud.google.com/vision/automl/object-detection/docs/how-to) on using Cloud AutoML Vision Object Detection, and I had no problem to follow it.
# 

# ## Step one: Prepare the dataset
# AutoML just needs a .csv file with one bounding box per line in addition to the .jpg files. Let us start with createing the .csv file. The ArTaxOr starter kernel outputs pickled objects data, we can simply load that file and export to .csv. The format of each line needs to be:  
# ```<set>, <image path>,<label>, x_relative_min, y_relative_min, x_relative_max, y_relative_min, x_relative_max, y_relative_max, x_relative_min, y_relative_max```
# where `<set>` is TRAIN, TEST, VALIDATE or UNASSIGNED. The latter means we leave the splitting to AutoML, which wants a 80%-10%-10% split of the dataset. `<image path>` is the path to the image in a Google storage bucket, and the four vertices of the bounding box follow last. Notice that AutoML will discard objects that are too small (less than 8x8pix), and also discard files that have no bounding boxes.
# 

# In[ ]:


import pandas as pd
import numpy as np

BUCKET = 'gs://<your-bucket-name>' # change this to actual storage bucket where images are stored

pickles='/kaggle/input/starter-arthropod-taxonomy-orders-data-exploring/'
labels=pd.read_pickle(pickles+'ArTaxOr_labels.pkl')
df=pd.read_pickle(pickles+'ArTaxOr_filelist.pkl')
anno=pd.read_pickle(pickles+'ArTaxOr_objects.pkl')


# In[ ]:


df = df.sample(frac=1).reset_index(drop=True) # shuffle dataset (not really required, AutoML will do)
automl=pd.DataFrame(columns=['set', 'file', 'label', 'xmin1', 'ymin1', 'xmax2', 'ymin2', 'xmax3', 'ymax3', 'xmin4' , 'ymax4'])
for i in range(len(df)):
    an=anno[anno.id == df.iloc[i].id]
    for j in range(len(an)):
        automl=automl.append({'set': 'UNASSIGNED',
                              'file': BUCKET+an.file.iloc[j].replace('/kaggle/input',''),
                              'label': an.label.iloc[j],
                              'xmin1': an.left.iloc[j],
                              'ymin1': an.top.iloc[j],
                              'xmax2': an.right.iloc[j],
                              'ymin2': an.top.iloc[j],
                              'xmax3': an.right.iloc[j],
                              'ymax3': an.bottom.iloc[j],
                              'xmin4': an.left.iloc[j],
                              'ymax4': an.bottom.iloc[j]}, ignore_index=True)
automl.to_csv('./ArTaxOr.csv', index=False, header=False)
get_ipython().system('head -3 ./ArTaxOr.csv')


# All that is needed now is to transfer the ArTaxOr dataset and the ArTaxOr.csv file to a storage bucket. One note about .zip files though: There is no direct way to unzip .zip files in a storage bucket. I found it most convenient to unzip ArTaxOr.zip on a local machine and then transfer the directory to the bucket with:  
# ```gsutil -m cp -r ArTaxOr gs://<bucket name>```  
# `gsutil` is part of the Google Cloud SDK

# ## Step 2: Run AutoML
# There are no hyperparameters to set, only to choose between a model that will be deployed to the cloud (best accuracy) or to an edge device. I choose the latter, to have a model I can run locally. There is also a tradeoff between latency and accuracy:
# ![optimize.png](attachment:2019-10-23%2020_36_49-Train%20new%20model%20%E2%80%93%20Vision%20%E2%80%93%20ArTaxOr%20%E2%80%93%20Google%20Cloud%20Platform.png)
# 
# Then there is the matter of setting the maximum number of training hours AutoML can use. AutoML will stop training automatically if it finishes before the quota is reached. At the time of writing, the first 20 hours are free the the price is $18 per hour after that.
# ![Train%20new%20model%20%E2%80%93%20Vision%20%E2%80%93%20ArTaxOr%20%E2%80%93%20Google%20Cloud%20Platform.png](attachment:Train%20new%20model%20%E2%80%93%20Vision%20%E2%80%93%20ArTaxOr%20%E2%80%93%20Google%20Cloud%20Platform.png)
# I set it to 20h and let it go...

# ## The resulting model
# AutoML did spend the 20h I set as maximum, which means it was not finished. It is impossible to know if AutoML was close to finish or just getting started. The ArTaxOr dataset is not an easy one, so I assume AutoML would spend quite a few more hours if allowed. Here is the reported precision:
# ![ArTaxOr_dataset%20%E2%80%93%20Vision%20%E2%80%93%20ArTaxOr%20%E2%80%93%20Google%20Cloud%20Platform.png](attachment:ArTaxOr_dataset%20%E2%80%93%20Vision%20%E2%80%93%20ArTaxOr%20%E2%80%93%20Google%20Cloud%20Platform.png)
# Not too bad really for a partly trained model. Now, what kind of neural network is AutoML creating? Let us check by loading the .tflite model into [Netron](https://github.com/lutzroeder/netron):
# ![tflite_model.png](attachment:tflite_model.png)

# The input layers look like this:
# ![model_input_stage.png](attachment:model_input_stage.png)
# While the output layers look like this:
# ![end%20model.png](attachment:end%20model.png)
# > Given that this is an intermediate stage, we would probably get a different architecture if AutoML was allowed to finish.

# AutoML has a nice UI, and after training lets you explore accuracy on the different labels as well as viewing predictions.

# ## Making predictions
# So, we have a (partly) trained model from AutoML, which I have copied to GitHub. Making predictions with images not part of the ArTaxOr dataset is the last step to check the performance. First, fetch the model file from GitHub:

# In[ ]:


import urllib.request

model_url='https://github.com/geddy11/ArTaxOr-models/raw/master/TensorFlow/AutoML/tflite_model-ArTaxOr1.0.0_dataset_20191023_model.tflite'
urllib.request.urlretrieve(model_url, 'automl_trained.tflite')


# Import TensorFlow, PIL and other libraries.

# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

get_ipython().system('pip install python-resize-image')
from PIL import Image, ImageFont, ImageDraw
from resizeimage import resizeimage
import glob, os.path
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Define a few helper functions.

# In[ ]:


def attribution(file):
    with Image.open(file) as img:
        exif_data = img._getexif()
    s='Photo: unknown'
    if exif_data is not None:
        if 37510 in exif_data:
            if len(exif_data[37510]) > 0:
                s = exif_data[37510][8:].decode('ascii')
        if 315 in exif_data:
            if len(exif_data[315]) > 0:
                s = 'Photo: ' + exif_data[315]
    return s

def resize_image(file, width, height, stretch=False):
    with Image.open(file) as im:
        img = im.resize((width, height)) if stretch else resizeimage.resize_contain(im, [width, height])
    img=img.convert("RGB")    
    return img, attribution(file)

fontname = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = ImageFont.truetype(fontname, 20) if os.path.isfile(fontname) else ImageFont.load_default()

def bbox(img, xmin, ymin, xmax, ymax, color, width, label, score):
    draw = ImageDraw.Draw(img)
    xres, yres = img.size[0], img.size[1]
    box = np.multiply([xmin, ymin, xmax, ymax], [xres, yres, xres, yres]).astype(int).tolist()
    txt = " {}: {}%" if score >= 0. else " {}"
    txt = txt.format(label, round(score, 1))
    ts = draw.textsize(txt, font=font)
    draw.rectangle(box, outline=color, width=width)
    if len(label) > 0:
        if box[1] >= ts[1]+3:
            xsmin, ysmin = box[0], box[1]-ts[1]-3
            xsmax, ysmax = box[0]+ts[0]+2, box[1]
        else:
            xsmin, ysmin = box[0], box[3]
            xsmax, ysmax = box[0]+ts[0]+2, box[3]+ts[1]+1
        draw.rectangle([xsmin, ysmin, xsmax, ysmax], fill=color)
        draw.text((xsmin, ysmin), txt, font=font, fill='white')


# In[ ]:


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))

def plot_img_pred(img, axes, scores, boxes, classes, title, by=''):
    for i in range(len(scores)):
        if scores[i]> 0.5 and classes[i]>0:
            label = labels.name.iloc[int(classes[i]-1)]
            color = hex_to_rgb(labels[labels.name == label].color.iloc[0])
            bbox(img, boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2], color, 2, label, 100*scores[i])
    plt.setp(axes, xticks=[], yticks=[])
    axes.set_title(title) if by == '' else axes.set_title(title+'\n'+by)
    plt.imshow(img)
    
def plot_img_gt(img, axes, boxes, stretch, title, by=''):
    wscale = 1. if stretch else min(1,boxes.xres.iloc[0]/boxes.yres.iloc[0])
    hscale = 1. if stretch else min(1,boxes.yres.iloc[0]/boxes.xres.iloc[0])
    for i in range(len(boxes)):
        label = boxes.label.iloc[i]
        color = hex_to_rgb(labels[labels.name == label].color.iloc[0])
        xmin = .5+(boxes.xcenter.iloc[i]-.5)*wscale-.5*wscale*boxes.width.iloc[i]
        ymin = .5+(boxes.ycenter.iloc[i]-.5)*hscale-.5*hscale*boxes.height.iloc[i]
        xmax = .5+(boxes.xcenter.iloc[i]-.5)*wscale+.5*wscale*boxes.width.iloc[i]
        ymax = .5+(boxes.ycenter.iloc[i]-.5)*hscale+.5*hscale*boxes.height.iloc[i]
        bbox(img, xmin, ymin, xmax, ymax, color, 2, label, -1)
    plt.setp(axes, xticks=[], yticks=[])
    axes.set_title(title) if by == '' else axes.set_title(title+'\n'+by)
    plt.imshow(img)


# TensorFlow Lite inference is quite straight forward using the `tf.lite.Interpreter`:

# In[ ]:


interpreter = tf.lite.Interpreter(model_path='automl_trained.tflite')
interpreter.allocate_tensors()
input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()

def predict(img):
    input_data = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num = interpreter.get_tensor(output_details[3]['index'])
    return scores, classes, boxes, num


# Fetch the testset metadata from the starter kernel:

# In[ ]:


pickles='/kaggle/input/starter-arthropod-taxonomy-orders-testset/'
labels=pd.read_pickle(pickles+'testset_labels.pkl')
df=pd.read_pickle(pickles+'testset_filelist.pkl')
anno=pd.read_pickle(pickles+'testset_objects.pkl')


# The labels in this dataset are systematic names rather than common names, so a quick review of their meaning:
# * Coleoptera: Beetles
# * Diptera: True flies, including mosquitoes, midges, crane flies etc.
# * Hymenoptera: Ants, bees and wasps
# * Lepidoptera: Butterflies and moths

# ## Make predictions on true negatives
# Lets start with making predictions on images that have no valid objects. Art and sculptures are not regarded as valid objects in the context of ArTaxOr, which is about species identification. Although false positives are not really a big problem.

# In[ ]:


negs='/kaggle/input/arthropod-taxonomy-orders-object-detection-testset/ArTaxOr_TestSet/negatives/*.jpg'
nlist=glob.glob(negs, recursive=False)
fig = plt.figure(figsize=(16,24))
for i in range(len(nlist)//2):
    for j in range(2):
        axes = fig.add_subplot(len(nlist)//2, 2, 1+i*2+j)
        img, by = resize_image(nlist[i*2+j], 512, 512, stretch=False)
        scores, classes, boxes,_ = predict(img)
        plot_img_pred(img, axes, scores.squeeze(), boxes.squeeze(), classes.squeeze(), 'Prediction', by)


# The model detects a few arty butterflies!  
# ## Make predictions on true positives

# In[ ]:


def pred_batch(idx, stretch):
    fig = plt.figure(figsize=(16,24))
    rows = 3
    for i in range(rows):
        img, by = resize_image(df.path.iloc[i+idx].replace('F:/', 'F:/'), 512, 512, stretch)
        axes = fig.add_subplot(rows, 2, 1+i*2)
        boxes = anno[anno.id == df.id.iloc[i+idx]][['label','xres', 'yres', 'xcenter', 'ycenter', 'width', 'height']]
        plot_img_gt(img, axes, boxes, stretch, 'Ground truth', by)
        img, by = resize_image(df.path.iloc[i+idx].replace('F:/', 'F:/'), 512, 512, stretch)
        scores, classes, boxes,_ = predict(img)
        axes = fig.add_subplot(rows, 2, 2+i*2)
        plot_img_pred(img, axes, scores.squeeze(), boxes.squeeze(), classes.squeeze(), 'Prediction', '') 


# In[ ]:


pred_batch(0, False)


# In[ ]:


pred_batch(3, False)


# In[ ]:


pred_batch(6, False)


# In[ ]:


pred_batch(12, False)


# ## Summary
# So, the model misses quite a few objects, but those it finds are pretty good on IoU. Again, this model training was stopped before it finished, so the result would most likely be better if it ran longer. Considering how easy it is to use AutoML, its definately thumbs up for this product! And in the future we can expect the performance of AutoML to improve dramatically. And no doubt other providers will launch similar services.
