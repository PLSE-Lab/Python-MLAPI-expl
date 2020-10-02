#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf # tensorflow package
import matplotlib.pyplot as plt 
import cv2 # computer vision library
import urllib # package which collect several modules for working with URLs
import os # operating system modules function
from tqdm.notebook import tqdm # progress bar library to support nested loops


# # Data Path

# In[ ]:


pwd


# In[ ]:


cd /kaggle/input/


# In[ ]:


cp-r keras-retinanet /kaggle/working/


# In[ ]:


cd /kaggle/working/keras-retinanet/keras-retinanet-master


# # Install: Keras - Retinanet

# In[ ]:


get_ipython().system('pip install .')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('python setup.py build_ext --inplace')


# In[ ]:


path = '/kaggle/input/global-wheat-detection'
trainFile = path + '/train.csv'
trainDataFile = path + '/train/'
testDataFile = path + '/test/'
testFile = path + '/sample_submission.csv'
names = ['ImgID', 'Width', 'Height', 'bbox', 'Source']
data = pd.read_csv(trainFile, skiprows=1,names=names)
print(data.shape)
data


# In[ ]:


# Seperate bboxs to x1,y1,x2,y2:
data_Frame = pd.DataFrame()
data_Frame['ImgID']=data['ImgID'].apply(lambda x: f'{trainDataFile}{x}.jpg')

# Add x1,y1,x2,y2 to data frame representation:
bbox = data.bbox.str.split(",",expand=True)
data_Frame['x1'] = bbox[0].str.strip('[').astype(float).apply(np.int64)
data_Frame['y1'] = bbox[1].str.strip(' ').astype(float).apply(np.int64)
data_Frame['x2'] = bbox[2].str.strip(' ').astype(float).apply(np.int64)+data_Frame['x1']
data_Frame['y2'] = bbox[3].str.strip(']').astype(float).apply(np.int64)+data_Frame['y1']
data_Frame['class_name'] = 'wheat'
data_Frame , data_Frame.dtypes


# # Visualizing Images With Bboxes

# In[ ]:


# Viusualise the data with bboxes:
imgSel = np.random.RandomState(50) # Generate random numbers drawn from variety of probability.
def show_images_with_box(df):
    """Subplot wheat images including bbox, based on data frame (df) input."""
    size = 3 # Reperesent the number of rows and colums
    fig, axs = plt.subplots(size, size, figsize=(25, 25), sharex=True, sharey=True)
    for row in range(size):
        for col in range(size):
            randomIdx = imgSel.choice(range(df.shape[0])) # Random  row index samples.
            img_name = df.iloc[randomIdx]['ImgID'] # Selecting data based on it's numerical position in the data frame.     
            image = plt.imread(img_name)                        
            # Draw boxes on images by label based selection command(loc):
            selectedImg = df.loc[df["ImgID"]==img_name,["x1","y1","x2","y2"]]
            class_name = 'wheat'
            bboxArray = np.array(selectedImg.values.tolist())
            for bbox in bboxArray:
                image = cv2.rectangle(image, (int(bbox[0]),
                                      int(bbox[1])), (int(bbox[2]),
                                      int(bbox[3])), color = (255,255,255), thickness=3) 
            axs[row, col].imshow(image)
            axs[row, col].axis('off')
            axs[row, col].set_title(f'#{class_name} marked = {bboxArray.shape[0]}',size='xx-large')
            
              
    plt.suptitle(f'{size*size} Random images',size='xx-large')
    plt.show() 
    '\]}'
show_images_with_box(data_Frame)


# # Import Models and function

# In[ ]:


from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color


# In[ ]:


# Coverting the data into the required representation :
data_Frame.to_csv('annotations.csv', index=False, header=None) # Write object to a separated csv file
with open('classes.csv', 'w') as f: # Create a file in writing mode
    f.write('wheat,0\n')
  
# Visualize the data as required by RetinaNet
get_ipython().system('head classes.csv')
get_ipython().system('head annotations.csv')


# In[ ]:



os.makedirs("snapshots", exist_ok=True) # Create a directory
preTrainModel = "./snapshots/_pretrained_model.h5" # Pretrained model name in keras-retinanet ripo
urlModel = "https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5"
# urllib.request.urlretrieve(urlModel, preTrainModel) # Copy a network object denoted by a URL to a local file


# # Train a model

# In[ ]:


#This section creates the train model and includes all parameters.
# Use this section if you need to train your model otherwise, if you all ready train data, please download as the follow section.
flag = False
if(flag):
    get_ipython().system('keras_retinanet/bin/train.py     --freeze-backbone     --random-transform     --weights {preTrainModel}     --batch-size 16     --steps 500     --epochs 10     csv annotations.csv classes.csv')


# In[ ]:



get_ipython().system('ls snapshots # Replicate snapshots')
model_path = os.path.join('/kaggle/input/pre-trained-model', sorted(os.listdir('/kaggle/input/pre-trained-model'), reverse=True)[0])
print(model_path)
# Load the model
model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)

labels_to_names = pd.read_csv('classes.csv', header=None).T.loc[0].to_dict()


# # Pre - detection

# In[ ]:


def pred(image):
#Retruns boxes, scores and label prediction by a given image.
  image = preprocess_image(image.copy())
#   image, scale = resize_image(image)
# Create prediction for each parameters on a single batch sample
  boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

#   boxes /= scale

  return boxes, scores, labels


# In[ ]:


scoreThreshold = 0.34
def drawDetections(image, boxes, scores, labels):
    #The function receive image, boxes, scores and label as input and then draw the detection based on a threshold (scoreThreshold).        
    
  for box, score, label in zip(boxes[0], scores[0], labels[0]):#This loop takes 3 iterable inputs and return them as an iterator (single entity).
    if score < scoreThreshold: #An inner condition in-order to detect wheater the input fit to the threshold.
        break

    color = (255,0,0)
    b = box.astype(int)
    draw_box(image, b, color=color)

    caption = "{} {:.3f}".format(labels_to_names[label], score)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


# In[ ]:


def showDetectedObjects(img_name):
    #The function receive image name and then plot wheat head detection. 
  img_path = testDataFile+img_name
  
  im = read_image_bgr(img_path)

  bx, scr, lb = pred(im)

  imgPlot = im.copy()
  imgPlot = cv2.cvtColor(imgPlot, cv2.COLOR_BGR2RGB)

  drawDetections(imgPlot, bx, scr, lb)
  plt.figure(figsize=(15,10))
  plt.axis('off')
  plt.imshow(imgPlot)
  plt.show()


# # Detection

# In[ ]:


imgs = os.listdir(testDataFile) # Create a list which contain the names of the entries given by the testDataFile.
for idx in imgs:
    showDetectedObjects(idx)


# # Pre - Submission

# In[ ]:


preds = []
imgid =  []
for img in tqdm(imgs,total = len(imgs)):
    predStr = ''
    img_path = testDataFile+img
    im = read_image_bgr(img_path)
    bx, scr, lb = pred(im)
    bx = bx[0]
    scr = scr[0]
    imgid.append(img.split(".")[0])
    for idx in range(bx.shape[0]):
        if scr[idx] > scoreThreshold:
            box,score = bx[idx],scr[idx]            
            predStr += (f'{score:.4} {int(box[0])} {int(box[1])} {int(box[2]-box[0])} {int(box[3]-box[1])} ') # Reshape coordinate to submssion format
    preds.append(predStr)


# In[ ]:


sub = {"image_id":imgid, "PredictionString":preds} # Submition format.
sub = pd.DataFrame(sub)
sub


# # Submission

# In[ ]:


sub.to_csv('/kaggle/working/submission.csv',index=False)

