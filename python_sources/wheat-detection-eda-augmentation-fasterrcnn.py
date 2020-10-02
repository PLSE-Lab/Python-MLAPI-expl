#!/usr/bin/env python
# coding: utf-8

# # <center><h1>Wheat Spike Detection<h1></center>
# 
# ![](https://cdn.glutenfreeliving.com/2015/01/wheat-starch-image-825x338.jpg)
# 
# Machine learning has a lot of applications in various industries. The recent development in technology has also enabled ML to step into the realm of agriculture. Image analysis has significantly enhanced the potential for achieving high-throughput analysis of crop fields. And has also enabled in detecting diseases in the crops at a very initial stage.  For wheat breeding purposes, assessing the production of wheat spikes, as the grain-bearing organ, is a useful proxy measure of grain production. Thus, being able to detect and characterize spikes from images of wheat fields is an essential component in a wheat breeding pipeline for the selection of high yielding varieties.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# import required libraries
import numpy as np


import PIL
from PIL import Image

# plotly libraries
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.express as px
import plotly.io as pio

import cv2


# In[ ]:


# initialize source paths
source_path = '/kaggle/input/global-wheat-detection/'
train_path = source_path + 'train/'
test_path = source_path + 'test/'

# read data table
train_df = pd.read_csv(source_path +'train.csv')


# # <center><h1>Basic Data Exploration<h1></center>
#     
#     
#  First we will look at the data source to get an intial understanding of what is there. It will help us to know how many images are there, what are the sources for the data, image sizes, bounding box information etc.

# In[ ]:


train_df.head()


# As a first step we will create the individual bbox columns and also create the bounding box area column. It will be helpful if we take care of it now since we have to analyze this later.
# 

# In[ ]:


# expand the bbox column into seprate columns
train_df[['bbox_xmin','bbox_ymin','bbox_width','bbox_height']] = train_df['bbox'].str.split(',',expand=True)
train_df['bbox_xmin'] = train_df['bbox_xmin'].str.replace('[','').astype(float)
train_df['bbox_ymin'] = train_df['bbox_ymin'].str.replace(' ','').astype(float)
train_df['bbox_width'] = train_df['bbox_width'].str.replace(' ','').astype(float)
train_df['bbox_height'] = train_df['bbox_height'].str.replace(']','').astype(float)

# add xmax, ymax, and area columns for bounding box
train_df['bbox_xmax'] = train_df['bbox_xmin'] + train_df['bbox_width']
train_df['bbox_ymax'] = train_df['bbox_ymin'] + train_df['bbox_height']
train_df['bbox_area'] = train_df['bbox_height'] * train_df['bbox_width']


# ## How are the images distributed by source?

# In[ ]:


# count distinct images by source
img_source_dist = train_df.groupby(['source']).agg(image_count=('image_id','nunique'),wheat_head=('image_id','size'))
img_source_dist.reset_index(inplace=True,drop=False)
fig = px.pie(img_source_dist, values='image_count', names='source', title='Spike Distribution by Source')
fig.show()


# ## How about 'em wheat heads?

# In[ ]:


img_source_dist['Avg_Wheat_Head'] = img_source_dist['wheat_head']/img_source_dist['image_count']
img_source_dist = img_source_dist.sort_values(by='Avg_Wheat_Head', ascending=True)

fig = go.Figure(data=[
    go.Bar(name='Avg Wheat Head Count', x=img_source_dist['Avg_Wheat_Head'], y=img_source_dist['source'],
           orientation='h',marker_color='salmon')
])
# Change the bar mode
fig.update_layout(title_text='Avg Number of Spikes',
                  height=400)
fig.show()


# The ethz_1 source has a significant number of wheat heads per image, approximately 69 per image. Maybe these are very small and will be highly concentrated. We will know better once we look at the images. Now let's look at the overall distribution for the wheat heads by image.

# In[ ]:


wheat_heads_per_image = train_df.groupby('image_id').agg(head_count=('image_id','size'))
wheat_heads_per_image.reset_index(inplace=True, drop=False)

fig = px.histogram(wheat_heads_per_image, x="head_count",marginal="box")
fig.update_layout(
    xaxis = dict(
        title_text = "Spike Count"), title = 'Spike Count per Image')
fig.show()


# The IQR is between 28 and 59 wheat heads per image, with a median value of 43. There are some images with a counts greater than 100 and as less as 1. We will look at images in these 3 ranges.

# In[ ]:


# create list of images per the regions identified 
heads_large =  list(wheat_heads_per_image[wheat_heads_per_image.head_count > 100]['image_id'].unique())
heads_normal = list(wheat_heads_per_image[(wheat_heads_per_image.head_count >= 30) & (wheat_heads_per_image.head_count <= 30)]['image_id'].unique())
heads_small =  list(wheat_heads_per_image[wheat_heads_per_image.head_count <= 5]['image_id'].unique())


# In[ ]:


# define a function to display the images
def get_bbox(df, image_id):
    bboxes = []
    image_bbox = df[df.image_id == image_id]
    
    for _,row in image_bbox.iterrows():
        bboxes.append([row.bbox_xmin, row.bbox_ymin, row.bbox_width, row.bbox_height])
        
    return bboxes
        
def plot_image(images, title=None):
    fig = plt.figure(figsize = (20,10))
    for i in range(1,4):
        ax = fig.add_subplot(1, 3, i)
        img = np.random.choice(images)
        image_path = os.path.join(train_path,img +'.jpg')
        image = Image.open(image_path)
        ax.imshow(image)
    
        b = get_bbox(train_df,img)
    
        for bbox in b:
                    rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='yellow',facecolor='none')
                    ax.add_patch(rect)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()


# In[ ]:


plot_image(heads_small,'Spike Count <= 5')


# For the images with less than 5 wheat heads we see these are very small heads. And Some of them are on the ground and a few are not clearly visbible. We also have heads with the shoe background. 
# 
# Also an other thing to note is there is a clear demarcation in the brightness of the pictures. Some of them are dark and some of them are light. We will have to use some augmentation techniques for these.

# In[ ]:


plot_image(heads_normal,'Spike Count >= 30 & <= 60')


# The images with count between 30 and 60 look good. The images seem sharper, but we need to look at more of these and also analyze those parameters.

# In[ ]:


plot_image(heads_large,'Spike Count > 100')


# Wow!! those look cluttered , but they are a good source of information.

# # Looking into the box...
# In this section we will look at the bounding boxes. Specifically, the area distribution and how many of these narrow boxes we have in the dataset. The area becomes important for IOU detection.

# In[ ]:


fig = px.histogram(train_df, x="bbox_area",marginal="box")
fig.update_layout(
    xaxis = dict(
        title_text = "Bounding Box Area"), title = 'Bounding Box Area Distribution')
fig.show()


# The bounding box area distribution is extremely skewed. We have some exteremly large bounding boxes with area greater than 100,000. The maximum area is approximately 500,000. As stated earlier, these will have an impact on the IOU and may have to be dealt with.
# 
# The smallest are of the bounding box is 2.0. This is could be one of those wheat heads that is still forming. Nevertheless let's a take a look at some of the images with both types of bounding boxes.

# In[ ]:


large_area = list(train_df[train_df.bbox_area > 100000]['image_id'].unique())
small_area = list(train_df[train_df.bbox_area <= 10]['image_id'].unique())


# In[ ]:


plot_image(large_area, title='Large Bounding Boxes')


# There are some really large boxes that don't make a lot sense!

# In[ ]:


plot_image(small_area, title='Small Bounding Boxes')


# # <center><h1>Image Augmentation<h1></center>
#     
# We have only about 3,300 images in the training data. Deep neural networks need a lot of data to be effective. That is where image augmentation comes into play. It is the process of creating more images from the existing training data by applying transformations. These include, but not limited to, flips, adding blur, increase sharpness and more. Some of these are very helpful to increase the accuracy of the models.
# 
# There is a fantastic library called 'Albumentations' that helps in creating augmenations quickly and effectively within a few lines of code. The github link is [here](https://github.com/albumentations-team/albumentations) and is a really good resource for beginners. The library was created by Kaggle Grandmasters and has helped win Kaggle competitions.
# 
# Since this is my first Kaggle competition and I have not really done object detection analysis, I am going to use the Albumentation library to play around a little bit to understand the different features available.
# 

# In[ ]:


import albumentations
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose,VerticalFlip,BboxParams,Rotate, ChannelShuffle, RandomRain)


# In[ ]:


# define functions for augmentation and display

def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params=BboxParams(format='pascal_voc', min_area=min_area, 
                                               min_visibility=min_visibility, label_fields=['labels']))

BOX_COLOR = (255,255,0)
def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2):
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)   
    return img

def visualize(annotations):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox)
    return img 
    
def aug_plots(image_id, aug, title=None):
    img_path = os.path.join(train_path,image_id +'.jpg')
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    bbox = train_df[train_df['image_id'] == image_id][['bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax']].astype(np.int32).values

    labels = np.ones((len(bbox), ))
    annotations = {'image': image, 'bboxes': bbox, 'labels': labels}
    
    aug = get_aug(aug)
    augmented = aug(**annotations)
    visualize(augmented)
    
    fig = plt.figure(figsize = (15,7))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(visualize(annotations))
    plt.title('Original')
    
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(visualize(augmented))
    plt.title(title)


# In[ ]:


aug_plots('b6ab77fd7',[VerticalFlip(p=1)], 'Vertical Flip')


# In[ ]:


aug_plots('69fc3d3ff',[HorizontalFlip(p=1)], 'Horizontal Flip')


# In[ ]:


aug_plots('69fc3d3ff',[Blur(blur_limit= 7,p=0.5)], 'Blur')


# In[ ]:


aug_plots('69fc3d3ff',[Rotate(p=0.5)], 'Rotate')


# In[ ]:


aug_plots('69fc3d3ff',[HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1)], 'Hue Saturation')


# This is a good one. The spikes stand out!

# In[ ]:


aug_plots('69fc3d3ff',[ChannelShuffle(p=1)], 'Channel Shuffle')


# In[ ]:


aug_plots('69fc3d3ff',[GaussNoise()], 'Gauss Noise')


# In[ ]:


aug_plots('69fc3d3ff',[RandomRain(p=1, brightness_coefficient=0.9, drop_width=1, blur_value=5)], 'Random Rain')


# This is really cool...adding weather aspects to the data.
# 
# I can keep playing with this all day :-) But it is time to move on!

# # <center><h1>Train Model<h1></center>
#     
# We will train a FasterRCNN model for this analysis. I am very new to DL and object detection. I am going to try my best to explain the proces in the most layman terms.
# 
# R-CNN extracts a bunch of regions from the given image using selective search, and then checks if any of these boxes contains an object. We first extract these regions, and for each region, CNN is used to extract specific features. Finally, these features are then used to detect objects. Unfortunately, R-CNN becomes rather slow due to these multiple steps involved in the process.
# 
# Fast R-CNN, on the other hand, passes the entire image to ConvNet which generates regions of interest (instead of passing the extracted regions from the image). Also, instead of using three different models (as we saw in R-CNN), it uses a single model which extracts features from the regions, classifies them into different classes, and returns the bounding boxes.
# 
# All these steps are done simultaneously, thus making it execute faster as compared to R-CNN. Fast R-CNN is, however, not fast enough when applied on a large dataset as it also uses selective search for extracting the regions.
# 
# Faster R-CNN fixes the problem of selective search by replacing it with Region Proposal Network (RPN). We first extract feature maps from the input image using ConvNet and then pass those maps through a RPN which returns object proposals. Finally, these maps are classified and the bounding boxes are predicted.
# 
# Below steps sumarize the steps:
# 
# * Take an input image and pass it to the ConvNet which returns feature maps for the image
# * Apply Region Proposal Network (RPN) on these feature maps and get object proposals
# * Apply ROI pooling layer to bring down all the proposals to the same size
# * Finally, pass these proposals to a fully connected layer in order to classify any predict the bounding boxes for the image
# 
# The Faster R-CNN model will be implemented with PyTorch. The first step is to write a function to prepare the dataset.

# In[ ]:


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler


# The data prep step in PyTorch requires three main parameters:
# * an image with size h and w. The minimum size required is a 800 x 800 image.
# * a target dictionary that needs the following mandatory fields:
#         1. coordinates of the bounding boxes
#         2. labels for each bounding box - background is always 0.
#         3. image id - unique identifier
#         4. area - area of the bounding box
#         5. iscrowd - instacnces with iscrowd = True will be ignored (i don't know what this means and have set it to False
#         
# In addition we can add masks if available and also specify transformations. 
# 
# The function below will do all of the above.

# In[ ]:


class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax']].values
        
        area = records['bbox_area'].values  # i already have the area in my dataframe
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class - so all will be 1
        labels = torch.ones((records.shape[0],), dtype=torch.int64) 
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.tensor(sample['bboxes'])
            target['boxes'] = target['boxes'].type(torch.float32)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]


# We have already seen that the albumentations library is a good resource. So we will start only with a flip for the train. Note that the validation set should not be augmented!

# In[ ]:


# define transformation functions
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# There are two common situations where one might want to modify one of the available models in torchvision modelzoo. The first is when we want to start from a pre-trained model, and just finetune the last layer. The other is when we want to replace the backbone of the model with a different one (for faster predictions, for example).
# 
# In this case we will use the pre-trained model and finetune the last layer since our dataset is not that large.

# In[ ]:


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# Next, we split the data into a train_set (2,708 images) and a validation set (665 images). 

# In[ ]:


image_ids = train_df['image_id'].unique()
valid_ids = image_ids[-665:]
train_ids = image_ids[:-665]

val_set = train_df[train_df['image_id'].isin(valid_ids)]
train_set = train_df[train_df['image_id'].isin(train_ids)]


# Next, we define the dataloaders. These steps are pretty self explanatory.

# In[ ]:


def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = WheatDataset(train_df, train_path, get_train_transform())
valid_dataset = WheatDataset(valid_df, train_path, get_valid_transform())


# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)


# Now, we setup the model.

# In[ ]:


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# define the number of classes
num_classes = 2 # one for wheat and one for background

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005) 


num_epochs = 4

#for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    #train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
    # update the learning rate
    #lr_scheduler.step()
    # evaluate on the test dataset
   # evaluate(model, valid_data_loader, device=device)


# In[ ]:


from engine import train_one_epoch

