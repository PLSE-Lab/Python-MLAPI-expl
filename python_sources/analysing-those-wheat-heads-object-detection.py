#!/usr/bin/env python
# coding: utf-8

# # [Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection)
# 
# In this notebook I will try to detect the wheat-heads by training the Faster R-CNN model implemented with PyTorch. I will try to keep this notebook clean, concise & well commented for everyone (including beginners) to understand. 
# 
# 
# 
# So, let's begin the fun... 
# 
# ![](https://i.imgflip.com/12aka8.jpg)

# # Libraries

# In[ ]:


#Generic Packages
import numpy as np
import os
import pandas as pd
import re
from PIL import Image
from IPython.display import display
import gc
from PIL import Image
import warnings 
warnings.filterwarnings("ignore")

#Data Augmentation Libraries
import albumentations as al
from albumentations.pytorch.transforms import ToTensorV2

#PyTorch Libraries
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

#openCV
import cv2  

#Plotting Libraries
import seaborn as sns; sns.set(font_scale=1.4)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

pd.plotting.register_matplotlib_converters()
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')
pd.set_option('display.max_columns', 500)

from tqdm import tqdm
from pathlib import Path


# In[ ]:


# Custom Class & Functions
def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

class WheatDS(Dataset):

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

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
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
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


# # Data Load & Split

# In[ ]:


#Define Directories
input_dir = '../input/global-wheat-detection'
train_dir = f'{input_dir}/train'
test_dir = f'{input_dir}/test'


# In[ ]:


#Train Image Number
img_number = -1 

#Load Training Data
train_data = pd.read_csv(f'{input_dir}/train.csv')

#Add New Columns
train_data['x'] = -1
train_data['y'] = -1
train_data['w'] = -1
train_data['h'] = -1

#Expand BBox
train_data[['x', 'y', 'w', 'h']] = np.stack(train_data['bbox'].apply(lambda x: expand_bbox(x)))

#Drop bbox column
train_data.drop(columns=['bbox'], inplace=True)

#Convert the data type for new columns
train_data['x'] = train_data['x'].astype(np.float)
train_data['y'] = train_data['y'].astype(np.float)
train_data['w'] = train_data['w'].astype(np.float)
train_data['h'] = train_data['h'].astype(np.float)

image_ids = train_data['image_id'].unique()
valid_ids = image_ids[img_number:]
train_ids = image_ids[:img_number]

test_data = train_data[train_data['image_id'].isin(valid_ids)]
train_data = train_data[train_data['image_id'].isin(train_ids)]

print("Validation Dataset Shape: ", test_data.shape, "-- ", "Training Dataset Shape: ",train_data.shape)


# # EDA - Numerical

# In[ ]:


#Inspect
train_data.head()


# In[ ]:


#Data Types
train_data.dtypes


# In[ ]:


#Images per Source
train_data.groupby("source").image_id.count()


# In[ ]:


print('Total Images in Training :', len(os.listdir(train_dir)))
print('Total Images in Test:', len(os.listdir(test_dir)))


# In[ ]:


#Unique Images
print("Unique Training Images: ", train_data['image_id'].nunique())


# # EDA - Visual

# In[ ]:


#Visualize the Image Distribution across the different Sources
plt.rcParams['figure.figsize'] = (10, 8)
sns.countplot(train_data['source'], palette='pastel')
plt.title('Image Distribution Across the Different Sources', fontsize= 15)
plt.legend()
plt.show()


# In[ ]:


# Function to view images from dataset
def view_images(image_ids):
    col = 5
    row = min(len(image_ids) // col, 5)
    fig, ax = plt.subplots(row, col, figsize=(16, 8))
    fig.suptitle('Random Images of Wheat Heads from Training Dataset', fontsize=15)
    ax = ax.flatten()
    

    for i, image_id in enumerate(image_ids):
        image = cv2.imread('../input/global-wheat-detection/train' + '/{}.jpg'.format(image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax[i].set_axis_off()
        ax[i].imshow(image)

#Running the function
view_images(train_data.sample(n=10)['image_id'].values)


# # Simple Object Detection 

# In[ ]:


# Function to read bbox and plot the image with the detected wheat heads

def get_bboxes(df, image_id):
    image_bboxes = df[df.image_id == image_id]
    
    bboxes = []
    for _,row in image_bboxes.iterrows():
        bboxes.append((row.x, row.y, row.w, row.h))
        
    return bboxes

def plot_image(df, rows=3, cols=3, title='Wheat Head Detection in Training Images'):
    fig, axs = plt.subplots(rows, cols, figsize=(10,10))
    for row in range(rows):
        for col in range(cols):
            idx = np.random.randint(len(df), size=1)[0]
            img_id = df.iloc[idx].image_id
            
            img = Image.open(train_dir + '/' + img_id + '.jpg')
            axs[row, col].imshow(img)
            
            bboxes = get_bboxes(df, img_id)
            
            for bbox in bboxes:
                rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')
                axs[row, col].add_patch(rect)
            
            axs[row, col].axis('off')
            
    plt.suptitle(title)


# In[ ]:


#Plotting the training image with detected wheat heads 
plot_image(train_data)


# # Build Model

# In[ ]:


# load a model; pre-trained on COCO

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=False)
model.load_state_dict(torch.load("../input/torchvisionfasterrcnn/fasterrcnn_resnet50_fpn.pth"))

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# ## Data Augmentation

# In[ ]:


# Albumentations
def get_train_transform():
    return al.Compose([
        al.RandomSizedBBoxSafeCrop(512, 512, erosion_rate=0.0, interpolation=1, p=1.0),
        al.HorizontalFlip(p=0.5),
        al.VerticalFlip(p=0.5),
        al.RandomContrast(p=1.0),
        al.RandomGamma(p=1.0),
        al.RandomBrightness(p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return al.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# # Dataset Creation

# In[ ]:


def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = WheatDS(train_data, train_dir, get_train_transform())
test_dataset = WheatDS(test_data, train_dir, get_valid_transform())


# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
#lr_scheduler = None

num_epochs = 9

loss_hist = Averager()
itr = 1000


# # Train the Model

# In[ ]:


for epoch in range(num_epochs):
    loss_hist.reset()
    
    for images, targets, image_ids in train_data_loader:
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 25 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1
    
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_hist.value}")


# # Submission

# In[ ]:


#Function for making predictions
def prediction(image_path,model,device):
    model.eval()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    images = torch.from_numpy(image).float().permute(2,0,1).unsqueeze(0).to(device)
    outputs = model(images)


    outputs = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in outputs]
    boxes = outputs[0]["boxes"]
    scores = outputs[0]["scores"]
    valid_boxes = boxes[scores > 0.5]
    valid_scores = scores[scores > 0.5]
    return valid_boxes, valid_scores


# In[ ]:


#Inspecting the sample submission
submission = pd.read_csv(f'{input_dir}/sample_submission.csv')
submission.head()


# In[ ]:


sub_dir = '../input/global-wheat-detection/test'

submission = pd.read_csv(f'{input_dir}/sample_submission.csv')



root_image = Path(test_dir)
test_images = [root_image / f"{img}.jpg" for img in submission.image_id]


submission = []
model.eval()

for image in tqdm(test_images):
    boxes, scores = prediction(str(image),model,device)
    prediction_string = []
    for (x_min,y_min,x_max,y_max),s in zip(boxes,scores):
        x = round(x_min)
        y = round(y_min)
        h = round(x_max-x_min)
        w = round(y_max-y_min)
        prediction_string.append(f"{s} {x} {y} {h} {w}")
    prediction_string = " ".join(prediction_string)
    
    submission.append([image.name[:-4],prediction_string])

sample_submission = pd.DataFrame(submission, columns=["image_id","PredictionString"])
sample_submission.to_csv('submission.csv', index=False)


# ## Please upvote if you find it helpful :-)
