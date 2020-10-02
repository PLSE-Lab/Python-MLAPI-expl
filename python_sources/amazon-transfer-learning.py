#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np # Torch wrapper for Numpy

import os
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.preprocessing import MultiLabelBinarizer


# In[ ]:


IMG_PATH = '../input/train-jpg/'
IMG_EXT = '.jpg'
TRAIN_DATA = '../input/train_v2.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


class KaggleAmazonDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None):
    
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(),         "Some images referenced in the CSV file were not found"
        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)


# In[ ]:


transformations = transforms.Compose([transforms.Scale(32), 
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                                       transforms.Normalize(mean=[-1,-1,-1],std=[2,2,2])
                                     ])

dataset = KaggleAmazonDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations)


# In[ ]:


from torch.utils.data.sampler import SubsetRandomSampler

batch_size = 256
validation_split = .2
shuffle_dataset = True
random_seed= 42

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)


# In[ ]:


train_loader = DataLoader(dataset,
                          batch_size=256,
                          sampler=train_sampler,
                          num_workers=1, # 1 for CUDA
                          pin_memory=True # CUDA only
                         )

test_loader = DataLoader(dataset,
                          batch_size=256,
                          sampler=valid_sampler,
                          num_workers=1, # 1 for CUDA
                          pin_memory=True # CUDA only
                         )


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

image, label = next(iter(train_loader.dataset))

plt.imshow(image[1])


# In[ ]:


class AmazonModel(nn.Module):
    def __init__(self, pretrained_model, in_features, out_features):
        super(AmazonModel, self).__init__()
        
        self.pretrained_model = pretrained_model
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features,out_features)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x


# In[ ]:


from torchvision import models

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.out_features
no_label = len(train_loader.dataset.y_train[0])

model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True

model = AmazonModel(model, num_ftrs, no_label)
model = model.to(device)

torch.backends.cudnn.benchmark=True

# for name, param in model.named_parameters():
#     print(name, param.requires_grad)


# In[ ]:


model.train()


# In[ ]:


optimizer = optim.Adam(model.parameters(), lr=0.003)


# In[ ]:


from sklearn import metrics

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
#         data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset) - split,
                100. * batch_idx / len(train_loader), loss.data.item()))


# In[ ]:


for epoch in range(2):
    train(epoch)


# In[ ]:


from sklearn.metrics import fbeta_score, confusion_matrix

f_scores = list()

def test(epoch):
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
#         data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        if batch_idx % 10 == 0:
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), split,
                100. * batch_idx / len(test_loader), loss.data.item()))
            f_scores.append([target, output])
            


# In[ ]:


# test(1)


# In[ ]:


for t, o in f_scores:
    print(fbeta_score(t.cpu().detach().numpy(), np.where(o.cpu().detach().numpy() > 0.5, 1, 0), beta=2, average='samples'))


# In[ ]:


# from sklearn.metrics import precision_score

# for t, o in f_scores:
#     print(precision_score(t.cpu().detach().numpy(), np.where(o.cpu().detach().numpy() > 0.5, 1, 0)))


# In[ ]:


# x = test_loader.dataset[40405][0].unsqueeze(0).to(device)
# y = test_loader.dataset[40405][1]

# model.eval()
# y_ = model(x)
# n = 6
# print(y_, torch.topk(y_, n)[1][0])
# print(y, torch.topk(y, n)[1])


# In[ ]:


# (dataset.mlb.inverse_transform(np.where(y_.cpu().detach().numpy() > 0.5, 1, 0))[0])


# In[ ]:


os.listdir('../input/')


# In[ ]:


import os

TEST_IMG_PATH = '../input/test-jpg-v2/'
TEST_IMG_EXT = '.jpg'
SUBMISSION_FILE = '../input/sample_submission_v2.csv'
SUB_MAPPING = '../input/test_v2_file_mapping.csv'

sub = pd.read_csv(SUBMISSION_FILE)
# os.listdir('../working/')
for index in range(len(sub)):
    img = Image.open(TEST_IMG_PATH + sub['image_name'][index] + TEST_IMG_EXT)
    img = img.convert('RGB')
    img = transformations(img)
    Y_ = model(img.unsqueeze(0).to(device))
    sub['tags'][index] = ' '.join(list(dataset.mlb.inverse_transform(np.where(Y_.cpu().detach().numpy() > 0.5, 1, 0))[0]))
    if index % 10 == 0:
        print('{} Files Completed!')
            
# sub.head()


# In[ ]:


sub.to_csv('submission_resnet18.csv', index = False)
os.listdir('../working/')


# # Thank you for your attention
# 
# Hopefully that will help you get started. I still have a lot to figure out in PyTorch like:
# 
# * Implementing the train / validation split
# * Figure out data augmentation (and not just random transformations or images)
# * Implementing early stopping
# * Automating computation of intermediate layers
# * Improving the display of each epochs
# 
# If you liked the kernel don't forget to vote and don't hesitate to comment.

# ## Full code
# 
# I have published my full code of the competition in my [GitHub](https://github.com/mratsim/Amazon_Forest_Computer_Vision). (Note: I only worked on it in the first 2 weeks, so it probably lacks the latest findings)
# 
# You will find:
#   - [A script that output the mean and stddev of your image if you want to train from scratch](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/compute-mean-std.py#L28)
# 
#   - [Using weighted loss function](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/main_pytorch.py#L61)
# 
#   - [Logging your experiment](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/main_pytorch.py#L89)
# 
#   - [Composing data augmentations](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/main_pytorch.py#L103), also [here](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p_data_augmentation.py#L181).
# Note use [Pillow-SIMD](https://python-pillow.org/pillow-perf/) instead of PIL/Pillow. It is even faster than OpenCV
# 
#   - [Loading from a CSV that contains image path - 61 lines yeah](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p2_dataload.py#L23)
# 
#   - [Equivalent in Keras - 216 lines ugh](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/k_dataloader.py). Note: so much lines were needed because by default in Keras you either have the data augmentation with ImageDataGenerator or lazy loading of images with "flow_from_directory" and there is no flow_from_csv
# 
#   - [Model finetuning with custom PyCaffe weights](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p_neuro.py#L139)
# 
#   - Train_test_split, [PyTorch version](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p_model_selection.py#L4) and [Keras version](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/k_model_selection.py#L4)
# 
# - [Weighted sampling training so that the model view rare cases more often](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/main_pytorch.py#L131-L140)
# 
#  - [Custom Sampler creation, example for the balanced sampler](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p_sampler.py)
# 
#  - [Saving snapshots each epoch](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/main_pytorch.py#L171)
# 
#  - [Loading the best snapshot for prediction](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/pytorch_predict_only.py#L83)
# 
#  - [Failed word embeddings experiments](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/Embedding-RNN-Autoencoder.ipynb) to [combine image and text data](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/Dual_Feed_Image_Label.ipynb)
# 
#  - [Combined weighted loss function (softmax for unique weather tags, BCE for multilabel tags)](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p2_loss.py#L36)
# 
#  - [Selecting the best F2-threshold](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p2_metrics.py#L38) via stochastic search at the end of each epoch to [maximize validation score](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/526128239a6abcbb32fbf5b34ed8cc7a3cd87c4e/src/p2_validation.py#L49). This is then saved along model parameter.
# 
#   - [CNN-RNN combination (work in progress)](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p3_neuroRNN.py#L10)
