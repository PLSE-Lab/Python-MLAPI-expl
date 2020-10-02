#!/usr/bin/env python
# coding: utf-8

# ## Skorch training baseline for Pandas
# 
# This is a simple training baseline for the challenge: [https://www.kaggle.com/c/prostate-cancer-grade-assessment](http://).
# Skorch introduces scikit-learn like functionality for PyTorch
# 
# I used reseized pandas dataset of size (512 x 512) which did not yeild any good results but I thought of sharing it anyway.
# I have also shareed the training files. 
# 
# Hope u find it useful! Upvote if u like it ! 

# In[ ]:


get_ipython().system('pip install -U skorch')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

# torch imports
import torch
from torchvision.transforms import transforms
import torch.optim as optim
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset


# skorch imports
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, Checkpoint, Freezer
from skorch.helper import predefined_split

import os

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('writefile', 'config.py', "\ntrain_dir = '../input/pandaresizeddataset512x512/train/'\ntest_dir = '../input/pandaresizeddataset512x512/test_images/'\ntrain_csv = '../input/prostate-cancer-grade-assessment/train.csv'\ntest_csv = '../input/prostate-cancer-grade-assessment/test.csv'\nbatch_size = 32")


# In[ ]:


get_ipython().run_cell_magic('writefile', 'data.py', "\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\nimport cv2\nimport config\n\n# torch imports\nimport torch\nfrom torch.utils.data import Dataset\n\nclass PandaDataset(Dataset):\n    def __init__(self, csv_file, transform=None):\n        self.df = pd.read_csv(csv_file)\n        self.transform = transform\n\n    def __getitem__(self, index):\n        image_ids = self.df['image_id'].values\n        labels = self.df['isup_grade'].values\n\n        image = cv2.imread(config.train_dir + image_ids[index] + '.png')\n        label = labels[index]\n\n        if self.transform:\n            image = self.transform(image)\n\n        image = image.clone().detach()\n        label = torch.tensor(label)\n\n        return image, label\n\n    def __len__(self):\n        return len(self.df)")


# In[ ]:


get_ipython().run_cell_magic('writefile', 'engine.py', "\n# torch imports\nimport torch\nfrom torchvision.transforms import transforms\nfrom torch.utils.data import Dataset, random_split\n\nfrom data import PandaDataset\nimport config\n\nclass Engine:\n    def __init__(self):\n        self.transforms = transforms.Compose(\n            [\n                transforms.ToPILImage(),\n                transforms.RandomHorizontalFlip(),\n                transforms.ToTensor()\n            ]\n        )\n        self.train_loss = []\n        self.loss_val = []\n\n    def create_data_loaders(self):\n        dataset = PandaDataset(config.train_csv, transform=self.transforms)\n\n        train_size = int(0.8 * len(dataset))\n        test_size = len(dataset) - train_size\n        train_dataset, valid_dataset = random_split(dataset, [train_size, test_size])\n\n        image_datasets = {\n            'train': train_dataset,\n            'validation': valid_dataset\n        }\n\n        return image_datasets")


# In[ ]:


get_ipython().run_cell_magic('writefile', 'train.py', "\n# torch imports\nimport torch\nimport torch.optim as optim\nimport torchvision\nimport torch.nn as nn\nfrom torch.optim.lr_scheduler import CyclicLR\n\n# skorch imports\nfrom skorch import NeuralNetClassifier\nfrom skorch.callbacks import LRScheduler, Checkpoint, Freezer\nfrom skorch.helper import predefined_split\n\nfrom engine import Engine\nfrom data import PandaDataset\nimport config\n\nclass PretrainedModel(nn.Module):\n    def __init__(self, output_features):\n        super().__init__()\n        model = torchvision.models.densenet121(pretrained=True)\n        num_ftrs = model.classifier.in_features\n        model.classifier = nn.Linear(num_ftrs, output_features)\n        self.model = model\n\n    def forward(self, x):\n        return self.model(x)\n\n# print(PretrainedModel(6))\n# exit(0)\ndatasets = Engine().create_data_loaders()\n\nlrscheduler = LRScheduler(\n    policy='StepLR',\n    step_size=7,\n    gamma=0.1\n)\n\ncheckpoint = Checkpoint(\n    f_params='densenet_skorch.pt',\n    monitor='valid_acc_best'\n)\n\nfreezer = Freezer(lambda x: not x.startswith('model.classifier'))\n\nnet = NeuralNetClassifier(\n    PretrainedModel,\n    criterion=nn.CrossEntropyLoss,\n    batch_size=config.batch_size,\n    max_epochs=5,\n    module__output_features=6,\n    optimizer=optim.SGD,\n    iterator_train__shuffle=True,\n    iterator_train__num_workers=4,\n    iterator_valid__shuffle=True,\n    iterator_valid__num_workers=4,\n    train_split=predefined_split(datasets['validation']),\n    callbacks=[lrscheduler, checkpoint, freezer],\n    device='cuda'  # comment to train on cpu\n)\n\n\n#start training\nnet.fit(datasets['train'], y=None)")


# In[ ]:


get_ipython().system('python train.py')


# In[ ]:




