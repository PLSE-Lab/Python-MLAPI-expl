#!/usr/bin/env python
# coding: utf-8

# ### Important Note
# This is just a reference model for the SIIM Pneumothorax Segmentation competition. The directories referenced in the code below are pointing to a **very** small sample of the training and test sets. Therefore, this kernel, as is, will not yield a good result. This is intended to be a framework, so you should expect to have to replace those directories with the actual train & test sets, once you've retrieved them [using the tutorial specified](https://storage.googleapis.com/kaggle-media/competitions/siim/SIIM%20Cloud%20Healthcare%20API%20Documentation.pdf).

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
print(os.listdir("../input/siim-acr-pneumothorax-segmentation/sample images"))
print(os.listdir("../input/example-test-images"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#----------------------------------------------------#
#  Created by Shunxing Bao.                          #
#  https://www.kaggle.com/onealbao                   #
#----------------------------------------------------#

# define DICE coefficient metric calculation 
import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    return 0


# In[ ]:


#----------------------------------------------------#
#  Created by Shunxing Bao.                          #
#  https://www.kaggle.com/onealbao                   #
#----------------------------------------------------#

# customized Data loader
# We recommend users firstly convert DICOM to JPEG, then load JPEG images in dataloader
from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import os
import nibabel as nib

class customDataLoader(Dataset):
    def __init__(self, img_list, seg_list, state, num_classes): #READ DATA
        self.img_list = img_list
        self.seg_list = seg_list
    
    def __getitem__(self, index): # RETURN ONE ITEM ON THE INDEX
        return 0
    
    def __len__(self): # RETURN THE DATA LENGTH
        return len(self.img_list)
        
        


# In[ ]:


#----------------------------------------------------#
#  Created by Shunxing Bao.                          #
#  contact: shunxing.bao@vanderbilt.edu              #
#----------------------------------------------------#

# a dummy Unet
import torch
import torch.nn as nn

class dummyUNet(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(dummyUNet, self).__init__()
        
    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        return None

    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        return None

    def forward(self, x):
        return x


# In[ ]:


#----------------------------------------------------#
#  Created by Shunxing Bao.                          #
#  https://www.kaggle.com/onealbao                   #
#----------------------------------------------------#

import os
import time
import datetime
import numpy as np
import random
import torch
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import nibabel as nib


def train(epoch, num_epochs, net, train_loader):
    mean_val_loss = []
    return mean_val_loss
    # End of train function

def main():
    print('::......STARTING SIIM SEGMENTATION ON MMWHS DATASET......::\n\n')

    # parameters setting
    num_epochs = 350
    last_epoch = 0
    in_channels = 1 # data per pixel
    n_classes = 1 # need to further set correctly ...

########################################################################################################################
    # DATA LOADING
########################################################################################################################

    print('Building Dataset....')

    # Define input training data and labels directories

    # TRAINING IMAGE PATHS
    train_data_path = os.path.normpath('../input/siim-acr-pneumothorax-segmentation/sample images/')
    # Label CSV Data Path
    train_labels_path = os.path.normpath('../input/siim-acr-pneumothorax-segmentation/sample images/train-rle-sample.csv')

    # Prepare training dataset for dataloading
    train_dataloader_input = customDataLoader(train_data_path, train_labels_path, 1, n_classes)

    # Training Data loader (Using built in Torch class Dataset, refer to customDataLoader)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataloader_input,
                                               batch_size=1,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=4) #

    print('....Dataset built!')

##############################################################s##########################################################
    # SETUP NEURAL NETWORK, LOSS FUNCTION
########################################################################################################################
    print('Initializing model...')

    # Automatically choose GPU when available, otherwise CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load model to device..
    net = dummyUNet(in_channels, n_classes)
    net.to(device)

    # Define loss criterion..
    # loss_criterion = DiceLoss()

########################################################################################################################
    # PERFORM NETWORK TRAINING AND SAVE LOSS DATA
########################################################################################################################

    print('\n===============================TRAINING BEGINS===============================')
    
    # Parse through epochs
    for epoch in range(last_epoch, num_epochs):
        train(epoch, num_epochs, net, train_loader)
    
    print('\n===============================TRAINING COMPLETE=============================')

if __name__ == '__main__':
    main()


# In[ ]:


#----------------------------------------------------#
#  Created by Shunxing Bao.                          #
#  https://www.kaggle.com/onealbao                   #
#----------------------------------------------------#

import os
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.backends.cudnn
from shutil import copyfile

def test():

    print('====================TESTING SIIM SEGMENTATION====================')
    print('Initializing Neural Network........................')
    # Automatically choose GPU when available, otherwise CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print('Device in use: ', device)

    in_channels = 1  # data per pixel
    n_classes = 1  # need to further set correctly

    # Define network
    net = dummyUNet(in_channels, n_classes)

    # change processing device to cpu
    net.eval()

    # Load old weights
    weights_path = '/PATH_TO_FINAL_WEIGHT.pth'
    #net.load_state_dict(torch.load(weights_path, map_location=device))
    print('....dummyUnet Network loaded with weights')

    # Build dataset for loading into network!!
    print('Building testing dataset......')
    #test_data_path = os.path.normpath('./sanbdox/Testing/img/')
    test_data_path = '../input/example-test-images'
    # Prepare validation dataset for dataloading
    test_dataloader_input = customDataLoader(test_data_path, "", 0, n_classes)

    # Testing Data loader (Using built in Torch class Dataset, refer to dataset.py)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataloader_input,
                                              batch_size=1,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=4)  
    print('...............Dataset Built!')
    print('===============================TESTING BEGINS===============================')
    
    mkdir('result')
    copyfile('../input/siim-acr-pneumothorax-segmentation/sample_submission.csv', './result/sample_result.csv')
    
    print('===============================TESTING COMPLETE===============================')
    print('segmentation result is saved into ./result/sample_result.csv ')
    
    
def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

if __name__ == '__main__':
    test()


# In[ ]:


ls


# In[ ]:


ls -l result/


# In[ ]:




