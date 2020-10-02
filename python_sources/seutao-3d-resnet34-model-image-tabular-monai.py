#!/usr/bin/env python
# coding: utf-8

# ### This is a notebook version of the code shared by SeuTao
# 
# #### His gihub repo 
# 
# ##### https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection
# 
# I made small modifications to adapt it to Kaggle Kernel for the people who does not have local GPU . 

# ### Install

# In[ ]:


## Pretrained weight from version 1 of notebook . trained for 3 epochs
get_ipython().system('ls ../input/3dpretrained/fold0_epoch_26_0.1700.pth.tar')


# In[ ]:


get_ipython().system('pip install monai')
get_ipython().system('pip install nilearn')


# ## Common Lib

# In[ ]:


'''
Written by SeuTao
'''
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ### Config

# In[ ]:


class Config :
    '''
Configs for training & testing
Written by Whalechen
Modified By Nirjhar for Kaggle Kernel
'''

    data_root ='./toy_data'
    img_list = './toy_data/test_ci.txt',
    n_seg_classes=2
    learning_rate=0.001
    phase='train'
    save_intervals=10
    input_D=56
    input_H=448
    input_W=448
    resume_path=''#'./TReNDs/exp1/models_resnet_10_B_fold_1/epoch_13_batch_134_loss_0.1707041710615158.pth.tar'
    pretrain_path= None
    new_layer_names=['conv_seg']
    no_cuda = False
    gpu_id = [1,2]
    model='resnet'
    model_depth=18# help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
#    resnet_shortcut='B' #help='Shortcut type of resnet (A | B)')
    manual_seed=1
    ci_test =True
    model_name = 'exp1'
    fold_index = 0
    no_cuda = False
    pretrain_path = ''# '../input/3dpretrained/fold0_epoch_26_0.1700.pth.tar'## This is pretrained model provided by Seutao

    batch_size = 32
    num_workers = 0
    model_depth = 34
    resnet_shortcut = 'B'

    n_epochs = 2
    fold_index = 0

    model_name = r'prue_3dconv'
    save_folder = r'.'

#if not os.path.exists(Config.save_folder):
#        os.makedirs(Config.save_folder)
        
## Load Config        
Config = Config()


# ### Datasets 

# In[ ]:


##Datasets
import os
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold,StratifiedKFold, GroupKFold, KFold
import nilearn as nl
import torch
import random
from tqdm import tqdm

import monai
from monai.transforms import     LoadNifti, LoadNiftid, AddChanneld, ScaleIntensityRanged,     Rand3DElasticd, RandAffined,     Spacingd, Orientationd

root = r'../input/trends-assessment-prediction'

train_df = pd.read_csv('{}/train_scores.csv'.format(root)).sort_values(by='Id')
loadings = pd.read_csv('{}/loading.csv'.format(root))
sample = pd.read_csv('{}/sample_submission.csv'.format(root))
reveal = pd.read_csv('{}/reveal_ID_site2.csv'.format(root))
ICN = pd.read_csv('{}/ICN_numbers.csv'.format(root))

"""
    Load and display a subject's spatial map
"""

def load_subject(filename, mask_niimg):
    """
    Load a subject saved in .mat format with the version 7.3 flag. Return the subject niimg, using a mask niimg as a template for nifti headers.
    Args:
        filename    <str>            the .mat filename for the subject data
        mask_niimg  niimg object     the mask niimg object used for nifti headers
    """
    subject_data = None
    with h5py.File(filename, 'r') as f:
        subject_data = f['SM_feature'][()]
        # print(subject_data.shape)
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
    # print(subject_data.shape)
    return subject_data
    # subject_niimg = nl.image.new_img_like(mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)
    # return subject_niimg

def read_data_sample():
    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
    mask_filename = r'{}/fMRI_mask.nii'.format(root)
    subject_filename = '{}/fMRI_train/10004.mat'.format(root)

    mask_niimg = nl.image.load_img(mask_filename)
    print("mask shape is %s" % (str(mask_niimg.shape)))

    subject_niimg = load_subject(subject_filename, mask_niimg)
    print("Image shape is %s" % (str(subject_niimg.shape)))
    num_components = subject_niimg.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))

class TReNDsDataset(Dataset):

    def __init__(self, mode='train', fold_index = 0):
        # print("Processing {} datas".format(len(self.img_list)))
        self.mode = mode
        self.fold_index = fold_index

        if self.mode=='train' or self.mode=='valid' or self.mode=='valid_tta':
            features = ('age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2')
            data = pd.merge(loadings, train_df, on='Id').dropna()
            id_train = list(data.Id)
            fea_train = np.asarray(data.drop(list(features), axis=1).drop('Id', axis=1))
            lbl_train = np.asarray(data[list(features)])

            self.all_samples = []
            for i in range(len(id_train)):
                id = id_train[i]
                fea = fea_train[i]
                lbl = lbl_train[i]
                filename = os.path.join('{}/fMRI_train/{}.mat'.format(root, id))
                self.all_samples.append([filename, fea, lbl, str(id)])

            fold = 0
            kf = KFold(n_splits=5, shuffle=True, random_state=1337)
            for train_index, valid_index in kf.split(self.all_samples):
                if fold_index == fold:
                    self.train_index = train_index
                    self.valid_index = valid_index
                fold+=1

            if self.mode=='train':
                self.train_index = [tmp for tmp in self.train_index if os.path.exists(self.all_samples[tmp][0])]
                self.len = len(self.train_index)
                print('fold index:',fold_index)
                print('train num:', self.len)

            elif self.mode=='valid' or self.mode=='valid_tta':
                self.valid_index = [tmp for tmp in self.valid_index if os.path.exists(self.all_samples[tmp][0])]
                self.len = len(self.valid_index)
                print('fold index:',fold_index)
                print('valid num:', self.len)

        elif  self.mode=='test':
            labels_df = pd.read_csv("{}/train_scores.csv".format(root))
            labels_df["is_train"] = True

            features = ('age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2')
            data = pd.merge(loadings, labels_df, on="Id", how="left")

            id_test = list(data[data["is_train"] != True].Id)
            fea_test = np.asarray(data.drop(list(features), axis=1).drop('Id', axis=1)[data["is_train"] != True].drop("is_train", axis=1))
            lbl_test = np.asarray(data[list(features)][data["is_train"] != True])

            self.all_samples = []
            for i in range(len(id_test)):
                id = id_test[i]
                fea = fea_test[i]
                lbl = lbl_test[i]

                filename = os.path.join('{}/fMRI_test/{}.mat'.format(root, id))
                if os.path.exists(filename):
                    self.all_samples.append([id, filename, fea, lbl])

            self.len = len(self.all_samples)
            print(len(id_test))
            print('test num:', self.len)

    def __getitem__(self, idx):
        
        def get_data(filename):
            with h5py.File(filename, 'r') as f:
                subject_data = f['SM_feature'][()]
                # print(subject_data.shape)
                # It's necessary to reorient the axes, since h5py flips axis order
            subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
            return subject_data        

        if self.mode == "train" :
            filename, fea, lbl, id =  self.all_samples[self.train_index[idx]]
            train_img = get_data(filename)
            train_img = train_img.transpose((3,2,1,0))
            # (53, 52, 63, 53)
            train_lbl = lbl

            data_dict = {'image':train_img}
            rand_affine = RandAffined(keys=['image'],
                                      mode=('bilinear', 'nearest'),
                                      prob=0.5,
                                      spatial_size=(52, 63, 53),
                                      translate_range=(5, 5, 5),
                                      rotate_range=(np.pi * 4, np.pi * 4, np.pi * 4),
                                      scale_range=(0.15, 0.15, 0.15),
                                      padding_mode='border')
            affined_data_dict = rand_affine(data_dict)
            train_img = affined_data_dict['image']

            return torch.FloatTensor(train_img),                    torch.FloatTensor(train_lbl),                   torch.FloatTensor(fea) 


        elif self.mode == "valid":
            filename, fea, lbl, id =  self.all_samples[self.valid_index[idx]]
            train_img = get_data(filename)
            train_img = train_img.transpose((3, 2, 1, 0))
            # (53, 52, 63, 53)
            train_lbl = lbl

            return torch.FloatTensor(train_img),                   torch.FloatTensor(train_lbl),                   torch.FloatTensor(fea) 

        elif self.mode == 'test':
            id, filename, fea, lbl =  self.all_samples[idx]
            test_img = get_data(filename)
            test_img = test_img.transpose((3, 2, 1, 0))

            return str(id),                    torch.FloatTensor(test_img),                   torch.FloatTensor(fea) 

    def __len__(self):
        return self.len

def run_check_datasets():
    dataset = TReNDsDataset(mode='test')
    for m in range(len(dataset)):
        tmp = dataset[m]
        print(m)

def convert_mat2nii2npy():

    def get_data(filename):
        with h5py.File(filename, 'r') as f:
            subject_data = f['SM_feature'][()]
            # print(subject_data.shape)
        # It's necessary to reorient the axes, since h5py flips axis order
        subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
        return subject_data

    # train_root = '{}/fMRI_train/'.format(root)
    # train_npy_root = '{}/fMRI_train_npy/'.format(root)
    train_root = '{}/fMRI_test/'.format(root)
    train_npy_root = '{}/fMRI_test_npy/'.format(root)
    os.makedirs(train_npy_root, exist_ok=True)

    mats = os.listdir(train_root)
    mats = [mat for mat in mats if '.mat' in mat]
    random.shuffle(mats)

    for mat in tqdm(mats):
        mat_path = os.path.join(train_root, mat)
        if os.path.exists(mat_path):
            print(mat_path)

        npy_path = os.path.join(train_npy_root, mat.replace('.mat','.npy'))
        if os.path.exists(npy_path):
            print(npy_path, 'exist')
        else:
            data = get_data(mat_path)
            print(npy_path,data.shape)
            np.save(npy_path,data.astype(np.float16))


# ## Check Datasets

# In[ ]:


#run_check_datasets() ## Uncomment this to check dataset. it will take some time .


# ### network

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)

def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet3D(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 num_class = 5,
                 no_cuda=False,
                 tab_feat=26):

        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet3D, self).__init__()

        # 3D conv net
        self.conv1 = nn.Conv3d(53, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        # self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 64*2, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 128*2, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 256*2, layers[3], shortcut_type, stride=1, dilation=4)

        self.fea_dim = 256*2 * block.expansion
        self.tab_feat = tab_feat
        self.tab_out = 512
       
        self.tab_fc = nn.Sequential(nn.Linear(self.tab_feat, 1024),
                                 nn.BatchNorm1d(1024),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.8),
                                 nn.Linear(1024, 512),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5))
        
        self.fc = nn.Sequential(nn.Linear(self.fea_dim+512, num_class, bias=True))


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x,tab_data):
        x = self.conv1( x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.tab_fc(tab_data)

        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        emb_3d = x.view((-1, self.fea_dim))
        x = torch.cat((x1,emb_3d),dim=1)
        out = self.fc(x)
        return out


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet3D(BasicBlock, [1, 1, 1, 1],**kwargs)
    return model

def resnet3d_10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet3D(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet3D(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet3D(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet3D(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet3D(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet3D(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


# ### model

# In[ ]:


## Works only for Resnet10 for now.
import torch
from torch import nn


def generate_model(opt):
    assert opt.model in [
        'resnet'
    ]

    print('model depth: ',opt.model_depth)

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
            model = resnet10(
                # sample_input_W=opt.input_W,
                # sample_input_H=opt.input_H,
                # sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                # num_seg_classes=opt.n_seg_classes,
            )
        elif opt.model_depth == 18:
            model = resnet18(
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
            )
        elif opt.model_depth == 34:
            model = resnet34(
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
            )
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
    
    if not opt.no_cuda:
            model = model.cuda()
            model = nn.DataParallel(model)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()
    
    # load pretrain
    if opt.pretrain_path:
        print ('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys() and 'conv1' not in k}
        print(pretrain_dict.keys())

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters,
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()


# In[ ]:



def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0) / np.sum(y_true, axis=0))

def weighted_nae(inp, targ):
    W = torch.FloatTensor([0.3, 0.175, 0.175, 0.175, 0.175])
    return torch.mean(torch.matmul(torch.abs(inp - targ), W.cuda() / torch.mean(targ, axis=0)))

def valid(data_loader, model, sets):
    # settings
    print("validation")
    model.eval()

    y_pred = []
    y_true = []
    loss_ave = []

    with torch.no_grad():
        for batch_data in tqdm(data_loader):
                # getting data batch
                volumes, label,features = batch_data
                if not sets.no_cuda:
                    volumes = volumes.cuda()
                    label = label.cuda()
                    features = features.cuda()

                logits = model(volumes,features)

                # calculating loss
                loss_value = weighted_nae(logits, label)
                y_pred.append(logits.data.cpu().numpy())
                y_true.append(label.data.cpu().numpy())
                loss_ave.append(loss_value.data.cpu().numpy())

    print('valid loss', np.mean(loss_ave))
    y_pred = np.concatenate(y_pred,axis=0)
    y_true = np.concatenate(y_true,axis=0)

    domain = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
    w = [0.3, 0.175, 0.175, 0.175, 0.175]

    m_all = 0
    for i in range(5):
        m = metric(y_true[:,i], y_pred[:,i])
        print(domain[i],'metric:', m)
        m_all += m*w[i]

    print('all_metric:', m_all)
    model.train()
    return np.mean(loss_ave)

def test(data_loader, model, sets, save_path):
    # settings
    print("validation")
    model.eval()

    y_pred = []
    ids_all = []
    with torch.no_grad():
        for batch_data in tqdm(data_loader):
                # getting data batch
                ids, volumes,features = batch_data
                if not sets.no_cuda:
                    volumes = volumes.cuda() 
                    features = features.cuda()

                logits = model(volumes,features)
                y_pred.append(logits.data.cpu().numpy())
                ids_all += ids

    y_pred = np.concatenate(y_pred, axis=0)
    np.savez_compressed(save_path,
                        y_pred = y_pred,
                        ids = ids_all)
    print(y_pred.shape)

def train(train_loader,valid_loader, model, optimizer, ajust_lr, total_epochs, save_interval, save_folder, sets):
    f = open(os.path.join(save_folder,'log.txt'),'w')

    # settings
    batches_per_epoch = len(train_loader)
    print("Current setting is:")
    print(sets)
    print("\n\n")

    model.train()
    train_time_sp = time.time()

    valid_loss = 99999
    min_loss = 99999

    for epoch in range(total_epochs):
        rate = ajust_lr(optimizer, epoch)

        # log.info('lr = {}'.format(scheduler.get_lr()))
        for batch_id, batch_data in enumerate(train_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, label,features = batch_data

            if not sets.no_cuda: 
                volumes = volumes.cuda()
                label = label.cuda()
                features =features.cuda()

            optimizer.zero_grad()
            logits = model(volumes,features)

            # calculating loss
            loss = weighted_nae(logits, label)
            loss.backward()                
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)

            log_ = '{} Batch: {}-{} ({}), '                    'lr = {:.5f}, '                    'train loss = {:.3f}, '                    'valid loss = {:.3f}, '                    'avg_batch_time = {:.3f} '.format(sets.model_name, epoch, batch_id, batch_id_sp, rate, loss.item(), valid_loss, avg_batch_time)

            print(log_)
            f.write(log_ + '\n')
            f.flush()

        if 1:
            valid_loss = valid(valid_loader,model,sets)

            if valid_loss < min_loss:
                min_loss = valid_loss
                model_save_path = '{}/epoch_{}_batch_{}_loss_{}.pth.tar'.format(save_folder, epoch, batch_id, valid_loss)

                model_save_dir = os.path.dirname(model_save_path)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)

                log_ = 'Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)
                print(log_)
                f.write(log_ + '\n')

                torch.save({'ecpoch': epoch,
                                    'batch_id': batch_id,
                                    'state_dict': model.state_dict(),
                                    'optimizer': optimizer.state_dict()},
                                    model_save_path)

    print('Finished training')
    f.close()


# ### Create model and dataloader

# In[ ]:


# getting model
torch.manual_seed(Config.manual_seed)
model, parameters = generate_model(Config)
print(model)

# optimizer
def get_optimizer(net):
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
    
    def ajust_lr(optimizer, epoch):
            if epoch < 24 :
                    lr = 1e-4
            elif epoch < 36:
                    lr = 0.5e-4
            else:
                    lr = 1e-5

            for p in optimizer.param_groups:
                    p['lr'] = lr
            return lr

    rate = ajust_lr(optimizer, 0)
    return  optimizer, ajust_lr

optimizer, ajust_lr = get_optimizer(model)
    # train from resume
if Config.resume_path:
    if os.path.isfile(Config.resume_path):
        print("=> loading checkpoint '{}'".format(Config.resume_path))
        checkpoint = torch.load(Config.resume_path)
        model.load_state_dict(checkpoint['state_dict'])

    # getting data
Config.phase = 'train'
if Config.no_cuda:
    Config.pin_memory = False
else:
    Config.pin_memory = True

train_dataset = TReNDsDataset(mode='train', fold_index=Config.fold_index)
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size,
                             shuffle=True, num_workers=Config.num_workers,
                             pin_memory=Config.pin_memory,drop_last=True)

valid_dataset = TReNDsDataset(mode='valid', fold_index=Config.fold_index)
valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size,
                             shuffle=False, num_workers=Config.num_workers,
                             pin_memory=Config.pin_memory, drop_last=False)


# ### Start Training

# In[ ]:


# # training
train(train_loader, valid_loader,model, optimizer,ajust_lr,
          total_epochs=Config.n_epochs,
          save_interval=Config.save_intervals,
          save_folder=Config.save_folder, sets=Config)


# ### Start Testing

# In[ ]:


test_dataset = TReNDsDataset(mode='test', fold_index=Config.fold_index)
test_loader  = DataLoader(test_dataset, batch_size=Config.batch_size,
                             shuffle=False, num_workers=Config.num_workers,
                             pin_memory=False, drop_last=False)
test(test_loader, model, Config, './pred.npz') ## Uncomment this to generate pred.npz in kaggle


# In[ ]:


## Loading the prediction done in local gpu


# In[ ]:


from sklearn.svm import SVR
from sklearn.model_selection import KFold

def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))

fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")
fnc_df
loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")
fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")
labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")
labels_df["is_train"] = True
df = df.merge(labels_df, on="Id", how="left")
test_df = df[df["is_train"] != True].copy()
df = df[df["is_train"] == True].copy()


# In[ ]:


test_numpy  = np.load('./pred.npz')
age_sub = pd.read_csv("../input/rapids-svm-on-trends-neuroimaging/submission.csv") ## The age is not doing well so thinking of predicting other 4 values 
sub2= age_sub["Predicted"].values.reshape(age_sub.shape[0]//5, 5)
sub2[:, 1:] = test_numpy["y_pred"][:,1:]
test_df[[ "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]] = sub2
sub_df = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
assert sub_df.shape[0] == test_df.shape[0]*5
sub_df.head(10)


# In[ ]:


sub_df.to_csv("submission.csv", index=False)

