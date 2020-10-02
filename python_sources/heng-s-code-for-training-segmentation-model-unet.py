#!/usr/bin/env python
# coding: utf-8

# Heng's Starter code for training the segmentation model on Kaggle.
# 
# I have not run the kernel with GPU enabled because I do not have much of Kaggle GPU left as of now. So the kernel as expected is giving CUDA error. This is just a simple kernel for training model on Kaggle easily. Made some minor changes to his code and seems like it will run fine here on kaggle. I have not tested the training time. It can exceed the 9 hour limit.
# 
# The kernel is based on Heng's starter kit version 20190910 you can find it [here](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/106462#latest-645576) . I have imported 2 utility scripts one for the utility functions with code for plotting and another one is for model. You can fork and edit the utility scripts and add the model classes as you feel like. The model architecture can be changed from this kernel below by changing the Net() class.
# 
# If you face any problems or errors then feel free to comment them. At last thank you very much Heng and other leaderboard rankers for helping newbies like me.

# In[ ]:


import numpy as np
import pandas as pd
import os

import random 
from timeit import default_timer as timer
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Sampler
import torch.utils.data as data
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
import torch

from fork_of_heng_s_utility_functions import *
from heng_s_models_all import *

PI = np.pi
IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]
DEFECT_COLOR = [(0,0,0),(0,0,255),(0,255,0),(255,0,0),(0,255,255)]


# In[ ]:


SPLIT_DIR = '../input/hengs-split'
DATA_DIR = '../input/severstal-steel-defect-detection'


# In[ ]:


class FourBalanceClassSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = (self.dataset.df['Label'].values)
        label = label.reshape(-1,4)
        label = np.hstack([label.sum(1,keepdims=True)==0,label]).T

        self.neg_index  = np.where(label[0])[0]
        self.pos1_index = np.where(label[1])[0]
        self.pos2_index = np.where(label[2])[0]
        self.pos3_index = np.where(label[3])[0]
        self.pos4_index = np.where(label[4])[0]

        #assume we know neg is majority class
        num_neg = len(self.neg_index)
        self.length = 4*num_neg


    def __iter__(self):
        neg = self.neg_index.copy()
        random.shuffle(neg)
        num_neg = len(self.neg_index)

        pos1 = np.random.choice(self.pos1_index, num_neg, replace=True)
        pos2 = np.random.choice(self.pos2_index, num_neg, replace=True)
        pos3 = np.random.choice(self.pos3_index, num_neg, replace=True)
        pos4 = np.random.choice(self.pos4_index, num_neg, replace=True)

        l = np.stack([neg,pos1,pos2,pos3,pos4]).T
        l = l.reshape(-1)
        return iter(l)

    def __len__(self):
        return self.length


# In[ ]:


# UNet
def upsize(x,scale_factor=2):
    #x = F.interpolate(x, size=e.shape[2:], mode='nearest')
    x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Decode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decode, self).__init__()

        self.top = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d( out_channel//2),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.1),

            nn.Conv2d(out_channel//2, out_channel//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel//2),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.1),

            nn.Conv2d(out_channel//2, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True), #Swish(), #
        )

    def forward(self, x):
        x = self.top(torch.cat(x, 1))
        return x

class Net(nn.Module):

    def load_pretrain(self, skip, is_print=True):
        conversion=copy.copy(CONVERSION)
        for i in range(0,len(conversion)-8,4):
            conversion[i] = 'block.' + conversion[i][5:]
        load_pretrain(self, skip, pretrain_file=PRETRAIN_FILE, conversion=conversion, is_print=is_print)

    def __init__(self, num_class=5, drop_connect_rate=0.2):
        super(Net, self).__init__()

        e = ResNet18()
        self.block = nn.ModuleList([
           e.block0,
           e.block1,
           e.block2,
           e.block3,
           e.block4
        ])
        e = None  #dropped

        self.decode1 =  Decode(512,     128)
        self.decode2 =  Decode(128+256, 128)
        self.decode3 =  Decode(128+128, 128)
        self.decode4 =  Decode(128+ 64, 128)
        self.decode5 =  Decode(128+ 64, 128)
        self.logit = nn.Conv2d(128,num_class, kernel_size=1)

    def forward(self, x):
        batch_size,C,H,W = x.shape

        #----------------------------------
        backbone = []
        for i in range( len(self.block)):
            x = self.block[i](x)
            #print(i, x.shape)

            if i in [0,1,2,3,4]:
                backbone.append(x)

        #----------------------------------
        x = self.decode1([backbone[-1], ])                   #; print('d1',d1.size())
        x = self.decode2([backbone[-2], upsize(x)])          #; print('d2',d2.size())
        x = self.decode3([backbone[-3], upsize(x)])          #; print('d3',d3.size())
        x = self.decode4([backbone[-4], upsize(x)])          #; print('d4',d4.size())
        x = self.decode5([backbone[-5], upsize(x)])          #; print('d5',d5.size())

        logit = self.logit(x)
        logit = F.interpolate(logit, size=(H,W), mode='bilinear', align_corners=False)
        return logit


# In[ ]:


# Class which is used by the infor object in __get_item__
class Struct(object):
    def __init__(self, is_copy=False, **kwargs):
        self.add(is_copy, **kwargs)

    def add(self, is_copy=False, **kwargs):
        #self.__dict__.update(kwargs)

        if is_copy == False:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, copy.deepcopy(value))
                    #setattr(self, key, value.copy())
                except Exception:
                    setattr(self, key, value)

    def __str__(self):
        text =''
        for k,v in self.__dict__.items():
            text += '\t%s : %s\n'%(k, str(v))
        return text

# Creating masks
def run_length_decode(rle, height=256, width=1600, fill_value=1):
    mask = np.zeros((height,width), np.float32)
    if rle != '':
        mask=mask.reshape(-1)
        r = [int(r) for r in rle.split(' ')]
        r = np.array(r).reshape(-1, 2)
        for start,length in r:
            start = start-1  #???? 0 or 1 index ???
            mask[start:(start + length)] = fill_value
        mask=mask.reshape(width, height).T
    return mask

# Collations
def null_collate(batch):
#     pdb.set_trace()
    batch_size = len(batch)
    input = []
    truth_mask  = []
    truth_label = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        #truth_mask.append(batch[b][1])
        infor.append(batch[b][2])

        mask  = batch[b][1]
        label = (mask.reshape(4,-1).sum(1)>0).astype(np.int32)

        num_class,H,W = mask.shape
        mask = mask.transpose(1,2,0)*[1,2,3,4]
        mask = mask.reshape(-1,4)
        mask = mask.max(-1).reshape(1,H,W)

        truth_mask.append(mask)
        truth_label.append(label)

    
    input = np.stack(input)
    input = image_to_input(input, IMAGE_RGB_MEAN,IMAGE_RGB_STD)
    input = torch.from_numpy(input).float()

    truth_mask = np.stack(truth_mask)
    truth_mask = torch.from_numpy(truth_mask).long()

    truth_label = np.array(truth_label)
    truth_label = torch.from_numpy(truth_label).float()

    return input, truth_mask, truth_label, infor

# Metric
def metric_dice(logit, truth, threshold=0.1, sum_threshold=1):

    with torch.no_grad():
        probability = torch.softmax(logit,1)
        probability = one_hot_encode_predict(probability)
        truth = one_hot_encode_truth(truth)

        batch_size,num_class, H,W = truth.shape
        probability = probability.view(batch_size,num_class,-1)
        truth = truth.view(batch_size,num_class,-1)
        p = (probability>threshold).float()
        t = (truth>0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)

        d_neg = (p_sum < sum_threshold).float()
        d_pos = 2*(p*t).sum(-1)/((p+t).sum(-1)+1e-12)

        neg_index = (t_sum==0).float()
        pos_index = 1-neg_index

        num_neg = neg_index.sum()
        num_pos = pos_index.sum(0)
        dn = (neg_index*d_neg).sum()/(num_neg+1e-12)
        dp = (pos_index*d_pos).sum(0)/(num_pos+1e-12)

        #----

        dn = dn.item()
        dp = list(dp.data.cpu().numpy())
        num_neg = num_neg.item()
        num_pos = list(num_pos.data.cpu().numpy())

    return dn,dp, num_neg,num_pos

def metric_hit(logit, truth, threshold=0.5):
    batch_size,num_class, H,W = logit.shape

    with torch.no_grad():
        logit = logit.view(batch_size,num_class,-1)
        truth = truth.view(batch_size,-1)

        probability = torch.softmax(logit,1)
        p = torch.max(probability, 1)[1]
        t = truth
        correct = (p==t)

        index0 = t==0
        index1 = t==1
        index2 = t==2
        index3 = t==3
        index4 = t==4

        num_neg  = index0.sum().item()
        num_pos1 = index1.sum().item()
        num_pos2 = index2.sum().item()
        num_pos3 = index3.sum().item()
        num_pos4 = index4.sum().item()

        neg  = correct[index0].sum().item()/(num_neg +1e-12)
        pos1 = correct[index1].sum().item()/(num_pos1+1e-12)
        pos2 = correct[index2].sum().item()/(num_pos2+1e-12)
        pos3 = correct[index3].sum().item()/(num_pos3+1e-12)
        pos4 = correct[index4].sum().item()/(num_pos4+1e-12)

        num_pos = [num_pos1,num_pos2,num_pos3,num_pos4,]
        tn = neg
        tp = [pos1,pos2,pos3,pos4,]

    return tn,tp, num_neg,num_pos

# Loss
def criterion(logit, truth, weight=None):
    logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, 5)
    truth = truth.permute(0, 2, 3, 1).contiguous().view(-1)

    if weight is not None: weight = torch.FloatTensor([1]+weight).cuda()
    loss = F.cross_entropy(logit, truth, weight=weight, reduction='none')

    loss = loss.mean()
    return loss

#One-Hot for segmentation
def one_hot_encode_truth(truth, num_class=4):
    one_hot = truth.repeat(1,num_class,1,1)
    arange  = torch.arange(1,num_class+1).view(1,num_class,1,1).to(truth.device)
    one_hot = (one_hot == arange).float()
    return one_hot

def one_hot_encode_predict(predict, num_class=4):
    value, index = torch.max(predict, 1, keepdim=True)
    value  = value.repeat(1,num_class,1,1)
    index  = index.repeat(1,num_class,1,1)
    arange = torch.arange(1,num_class+1).view(1,num_class,1,1).to(predict.device)
    one_hot = (index == arange).float()
    value = value*one_hot
    return value

# Learning Rate Adjustments
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]
    return lr

# Learning Rate Schedule
class NullScheduler():
    def __init__(self, lr=0.01 ):
        super(NullScheduler, self).__init__()
        self.lr    = lr
        self.cycle = 0

    def __call__(self, time):
        return self.lr

    def __str__(self):
        string = 'NullScheduler\n'                 + 'lr=%0.5f '%(self.lr)
        return string


# In[ ]:


# import pdb; 

# def abc():

#     next(iter(train_loader))
# abc()


# In[ ]:


schduler = NullScheduler(lr=0.001)
batch_size = 4 #8
iter_accum = 8
loss_weight = None#[5,5,2,5] #
train_sampler = FourBalanceClassSampler #RandomSampler


# In[ ]:


class SteelDataset(Dataset):
    def __init__(self, split, csv, mode, augment=None):
#         import pdb; pdb.set_trace()
        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.augment = augment

        self.uid = list(np.concatenate([np.load(SPLIT_DIR + '/%s'%f , allow_pickle=True) for f in split]))
        df = pd.concat([pd.read_csv(DATA_DIR + '/%s'%f) for f in csv])
        df.fillna('', inplace=True)
        df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
        df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
        df = df_loc_by_list(df, 'ImageId_ClassId', [ u.split('/')[-1] + '_%d'%c  for u in self.uid for c in [1,2,3,4] ])
        self.df = df

    def __str__(self):
        num1 = (self.df['Class']==1).sum()
        num2 = (self.df['Class']==2).sum()
        num3 = (self.df['Class']==3).sum()
        num4 = (self.df['Class']==4).sum()
        pos1 = ((self.df['Class']==1) & (self.df['Label']==1)).sum()
        pos2 = ((self.df['Class']==2) & (self.df['Label']==1)).sum()
        pos3 = ((self.df['Class']==3) & (self.df['Label']==1)).sum()
        pos4 = ((self.df['Class']==4) & (self.df['Label']==1)).sum()

        length = len(self)
        num = len(self)*4
        pos = (self.df['Label']==1).sum()
        neg = num-pos

        #---

        string  = ''
        string += '\tmode    = %s\n'%self.mode
        string += '\tsplit   = %s\n'%self.split
        string += '\tcsv     = %s\n'%str(self.csv)
        string += '\t\tlen   = %5d\n'%len(self)
        if self.mode == 'train':
            string += '\t\tnum   = %5d\n'%num
            string += '\t\tneg   = %5d  %0.3f\n'%(neg,neg/num)
            string += '\t\tpos   = %5d  %0.3f\n'%(pos,pos/num)
            string += '\t\tpos1  = %5d  %0.3f  %0.3f\n'%(pos1,pos1/length,pos1/pos)
            string += '\t\tpos2  = %5d  %0.3f  %0.3f\n'%(pos2,pos2/length,pos2/pos)
            string += '\t\tpos3  = %5d  %0.3f  %0.3f\n'%(pos3,pos3/length,pos3/pos)
            string += '\t\tpos4  = %5d  %0.3f  %0.3f\n'%(pos4,pos4/length,pos4/pos)
        return string


    def __len__(self):
        return len(self.uid)


    def __getitem__(self, index):
        # print(index)
        folder, image_id = self.uid[index].split('/')

        rle = [
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_1','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_2','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_3','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_4','EncodedPixels'].values[0],
        ]
        image = cv2.imread(DATA_DIR + '/%s/%s'%(folder,image_id), cv2.IMREAD_COLOR)
        mask  = np.array([run_length_decode(r, height=256, width=1600, fill_value=1) for r in rle])

        infor = Struct(
            index    = index,
            folder   = folder,
            image_id = image_id,
        )

        if self.augment is None:
            return image, mask, infor
        else:
            return self.augment(image, mask, infor)


# In[ ]:


def do_valid(net, valid_loader, displays=None):
    valid_num  = np.zeros(11, np.float32)
    valid_loss = np.zeros(11, np.float32)
    
    for t, (input, truth_mask, truth_label, infor) in enumerate(valid_loader):

        #if b==5: break
        net.eval()
        input = input.cuda()
        truth_mask  = truth_mask.cuda()
        truth_label = truth_label.cuda()

        with torch.no_grad():
            logit = net(input) #data_parallel(net, input)
            loss  = criterion(logit, truth_mask)
            tn,tp, num_neg,num_pos = metric_hit(logit, truth_mask)
            dn,dp, num_neg,num_pos = metric_dice(logit, truth_mask, threshold=0.5, sum_threshold=100)
            
            #zz=0
        #---
        batch_size = len(infor)
        l = np.array([ loss.item(), tn,*tp, dn,*dp ])
        n = np.array([ batch_size, num_neg,*num_pos, num_neg,*num_pos ])
        valid_loss += l*n
        valid_num  += n

        #debug-----------------------------
        if displays is not None:
            probability = torch.sigmoid(logit)
            image = input_to_image(input, IMAGE_RGB_MEAN,IMAGE_RGB_STD)

            probability = one_hot_encode_predict(probability)
            truth_mask  = one_hot_encode_truth(truth_mask)
            
            probability_mask = probability.data.cpu().numpy()
            truth_label = truth_label.data.cpu().numpy()
            truth_mask  = truth_mask.data.cpu().numpy()

            for b in range(0, batch_size, 4):
                image_id = infor[b].image_id[:-4]
                result = draw_predict_result_label(image[b], truth_mask[b], truth_label[b], probability_mask[b], stack='vertical')
                draw_shadow_text(result,'%05d    %s.jpg'%(valid_num[0]-batch_size+b, image_id),(5,24),1,[255,255,255],2)
                image_show('result',result,resize=1)
#                 cv2.imwrite(out_dir +'/valid/%s.png'%(image_id), result)
#                 cv2.waitKey(1)
                pass
        #debug-----------------------------

        #print(valid_loss)
        print('\r %8d /%8d'%(valid_num[0], len(valid_loader.dataset)),end='',flush=True)

        pass  #-- end of one data loader --
    assert(valid_num[0] == len(valid_loader.dataset))
    valid_loss = valid_loss/valid_num

    return valid_loss


# In[ ]:


def run_train():
    batch_size = 4

    initial_checkpoint =     '/root/share/project/kaggle/2019/steel/result1/resnet34-cls-full-foldb0-0/checkpoint/00007500_model.pth'
    
    train_dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train_b1_11568.npy',],
        augment = train_augment,
    )
    train_loader  = DataLoader(
        train_dataset,
        #sampler     = BalanceClassSampler(train_dataset, 3*len(train_dataset)),
        #sampler    = SequentialSampler(train_dataset),
        sampler    = train_sampler(train_dataset),
        batch_size  = batch_size,
        drop_last   = True,
        num_workers = 2,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    valid_dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['valid_b1_1000.npy',],
        augment = valid_augment,
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler    = SequentialSampler(valid_dataset),
        #sampler     = RandomSampler(valid_dataset),
        batch_size  = 4,
        drop_last   = False,
        num_workers = 2,
        pin_memory  = True,
        collate_fn  = null_collate
    )
    
    assert(len(train_dataset)>=batch_size)
    
    net = Net().cuda()
    
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.9, weight_decay=0.0001)

    num_iters   = 3000*1000
    iter_smooth = 50
    iter_log    = 500
    iter_valid  = 1500
    iter_save   = [0, num_iters-1]                   + list(range(0, num_iters, 1500))#1*1000

    start_iter = 0
    start_epoch= 0
    rate       = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']
            #optimizer.load_state_dict(checkpoint['optimizer'])
        pass
    
    train_loss = np.zeros(20,np.float32)
    valid_loss = np.zeros(20,np.float32)
    batch_loss = np.zeros(20,np.float32)
    iter = 0
    i    = 0
    
    start = timer()
    
    while  iter<num_iters:
        sum_train_loss = np.zeros(20,np.float32)
        sum = np.zeros(20,np.float32)

        optimizer.zero_grad()
#         import pdb; pdb.set_trace()
        for t, (input, truth_mask, truth_label, infor) in enumerate(train_loader):
            batch_size = len(infor)
            iter  = i + start_iter
            epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch
            
            # Weather to display images or not! While in validation loss
            displays = None
            #if 0:
            if (iter % iter_valid==0):
                valid_loss = do_valid(net, valid_loader, displays) # omitted outdir variable
                #pass

            if (iter % iter_log==0):
                print('\r',end='',flush=True)
                asterisk = '*' if iter in iter_save else ' '
                print('%0.5f  %5.1f%s %5.1f |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  | %s' % (                         rate, iter/1000, asterisk, epoch,
                         *valid_loss[:11],
                         *train_loss[:6],
                         time_to_str((timer() - start),'min'))
                )
                print('\n')
                
            #if 0:
            if iter in iter_save:
                torch.save(net.state_dict(),'../working/%08d_model.pth'%(iter))
                torch.save({
                    #'optimizer': optimizer.state_dict(),
                    'iter'     : iter,
                    'epoch'    : epoch,
                }, '../working/%08d_optimizer.pth'%(iter))
                pass

            # learning rate schduler -------------
            lr = schduler(iter)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)
            
            net.train()
            input = input.cuda()
            truth_label = truth_label.cuda()
            truth_mask  = truth_mask.cuda()

            logit =  net(input) #data_parallel(net,input)  
            loss = criterion(logit, truth_mask, loss_weight)
            tn,tp, num_neg,num_pos = metric_hit(logit, truth_mask)
            
            (loss/iter_accum).backward()
            if (iter % iter_accum)==0:
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  ------------
            l = np.array([ loss.item(), tn,*tp ])
            n = np.array([ batch_size, num_neg,*num_pos ])

            batch_loss[:6] = l
            sum_train_loss[:6] += l*n
            sum[:6] += n
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss/(sum+1e-12)
                sum_train_loss[...] = 0
                sum[...]            = 0


            print('\r',end='',flush=True)
            asterisk = ' '
            print('%0.5f  %5.1f%s %5.1f |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  | %s' % (                     rate, iter/1000, asterisk, epoch,
                     *valid_loss[:11],
                     *train_loss[:6],
                     time_to_str((timer() - start),'min'))
            )
            print('\n')
            i=i+1
            
            # debug-----------------------------
            if 1:
                for di in range(3):
                    if (iter+di)%1000==0:

                        probability = torch.softmax(logit,1)
                        image = input_to_image(input, IMAGE_RGB_MEAN,IMAGE_RGB_STD)
                        
                        probability = one_hot_encode_predict(probability)
                        truth_mask  = one_hot_encode_truth(truth_mask)
                        
                        probability_mask = probability.data.cpu().numpy()
                        truth_label = truth_label.data.cpu().numpy()
                        truth_mask  = truth_mask.data.cpu().numpy()


                        for b in range(batch_size):
                    
                            result = draw_predict_result_label(image[b], truth_mask[b], truth_label[b], probability_mask[b], stack='vertical')

                            image_show('result',result,resize=1)
#                             cv2.imwrite('../working/%05d.png'%(di*100+b), result)
#                             cv2.waitKey(1)
                            pass
        pass  #-- end of one data loader --
    pass #-- end of all iterations --


# In[ ]:


print('                      |-------------------------------- VALID-----------------------------|---------- TRAIN/BATCH ------------------------------\n')
print('rate     iter   epoch |  loss    hit_neg,pos1,2,3,4           dice_neg,pos1,2,3,4         |  loss    hit_neg,pos1,2,3,4          | time         \n')
print('------------------------------------------------------------------------------------------------------------------------------------------------\n')
          #0.00000    0.0*   0.0 |  0.690   0.50 [0.00,1.00,0.00,1.00]   0.44 [0.00,0.02,0.00,0.15]  |  0.000   0.00 [0.00,0.00,0.00,0.00]  |  0 hr 00 min
run_train()

