#!/usr/bin/env python
# coding: utf-8

# **Heng's Starter code for training the classification model on Kaggle.**
# 
# I have not run the kernel with GPU enabled because I do not have much of Kaggle GPU left as of now. So the kernel as expected is giving CUDA error.
# This is just a simple kernel for training model on Kaggle easily. Made some minor changes to his code and seems like it will run fine here on kaggle.
# I have not tested the training time. It can exceed the 9 hour limit.
# 
# The kernel is based on Heng's starter kit version 20190910 you can find it [here](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/106462#latest-645576) .
# I have imported 2 utility scripts one for the utility functions with plotting code and another one is for model.
# You can fork and edit the utility scripts and add the model classes as you feel like.
# The model architecture can be changed from this kernel below by changing the Net() class.
# 
# If you face any problems or errors then feel free to comment them.
# At last thank you very much [Heng](https://www.kaggle.com/hengck23) and other leaderboard rankers for helping newbies like me.

# In[ ]:


import numpy as np
import pandas as pd
import os

from timeit import default_timer as timer
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch.utils.data as data
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
import torch

from heng_s_utility_functions import *
from heng_s_models_all import *

PI = np.pi
IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]
DEFECT_COLOR = [(0,0,0),(0,0,255),(0,255,0),(255,0,0),(0,255,255)]


# In[ ]:


SPLIT_DIR = '../input/hengs-split'
DATA_DIR = '../input/severstal-steel-defect-detection'


# In[ ]:


class Net(nn.Module):
    def load_pretrain(self, skip, is_print=True):
        conversion=copy.copy(CONVERSION)
        for i in range(0,len(conversion)-8,4):
            conversion[i] = 'block.' + conversion[i][5:]
        load_pretrain(self, skip, pretrain_file=PRETRAIN_FILE, conversion=conversion, is_print=is_print)

    def __init__(self, num_class=4, drop_connect_rate=0.2):
        super(Net, self).__init__()

        e = ResNet34()
        self.block = nn.ModuleList([
            e.block0,
            e.block1,
            e.block2,
            e.block3,
            e.block4,
        ])
        e = None  #dropped
        self.feature = nn.Conv2d(512,32, kernel_size=1) #dummy conv for dim reduction
        self.logit = nn.Conv2d(32,num_class, kernel_size=1)

    def forward(self, x):
        batch_size,C,H,W = x.shape

        for i in range( len(self.block)):
            x = self.block[i](x)
            #print(i, x.shape)

        x = F.dropout(x,0.5,training=self.training)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.feature(x)
        logit = self.logit(x)
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
    batch_size = len(batch)

    input = []
    truth_mask  = []
    truth_label = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth_mask.append(batch[b][1])
        infor.append(batch[b][2])

        label = (batch[b][1].reshape(4,-1).sum(1)>8).astype(np.int32)
        truth_label.append(label)


    input = np.stack(input)
    input = image_to_input(input, IMAGE_RGB_MEAN,IMAGE_RGB_STD)
    input = torch.from_numpy(input).float()

    truth_mask = np.stack(truth_mask)
    truth_mask = (truth_mask>0.5).astype(np.float32)
    truth_mask = torch.from_numpy(truth_mask).float()

    truth_label = np.array(truth_label)
    truth_label = torch.from_numpy(truth_label).float()

    return input, truth_mask, truth_label, infor

# Metric
def metric_hit(logit, truth, threshold=0.5):
    batch_size,num_class, H,W = logit.shape

    with torch.no_grad():
        logit = logit.view(batch_size,num_class,-1)
        truth = truth.view(batch_size,num_class,-1)

        probability = torch.sigmoid(logit)
        p = (probability>threshold).float()
        t = (truth>0.5).float()

        tp = ((p + t) == 2).float()  # True positives
        tn = ((p + t) == 0).float()  # True negatives

        tp = tp.sum(dim=[0,2])
        tn = tn.sum(dim=[0,2])
        num_pos = t.sum(dim=[0,2])
        num_neg = batch_size*H*W - num_pos

        tp = tp.data.cpu().numpy()
        tn = tn.data.cpu().numpy().sum()
        num_pos = num_pos.data.cpu().numpy()
        num_neg = num_neg.data.cpu().numpy().sum()

        tp = np.nan_to_num(tp/(num_pos+1e-12),0)
        tn = np.nan_to_num(tn/(num_neg+1e-12),0)

        tp = list(tp)
        num_pos = list(num_pos)

    return tn,tp, num_neg,num_pos

# Loss
def criterion(logit, truth, weight=None):
    batch_size,num_class, H,W = logit.shape
    logit = logit.view(batch_size,num_class)
    truth = truth.view(batch_size,num_class)
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    if weight is None:
        loss = loss.mean()

    else:
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_sum = pos.sum().item() + 1e-12
        neg_sum = neg.sum().item() + 1e-12
        loss = (weight[1]*pos*loss/pos_sum + weight[0]*neg*loss/neg_sum).sum()
        #raise NotImplementedError

    return loss

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


schduler = NullScheduler(lr=0.001)
batch_size = 4 #8
iter_accum = 8


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
    valid_num  = np.zeros(6, np.float32)
    valid_loss = np.zeros(6, np.float32)
    
    for t, (input, truth_mask, truth_label, infor) in enumerate(valid_loader):

        #if b==5: break
        net.eval()
        input = input.cuda()
        truth_mask  = truth_mask.cuda()
        truth_label = truth_label.cuda()

        with torch.no_grad():
            logit = net(input) #data_parallel(net, input)
            loss  = criterion(logit, truth_label)
            tn,tp, num_neg,num_pos = metric_hit(logit, truth_label)


            #zz=0
        #---
        batch_size = len(infor)
        l = np.array([ loss.item(), tn,*tp])
        n = np.array([ batch_size, num_neg,*num_pos])
        valid_loss += l*n
        valid_num  += n

        #debug-----------------------------
        if displays is not None:
            probability = torch.sigmoid(logit)
            image = input_to_image(input, IMAGE_RGB_MEAN,IMAGE_RGB_STD)

            probability_label = probability.data.cpu().numpy()
            truth_label = truth_label.data.cpu().numpy()
            truth_mask  = truth_mask.data.cpu().numpy()

            for b in range(0, batch_size, 4):
                image_id = infor[b].image_id[:-4]
                result = draw_predict_result_label(image[b], truth_mask[b], truth_label[b], probability_label[b], stack='vertical')
                draw_shadow_text(result,'%05d    %s.jpg'%(valid_num[0]-batch_size+b, image_id),(5,24),0.75,[255,255,255],1)
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
        sampler    = RandomSampler(train_dataset),
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
    
#     if initial_checkpoint is not None:
#         state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
#         #for k in ['logit.weight','logit.bias']: state_dict.pop(k, None)

#         net.load_state_dict(state_dict,strict=False)
#     else:
#         load_pretrain(net.e, skip=['logit'], is_print=False)
        
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
#     import pdb; pdb.set_trace()
    while  iter<num_iters:
        sum_train_loss = np.zeros(20,np.float32)
        sum = np.zeros(20,np.float32)

        optimizer.zero_grad()
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
                print('%0.5f  %5.1f%s %5.1f |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  | %s' % (                         rate, iter/1000, asterisk, epoch,
                         *valid_loss[:6],
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
            loss = criterion(logit, truth_label)
            tn,tp, num_neg,num_pos = metric_hit(logit, truth_label)
            
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
            print('%0.5f  %5.1f%s %5.1f |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  | %s' % (                         rate, iter/1000, asterisk, epoch,
                         *valid_loss[:6],
                         *batch_loss[:6],
                         time_to_str((timer() - start),'min'))
            , end='',flush=True)
            i=i+1
            
            # debug-----------------------------
            if 1:
                for di in range(3):
                    if (iter+di)%1000==0:

                        probability = torch.sigmoid(logit)
                        image = input_to_image(input, IMAGE_RGB_MEAN,IMAGE_RGB_STD)

                        probability_label = probability.data.cpu().numpy()
                        truth_label = truth_label.data.cpu().numpy()
                        truth_mask  = truth_mask.data.cpu().numpy()


                        for b in range(batch_size):
                            result = draw_predict_result_label(image[b], truth_mask[b], truth_label[b], probability_label[b], stack='vertical')

#                             image_show('result',result,resize=1)
#                             cv2.imwrite('../working/%05d.png'%(di*100+b), result)
#                             cv2.waitKey(1)
                            pass



        pass  #-- end of one data loader --
    pass #-- end of all iterations --


# In[ ]:


print('rate     iter   epoch |  loss    tn, [tp1,tp2,tp3,tp4]       |  loss    tn, [tp1,tp2,tp3,tp4]       | time           ')
print('--------------------------------------------------------------------------------------------------------------------\n')
run_train()

