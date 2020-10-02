#!/usr/bin/env python
# coding: utf-8

# It's the first time I use fastai in Kaggle competition, I use fastai with 7fold cross validation, and inceptionresnetv2 + resnet50 ensemble. Pretty simple edition, and may seems to be stupid to some experienced Kaggler.
# This kernel can get both 0.626 in private & public LB, but I didn't select is for my another kernel get 0.627 in public LB and 0.623 in Private LB, which makes me get 71st place.  
# Well, fastai is easy to use, but it's difficult to get an outstanding result.
# BTW, thanks for [ods.ai] Alex Lekov for his kernel https://www.kaggle.com/itslek/fastai-resnet50-imet-v4-2 and thanks for Miguel Pinto for his kernel https://www.kaggle.com/mnpinto/imet-fastai-starter

# The following part is similar to the kernels mentioned above, you can ignore it

# In[ ]:



#======================================From here to end just a demo to train the resnet50 models===========================================
#I also trained inceptionresnetv2, with the internel open and use !pip install pretrainedmodels to directly use inceptionresnetv2. 
#I only need the imet pretrained resnet50 models, internel open but cannot submit result is ok for me
import fastai
from fastai.vision import *
fastai.__version__

#Here just for example, use BATCH  = 72 and SIZE   = 320. It's appropriate for pre-trained at first SIZE= 256 and then 288, at last, 320.

BATCH  = 72
SIZE   = 320
path = Path('../input/imet-2019-fgvc6/') # iMet data path

from torch.utils import model_zoo
Path('models').mkdir(exist_ok=True)
get_ipython().system("cp '../input/resnet50/resnet50.pth' 'models/'")
def load_url(*args, **kwargs):
    model_dir = Path('models')
    filename  = 'resnet50.pth'
    if not (model_dir/filename).is_file(): raise FileNotFoundError
    return torch.load(model_dir/filename)
model_zoo.load_url = load_url

train_df = pd.read_csv(path/'train.csv')
train_df.head()

labels_df = pd.read_csv(path/'labels.csv')
labels_df.head()

test_df = pd.read_csv(path/'sample_submission.csv')
test_df.head()


tfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=0.10, max_zoom=1.5, max_warp=0.2, max_lighting=0.2,
                     xtra_tfms=[(symmetric_warp(magnitude=(-0,0), p=0)),])

train, test = [ImageList.from_df(df, path=path, cols='id', folder=folder, suffix='.png') 
               for df, folder in zip([train_df, test_df], ['train', 'test'])]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val +                ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()


# Here I use 7 fold cross-validation. However, fastai **automatically determined the number of class** according to training dataset, which will lead to 1102 or 1101, 1100 classes rather than 1103 classes. Because the categories are imbalanced, some of categories only appears once: if it is divided into validation dataset, it will not be recoreded in training dataset. 
# So I use n_splits=15, and find out the divisions that number of classes are 1103 in training dataset, they are: set0, set1, set2, set3, set7, set8 and set10.

# In[ ]:



from sklearn.model_selection import KFold
import numpy as np
kf = KFold(n_splits=15, random_state=43, shuffle=True)

kf

n = 0
for train_index, test_index in kf.split(train):
    print("TRAIN:", train_index, "TEST:", test_index)
# Change if n == 0 to n == 1,2,3,7,8,10 to use other divisions
    if n == 0:
        break
    n+=1

X_train, X_test = train[train_index], train[test_index]


# Here, for example, train resnet50 for 1 time. In fact, you can train here at least 15 times with resnet50 in 32400 seconds, save the model again, next, load it again and again by learn.load function.

# In[ ]:


data = (train.split_by_list(X_train,X_test)
        .label_from_df(cols='attribute_ids', label_delim=' ')
        .add_test(test)
        .transform(tfms, size=SIZE, resize_method=ResizeMethod.PAD, padding_mode='border',)
        .databunch(path=Path('.'), bs=BATCH).normalize(imagenet_stats)

       )
print('Data loaded')
learn = cnn_learner(data, base_arch=models.resnet50, loss_func=FocalLoss(), metrics=fbeta,pretrained=True)
#use learn.load function to use your previous trained model, move the model to correct file position by using !cp function
#learn.load('resnet50-0-v4-35')
learn.unfreeze()
learn.fit_one_cycle(1, slice(1e-5,1e-2))
learn.freeze()
learn.save('resnet50-0-v4-1', return_path=True)
#=================================This is the end of pre-trained resnet50 demo====================================================


# The following code is actually what I used in the last submission result file. It's time saving by only loading models and do predictions. In public datasets, I only used less than 1500 seconds to do prediction in 7-fold inceptionresnetv2 and resnet50.

# In[ ]:


Incepres_BATCH  = 36
Incepres_SIZE   = 320
path = Path('../input/imet-2019-fgvc6/') # iMet data path


# In[ ]:


Res_BATCH  = 36
Res_SIZE   = 320


# In[ ]:


train, test = [ImageList.from_df(df, path=path, cols='id', folder=folder, suffix='.png') 
               for df, folder in zip([train_df, test_df], ['train', 'test'])]
Incepres_data = (train.split_by_rand_pct(0.1, seed=42)
        .label_from_df(cols='attribute_ids', label_delim=' ')
        .add_test(test)
        .transform(tfms, size=Incepres_SIZE, resize_method=ResizeMethod.PAD, padding_mode='border',)
        .databunch(path=Path('.'), bs=Incepres_BATCH).normalize(imagenet_stats))
Res_data = (train.split_by_rand_pct(0.1, seed=42)
        .label_from_df(cols='attribute_ids', label_delim=' ')
        .add_test(test)
        .transform(tfms, size=Res_SIZE, resize_method=ResizeMethod.PAD, padding_mode='border',)
        .databunch(path=Path('.'), bs=Res_BATCH).normalize(imagenet_stats))


# Here you should copy the key source codes in github repositories in pytorch pretrained-models cadene https://github.com/Cadene/pretrained-models.pytorch and fastai https://github.com/fastai/fastai to avoid using internet

# In[ ]:


from fastai.vision import learner


# In[ ]:


# Copied from https://github.com/Cadene/pretrained-models.pytorch, and just use for inceptionresnetv2
from __future__ import print_function, division, absolute_import

import torch

import torch.nn as nn

import torch.utils.model_zoo as model_zoo

import os

import sys



__all__ = ['InceptionResNetV2', 'inceptionresnetv2']



pretrained_settings = {

    'inceptionresnetv2': {

        'imagenet': {

            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',

            'input_space': 'RGB',

            'input_size': [3, 299, 299],

            'input_range': [0, 1],

            'mean': [0.5, 0.5, 0.5],

            'std': [0.5, 0.5, 0.5],

            'num_classes': 1000

        },

        'imagenet+background': {

            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',

            'input_space': 'RGB',

            'input_size': [3, 299, 299],

            'input_range': [0, 1],

            'mean': [0.5, 0.5, 0.5],

            'std': [0.5, 0.5, 0.5],

            'num_classes': 1001

        }

    }

}





class BasicConv2d(nn.Module):



    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):

        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,

                              kernel_size=kernel_size, stride=stride,

                              padding=padding, bias=False) # verify bias false

        self.bn = nn.BatchNorm2d(out_planes,

                                 eps=0.001, # value found in tensorflow

                                 momentum=0.1, # default pytorch value

                                 affine=True)

        self.relu = nn.ReLU(inplace=False)



    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        x = self.relu(x)

        return x





class Mixed_5b(nn.Module):



    def __init__(self):

        super(Mixed_5b, self).__init__()



        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)



        self.branch1 = nn.Sequential(

            BasicConv2d(192, 48, kernel_size=1, stride=1),

            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)

        )



        self.branch2 = nn.Sequential(

            BasicConv2d(192, 64, kernel_size=1, stride=1),

            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),

            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)

        )



        self.branch3 = nn.Sequential(

            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),

            BasicConv2d(192, 64, kernel_size=1, stride=1)

        )



    def forward(self, x):

        x0 = self.branch0(x)

        x1 = self.branch1(x)

        x2 = self.branch2(x)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)

        return out





class Block35(nn.Module):



    def __init__(self, scale=1.0):

        super(Block35, self).__init__()



        self.scale = scale



        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)



        self.branch1 = nn.Sequential(

            BasicConv2d(320, 32, kernel_size=1, stride=1),

            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        )



        self.branch2 = nn.Sequential(

            BasicConv2d(320, 32, kernel_size=1, stride=1),

            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),

            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)

        )



        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=False)



    def forward(self, x):

        x0 = self.branch0(x)

        x1 = self.branch1(x)

        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)

        out = self.conv2d(out)

        out = out * self.scale + x

        out = self.relu(out)

        return out





class Mixed_6a(nn.Module):



    def __init__(self):

        super(Mixed_6a, self).__init__()



        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)



        self.branch1 = nn.Sequential(

            BasicConv2d(320, 256, kernel_size=1, stride=1),

            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),

            BasicConv2d(256, 384, kernel_size=3, stride=2)

        )



        self.branch2 = nn.MaxPool2d(3, stride=2)



    def forward(self, x):

        x0 = self.branch0(x)

        x1 = self.branch1(x)

        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)

        return out





class Block17(nn.Module):



    def __init__(self, scale=1.0):

        super(Block17, self).__init__()



        self.scale = scale



        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)



        self.branch1 = nn.Sequential(

            BasicConv2d(1088, 128, kernel_size=1, stride=1),

            BasicConv2d(128, 160, kernel_size=(1,7), stride=1, padding=(0,3)),

            BasicConv2d(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))

        )



        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=False)



    def forward(self, x):

        x0 = self.branch0(x)

        x1 = self.branch1(x)

        out = torch.cat((x0, x1), 1)

        out = self.conv2d(out)

        out = out * self.scale + x

        out = self.relu(out)

        return out





class Mixed_7a(nn.Module):



    def __init__(self):

        super(Mixed_7a, self).__init__()



        self.branch0 = nn.Sequential(

            BasicConv2d(1088, 256, kernel_size=1, stride=1),

            BasicConv2d(256, 384, kernel_size=3, stride=2)

        )



        self.branch1 = nn.Sequential(

            BasicConv2d(1088, 256, kernel_size=1, stride=1),

            BasicConv2d(256, 288, kernel_size=3, stride=2)

        )



        self.branch2 = nn.Sequential(

            BasicConv2d(1088, 256, kernel_size=1, stride=1),

            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),

            BasicConv2d(288, 320, kernel_size=3, stride=2)

        )



        self.branch3 = nn.MaxPool2d(3, stride=2)



    def forward(self, x):

        x0 = self.branch0(x)

        x1 = self.branch1(x)

        x2 = self.branch2(x)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)

        return out





class Block8(nn.Module):



    def __init__(self, scale=1.0, noReLU=False):

        super(Block8, self).__init__()



        self.scale = scale

        self.noReLU = noReLU



        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)



        self.branch1 = nn.Sequential(

            BasicConv2d(2080, 192, kernel_size=1, stride=1),

            BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1)),

            BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))

        )



        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)

        if not self.noReLU:

            self.relu = nn.ReLU(inplace=False)



    def forward(self, x):

        x0 = self.branch0(x)

        x1 = self.branch1(x)

        out = torch.cat((x0, x1), 1)

        out = self.conv2d(out)

        out = out * self.scale + x

        if not self.noReLU:

            out = self.relu(out)

        return out





class InceptionResNetV2(nn.Module):



    def __init__(self, num_classes=1001):

        super(InceptionResNetV2, self).__init__()

        # Special attributs

        self.input_space = None

        self.input_size = (299, 299, 3)

        self.mean = None

        self.std = None

        # Modules

        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)

        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)

        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.maxpool_3a = nn.MaxPool2d(3, stride=2)

        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)

        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)

        self.maxpool_5a = nn.MaxPool2d(3, stride=2)

        self.mixed_5b = Mixed_5b()

        self.repeat = nn.Sequential(

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17)

        )

        self.mixed_6a = Mixed_6a()

        self.repeat_1 = nn.Sequential(

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10)

        )

        self.mixed_7a = Mixed_7a()

        self.repeat_2 = nn.Sequential(

            Block8(scale=0.20),

            Block8(scale=0.20),

            Block8(scale=0.20),

            Block8(scale=0.20),

            Block8(scale=0.20),

            Block8(scale=0.20),

            Block8(scale=0.20),

            Block8(scale=0.20),

            Block8(scale=0.20)

        )

        self.block8 = Block8(noReLU=True)

        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)

        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)

        self.last_linear = nn.Linear(1536, num_classes)



    def features(self, input):

        x = self.conv2d_1a(input)

        x = self.conv2d_2a(x)

        x = self.conv2d_2b(x)

        x = self.maxpool_3a(x)

        x = self.conv2d_3b(x)

        x = self.conv2d_4a(x)

        x = self.maxpool_5a(x)

        x = self.mixed_5b(x)

        x = self.repeat(x)

        x = self.mixed_6a(x)

        x = self.repeat_1(x)

        x = self.mixed_7a(x)

        x = self.repeat_2(x)

        x = self.block8(x)

        x = self.conv2d_7b(x)

        return x



    def logits(self, features):

        x = self.avgpool_1a(features)

        x = x.view(x.size(0), -1)

        x = self.last_linear(x)

        return x



    def forward(self, input):

        x = self.features(input)

        x = self.logits(x)

        return x



def inceptionresnetv2(num_classes=1000, pretrained='imagenet'):

    r"""InceptionResNetV2 model architecture from the

    `"InceptionV4, Inception-ResNet..." <https://arxiv.org/abs/1602.07261>`_ paper.

    """

    if pretrained:

        settings = pretrained_settings['inceptionresnetv2'][pretrained]

        assert num_classes == settings['num_classes'],"num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)



        # both 'imagenet'&'imagenet+background' are loaded from same parameters

        model = InceptionResNetV2(num_classes=1001)

        model.load_state_dict(model_zoo.load_url(settings['url']))



        if pretrained == 'imagenet':

            new_last_linear = nn.Linear(1536, 1000)

            new_last_linear.weight.data = model.last_linear.weight.data[1:]

            new_last_linear.bias.data = model.last_linear.bias.data[1:]

            model.last_linear = new_last_linear



        model.input_space = settings['input_space']

        model.input_size = settings['input_size']

        model.input_range = settings['input_range']



        model.mean = settings['mean']

        model.std = settings['std']

    else:

        model = InceptionResNetV2(num_classes=num_classes)

    return model



'''

TEST

Run this code with:

```

cd $HOME/pretrained-models.pytorch

python -m pretrainedmodels.inceptionresnetv2

```

'''
# Comment these code, we will not use them

#if __name__ == '__main__':



 #   assert inceptionresnetv2(num_classes=10, pretrained=None)

  #  print('success')

   # assert inceptionresnetv2(num_classes=1000, pretrained='imagenet')

    #print('success')

   # assert inceptionresnetv2(num_classes=1001, pretrained='imagenet+background')

    #print('success')



    # fail

   # assert inceptionresnetv2(num_classes=1001, pretrained='imagenet')


# In[ ]:


# copied from https://github.com/fastai/fastai/blob/master/fastai/vision/models/cadene_models.py
def get_incepres_model(model_name:str, pretrained:bool, seq:bool=False, pname:str='imagenet', **kwargs):

    pretrained = pname if pretrained else None

    model = inceptionresnetv2(pretrained=pretrained, **kwargs)

    return nn.Sequential(*model.children()) if seq else model


# In[ ]:


def myinceptionresnetv2(pretrained:bool=False):  
    return get_incepres_model('inceptionresnetv2', pretrained, seq=True)


learner.model_meta[myinceptionresnetv2] = {'cut': -2, 'split': lambda m: (m[0][9],     m[1])}


# In[ ]:





# In[ ]:


#!ls ../input/15foldincepres/15foldincepres.pth


# In[ ]:




get_ipython().system('ls ../input/incepres15fold320 ')


# The following code just show the prediction matrix, in fastai, don't use learn.TTA for it required 8 times of running time!

# In[ ]:


#cv0.593
get_ipython().system('cp ../input/incepres15fold320/inceptionres-0-v4-52.pth ./models/')
Incepres_learn0 = cnn_learner(Incepres_data, base_arch=myinceptionresnetv2, loss_func=FocalLoss(), metrics=fbeta,pretrained=False)
Incepres_learn0.load('inceptionres-0-v4-52')
#test_preds = learn0.TTA(ds_type=DatasetType.Test)
Incepres_test_preds0 = Incepres_learn0.get_preds(ds_type=DatasetType.Test)
Incepres_test_preds0


# In[ ]:


#cv0.597
get_ipython().system('cp ../input/incepres15fold320/inceptionres-1-v4-54.pth ./models/')
Incepres_learn1 = cnn_learner(Incepres_data, base_arch=myinceptionresnetv2, loss_func=FocalLoss(), metrics=fbeta,pretrained=False)
Incepres_learn1.load('inceptionres-1-v4-54')
#test_preds = learn0.TTA(ds_type=DatasetType.Test)
Incepres_test_preds1 = Incepres_learn1.get_preds(ds_type=DatasetType.Test)
Incepres_test_preds1


# In[ ]:


#cv0.588
get_ipython().system('cp ../input/incepres15fold320/inceptionres-2-v4-52.pth ./models/')
Incepres_learn2 = cnn_learner(Incepres_data, base_arch=myinceptionresnetv2, loss_func=FocalLoss(), metrics=fbeta,pretrained=False)
Incepres_learn2.load('inceptionres-2-v4-52')
#test_preds = learn0.TTA(ds_type=DatasetType.Test)
Incepres_test_preds2 = Incepres_learn2.get_preds(ds_type=DatasetType.Test)
Incepres_test_preds2


# In[ ]:


#cv0.597
get_ipython().system('cp ../input/incepres15fold320/inceptionres-8-v4-60.pth ./models/')
Incepres_learn8 = cnn_learner(Incepres_data, base_arch=myinceptionresnetv2, loss_func=FocalLoss(), metrics=fbeta,pretrained=False)
Incepres_learn8.load('inceptionres-8-v4-60')
#test_preds = learn0.TTA(ds_type=DatasetType.Test)
Incepres_test_preds8 = Incepres_learn8.get_preds(ds_type=DatasetType.Test)
Incepres_test_preds8


# In[ ]:


#cv0.588
get_ipython().system('cp ../input/incepres15fold320/inceptionres-3-v4-49-n.pth ./models/')
Incepres_learn3 = cnn_learner(Incepres_data, base_arch=myinceptionresnetv2, loss_func=FocalLoss(), metrics=fbeta,pretrained=False)
Incepres_learn3.load('inceptionres-3-v4-49-n')
#test_preds = learn0.TTA(ds_type=DatasetType.Test)
Incepres_test_preds3 = Incepres_learn3.get_preds(ds_type=DatasetType.Test)
Incepres_test_preds3


# In[ ]:


#cv0.588
get_ipython().system('cp ../input/incepres15fold320/inceptionres-7-v4-60-n.pth ./models/')
Incepres_learn7 = cnn_learner(Incepres_data, base_arch=myinceptionresnetv2, loss_func=FocalLoss(), metrics=fbeta,pretrained=False)
Incepres_learn7.load('inceptionres-7-v4-60-n')
#test_preds = learn0.TTA(ds_type=DatasetType.Test)
Incepres_test_preds7 = Incepres_learn7.get_preds(ds_type=DatasetType.Test)
Incepres_test_preds7


# In[ ]:


#Res_test_preds0


# In[ ]:


#cv0.594
get_ipython().system('cp ../input/incepres15fold320/inceptionres-10-v4-48.pth ./models/')
Incepres_learn10 = cnn_learner(Incepres_data, base_arch=myinceptionresnetv2, loss_func=FocalLoss(), metrics=fbeta,pretrained=False)
Incepres_learn10.load('inceptionres-10-v4-48')
#test_preds = learn0.TTA(ds_type=DatasetType.Test)
Incepres_test_preds10 = Incepres_learn10.get_preds(ds_type=DatasetType.Test)
Incepres_test_preds10


# In[ ]:


get_ipython().system('ls ../input/resnet50imet15fold320')


# In[ ]:


#!ls ../input/resnet50-imet-15fold/resnet50-0-v4-35.pth
get_ipython().system('cp ../input/resnet50imet15fold320/resnet50-0-v4-597.pth ./models/')
Res_learn0 = cnn_learner(Res_data, base_arch=models.resnet50, loss_func=FocalLoss(), metrics=fbeta,pretrained=False)
Res_learn0.load('resnet50-0-v4-597')

Res_test_preds0 = Res_learn0.get_preds(ds_type=DatasetType.Test)
Res_test_preds0


# In[ ]:


get_ipython().system('cp ../input/resnet50imet15fold320/resnet50-1-v4-603.pth ./models/')
Res_learn1 = cnn_learner(Res_data, base_arch=models.resnet50, loss_func=FocalLoss(), metrics=fbeta,pretrained=False)
Res_learn1.load('resnet50-1-v4-603')

Res_test_preds1 = Res_learn1.get_preds(ds_type=DatasetType.Test)
Res_test_preds1


# In[ ]:


get_ipython().system('cp ../input/resnet50imet15fold320/resnet50-2-v4-597.pth ./models/')
Res_learn2 = cnn_learner(Res_data, base_arch=models.resnet50, loss_func=FocalLoss(), metrics=fbeta,pretrained=False)
Res_learn2.load('resnet50-2-v4-597')

Res_test_preds2 = Res_learn2.get_preds(ds_type=DatasetType.Test)
Res_test_preds2


# In[ ]:


get_ipython().system('cp ../input/resnet50imet15fold320/resnet50-3-v4-597.pth ./models/')
Res_learn3 = cnn_learner(Res_data, base_arch=models.resnet50, loss_func=FocalLoss(), metrics=fbeta,pretrained=False)
Res_learn3.load('resnet50-3-v4-597')

Res_test_preds3 = Res_learn3.get_preds(ds_type=DatasetType.Test)
Res_test_preds3


# In[ ]:


get_ipython().system('cp ../input/resnet50imet15fold320/resnet50-7-v4-597.pth ./models/')
Res_learn7 = cnn_learner(Res_data, base_arch=models.resnet50, loss_func=FocalLoss(), metrics=fbeta,pretrained=False)
Res_learn7.load('resnet50-7-v4-597')

Res_test_preds7 = Res_learn7.get_preds(ds_type=DatasetType.Test)
Res_test_preds7


# In[ ]:


get_ipython().system('cp ../input/resnet50imet15fold320/resnet50-8-v4-603.pth ./models/')
Res_learn8 = cnn_learner(Res_data, base_arch=models.resnet50, loss_func=FocalLoss(), metrics=fbeta,pretrained=False)
Res_learn8.load('resnet50-8-v4-603')

Res_test_preds8 = Res_learn8.get_preds(ds_type=DatasetType.Test)
Res_test_preds8


# In[ ]:


get_ipython().system('cp ../input/resnet50imet15fold320/resnet50-10-v4-599.pth ./models/')
Res_learn10 = cnn_learner(Res_data, base_arch=models.resnet50, loss_func=FocalLoss(), metrics=fbeta,pretrained=False)
Res_learn10.load('resnet50-10-v4-599')

Res_test_preds10 = Res_learn10.get_preds(ds_type=DatasetType.Test)
Res_test_preds10


# In[ ]:


#import numpy as np
#np.shape(np.array(Res_test_preds0[0]))


# In[ ]:


#Here use take average of everything, or you can appoint the weight, it's simple
test_preds = ((Incepres_test_preds0[0]*1+Incepres_test_preds1[0]*1.1
               +Incepres_test_preds2[0]*0.9+Incepres_test_preds3[0]*0.9
               +Incepres_test_preds7[0]*0.9+Incepres_test_preds8[0]*1.1
               +Incepres_test_preds10[0]*1.1)/14+(Res_test_preds0[0]*1+Res_test_preds1[0]*1
               +Res_test_preds2[0]*1+Res_test_preds3[0]*1
               +Res_test_preds7[0]*1+Res_test_preds8[0]*1
               +Res_test_preds10[0]*1)/14,Incepres_test_preds1[1])
#test_preds = ((test_preds0[0]+test_preds1[0]+test_preds2[0]+test_preds3[0]+test_preds7[0]+test_preds8[0]+test_preds10[0])/7,test_preds1[1])


# In[ ]:


i2c = np.array([[i, c] for c, i in Incepres_learn0.data.train_ds.y.c2i.items()]).astype(int)
def join_preds(preds, thr):
    return [' '.join(i2c[np.where(t==1)[0],1].astype(str)) for t in (preds[0].sigmoid()>thr).long()]


# In[ ]:


# Most of models get good result around 0.260, 0.270 is also an applicable threshold
test_df.attribute_ids = join_preds(test_preds, 0.260)
test_df.head()


# In[ ]:


test_df.to_csv('submission.csv', index=False) 


# In[ ]:




