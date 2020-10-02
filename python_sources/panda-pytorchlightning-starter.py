#!/usr/bin/env python
# coding: utf-8

# # Pytorch Lightning Starter - PANDA competition
# 
# Here I won't explain again what this competition is about. This kernel is simply to share my (local) work on how I use pytorch lightning both for this competition and in my day to day work. I hope it can serve as a useful tutorial for fellow kagglers. This is also my first public kernel :) !
# 
# #### Why use Pytorch-Lightning ?
# 
# I preffer code that is in general organized in small .py files with classes. For machine learning training you should be able to modify the class you want and easily add options that you switch on and off by passing the required argument to the class or its methods. Pytorch lighntning is designed to help you easily follow a pytorch based training loop and ease modifications that you may want. Want to use a new scheduler ? Then simply modify the configure_optimizer method ! The beauty of it is that it automates all the boring stuff that clogs a pure pytorch code. All these loops, .zero_grad(), .eval(), torch.save etc. are gone and handled by the framework. You just have to focus on the ML part of it.
# 
# One of the great benefit is also the ability to automate logging with tensorboard which eases comparing multiple experiments. Obviously kaggle is not very friendly with that so I suggest reproducing the code of this kernel in a local environment and use tensorboard there.
# 
# You may ask why not simply use fastai. This is now a matter of preference. Fastai automates a lot of stuff with best practices like .fit_one_cycle. But on the other hand unless you have a lot of experience with it I find it rather opaque in what is happening behind the scenes. It's a framework designed to go with doing the fastai course so that you understand the options. If like me you learnt deep learning in a more academic environment in pure pytorch or pure tensorflow then you may find fastai hard to understand without listening to the great J. Howard courses. Similarly as soon as you want to do something a bit different it can become hard to understand how to change anything. On a personal note, I'll wait for the fastai v2 course before delving into it.

# In[ ]:


# Let's install it as it not in kaggle by default.
get_ipython().system('pip install pytorch_lightning')


# ## Loading data
# Here I will simply create a pytorch dataset to load the tiles from the popular notebook https://www.kaggle.com/iafoss/panda-16x128x128-tiles. A dataset should simply return all the information necessary for a sample by defining the __getitem__ and __len__ magic methods.

# In[ ]:


import zipfile
import datetime
import random
import os
import pandas as pd
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision.transforms as transforms
import pytorch_lightning as pl
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from mish_activation import Mish


# In[ ]:


class TileDataset(tdata.Dataset):

    def __init__(self, img_zip_path, dataframe, num_tiles, transform=None):
        """
        img_zip: Where the images are stored
        dataframe: The train.csv dataframe
        num_tiles: How many tiles should the dataset return per sample
        transform: The function to apply to the image. Usually dataaugmentation. DO NOT DO NORMALIZATION here.
        """
        # Here I am using an already existing kernel output with a zipfile. 
        # I suggest extracting files as it can lead to issue with multiprocessing.
        self.zip_img = zipfile.ZipFile(img_zip_path) 
        self.df = dataframe
        self.num_tiles = num_tiles
        self.img_list = self.df['image_id'].values
        
        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.img_list[idx]

        tiles = [img_id + '_' + str(i) + '.png' for i in range(0, self.num_tiles)]
        metadata = self.df.iloc[idx]
        image_tiles = []

        for tile in tiles:
            image = Image.open(self.zip_img.open(tile))

            if self.transform is not None:
                image = self.transform(image)

            image = 1 - image
            image = transforms.Normalize([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304],
                                         [0.1279171 , 0.24528177, 0.16098117])(image)
            image_tiles.append(image)

        image_tiles = torch.stack(image_tiles, dim=0)

        return {'image': image_tiles, 'provider': metadata['data_provider'],
                'isup': metadata['isup_grade'], 'gleason': metadata['gleason_score']}

    def __len__(self):
        return len(self.img_list)


# You are better off creating an instance of TileDataset in a pytorch lightning module. But here for demonstration purposes let's try our dataset class.

# In[ ]:


transform_train = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                      transforms.RandomVerticalFlip(0.5),
                                      transforms.ToTensor()])
train_df = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
trainset = TileDataset('/kaggle/input/panda-16x128x128-tiles/train.zip', train_df, 12, transform=transform_train)


# In[ ]:


image = trainset[0]['image']
print(image.shape, image.mean(), image.std())


# One element of the dataset is 12 tiles, of 128x128 RGB color images in a N, C, H, W tensor.

# ## Making a Model
# 
# Inspired by other popular kernel here is a simple resnext50 pytorch model. As explained at the start I suggest you define the head as a module itself. This will allow you to easily create new heads and then switch them in the main module.

# In[ ]:


class AdaptiveConcatPool2d(nn.Module):
    # This layer will concatenate both average and max pool
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        avg_x = self.avg(x)
        max_x = self.max(x)
        return torch.cat([avg_x, max_x], dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


# In[ ]:


class BasicHead(nn.Module):
    # The head of our model
    def __init__(self, c_in, c_out, n_tiles):
        self.n_tiles = n_tiles
        super().__init__()
        self.fc = nn.Sequential(AdaptiveConcatPool2d(),
                                Flatten(),
                                nn.Dropout(0.5),
                                nn.Linear(c_in * 2, 512),
                                Mish(),
                                nn.BatchNorm1d(512),
                                nn.Dropout(0.5),
                                nn.Linear(512, c_out))

    def forward(self, x):

        bn, c, height, width = x.shape
        h = x.view(-1, self.n_tiles, c, height, width).permute(0, 2, 1, 3, 4)             .contiguous().view(-1, c, height * self.n_tiles, width)
        h = self.fc(h)
        return h


# In[ ]:


class Model(nn.Module):
    # The mnain model combining a backbone and a head
    def __init__(self, c_out=6, n_tiles=12, tile_size=128, backbone='resnext50_semi', head='basic', **kwargs):
        super().__init__()
        if backbone == 'resnext50_semi':
            m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
        elif backbone == 'resnet50':
            m = models.resnet50(pretrained=True)
        
        c_feature = list(m.children())[-1].in_features  
        self.feature_extractor = nn.Sequential(*list(m.children())[:-2])  # Remove resnet head
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        if head == 'basic':
            self.head = BasicHead(c_feature, c_out, n_tiles)

    def forward(self, x):
        h = x.view(-1, 3, self.tile_size, self.tile_size)
        h = self.feature_extractor(h)
        h = self.head(h)

        return h


# ## Pytorch Lightning module definition
# 
# In a normal pytorch code you probably would instantiate the model, dataloaders and make a nested for loop for epochs and batches. Pytorch lightning automates the engineering parts like the loops so that you focus on the ML part. To do that you create a pytorch lightning model and then define every ML step inside of it. To help you understand I have added comments under every method you need to implement.

# In[ ]:


class LightModel(pl.LightningModule):

    def __init__(self, df_train, train_idx, val_idx, hparams):
        # This is where paths and options should be stored. I also store the
        # train_idx, val_idx for cross validation since the dataset are defined 
        # in the module !
        super().__init__()
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.df_train = df_train

        self.model = Model(c_out=hparams.c_out,  # This would be different for regression or classification
                           n_tiles=hparams.n_tiles,
                           tile_size=hparams.tile_size,
                           backbone=hparams.backbone,
                           head=hparams.head)

        self.hparams = hparams
        self.trainset = None
        self.valset = None

    def forward(self, batch):
        # What to do with a batch in a forward. Usually simple if everything is already defined in the model.
        return self.model(batch['image'])

    def prepare_data(self):
        # This is called at the start of training and is where everything data related should be initialized.
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                              transforms.RandomVerticalFlip(0.5),
                                              transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        
        self.trainset = TileDataset('/kaggle/input/panda-16x128x128-tiles/train.zip', self.df_train.iloc[self.train_idx], self.hparams.n_tiles, transform=transform_train)
        self.valset = TileDataset('/kaggle/input/panda-16x128x128-tiles/train.zip', self.df_train.iloc[self.val_idx], self.hparams.n_tiles, transform=transform_test)

    def train_dataloader(self):
        # Simply define a pytorch dataloader here that will take care of batching. Note it works well with dictionnaries !
        train_dl = tdata.DataLoader(self.trainset, batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=0)  # Using only one worker can be slow but zipfile can lead to bugs. You may try with multiple workers. In general use extracted files.
        return train_dl

    def val_dataloader(self):
        # Same but for validation. Pytorch lightning allows multiple validation dataloaders hence why I return a list.
        val_dl = tdata.DataLoader(self.valset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0)
        return [val_dl]

    def cross_entropy_loss(self, logits, gt):
        # How to calculate the loss. Note this method is actually not a part of pytorch lightning ! It's only good practice
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, gt)

    def configure_optimizers(self):
        # Optimizers and schedulers. Note that each are in lists of equal length to allow multiple optimizers (for GAN for example)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=3e-6)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=10 * self.hparams.lr, 
                                                        epochs=self.hparams.epochs, steps_per_epoch=len(self.train_dataloader()))
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # This is where you must define what happens during a training step (per batch)
        logits = self(batch)
        loss = self.cross_entropy_loss(logits, batch['isup']).unsqueeze(0)  # You need to unsqueeze in case you do multi-gpu training
        preds = logits.argmax(1)
        # Pytorch lightning will call .backward on what is called 'loss' in output
        # 'log' is reserved for tensorboard and will log everything define in the dictionary
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        # This is where you must define what happens during a validation step (per batch)
        logits = self(batch)
        loss = self.cross_entropy_loss(logits, batch['isup']).unsqueeze(0)
        preds = logits.argmax(1)
        return {'val_loss': loss, 'preds': preds, 'gt': batch['isup']}

    def validation_epoch_end(self, outputs):
        # This is what happens at the end of validation epoch. Usually gathering all predictions
        # outputs is a list of dictionary from each step.
        avg_loss = torch.cat([out['val_loss'] for out in outputs], dim=0).mean()
        preds = torch.cat([out['preds'] for out in outputs], dim=0)
        gt = torch.cat([out['gt'] for out in outputs], dim=0)
        preds = preds.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()

        kappa = cohen_kappa_score(preds, gt, weights='quadratic')
        tensorboard_logs = {'val_loss': avg_loss, 'kappa': kappa}
        print(f'Epoch {self.current_epoch}: {avg_loss:.2f}, kappa: {kappa:.4f}')

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


# This is a small helper to make hparams act like args. Making it easier to then switch the code to argparser.
# This is also what allows you to use hparams.arg instead of hparams['arg']

# In[ ]:


import argparse
def dict_to_args(d):
    args = argparse.Namespace()

    def dict_to_args_recursive(args, d, prefix=''):
        for k, v in d.items():
            if type(v) == dict:
                dict_to_args_recursive(args, v, prefix=k)
            elif type(v) in [tuple, list]:
                continue
            else:
                if prefix:
                    args.__setattr__(prefix + '_' + k, v)
                else:
                    args.__setattr__(k, v)

    dict_to_args_recursive(args, d)
    return args


# ## Training
# Let's start by specifying parameters, the seed and output folder.

# In[ ]:


SEED = 33
BATCH_SIZE = 4
NAME = 'resnext50'
OUTPUT_DIR = './lightning_logs'
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# In[ ]:


# This is where you should specify for an experiment what the model should use.
hparams = {'backbone': 'resnext50_semi', 
           'head': 'basic',
           'lr': 1e-4,
           'n_tiles': 12,
           'c_out': 6,
           'epochs':2,  # You obviously want to increase this :)
           'tile_size': 128}

hparams = dict_to_args(hparams) 


# Before start of training, the kernel output that I am using https://www.kaggle.com/iafoss/panda-16x128x128-tiles does not process all images in the zip file (missing those with a mask). Therefore I manually remove them from the dataframe to avoid issues with the datasets. **This is not necessary if you processed all tiles !**.

# In[ ]:


train_df = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
zip_img = zipfile.ZipFile('/kaggle/input/panda-16x128x128-tiles/train.zip')
not_in_image_zip = []
for img_id in train_df['image_id']:
    try:
        zip_img.open(img_id + '_' + '0.png')
    except KeyError:
        not_in_image_zip.append(img_id)
train_df = train_df[~train_df['image_id'].isin(not_in_image_zip)]


# For training we just need to instantiate the pytorch lightning module and a trainer with a few options. Most importantly this is where you specify how many GPU to use (or TPU) and if you want to do mixed precision training (with apex). For the purpose of this kernel I just do FP32 1GPU training but please read the pytorch lightning doc if you want to try TPU and/or mixed precision.
# 
# Note: For some reason the kernel has a bit of trouble with the zipfile and I have to usually

# In[ ]:


kfold = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
splits = kfold.split(train_df, train_df['isup_grade'])
date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

# Checkpoints and logs will be in ./OUTPUT_DIR/NAME-DATE/fold_i
NAME = 'resnext50'
OUTPUT_DIR = './lightning_logs'

for fold, (train_idx, val_idx) in enumerate(splits):
    print(f'Fold {fold + 1}')
    # Defining clearly to the tensorboard logger in order to put every fold under the same directory.
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=OUTPUT_DIR,
                                             name=f'{NAME}' + '-' + date,
                                             version=f'fold_{fold + 1}')

    # Define what metric the checkpoint should track (can be anything returned from the validation_end method)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=tb_logger.log_dir + "/{epoch:02d}-{kappa:.4f}",
                                                       monitor='kappa', mode='max')

    # Initiate model
    model = LightModel(train_df, train_idx, val_idx, hparams)
    
    # Define trainer
    # Here you can 
    trainer = pl.Trainer(gpus=[0], max_nb_epochs=hparams.epochs, auto_lr_find=False,
                         gradient_clip_val=1,
                         logger=tb_logger,
                         accumulate_grad_batches=1,              # BatchNorm ?
                         checkpoint_callback=checkpoint_callback
                         )
    # lr_finder = trainer.lr_find(model)
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig('lr_plot.png')
    trainer.fit(model)
    
    # One last thing. In checkpoints, pytorch lightning will save the weights and the state of the optimizer. 
    # This makes the weights very large. If you want to isolate the model weight simply use:
    torch.save(model.model.state_dict(), OUTPUT_DIR + '/' + NAME + '-' + date + '/' + f'fold_{fold}.pth')
    # One fold training (Remove this for training all folds)
    break


# That's it ! Hope it was useful. 
