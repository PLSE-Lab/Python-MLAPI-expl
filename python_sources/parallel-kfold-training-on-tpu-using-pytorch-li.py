#!/usr/bin/env python
# coding: utf-8

# # Parallel KFold training on TPU using Pytorch Lightning
# This kernel demonstrates training K instances of a model parallely on each TPU core.

# ### Install XLA
# XLA powers the TPU support for PyTorch

# In[ ]:


get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
get_ipython().system('python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev')


# ## Install PyTorch Lightning

# In[ ]:


get_ipython().system('pip uninstall -q typing --yes')
get_ipython().system('pip install -qU git+https://github.com/lezwon/pytorch-lightning.git@2016_test')


# In[ ]:


import os

import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler
import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import KFold
import torch_xla.core.xla_model as xm


import torch_xla


# In[ ]:


# Set a seed for numpy for a consistent Kfold split
np.random.seed(123)


# In[ ]:


# Download the dataset in advance
MNIST(os.getcwd(), train=True, download=True)
MNIST(os.getcwd(), train=True, download=True)


# # **Define a Lightning Module**
# Define a lightning module that takes in fold number in hparams.

# In[ ]:


class MNISTModel(pl.LightningModule):

    def __init__(self, hparams):
        super(MNISTModel, self).__init__()
        self.hparams = hparams
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': {'tpu': torch_xla._XLAC._xla_get_default_device()}}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.nll_loss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    
    def prepare_data(self):
        dataset = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
        self.mnist_test = MNIST(os.getcwd(), train=False, download=False, transform=transforms.ToTensor())
        
        kf = KFold(n_splits=8)
        splits = list(kf.split(dataset))
        train_indices, val_indices = splits[self.hparams['fold']]
        
        self.mnist_train = torch.utils.data.Subset(dataset, train_indices)
        self.mnist_val = torch.utils.data.Subset(dataset, val_indices)
                

    def train_dataloader(self):
        loader = DataLoader(self.mnist_train, batch_size=64, num_workers=4)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.mnist_val, batch_size=32, num_workers=4)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.mnist_test, batch_size=32, num_workers=4)
        return loader


# ## Train
# Use the `trainer` to train an instance of a model. It takes care of all the TPU setup given a `tpu_id` in `tpu_cores`.

# In[ ]:


hparams = {'lr': 6.918309709189366e-07, 'fold': 1}


# In[ ]:


# Define a function to initialize and train a model
def train(tpu_id):
    model = MNISTModel(hparams)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath='checkpoints/tpu' + str(tpu_id) + '-{epoch}-{val_loss:.2f}',
        monitor='avg_val_loss',
        mode='min'
    )
    
    trainer = pl.Trainer(tpu_cores=[tpu_id], precision=16, max_epochs=5, checkpoint_callback=checkpoint_callback)    
    trainer.use_native_amp = False
    trainer.fit(model)
    trainer.test()


# In[ ]:


#use joblib to run the train function in parallel on different folds
import joblib as jl
parallel = jl.Parallel(n_jobs=8, backend="threading", batch_size=1)
parallel(jl.delayed(train)(i+1) for i in range(8))


# In[ ]:


# weights are saved to checkpoints
get_ipython().system('ls -lh checkpoints/ ')


# In[ ]:




