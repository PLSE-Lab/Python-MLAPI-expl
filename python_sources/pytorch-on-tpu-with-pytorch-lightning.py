#!/usr/bin/env python
# coding: utf-8

# # PyTorch Lightning TPU kernel
# Use this kernel to bootstrap a PyTorch project on TPUs using PyTorch Lightning
# 
# ## What is PyTorch Lightning?
# Lightning is simply organized PyTorch code. There's NO new framework to learn.
# For more details about Lightning visit the repo:
# 
# https://github.com/PyTorchLightning/pytorch-lightning

# ### Install XLA
# XLA powers the TPU support for PyTorch

# In[ ]:


get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
get_ipython().system('python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev')


# ## Install PyTorch Lightning

# In[ ]:


get_ipython().system(' pip install pytorch-lightning')


# ## Define the LightningModule
# This is just regular PyTorch organized in a specific format.
# 
# Notice the following:
# - no TPU specific code
# - no .to(device)
# - np .cuda()
# 
# For a full intro, read the following:   
# https://pytorch-lightning.readthedocs.io/en/stable/introduction_guide.html

# In[ ]:


import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

class MNISTModel(pl.LightningModule):

    def __init__(self):
        super(MNISTModel, self).__init__()
        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        # called with self(x)
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

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

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def prepare_data(self):
        self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    def train_dataloader(self):
        loader = DataLoader(self.mnist_train, batch_size=64, num_workers=4)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.mnist_test, batch_size=64, num_workers=4)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.mnist_test, batch_size=64, num_workers=4)
        return loader


# ## Train
# The Trainer automates the rest.
# 
# Trains on 8 cores

# In[ ]:


mnist_model = MNISTModel()

# most basic trainer, uses good defaults (1 TPU)
trainer = pl.Trainer(num_tpu_cores=8)    
trainer.fit(mnist_model)   


# ## Run test set
# In this example we used the test set to validation (a big no no), it's just for simplicity.
# In real training, make sure to split the train set into train/val and use test for testing.

# In[ ]:


trainer.test()


# ## View logs in tensorboard

# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir lightning_logs/')


# In[ ]:




