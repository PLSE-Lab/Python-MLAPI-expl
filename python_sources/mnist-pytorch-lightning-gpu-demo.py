#!/usr/bin/env python
# coding: utf-8

# # PyTorch Lightning TPU/GPU demo
# 
# On Lightning you can train a model using CPUs, TPUs and GPUs without changing ANYTHING about your code.
# 
# Let's walk through an example!

# ---
# ## Install XLA
# First, we have to install the XLA library to support TPUs on PyTorch

# In[ ]:


# VERSION = "20200325"  #@param ["1.5" , "20200325", "nightly"]
# !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# !python pytorch-xla-env-setup.py --version $VERSION


# ## Install PyTorch Lightning
# Next, we install PyTorch Lightning

# In[ ]:


get_ipython().system(' pip install -qU pip pytorch-lightning torch==1.5.1')


# # NVidia AMP

# In[ ]:


get_ipython().system('nvcc --version')
import torch
torch.__version__


# In[ ]:


try:
    from apex import amp
except:
    get_ipython().system('git clone https://github.com/NVIDIA/apex nv_apex')
    get_ipython().system('pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./nv_apex')
    from apex import amp


# ## Define lightning model

# In[ ]:


import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl

class CoolSystem(pl.LightningModule):

    def __init__(self, classes=10):
        super().__init__()
        self.save_hyperparameters()

        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, self.hparams.classes)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())

    def train_dataloader(self):
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
        loader = DataLoader(mnist_train, batch_size=32, num_workers=4)
        return loader


# ## Run training

# In[ ]:


from pytorch_lightning import Trainer

model = CoolSystem()

# most basic trainer, uses good defaults
trainer = Trainer(gpus=1, precision=16, progress_bar_refresh_rate=5, max_epochs=10)
trainer.fit(model)


# In[ ]:


# Start tensorboard.
get_ipython().run_line_magic('reload_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir lightning_logs/')

