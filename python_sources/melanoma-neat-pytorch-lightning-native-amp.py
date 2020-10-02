#!/usr/bin/env python
# coding: utf-8

# # Melanoma classification with PyTorch Lightning
# 
# Using EfficientNet on PyTorch Lightning, with its amazing hardware agnostic and mixed precision implementation.
# 
# This is still work in progress, so please bear with me

# In[ ]:


fold_number = 1
seed  = 66
debug = False
tta   = 2 if debug else 20

batch_size = {
    'tpu': 10, # x8
    'gpu': 22, # 10 without AMP
    'cpu': 4,
}

arch = 'efficientnet-b5'
resolution = 456  # orignal res for B5
input_res  = 512

lr = 8e-6   # * batch_size
weight_decay = 2e-5
pos_weight   = 3.2
label_smoothing = 0.03

max_epochs = 7


# # Why PyTorch Lightning?
# Lightning is simply organized PyTorch code. There's NO new framework to learn.
# For more details about Lightning visit the repo:
# 
# https://github.com/PyTorchLightning/pytorch-lightning
# 
# - Run on CPU, GPU clusters or TPU, without any code changes
# - Transparent use of AMP (automatic mixed precision)
# 
# ![lightning structure](https://raw.githubusercontent.com/PyTorchLightning/pytorch-lightning/master/docs/source/_images/lightning_module/pt_to_pl.png)

# # Install modules
# 
# Update PyTorch to enable its native support to Mixed Precision or XLA for TPU

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os

if 'TPU_NAME' in os.environ.keys():
    try:
        import torch_xla
    except:
        # XLA powers the TPU support for PyTorch
        get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
        get_ipython().system('python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev')
else:
    # Update PyTorch to enable its native support to Mixed Precision
    get_ipython().system('pip install --pre torch==1.7.0.dev20200701+cu101 torchvision==0.8.0.dev20200701+cu101 -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html')

get_ipython().system('pip install -U pip albumentations==0.4.5 PyYAML pytorch-lightning==0.8.5 efficientnet_pytorch')


# # Hardware lookup

# In[ ]:


import os
import torch

num_workers = 2  # os.cpu_count()
gpus = 1 if torch.cuda.is_available() else None

try:
    import torch_xla.core.xla_model as xm
    tpu_cores = 8 #xm.xrt_world_size()
except:
    tpu_cores = None

if isinstance(batch_size, dict):
    if tpu_cores:
        batch_size = batch_size['tpu']
        lr *= tpu_cores
        num_workers = 1
    elif gpus:
        batch_size = batch_size['gpu']
        # support for free Colab GPU's
        if 'K80' in torch.cuda.get_device_name():
            batch_size = batch_size//3
        elif 'T4' in torch.cuda.get_device_name():
            batch_size = int(batch_size * 0.66)
    else:
        batch_size = batch_size['cpu']

lr *= batch_size

dict(
    num_workers=num_workers,
    tpu_cores=tpu_cores,
    gpus=gpus,
    batch_size=batch_size,
    lr=lr,
)


# # Automatic Mixed Precision
# 
# NVIDIA Apex is required only prior to PyTorch 1.6

# In[ ]:


# check for torch's native mixed precision support (pt1.6+)
if gpus and not hasattr(torch.cuda, "amp"):
    try:
        from apex import amp
    except:
        get_ipython().system('git clone https://github.com/NVIDIA/apex  nv_apex')
        get_ipython().system('pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./nv_apex')
        from apex import amp
    # with PyTorch Lightning all you need to do now is set precision=16


# # Imports

# In[ ]:


import os
import time
import random
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
from skimage import io
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from glob import glob
import sklearn

import pytorch_lightning as pl
import warnings

warnings.filterwarnings("ignore") 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(seed*6 + fold_number)

torch.__version__


# # Dataset
# 
# We will be using @shonenkov dataset with external data: https://www.kaggle.com/shonenkov/melanoma-merged-external-data-512x512-jpeg 
# 
# thank you @shonenkov

# In[ ]:


from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, path, image_ids, labels=None, transforms=None):
        super().__init__()
        self.path = path
        self.image_ids = image_ids
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image = cv2.imread(f'{self.path}/{image_id}.jpg', cv2.IMREAD_COLOR)

        if self.transforms:
            sample = self.transforms(image=image)
            image  = sample['image']

        label = self.labels[idx] if self.labels is not None else 0.5
        return image, label

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def get_labels(self):
        return list(self.labels)


# # Augmentations

# In[ ]:


def get_train_transforms():
    return A.Compose([
            A.JpegCompression(p=0.5),
            A.Rotate(limit=80, p=1.0),
            A.OneOf([
                A.OpticalDistortion(),
                A.GridDistortion(),
                A.IAAPiecewiseAffine(),
            ]),
            A.RandomSizedCrop(min_max_height=(int(resolution*0.7), input_res),
                              height=resolution, width=resolution, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussianBlur(p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(),   
                A.HueSaturationValue(),
            ]),
            A.Cutout(num_holes=8, max_h_size=resolution//8, max_w_size=resolution//8, fill_value=0, p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
            A.CenterCrop(height=resolution, width=resolution, p=1.0),
            A.Normalize(),
            ToTensorV2(),
        ], p=1.0)

def get_tta_transforms():
    return A.Compose([
            A.JpegCompression(p=0.5),
            A.RandomSizedCrop(min_max_height=(int(resolution*0.9), int(resolution*1.1)),
                              height=resolution, width=resolution, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ], p=1.0)


# # Setup dataset

# In[ ]:


DATA_PATH = '../input/melanoma-merged-external-data-512x512-jpeg'
TRAIN_ROOT_PATH = f'{DATA_PATH}/512x512-dataset-melanoma/512x512-dataset-melanoma'
TEST_ROOT_PATH = f'{DATA_PATH}/512x512-test/512x512-test'

df_folds = pd.read_csv(f'{DATA_PATH}/folds.csv', index_col='image_id',
                       usecols=['image_id', 'fold', 'target'], dtype={'fold': np.byte, 'target': np.byte})

_ = df_folds.groupby('fold').target.hist(alpha=0.4)
df_folds.groupby('fold').target.mean().to_frame('ratio').T


# In[ ]:


df_test = pd.read_csv(f'../input/siim-isic-melanoma-classification/test.csv', index_col='image_name')

if debug:
    df_folds = df_folds.sample(batch_size * 80)

df_folds = df_folds.sample(frac=1.0, random_state=seed*6+fold_number)


# In[ ]:


ds_train = ImageDataset(
    path=TRAIN_ROOT_PATH,
    image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
    labels=df_folds[df_folds['fold'] != fold_number].target.values,
    transforms=get_train_transforms(),
)

ds_val = ImageDataset(
    path=TRAIN_ROOT_PATH,
    image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
    labels=df_folds[df_folds['fold'] == fold_number].target.values,
    transforms=get_valid_transforms(),
)

ds_test = ImageDataset(
    path=TEST_ROOT_PATH,
    image_ids=df_test.index.values,
    transforms=get_tta_transforms(),
)

del df_folds
len(ds_train), len(ds_val), len(ds_test)


# # Model

# In[ ]:


from efficientnet_pytorch import EfficientNet
from pytorch_lightning.metrics.classification import AUROC
from sklearn.metrics import roc_auc_score

class Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = EfficientNet.from_pretrained(arch, advprop=True)
        self.net._fc = nn.Linear(in_features=self.net._fc.in_features, out_features=1, bias=True)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            max_lr=lr,
            epochs=max_epochs,
            optimizer=optimizer,
            steps_per_epoch=int(len(ds_train) / batch_size),
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100,
            base_momentum=0.90,
            max_momentum=0.95,
        )
        return [optimizer], [scheduler]

    def step(self, batch):
        # return batch loss
        x, y  = batch
        y_hat = self(x).flatten()
        y_smo = y.float() * (1 - label_smoothing) + 0.5 * label_smoothing
        loss  = F.binary_cross_entropy_with_logits(y_hat, y_smo.type_as(y_hat),
                                                   pos_weight=torch.tensor(pos_weight))
        return loss, y, y_hat.sigmoid()

    def training_step(self, batch, batch_nb):
        # hardware agnostic training
        loss, y, y_hat = self.step(batch)
        acc = (y_hat.round() == y).float().mean().item()
        tensorboard_logs = {'train_loss': loss, 'acc': acc}
        return {'loss': loss, 'acc': acc, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss, y, y_hat = self.step(batch)
        return {'val_loss': loss,
                'y': y.detach(), 'y_hat': y_hat.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        auc = AUROC()(pred=y_hat, target=y) if y.float().mean() > 0 else 0.5 # skip sanity check
        acc = (y_hat.round() == y).float().mean().item()
        print(f"Epoch {self.current_epoch} acc:{acc} auc:{auc}")
        tensorboard_logs = {'val_loss': avg_loss, 'val_auc': auc, 'val_acc': acc}
        return {'avg_val_loss': avg_loss,
                'val_auc': auc, 'val_acc': acc,
                'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, _ = batch
        y_hat = self(x).flatten().sigmoid()
        return {'y_hat': y_hat}

    def test_epoch_end(self, outputs):
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        df_test['target'] = y_hat.tolist()
        N = len(glob('submission*.csv'))
        df_test.target.to_csv(f'submission{N}.csv')
        return {'tta': N}

    def train_dataloader(self):
        return DataLoader(ds_train, batch_size=batch_size, num_workers=num_workers,
                          drop_last=True, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers,
                          drop_last=False, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(ds_test, batch_size=batch_size, num_workers=num_workers,
                          drop_last=False, shuffle=False, pin_memory=False)

model = Model()


# In[ ]:


# Plot some training images
import torchvision.utils as vutils
batch, targets = next(iter(model.train_dataloader()))

plt.figure(figsize=(16, 8))
plt.axis("off")
plt.title("Training Images")
_ = plt.imshow(vutils.make_grid(
    batch[:16], nrow=8, padding=2, normalize=True).cpu().numpy().transpose((1, 2, 0)))

targets[:16].reshape([2, 8]) if len(targets) >= 16 else targets


# In[ ]:


# # test the same images
# with torch.no_grad():
#     print(model(batch[:16]).reshape([len(targets)//8,8]).sigmoid())
del batch; del targets


# # Train
# The Trainer automates the rest.
# 
# Trains on 8 TPU cores, GPU or CPU - whatever is available.

# In[ ]:


# # View logs life in tensorboard
# Unfortunately broken again in the Kaggle notebooks :(
# however, it still works nicely in Colab or locally :)

# if gpus:
#     !pip install -qU tensorboard-plugin-profile
# %reload_ext tensorboard
# %tensorboard --logdir lightning_logs/


# In[ ]:


checkpoint_callback = pl.callbacks.ModelCheckpoint("{epoch:02d}_{val_auc:.4f}",
                                                   save_top_k=1, monitor='val_auc', mode='max')
trainer = pl.Trainer(
    tpu_cores=tpu_cores,
    gpus=gpus,
    precision=16 if gpus else 32,
    max_epochs=max_epochs,
    num_sanity_val_steps=1 if debug else 0,
    checkpoint_callback=checkpoint_callback,
#     val_check_interval=0.25, # check validation 4 times per epoch
)


# In[ ]:


# clean up gpu in case you are debugging 
import gc
torch.cuda.empty_cache(); gc.collect()
torch.cuda.empty_cache(); gc.collect()


# In[ ]:


trainer.fit(model)


# In[ ]:


# import pdb; pdb.pm()


# # Submission
# Infer on test set using a simple random TTA (test-time augmentation)

# In[ ]:


get_ipython().run_cell_magic('time', '', "for _ in range(tta):\n    trainer.test(ckpt_path='best')")


# In[ ]:


# merge TTA
submission = df_test[['target']]
submission.target = 0.0
for sub in glob('submission*.csv'):
    submission.target += pd.read_csv(sub, index_col='image_name').target

# min-max norm
submission.target -= submission.target.min()
submission.target /= submission.target.max()

submission.to_csv(f'submission_fold{fold_number}.csv')

submission.hist(bins=100, log=True, alpha=0.6)
submission.target.describe()


# In[ ]:


submission


# # K-Fold blend

# In[ ]:


folds_path = '../input/melanoma-neat-pytorch-lightning'
get_ipython().system('cp {folds_path}/*_fold*.csv .')
get_ipython().system('cp {folds_path}/*.ckpt .')


# In[ ]:


folds_sub = pd.read_csv(f'{folds_path}/submission.csv', index_col='image_name')

# incremental blend with equal weights for all folds
submission.target += folds_sub.target * (fold_number + 4)
submission.target /= (fold_number + 5)

submission.to_csv('submission.csv')

submission.hist(bins=100, log=True, alpha=0.6)
submission.target.describe()


# In[ ]:


if not debug and gpus:
    get_ipython().system('rm nv_apex -rf')
get_ipython().system('ls -sh')

