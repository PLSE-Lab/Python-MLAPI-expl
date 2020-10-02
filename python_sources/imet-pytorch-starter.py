#!/usr/bin/env python
# coding: utf-8

# ## Problem description
# 
# In this kernel, we are going to use Resnet34 pretrained model to fine tune with Pytorch.

# ## Libraries

# In[1]:


import gc
import os
import sys
import time
import random
import logging
import datetime as dt

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision as vision

from torch.optim.lr_scheduler import CosineAnnealingLR

from pathlib import Path
from PIL import Image
from contextlib import contextmanager

from joblib import Parallel, delayed
from tqdm import tqdm
from fastprogress import master_bar, progress_bar

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import fbeta_score

torch.multiprocessing.set_start_method("spawn")


# ## Utilities

# In[2]:


@contextmanager
def timer(name="Main", logger=None):
    t0 = time.time()
    yield
    msg = f"[{name}] done in {time.time() - t0} s"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
        

def get_logger(name="Main", tag="exp", log_dir="log/"):
    log_path = Path(log_dir)
    path = log_path / tag
    path.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(
        path / (dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"))
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# In[8]:


logger = get_logger(name="Main", tag="Pytorch-ResNet34")


# ## Data Loading

# In[4]:


get_ipython().system('ls ../input/imet-2019-fgvc6/')


# In[5]:


labels = pd.read_csv("../input/imet-2019-fgvc6/labels.csv")
train = pd.read_csv("../input/imet-2019-fgvc6/train.csv")
sample = pd.read_csv("../input/imet-2019-fgvc6/sample_submission.csv")
train.head()


# In[6]:


get_ipython().system('cp ../input/pytorch-pretrained-image-models/* ./')
get_ipython().system('ls')


# ## DataLoader

# In[9]:


class IMetImageDataset(data.DataLoader):
    def __init__(self, root_dir: Path, 
                 df: pd.DataFrame, 
                 mode="train",
                 device="cuda:0",
                 transforms=None):
        self._root = root_dir
        self.transform = transforms[mode]
        self._img_id = (df["id"] + ".png").values
        self.labels = df.attribute_ids.map(lambda x: x.split()).values
        self.mode = mode
        self.device = device
        
    def __len__(self):
        return len(self._img_id)
    
    def __getitem__(self, idx):
        img_id = self._img_id[idx]
        file_name = self._root / img_id
        img = Image.open(file_name)
        
        if self.transform:
            img = self.transform(img)
        if self.mode == "train" or self.mode == "val":
            label = self.labels[idx]
            label_tensor = torch.zeros((1, 1103))
            for i in label:
                label_tensor[0, int(i)] = 1
            label_tensor = label_tensor.to(self.device)
            return [img.to(self.device), label_tensor]
        else:
            return [img.to(self.device)]
    
    
data_transforms = {
    'train': vision.transforms.Compose([
        vision.transforms.RandomResizedCrop(224),
        vision.transforms.RandomHorizontalFlip(),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225])
    ]),
    'val': vision.transforms.Compose([
        vision.transforms.Resize(256),
        vision.transforms.CenterCrop(224),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225])
    ]),
}

data_transforms["test"] = data_transforms["val"]


# ## Model

# In[19]:


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(512, 1103)
        self.drop = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.drop(self.linear(x))
        return torch.sigmoid(x)


class ResNet34(nn.Module):
    def __init__(self, pretrained: Path):
        super(ResNet34, self).__init__()
        self.resnet34 = vision.models.resnet34()
        self.resnet34.load_state_dict(torch.load(pretrained))
        self.resnet34.fc = Classifier()
        
    def forward(self, x):
        return self.resnet34(x)


# ## Train Utilities

# In[20]:


class Trainer:
    def __init__(self, 
                 model, 
                 logger,
                 n_splits=5,
                 seed=42,
                 device="cuda:0",
                 train_batch=32,
                 valid_batch=128,
                 kwargs={}):
        self.model = model
        self.logger = logger
        self.device = device
        self.n_splits = n_splits
        self.seed = seed
        self.train_batch = train_batch
        self.valid_batch = valid_batch
        self.kwargs = kwargs
        
        self.best_score = None
        self.tag = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.loss_fn = nn.BCELoss(reduction="mean").to(self.device)
        
        path = Path(f"bin/{self.tag}")
        path.mkdir(exist_ok=True, parents=True)
        self.path = path
        
    def fit(self, X, n_epochs=10, kfold=False):
        train_preds = np.zeros((len(X), 1103))
        if kfold:
            fold = KFold(n_splits=self.n_splits, random_state=self.seed)
            for i, (trn_idx, val_idx) in enumerate(fold.split(X)):
                self.fold_num = i
                self.logger.info(f"Fold {i + 1}")
                X_train, X_val = X.loc[trn_idx, :], X.loc[val_idx, :]

                valid_preds = self._fit(X_train, X_val, n_epochs)
                train_preds[val_idx] = valid_preds
            return train_preds
        else:
            idx = np.arange(X.shape[0])
            self.fold_num = 0
            trn_idx, val_idx = train_test_split(
                idx, test_size=0.2, random_state=self.seed)
            X_train, X_val = X.loc[trn_idx, :], X.loc[val_idx, :]
            valid_preds = self._fit(X_train, X_val, n_epochs)
            train_preds = valid_preds
            return train_preds, y_val
    
    def _fit(self, X_train, X_val, n_epochs):
        seed_torch(self.seed)
        train_dataset = IMetImageDataset(root_dir=Path("../input/imet-2019-fgvc6/train/"), 
                                         df=X_train, 
                                         mode="train", 
                                         device=self.device, 
                                         transforms=data_transforms)
        train_loader = data.DataLoader(train_dataset, 
                                       batch_size=self.train_batch,
                                       shuffle=True)

        valid_dataset = IMetImageDataset(root_dir=Path("../input/imet-2019-fgvc6/train/"), 
                                         df=X_val, 
                                         mode="val", 
                                         device=self.device, 
                                         transforms=data_transforms)
        valid_loader = data.DataLoader(valid_dataset,
                                       batch_size=self.valid_batch,
                                       shuffle=False)
        
        model = self.model(**self.kwargs)
        model.to(self.device)
        
        optimizer = optim.Adam(params=model.parameters(), 
                                lr=0.0001)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
        best_score = np.inf
        mb = master_bar(range(n_epochs))
        for epoch in mb:
            model.train()
            avg_loss = 0.0
            for i_batch, y_batch in progress_bar(train_loader, parent=mb):
                y_pred = model(i_batch)
                loss = self.loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
            valid_preds, avg_val_loss = self._val(valid_loader, model)
            scheduler.step()

            self.logger.info("=========================================")
            self.logger.info(f"Epoch {epoch + 1} / {n_epochs}")
            self.logger.info("=========================================")
            self.logger.info(f"avg_loss: {avg_loss:.8f}")
            self.logger.info(f"avg_val_loss: {avg_val_loss:.8f}")
            
            if best_score > avg_val_loss:
                torch.save(model.state_dict(),
                           self.path / f"best{self.fold_num}.pth")
                self.logger.info(f"Save model at Epoch {epoch + 1}")
                best_score = avg_val_loss
        model.load_state_dict(torch.load(self.path / f"best{self.fold_num}.pth"))
        valid_preds, avg_val_loss = self._val(valid_loader, model)
        self.logger.info(f"Best Validation Loss: {avg_val_loss:.8f}")
        return valid_preds
    
    def _val(self, loader, model):
        model.eval()
        valid_preds = np.zeros((len(loader.dataset), 1103))
        avg_val_loss = 0.0
        for i, (i_batch, y_batch) in enumerate(loader):
            with torch.no_grad():
                y_pred = model(i_batch).detach()
                avg_val_loss += self.loss_fn(y_pred, y_batch).item() / len(loader)
                valid_preds[i * self.valid_batch:(i + 1) * self.valid_batch] =                     y_pred.cpu().numpy()
        return valid_preds, avg_val_loss
    
    def predict(self, X):
        dataset = IMetImageDataset(root_dir=Path("../input/imet-2019-fgvc6/test/"), 
                                   df=X, 
                                   mode="test", 
                                   device=self.device, 
                                   transforms=data_transforms)
        loader = data.DataLoader(dataset, 
                                 batch_size=self.valid_batch, 
                                 shuffle=False)
        model = self.model(**self.kwargs)
        preds = np.zeros((X.size(0), 1103))
        for path in self.path.iterdir():
            with timer(f"Using {str(path)}", self.logger):
                model.load_state_dict(torch.load(path))
                model.to(self.device)
                model.eval()
                temp = np.zeros_like(preds)
                for i, (i_batch, ) in enumerate(loader):
                    with torch.no_grad():
                        y_pred = model(i_batch).detach()
                        temp[i * self.valid_batch:(i + 1) * self.valid_batch] =                             y_pred.cpu().numpy()
                preds += temp / self.n_splits
        return preds


# ## Training

# In[21]:


trainer = Trainer(ResNet34, 
                  logger, 
                  train_batch=64, 
                  kwargs={"pretrained": "resnet34.pth"})
gc.collect()


# In[ ]:


y = train.attribute_ids.map(lambda x: x.split()).values
valid_preds, y_val = trainer.fit(train, n_epochs=22, kfold=False)


# ## Post process - threshold search -

# Since I used sigmoid for the activation, I've got the 1103 probability output for each data row.
# 
# I need to decide threshold for this.There are two ways to deal with this.
# 
# - Class-wise threshold search
#   - Takes some time but it's natural.
# - One threshold for all the class
#   - Low cost way.
# 
# **UPDATE**
# I will use the first -> second one.

# In[ ]:


def threshold_search(y_pred, y_true):
    score = []
    candidates = np.arange(0, 1.0, 0.01)
    for th in progress_bar(candidates):
        yp = (y_pred > th).astype(int)
        score.append(fbeta_score(y_pred=yp, y_true=y_true, beta=2, average="samples"))
    score = np.array(score)
    pm = score.argmax()
    best_th, best_score = candidates[pm], score[pm]
    return best_th, best_score


# In[ ]:


best_threshold, best_score = threshold_search(valid_preds, y_val)
best_score


# ## Prediction for test data

# In[ ]:


test_preds = trainer.predict(sample)


# In[ ]:


preds = (test_preds > best_threshold).astype(int)


# In[ ]:


prediction = []
for i in range(preds.shape[0]):
    pred1 = np.argwhere(preds[i] == 1.0).reshape(-1).tolist()
    pred_str = " ".join(list(map(str, pred1)))
    prediction.append(pred_str)
    
sample.attribute_ids = prediction
sample.to_csv("submission.csv", index=False)
sample.head()


# In[ ]:




