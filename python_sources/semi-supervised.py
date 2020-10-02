#!/usr/bin/env python
# coding: utf-8

# <h2>Pipeline description:</h2>
# <h3>
#    1) Get features of images from train using pretrained resnet50 (with augmentation)<br><br>
#    2) Train simple logistic regression<br><br>
#    3) Use semi-supervised learning: add to train data those objects from test where probability is very high or very small (also with augmentation)<br><br>
#    4) Retrain logistic regression with new data
# </h3>

# In[ ]:


import os
import shutil
import zipfile

import numpy as np
import pandas as pd
import torch
import torchvision

from distutils.dir_util import copy_tree

from torchvision import transforms, models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler


# In[ ]:


with zipfile.ZipFile('../input/platesv2/plates.zip', 'r') as zip_obj:
   # Extract all the contents of zip file in current directory
   zip_obj.extractall('/kaggle/working/')


# In[ ]:


data_root = '/kaggle/working/plates/'
train_dir = 'train'
test_dir = 'test'
class_names = ['cleaned', 'dirty']


# In[ ]:


os.makedirs(os.path.join(train_dir), exist_ok=True)
os.makedirs(os.path.join(train_dir, class_names[0]), exist_ok=True)
os.makedirs(os.path.join(train_dir, class_names[1]), exist_ok=True)
copy_tree(os.path.join(f"{data_root}/{train_dir}"), os.path.join(f"{train_dir}"))


# In[ ]:


os.makedirs(os.path.join(test_dir), exist_ok=True)
for folder, folders, names in os.walk(f"{data_root}/{test_dir}"):
    for name in names:
        ind, ttype = name.split('.')
        if ttype != 'jpg':
            continue
        os.makedirs(os.path.join(test_dir, ind))
        shutil.copy(os.path.join(data_root, test_dir, name), os.path.join(test_dir, ind, name))


# In[ ]:


means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
batch_size = 8

train_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(means, stds)
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(means, stds)
])

train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)
test_dataset = torchvision.datasets.ImageFolder(test_dir, test_transforms)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)


# In[ ]:


class Identity(torch.nn.Module):
    """layer which just returns input"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

model = models.resnet50(pretrained=True)
model.fc = Identity()


# In[ ]:


def get_outputs(model, data_loader, num_epochs=1):
    """gets features of objects and labels for them (object path for test)"""
    
    X, y = [], []
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    
    for i in range(num_epochs):
        print(f"Epoch {i}")
        
        for inputs, labels in data_loader:
            with torch.set_grad_enabled(False):
                preds = model(inputs)
            X.extend(preds.tolist())
            y.extend(labels.tolist())
       
    return np.array(X), np.array(y)


# In[ ]:


X_train, y_train = get_outputs(model, train_dataloader, 50)


# In[ ]:


X_test, test_paths = get_outputs(model, test_dataloader)


# In[ ]:


scaler = StandardScaler()
scaler.fit(np.concatenate([X_train, X_test], axis=0))
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)


# In[ ]:


def partial_learning(model, output_model, scaler,
                     X_train, y_train, X_test, test_paths,
                     test_path, train_path, train_transforms, epochs_num=3):
    # folder to store new labeled images
    os.makedirs(os.path.join('additional'), exist_ok=True)

    batch_size = 8
    first_threshold = 0.95  # when model is 'sure' that plate is dirty
    second_threshold = 0.05  # when model is 'sure' that plate is clear
    epochs = 50  # number of times each new picture should be augmented
    
    os.makedirs(os.path.join('additional', '0'), exist_ok=True)
    os.makedirs(os.path.join('additional', '1'), exist_ok=True)

    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]

    add_1 = pred > first_threshold
    add_0 = pred < second_threshold

    print(add_1.sum(), add_0.sum())

    for i in test_paths[add_1]:
        folder = '0' * (4 - len(str(i))) + str(i)
        path = f"{test_path}/{folder}/{folder}.jpg"
        dest_path = f"additional/1/{i}.jpg"
        shutil.copy(os.path.join(path), os.path.join(dest_path))
    for i in test_paths[add_0]:
        folder = '0' * (4 - len(str(i))) + str(i)
        path = f"{test_path}/{folder}/{folder}.jpg"
        dest_path = f"additional/0/{i}.jpg"
        shutil.copy(os.path.join(path), os.path.join(dest_path))

    add_dataset = torchvision.datasets.ImageFolder(f"additional", train_transforms)

    add_dataloader = torch.utils.data.DataLoader(
        add_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)

    X_add, y_add = get_outputs(output_model, add_dataloader, 50)
    X_add = scaler.transform(X_add)
    X_train = np.concatenate([X_train, X_add], axis=0)
    y_train = np.hstack([y_train, y_add])

    model.fit(X_train, y_train)
    return X_train, y_train


# In[ ]:


lr = LogisticRegression(C=0.001, max_iter=1000)
new_X, new_y = partial_learning(lr, model, scaler, X_train_transformed, y_train, X_test_transformed, test_paths,                 'test', 'train', train_transforms)


# In[ ]:


get_ipython().system('rm -r additional')


# In[ ]:


new_y.shape[0] - y_train.shape[0]


# <h4> We got 70(3500 with augmentation) additional dirty plates and 70(3500) cleaned plates<h4>

# In[ ]:


pred = lr.predict_proba(X_test_transformed)[:, 1]


# In[ ]:


submission = pd.DataFrame.from_dict({'id': test_paths, 'label': pred})
submission['label'] = submission['label'].map(lambda x: 'dirty' if x > 0.1 else 'cleaned')
submission.set_index('id', inplace=True)
submission.to_csv('submission.csv')


# In[ ]:




