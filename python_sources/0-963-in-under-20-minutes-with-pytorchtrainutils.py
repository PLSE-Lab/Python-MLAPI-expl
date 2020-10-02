#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import torch
import torch.nn.functional as F
import torchvision
import cv2
import albumentations
import albumentations.pytorch
import os
import PIL
import copy
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# ### Install pytorchtrainutils and efficientnet-pytorch

# Very simple yet useful set of util functions for training models in pytorch https://github.com/carloalbertobarbano/pytorch-train-utils

# In[ ]:


get_ipython().system('pip install --upgrade git+git://github.com/carloalbertobarbano/pytorch-train-utils')


# In[ ]:


from pytorchtrainutils import trainer
from pytorchtrainutils import metrics
from pytorchtrainutils import utils


# # Utilities functions
# Unhide code cell

# In[ ]:


def plot_cm(logs):
    accs = logs['top1-acc']
    cms = logs['cm']

    classes = ['healthy', 'multiple_diseases', 'rust', 'scab']

    plt.figure(figsize=(20, 3))
    plt.suptitle(f'CM')
    for class_idx, class_name in enumerate(classes):
        plt.subplot(1, 4, class_idx+1)
        plt.title(f'{class_name}')
        ax = sns.heatmap(
            cms.class_cm(class_idx, normalized=True), 
            annot=True, fmt=".2f", vmin=0., vmax=1.
        )

    plt.savefig('cm.png')
    plt.show()

    plt.figure(figsize=(20, 3))
    plt.suptitle(f'CM (best threshold)')
    for class_idx, class_name in enumerate(classes):
        plt.subplot(1, 4, class_idx+1)
        plt.title(f'{class_name}')

        best_threshold = accs.get_best_threshold(class_idx)
        ax = sns.heatmap(
            cms.class_cm(class_idx, normalized=True, threshold=best_threshold), 
            annot=True, fmt=".2f", vmin=0., vmax=1.
        )

    plt.savefig('cm-t')
    plt.show()
    

def plot_roc_auc(logs):
    aucs = logs['col-auc']
    accs = logs['top1-acc']
    classes = ['healthy', 'multiple_diseases', 'rust', 'scab']

    plt.figure(figsize=(20, 5))
    plt.suptitle(f'Classification report - average AUC: {aucs.get():.4f}')
    for class_idx, class_name in enumerate(classes):
        plt.subplot(1, 4, class_idx+1)
        plt.title(f'{class_name} BA={accs.class_ba(class_idx):.4f}')
        fpr, tpr, _ = aucs.class_curve(class_idx)
        plt.plot(fpr, tpr, label=f'AUC: {aucs.class_auc(class_idx):.4f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')
        plt.legend(loc='lower right')
    plt.savefig('auc.png')
    plt.show()


# # Hyperparams & values

# In[ ]:


seed = 42
utils.set_seed(seed)
device = torch.device('cuda')

lr = 1e-2
batch_size = 8
n_epochs_224 = 30
n_epochs_448 = 20

arch = 'resnet18'

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] #Imagenet


# # Data loading
# 
# I converted the competition dataset to pickles (.npy). 
# Images are already downscaled to 600x600. You can find the dataset (along with the generating script) [here](https://www.kaggle.com/carloalbertobarbano/plantpathology2020fgvc7pickles).
# The speedup is noticeable.

# In[ ]:


get_ipython().system('ls /kaggle/input')


# In[ ]:


dataset_path = '/kaggle/input/plantpathology2020fgvc7pickles/plant-pathology-2020-fgvc7-pickles'
train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))


# In[ ]:


def preprocess_df(df):
    df['label'] =  df.multiple_diseases * 2 +                 df.rust * 3 +                 df.scab * 4 +                 df.healthy
    df.label -= 1
    return df


# In[ ]:


train_df = preprocess_df(train_df)
train_df.head()


# In[ ]:


train_df.iloc[:, 1:].sum()


# As you can see the *multiple_diseases* class is very unbalanced, so we will apply weights to the loss function. Undersampling of the other classes is not really an option here

# In[ ]:


class PlantDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, df, path, transform):
        super().__init__()

        self.df = df
        self.path = os.path.join(path, 'images')
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        entry = self.df.iloc[index]
        fname = entry.image_id + '.npy'
        fname = os.path.join(self.path, fname)

        img = np.load(fname)
        img = self.transform(img)

        return img, entry.values[1:5].astype('float64')


# In[ ]:


train_df, val_df = train_test_split(train_df, test_size=0.3, random_state=seed, stratify=train_df.label)


# We'll treat this task as a multilabel classification problem; even if the given data is actually a single-class classification. 

# In[ ]:


train_weight = val_weight = torch.tensor([
    [1., 1.],
    [1.5, 2.],
    [1., 1.],
    [1., 1.]
]).to(device)


# I will also apply a slight oversampling on the minority class, by giving it a higher weight.

# In[ ]:


class_weights = torch.tensor([1., 1.3, 1.1, 1.])
sampler_weights = class_weights[train_df.label.values]


# # Data transforms & aug

# Applying TTA with albumentations and [pytorchtrainutils](https://github.com/carloalbertobarbano/pytorch-train-utils) is very easy

# In[ ]:


def stack_image(image, **kwargs):
    to_tensor = albumentations.pytorch.ToTensor()
    vflip = albumentations.VerticalFlip(always_apply=True)
    hflip = albumentations.HorizontalFlip(always_apply=True)
    
    return torch.stack([
        to_tensor(image=vflip(image=image)['image'])['image'],
        to_tensor(image=hflip(image=image)['image'])['image'],
        to_tensor(image=vflip(image=hflip(image=image)['image'])['image'])['image'],
        to_tensor(image=image)['image']
    ])

def get_transform(img_size, crop_size):
    train_transform = albumentations.Compose([
        albumentations.Resize(img_size, img_size, always_apply=True),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.ShiftScaleRotate(rotate_limit=30.0, scale_limit=0.2, shift_limit=0.15, p=0.7),
        albumentations.CenterCrop(crop_size, crop_size, always_apply=True),
        albumentations.Normalize(mean, std),
        albumentations.pytorch.ToTensor()
    ])

    transform = albumentations.Compose([
        albumentations.Resize(img_size, img_size, always_apply=True),
        albumentations.CenterCrop(crop_size, crop_size, always_apply=True),
        albumentations.Normalize(mean, std),
        albumentations.pytorch.ToTensor()
    ])

    tta_transform = albumentations.Compose([
        albumentations.Resize(img_size, img_size, always_apply=True),
        albumentations.CenterCrop(crop_size, crop_size, always_apply=True),
        albumentations.Normalize(mean, std),
        albumentations.Lambda(stack_image, always_apply=True) 
    ])
    
    lambda_train = lambda image: train_transform(image=image)['image']
    lambda_valid = lambda image: transform(image=image)['image']
    lambda_tta = lambda image: tta_transform(image=image)['image']
    
    return lambda_train, lambda_valid, lambda_tta


# # Model creation

# We actually don't need a very large or sophisticated model to reach LB ~0.963. A resnet18 is enough, though you can probably do better with a slightly bigger model

# In[ ]:


model = torchvision.models.resnet18(pretrained=True)
num_ft = model.fc.in_features
model.fc = torch.nn.Linear(in_features=num_ft, out_features=4, bias=True)
model = model.to(device)


# In[ ]:


class Softmaxer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        x = self.model(x)
        return F.softmax(x, dim=1)


# # Loss function

# The loss function is a normal binary cross entropy (with logits), we only need some indexing magic to correctly apply the weight

# In[ ]:


def bce(preds, targets, weight=None):    
    loss = F.binary_cross_entropy_with_logits(preds, targets.type(preds.dtype), reduction='none')
    if weight is not None:
        weight = weight[:, targets.T.long()]
        idx = np.diag_indices(weight.shape[0])
        weight = weight[idx[0], idx[1], :]
        loss *= weight.T
    return loss.mean()


# # Train on 224

# In[ ]:


img_size = 250
crop_size = 224
name = f'{arch}-{crop_size}'
print(f'{name} image size: {img_size}, crop size: {crop_size}')


# In[ ]:


train_transform, valid_transform, tta_transform = get_transform(img_size=img_size, crop_size=crop_size)

train_dataset = PlantDataset(train_df, dataset_path, train_transform)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sampler_weights, len(train_df))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=4, 
    sampler=train_sampler, shuffle=False
)

val_dataset = PlantDataset(val_df, dataset_path, valid_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, num_workers=4, shuffle=False)


# In[ ]:


criterion = bce
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)


# Pytorchtrainutils provides utilities functions and classes (i.e.) metrics

# In[ ]:


tracked_metrics = [
    metrics.MultilabelAccuracy(),
    metrics.MultilabelRocAuc(),
    metrics.MultilabelConfusionMatrix()
]

best_model = trainer.fit(
    model, train_dataloader=train_loader, val_dataloader=val_loader,
    test_dataloader=None, test_every=0, criterion=criterion,
    optimizer=optimizer, scheduler=lr_scheduler, metrics=tracked_metrics, n_epochs=n_epochs_224,
    metric_choice='col-auc', mode='max',
    name=name, device=device, weight={'train': train_weight, 'val': val_weight}
)


# Let's see how the best model does on the validation set, with 4xTTA.

# In[ ]:


tracked_metrics = [
    metrics.MultilabelAccuracy(metric='top1-acc', apply_sigmoid=False),
    metrics.MultilabelRocAuc(apply_sigmoid=False),
    metrics.MultilabelConfusionMatrix()
]

softmaxer = Softmaxer(model)
softmaxer_best = Softmaxer(best_model)

val_dataset = PlantDataset(val_df, dataset_path, tta_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, num_workers=4, shuffle=False)

val_logs = trainer.test(softmaxer, criterion=criterion, test_dataloader=val_loader, metrics=tracked_metrics, weight=val_weight, device=device, tta=True)
best_val_logs = trainer.test(softmaxer_best, criterion=criterion, test_dataloader=val_loader, metrics=tracked_metrics, weight=val_weight, device=device, tta=True)


# In[ ]:


print(f'Final {name} val:', trainer.summarize_metrics(val_logs))
print(f'Best {name} val:', trainer.summarize_metrics(best_val_logs))


# In[ ]:


ax = sns.heatmap(val_logs['cm'].get(normalized=True), annot=True, fmt=".2f", vmin=0., vmax=1.)


# In[ ]:


ax = sns.heatmap(best_val_logs['cm'].get(normalized=True), annot=True, fmt=".2f", vmin=0., vmax=1.)


# In[ ]:


plot_cm(best_val_logs)


# In[ ]:


plot_roc_auc(best_val_logs)


# # Train on 448

# Let's increase the resolution up to 448 and apply transfer learning

# In[ ]:


img_size *= 2
crop_size *= 2
name = f'{arch}-{crop_size}'
print(f'{name} image size: {img_size}, crop size: {crop_size}')


# In[ ]:


train_transform, valid_transform, tta_transform = get_transform(img_size=img_size, crop_size=crop_size)

train_dataset = PlantDataset(train_df, dataset_path, train_transform)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sampler_weights, len(train_df))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=4, 
    sampler=train_sampler, shuffle=False
)

val_dataset = PlantDataset(val_df, dataset_path,  valid_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, num_workers=4, shuffle=False)


# In[ ]:


criterion = bce
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)


# In[ ]:


tracked_metrics = [
    metrics.MultilabelAccuracy(apply_sigmoid=True),
    metrics.MultilabelRocAuc(apply_sigmoid=True),
    metrics.MultilabelConfusionMatrix()
]

best_model = trainer.fit(
    model, train_dataloader=train_loader, val_dataloader=val_loader,
    test_dataloader=None, test_every=0, criterion=criterion,
    optimizer=optimizer, scheduler=lr_scheduler, metrics=tracked_metrics, n_epochs=n_epochs_448,
    metric_choice='col-auc', mode='max',
    name=name, device=device, weight={'train': train_weight, 'val': val_weight}
)


# In[ ]:


tracked_metrics = [
    metrics.MultilabelAccuracy(metric='top1-acc', apply_sigmoid=False),
    metrics.MultilabelRocAuc(apply_sigmoid=False),
    metrics.MultilabelConfusionMatrix()
]

softmaxer = Softmaxer(model)
softmaxer_best = Softmaxer(best_model)

val_dataset = PlantDataset(val_df, dataset_path, tta_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, num_workers=4, shuffle=False)

val_logs = trainer.test(softmaxer, criterion=criterion, test_dataloader=val_loader, metrics=tracked_metrics, weight=val_weight, device=device, tta=True)
best_val_logs = trainer.test(softmaxer_best, criterion=criterion, test_dataloader=val_loader, metrics=tracked_metrics, weight=val_weight, device=device, tta=True)


# In[ ]:


print(f'Final {name} val:', trainer.summarize_metrics(val_logs))
print(f'Best {name} val:', trainer.summarize_metrics(best_val_logs))


# In[ ]:


ax = sns.heatmap(val_logs['cm'].get(normalized=True), annot=True, fmt=".2f", vmin=0., vmax=1.)


# In[ ]:


ax = sns.heatmap(best_val_logs['cm'].get(normalized=True), annot=True, fmt=".2f", vmin=0., vmax=1.)


# In[ ]:


plot_cm(best_val_logs)


# In[ ]:


plot_roc_auc(best_val_logs)


# # Inference & submission

# In[ ]:


test_dataset = PlantDataset(test_df, dataset_path, tta_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)


# In[ ]:


def get_test_preds(model, test_loader):
    outputs = []
    for batch_idx, (data, labels) in enumerate(tqdm(test_loader)):
        batch_size, n_crops, c, h, w = data.size()
        data = data.view(-1, c, h, w)
        with torch.no_grad():
            output = model(data.to(device))
        output = output.view(batch_size, n_crops, -1).mean(1)
        outputs.append(output.cpu())

    return torch.cat(outputs, dim=0).numpy()


# In[ ]:


def make_submission_df(df, preds):
    classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
    for idx, name in enumerate(classes):
        df[name] = pd.Series(preds[:, idx])
    return df


# In[ ]:


softmaxer.eval()
test_preds = get_test_preds(softmaxer, test_loader)
test_submission = make_submission_df(test_df, test_preds)
test_submission.to_csv(f'submission-{name}.csv', index=False)


# In[ ]:


softmaxer_best.eval()
test_preds = get_test_preds(softmaxer_best, test_loader)
test_submission = make_submission_df(test_df, test_preds)
test_submission.to_csv(f'submission-best-{name}.csv', index=False)


# In[ ]:


pd.read_csv(f'submission-{name}.csv').head()


# In[ ]:


pd.read_csv(f'submission-best-{name}.csv').head()

