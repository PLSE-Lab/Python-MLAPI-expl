#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas'

TRAIN_DIR = DATA_DIR + '/train'                           
TEST_DIR = DATA_DIR + '/test'                             

TRAIN_CSV = DATA_DIR + '/train.csv'                       
TEST_CSV = '../input/jovian-pytorch-z2g/submission.csv' 


# In[ ]:


csv = pd.read_csv(TRAIN_CSV)
csv.head()


# In[ ]:


labels = {
    0: 'Mitochondria',
    1: 'Nuclear bodies',
    2: 'Nucleoli',
    3: 'Golgi apparatus',
    4: 'Nucleoplasm',
    5: 'Nucleoli fibrillar center',
    6: 'Cytosol',
    7: 'Plasma membrane',
    8: 'Centrosome',
    9: 'Nuclear speckles'
}

num_cls = len(labels)
print("Num classes: {}".format(num_cls))


# In[ ]:


"""
encode_lbl

target: target from the dataset
keys: dictionary of classes {id:label}

example -- [2 4 5] => [0 0 1 0 1 1 0 0 0 0] 

"""
def encode_lbl(target, keys=labels):
    
    ohv = torch.zeros(num_cls)
    for cls_id in str(target).split(' '):
        ohv[int(cls_id)] = 1
    
    return ohv
            
"""
decode_lbl

y: target from the dataset/prediction
keys: dictionary of classes {id:label}
show_lbl: Boolean flag to show/hide text label
threshold: Threshold to assign class label

"""
def decode_lbl(y, keys=labels, show_lbl=False, threshold=0.5):
    
    result = []
    for i,v in enumerate(y):
        if (v >= threshold):
            if show_lbl:
                result.append(keys[i] + '-' + str(i))
            else:
                result.append(str(i))
        
    return ' '.join(result)


# In[ ]:


a = encode_lbl('2 4 5')


# In[ ]:


decode_lbl(a, show_lbl=True)


# In[ ]:


# Dataset enhancement

"""
HumanProteinDataset

Reads the image based on the id from dataframe and the dir path.
Returns the (transformed) image along with encoded label.
"""
class HumanProteinDataset(Dataset):
    
    def __init__(self, df, base_dir, tsfm=None):
        self.df = df
        self.base_dir = base_dir
        self.tsfm = tsfm
    
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        
        obs = self.df.loc[idx]
        img_id, img_lbl = obs['Image'], obs['Label']
        im_path = os.path.join(self.base_dir, str(img_id)+".png")
        
        img = Image.open(im_path)
        if self.tsfm:
            img = self.tsfm(img)
        
        return img, encode_lbl(img_lbl)
        


# In[ ]:


imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tfms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(512, padding=8, padding_mode='reflect'),
#     torchvision.transforms.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
#     torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    torchvision.transforms.RandomHorizontalFlip(), 
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize(*imagenet_stats,inplace=True), 
#     torchvision.transforms.RandomErasing(inplace=True)
])

valid_tfms = torchvision.transforms.Compose([
#     torchvision.transforms.Resize(256), 
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize(*imagenet_stats)
])


# In[ ]:


np.random.seed(42)
train_idx = np.random.random(csv.shape[0]) < 0.9

train_csv = csv[train_idx].reset_index(drop=True)
val_csv = csv[~train_idx].reset_index(drop=True)

print(train_csv.shape[0], val_csv.shape[0])


# In[ ]:


train_csv.head()


# In[ ]:


train_ds = HumanProteinDataset(train_csv, TRAIN_DIR, train_tfms)
val_ds = HumanProteinDataset(val_csv, TRAIN_DIR, valid_tfms)

print(len(train_ds[0]), len(val_ds[0]))


# In[ ]:


test_csv = pd.read_csv(TEST_CSV)
test_dataset = HumanProteinDataset(test_csv, TEST_DIR, valid_tfms)


# In[ ]:


test_csv.head()


# In[ ]:


def show_sample(img, target, invert=False):
    
    if invert:
        plt.imshow(1 - img.permute(1, 2, 0))
    else:
        plt.imshow(img.permute(1, 2, 0))
        
    print("Label: {}".format(decode_lbl(target, show_lbl=True)))


# In[ ]:


show_sample(*train_ds[10])


# In[ ]:


show_sample(*train_ds[10], invert=True)


# In[ ]:


batch_size = 16

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=batch_size*2, shuffle=True, num_workers=3, pin_memory=True)

test_dl = DataLoader(test_dataset, batch_size, num_workers=3, pin_memory=True)


# In[ ]:


def F_score(output, label, threshold=0.5, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)


# In[ ]:


class MultilabelImageClassificationBase(torch.nn.Module):
    def training_step(self, batch):
        images, targets = batch 
        out = self(images)                      
        loss = F.binary_cross_entropy(out, targets)      
        return loss
    
    def validation_step(self, batch):
        images, targets = batch 
        out = self(images)                           # Generate predictions
        loss = F.binary_cross_entropy(out, targets)  # Calculate loss
        score = F_score(out, targets)
        return {'val_loss': loss.detach(), 'val_score': score.detach() }
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_score']))


# In[ ]:


class ProteinClassifierTransLearn(MultilabelImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = torchvision.models.resnet34(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = torch.nn.Linear(num_ftrs, 10)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))                


# In[ ]:



def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[ ]:


device = get_default_device()
device


# In[ ]:


train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device)


# In[ ]:


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[ ]:


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    
    for e in range(epochs):
        
        model.train()
        train_losses = []
        lrs = []
        
        for batch in train_loader:
            losses = model.training_step(batch)
            
            train_losses.append(losses)
            losses.backward()
            
            if grad_clip:
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            sched.step()
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        
        model.epoch_end(e, result)
        history.append(result)
        
    return history


# In[ ]:


model = to_device(ProteinClassifierTransLearn(), device)


# In[ ]:


@torch.no_grad()
def predict_dl(dl, model):

    torch.cuda.empty_cache()

    batch_probs = []
    for xb, _ in dl:
        probs = model(xb)
        batch_probs.append(probs.cpu().detach())
    batch_probs = torch.cat(batch_probs)
    return [decode_lbl(x) for x in batch_probs]


# In[ ]:


history = [evaluate(model, val_dl)]

history


# In[ ]:


epochs = 2
max_lr = 5e-3
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam


# In[ ]:


# history += fit(epochs, max_lr, model, train_dl, val_dl, opt_func=torch.optim.Adam)
history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, weight_decay=weight_decay, grad_clip=grad_clip, opt_func=opt_func)


# In[ ]:


weights_fname = 'protein-resnet.pth'
torch.save(model.state_dict(), weights_fname)


# In[ ]:


test_preds = predict_dl(test_dl, model)


# In[ ]:


test_preds[:10]


# In[ ]:


submission_df = pd.read_csv(TEST_CSV)
submission_df.Label = test_preds

print(submission_df.head())
sub_fname = 'resnet_submission.csv'
submission_df.to_csv(sub_fname, index=False)


# In[ ]:


submission_df.to_csv('/kaggle/working/kaggle_submission.csv', index=False)

