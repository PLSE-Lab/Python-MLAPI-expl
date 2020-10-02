#!/usr/bin/env python
# coding: utf-8

# # Intel-Image-classification

# ## Importing the modules

# In[ ]:


import torch
import torchvision
import torch.nn as nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.data import random_split
import torchvision.transforms as transforms
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Preparing the data

# In[ ]:


labels = {
    0: "buildings",
    1: "forest",
    2: "glacier",
    3: "mountain",
    4: "sea",
    5: "street"
}


# In[ ]:


def encode_label(label):
    target = torch.zeros(6)
    for l in str(label).split(" "):
        target[int(l)] = 1.
    return target
    
def decode_target(target, text_labels=False, threshold=0.5):
    result = []
    for i, x in enumerate(target):
        if (x >=threshold):
            if text_labels:
                result.append(labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return " ".join(result)


# In[ ]:


def show_sample(img, target, invert=True):
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute(1, 2, 0))
    print('Labels:', decode_target(target, text_labels=True))
    
show_sample(*train_ds[200])


# In[ ]:


transform_ds= transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

transform_dt= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.5, .5, .5], [.5, .5, .5])
])

ds = torchvision.datasets.ImageFolder(root= "../input/intel-image-classification/seg_train/seg_train", 
                                      transform=transform_ds)

test_ds = torchvision.datasets.ImageFolder(root= "../input/intel-image-classification/seg_test/seg_test", 
                                           transform=transform_dt)


# In[ ]:


train_ds_size = 12631
val_ds_size = 1403
train_ds, val_ds = random_split(ds, [train_ds_size, val_ds_size])


# In[ ]:


batch_size = 128
train_dl = DataLoader(ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=3, pin_memory=True)


# In[ ]:


def show_batch(train_dl):
    for images, labels in train_dl:
        fig, ax = plt.subplots(figsize=(12,12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1,2,0))
        break


# In[ ]:


show_batch(train_dl)


# ## Defining the model

# In[ ]:


def accuracy(out, labels):
    _, preds = torch.max(out, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}
    
    def validation_epoch_end(self, outputs):
        batch_loss = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
            print("Epoch: [{}], last_lr: {:.8f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result["lrs"][-1], result["train_loss"], result["val_loss"], result["val_acc"]))


# In[ ]:


class IntelResnet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(pretrained=True)
        number_of_features = self.network.fc.in_features
        self.network.fc = nn.Linear(number_of_features, 10)
        
    def forward(self, xb):
        return self.network(xb)
        
    def freeze(self):
        for param in self.network.parameters():
            param.require_grad = False
        for param in self. network.fc.parameters():
            param.require_grad = True
            
    def unfreeze(self):
        for param in self.network.parameters():
            param.require_grad = True


# In[ ]:


class IntelModel(ImageClassificationBase):
    
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        
        nn.Conv2d(16, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        
        nn.Flatten(),
        nn.Linear(256*2*2, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10))
        
    def forward(self, xb):
        return self.cnn(xb)
    
model = IntelModel()
model


# ## Moving the model

# In[ ]:


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for x in self.dl:
            yield to_device(x, self.device)
            
    def __len__(self):
        return len(self.dl)


# In[ ]:


device = get_default_device()
device


# In[ ]:


train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)


# In[ ]:


@torch.no_grad()
def evaluate(model, val_dl):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_dl]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_dl):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            sched.step()
            
        result = evaluate(model, val_dl)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[ ]:


#model_resnet18 = to_device(IntelResnet(), device)
model_resnet34 = to_device(IntelResnet(), device)


# ## Evaluating

# In[ ]:


history = [evaluate(model_resnet18, val_dl)]
history


# In[ ]:


history = [evaluate(model_resnet34, val_dl)]
history


# In[ ]:


model_resnet34.freeze()


# In[ ]:


epochs = 5
max_lr = 10e-4
grad_clip = 0.1
weight_decay = 1e-3
opt_func = torch.optim.Adam


# In[ ]:


history += fit_one_cycle(epochs, max_lr, model_resnet34, train_dl, val_dl,
             grad_clip=grad_clip, weight_decay=weight_decay,
             opt_func=opt_func)
history


# In[ ]:


model.unfreeze()


# In[ ]:


epochs = 10
max_lr = 10e-5
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam


# In[ ]:


history = fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
             grad_clip=grad_clip, weight_decay=weight_decay,
             opt_func=opt_func)
history


# ## Let's have a look at the performance of the model

# In[ ]:


def plot_accuracy(history):
    scores = [x["val_acc"] for x in history]
    plt.plot(scores, "-x")
    plt.xlabel("epoch")
    plt.ylabel("val_acc")
    plt.title("Accuracy vs. number of epochs")

plot_accuracy(history)


# In[ ]:


def plot_loss(history):
    loss = [x["val_loss"] for x in history]
    plt.plot(loss, "-x")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs. number of epochs")
    
plot_loss(history)


# ## Let's make some predictions

# In[ ]:


def prediction(images, model):
    xb = to_device(images.unsqueeze(0), device)
    out = model(xb)
    _, preds = torch.max(out, dim=1)
    prediction = preds[0].item()
    return prediction


# In[ ]:


images, labels = test_ds[16]
print("Label: ", labels, "Prediction:", prediction(images, model))
plt.imshow(images.permute(1,2,0))


# In[ ]:


images, labels = test_ds[400]
print("Label: ", labels, prediction(images, model))
plt.imshow(images.permute(1,2,0))


# In[ ]:




