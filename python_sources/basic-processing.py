#!/usr/bin/env python
# coding: utf-8

# # USG Liver health classification
# 
# Useful links:
# - https://pytorch.org/tutorials/
# - https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
# - https://software.intel.com/en-us/articles/hands-on-ai-part-14-image-data-preprocessing-and-augmentation
# - https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9
# - https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# 
# **Import notes**:
# Using a neural network here is just an example. Look at other methods, such as other classifiers (Random Forests, SVMs, all available in `scikit-learn` library) or methods based on color or other features (simple thresholding based on a color histogram).
# 
# Consider also different methods of processing data:
# - using cross validation (`KFold` in `sklearn`)
# - augmentation methods available in `torchvision`

# In[ ]:


from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
from IPython.display import display
from collections import defaultdict

import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import tqdm
import pandas as pd

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch


# ## Data loading

# In[ ]:


train_paths = list(Path("/datasets/usg-kaggle/train/").rglob("circle.png"))
test_paths = list(Path("/datasets/usg-kaggle/test").rglob("circle.png"))


# ## Data analysis

# In[ ]:


print(f"Training size: {len(train_paths)}, Test size: {len(test_paths)}")


# In[ ]:


training_classes = [path.parent.parent.name for path in train_paths]
counts = Counter(training_classes)
display(counts)
plt.bar(counts.keys(), counts.values())


# In[ ]:


f, ax = plt.subplots(8, 4, figsize=(12, 24))
ax = np.asarray(ax).flatten()
random_images = np.random.choice(train_paths, size=64)
plt.axis('off')
for i in range(len(ax)):
    ax[i].imshow(cv2.imread(random_images[i].as_posix()))
    ax[i].set_title("class: " + random_images[i].parent.parent.name)
    ax[i].axis('off')


# In[ ]:


f, ax = plt.subplots(8, 4, figsize=(12, 24))
ax = np.asarray(ax).flatten()
random_images = np.random.choice(test_paths, size=64)
plt.axis('off')
for i in range(len(ax)):
    ax[i].imshow(cv2.imread(random_images[i].as_posix()))
    ax[i].set_title("class: " + random_images[i].parent.parent.name)
    ax[i].axis('off')


# ## Model definition

# In[ ]:


class MLP(nn.Module):
    def __init__(self, input_size=32 * 32):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer_2 = nn.Linear(256, 2)
    
    def forward(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        
        return x
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2)
        )
        
        self.mlp = MLP(5 * 5 * 64)
        
    def forward(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = x.view(-1, 5 * 5 * 64)
        x = self.mlp(x)
        return x
        


# ## Dataset definition

# In[ ]:


class UsgDataset(Dataset):
    def __init__(self, images_paths, is_training):
        super().__init__()
        self.images_paths = images_paths
        self.is_training = is_training
        
    def __getitem__(self, index):
        a_path = self.images_paths[index]
        img = cv2.imread(a_path.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32, 32)).astype(np.float32) / 255
        img = img[np.newaxis]
        
        if self.is_training:
            a_class = a_path.parent.parent.name
        else:
            a_class = a_path.parent.name
        
        expected_output = int(a_class)
        return img, expected_output
    
    def __len__(self):
        return len(self.images_paths)


# In[ ]:


train_samples, valid_samples = train_test_split(train_paths)
train_dataset = UsgDataset(train_samples, True)
valid_dataset = UsgDataset(valid_samples, True)
test_dataset = UsgDataset(test_paths, False)

print(f"Train length: {len(train_dataset)}")
print(f"Valid length: {len(valid_dataset)}")
print(f"Test length: {len(test_dataset)}")


# ## Training

# In[ ]:


batch_size = 32
learning_rate = 0.001
num_epochs = 100
use_cuda = torch.cuda.is_available()


# In[ ]:


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True)


# In[ ]:


model = CNN()

loss_function = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters())

# optimiser = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

if use_cuda:
    model = model.cuda()
    loss_function = loss_function.cuda()
    
train_stats = defaultdict(list)
    
for epoch in range(num_epochs):
    print("Epoch: {}/{}".format(epoch + 1, num_epochs))
    
    total_correct = 0
    total_samples = 0
    train_losses = []
    
    model.train()
    
    with tqdm.tqdm_notebook(total=len(train_loader)) as pbar:
        for x_train, y_train in train_loader:
            optimiser.zero_grad()
            if use_cuda:
                x_train, y_train = x_train.cuda(), y_train.cuda()
            
            predicted_output = model(x_train)

            predicted_class = predicted_output.argmax(dim=1)

            total_correct += (predicted_class == y_train).float().sum()
            total_samples += len(y_train)

            loss = loss_function(predicted_output, y_train)
            loss.backward()
            optimiser.step()
            
            train_losses.append(loss.cpu().item())
            
            pbar.set_postfix(loss=loss.cpu().item(), acc=(total_correct / total_samples * 100).cpu().item())
            pbar.update(1)
            
    train_stats["train_loss"].append(np.mean(train_losses))
    train_stats["train_acc"].append(total_correct / total_samples * 100)
            
    validation_correct = 0
    validation_total_samples = 0
    validation_losses = []
    
    model.eval()
    for x_train, y_train in valid_loader:
        if use_cuda:
            x_train, y_train = x_train.cuda(), y_train.cuda()

        predicted_output = model(x_train)

        predicted_class = predicted_output.argmax(dim=1)
        loss = loss_function(predicted_output, y_train)

        validation_correct += (predicted_class == y_train).float().sum().cpu().item()
        validation_total_samples += len(y_train)
        validation_losses.append(loss.cpu().item())
        
    train_stats["valid_loss"].append(np.mean(validation_losses))
    train_stats["valid_acc"].append(validation_correct / validation_total_samples * 100)
        
    print("Validation loss: {:.4f}, validation accuracy: {:.4f}".format(
        np.mean(validation_losses), validation_correct / validation_total_samples * 100
    ))
    


# In[ ]:


plt.figure()
plt.plot(list(range(num_epochs)), train_stats["train_loss"], label="Training")
plt.plot(list(range(num_epochs)), train_stats["valid_loss"], label="Validation")
plt.legend(loc="best")
plt.xlabel("Epoch")
plt.ylabel("Loss (the lower, the better)")
print()


# In general validation plot should be close to the training one. This one, as above, indicates that something is wrong with our model (or data)

# In[ ]:


plt.figure()
plt.plot(list(range(num_epochs)), train_stats["train_acc"], label="Training")
plt.plot(list(range(num_epochs)), train_stats["valid_acc"], label="Validation")
plt.legend(loc="best")
plt.xlabel("Epoch")
plt.ylabel("Loss (the lower, the better)")
print()


# ## Generating submission

# In[ ]:


test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
out_data = defaultdict(list)
model.eval()
for x_train, ids in tqdm.tqdm_notebook(test_loader):
    if use_cuda:
        x_train = x_train.cuda()

    predicted_output = model(x_train)

    predicted_class = predicted_output.argmax(dim=1).cpu().numpy()
    ids = ids.cpu().numpy()
    
    for an_id, a_class in zip(ids, predicted_class):
        out_data["id"].append(an_id)
        out_data["label"].append(a_class)

out_data = pd.DataFrame(out_data)
out_data = out_data.sort_values(by="id")
out_data.to_csv("submission_{}.csv".format(np.mean(validation_losses)), index=False)


# In[ ]:




