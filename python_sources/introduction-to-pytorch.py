#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
from torch import nn, optim
from torch.functional import F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time, copy, os, glob

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt
from PIL import Image


# In[ ]:


import requests, zipfile, io, os
def download_dataset():
    if not os.path.exists("../input/flower-data/"):
        url = 'https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip'
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('../input/')

        with open('../input/flower_data.zip', 'wb') as file:
            file.write(r.content)
        file.close()
    else:
        print("Data directory already exists")


# In[ ]:


# download_dataset()


# In[ ]:


data_dir = '../input/flower-data/flower_data/flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'


# In[ ]:


# TODO: Define your transforms for the training and validation sets
data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                     transforms.RandomRotation(degrees=15),
                                     transforms.ColorJitter(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.CenterCrop(size=224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(root=train_dir, transform=data_transforms),
    'valid': datasets.ImageFolder(root=valid_dir, transform=data_transforms)
}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=36, shuffle=True),
    'valid' : DataLoader(image_datasets['valid'], batch_size=36, shuffle=True)
}


# In[ ]:


import json

with open('../input/flowerjsonmap/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print(len(cat_to_name))


# In[ ]:


# TODO: Build and train your network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[ ]:


# Load pretrained model (VGG-16)
model = models.vgg16(pretrained=True)
model


# In[ ]:


in_features = model.classifier[6].in_features
in_features


# In[ ]:


# Freeze model weights
for param in model.parameters():
    param.requires_grad = (False)
    
model_backup = model


# In[ ]:


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # convolutional layer (sees 224x224x3 image tensor)
        
        self.fc1 = nn.Linear(in_features=in_features, out_features=256)
        self.fc2 = nn.Linear(256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=len(cat_to_name))
        
        self.dropout = nn.Dropout(p=0.4)
        
        # max pooling layer
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # flatten inputs
        x = x.view(-1, in_features)
         
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        x = self.fc4(x)
        return x

classifier = nn.Sequential(
    nn.Linear(in_features=in_features, out_features=256),
    nn.ReLU(),
    nn.Dropout(p=0.4),
    nn.Linear(in_features=256, out_features=len(cat_to_name)),
    nn.LogSoftmax(dim=1)
)
    
net = Network()
net


# In[ ]:


model.classifier[6] = classifier
model.to(device)


# In[ ]:


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[ ]:


d = torch.Tensor(
    [
        [
            [1,2,3], [4,5,6], [1,5,6], [12, 13, 14]
        ],
        [
            [1,2,3], [4,5,6], [1,5,6], [12, 13, 14]
        ]
    ])
d.size()


# In[ ]:


epochs = 13
best_valid_acc = 0.0
min_valid_loss = np.inf
start_time = time.time()
train_losses = []
valid_losses = []
for epoch in range(1, epochs+1):
    running_loss = 0.0
    valid_loss = 0
    for data, target in dataloaders['train']:
        # Remove all accumulated gradients
        optimizer.zero_grad()
        # Move data to device
        data, target = data.to(device), target.to(device)
        # Feed forward the network
        output = model.forward(data)
        # Calculated loss output
        loss = criterion(output, target)
        # Backpropagated weights
        loss.backward()
        # Update gradients
        optimizer.step()
        
        running_loss += loss.item()
    training_loss = running_loss/len(dataloaders['train'])
    print("Epoch: {}/{}.. Running loss: {:.5f}".format(epoch, epochs, training_loss))
    # Data validation
    model.eval()
    valid_acc = 0.0
    for data, target in dataloaders['valid']:
        data, target = data.to(device), target.to(device)
        output = model.forward(data)
        valid_loss += criterion(output, target).item()*data.size(0)
        _, top_classes = output.topk(1, dim=1)
        equality = top_classes == target.view(*top_classes.shape)
        valid_acc += torch.mean(equality.type(torch.FloatTensor))
        
    valid_loss = valid_loss/len(dataloaders['valid'].dataset)
    valid_acc = valid_acc/len(dataloaders['valid'])
    
    # Collect visualization data
    train_losses.append(training_loss)
    valid_losses.append(valid_loss)
    
    print("Validation loss: {:.9f}.. Validation accuracy: {:.9f}".format(valid_loss, valid_acc))
    
    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        
    if valid_acc > best_valid_acc:
        print("---------- Validation accuracy increased ({:.9f} ---> {:.9f}). Saving Model state --------------".format(best_valid_acc, valid_acc))
        best_weights = {
            'state_dict': copy.deepcopy(model.state_dict()),
            'epoch': epoch,
            'class_to_idx': image_datasets['train'].class_to_idx,
            'optimizer': optimizer.state_dict(),
            'valid_loss': valid_loss,
            'valid_acc': valid_acc }
        best_valid_acc = valid_acc
        
    model.train()

time_elapsed = time.time() - start_time
print("\n\nTraining completed in {:.0f}m {:.0f}s\nMinimum validation loss: {:.9f}\nBest validation accuracy: {:.9f}\n\n"
      .format(time_elapsed/60, time_elapsed%60, min_valid_loss, best_valid_acc))
torch.save(best_weights, 'model.pt')


# In[ ]:


# Visualize training stats


# In[ ]:


plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()


# In[ ]:


os.listdir('./..')


# In[ ]:


def load_checkpoint(path='model.pt'):
    checkpoint = torch.load(path)
    print(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
#load_checkpoint()


# In[ ]:


test_images = glob.glob('../input/flower-data/flower_data/flower_data/test/1/*')
test_images


# In[ ]:


#for i in test_images:
fig, axs = plt.subplots(nrows=int(len(test_images)/7), ncols=7, figsize=(6,9))
row = 0
col = 0

for ii, image in enumerate(test_images):
    im = Image.open(image)
    im.thumbnail((604, 604), Image.ANTIALIAS)
    axs[row][col].imshow(im)
    axs[row][col].axis('off')
    if col != 0 and col % 6 == 0: # Index starts at 0
        col = -1
        row += 1
    col += 1

fig.subplots_adjust(hspace=0.3)
plt.show()


# In[ ]:


def process_image(image_path):
    """Imshow for Tensor."""
    image = Image.open(image_path)#.thumbnail(size).crop(224)
    new_size = [0, 0] ##
    if image.size[0] > image.size[1]:
        new_size = [image.size[0], 256]
    else:
        new_size = [256, image.size[1]]
    
    image.thumbnail(new_size, Image.ANTIALIAS)
    width, height = image.size
    
    l,t,r,b = (256 - 224)/2 , (256 - 224)/2, (256 + 224)/2 , (256 + 224)/2
    image = image.crop((l, t, r, b))
    
    image = np.array(image)
    image = image/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    return image


# In[ ]:



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))#.numpy()
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# In[ ]:



def predict(image, model):
    model.to(device)
    model.eval()
    
    # image = process_image(image_path)
    image = torch.from_numpy(image).to(device)
    if torch.cuda.is_available():
        image = image.float()
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logps = model(image)
    ps = torch.exp(logps)
    top_ps, top_classes = ps.topk(5, dim=1)
    return (top_ps.cpu().numpy().squeeze(), top_classes.cpu().numpy().squeeze())


# In[ ]:


def visualize_model(vgg, num_images=6):
    was_training = vgg.training
    
    # Set model for evaluation
    vgg.train(False)
    vgg.eval() 
    
    images_so_far = 0

    for i, data in enumerate(dataloaders[TEST]):
        inputs, labels = data
        size = inputs.size()[0]
        
        if use_gpu:
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
        
        outputs = vgg(inputs)
        
        _, preds = torch.max(outputs.data, 1)
        predicted_labels = [preds[j] for j in range(inputs.size()[0])]
        
        print("Ground truth:")
        show_databatch(inputs.data.cpu(), labels.data.cpu())
        print("Prediction:")
        show_databatch(inputs.data.cpu(), predicted_labels)
        
        del inputs, labels, outputs, preds, predicted_labels
        torch.cuda.empty_cache()
        
        images_so_far += size
        if images_so_far >= num_images:
            break
        
    vgg.train(mode=was_training)


# In[ ]:


# Visualize accuracy probabilities
def display_stats(image_path):
    
    image = process_image(image_path)
    ps, classes = predict(image, model)
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.axis('off')
    # ax1.imshow(process_image(test_images[1]))
    imshow(image, ax=ax1)

    ax2.barh(np.arange(5), ps, color='g')
    ax2.set_aspect(0.2)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels([cat_to_name[str(i)] for i in classes], size='small')
    ax2.set_title('Class Probabilities')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


# In[ ]:


ps, classes = predict(test_images[9], model)
ps, classes


# In[ ]:



display_stats(test_images[9])


# In[ ]:


url = 'https://images.pexels.com/photos/132472/pexels-photo-132472.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260'
def get_image(url):
    r = requests.get(url)
    name = str(time.time()) + '.png'
    with open('../input/{}'.format(name), 'wb') as f:
        f.write(r.content)
    return '../input/'+name

display_stats(get_image(url))


# In[ ]:


def download_model(name='model.pt'):
    # r = request.get('model.pt', allow_redirects=True)


# In[ ]:





# In[ ]:




