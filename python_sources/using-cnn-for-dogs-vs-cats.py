#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, datasets
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


torch.__version__


# In[ ]:


import sys
sys.version


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu : %s ' % torch.cuda.is_available())


# ## data processing

# In[ ]:


data_dir = '../input/dogscats/dogscats/dogscats/'
print(os.listdir('../input/dogscats/dogscats/dogscats/'))


# In[ ]:


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])
vgg_format = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


# In[ ]:


dsets = {x:datasets.ImageFolder(os.path.join(data_dir, x), vgg_format)
        for x in ['train', 'valid']}


# In[ ]:


dsets['train'].class_to_idx


# In[ ]:


dset_sizes = {x : len(dsets[x]) for x in ['train', 'valid']}
dset_sizes


# In[ ]:


dset_classes = dsets['valid'].classes
dset_classes


# In[ ]:


load_train = torch.utils.data.DataLoader(dsets['train'], batch_size=64, 
                                        shuffle=True, num_workers=6)


# In[ ]:


load_test = torch.utils.data.DataLoader(dsets['valid'], batch_size=5, 
                                        shuffle=True, num_workers=6)


# In[ ]:


count = 1
for data in load_test:
    print(count , end=',')
    if count == 1:
        inputs_try, labels_try = data
    count += 1


# In[ ]:


labels_try


# In[ ]:


inputs_try.shape


# In[ ]:


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = np.clip(std * inp + mean, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# In[ ]:


out = torchvision.utils.make_grid(inputs_try)
imshow(out, title=[dset_classes[x] for x in labels_try])


# In[ ]:


inputs, classes = next(iter(load_train))

n_images =  8

out = torchvision.utils.make_grid(inputs[0:n_images])

imshow(out, title=[dset_classes[x] for x in classes[0:n_images]])


# In[ ]:


inputs, classes = next(iter(load_test))

n_images =  8

out = torchvision.utils.make_grid(inputs[0:n_images])

imshow(out, title=[dset_classes[x] for x in classes[0:n_images]])


# ### Creating VGG Model

# In[ ]:


model_vgg = models.vgg16(pretrained=True)


# In[ ]:


import json
fpath = '../input/imagenet-class-index/imagenet_class_index.json'
with open(fpath) as f:
    class_dict = json.load(f)


# In[ ]:


dic_imagenet = [class_dict[str(i)][1] for i in range(len(class_dict))]


# In[ ]:


inputs_try, lables_try = inputs_try.to(device), labels_try.to(device)

model_vgg = model_vgg.to(device)


# In[ ]:


outputs_try = model_vgg(inputs_try)


# In[ ]:


outputs_try


# In[ ]:


outputs_try.shape


# In[ ]:


m_softm = nn.Softmax(dim=1)
probs = m_softm(outputs_try)
vals_try, preds_try = torch.max(probs, dim=1)


# In[ ]:


torch.sum(probs, 1)


# In[ ]:


vals_try


# In[ ]:


print([dic_imagenet[i] for i in preds_try.data])


# In[ ]:


out = torchvision.utils.make_grid(inputs_try.data.cpu())
imshow(out, title=[dset_classes[x] for x in labels_try.data.cpu()])


# ### Modifying the last layer and setting the gradient false to all layers

# In[ ]:


print(model_vgg)


# In[ ]:


for param in model_vgg.parameters():
    param.requires_grad = False
model_vgg.classifier._modules['6'] = nn.Linear(4096, 2)
model_vgg.classifier._modules['7'] =  torch.nn.LogSoftmax(dim=1)


# `LogSoftmax`:
#         $$LogSoftmax(x_i) = log(\frac{exp(x_i)}{\sum_jexp(x_j)})$$

# In[ ]:


print(model_vgg.classifier)


# In[ ]:


model_vgg = model_vgg.to(device)


# ### Training fully connected module

# #### Creating loss function and optimizer

# In[ ]:


criterion = nn.NLLLoss()
lr = 0.001
optimizer_vgg = torch.optim.SGD(model_vgg.classifier[6].parameters(), lr = lr)


# #### training the model

# In[ ]:


def train_model(model, dataloader, size, epochs=1, optimizer=None):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        for inputs, classes in dataloader:
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, classes)
            optimizer = optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs.data, 1)
            #statistics
            running_loss += loss.data.item()
            running_corrects += torch.sum(preds == classes.data)
        epoch_loss = running_loss / size
        epoch_acc = running_corrects.data.item() / size
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


# In[ ]:


get_ipython().run_line_magic('time', '')
train_model(model_vgg, load_train,
            size=dset_sizes['train'], epochs=2, optimizer=optimizer_vgg)


# In[ ]:


def test_model(model, dataloader, size):
    model.eval()
    predictions = np.zeros(size)
    all_classes = np.zeros(size)
    all_proba = np.zeros((size, 2))
    i = 0
    running_loss = 0.0
    running_corrects = 0
    for inputs, classes in dataloader:
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, classes)
        _, preds = torch.max(outputs.data, 1)
        # statistics
        running_loss += loss.data.item()
        running_corrects += torch.sum(preds == classes.data)
        predictions[i : i + len(classes)] = preds.to('cpu').numpy()
        all_classes[i : i + len(classes)] = classes.to('cpu').numpy()
        all_proba[i : i + len(classes)] = outputs.data.to('cpu').numpy()
        
        i += len(classes)
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.data.item() / size
    print('Loss : {:.4f} Acc : {:.4f}'.format(epoch_loss, epoch_acc))
    return predictions, all_proba, all_classes


# In[ ]:


get_ipython().run_line_magic('time', '')
predictions, all_proba, all_classes = test_model(model_vgg, load_test, size=
                                                dset_sizes['valid'])


# In[ ]:


# Get a batch of training data
inputs, classes = next(iter(load_test))

out = torchvision.utils.make_grid(inputs[0: n_images])

imshow(out, title=[dset_classes[x] for x in classes[0:n_images]])


# In[ ]:


outputs = model_vgg(inputs[:n_images].to(device))
print(torch.exp(outputs))


# In[ ]:


classes[:n_images]


# ### Speeding up the learning by precomputing features

# In[ ]:


x_try = model_vgg.features(inputs_try)
x_try.shape


# In[ ]:


def preconvfeat(dataloader):
    conv_features = []
    labels_list = []
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        x = model_vgg.features(inputs)
        conv_features.extend(x.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())
        conv_features = np.concatenate([[feat] for feat in conv_features])
        
        return (conv_features, labels_list)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'conv_feat_train, labels_train = preconvfeat(load_train)')


# In[ ]:


conv_feat_train.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', 'conv_feat_valid, labels_valid = preconvfeat(load_test)')


# ### Creating a new data generator

# In[ ]:


def data_gen(conv_feat, labels, batch_size=64, shuffle=True):
    labels = np.array(labels)
    if shuffle:
        index = np.random.permutation(len(conv_feat))
        conv_feat = conv_feat[index]
        labels = labels[index]
    for idx in range(0, len(conv_feat), batch_size):
        yield(conv_feat[idx:idx + batch_size], labels[idx : idx + batch_size])


# In[ ]:


def train_model(model, size, epochs=1, optimizer=None):
    model.train()
    
    for epoch in range(epochs):
        dataloader = data_gen(conv_feat_train, labels_train)
        running_loss = 0.0
        running_corrects = int(0)
        for inputs, classes in dataloader:
            
            inputs, classes = torch.from_numpy(inputs).to(device), torch.from_numpy(classes).to(device)
            inputs = inputs.to(device)
            classes = classes.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            loss = criterion(outputs, classes)
            optimizer = optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs.data, 1)
            # statistics
            running_loss += loss.data.item()
            running_corrects += torch.sum(preds == classes.data)
        epoch_loss = running_loss / size
        epoch_acc = running_corrects.item() / size
        print('Loss : {:.4f} Acc : {:.4f}'.format(epoch_loss, epoch_acc))


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_model(model_vgg.classifier, size=dset_sizes['train'], epochs=50,\n           optimizer=optimizer_vgg)")


# In[ ]:


def test_model(model, size):
    model.eval()
    predictions = np.zeros(size)
    all_classes = np.zeros(size)
    all_proba = np.zeros((size, 2))
    i = 0
    running_loss = 0.0
    running_corrects = 0
    for inputs, classes in data_gen(conv_feat_valid, labels_valid, shuffle=False):
        inputs, classes = torch.from_numpy(inputs).to(device),torch.from_numpy(classes).to(device)
        inputs = inputs.to(device)
        classes = classes.to(device)
        inputs = inputs.view(inputs.size(0), -1) 
        outputs = model(inputs)
        loss = criterion(outputs, classes)
        _, preds = torch.max(outputs.data, 1)
        # statistics
        running_loss += loss.data.item()
        running_corrects += torch.sum(preds == classes.data)
        #print(i)
        predictions[i:i+len(classes)] = preds.to('cpu').numpy()
        all_classes[i : i + len(classes)] = classes.to('cpu').numpy()
        all_proba[i : i + len(classes),:] = outputs.data.to('cpu').numpy()
        #print(preds.cpu())
        i += len(classes)
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.data.item() / size
    print('Loss {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return predictions , all_proba, all_classes
        


# In[ ]:


predictions, all_proba, all_classes = test_model(model_vgg.classifier,size=dset_sizes['valid'])


# ### Viewing model prediction

# In[ ]:


n_view = 8


# In[ ]:


correct = np.where(predictions == all_classes)[0]
correct


# In[ ]:


len(correct) / dset_sizes['valid']


# In[ ]:


from numpy.random import random, permutation
idx = permutation(correct)[:n_view]


# In[ ]:


idx


# In[ ]:


loader_correct = torch.utils.data.DataLoader([dsets['valid'][x] for x
                                             in idx], batch_size = n_view, shuffle=True)


# In[ ]:


for data in loader_correct:
    inputs_cor, labels_cor = data


# In[ ]:


out = torchvision.utils.make_grid(inputs_cor)
imshow(out, title=[l.item() for l in labels_cor])


# In[ ]:


from IPython.display import Image, display
for x in idx:
    display(Image(filename=dsets['valid'].imgs[x][0], retina=True))


# In[ ]:


incorrect = np.where(predictions!=all_classes)[0]
for x in permutation(incorrect)[:n_view]:
    display(Image(filename=dsets['valid'].imgs[x][0], retina=True))


# In[ ]:


correct_cats = np.where((predictions == 0) & (predictions == all_classes))[0]
most_correct_cats = np.argsort(all_proba[correct_cats, 0])[:n_view]


# In[ ]:


for x in most_correct_cats:
    display(Image(filename=dsets['valid'].imgs[correct_cats[x]][0], retina=True))


# In[ ]:


#3. The images we most confident were dogs, and are actually dogs
correct_dogs = np.where((predictions==1) & (predictions==all_classes))[0]
most_correct_dogs = np.argsort(all_proba[correct_dogs,0])[:n_view]


# In[ ]:


for x in most_correct_dogs:
    display(Image(filename=dsets['valid'].imgs[correct_dogs[x]][0], retina=True))

