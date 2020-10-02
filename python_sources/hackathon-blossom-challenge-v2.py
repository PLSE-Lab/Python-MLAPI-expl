#!/usr/bin/env python
# coding: utf-8

# Model: I have used the Densenet 201 pretrained model, froze the first layers and retrained the rest.
# I have used optimizer SGD with momentum, a learning rate of 0.001 and a sheduler StepLR to decrease the learning rate with steps of 5-7 epochs. I trained first with frozen parameters, then retrained the same model with unfrozen parameters and with different hyperparameters.
# Most of the code is taken from a previous Pytorch challenge where I have used different resources, mainly Pytorch Tutorials and Udacity code. Unfortunately the classifier I use here to produce the results is not the one I have trained. I lost my work a couple of times and not everything was saved.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
import numpy as np
import torch

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from collections import OrderedDict
import time
import copy
from PIL import Image

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data_dir = '../input/hackathon-blossom-flower-classification/'
train_dir = data_dir + 'flower_data/flower_data/train'
valid_dir = data_dir + 'flower_data/flower_data/valid'
test_dir = data_dir + 'test set/test set/'


# In[ ]:


import json

with open('../input/hackathon-blossom-flower-classification/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# In[ ]:


# Data augmentation and normalization for training
# Just normalization for validation
batch_size=32
num_workers=0

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.Resize(226),
        transforms.CenterCrop(224),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.ColorJitter(hue=.08, saturation=.08),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(226),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    'test': transforms.Compose([
        transforms.Resize(226),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
}


image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid':datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test':datasets.ImageFolder('../input/hackathon-blossom-flower-classification/test set/', transform=data_transforms['test'])
}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
              for x in ['train', 'valid', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


classes=[]
for i in image_datasets['train'].classes:
    classes.append(cat_to_name[i])


# In[ ]:


from PIL import Image
def imshow1(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)


# In[ ]:


images, labels = next(iter(dataloaders['train']))                      
print(labels)
#images = images.numpy()
# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(35, 6))
for idx in np.arange(batch_size):
    ax = fig.add_subplot(batch_size/16, batch_size/2, idx+1, xticks=[], yticks=[])
    imshow1(images[idx])
    ax.set_title(classes[labels[idx]])


# In[ ]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        """if epoch > 20:
            for param in model_ft.parameters():
              param.requires_grad = True"""
            
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                print('Accuracy increased ({:.6f} --> {:.6f}).  Saving model checkpoint13jul ...'.format(best_acc, epoch_acc))
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # save model if loss has decreased
                model.cpu()
                torch.save({'epoch': epoch + 1,
                                 'state_dict': model.state_dict(),
                                 'optimizer' : optimizer.state_dict(),
                                 'model': 'densenet161',
                                 'loss': loss,
                                 'class_to_idx': model_ft.class_to_idx},
                                 './checkpoint1.pth'
                                )
                model.cuda()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[ ]:


model_ft = models.densenet201(pretrained=True)
model_ft.class_to_idx = image_datasets['train'].class_to_idx
model_ft.classifier.cpu()


# In[ ]:


# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
print(torch.cuda.is_available())
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    


# In[ ]:


n_inputs = 1920
# Freeze parameters so we don't backprop through them


  
for i in range(len(model_ft.features)-6):
    model_ft.features[i].requires_grad = False

for i in range(6,12):
    model_ft.features[i].requires_grad=True
    
""" 
ct = 0
for child in model_ft.children():
  ct += 1
  if ct < 10:
    for param in child.parameters():
        param.requires_grad = False"""

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(n_inputs, 500)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.4)),
                          ('fc2', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model_ft.classifier = classifier
model_ft = model_ft.to(device)

# Criteria NLLLoss which is recommended with Softmax final layer
criterion = nn.NLLLoss()

# Observe that all parameters are being optimized
#optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
#try with
#optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[ ]:


#model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=25)


# In[ ]:


#torch.save({'model': 'densenet201',
#            'state_dict': model_ft.state_dict(), 
#           'class_to_idx': model_ft.class_to_idx}, 
#            'classifier13_1.pt')


# In[ ]:


#unfreeze parameters
#for param in model_ft.parameters():
#    param.requires_grad = True

#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[ ]:


#model_rt = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)


# In[ ]:


#torch.save({'model': 'densenet201',
#            'state_dict': model_ft.state_dict(), 
#            'class_to_idx': model_ft.class_to_idx}, 
#            'classifier13_98.pt')


# In[ ]:


def load_model(checkpoint_path):
    chpt = torch.load(checkpoint_path)
    
    if chpt['model'] == 'densenet201':
        model_ft = models.densenet201(pretrained=False)
        for param in model_ft.parameters():
            param.requires_grad = False
    else:
        print("Sorry base architecture note recognized")
      
    
    model_ft.class_to_idx = chpt['class_to_idx']
    from collections import OrderedDict
    # Create the classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(n_inputs, 500)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    # Put the classifier on the pretrained network
    model_ft.classifier = classifier
    
    model_ft.load_state_dict(chpt['state_dict'])
    
    return model_ft


# In[ ]:


# I am reloading a previously trained model because my kernel just shut down 
#suddendly while I was finishing up 
model_rl =load_model('../input/hackathon-blossom-challenge-v2/classifier13_98.pt')
model_rl.cuda()


# In[ ]:


def visualize_model(model, num_images=8):
    model.cuda()
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['valid']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            print(labels)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            print(inputs.size()[0])
            
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far) #subplot(nrows, ncols, index, **kwargs)
                #ax = plt.subplot(num_images//2, 2, j+1, xticks=[], yticks=[])
                ax.axis('off')
                ax.set_title('{}'.format(classes[preds[j]]) + ' ({})'.format(classes[labels[j]]), color=("green" if preds[j]==labels[j] else "red"))
               
                imshow1(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# In[ ]:


visualize_model(model_rl)


# In[ ]:


def calc_accuracy(model, data):
    model.eval()
    model.to(device)    
    
    with torch.no_grad():
        batch_accuracy=[]
        for idx, (inputs, labels) in enumerate(data):
            inputs, labels = inputs.cuda(), labels.cuda()
            # obtain the outputs from the model
            outputs = model.forward(inputs)
            # max provides the (maximum probability, max value)
            _, predicted = outputs.max(dim=1)
            # check the 
            #if idx == 0:
                #print("predicted flower type", predicted) #the predicted class
                #print("label data", labels.data)
                #print("how sure is model", torch.exp(_)) # the predicted probability
            equals = predicted == labels.data
            #if idx == 0:
               # print("correct or not", equals)
            #print("accuracy for each batch", equals.float().mean())
            batch_accuracy.append(equals.float().mean().cpu().numpy())
            mean_acc = np.mean(batch_accuracy)
            print("Mean accuracy: {}".format(mean_acc))
    return mean_acc


# In[ ]:


calc_accuracy(model_rl, dataloaders['valid'])


# In[ ]:


# track test loss 
def calculate_loss(model, dataloader):
    test_loss = 0.0
    class_correct = list(0. for i in range(102))
    class_total = list(0. for i in range(102))

    model.eval() # eval mode

    # iterate over test data
    for data, target in dataloader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update  test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(len(data)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate avg test loss
    test_loss = test_loss/len(dataloader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))
      
    
    


# In[ ]:


calculate_loss(model_rl, dataloaders['valid'])


# **Prediction results**

# In[ ]:


def predict_flowername(image_path, model):
    
    # Process image
    img = transform_image(image_path)
        
    # Predict     
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    #remove alpha channel in png
    image_tensor = image_tensor[:3,:,:]
    #print(image_tensor.shape)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    #print(model_input.shape)
    # Probs
    probs = torch.exp(model.forward(model_input.cuda()))
    
    # Top probs
    probs, labs = probs.topk(5)
    #prob=probs[0].cpu()
    #print(labs)
    lab = labs[0].cpu()
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    label = idx_to_class[lab[0].item()]
    flower_name = cat_to_name[label]

    return flower_name


# In[ ]:


def transform_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
    # Open the image
    from PIL import Image
    img = Image.open(image_path)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 226))
    else:
        img.thumbnail((226, 10000))

    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    #img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
   # print(img.shape)
    
    return img


# In[ ]:


#create list of url and flower names
data = []
for img_filename in os.listdir(test_dir):
    image_path= test_dir + img_filename
    
    flower_name = predict_flowername(image_path, model_rl)
   # print(image_path, flower_name)
    if flower_name != None:
        data.append([img_filename, flower_name])    


# In[ ]:


len(data)


# In[ ]:


df = pd.DataFrame(data, columns = ['file','species'])
pd.set_option('display.max_colwidth', -1)
result = df.sort_values(['file', 'species'], ascending=[1, 0])
result


# In[ ]:


df.to_csv(r'./flowers_FP.csv')


# In[ ]:


def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
    # Open the image
    from PIL import Image
    img = Image.open(image_path)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 226))
    else:
        img.thumbnail((226, 10000))
    # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img


# In[ ]:


def predict(image_path, model, top_num=5):
    # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers


# In[ ]:


import seaborn as sns
def plot_solution(image_path, model):
    # Set up plot
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    # Set up title
    #flower_num = image_path.split('/')[path_num]
    #print(flower_num)
    
    # Plot flower
    
    img = process_image(image_path)
    
    # Make prediction
    probs, labs, flowers = predict(image_path, model.cpu()) 
    title_ = cat_to_name[labs[0]]
    imshow(img, ax, title = title_);
    print(labs)
    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    plt.show()


# In[ ]:


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is first
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# In[ ]:


image_path = "../input/hackathon-blossom-flower-classification/test set/test set/nic23.jpg"
plot_solution(image_path, model_rl)


# In[ ]:


image_path = "../input/hackathon-blossom-flower-classification/test set/test set/ab42.jpg"
plot_solution(image_path, model_rl)


# In[ ]:


image_path = "../input/hackathon-blossom-flower-classification/test set/test set/ab44.jpg"
plot_solution(image_path, model_rl)


# In[ ]:


image_path = "../input/hackathon-blossom-flower-classification/test set/test set/ab49.jpg"
plot_solution(image_path, model_rl)


# In[ ]:


image_path = "../input/hackathon-blossom-flower-classification/test set/test set/ab50.jpg"
plot_solution(image_path, model_rl)


# In[ ]:


image_path = "../input/hackathon-blossom-flower-classification/test set/test set/ab52.jpeg"
plot_solution(image_path, model_rl)

