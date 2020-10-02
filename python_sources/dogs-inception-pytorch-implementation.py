#!/usr/bin/env python
# coding: utf-8

# In this kernel, I'll be using an object detection model (YOLOv3) to detect the location of the dogs within the images, followed by using the InceptionV3 model re-trained specifically on this dataset to classify the dogs into their unique breeds.
# 
# NOTE: As I'm unable to include the YOLO model in this kernel, this [link](https://github.com/gabrielloye/yolov3-stanford-dogs/blob/master/main.ipynb) to my repository includes the training script for YOLO and how it can be combined with the model trained in this kernel.
# 
# Any feedback on areas of improvement and better practices that I should adopt is definitely welcome.

# In[ ]:


from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler, random_split
import torch.nn as nn

from PIL import Image
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import xml.etree.ElementTree as ET


# After importing the necessary modules, we'll first define a function that finds the bounding boxes using the maximum and minimum x and y values from the annotation files and crops the images accordingly

# In[ ]:


def crop_image(breed, dog, data_dir):
  img = plt.imread(data_dir + 'images/Images/' + breed + '/' + dog + '.jpg')
  tree = ET.parse(data_dir + 'annotations/Annotation/' + breed + '/' + dog)
  xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
  xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
  ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
  ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
  img = img[ymin:ymax, xmin:xmax, :]
  return img


# To help visualise and ensure that we are cropping correctly, I've randomly chosen 4 images and displayed how the images are cropped with the annotations below.

# In[ ]:


data_dir = '../input/stanford-dogs-dataset/'
breed_list = os.listdir(data_dir + 'images/Images/')

plt.figure(figsize=(20, 20))
for i in range(4):
  plt.subplot(421 + (i*2))
  breed = np.random.choice(breed_list)
  dog = np.random.choice(os.listdir(data_dir + 'annotations/Annotation/' + breed))
  img = plt.imread(data_dir + 'images/Images/' + breed + '/' + dog + '.jpg')
  plt.imshow(img)  
  
  tree = ET.parse(data_dir + 'annotations/Annotation/' + breed + '/' + dog)
  xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
  xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
  ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
  ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
  plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
  crop_img = crop_image(breed, dog, data_dir)
  plt.subplot(422 + (i*2))
  plt.imshow(crop_img)


# Next, we'll create new folders to store all the cropped images, which we will use to train the classifier

# In[ ]:


if 'data' not in os.listdir():
    os.mkdir('data')
for breed in breed_list:
    os.mkdir('data/' + breed)
print('Created {} folders to store cropped images of the different breeds.'.format(len(os.listdir('data'))))


# And we'll save the cropped images to their breed folders

# In[ ]:


for breed in os.listdir('data'):
    for file in os.listdir(data_dir + 'annotations/Annotation/' + breed):
        img = Image.open(data_dir + 'images/Images/' + breed + '/' + file + '.jpg')
        tree = ET.parse(data_dir + 'annotations/Annotation/' + breed + '/' + file)
        xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
        xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
        ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
        ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
        img = img.crop((xmin,ymin,xmax,ymax))
        img = img.convert('RGB')
        img.save('data/' + breed + '/' + file + '.jpg')


# In[ ]:


img_count = 0
for folder in os.listdir('data'):
    for _ in os.listdir('data/' + folder):
        img_count += 1
print('No. of Images: {}'.format(img_count))


# Great, now all our cropped images are in place. We'll now define the data augmentation and normalizations for the data. The training data will have augmentations to increase the variability of training images. We'll use random rotations, random crops and some other augmentations for the training transforms. Both training and test data will then be resized to 299 * 299 as this is the standard input size for the inception model. The images will be normalized as well according to the ImageNet standards.
# 
# 

# In[ ]:


image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=315, scale=(0.95, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=299),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    'test':
    transforms.Compose([
        transforms.Resize(size=299),
        transforms.CenterCrop(size=299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# We'll now define the batch size (which can be modified depending on how you want to train your model) and split the data into training, validation and test sets.

# In[ ]:


batch_size = 128

all_data = datasets.ImageFolder(root='data')
train_data_len = int(len(all_data)*0.8)
valid_data_len = int((len(all_data) - train_data_len)/2)
test_data_len = int(len(all_data) - train_data_len - valid_data_len)
train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
train_data.dataset.transform = image_transforms['train']
val_data.dataset.transform = image_transforms['test']
test_data.dataset.transform = image_transforms['test']
print(len(train_data), len(val_data), len(test_data))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# In[ ]:


trainiter = iter(train_loader)
features, labels = next(trainiter)
print(features.shape, labels.shape)


# After making sure that our data loaders are outputting data of the correct dimensions (batch_size, 3, 299, 299), we can now instantiate the model by loading it with pre-trained weights. Specifically for the Inception model, we'll have to set the aux_logits property to False. We can then see the model architecture(which is very long).

# In[ ]:


model = models.inception_v3(pretrained=True)
model.aux_logits=False
model


# Next, we'll freeze the earlier layers in the model as we do not want to re-train these weights, which are responsible for the model's success in picking out key features of the image.

# In[ ]:


# Freeze early layers
for param in model.parameters():
    param.requires_grad = False


# Instead, the only layers which we will be training is the classifier layer, or defined as the fc layer for the case of the Inception model. We have to ensure that the final output will be 120 as we have 120 possible different classes.

# In[ ]:


n_classes = 120
n_inputs = model.fc.in_features
# n_inputs will be 4096 for this case
# Add on classifier
model.fc = nn.Sequential(
    nn.Linear(n_inputs, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, n_classes),
    nn.LogSoftmax(dim=1))

model.fc


# Sending the model to GPU, if not it'll take forever to train. This is followed by defining the criterion and optimizer.

# In[ ]:


model.cuda()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


# Creating an index for the 120 different classes/breeds below

# In[ ]:


model.class_to_idx = all_data.class_to_idx
model.idx_to_class = {
    idx: class_
    for class_, idx in model.class_to_idx.items()
}

list(model.idx_to_class.items())


# Now, we'll define the training function which will run over the specified number of epochs. It will take the data from the train loader and update the weights of the fc layer according to the labels.
# 
# Every 2 epochs, the function will do validation using the data from the validation loader and ensure that the loss is still decreasing. If the loss stops decreasing for a certain number of epochs (which is defined through the early_stop hyper-parameter), the model will stop training and save the previous the best weights instead

# In[ ]:


def train(model,
         criterion,
         optimizer,
         train_loader,
         val_loader,
         save_location,
         early_stop=3,
         n_epochs=20,
         print_every=2):
    #Initializing some variables
    valid_loss_min = np.Inf
    stop_count = 0
    valid_max_acc = 0
    history = []
    model.epochs = 0

    #Loop starts here
    for epoch in range(n_epochs):
        
        train_loss = 0
        valid_loss = 0

        train_acc = 0
        valid_acc = 0

        model.train()
        ii = 0

        for data, label in train_loader:
            ii += 1
            data, label = data.cuda(), label.cuda()
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1) # first output gives the max value in the row(not what we want), second output gives index of the highest val
            correct_tensor = pred.eq(label.data.view_as(pred)) # using the index of the predicted outcome above, torch.eq() will check prediction index against label index to see if prediction is correct(returns 1 if correct, 0 if not)
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor)) #tensor must be float to calc average
            train_acc += accuracy.item() * data.size(0)
            if ii%10 == 0:
                print(f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete.')

        model.epochs += 1
        with torch.no_grad():
            model.eval()

            for data, label in val_loader:
                data, label = data.cuda(), label.cuda()

                output = model(data)
                loss = criterion(output, label)
                valid_loss += loss.item() * data.size(0)

                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(label.data.view_as(pred))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                valid_acc += accuracy.item() * data.size(0)

            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(val_loader.dataset)

            train_acc = train_acc / len(train_loader.dataset)
            valid_acc = valid_acc / len(val_loader.dataset)

            history.append([train_loss, valid_loss, train_acc, valid_acc])

            if (epoch + 1) % print_every == 0:
                print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
                print(f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')

            if valid_loss < valid_loss_min:
                torch.save({
                    'state_dict': model.state_dict(),
                    'idx_to_class': model.idx_to_class
                }, save_location)
                stop_count = 0
                valid_loss_min = valid_loss
                valid_best_acc = valid_acc
                best_epoch = epoch

            else:
                stop_count += 1

                # Below is the case where we handle the early stop case
                if stop_count >= early_stop:
                    print(f'\nEarly Stopping Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
                    model.load_state_dict(torch.load(save_location)['state_dict'])
                    model.optimizer = optimizer
                    history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc','valid_acc'])
                    return model, history

    model.optimizer = optimizer
    print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')

    history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history


# In[ ]:


model, history = train(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    save_location='./dog_inception.pt',
    early_stop=3,
    n_epochs=30,
    print_every=2)


# In[ ]:


history


# After training for just a few epochs, we can see that the accuracy on the validation training set is around 75%. To ensure that the model isn't overfitting for the training and validation data, let's test the model out using the test data which it has not seen at all yet.

# In[ ]:


def test(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        test_acc = 0
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()

            output = model(data)

            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(label.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            test_acc += accuracy.item() * data.size(0)

        test_acc = test_acc / len(test_loader.dataset)
        return test_acc


# In[ ]:


model.load_state_dict(torch.load('./dog_inception.pt')['state_dict'])
test_acc = test(model.cuda(), test_loader, criterion)
print(f'The model has achieved an accuracy of {100 * test_acc:.2f}% on the test dataset')


# The accuracy we achieved is not too bad considering that some dog breeds can be extremely similar and we may not be able to distinguish them with our eyes ourselves

# In[ ]:


def evaluate(model, test_loader, criterion):
  
    classes = []
    acc_results = np.zeros(len(test_loader.dataset))
    i = 0

    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.cuda(), labels.cuda()
            output = model(data)

            for pred, true in zip(output, labels):
                _, pred = pred.unsqueeze(0).topk(1)
                correct = pred.eq(true.unsqueeze(0))
                acc_results[i] = correct.cpu()
                classes.append(model.idx_to_class[true.item()][10:])
                i+=1

    results = pd.DataFrame({
      'class': classes,
      'results': acc_results    
    })
    results = results.groupby(classes).mean()

    return results


# In[ ]:


print(evaluate(model, test_loader, criterion))


# The table above shows the accuracy of our model on the specific breeds, with a score of 1 showing that our model classified that breed correctly. As we can see, some of the breeds have rather bad accuracy, probably because of their similarities, which the model could not pick out the fine-grained differences.

# To-Do:
# - Improve the model by possibly unfreezing some layers and training them.

# In[ ]:


get_ipython().system('rm -rf  data/*')
#deleting the data of the cropped images we saved earlier on so this notebook can be posted

