#!/usr/bin/env python
# coding: utf-8

# **Stanford car dataset Classification**

# We have to make a model that can tell the difference between cars by type or colour? Which cars are manufactured by Tesla vs BMW?

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import time
import os
import PIL.Image as Image
from IPython.display import display

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(device))


# My work approach:: I used pytorch.imageloader becuase it directly loads dataset from given path like input/flower_data/train or input/flower_data/valid. Earlier i tried to load it using numpy/pandas, then i need to write code for making train and valid set, each having images of 196 categories with proper labeling. Like.. 1 folder will have all images inside 1 floder and label as 1, which we will replace with name(help from json cat_name). But, using pytorch, it was easy to do so with only few lines of code.
# 
# Algoritm Optimization:: We iincreased model accuracy by changing parameters like epoc.We changed epoc from 3 to 9.Accurancy increased slowly slowly.
# 
# Then we pre-process the dataset, using pytorch, rotate,resize , normailze etc methods.
# 
# Algorithm used for this image classification:: Transfer learning becuase making model from scratch would be very complex and computation expensive, would require lot of time. Transfer learning allows us to train deep networks using significantly less data then we would need if we had to train from scratch. In recent years, a number of models have been created for reuse in computer vision problems. Using these pre-trained models is known as transfer learning These pre-trained models allow others to quickly obtain cutting edge results in computer vision without needing the large amounts of compute power, time, and patience in finding the right training technique to optimize the weights. 

# In[ ]:


dataset_dir = "../input/car_data/car_data/"


# In[ ]:


indices = np.arange(8144)
np.random.shuffle(indices)
train_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
val_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_loader = torch.utils.data.DataLoader(dataset = torchvision.datasets.ImageFolder(root=dataset_dir+"train", transform = train_tfms),
batch_size=32, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[:7500]))


val_loader =torch.utils.data.DataLoader(dataset = torchvision.datasets.ImageFolder(root=dataset_dir+"train", transform = val_tfms),
batch_size=32, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[7500:]))


dataset2 = torchvision.datasets.ImageFolder(root=dataset_dir+"test", transform = test_tfms)
testloader = torch.utils.data.DataLoader(dataset2, batch_size = 32, shuffle=False, num_workers = 2)


# In[ ]:


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images[1].shape)


# In[ ]:


grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid, (1,2,0)))


# In[ ]:


train_dir = "../input/car_data/car_data/train"
#os.listdir(train_dir)
os.listdir(train_dir + '/' + 'Acura ZDX Hatchback 2012')


# In[ ]:


train_dir = "../input/car_data/car_data/train"
test_dir = "../input/car_data/car_data/test"
os.listdir(train_dir)
#Dictionary that has name of car class as key and image name as its values
car_names_train = {}

for i in os.listdir(train_dir):
    #print()
    car_names_train[i] = os.listdir(train_dir + '/' + i)
car_names_train


# In[ ]:


for i in os.listdir(test_dir):
    #print()
    car_names_test[i] = os.listdir(test_dir + '/' + i)
car_names_test


# In[ ]:


#Code to create two lists for class name and image directories corresponding to it
car_images_ls = []
car_names_ls = []
car_classes = []
car_directories = []

for i in car_names_train:
    car_classes.append(i)

for i,j in enumerate(car_names_train.values()):
    for img in j:
        car_images_ls.append(img)
        car_names_ls.append(car_classes[i])
        
        
for i in range(len(car_names_ls)):
    car_directories.append(train_dir + '/' + car_names_ls[i] + '/' + car_images_ls[i])
    
car_images_ls[1]


# In[ ]:


def train_model(model_ft, criterion, optimizer, scheduler, n_epochs = 5):
    
    losses = []
    accuracies = []
    val_accuracies = []
    # set the model to train mode initially
    model_ft.train()
    for epoch in range(n_epochs):
        since = time.time()
        running_loss = 0.0
        running_correct = 0.0
        for i, data in enumerate(train_loader, 0):

            # get the inputs and assign them to cuda
            inputs, labels = data
            #inputs = inputs.to(device).half() # uncomment for half precision model
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model_ft(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
             # calculate the loss/acc later
            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()

        epoch_duration = time.time()-since
        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100/32*running_correct/len(train_loader)
        print("Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (epoch+1, epoch_duration, epoch_loss, epoch_acc))
        
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        
        # switch the model to eval mode to evaluate on validation data
        model_ft.eval()
        val_acc = eval_model(model_ft)
        val_accuracies.append(val_acc)
        
        # re-set the model to train mode after validating
        model_ft.train()
        scheduler.step(val_acc)
        since = time.time()
    print('Finished Training')
    return model_ft, losses, accuracies, val_accuracies


# In[ ]:


def eval_model(model_ft):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            images, labels = data
            #images = images.to(device).half() # uncomment for half precision model
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model_ft(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100.0 * correct / total
    print('Accuracy of the network on the validation images: %d %%' % (
        val_acc))
    return val_acc


# In[ ]:


model_ft = models.resnet34(pretrained=True)
num_ftrs = model_ft.fc.in_features

# replace the last fc layer with an untrained one (requires grad by default)
model_ft.fc = nn.Linear(num_ftrs, 196)
model_ft = model_ft.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)


lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)


# In[ ]:


model_ft, training_losses, training_accs, val_accs = train_model(model_ft, criterion, optimizer, lrscheduler, n_epochs=10)


# In[ ]:


colnames=['Name'] 
user1 = pd.read_csv('../input/names.csv',names=colnames)
dict1={}
for i in user1.index:
    dict1[i]=user1['Name'][i]
    
dict1


# In[ ]:


###Testing with random validation samples###
with torch.no_grad():
    model_ft.eval()
    model_ft.eval()
    print ("Evaluating random validation samples:")
    val_data, val_targets = next(iter(val_loader))
    outputs = model_ft(val_data[0].unsqueeze(0).to(device))
    _, pred = torch.max(outputs, 1)
    
    print ("Actual Class - {0}".format(dict1[int(val_targets[0]) ]))
    print ("Predicted Class - {0}".format(dict1[int(pred)]))


# In[ ]:


# plot the stats
f, axarr = plt.subplots(2,2, figsize = (12, 8))
axarr[0, 0].plot(training_losses)
axarr[0, 0].set_title("Training loss")
axarr[0, 1].plot(training_accs)
axarr[0, 1].set_title("Training acc")
axarr[1, 0].plot(val_accs)

axarr[1, 0].set_title("Val acc")


# In[ ]:


#Testing on Test loader.Getting list of image names and class labels.
labels1=[]
image_nm=[]
img_indx=0
with torch.no_grad():
    for i, (images, labels) in enumerate(testloader, 0):
        images = images.to(device)
        outputs = model_ft(images)
        _, predicted = torch.max(outputs.data, 1)
        #print(len(predicted))
        
        for j in range(len(predicted)):
                #print(idx_class_mapping[i.item()])
                labels1.append(int(predicted[j]) + 1)
                sample_fname, _ = testloader.dataset.samples[img_indx]
                img_indx+=1
                sample_fname_file=os.path.splitext(os.path.basename(sample_fname))[0]
                image_nm.append(sample_fname_file)
                #image_nm.append()
        
      #print("{}, {}\n".format(sample_fname[i], predicted[i].item())) 


# In[ ]:


#Making Dataframe for class labels and image names
predicted_df = pd.DataFrame(
    {'Image': image_nm,
     'Prediction': labels1     
    })

predicted_df.to_csv()


# In[ ]:




