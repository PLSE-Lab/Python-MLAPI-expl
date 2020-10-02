#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from IPython.core.debugger import set_trace
import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


all_data = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/"  + "train.csv")
all_data.head()


# # Lets look at the distribution of different training labels in the original data set

# In[ ]:


all_data["diagnosis"].value_counts().plot(kind="pie")


# In[ ]:


train_data, validation_data = train_test_split(all_data, stratify = all_data.diagnosis.values, test_size=0.01)


# lets looks at the distribution of labels in both train and validation set

# In[ ]:



fig, axes = plt.subplots(nrows=1, ncols=2)
train_data.diagnosis.value_counts().plot(kind="pie", ax=axes[0],title="train data")
validation_data.diagnosis.value_counts().plot(kind="pie", ax=axes[1], title="validation data")


# In[ ]:


validation_data.shape


# In[ ]:


class dataSet_(data_utils.Dataset):
    def __init__(self, data_dir, data_frame, image_transform):
        super().__init__()
        self.dir = data_dir
        self.label = data_frame.values[:,0]
        self.diagnosis = data_frame.values[:,1]
        self.transfrom = image_transform

    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        file_to_be_loaded = self.label[idx]        
        array = Image.open(self.dir + file_to_be_loaded + ".png")
        array = self.transfrom(array)
        return array, torch.tensor(self.diagnosis[idx], dtype=torch.float32)
        
        
        


# In[ ]:


train_dir = '/kaggle/input/aptos2019-blindness-detection/train_images/'
valid_dir = '/kaggle/input/aptos2019-blindness-detection/train_images/'
test_dir = '/kaggle/input/aptos2019-blindness-detection/test_images/'


# In[ ]:


image_transform={"train":transforms.Compose([transforms.RandomRotation(degrees=50),transforms.RandomResizedCrop((224,224)),transforms.RandomHorizontalFlip(p=0.5),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),
                "test": transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])}


# In[ ]:


TrainData = dataSet_(train_dir, train_data, image_transform["train"])
trainloader = data_utils.DataLoader(TrainData,batch_size=64)
##############################
## Validation data

ValidData = dataSet_(valid_dir, validation_data, image_transform["test"])
validationloader = data_utils.DataLoader(ValidData, batch_size=64)

### test data
test_data = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/"  + "sample_submission.csv")
TestData = dataSet_(test_dir, test_data, image_transform["test"])
testloader = data_utils.DataLoader(TestData, batch_size=64)


# In[ ]:


images,_ = next(iter(testloader))
plt.imshow(images.data.numpy().squeeze()[0,:,:,:].reshape(224,224,3))


# In[ ]:


model = resnet50(pretrained=True)


# In[ ]:


from collections import OrderedDict
import torch.functional as F

for param in model.parameters():
    param.requires_grad = False

in_feature = model.fc.in_features

fc = nn.Sequential(OrderedDict([("fc1", nn.Linear(in_feature, 512)),
                                ("relu1", nn.ReLU()),
                                ("fc2", nn.Linear(512, 5)),
                                ("output",nn.LogSoftmax(dim=1))]))
model.fc = fc


# In[ ]:


device = ("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)


# In[ ]:


valid_loss_min = np.Inf
model.to(device)
for epoch in range(10):
    running_loss = 0.0
    model.train()
    for train_data, train_label in trainloader:
        train_data, train_label = train_data.to(device), train_label.to(device)
        log_ps = model.forward(train_data)
        optimizer.zero_grad()        
        train_loss = criterion(log_ps, train_label.long())
        running_loss += train_loss.item()
        train_loss.backward()
        optimizer.step()
    else:
        running_test_loss = 0.0
        with torch.no_grad():
            model.eval()
            for test_data, test_label in validationloader:
                test_data, test_label = test_data.to(device), test_label.to(device)
                test_log_ps = model.forward(test_data)
                test_loss = criterion(test_log_ps, test_label.long())
                running_test_loss += test_loss.item()
    if test_loss <= valid_loss_min:
        print("Validation loss decreased from {:.7f} -----> {:.7f}".format(valid_loss_min,test_loss))
        # save the model if validation error decreases 
        torch.save(model.state_dict(), 'model_augmented.pt')
        valid_loss_min = test_loss
                

    print(f" Epoch {epoch} , Training loss = {running_loss/len(trainloader)} Test loss = {running_test_loss/len(validationloader)}")


# # Load the model from the disk

# In[ ]:


checkpoint = torch.load('model_augmented.pt')
model.load_state_dict(checkpoint)


# # Confusion matrix
# Lets print the confusion Matrix

# In[ ]:


nb_classes = 5

confusion_matrix = torch.zeros(nb_classes, nb_classes)
model.to(device)
model.eval()
with torch.no_grad():
    for i, (inputs, classes) in enumerate(validationloader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model.forward(inputs)
        _, preds = torch.max(outputs.data.cpu(), 1)
        #set_trace()
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)


# # Classwise accuracy of the results

# In[ ]:


# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))


model.eval() # prep model for evaluation
with torch.no_grad():    
    for data, target in validationloader:
        # forward pass: compute predicted outputs by passing inputs to the model
        data, target = data.to(device), target.to(device)
        output = model(data)
        # calculate the loss
        loss = criterion(output, target.long())
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.long().data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            #set_trace()
            label = target[i].long().item()
            #set_trace()
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss/len(validationloader)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Validation Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Validation Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\n Validation Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


# # Visualize the results
# This cell displays test images and their labels in this format: predicted (ground-truth). The text will be green for accurately classified examples and red for incorrect predictions.

# In[ ]:


# obtain one batch of test images
dataiter = iter(validationloader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)
with torch.no_grad():  
    model.eval()
    # get sample outputs
    output = model.forward(images)
    # convert output probabilities to predicted class
    _, preds = torch.max(output, 1)
    # prep images for display
    images = images.data.cpu().numpy()

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        #set_trace()
        ax.imshow(images[idx].reshape(224,224,3))
        ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].long().item())),
                     color=("green" if preds[idx]==labels[idx].long() else "red"))


# # Test data

# In[ ]:


#filename_arrays = []
test_output =[]
with torch.no_grad():
    model.eval()
    for testdata,_ in testloader:
        testdata = testdata.to(device)
        #filename_arrays += list(filename)
        log_ps = model.forward(testdata)
        test_prob = torch.exp(log_ps)
        _, test_pred_class = torch.max(test_prob, dim=1)
        test_output += test_pred_class.cpu().data.tolist()


# In[ ]:


sample_output = {"id": test_data.id_code.values.tolist(), "diagnosis": test_output}
sample_output_dataframe = pd.DataFrame(sample_output)
sample_output_dataframe.to_csv("submission.csv",index=False)


# In[ ]:


import os
os.chdir(r'../working')
from IPython.display import FileLink
FileLink(r'submission.csv')

