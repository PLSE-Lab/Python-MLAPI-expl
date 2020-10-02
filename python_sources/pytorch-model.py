#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Lets import important Libraries

# In[ ]:


from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch
from torch.utils.data import Dataset,DataLoader
import  cv2
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import roc_auc_score
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
torch.manual_seed(42)


# # Read the data

# In[ ]:


Data_dir = "/kaggle/input/plant-pathology-2020-fgvc7/images/"
dataframe = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
submit = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")


# In[ ]:


dataframe['image_id']+='.jpg'
test_df['image_id']+='.jpg'


# In[ ]:


label = dataframe.loc[:, 'healthy':'scab']
label = label.values
x = np.argmax(label,axis=1)
dataframe['class'] = x


# # Split the dataframe into train and vaidation dataframe
# ### Also don't forget to reset index of train and validation data frame

# In[ ]:


train_df, valid_df = train_test_split(dataframe,test_size=0.2,shuffle=True,stratify=dataframe['class'])
train_label = train_df.loc[:, 'healthy':'scab']
valid_label = valid_df.loc[:,'healthy':'scab']
train_df.reset_index(drop=True,inplace=True)
train_label.reset_index(drop=True,inplace=True)
valid_df.reset_index(drop=True,inplace=True)
valid_label.reset_index(drop=True,inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


train_df.shape


# # Lets's built custom dataset for our work

# In[ ]:


class Plant_dataset(Dataset):
    
    def __init__(self,df,data_dir,transform,train=True,label=None):
        self.len = df.shape[0]
        self.dir = data_dir
        self.transform=transform
        self.df = df
        self.train=train
        self.label=label
        
    def __len__(self):
        return self.len
    
    def __getitem__(self,idx):
        img_pth = os.path.join(self.dir,self.df['image_id'].loc[idx])
        image = mpimg.imread(img_pth)
        image = cv2.resize(image,(224,224))
        if self.transform:
            image = self.transform(image)
        if(self.train==True):
            labels =np.argmax(self.label.loc[idx,:].values)
        else:
            return image
        return image,labels
    


# # Then apply the following transformation:
# ### 1. Transform to PIL Image
# ### 2. Transform to Tensor
# ### 3. Normalization 

# In[ ]:


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
train_transf = transforms.Compose([transforms.ToPILImage(),
                                   transforms.ToTensor(),normalize])

valid_transf = transforms.Compose([transforms.ToPILImage(),
                                  transforms.ToTensor(),normalize])
test_transf = transforms.Compose([transforms.ToPILImage(),
                                  transforms.ToTensor(),normalize])


# In[ ]:


train_dataset = Plant_dataset(train_df,Data_dir,transform=train_transf,train=True,label=train_label)
valid_dataset = Plant_dataset(valid_df,Data_dir,transform=valid_transf,train=True,label=valid_label)
test_dataset = Plant_dataset(test_df,Data_dir,transform=test_transf,train=False)


# #### Let's view some of our image dataset with labels

# In[ ]:


col = train_df.columns
fig = plt.figure(figsize=(25, 8))
for i, idx in enumerate(np.random.choice(train_df.index, 10)):
    ax = fig.add_subplot(2, 10//2, i+1, xticks=[], yticks=[])
    pth = os.path.join(Data_dir,train_df.image_id[idx])
    im = mpimg.imread(pth)
    im = cv2.resize(im, (100, 100)) 
    plt.imshow(im, cmap="hot")
    lab =col[np.argmax(train_df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']].values)+1]
    ax.set_title(lab)


# ### Make a dictionary for train_loader and validation_loader

# In[ ]:


batch_size=32
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size)
dataloaders_dict ={'train':train_loader,'val':val_loader}


# #### I am using densenet architecture.
# #### You can select any of these given model: [resnet, alexnet, vgg, squeezenet, densenet, inception], you just have to replace model_name with your desired model in the given list.

# In[ ]:


model_name = "densenet"
num_classes = 4
num_epochs = 2 
feature_extract = False
learning_rate = 0.0008


# ## Note: For fine tunning you just have to make the feature_extract False in above cell. And for feature extraction make the feature_extract True in above cell.
# #### Note: During fine tunning try to keep learning rate very small because the model is already trained.

# ### Lets define training function. For more information you can see pytorch documentation

# In[ ]:


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
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
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# ### If you are using fine tunning then there is no need to run the below cell as the default value of requires_grad of model.parameters() is true

# In[ ]:


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# ### Below cell contain all the model given in the list above. You can try different models and also ensemble all of them for further improvement in accuracy

# In[ ]:


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)


# In[ ]:


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


# Send the model to GPU
model_ft = model_ft.to(device)


# In[ ]:


##Print all the trainable layer 
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update, lr=0.001)


# In[ ]:


criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=2, is_inception=(model_name=="inception"))


# #### Lets predict on test data

# In[ ]:


def prediction(model,dataloader):
    model.eval()
    test_preds = None
    for i,batch in enumerate(dataloader):
        image = batch
        image = image.to(device)
        with torch.no_grad():
            out = model(image)
            if test_preds is None:
                test_preds = out.data.cpu()
            else:
                test_preds = torch.cat((test_preds, out.data.cpu()), dim=0)
    return test_preds


# In[ ]:


test_pred = prediction(model_ft,test_loader)


# In[ ]:


submit[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(test_pred, dim=1)


# In[ ]:


submit.to_csv("sample_submission.csv",index=False)


# # Please upvote this kernel if you find this Notebook useful :)

# In[ ]:




