#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import model_selection
import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as model
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score,accuracy_score
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensor 
import cv2


# In[ ]:


class MyDataSet(torch.utils.data.Dataset):

    def __init__(self, image_path, targets, transforms=None):
        self.image_path = image_path
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        image = self.image_path[item]
        targets = self.targets[item]
        #img = Image.open(image)
        #img = np.array(img)
        #print(img.shape)
        img = cv2.imread(image,cv2.cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            sample = {'image':img}
            sample = self.transforms(**sample)
            img = sample['image']
            #img = self.transforms(img)
            #print(img.shape)
        else:
            img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float), torch.tensor(targets, dtype=torch.float)


# In[ ]:


def train_fold(fold):
    training_path = "../input/melanoma-custom/Data/Data/Train_512"
    df = pd.read_csv('../input/melanoma-custom/train_Kfold.csv')

    df_train = df[df['k-fold'] != fold].reset_index(drop=True)
    df_valid = df[df['k-fold'] == fold].reset_index(drop=True)

    train_images = list(df_train.image_name)
    train_images = [os.path.join(training_path,i+'.jpg') for i in train_images]
    train_targets = df_train.target.values

    valid_images = list(df_valid.image_name)
    valid_images = [os.path.join(training_path,i+'.jpg') for i in valid_images]
    valid_targets = df_valid.target.values

    train_transform = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RGBShift(r_shift_limit=40),
        A.MultiplicativeNoise(p=1.0),
        A.RandomBrightness(0.1),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensor()
    ])

    valid_transform = A.Compose([
        A.RandomRotate90(),
        A.RandomBrightness(0.1),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensor()
    ])

    train_dataset = MyDataSet(train_images, train_targets, train_transform)
    valid_dataset = MyDataSet(valid_images, valid_targets, valid_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=256,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=False)

    return train_loader, valid_loader,valid_targets


# In[ ]:


class Resnet_152(nn.Module):
  def __init__(self,model):
    super(Resnet_152, self).__init__()
    self.model = model
    self.ext = nn.Sequential(
        nn.ReLU(),
        nn.Linear(100,1)
    )

  def forward(self,images):
    out = self.model(images)
    out = self.ext(out)
    return out


# In[ ]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

print(device)

model = model.resnet152(pretrained=True)

for param in model.parameters():
    param.requires_grad=False


in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 100)


mymodel = Resnet_152(model)

mymodel = mymodel.to(device)
criteria = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(mymodel.parameters(), lr=0.0001,weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='max',verbose=True)

#es = EarlyStopping(patience=5, mode="max")

for i in range(5):
    print('Fold={}'.format(i))
    tr_loader, val_loader,targets = train_fold(i)
    epochs = 10
    for j in range(epochs):
        loss_arr = []
        mymodel.train()
        for data in tr_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            del data
            optimizer.zero_grad()
            output = mymodel(images)
            #print(output)
            loss = criteria(output, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            loss_arr.append(loss.item())
            del images,labels
        print("epoch={},loss={}".format(j, sum(loss_arr)/len(loss_arr)))
        mymodel.eval()
        final_predictions = []
        for val_data in val_loader:
            val_images, val_labels = val_data
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            del  val_data
            with torch.no_grad():
                val_output = mymodel(val_images)
                #proba, pred = torch.max(val_output.data, 1)
                #print(val_output)
                val_output = torch.sigmoid(val_output)
                pred = val_output.cpu()
                #final_predictions.extend(pred)
                final_predictions.append(pred)
                del val_images, val_labels
        #predictions = np.array(final_predictions)
        predictions = np.vstack(final_predictions).ravel()
        k=roc_auc_score(targets, predictions)
        #l=accuracy_score(targets, predictions)
        print('val_auc_acore={}'.format(k))
        scheduler.step(k)
    torch.save(mymodel.state_dict(), 'resnet152_{}.pth'.format(i))   


# In[ ]:


##Testing
df = pd.read_csv("../input/melanoma-custom/test.csv")
test_path = "../input/melanoma-custom/Test_512/Test_512"
test_images = list(df['image_name'])
test_images = [os.path.join(test_path,i+'.jpg') for i in test_images]
test_transform = A.Compose([
    A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ToTensor()
])
targets = np.zeros(len(test_images))
test_dataset = MyDataSet(test_images, targets , test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

mymodel.eval()
final_predictions = []
for test_data in test_loader:
    test_images,test_labels = test_data
    test_images = test_images.to(device)
    with torch.no_grad():
        test_output = mymodel(test_images)
        #proba, pred = torch.max(val_output.data, 1)
        #print(val_output)
        test_output = torch.sigmoid(test_output)
        pred = test_output.cpu()
        #final_predictions.extend(pred)
        final_predictions.append(pred)
        del test_images,test_data
#predictions = np.array(final_predictions)
predictions = np.vstack(final_predictions).ravel()
sample = pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")
sample.loc[:, "target"] = predictions
sample.to_csv("submission.csv", index=False)


# In[ ]:




