# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.tensor as Variable

train_df=pd.read_csv('../input/train.csv').values
test_df=pd.read_csv('../input/test.csv').values

class MNIST(Dataset):

    def __init__(self, frame, train = True, transform=None):
        
        self.frame = frame
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if self.train:
            image = self.frame[idx, 1: ].reshape((28, 28, 1)).astype(np.float32)
            label = self.frame[idx, 0]
            sample = {'image': image, 'label': label}
            if self.transform:
                sample['image'] = self.transform(sample['image'])
        else:
            image = self.frame[idx].reshape((28, 28, 1)).astype(np.float32)
            sample = image
            if self.transform:
                sample = self.transform(sample)
        
        return sample


train_data=MNIST(frame=train_df,train=True,
                      transform=transforms.Compose([transforms.ToTensor()]))

dataloader=DataLoader(train_data,batch_size=8,shuffle=True,num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=1 , out_channels=16 , kernel_size=3 , stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2 , stride=2),  
       )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(in_channels=16 , out_channels=48 , kernel_size=3 , stride=1 , padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(in_channels=48 , out_channels=64 , kernel_size=3 , stride=1 , padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2 , stride=2), 
        )
        self.conv4 = nn.Sequential(         
            nn.Conv2d(in_channels=64 , out_channels=32 , kernel_size=3 , stride=1 , padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2 , stride=2), 
        )
        self.linear1 = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(32*3*3 , 120),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(120 , 10),
        )

    def forward(self , data):
               
        out = self.conv1(data)
        #print(out.shape)
        out = self.conv2(out)
        #print(out.shape)
        out = self.conv3(out)
        #print(out.shape)
        out = self.conv4(out) 
        #print(out.shape)
        out = out.view(-1, 32 * 3 * 3)
        out = self.linear1(out)
        #print(out.shape)
        out=F.log_softmax(self.linear2(out))
         
        return out


model=Net()
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(3):
    
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data['image'], data['label']
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

mnist_test = MNIST(frame=test_df, train=False, transform=transforms.Compose([transforms.ToTensor()]))

dataloader_test = DataLoader(mnist_test, batch_size=8,
                        shuffle=False, num_workers=0)

res = []
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        inputs = data
        outputs = torch.exp(model(inputs)).argmax(dim=1).tolist()
        res += outputs
    print('Finished Training')  
res = np.array(res)

res = pd.DataFrame(res, columns=['Label'])
res['ImageId'] = np.arange(1, len(res) + 1)

res = res[['ImageId', 'Label']]

example_sub = pd.read_csv('../input/sample_submission.csv')
res.to_csv('res.csv', index=False)