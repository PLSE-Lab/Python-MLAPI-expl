import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.nn.modules import MSELoss, L1Loss

import glob
import csv
import cv2
from numpy import array, asarray, ndarray, swapaxes

#training controls
batch_size = 20
epochs = 2
training_size = 0.7
learning_rate = 0.001
dropout = [0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.15]
# input image dimensions
img_rows, img_cols = 268, 182

# the data holders
x_test = []
x_train = []
y_test= []
y_train= []
tempY = []

#opening the dataset
dataset = csv.reader(open("../input/MovieGenre.csv",encoding="utf8",errors='replace'), delimiter=",")

#skipping the header line
next(dataset)

#the list of image files in SampleMoviePosters folder
flist=glob.glob('../input/SampleMoviePosters/*.jpg')  

#extracting the data from the CSV file
for imdbId, Link, Title, Score, Genre, Poster in dataset:
    if(Score!=""):
        if(len((int(imdbId),float(Score)))==2):
            tempY.append((int(imdbId),float(Score)))


#setting the length of training data
length=int(len(flist)*training_size)

#extracting the data about the images that are available
i=0
for filename in flist:
    name=int(filename.split('/')[-1][:-4])
    for z in tempY:
        if(z[0]==name):
            
            img = array(cv2.imread(filename))
            img = swapaxes(img, 2,0)
            img = swapaxes(img, 2,1)

            if(i<length):
                x_train.append(img)
                y_train.append(z[1])
            else:
                x_test.append(img)
                y_test.append(z[1])
    i+=1
    
#converting the data from lists to numpy arrays
x_train=asarray(x_train,dtype=float)
x_test=asarray(x_test,dtype=float)
y_train=asarray(y_train,dtype=float)
y_test=asarray(y_test,dtype=float)

#scaling down the RGB data
x_train /= 255
x_test /= 255

#printing stats about the features
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

train_length = x_train.shape[0]

x_train=torch.from_numpy(x_train)
x_test=torch.from_numpy(x_test)
y_train=torch.from_numpy(y_train)
y_test=torch.from_numpy(y_test)

train = data_utils.TensorDataset(x_train, y_train)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

test = data_utils.TensorDataset(x_test, y_test)
test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self, input_shape=(3, img_rows, img_cols)):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3)
        self.conv1_drop = nn.Dropout2d(p=dropout[0])
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=dropout[1])
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3_drop = nn.Dropout2d(p=dropout[2])
        self.conv4 = nn.Conv2d(64, 64, kernel_size=2)
        self.conv4_drop = nn.Dropout2d(p=dropout[3])
        self.conv5 = nn.Conv2d(64, 32, kernel_size=2)
        self.conv5_drop = nn.Dropout2d(p=dropout[4])
        self.conv6 = nn.Conv2d(32, 16, kernel_size=2)
        self.conv6_drop = nn.Dropout2d(p=dropout[5])
        
        n_size = self._get_conv_output(input_shape)
        
        self.fc1 = nn.Linear(n_size, 16)
        self.fc1_drop = nn.Dropout(p=dropout[6])
        self.fc2 = nn.Linear(16, 16)
        self.fc2_drop = nn.Dropout(p=dropout[7])
        self.fc3 = nn.Linear(16, 8)
        self.fc3_drop = nn.Dropout(p=dropout[8])
        self.fc4 = nn.Linear(8, 1)
        
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size
        
    def _forward_features(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(x)), 2))
        x = F.relu(F.max_pool2d(self.conv5_drop(self.conv5(x)), 2))
        x = F.relu(F.max_pool2d(self.conv6_drop(self.conv6(x)), 2))
        return x
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(x)), 2))
        x = F.relu(F.max_pool2d(self.conv5_drop(self.conv5(x)), 2))
        x = F.relu(F.max_pool2d(self.conv6_drop(self.conv6(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_drop(self.fc1(x)))
        x = F.relu(self.fc2_drop(self.fc2(x)))
        x = F.relu(self.fc3_drop(self.fc3(x)))
        x = self.fc4(x)
        return x

model = Net()
criterion = MSELoss(size_average=True)
human_criterion = L1Loss(size_average=True)
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate,
            alpha=0.9, eps=1e-08, weight_decay=0.0)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).float(), Variable(target).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        human_loss= human_criterion(output, target)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0], human_loss.data[0]))

def test():
    print('test')
    model.eval()
    test_loss = 0
    correct = 0
    human_loss = 0
    i = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        i+=1
        data, target = Variable(data, volatile=True).float(), Variable(target).float()
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss
        human_loss += human_criterion(output, target)
        if loss==0:
            correct+=1

    print('\nTest set: \nAverage sq_loss: {:.4f} \nAverage abs_loss: {:.4f} \nGuessed 100% correct: {:.4f}\n'.format(test_loss.data[0]/i, human_loss.data[0]/i, correct))

model.float()
print(model)
for epoch in range(0, epochs):
    train(epoch)
    test()
