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
from numpy import array, asarray, ndarray, swapaxes, append, empty, NAN, nanmean, isnan

#training controls
batch_size = 1
epochs = 1
train_size = 0.6
learning_rate = 0.001
dropout = [0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15, 0.1, 0.1]
clear_grad_train = True

k = 2
mem_trust = 0.5
mem_const = 400*1e-05
rem_trust = 0.3
rem_mem = 10
knn = 600

correct_score = 1
inform_rate = 50
# input image dimensions
img_rows, img_cols = 268, 182

# the data holders
x = []
y = []
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

#extracting the data about the images that are available
for filename in flist:
    name=int(filename.split('/')[-1][:-4])
    for z in tempY:
        if(z[0]==name):
            img = array(cv2.imread(filename))
            img = swapaxes(img, 2,0)
            img = swapaxes(img, 2,1)
            x.append(img)
            y.append(z[1])
    
#converting the data from lists to numpy arrays
x=asarray(x,dtype=float)
y=asarray(y,dtype=float)

train_size = int(len(x)*train_size)

x_train = x[:train_size]
y_train= y[:train_size]

x_test = x[train_size:]
y_test= y[train_size:]
#scaling down the RGB data
x_train /= 255
x_test /= 255

#printing stats about the features
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

train_length = x_train.shape[0]

x_train=torch.from_numpy(x_train).float()
x_test=torch.from_numpy(x_test).float()
y_train=torch.from_numpy(y_train).float()
y_test=torch.from_numpy(y_test).float()

train = data_utils.TensorDataset(x_train, y_train)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

test = data_utils.TensorDataset(x_test, y_test)
test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self, input_shape=(3, img_rows, img_cols)):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=2)
        self.conv1_drop = nn.Dropout2d(p=dropout[0])
        self.conv2 = nn.Conv2d(128, 64, kernel_size=2)
        self.conv2_drop = nn.Dropout2d(p=dropout[1])
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.conv3_drop = nn.Dropout2d(p=dropout[2])
        self.conv4 = nn.Conv2d(64, 64, kernel_size=2)
        self.conv4_drop = nn.Dropout2d(p=dropout[3])
        self.conv5 = nn.Conv2d(64, 32, kernel_size=2)
        self.conv5_drop = nn.Dropout2d(p=dropout[4])
        self.conv6 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv6_drop = nn.Dropout2d(p=dropout[5])
        
        n_size = self._get_conv_output(input_shape)
        
        self.fc1 = nn.Linear(n_size, 16)
        self.fc1_drop = nn.Dropout(p=dropout[6])
        self.fc2 = nn.Linear(16, 16)
        self.fc2_drop = nn.Dropout(p=dropout[7])
        self.fc3 = nn.Linear(16, 8)
        self.fc3_drop = nn.Dropout(p=dropout[8])
        self.fc4 = nn.Linear(8, 1)
        
        self.reveries = []
        self.use_memory = False
        
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
        
        self.short_term_mem = x.clone().detach()
        memory = None
        if(self.use_memory==True):
            memory = self.remember(x)
        if(memory!=None):
            mem = asarray(memory)
            mem = torch.from_numpy(mem).float()
            return (Variable(mem),1)
        
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1_drop(self.fc1(x)))
        x = F.relu(self.fc2_drop(self.fc2(x)))
        x = F.relu(self.fc3_drop(self.fc3(x)))
        x = self.fc4(x)
        return (x,0)
        
    def memorize(self,y):
        self.reveries.append((self.short_term_mem,y))
        
    def remember(self,x):
        sims = empty(k)
        sims[:] = NAN
        targets = empty(k)
        targets[:] = NAN
        similarity = empty(1)
        for revery, target in self.reveries:
            similarity = 1/torch.abs(cc(x, revery))/mem_const
            #print(similarity.data[0])
            if similarity.data[0] >= mem_trust:
                for i in range(0,k):
                    if sims[i]<similarity:
                        sims[i] = similarity.data[0] 
                        targets[i] = target.data[0]
                        break
        
        y_tr = asarray([nanmean(targets)])
        if isnan(y_tr):
            return None
        else:
            #print("-----------------",sims[0])
            return y_tr

model = Net().double()
criterion = MSELoss(size_average=True)
cc = MSELoss()
human_criterion = L1Loss(size_average=True)
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate,
            alpha=0.9, eps=1e-08, weight_decay=0.0)

def train(epoch):
    model.train()
    temp = []
    loss_mem=array(temp)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target).float()
        if clear_grad_train:
                    optimizer.zero_grad()
        output = model(data)
        loss = criterion(output[0], target)
        if(loss.data[0]<=loss_mem[len(loss_mem)-rem_mem:len(loss_mem)].mean()*(1+rem_trust)):
            loss_mem = append(loss_mem,loss.data[0])
        else:
            model.memorize(target)
            loss_mem = append(loss_mem,loss.data[0])
        if(output[1]==0):
            loss.backward()
            optimizer.step()
        if batch_idx%inform_rate==0:
            human_loss= human_criterion(output[0], target)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0], human_loss.data[0]))
        if (batch_idx+1)%knn==0:
            model.use_memory = True
        

def test():
    model.eval()
    test_loss = 0
    correct = 0
    human_loss = 0
    i = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        i+=1
        data, target = Variable(data, volatile=True).float(), Variable(target).float()
        output = model(data)
        loss = criterion(output[0], target)
        test_loss += loss.data[0]
        human_loss += human_criterion(output[0], target).data[0]
        if loss.data[0]<=correct_score:
            correct+=1

    print('\nTest set: \nAverage sq_loss: {:.4f} \nAverage abs_loss: {:.4f} \nGuessed correctly: {:.4f}\n'.format(test_loss/i, human_loss/i, correct))
    
    
model.float()
print(model)
model.use_memory = False
for epoch in range(0, epochs):
    train(epoch)
test()