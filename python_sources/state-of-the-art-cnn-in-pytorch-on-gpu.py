import torch
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5,5))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(5,5))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2_drop = nn.Dropout2d(p = 0.2)
        self.fc1 = nn.Linear(128, 100)
        self.fc2 = nn.Linear(100, 10)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,3))
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3,3))
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(1,1))
        self.bn6 = nn.BatchNorm2d(128)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.conv2_drop(F.max_pool2d(self.bn2(x),2))
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.max_pool2d(x,2)
        x = self.conv2_drop(x)
        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = F.relu(self.conv6(x))
        x = self.bn6(x)
        size =  x.size()[1]*x.size()[2]*x.size()[3]
        #print(size)
        x = x.view(-1, size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

pass

net = MnistNet()

print(net)

use_gpu = torch.cuda.is_available()
if use_gpu:
	net = net.cuda()
	print ('USE GPU')
else:
	print ('USE CPU')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.3, momentum = 0.1)

print ("1. Loading data")
train = pd.read_csv("../input/train.csv").values
test  = pd.read_csv("../input/test.csv").values

print ("2. Converting data")
X_data  = train[:, 1:].reshape(train.shape[0], 1, 28, 28)
X_data  = X_data.astype(float)
X_data /= 255.0
X_data  = torch.from_numpy(X_data);
X_label = train[:,0];
X_label = X_label.astype(int);
X_label = torch.from_numpy(X_label);
X_label = X_label.view(train.shape[0],-1);
print (X_data.size(), X_label.size())

print ("3. Training phase")
nb_train = train.shape[0]
nb_epoch = 20000
nb_index = 0
nb_batch = 250

for epoch in range(nb_epoch):
	if nb_index + nb_batch >= nb_train:
		nb_index = 0
	else:
		nb_index = nb_index + nb_batch

	mini_data  = Variable(X_data[nb_index:(nb_index+nb_batch)].clone())
	mini_label = Variable(X_label[nb_index:(nb_index+nb_batch)].clone(), requires_grad = False)
	mini_data  = mini_data.type(torch.FloatTensor)
	mini_label = mini_label.type(torch.LongTensor)
	if use_gpu:
		mini_data  = mini_data.cuda()
		mini_label = mini_label.cuda()
	optimizer.zero_grad()
	mini_out   = net(mini_data)
	mini_label = mini_label.view(nb_batch)
	mini_loss  = criterion(mini_out, mini_label)
	mini_loss.backward()
	optimizer.step() 

	if (epoch + 1) % 2000 == 0:
		print("Epoch = %d, Loss = %f" %(epoch+1, mini_loss.data[0]))

print ("4. Testing phase")

Y_data  = test.reshape(test.shape[0], 1, 28, 28)
Y_data  = Y_data.astype(float)
Y_data /= 255.0
Y_data  = torch.from_numpy(Y_data);
print (Y_data.size())
nb_test = test.shape[0]

net.eval()

final_prediction = np.ndarray(shape = (nb_test, 2), dtype=int)
for each_sample in range(nb_test):
	sample_data = Variable(Y_data[each_sample:each_sample+1].clone())
	sample_data = sample_data.type(torch.FloatTensor)
	if use_gpu:
		sample_data = sample_data.cuda()
	sample_out = net(sample_data)
	pred = torch.max(sample_out, 1)
	final_prediction[each_sample][0] = 1 + each_sample
	final_prediction[each_sample][1] = pred[1][0]
	if (each_sample + 1) % 2000 == 0:
		print("Total tested = %d" %(each_sample + 1))

print ('5. Generating submission file')

submission = pd.DataFrame(final_prediction, dtype=int, columns=['ImageId', 'Label'])
submission.to_csv('submission.csv', index=False, header=True)

