# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
## design a pytorch convolutional neural net
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
print(train.columns)
y = train['label'];
train.drop('label', axis = 1, inplace = True)

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse = False)
yhot = onehot_encoder.fit_transform(y.values.reshape(-1,1))
print(yhot.shape)

## ==================================================================##
print(train.shape)
#image shape is 28 by 28

nptrain = train.values/255.0;
nptrain = np.reshape(nptrain, (42000, 1,28,28))
print(nptrain.shape)
## ==================================================================##

from torch.autograd import Variable
import torch.nn.functional as F

class MnistCNN(torch.nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        #Input channels = 3, output channels = 18
        # we only specify layers, but not for example activations
        
        #we have to be responsible for all sizing operations
        #first two arguments are in_channels (input filter size), out_channels (size of filter)
        # kernel size is number of filter stacks
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=1)
        ## output is 26x26x3
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #self.d1 = torch.nn.Dropout(p = 0.1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size = 2, stride = 1, padding = 1);
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv_bn2 = torch.nn.BatchNorm2d(32)

        self.conv3 = torch.nn.Conv2d(32,8, kernel_size = 1, stride = 1, padding = 1)
        torch.nn.init.xavier_uniform_(self.conv3.weight)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.d3 = torch.nn.Dropout(p = 0.2)
        ## reduces size  to 13x13x 3
        # in_features, out_features
        self.fc1 = torch.nn.Linear(np.prod(8*8*8), 64)
        self.bn1 = torch.nn.BatchNorm1d(64) #parameter here is...

        #64 input features, 10 output features for our 10 defined classes, 0-9
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        #batch has shape (num_samples, 1, 28, 28)
        x = self.pool1(self.conv1(x));
        x = self.conv_bn2(self.conv2(x));
        x = F.relu(self.conv3(x));
        x = self.pool(x)
        #print(x.shape)
        #Recall that the -1 infers this dimension from the other given dimension
        #print(x.shape)
        x = x.view(-1, np.prod(8*8*8))
        #print(x.shape)
        #Computes the activation of the first fully connected layer
        x = self.d3(self.bn1(F.relu(self.fc1(x))))
        #Computes the second fully connected layer (activation applied later)
        x = self.fc2(x)
        
        #get softmax
        
        return(x)
    
    def evaluate(self, image_batch, batch_labels):
        image_tensor = image_batch.float()
        input = Variable(image_tensor)
        output = self.forward(input)
        index = output.data.cpu().numpy().argmax(axis = 1)
        batch_labels = batch_labels.data.cpu().numpy();
        return 1-np.count_nonzero(index-batch_labels)/len(batch_labels);        
        
        
## ==================================================================##


epochs = 500;
num_examples = 42000;
minibatch_size = 500;
num_mini_batches = num_examples//minibatch_size;
train_data = torch.from_numpy(nptrain[0:num_examples,:,:])
train_data = train_data.type('torch.DoubleTensor')
criterion = torch.nn.CrossEntropyLoss()
ytrain = torch.from_numpy(y.values);
ytrain = ytrain.type('torch.DoubleTensor')
print(ytrain.shape)
#convert to one hot

## =================== TRAIN VALIDATION SPLIT ======================== ##
from sklearn.model_selection import train_test_split
Xtrain, Xval, Ytrain, Yval = train_test_split(train_data.numpy(), ytrain.numpy(), test_size=0.2, random_state=42);
print(Xtrain.shape)
Xtrain = torch.from_numpy(Xtrain).type('torch.DoubleTensor').cuda();
Xval = torch.from_numpy(Xval).type('torch.DoubleTensor').cuda();
Ytrain = torch.from_numpy(Ytrain).type('torch.DoubleTensor').cuda();
Yval = torch.from_numpy(Yval).type('torch.DoubleTensor').cuda();
print(Xtrain.shape)
print(Yval.shape)
num_mini_batches = Xtrain.shape[0]//minibatch_size;



#optimizer = torch.optim.SGD(cnn.parameters(), lr = 0.1, momentum=0.4)
cnn = MnistCNN();
cnn.cuda()
cnn.zero_grad()
optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.001,betas=(0.9, 0.999));
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97, last_epoch=-1)
for i in range(epochs):
    for j in range(num_mini_batches):
        optimizer.zero_grad()
        ybatch = Ytrain[j*minibatch_size:(j+1)*minibatch_size]
        yhat = cnn.forward(Xtrain[j*minibatch_size:(j+1)*minibatch_size,:,:,:].cuda().float());
        loss = criterion(yhat, ybatch.cuda().long())
        loss.backward();
        optimizer.step();
    dev_correct = cnn.evaluate(Xval, Yval)
        
    if(i%10 == 0):
        print('val correct: '+str(dev_correct))
        print('loss: '+str(loss))
        scheduler.step();
    
## ==================================================================##
def predict_batch(image_batch, model):
    image_tensor = image_batch.float()
    input = Variable(image_tensor)
    output = model(input)
    index = output.data.cpu().numpy().argmax(axis = 1)
    return index
    
## generate predictions for everything in test
test = pd.read_csv("../input/test.csv")
print(test.columns)
nptest = test.values/255.0;
print(nptest.shape)
nptest = np.reshape(nptest, (28000, 1,28,28))

#run every test image through the model
print(nptest.shape)
test_data = torch.from_numpy(nptest)
test_data = test_data.type('torch.DoubleTensor')
print(test_data.shape)

batch_size = 1000;
num_test_batches = 28000//batch_size;
predicted_labels = [];
for i in range(num_test_batches):
    #print((i*batch_size,(i+1)*batch_size))
    test_batch = test_data[i*batch_size:(i+1)*batch_size, :,:,:].cuda().float();
    #print(test_batch.shape)
    preds = predict_batch(test_batch, cnn)
    predicted_labels +=  list(preds);
    #print(len(predicted_labels))
print(len(predicted_labels))

predictions = pd.DataFrame(predicted_labels, columns = ['Label']);
#predictions['Label'] = predicted_labels;
#print(predictions)
predictions.index+=1;
predictions.to_csv('predictions.csv', index_label = 'ImageId')