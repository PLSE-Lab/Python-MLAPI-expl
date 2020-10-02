import random
import csv
import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# function to load the data into tensors
def loadData(trainData, testData):

	features = []
	for col in trainData.columns:
		if col != "label":
			features.append(col)

	trainDataFeats = np.float32(trainData[features].values)
	trainDataLabels = np.int32(trainData["label"].values)

	testDataFeats = np.float32(testData[features].values)

	return trainDataFeats, trainDataLabels, testDataFeats

def generateBatch(trainDataFeats, trainDataLabels, batchSize):

	ImgBatch = []
	LabelBatch = []

	for i in range(0,batchSize):

		randInt = random.randint(0, np.shape(trainDataLabels)[0]-1)
		img = np.reshape(trainDataFeats[randInt,:],(28,28))

        #normalizing the image 
        
		img[:,:] = 2*(img[:,:]-img[:,:].min())/(img[:,:].max()-img[:,:].min()) - 1
		img = np.expand_dims(img,axis=0)

		label = [trainDataLabels[randInt]]
		
		ImgBatch.append(torch.Tensor(img))
		LabelBatch.append(torch.Tensor(label))
	
	ImgBatchTensor = torch.stack(ImgBatch)
	LabelBatchTensor = torch.cat(LabelBatch)
	LabelBatchTensor = LabelBatchTensor.long()	


	return ImgBatchTensor, LabelBatchTensor

class NNet(nn.Module):

    def __init__(self):
        super(NNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1BN = nn.BatchNorm2d(32)
        self.conv1Drop = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2BN = nn.BatchNorm2d(64)
        self.conv2Drop = nn.Dropout2d(0.25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3BN = nn.BatchNorm2d(128)
        self.conv3Drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.fc1BN = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc2BN = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.conv1BN(nn.functional.max_pool2d(x, 2, 2))
        x = nn.functional.relu(self.conv1Drop(self.conv2(x)))
        x = self.conv2BN(nn.functional.max_pool2d(x, 2, 2))
        x = nn.functional.relu(self.conv2Drop(self.conv3(x)))
        x = self.conv3BN(nn.functional.max_pool2d(x, 2, 2))
        x = self.conv3Drop(x)
        x = x.view(-1, 128)
        x = self.fc1BN(nn.functional.relu(self.fc1(x)))
        x = self.fc2BN(nn.functional.relu(self.fc2(x)))
        x = self.fc3(x)

        return nn.functional.log_softmax(x, dim=1)

def train(model, trainDataFeats, trainDataLabels, optimizer, loss_func, batchSize, epoch):
	model.train()
	numIterationsPerEpoch = int(np.shape(trainDataLabels)[0]/float(batchSize))
	for i in (range(numIterationsPerEpoch)):

		imgBatch, labelBatch = generateBatch(trainDataFeats, trainDataLabels, batchSize)
		imgBatch, labelBatch = imgBatch.cuda(), labelBatch.cuda()

		optimizer.zero_grad()
		output = model(imgBatch)
		loss = loss_func(output, labelBatch)
		loss.backward()
		optimizer.step()
		if i % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * np.shape(imgBatch)[0], np.shape(trainDataLabels)[0],
                100. *  i * np.shape(imgBatch)[0]/ np.shape(trainDataLabels)[0], loss.item()))

def test(model, testFeats):

	model.eval()

	with torch.no_grad():

		with open('../working/submission.csv', 'w') as csvFile:
			csvWriter = csv.writer(csvFile, delimiter=',')
			csvWriter.writerow(['ImageId', 'Label'])

			for i in tqdm(range(np.shape(testFeats)[0])):
				
				img = np.reshape(testFeats[i,:],(28,28))
				#normalizing the image 
				img[:,:] = 2*(img[:,:]-img[:,:].min())/(img[:,:].max()-img[:,:].min()) - 1
				img = np.expand_dims(img,axis=0)
				img = np.expand_dims(img,axis=0)
				
				img = torch.Tensor(img)
				img = img.cuda()
				output = model(img)

				pred = output.argmax(dim=1)

				csvWriter.writerow([(i+1),pred.item()])

def main():
    # Read the data
    print("...........Reading the data....................")
    trainData = pd.read_csv('../input/digit-recognizer/train.csv')	
    testData = pd.read_csv('../input/digit-recognizer/test.csv')
    print("...........Done reading the data...............")

    # Loading data into numpy arrays
    trainDataFeats, trainDataLabels, testDataFeats = loadData(trainData, testData)

    # Fixing the seed of random number generator
    torch.manual_seed(0)
    random.seed(0)

    # Declaring the model, optimizer and loss function
    model = NNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    loss_func = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1,50)):

    	train(model, trainDataFeats, trainDataLabels, optimizer, loss_func, batchSize=60, epoch=epoch)
    
    test(model, testDataFeats)

    torch.save(model.state_dict(), "../working/mnist_myNet.pt")

if __name__ == '__main__':
	main()