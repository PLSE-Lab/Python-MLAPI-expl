#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../working"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import scipy.io as sio 
import os
import cv2
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import pprint


# In[ ]:


from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class IMAGE_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.x = []
        self.y = []
        self.transform = transform
        self.num_classes = 0
        #print(self.root_dir.name)
        for i, _dir in enumerate(self.root_dir.glob('*')):
            for file in _dir.glob('*'):
                self.x.append(file)
                self.y.append(i)

            self.num_classes += 1
            #print(self.num_classes)
        #print(self.num_classes)
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, self.y[index]


# In[ ]:


annos = sio.loadmat('../input/devkit/devkit/cars_train_annos.mat')
pprint.pprint(annos)
pprint.pprint(annos["annotations"][:,0])
for i in range(6):
    pprint.pprint(annos["annotations"][:,0][0][i][0])
path = annos["annotations"][:,0][0][5][0].split(".")
pprint.pprint(int(path[0]) - 1)


# In[ ]:


def get_train_labels():
    annos = sio.loadmat('../input/devkit/devkit/cars_train_annos.mat')
    _, total_size = annos["annotations"].shape
    print("total sample size is ", total_size)
    labels = np.zeros((total_size, 5))
    for i in range(total_size):
        path = annos["annotations"][:,i][0][5][0].split(".")
        id = int(path[0]) - 1
        for j in range(5):
            labels[id, j] = int(annos["annotations"][:,i][0][j][0])
    return labels
train_labels = get_train_labels()
pprint.pprint(train_labels)


# In[ ]:


def get_test_labels():
    annos = sio.loadmat('../input/devkit/devkit/cars_test_annos_withlabels.mat')
    _, total_size = annos["annotations"].shape
    print("total sample size is ", total_size)
    labels = np.zeros((total_size, 5))
    for i in range(total_size):
        path = annos["annotations"][:,i][0][5][0].split(".")
        id = int(path[0]) - 1
        for j in range(5):
            labels[id, j] = int(annos["annotations"][:,i][0][j][0])
    return labels
test_labels = get_test_labels()
pprint.pprint(test_labels)


# In[ ]:


print(os.path.isdir('../working/cars_train_cut'))


# In[ ]:


def cut_train_images():    
    if not os.path.isdir('../working/cars_train_cut'):
        os.mkdir('../working/cars_train_cut')
        
    image_names = os.listdir('../input/cars_train/cars_train')
    for index in range(len(image_names)):
        img = cv2.imread("../input/cars_train/cars_train/" + image_names[index])[:,:,::-1]
        name = image_names[index].split('.')
        image_label = train_labels[int(name[0]) - 1]
        image_class = int(image_label[4])
        img = img[int(image_label[1]):int(image_label[3]),int(image_label[0]):int(image_label[2])]
        
        if not os.path.isdir('../working/cars_train_cut/{}'.format(image_class)):
            os.mkdir('../working/cars_train_cut/{}'.format(image_class))
        
        cv2.imwrite('../working/cars_train_cut/{}/{}'.format(image_class, image_names[index]), img)

cut_train_images()


# In[ ]:


count = 0
for image_class in os.listdir('../working/cars_train_cut/'):
    count = count + len(os.listdir('../working/cars_train_cut/{}'.format(image_class)))
print(count)


# In[ ]:


def cut_test_images():
    # create new folder if not exists
    if not os.path.isdir('../working/cars_test_cut'):
        os.mkdir('../working/cars_test_cut')
        
    image_names = os.listdir('../input/cars_test/cars_test')
    for index in range(len(image_names)):
        # load the image data
        img = cv2.imread("../input/cars_test/cars_test/" + image_names[index])[:,:,::-1]
        name = image_names[index].split('.')
        image_label = test_labels[int(name[0]) - 1]
        image_class = int(image_label[4])
        # cut the image
        img = img[int(image_label[1]):int(image_label[3]),int(image_label[0]):int(image_label[2])]

        # create new folder if not exists
        if not os.path.isdir('../working/cars_test_cut/{}'.format(image_class)):
            os.mkdir('../working/cars_test_cut/{}'.format(image_class))
        
        # save the cut image
        cv2.imwrite('../working/cars_test_cut/{}/{}'.format(image_class, image_names[index]), img)

cut_test_images()


# In[ ]:


count = 0
for image_class in os.listdir('../working/cars_test_cut/'):
    count = count + len(os.listdir('../working/cars_test_cut/{}'.format(image_class)))
print(count)


# In[ ]:


import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy

##REPRODUCIBILITY
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#args = parse_args()
#CUDA_DEVICES = args.cuda_devices
#DATASET_ROOT = args.path
CUDA_DEVICES = 0
DATASET_ROOT = '../working/cars_train_cut'

def train():
	data_transform = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	print(DATASET_ROOT)
	train_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)
	print(len(train_set))
	data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)
	print(train_set.num_classes)
# 	model = models(num_classes=train_set.num_classes)
	model = models.resnet101(pretrained=False)
	model = model.cuda(CUDA_DEVICES)
	model.train()

	best_model_params = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	num_epochs = 50
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(params=model.parameters(), lr=0.005, momentum=0.9)

	TenCounter = 0
	for epoch in range(num_epochs):
		print(f'Epoch: {epoch + 1}/{num_epochs}')
		print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

		training_loss = 0.0
		training_corrects = 0

		for i, (inputs, labels) in enumerate(data_loader):
			inputs = Variable(inputs.cuda(CUDA_DEVICES))
			labels = Variable(labels.cuda(CUDA_DEVICES))			

			optimizer.zero_grad()

			outputs = model(inputs)
			_, preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			training_loss += loss.item() * inputs.size(0)
			#revise loss.data[0]-->loss.item()
			training_corrects += torch.sum(preds == labels.data)
			#print(f'training_corrects: {training_corrects}')

		training_loss = training_loss / len(train_set)
		training_acc =training_corrects.double() /len(train_set)
		print(training_acc.type())
		print(f'training_corrects: {training_corrects}\tlen(train_set):{len(train_set)}\n')
		print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')
		LogMessage = f'Training loss: {training_loss:.4f}\t accuracy: {training_acc:.4f}\n'
		with open("log.txt", "a") as text_file:
			text_file.write(str(epoch) + "   " + LogMessage)
		print(LogMessage)
        
		if training_acc > best_acc:
			best_acc = training_acc
			best_model_params = copy.deepcopy(model.state_dict())

		if (epoch + 1) / 10 >= TenCounter:
			TenCounter += 1
			model.load_state_dict(best_model_params)
			torch.save(model, f'{TenCounter}-model-{best_acc:.02f}-best_train_acc.pth')
			print(f'save model: {TenCounter}-model-{best_acc:.02f}-best_train_acc.pth')
            
if __name__ == '__main__':
	train()


# In[ ]:


print(os.listdir('../working/'))


# In[ ]:


import torch
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
CUDA_DEVICES = 0
DATASET_ROOT = '../working/cars_test_cut/'
PATH_TO_WEIGHTS = '../working/5-model-1.00-best_train_acc.pth'

def test():
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    test_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)

    data_loader = DataLoader(
        dataset=test_set, batch_size=32, shuffle=True, num_workers=1)
    classes = [_dir.name for _dir in Path(DATASET_ROOT).glob('*')]
    print(classes)
    return

    model = torch.load(PATH_TO_WEIGHTS)
    model = model.cuda(CUDA_DEVICES)
    model.eval()

    total_correct = 0
    total = 0
    class_correct = list(0. for i in enumerate(classes))
    class_total = list(0. for i in enumerate(classes))
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # totoal
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            # batch size
            for i in range(labels.size(0)):
                label =labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy on the ALL test images: %d %%'
          % (100 * total_correct / total))

    for i, c in enumerate(classes):
        print('Accuracy of %5s : %2d %%' % (
        c, 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    test()


# In[ ]:


get_ipython().system('cat log.txt')

