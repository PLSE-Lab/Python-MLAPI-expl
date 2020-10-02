#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
torch.cuda.is_available()


# In[2]:


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


# In[19]:


import torch.nn as nn
class My_CNN(nn.Module):
    def __init__(self, num_classes):
        super(My_CNN, self).__init__()
        self.conv = nn.Sequential(  # input shape (3, 224, 224)
            nn.Conv2d(in_channels=3,out_channels=25,kernel_size=3,padding=1),      # output shape (16, 224, 224)
            nn.ReLU(),
            nn.Conv2d(in_channels=25,out_channels=75,kernel_size=3,padding=1),      # output shape (16, 224, 224)
            nn.ReLU(),
            nn.Conv2d(in_channels=75,out_channels=75,kernel_size=3,padding=1),      # output shape (32, 224, 224)
            nn.ReLU(),
            nn.MaxPool2d(2),  # output shape (32, 112, 112)
            
            nn.Conv2d(in_channels=75,out_channels=125,kernel_size=3,padding=1),      # output shape (64, 112, 112)
            nn.ReLU(),
            nn.Conv2d(in_channels=125,out_channels=125,kernel_size=3,padding=1),      # output shape (64, 112, 112)
            nn.ReLU(),
            nn.Conv2d(in_channels=125,out_channels=125,kernel_size=3,padding=1),      # output shape (128, 112, 112)
            nn.ReLU(),
            nn.MaxPool2d(2),  # output shape (128, 56, 56)
            
            nn.Conv2d(in_channels=125, out_channels=175,kernel_size=3,padding=1),      # output shape (256, 56, 56)
            nn.ReLU(),
            nn.Conv2d(in_channels=175,out_channels=175,kernel_size=3,padding=1),      # output shape (512, 56, 56)
            nn.ReLU(),
            nn.Conv2d(in_channels=175,out_channels=225,kernel_size=3,padding=1),      # output shape (512, 56, 56)
            nn.ReLU(),
            nn.MaxPool2d(2),  # output shape (512, 28, 28)
            
            nn.Conv2d(in_channels=225,out_channels=275,kernel_size=3,padding=1),      # output shape (512, 28, 28)
            nn.ReLU(),
            nn.Conv2d(in_channels=275,out_channels=275,kernel_size=3,padding=1),      # output shape (512, 28, 28)
            nn.ReLU(),
            nn.Conv2d(in_channels=275,out_channels=325,kernel_size=3,padding=1),      # output shape (512, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2), # output shape (512, 14, 14)
            
            nn.Conv2d(in_channels=325,out_channels=375,kernel_size=3,padding=1),      # output shape (512, 28, 28)
            nn.ReLU(),
            nn.Conv2d(in_channels=375,out_channels=375,kernel_size=3,padding=1),      # output shape (512, 14, 14)
            nn.ReLU(),
            nn.Conv2d(in_channels=375,out_channels=425,kernel_size=3,padding=1),      # output shape (2048, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2) # output shape (2048, 7, 7)
            
            
#             nn.Conv2d(in_channels=3, out_channels=100, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=100, out_channels=150, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=150, out_channels=150, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=150, out_channels=300, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=300, out_channels=300, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=300, out_channels=300, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=300, out_channels=550, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=550, out_channels=550, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=550, out_channels=550, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=550, out_channels=550, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=550, out_channels=550, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=550, out_channels=550, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
        )        
        
        self.out = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 425, out_features=4250),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4250, out_features=4250),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4250, out_features=num_classes)
        )
        #nn.Linear(512 * 7 * 7, num_classes)   # fully connected layer, output 10 classes
        
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()
                
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)   #(512 * 7 * 7)
        output = self.out(x)
        return output


# In[16]:


import math
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            # input shape: (batch_size, 3, 224, 224) and
            # downsampled by a factor of 2^5 = 32 (5 times maxpooling)
            # So features' shape is (batch_size, 7, 7, 512)
            nn.Linear(in_features=7 * 7 * 512, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        # initialize parameters
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# In[6]:


get_ipython().system('nvidia-smi')


# In[7]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[20]:


import torch
import torch.nn as nn
# from models import VGG16
# from dataset import IMAGE_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy

##REPRODUCIBILITY
torch.manual_seed(233)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#CUDA_DEVICES = args.cuda_devices
#DATASET_ROOT = args.path
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CUDA_DEVICES = 0
DATASET_ROOT = '../input/cars_train/cars_train/'

def train():
	data_transform = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	#print(DATASET_ROOT)
	train_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)
	data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)
	#print(train_set.num_classes)
# 	model = models.resnet101(pretrained=True)
	model = My_CNN(num_classes=train_set.num_classes)
# 	model = VGG16(num_classes=train_set.num_classes)
	model = model.cuda(CUDA_DEVICES)
	model.train()

	best_model_params = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	num_epochs = 50
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
# 	optimizer = torch.optim.ASGD(params=model.parameters(), lr=0.01)

	tenTime = 0
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
			#torch.cuda.empty_cache()

		training_loss = training_loss / len(train_set)
		training_acc =training_corrects.double() /len(train_set)
		#print(training_acc.type())
		#print(f'training_corrects: {training_corrects}\tlen(train_set):{len(train_set)}\n')
		outMessage = f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n'
		with open("Output.txt", "a") as text_file:
			text_file.write(str(epoch) + "    " + outMessage)
		print(outMessage)

      
		if training_acc > best_acc:
			best_acc = training_acc
			best_model_params = copy.deepcopy(model.state_dict())

		if (epoch+1)/10>=tenTime:

			tenTime += 1
			model.load_state_dict(best_model_params)
			torch.save(model, f'{tenTime}-model-{best_acc:.02f}-best_train_acc.pth')
			print(f'save model: {tenTime}-model-{best_acc:.02f}-best_train_acc.pth')


if __name__ == '__main__':
	train()


# In[21]:


get_ipython().system('ls')


# In[22]:


get_ipython().system('cat Output.txt')


# In[24]:


import torch
# from utils import parse_args
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
from PIL import Image
# from torch.utils.data import DataLoader
# from dataset import IMAGE_Dataset
CUDA_DEVICES = 0
DATASET_ROOT = '../input/cars_test/cars_test'
PATH_TO_WEIGHTS = './6-model-0.99-best_train_acc.pth'


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




