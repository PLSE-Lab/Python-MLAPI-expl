# %% [code] {"id":"k6TZJR2F5ufq"}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %% [code] {"id":"vZZPtkUX5ugG"}
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import torch
from torchvision import datasets, transforms


# %% [code] {"id":"VK74mBYD5ugw"}
def imshow(image, ax=None, title=None, normalize=True):
  """Imshow for Tensor."""
  if ax is None:
      fig, ax = plt.subplots()
  image = image.numpy().transpose((1, 2, 0))

  if normalize:
      mean = np.array([0.485, 0.456, 0.406])
      std = np.array([0.229, 0.224, 0.225])
      image = std * image + mean
      image = np.clip(image, 0, 1)

  ax.imshow(image)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.tick_params(axis='both', length=0)
  ax.set_xticklabels('')
  ax.set_yticklabels('')

  return ax

# %% [code] {"id":"W6q-nTsJ5uhH"}
data_dir=r'/kaggle/input/fruits/fruits-360/Training'

transform = transforms.Compose([transforms.Resize(100),
                               transforms.CenterCrop(100),
                               transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform = transform)

# %% [code] {"id":"dLmXKvPF5uhb"}
testset = datasets.ImageFolder(r'/kaggle/input/fruits/fruits-360/Test', transform=transform)
test_Loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

# %% [code] {"id":"HMTo8PqO5uhw","outputId":"0a374561-c3fa-48c2-cd38-8656ba885b43"}
images, labels = next(iter(test_Loader))

print(labels)
imshow(images[0], normalize=False)

# %% [code] {"id":"Bd5fV40b5uiH"}
from torch.utils.data.sampler import SubsetRandomSampler
valid_size = 0.2
num_workers = 0
num_train = len(dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size*num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_Loader = torch.utils.data.DataLoader(dataset, batch_size=32,
               sampler=train_sampler, num_workers=num_workers)
valid_Loader = torch.utils.data.DataLoader(dataset, batch_size=32, 
               sampler=valid_sampler, num_workers=num_workers)

# %% [code] {"id":"b7Tnr5v95uiW"}
classes = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Dates', 'Eggplant', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Williams', 'Pepino', 'Pepper Green', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Maroon', 'Tomato Yellow', 'Walnut']


# %% [code] {"id":"vPeo_6685uik","outputId":"afdcb6e9-5ed5-41e9-f7d3-784b025ee431"}
dataiter = iter(train_Loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)

# %% [code] {"id":"TnA3xGgG5ui1"}
def imageshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  

# %% [code] {"id":"ZB24PGVC5ujG","outputId":"7b1ed614-51bd-4e90-de82-29d798e78312"}
import matplotlib.pyplot as plt
%matplotlib inline
dataiter = iter(test_Loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imageshow(images[idx])
    ax.set_title(classes[labels[idx]])

# %% [code] {"id":"0JUD7RWqUTPk","outputId":"a4023d22-08f3-4979-cbf3-8d76a70f0208"}
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

# %% [code] {"id":"Z6P4FE9T5ujX","outputId":"27e313d2-8e19-4af1-c46f-95acc54abc56"}
import torch.nn as nn
import torch.nn.functional as F

#Defineing CNN Architecture

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3,16, 3, padding=1)
        
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        
        self.conv4 = nn.Conv2d(64,128,3, padding=1)
        
        self.pool = nn.MaxPool2d(2,2)
        #self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4608,2056)
        
        self.fc2 = nn.Linear(2056,512)
        self.fc3 = nn.Linear(512,512)
        self.fc4 = nn.Linear(512,120)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        
        x =self.pool(F.relu(self.conv1(x)))
        x =self.pool(F.relu(self.conv2(x)))
        x =self.pool(F.relu(self.conv3(x)))
        x =self.pool(F.relu(self.conv4(x)))
        
        x = x.view(x.size(0), -1) 
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc4(x),dim=1)
        
        return x
    
model = Net()
print(model)
if train_on_gpu:
    model.cuda()

# %% [code] {"id":"yuBtBeIy5ujm"}
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= 0.001)

# %% [code] {"id":"Szdq_wzz5uj1","outputId":"4fe5cd31-9900-43b5-c75a-7f29a48f3e45"}
n_epochs = 15

valid_loss_min = np.Inf 
for epoch in range(1, n_epochs+1):

    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_Loader:

        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
       
        output = model(data)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_Loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
    
    train_loss = train_loss/len(train_Loader.sampler)
    valid_loss = valid_loss/len(valid_Loader.sampler)
        
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_Fruits.pt')
        valid_loss_min = valid_loss


# %% [code] {"id":"WMzocbfq5ukE","outputId":"b9e2d465-3d08-47db-fb64-38bcaf1e625d"}
test_loss = 0.0
class_correct = list(0. for i in range(120))
class_total = list(0. for i in range(120))

model.eval()
for data, target in test_Loader:
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    _, pred = torch.max(output, 1)    
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

test_loss = test_loss/len(test_Loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(120):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

