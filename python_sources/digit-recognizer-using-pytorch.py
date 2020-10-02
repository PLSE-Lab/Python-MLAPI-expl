#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import random_split
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# torch.Tensor.ndim = property(lambda x: len(x.shape))


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


get_ipython().system('unzip digit-recognizer.zip')


# In[ ]:


# path = "/content/Mnist"
train_path = "../input/digit-recognizer/train.csv"
test_path = "../input/digit-recognizer/test.csv"


# In[ ]:


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


# In[ ]:


# val_size = 8000
# train_size = len(train) - val_size
# train_data, val_data = random_split(train, [train_size, val_size])


# In[ ]:


Y = train["label"]
X = train.drop(labels = ["label"],axis = 1)


# In[ ]:


#Splitting
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2)
X_train.shape, Y_train.shape, X_val.shape, Y_val.shape


# In[ ]:


train_size = X_train.shape[0]
val_size = X_val.shape[0]
print("Training size {}, Validation size {} ".format(train_size, val_size))


# In[ ]:


#Normalization 
X_train = X_train / 255.0
X_val = X_val / 255.0
test = test / 255.0


# In[ ]:


X_train = np.array(X_train, np.float32)
test = np.array(test, np.float32)
Y_train = np.array(Y_train, np.long)
X_val = np.array(X_val, np.float32)
Y_val = np.array(Y_val, np.long)
print(X_train.shape)
print(test.shape)
print(X_val.shape)


# In[ ]:


plt.imshow(X_train[10].reshape(28, 28), cmap='gray')
print(Y_train[10])


# In[ ]:


type(X_train)


# In[ ]:


X_train = torch.from_numpy(X_train)
Y_train = torch.from_numpy(Y_train)
X_val = torch.from_numpy(X_val)
Y_val = torch.from_numpy(Y_val)
test = torch.from_numpy(test)


# In[ ]:


# transform_img = torchvision.transforms.Compose([
# #                                           torchvision.transforms.Resize((224, 224)),
#                                           torchvision.transforms.ToTensor(),
#                                           torchvision.transforms.Normalize(mean=[0.485], std=[0.229])
#                                           ])


# In[ ]:


# print(transform_img)


# In[ ]:


# data = torchvision.datasets.MNIST(path, train=True, transform=transform_img, target_transform=None, download=True)


# In[ ]:


train_data = torch.utils.data.TensorDataset(X_train, Y_train)
val_data = torch.utils.data.TensorDataset(X_val, Y_val)


# In[ ]:


# inputs = data.data
# labels = data.targets
# print(inputs.shape)


# print(labels[2])
# plt.imshow(inputs[2].numpy())


# In[ ]:


# dir(data)


# In[ ]:


batch = 32


# In[ ]:


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=2*batch, shuffle=True, num_workers=4)


# In[ ]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# In[ ]:


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(784, 32)
        # nn.ReLU(),
        self.fc2 = torch.nn.Linear(32, 16)
        # nn.ReLU(),
        self.fc3 = torch.nn.Linear(16, 10)
        # nn.Softmax()
#         self.fc1 = torch.nn.Linear(784, 10)
          
    def forward(self, x):
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
#         x = self.fc1(x)
        return x

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        preds = self(images)
        loss = F.cross_entropy(preds, labels)
        acc = accuracy(preds, labels)
        return {'val_loss': loss, 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


# In[ ]:


model = MLP()


# In[ ]:


print(model)


# In[ ]:


for para in model.parameters():
  print(para.shape)


# In[ ]:


# loss_fn = F.cross_entropy
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# In[ ]:


if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print(device)


# In[ ]:


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# In[ ]:


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[ ]:


train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)


# In[ ]:


# model.to(device=device)


# In[ ]:


# images, labels = next(iter(train_loader))


# In[ ]:


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# ## Training the model

# In[ ]:


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[ ]:


to_device(model, device)


# In[ ]:


history = [evaluate(model, val_loader)]
history


# In[ ]:


history += fit(5, 0.5, model, train_loader, val_loader)


# In[ ]:


history += fit(5, 0.1, model, train_loader, val_loader)


# In[ ]:


losses = [x['val_loss'] for x in history]
plt.plot(losses, '-x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs');


# In[ ]:


accuracies = [x['val_acc'] for x in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');


# ## Prediction on Samples

# In[ ]:


# test_data = torchvision.datasets.MNIST(path, train=False, transform=transform_img, target_transform=None, download=True)


# In[ ]:


# inputs = test_data.test_data
# labels = test_data.test_labels
# print(inputs.shape)


# print(labels[1])
# plt.imshow(inputs[1].numpy())


# In[ ]:


test_data = torch.utils.data.TensorDataset(test)


# In[ ]:


test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)


# In[ ]:


train_loader = DeviceDataLoader(train_loader, device)


# In[ ]:


model.eval()


# In[ ]:


def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()


# In[ ]:


img = test_data[1]
plt.imshow(img[0].reshape(28, 28), cmap='gray')
print('Predicted: ', predict_image(img[0], model))


# In[ ]:


img = test_data[1]
print(type(img))
print(type(img[0]))


# In[ ]:


# evaluate(model, test_loader)


# In[ ]:


# correct_ans = 0


# In[ ]:


# def correct_pred_count(pred, answer):
#     pred = torch.argmax(pred, dim=1)
#     correct_count_vector = (pred.data == answer.data)
#     correct_count = correct_count_vector.sum()
#     return correct_count


# In[ ]:


# for images, labels in test_loader:
    
#     images = images.view(images.size(0), -1)

#     images = images.to(device=device)
#     labels = labels.to(device=device)

#     preds = model(images)
    
#     correct_ans += correct_pred_count(preds, labels)


# In[ ]:


# accuracy = (correct_ans /  10000.0)


# In[ ]:


# accuracy


# In[ ]:


# import csv


# In[ ]:


# csv_path = "mnist.csv"
# file = open(csv_path, 'w')
# writer = csv.writer(file)
# writer.writerow(["ImageId", "Label"])


# In[ ]:


prediction = []
for i, images in enumerate(test_loader, 1):
  images = images[0]
#   images = images.to(device=device)

  preds = model(images)
#   print(i)
  preds = torch.argmax(preds, dim=1)
  prediction.append(preds.item())
#   writer.writerow([i, preds.item()])


# In[ ]:


len(prediction)


# In[ ]:


path = "CNN_submission2.csv"


# In[ ]:


sample_sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')


# In[ ]:


sample_sub = pd.DataFrame({"ImageId": list(range(1,len(prediction)+1)),
                         "Label": prediction})
sample_sub.to_csv('CNN_submission2.csv', index=False)
sample_sub.head()


# In[ ]:


file = [row.strip().split() for row in open(path)]


# In[ ]:


len(file)


# In[ ]:


file[len(file) - 1]


# In[ ]:


len(test_loader)


# ## Save and upload

# In[ ]:


saved_weights_fname='MNIST-feedforward.pth'


# In[ ]:


torch.save(model.state_dict(), saved_weights_fname)


# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


import jovian


# In[ ]:


project_name = "MNISt Feed forward"


# In[ ]:


jovian.commit(project=project_name, environment=None, outputs=[saved_weights_fname])


# In[ ]:




