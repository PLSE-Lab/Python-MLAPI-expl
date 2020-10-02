#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will use a vgg16 pretrained model to classify dogs vs cats. Moreover, we use an antialiased model from the paper *Making Convolutional Networks Shift-Invariant Again*, https://richzhang.github.io/antialiased-cnns/
# 
# 
# 

# In[ ]:


get_ipython().system(' echo "Downloading antialiased vgg16 model and pre-trained weights"')
get_ipython().system(' git clone https://github.com/adobe/antialiased-cnns.git')
get_ipython().system(' cd ./antialiased-cnns/;  wget -q https://www.dropbox.com/s/2a4fylumsxasie0/vgg16_lpf3.pth.tar?dl=0 -O ./weights/vgg16_lpf3.pth.tar')


# Setting the random seeds so we can reproduce the results.

# In[ ]:


import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(0)
import torchvision
import torchvision.transforms as transforms
import PIL


# Load the training set

# In[ ]:


DC = torchvision.datasets.ImageFolder(
    root      = "/kaggle/input/training_set/training_set", 
    transform = transforms.Compose([
        transforms.Resize(200),
        transforms.Pad(100, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ]))

dataloader = torch.utils.data.DataLoader(DC, batch_size=16, shuffle=True, num_workers=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the pretrained VGG16 model. We replace the last linear classifier layer by a new linear layer with two output neurons.

# In[ ]:


get_ipython().run_line_magic('cd', 'antialiased-cnns')
import models_lpf.vgg
model = models_lpf.vgg.vgg16(filter_size=3)
model.load_state_dict(torch.load('./weights/vgg16_lpf3.pth.tar')['state_dict'])
get_ipython().run_line_magic('cd', '..')

for param in model.parameters():
    param.requires_grad = False

feature_size = model.classifier[6].in_features
model.classifier[6] =  torch.nn.Linear(feature_size, 2)

model = model.to(device)
weights = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        weights.append(param)
        print("\t", name)


# Setting optimizer and loss function.

# In[ ]:


optimizer_ft = torch.optim.Adam(weights, lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
iteration = 0
num_epoch = 2


# Train the last layer for 2 epochs.

# In[ ]:


model.train()
for epoch in range(num_epoch):
    for x, y in dataloader:
        pred = model(x.to(device))
        loss = loss_fn(pred, y.to(device))
        optimizer_ft.zero_grad()
        loss.backward()
        optimizer_ft.step()
        iteration = iteration + 1
        print(f"\r[{iteration * 100.0 / num_epoch/len(dataloader):7.3f} % ] loss: {loss.item():5.3f}", end="")


# In[ ]:


torch.save(model.state_dict(), "vgg16_dogvscat_model.pt")
get_ipython().run_line_magic('rm', '-r antialiased-cnns')


# Compute the prediction accuracy on the test set

# In[ ]:


DC_test = torchvision.datasets.ImageFolder(
    root      = "/kaggle/input/test_set/test_set", 
    transform = transforms.Compose([
        transforms.Resize((224,224)),
#         transforms.Pad(100, padding_mode="reflect"),
#         transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ]))
    
def acc(model, dataset):
    model.eval()
    hit = 0
    total = 0
    dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    for x, y in dataloader_test:
        pred = model(x.to(device))
        hit = hit + sum( pred.argmax(dim=-1).cpu() == y ).item()
        total = total + x.size(0)

    model.train()
    return hit*100.0 / total


print("Test accuracy: ", acc(model, DC_test))
# print("Train acc: ", acc(model, DC))


# In[ ]:


dataloader_test = torch.utils.data.DataLoader(DC_test, batch_size=32, shuffle=True)

batch = next(iter(dataloader_test))
o = model(batch[0].to(device)).argmax(dim=-1)
labels =  [DC_test.classes[i][:-1] for i in o.tolist()]
true_labels =  [DC_test.classes[i] for i in (model(batch[0].to(device)).argmax(dim=-1).tolist())]
imgs =  batch[0] * torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1) +                     torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)

imgs = imgs.permute(0, 2, 3, 1)
import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 20, 10


for i in range(imgs.size(0)):
    plt.subplot(4, 8, i+1)
    plt.imshow(imgs[i])
    c = "b" if o[i].item() == batch[1][i].item() else "r"
    plt.text(0,20, labels[i], fontsize=16, color=c,
                      bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
    plt.axis("off")

