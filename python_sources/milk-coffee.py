#!/usr/bin/env python
# coding: utf-8

# **0. Setup**
# 
# We used PyTorch as our main Deep Learning framework to implement our algorithms. We also made use of pretrainedmodels package from https://github.com/Cadene/pretrained-models.pytorch for latest state-of-the-art architectures.
# Besides that, scikit-learn was used for some metrics and Machine Learning algorithm as post-processing.

# In[ ]:


# import libraries
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import torchvision.models.resnet as resnet
import torchvision.models.densenet as densenet
import torchvision.models.inception as inception

import pretrainedmodels

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

import glob
import os
import random
import tqdm

from PIL import Image, ImageEnhance

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score


# **1. Data Preprocessing and Augmentation** 
# 
# The training data was splitted into 80% for training and 20% for validation. We have trained and tested models with different types of augmentation such as random horizontal flip, various scales, brightness, contrast and colors. However, changes in brightness, contrast and colors didn't improve our model accuracy, so we stuck to the horizontal flip, random cropping and shifting.

# In[ ]:


img_folder = 'data/train' # contain 30563
val_folder = 'data/val' # contain 7648 images
test_folder = 'data/test' # contain 16111 images

# data preprocessing and augmentation
transform_1 = transforms.Compose([transforms.Resize((248, 248)),
                                  transforms.RandomCrop((224, 224)),
                                  ])

transform_2 = transforms.Resize((224, 224))
trans = [transform_1, transform_2]

class Augment(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return transform_1(img)
        return transform_2(img)

# image augmentation and normalization
img_transform = transforms.Compose([Augment(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ])
test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                     ])


# **2. Training, Validating and Testing**
# 
# We trained different architectures for ensembling. For all the networks except Resnet18, we changed the last linear layer to match the number of classes (18) and finetuned about 15~20% number of layers (for examples, resnet152 has 152 layers so we trained the last 30 layers). For Resnet18, we trained it from scratch and got higher accuracy than finetuning. However, training other very deep models from scratch would take days so we decided not to do that. Most of the model would get 85+% on our validation set and 83+% on public score. Here are the wrapped models for finetuning.

# In[ ]:


class Dense161(nn.Module):
    def __init__(self):
        super(Dense161, self).__init__()
        self.net = densenet.densenet161(pretrained=True)
        for param in self.net.features.parameters():
            param.requires_grad = False
        for param in self.net.features.denseblock4.parameters():
            param.requires_grad = True
        self.net.classifier = nn.Linear(2208, 18)

    def forward(self, x):
        return self.net(x)
    
class Res101(nn.Module):
    def __init__(self):
        super(Res101, self).__init__()
        self.net = resnet.resnet101(pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False

        unfreezes = range(16, 23)
        for l in unfreezes:
            for param in self.net.layer3[l].parameters():
                param.requires_grad = True
        for param in self.net.layer4.parameters():
            param.requires_grad = True
        for param in self.net.fc.parameters():
            param.requires_grad = True
        self.net.fc = nn.Linear(2048, 18)

    def forward(self, x):
        return self.net(x)
    
class ResNext(nn.Module):
    def __init__(self):
        super(ResNext, self).__init__()
        self.net = pretrainedmodels.__dict__['resnext101_64x4d'](num_classes=1000, pretrained='imagenet')
        for param in self.net.parameters():
            param.requires_grad = False

        unfreezes = range(16, 23)
        for l in unfreezes:
            for param in self.net.features[6][l].parameters():
                param.requires_grad = True
        for param in self.net.features[7].parameters():
            param.requires_grad = True
        for param in self.net.last_linear.parameters():
            param.requires_grad = True
        self.net.last_linear = nn.Linear(2048, 18)

    def forward(self, x):
        return self.net(x)


# Each model was trained for 15 epochs with Adam optimizer; initial learning rate 1e-3, and decreased to 1e-4 at 7th epoch and 2e-5 at 12th epoch; weight decay 1e-4; and multiclass cross entropy loss. The model was validated after each epoch and the best model with best validation accuracy score was saved.

# In[ ]:


num_epoch = 15
num_workers = 6
batch_size = 128
checkpoint = ''

# load data using PyTorch Data Loader
train_set = ImageFolder(img_folder, transform=img_transform)
loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

valid_set = ImageFolder(valid_folder, transform=test_transform)
valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers)

gpu_option = torch.cuda.is_available()

model = Res101().cuda()
if checkpoint:
    model.load_state_dict(torch.load(checkpoint))
    
softmax = nn.Softmax().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 12], gamma=0.1)

model_name = 'res101'
log_file = open(model_name + '_log.txt', mode='w')
max_iter = len(loader)
iter_log = max_iter//15
best_model = None
best_accuracy = 0

for epoch in range(1, num_epoch+1):
    # train
    model.train()
    epoch_losses = []
    
    for i, data in enumerate(loader, start=1):
        img, label = data
        img = Variable(img).float().cuda()
        label = Variable(label).cuda()
        
        output = model(img)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.data[0])
        avg_loss = sum(epoch_losses)/i

        if i % iter_log == 1:
            print('Epoch: %2d/%2d  Iter: %4d/%4d  Average Loss: %.5f    Loss: %.5f'
                  % (epoch, num_epoch, i, max_iter, avg_loss, loss.data[0]))

    # validate
    model.eval()

    true_labels = []
    pred_labels = []

    test_losses = []
    for data in tqdm.tqdm(valid_loader):
        img, label = data
        img = Variable(img, volatile=True).float().cuda()
        label = Variable(label, volatile=True).cuda()

        output = model(img)

        output = softmax(output)
        loss = criterion(output, label)
        test_losses.append(loss.data[0])

        output = torch.max(output, 1)[1]

        true_labels += list(label.data.cpu().squeeze().numpy())

        pred = output.squeeze().data.cpu().numpy()
        pred_labels += list(pred)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    report = classification_report(true_labels, pred_labels, target_names=valid_set.classes)
    print(report)

    accuracy = accuracy_score(true_labels, pred_labels)
    print('Average Test Loss: %.5f' % (sum(test_losses)/len(valid_loader)))
    print('Accuracy: {}\n'.format(accuracy))
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model.state_dict()

    log_file.write('Validating epoch {}:\n'.format(epoch))
    log_file.write(report + '\n')
    log_file.write('Average Test Loss: %.5f\n' % (sum(test_losses) / len(valid_loader)))
    log_file.write('Accuracy: {}\n\n'.format(accuracy))

save_name = 'model/{}_{}.pth'.format(model_name, epoch)
torch.save(model.state_dict(), save_name)

log_file.close()


# After getting the best parameters, we saved the output probabilities of the testing images for further post processing. 

# In[ ]:


test_set = TestSet(test_folder, transform=test_transform)
loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

if checkpoint:
    model.load_state_dict(torch.load(checkpoint))
    print('Loading model', checkpoint)
else:
    raise ValueError

model.eval()
softmax = nn.Softmax().cuda()

pred_labels = []
all_prob = []

for data in tqdm.tqdm(loader):
    img = Variable(data, volatile=True).float().cuda()
    output = model(img)
    output = softmax(output)

    prob = output.squeeze().data.cpu().numpy()
    all_probs.append(probs)

    output = torch.max(output, 1)[1]

    pred = output.squeeze().data.cpu().numpy()
    pred_labels += list(pred)

submission = {'id': range(1, len(pred_labels) + 1), 
              'category': pred_labels}
df = pd.DataFrame(submission)
df.to_csv(out_name + '_sub.csv', columns=['id', 'category'], index=False)

all_probs = np.concatenate(all_probs)
np.save('test_probs/' + model_name + '_prob.npy', all_probs)


# **3. Ensembling**
# 
# Ensembling is a trick to increase overall accuracy by averaging the output of several model. This works only when the features extracted from different models are uncorrelated. After averaging the probabilities of 6 models including densenet161, resnet152, resnet101, resnet50, resnet18 and inceptionresnetv2, we got 85+% on public leaderboard.

# In[ ]:


num_imgs = [677, 530, 386, 383, 395, 423, 621, 638, 290, 458, 598, 584, 364, 324, 336, 259, 162, 220]
labels = []
for i in range(len(num_imgs)):
    labels += [i] * num_imgs[i]
labels = np.array(labels)

dense = np.load('test_probs/dense161_prob.npy')
res152 = np.load('test_probs/res152_prob.npy')
res101 = np.load('test_probs/res101_prob.npy')
res50 = np.load('test_probs/res50_prob.npy')
res18 = np.load('test_probs/res18_prob.npy')
incep = np.load('test_probs/inceptionresv2_prob.npy')

final_result = np.argmax((dense + res152 + res101 + res50 + res18 + incep), axis=1)
df = pd.DataFrame({'id': range(1, len(pred) + 1),
                   'category': pred})
df.to_csv('ensemble.csv', columns=['id', 'category'], index=False)


# **4. Weighted Ensembling**
# 
# However, we wanted to do more than just averaging the probabilities. As each model would contribute different portions of the final accuracy. We found the best combination of weights by searching in a specific range. After weighing the models, the accuracy improve from 85.5+% to 86.2+% on leaderboard. Using this way, we got our final score on the private leaderboard 0.85662.

# In[ ]:


start = 5
end = 12

best_score = 0
best_coefs = None
best_result = None
for x1 in range(start, end+1, 1):
    for x2 in range(start, end+1, 1):
        for x3 in range(start, end+1, 1):
            for x4 in range(start, end+1, 1):
                for x5 in range(start, end+1, 1):
                    for x6 in range(start, end+1, 1):
                        final = np.argmax(x1*dense + x2*res152 + x3*res101 + x4*res50 + x5*res18 + x6*incep, axis=1)
                        acc = accuracy_score(labels, final)
                        if acc > best_score:
                            best_score = acc
                            best_coefs = [x1, x2, x3, x4, x5, x6]
                            best_result = final

# best coefs are 9, 11, 9, 8, 10, 7 or
# best_result = np.argmax((9 * dense + 11 * res152 + 9 * res101 + 8 * res50 + 10 * res18 + 7 * incep), axis=1)

df = pd.DataFrame({'id': range(1, len(pred) + 1),
                   'category': pred})
df.to_csv('weighted_ensemble.csv', columns=['id', 'category'], index=False)


# **5. Machine Learning Ensembling**
# 
# Weight searching is very inefficient and the fact is that we cannot search for all the combination. We applied some Machine Learning algorithms to get the best ensembling result. We came up with the best 3 that improve our validation accuracy:
# * Weight Layer: training a Convolution layer of filter size (6, 1) on the (6, 18) probability array.
# * Support Vector Machine: tuning a svm using the probability of 6 models and output the probabilities of 18 classes.
# * K-Nearest Neighbor: finding the nearest 15 datapoints in the dataset to find out the most relevant class of the image.
# 
# Using any of these 3 algorithms would boost our validation accuracy by 1% (89%). Strangely, when tested on the 30% of the test set, our public score is lower than the previous weight searching submission; however, when tested on the other 70% of the test set, all 3 of them got higher score than the weight searching on the private score (from 0.85662 to 0.85857). In the end, we chose the weight searching method (0.85662) as the final submission since it performed better than the Machine Learning on public leaderboard. (One of the reason is maybe that the public test set only has about 4800 images and does not reflect the accurate score)

# In[ ]:


kfold = KFold(n_splits=6, shuffle=True, random_state=123)

#######################
# tuning SVM
best_score = 0
best_param = 0
for c in [0.001 ,0.01, 0.1, 1, 2, 3, 4, 5, 10, 20, 50, 100, 1000, 10000]:
    clf = SVC(C=c, kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=1e-3)

    final_score = 0
    for train_ids, test_ids in kfold.split(all_data, labels):
        x_train, y_train = all_data[train_ids], labels[train_ids]
        x_test, y_test = all_data[test_ids], labels[test_ids]

        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)

        final_score += accuracy_score(y_test, pred)
    
    final_score /= 6
    
    if final_score > best_score:
        best_score = final_score
        best_param = c
# best c = 2

#######################
# tuning K-Nearest Neighbor
best_score = 0
best_param = 0
for n in range(5, 51, 5):
    clf = KNeighborsClassifier(n_neighbors=n, weights='uniform', leaf_size=30, p=1, metric='minkowski')

    final_score = 0
    for train_ids, test_ids in kfold.split(all_data, labels):
        x_train, y_train = all_data[train_ids], labels[train_ids]
        x_test, y_test = all_data[test_ids], labels[test_ids]

        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)

        final_score += accuracy_score(y_test, pred)
    
    final_score /= 6
    
    if final_score > best_score:
        best_score = final_score
        best_param = n
# best n is 15

#######################
# training weight layer
num_models = 6
weight_layer = nn.Conv2d(1, 1, kernel_size=(num_models, 1), stride=1, padding=0, bias=False)
# input_prob of size 7648x6x18
max_iter = 1000
softmax = nn.Softmax()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(weight_layer.parameters(), lr=1e-3, weight_decay=1e-4)

for i in max_iter:
    input_prob = Variable(input_prob)
    labels = Variable(labels)
    
    out_prob = weight_layer(input_prob)
    loss = criterion(out_prob, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# **7. Other techniques**
# 
# We also applied some other techniques but they didn't help improve the accuracy:
# * Oversampling: to deal with the unbalanced dataset, oversample classes which have fewer images.
# * Weighing each class: weigh low accuracy classes with higher coefficients when calculating the loss.
# * Hard negative sampling: sample the negative samples separately with small learning rate to improve the model on difficult samples.
# * Scrapping google images: getting more data.
# 
# We found that some classes are not mutually exclusive such as different women top classes. Therefore, techniques like oversampling and negative sampling may worsen the model.
