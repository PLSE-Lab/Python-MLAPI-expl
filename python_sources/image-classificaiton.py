#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import math
import time
import itertools
import torch.nn as nn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

import os
import cv2
import matplotlib.pyplot as plt
import glob
import pandas as pd
import psutil
print(os.listdir("../input"))

np.random.seed(10)
USE_GPU = True


# In[ ]:


def load_sudo_data(directory_location, print_random=False, size=(100,100)):
    files = glob.glob(directory_location)
    img_file_name, img_category = [], []
    
    for file in files:
        img_category.append(file.split("/")[-1].split(".")[0])
        img_file_name.append(file)
    
    df = pd.DataFrame({"Img_filename": img_file_name, "Img_Category": img_category})
    
    if print_random:
        v = np.random.randint(0,len(files))
        image = cv2.resize(cv2.imread(df.Img_filename[v], cv2.IMREAD_GRAYSCALE), size)
        image_cat = df.Img_Category[v]
        
        plt.imshow(image, cmap="gray")
        plt.title(image_cat)
    
    df = shuffle(df, random_state=10)
    df.reset_index(drop=True, inplace=True)
    return df

df = load_sudo_data("../input/train/train/*", print_random=True)
df_train, df_val = train_test_split(df, test_size=0.05, random_state=10)


# In[ ]:


print (df_train.shape)
print (df_val.shape)


# In[ ]:


def load_batch(df, batch_number, batch_size=100, size=(100, 100)):
    
    batch_start_index = batch_number * batch_size
    if batch_start_index + batch_size >= len(df):
        batch_end_index = len(df)
    else:
        batch_end_index = batch_start_index + batch_size
        
    X, y = [], [] 
    category_to_index = {'cat': 0, 'dog': 1}
    index_to_category = {0: 'cat', 1: 'dog'}
    
    for index, dp in df[batch_start_index:batch_end_index].iterrows():
        X.append(cv2.resize(cv2.imread(dp.Img_filename, cv2.IMREAD_GRAYSCALE), size))
        y.append(category_to_index[dp.Img_Category])
            
    return X, y


# In[ ]:


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[ ]:


def print_mem(itnum,bnum): 
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**20
    return 'iteration: {} batchnum {} memory use: {}MB'.format(itnum, bnum, memoryUse)


# In[ ]:


class ImageClassifier2layer(nn.Module):
    def __init__(self, input_channels, conv_l1_channels, conv_l2_channels, 
                 conv_l1_kernelsize, conv_l2_kernelsize, 
                 conv_l1_padding, conv_l2_padding, 
                 conv_l1_stride, conv_l2_stride,
                 conv_l1_pool, conv_l2_pool, 
                 size, dropout=0.2):
        super(ImageClassifier2layer, self).__init__()
        self.size = size
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(input_channels, conv_l1_channels, kernel_size=conv_l1_kernelsize, padding=conv_l1_padding, stride=conv_l1_stride),
            nn.BatchNorm2d(conv_l1_channels),
            nn.ReLU(),
            nn.MaxPool2d(conv_l1_pool))
             
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(conv_l1_channels, conv_l2_channels, kernel_size=conv_l2_kernelsize, padding=conv_l2_padding, stride=conv_l2_stride),
            nn.BatchNorm2d(conv_l2_channels),
            nn.ReLU(),
            nn.MaxPool2d(conv_l2_pool)
        )
        self.fc1 = nn.Linear(400, 10)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(10, 1)
        self.dropout2 = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, batch_size):
        x = x.view(batch_size, 1, self.size[0], self.size[1])
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        
        out = out.view(batch_size, -1)
        out = self.fc1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


# In[ ]:


def train(classifier, criterion, optimizer, batch_size, data, size, iter_num, calc_accuracy=False):
    iteration_loss = 0.0
    correct_predicted = 0.0
    
    classifier = classifier.train()
    
    for batch_number in range(0, math.ceil(len(data)/batch_size)):
#         print (print_mem(iter_num, batch_number))
        input, output = load_batch(data, batch_number, batch_size=batch_size, size=size)
        current_batch_size = len(input)
        
        if USE_GPU and torch.cuda.is_available():
            input_tensor = torch.tensor(input, dtype=torch.float).cuda()
            output_tensor = torch.tensor(output, dtype=torch.float).view(-1, 1).cuda()
        else:
            input_tensor = torch.tensor(input, dtype=torch.float)
            output_tensor = torch.tensor(output, dtype=torch.float).view(-1, 1)
            
        optimizer.zero_grad()
        output = classifier(input_tensor, current_batch_size)
        
        loss = criterion(output, output_tensor)
        loss.backward()
        
        optimizer.step()
        
        iteration_loss += loss.item() * current_batch_size
        correct_predicted += get_accuracy(output_tensor, output) * current_batch_size
        torch.cuda.empty_cache()
        
    return iteration_loss/float(len(data)), correct_predicted/float(len(data))


# In[ ]:


def validate(classifier, criterion, data, size):
    classifier = classifier.eval()
    
    input, output = load_batch(data, 0, batch_size=len(data), size=size)
    
    if USE_GPU and torch.cuda.is_available():
        input_tensor = torch.tensor(input, dtype=torch.float).cuda()
        output_tensor = torch.tensor(output, dtype=torch.float).view(-1, 1).cuda()
    else:
        input_tensor = torch.tensor(input, dtype=torch.float)
        output_tensor = torch.tensor(output, dtype=torch.float).view(-1, 1)
    
    output = classifier(input_tensor, len(data))
    
    loss = criterion(output, output_tensor)
    acc = get_accuracy(output_tensor, output)
    
    return loss.item(), acc


# In[ ]:


def get_accuracy(true, predicted):
    true = true.tolist()
    pred = predicted.round().tolist()
    
    return accuracy_score(true, pred)


# In[ ]:


def trainiter(classifier, criterion, optimizer, n_iters, batch_size, size, print_every=100):
    start = time.time()
    training_error, validation_error, training_accuracy, validation_accuracy = [], [], [], []
    for iter in range(1, n_iters + 1):
        train_error, train_acc = train(classifier, criterion, optimizer, batch_size, df_train, size, iter)
        val_error, val_acc = validate(classifier, criterion, df_val, size)
        
        training_error.append(train_error)
        training_accuracy.append(train_acc)
        validation_error.append(val_error)
        validation_accuracy.append(val_acc)
        
        if iter % print_every == 0:
            print('%s (%d %d%%) Train Error = %.4f Train Acc = %.4f Val Error = %.4f Val Acc = %.4f' 
                  % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, train_error, train_acc, val_error, val_acc))

    return training_error, training_accuracy, validation_error, validation_accuracy


# In[ ]:


INP_CHANNELS = 1
CONV1_CHANNELS = 32
CONV2_CHANNELS = 16
CONV1_KERNEL_SIZE = 5
CONV2_KERNEL_SIZE = 3
CONV1_PADDING = 2
CONV2_PADDING = 1
CONV1_STRIDE = 3
CONV2_STRIDE = 2
CONV1_POOL = 2
CONV2_POOL = 2
BATCH_SIZE = 1000
SIZE = (128, 128)
DROPOUT = 0.2
LEARNING_RATE = 0.001

clf = ImageClassifier2layer(INP_CHANNELS, CONV1_CHANNELS, CONV2_CHANNELS, CONV1_KERNEL_SIZE, CONV2_KERNEL_SIZE, 
                            CONV1_PADDING, CONV2_PADDING, CONV1_STRIDE, CONV2_STRIDE, CONV1_POOL, CONV2_POOL, 
                            SIZE, DROPOUT)

if USE_GPU and torch.cuda.is_available():
    clf.cuda()
else:
    pass

criterion = nn.BCELoss();
optimizer = torch.optim.Adam(clf.parameters(), lr=LEARNING_RATE);


# In[ ]:


training_error, training_accuracy, validation_error, validation_accuracy = trainiter(clf, criterion, optimizer, 200, BATCH_SIZE, SIZE, print_every=20)


# In[ ]:


def load_test_data(directory_location, size=(100,100)):
    files = glob.glob(directory_location)
    
    X_test = []
    id = []
    for file in files:
        X_test.append(cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), size))
        id.append(file.split("/")[-1].split(".")[0])
        
    return X_test, id


# In[ ]:


def predict(classifier, X):
    classifier = classifier.eval()
    
    if USE_GPU and torch.cuda.is_available():
        X_tensor = torch.tensor(X, dtype=torch.float).cuda()
    else:
        X_tensor = torch.tensor(X, dtype=torch.float)
    
    predicted = classifier(X_tensor, len(X))
    
    return predicted.round().tolist()


# In[ ]:


X_test, ids = load_test_data("../input/test1/test1/*", size=SIZE)
predicted = predict(clf, X_test)
pred = list(itertools.chain(*predicted))

df = pd.DataFrame({"id": ids, "label": pred})
df.to_csv("submission.csv", index=False)


# In[ ]:


plt.plot(range(0, len(training_error)), training_error)
plt.plot(range(0, len(validation_error)), validation_error)

