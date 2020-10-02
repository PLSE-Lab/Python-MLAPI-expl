#!/usr/bin/env python
# coding: utf-8

# # Hi!
# This is my first kernel on Kaggle. This is Molecular Classification of Cancer by Gene Expression Monitoring -- Pytorch Baseline. I hope you can understand the code and it will help you.  

# # Basic Steps
# 
# * --> Importing dependecies
# * --> Define some parameters
# * --> Preparing CSV to Tensor
# * --> Create Artifical Neural Network
# * --> Train the model
# * --> See the Results and Benchmark on Test set which is unseen by model.

# # Dependecies

# In[ ]:


import os
import time 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# # Some Parameters

# In[ ]:


DIR = "../input/" # DIR must consist "actual.csv", "data_set_ALL_AML_independent.csv", "data_set_ALL_AML_train.csv" files.
MODEL_PATH = "./model.pt"

EPOCHS = 100000 # This model -cause of data and model parameters-  is pretty fast.
BATCH_SIZE  = 72
THRESH_VAL  = 0.5

# All ratios sum must be 1.
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
TEST_RATIO  = 0.2


# In[ ]:


# training and evaluating model on cpu or gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# # Preparing Data

# In[ ]:


df1 = pd.read_csv(DIR+"actual.csv")
df2 = pd.read_csv(DIR+"data_set_ALL_AML_independent.csv")
df3 = pd.read_csv(DIR+"data_set_ALL_AML_train.csv")


# In[ ]:


# prepare the csv
def csv2data(_df_):
    df = _df_[[col for col in _df_.columns if "call" not in col]]  # drop "call" columns
    df = df.T
    df2 = df.drop(['Gene Description','Gene Accession Number'],axis=0)
    df2.index = pd.to_numeric(df2.index)
    df2.sort_index(inplace=True)
    return df2


# In[ ]:


final_x_df = pd.concat([csv2data(df3), csv2data(df2)])
final_y_df = df1


# In[ ]:


x = np.empty((len(final_x_df), len(final_x_df.columns)), dtype=np.float32)
y = np.empty((len(final_x_df), 1), dtype=np.float32)

for ix in range(len(final_x_df)):
    # x
    patient_gene_vals = final_x_df.iloc[ix].tolist()
    
    # y
    # Binary classification
    cancer_type = final_y_df['cancer'][ix]
    if cancer_type == 'ALL':
        label = 0
    elif cancer_type == 'AML':
        label = 1
        
    # Store the values
    x[ix] = patient_gene_vals 
    y[ix] = label


# In[ ]:


x_ = torch.from_numpy(x)
y_ = torch.from_numpy(y)


# In[ ]:


dataset = TensorDataset(x_, y_)

train_size = int(TRAIN_RATIO * len(dataset))
val_size   = int(VAL_RATIO * len(dataset))
test_size   = len(dataset) - train_size - val_size

train_data, test_data, val_data = random_split(dataset, [train_size, test_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data  , batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


# # Creating Model

# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # LAYER 1
        self.fc1 = nn.Linear(len(final_x_df.columns), 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        
        # LAYER 2
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        # LAYER 3
        self.fc3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
         
        # OUTPUT LAYER
        self.fc4 = nn.Linear(128, 1)
        

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))), p=0.5)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))), p=0.5)
        x = F.dropout(F.relu(self.bn3(self.fc3(x))), p=0.5)
        x = torch.sigmoid(self.fc4(x))
        return x


# In[ ]:


net = Net()
net.to(device)
print(net)


# In[ ]:


criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)


# In[ ]:


def threshold_arr(array):
    for ix, val in enumerate(array):
        if val>=THRESH_VAL:
            array[ix]=1
        else:
            array[ix]=0
    return array


# # Training

# In[ ]:


for epoch in range(EPOCHS):  # loop over the dataset multiple times
    net.train()
    running_loss = 0.0
    running_acc = 0.0
    running_f1 = 0.0
    for i, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        predictions = net(inputs)

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        predictions = np.array(predictions.cpu().detach())
        labels      = np.array(labels.cpu().detach())
        acc = accuracy_score(labels, threshold_arr(predictions))
        f1  = f1_score(labels, threshold_arr(predictions))
        
        # print statistics
        running_loss += loss.item()
        running_acc += acc 
        running_f1  += f1
        
        if epoch%1000==999:
            net.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_f1 = 0.0
            
            for ix, val_data in enumerate(val_loader):
                inputs, labels = val_data
                inputs, labels = inputs.to(device), labels.to(device)
                
                predictions = net(inputs)
                loss = criterion(predictions, labels)
                
                predictions = np.array(predictions.cpu().detach())
                labels      = np.array(labels.cpu().detach())
                acc = accuracy_score(labels, threshold_arr(predictions))
                f1  = f1_score(labels, threshold_arr(predictions))
                
                val_loss += loss.item()
                val_acc += acc 
                val_f1  += f1
                      
            print('Epoch {}/{}, loss: {}, acc: {}, f1: {}, val_loss: {}, val_acc: {}, val_f1: {}'.format(epoch + 1, EPOCHS, round(running_loss,6), round(running_acc,3), round(running_f1,3), round(val_loss,6), round(val_acc,3), round(val_f1,3)))  
            running_loss = 0.0
            running_acc  = 0.0 
            running_f1   = 0.0
            
print('Finished Training')


# In[ ]:


torch.save(net.state_dict(), MODEL_PATH)


# # Testing the Model

# In[ ]:


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[ ]:





# In[ ]:


# CPU AND GPU BENCHMARK
net.load_state_dict(torch.load(MODEL_PATH))

archs = ["cpu", "cuda"]

for l in range(10):
    for arch in archs:
        net.to(arch)
        net.eval()
        test_loss = 0.0
        for ix, test_data in enumerate(test_loader):
            inputs, labels = test_data
            inputs, labels = inputs.to(arch), labels.to(arch)

            tic =  time.perf_counter()
            predictions = net(inputs)
            toc =  time.perf_counter()
            loss = criterion(predictions, labels)
            test_loss += loss.item()   

        print("{}: Single row of data costs {} ms".format(arch,(toc-tic)*1000/inputs.shape[0]))
        print('Test Loss: {}'.format(test_loss)) 
        print("____________")
    print("***************")


# In[ ]:


# convert tensors to numpy array
predictions = np.array(predictions.cpu().detach())
labels      = np.array(labels.cpu().detach())


# In[ ]:


# Calculate Metrics 
acc = accuracy_score(labels, threshold_arr(predictions))
f1  = f1_score(labels, threshold_arr(predictions))
print(classification_report(labels, predictions))
print("\n")
print("ACCURACY: {}, F1_SCORE: {}".format(acc, f1))
print("\n")
cnf_matrix = confusion_matrix(labels, threshold_arr(predictions))

plot_confusion_matrix(cm           = cnf_matrix, 
                      normalize    = False,
                      target_names = ['0', '1'],
                      title        = "Confusion Matrix")


# 
# I hope you have enjoyed reading this workflow and have learned something new :)
# 
# If you have some advices or bug about my work, please share to all of us in the comments.
