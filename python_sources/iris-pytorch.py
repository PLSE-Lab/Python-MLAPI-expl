import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def load_data():
    # Load the data
    data = pd.read_csv('../input/iris.data', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type'])
    
    # Convert categorical value to numerical value
    classes = data['type'].unique()
    data['type'] = data['type'].map({name:i for i, name in enumerate(classes)})
    
    # Seperate features and labels from data
    y = data['type']
    x = data.drop('type', axis=1)
    
    # Split the data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,shuffle=True)
    
    # Convert to arrays
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values

    # Map to tensor
    x_train, x_test, y_train, y_test = map(torch.Tensor, (x_train, x_test, y_train, y_test))
    y_train = y_train.long()
    y_test = y_test.long()
    
    return x_train, x_test, y_train, y_test
    
class Net(nn.Module):
    
    def __init__(self, n_input, n_output, n_hidden):
        super(Net, self).__init__()
        
        self.hidden = nn.Linear(n_input, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)
        
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x
        
def accuracy(x, y):
    out = model(x)
    _, predicted = torch.max(out.data, 1)
    
    correct_pred = torch.sum(predicted == y)
    total = len(x)
    accuracy = 100 * correct_pred / total
    
    return accuracy
    

lr = 0.05
epochs = 100
bs = 24

x_train, x_test, y_train, y_test = load_data()

n_features = x_train.shape[1]
n_classes = 3
n_hidden = 8

# Convert to TensorDataset
train_ds = TensorDataset(x_train, y_train)
test_ds = TensorDataset(x_test, y_test)

train_dl = DataLoader(train_ds, batch_size=bs)

model = Net(n_features, n_classes, n_hidden)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),lr=lr)

for epoch in range(epochs):
    
    for xb, yb in train_dl:
        
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        
    if (epoch + 1) % 10 == 0:
            print('Epoch:', epoch+1,'/',epochs,'\tLoss:',loss.item())
            
print('Train accuracy:', accuracy(x_train, y_train))
print('Test accuracy:',accuracy(x_test, y_test))