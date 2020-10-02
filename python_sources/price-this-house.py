#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# In[ ]:


house_prices = train_data['SalePrice']
ax = sns.distplot(house_prices / 1000, kde=False)
ax.set_xlabel('House Prices * 1000')
ax.set_title('House prices')


# In[ ]:


train_data.info()


# In[ ]:


def plot_classify_feature(feature):
    facet = sns.FacetGrid(train_data, hue=feature,aspect=4)
    facet.map(sns.distplot,'SalePrice', kde=False)
    facet.add_legend()
 
    plt.show()


# In[ ]:


def plot_number_feature(feature):
    sns.regplot(x=feature, y = 'SalePrice', data=train_data)


# In[ ]:


plot_classify_feature('MSSubClass')


# In[ ]:


plot_classify_feature('MSZoning')


# In[ ]:


plot_number_feature('LotFrontage')


# In[ ]:


plot_number_feature('LotArea')


# In[ ]:


plot_classify_feature('OverallQual')


# In[ ]:


# train_data['SalePrice'] = np.log1p(train_data['SalePrice'])


# In[ ]:


# #MSSubClass=The building class
# train_data['MSSubClass'] = train_data['MSSubClass'].apply(str)

# test_data['MSSubClass'] = test_data['MSSubClass'].apply(str)


# #Changing OverallCond into a categorical variable
# train_data['OverallCond'] = train_data['OverallCond'].astype(str)
# train_data['OverallQual'] = train_data['OverallQual'].astype(str)

# test_data['OverallCond'] = test_data['OverallCond'].astype(str)
# test_data['OverallQual'] = test_data['OverallQual'].astype(str)


# #Year and month sold are transformed into categorical features.
# train_data['YrSold'] = train_data['YrSold'].astype(str)
# train_data['MoSold'] = train_data['MoSold'].astype(str)

# test_data['YrSold'] = test_data['YrSold'].astype(str)
# test_data['MoSold'] = test_data['MoSold'].astype(str)


# In[ ]:


# from scipy.stats import skew

# for feature in train_data.columns.values:
#     if train_data[feature].dtype == 'object':
#         train_data[feature].fillna('UNKNOWN', inplace=True)
#     else:
#         train_data[feature].fillna(train_data[feature].median(), inplace=True)
# #         if abs(skew(train_data[feature])) > 0.5:
# #             train_data[feature] = np.log1p(train_data[feature])
        
# for feature in test_data.columns.values:
#     if test_data[feature].dtype == 'object':
#         test_data[feature].fillna('UNKNOWN', inplace=True)
#     else:
#         test_data[feature].fillna(test_data[feature].median(), inplace=True)
# #         if abs(skew(test_data[feature])) > 0.5:
# #             test_data[feature] = np.log1p(test_data[feature])


# In[ ]:


alldata = pd.concat([train_data.drop('SalePrice', axis=1), test_data])
alldata = alldata.drop('Id', axis=1)


# In[ ]:


alldata['MSSubClass'] = alldata['MSSubClass'].apply(str)


# In[ ]:


alldata = pd.get_dummies(alldata, dummy_na=True)


# In[ ]:


alldata = alldata.fillna(alldata.median())


# In[ ]:


alldata.isnull().values.any()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
alldata = pd.DataFrame(scaler.fit_transform(alldata), columns = alldata.columns)

alldata.head()


# In[ ]:


alldata.shape


# In[ ]:


y = train_data['SalePrice']

X = alldata[:len(train_data)]
X_test = alldata[len(train_data):]


# In[ ]:


# for feature in X.columns.values:
#     if feature not in X_test:
#         X_test[feature] = 0
        
# for feature in X_test.columns.values:
#     if feature not in X:
#         X[feature] = 0


# In[ ]:


# min_value = X.min()
# max_value = X.max()

# min_max_equal = min_value == max_value


# In[ ]:


# for feature in min_max_equal.index:
#     if not min_max_equal[feature]:
#         X[feature] = (X[feature] - min_value[feature]) / (max_value[feature] - min_value[feature])
#         X_test[feature] = (X_test[feature] - min_value[feature]) / (max_value[feature] - min_value[feature])


# In[ ]:


# from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# model.fit(X, y)
# predictions = model.predict(X_test)


# In[ ]:


# important_features = []
# not_important_features = []

# # Print the name and gini importance of each feature
# for feature in zip(X.columns.values, model.feature_importances_):
#     if feature[1] > 0.01:
#         important_features.append(feature)
#     else:
#         not_important_features.append(feature)


# In[ ]:


# X = X[[important_feature[0] for important_feature in important_features]]
# X_test = X_test[[important_feature[0] for important_feature in important_features]]


# In[ ]:


# from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# model.fit(X, y)
# predictions = model.predict(X_test)

# output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': np.floor(np.expm1(predictions / 1000))})
# output.to_csv('random_forest.csv', index=False)


# In[ ]:


from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np
import sys


# In[ ]:


##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Hyperparameters
random_seed = 0
learning_rate = 0.01
num_epochs = 1428
batch_size = 30
print(batch_size)

# Architecture
num_features = len(X.columns.values)
print(len(X.columns.values))


# In[ ]:


X = torch.tensor(X.values)
y = torch.tensor(y.values).view(-1, 1)

train_loader = DataLoader(dataset=list(zip(X, y)),
                          batch_size=batch_size,
                          shuffle=True)


# In[ ]:


X_test = torch.tensor(X_test.values)


# In[ ]:


##########################
### MODEL
##########################

class HousePricingModel(torch.nn.Module):
    def __init__(self, num_features):
        super(HousePricingModel, self).__init__()
        self.fc1 = torch.nn.Linear(num_features, 144)
        self.fc2 = torch.nn.Linear(144, 72)
        self.fc3 = torch.nn.Linear(72, 18)
        self.fc4 = torch.nn.Linear(18, 1)
        
        self.bn1 = torch.nn.BatchNorm1d(144)
        self.bn2 = torch.nn.BatchNorm1d(72)
        self.bn3 = torch.nn.BatchNorm1d(18)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = F.relu(self.fc4(x)) + sys.float_info.epsilon

        return x

model = HousePricingModel(num_features=num_features)

model.to(device)


# In[ ]:


##########################
### COST AND OPTIMIZER
##########################

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[ ]:


lowest_cost = 100
lowest_cost_epoch = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    batch_number = 0
    for batch_idx, (features, targets) in enumerate(train_loader):        
        ### FORWARD AND BACK PROP
        logits = model(features.float().to(device))

        cost = torch.sqrt(criterion(torch.log(logits), torch.log(targets.float().to(device))))
        optimizer.zero_grad()
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        train_loss += cost.item()
        batch_number += 1        
    else:
        print ('Epoch: %03d/%03d | Cost: %.4f' %(epoch+1, num_epochs, train_loss/batch_number))    
        lowest_cost, lowest_cost_epoch = (train_loss/batch_number, epoch) if lowest_cost > train_loss/batch_number else (lowest_cost, lowest_cost_epoch)


print(f'lowest cost epoch is {lowest_cost_epoch} with cost of {lowest_cost}')


# In[ ]:


predictions = model.forward(X_test.float().to(device))


# In[ ]:


predictions = [element.item() for element in predictions.flatten()]


# In[ ]:


output = pd.DataFrame({'Id': test_data.Id.tolist(), 'SalePrice': predictions})
output.to_csv('deep_learning.csv', index=False)


# In[ ]:


output.head()


# In[ ]:




