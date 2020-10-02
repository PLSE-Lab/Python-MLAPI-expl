#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Importing necessary libraries**

# In[ ]:


import torch 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


admissiondata=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')


# In[ ]:


admissiondata


# **Since,there is no use of serial no its better to remove it**

# In[ ]:


del admissiondata['Serial No.']


# **Checking datatypes**

# In[ ]:


admissiondata.dtypes


# **Checking if null exists in dataset**

# In[ ]:


admissiondata.isna().sum()


# # **Visualizing relationship between values** 

# In[ ]:


g = sns.PairGrid(admissiondata)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3, legend=False);


# * **From above graphs you can see the chance of admission has relation with GRE,TOEFl score,University rating,SOP,LOR,CGPA but not research**
# * **Other relations in between these values are also visible but for this specific problem will focus in the the above mentioned**

# In[ ]:


#GRE score,TOEFl score,University rating,SOP,LOR,CGPA
input_arrays=admissiondata[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA']].to_numpy()
target_array=admissiondata[['Chance of Admit ']].to_numpy()


# **Converting pandas dataframe to numpy to feed it in as input in neural network**

# In[ ]:


input_arrays=torch.tensor(input_arrays)
target_arrays=torch.tensor(target_array)


# # Model making
# Making model with a linear layer for prediction

# In[ ]:


import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module): 
  
    def __init__(self): 
        super(Model, self).__init__() 
        self.linear = torch.nn.Linear(input_arrays.shape[1], target_arrays.shape[1]) 
  
    def forward(self, x): 
        y_pred = self.linear(x) 
        return y_pred
    
    
model = Model()
criterion = nn.MSELoss(size_average = False) 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01) 


# # Model training
# training for 5000 epochs

# In[ ]:


for epoch in range(5000): 

    pred_y = model(input_arrays.float()) 
   
    loss = criterion(pred_y, target_arrays.float()) 
 
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 
    print('epoch {}, loss {}'.format(epoch, loss.item())) 


# # Predicting values for input values
# Pretty close . Nice :)

# In[ ]:


preds=model(input_arrays.float())
preds


# **Comparing the targets values with predicted values**

# In[ ]:


plt.scatter(target_arrays.detach().numpy(),preds.detach().numpy())


# **Finally model is ready for predicting chance of admmission**

# In[ ]:




