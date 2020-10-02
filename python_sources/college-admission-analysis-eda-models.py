#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install --quiet colored')


# # Importing and Data Loading
# Not using Plotly for this dataset, since it weirdly keeps crashing the notebook. It would be great if anyone can give some potential reasons as to why this is happening.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.autograd import Variable
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from colored import fore, style
plt.style.use('fivethirtyeight')


# In[ ]:


data = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")
data.head()


# In[ ]:


data.info()


# ## Obseravations from the dataset
# 
# * Students in this dataset had a minimum chance of **34%** and a maximum of **97%** of getting selected
# * *75%* of students had less than **82%** chances of getting selected.  
# * *50%* of students in the data had less than **72%** chances of getting selected.
# * On an average a student had **8.5** GPA (out of a maximum 10). In this, 75% of all students have less than 9.0 GPA and 50% of the students have less than 8.5 GPA.
# * Also, the minimum present GPA present is **6.8** and the maximum present is **9.9**
# * SOP, LOR and University Rating Descriptives are fairly straight forward and you can take a look at them yourself below
# * Maximum TOEFL Score is **120** and minimum is **92**. Of this, 75% of all students have less than 112 score in TOEFL and 50% of the students have less than 107 score. On an average the student has 107 score.
# * Maximum GRE Score is **340** and minimum is **290**. Of this, 75% of all students have less than 325 score in GRE and 50% of the students have less than 317 score. On an average the student has 316 score.

# In[ ]:


# Look at the descriptive statistics of the data
data.describe()


# In[ ]:


# Drop useless column for further visualization and also check for any null values
data = data.drop(['Serial No.'], axis=1)
data.isna().sum()


# In[ ]:


# A pairplot visualizes how each variable depends on other variables (If you have no idea what that is, pick a stats book for god's sake)
sns.pairplot(data)


# In[ ]:


# This is a heatmap.
# It shows the correlation between different variables at play
fig = plt.figure(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True)
fig.show()
print("Correlation in a nutshell: ")
print(fore.GREEN+"More Correlation between 2 features => More closely they affect each other and vice-versa"+style.RESET)


# In[ ]:


feature_importance = dict(data.corr()['Chance of Admit '])
sort_orders = sorted(feature_importance.items(), key=lambda x: x[1])
sort_orders.pop()

print(fore.GREEN+f"Most Important feature for getting selected is: {sort_orders[-1][0]}"+style.RESET)
print(fore.RED+f"Least Important feature for getting selected is: {sort_orders[0][0]}"+style.RESET)


# In[ ]:


# Order of Most Important to Least Important Features
print("Following are the features from most important to least important (Darker Blue Shade = More Important) and (Lesser Blue Shade = Less Important)")
i=len(sort_orders)-1
colors = [fore.BLUE_VIOLET, fore.VIOLET, fore.BLUE, fore.GREEN, fore.YELLOW, fore.ORANGE_1, fore.RED][::-1]
while i>=0:
    print(colors[i]+f"{sort_orders[i][0]}"+style.RESET)
    i-=1


# In[ ]:


# Split the data
X = data.drop('Chance of Admit ', axis=1).values
y = data['Chance of Admit '].values
# Just taking 5% of the total data for validation
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.05)


# # Training a Model
# 
# I have tried training the model using different libraries and here is what I observed;
# 
# 1. PyTorch: Made a Custom Linear Regression model and got ~91% accuracy on test set
# 2. Scikit-learn: Just used the pre-built `LogisticRegression` class and got ~92% accuracy on test set
# 
# Conclusion: You can use both, however for much bigger and diverse datasets, I would rather use the pre-built Logistic Regression model from scikit-learn as it is more efficient and does steps such as Data Normalization and better weights initialization.

# In[ ]:


# Define a linear regression model
class LinearRegressionTorch(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionTorch, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out


# In[ ]:


# And Hyperparamters
inp_dim = 7
op_dim = 1
learningRate = 0.001
epochs = 15000


# In[ ]:


# See if CUDA is available
model = LinearRegressionTorch(inp_dim, op_dim)
try:
    model.cuda()
except AssertionError:
    print("GPU isn't enabled")


# In[ ]:


# Define the loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)


# In[ ]:


# Reshape the labels
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)


# In[ ]:


verbose=True
all_loss = []
for epoch in range(epochs+1):
    
    # Convert the data into torch tensors
    inputs = Variable(torch.from_numpy(x_train)).float()
    label = Variable(torch.from_numpy(y_train)).float()
    
    # Clear the existing Gradients from the optimizer
    optimizer.zero_grad()
    
    # Pass the data thorugh model and get one set of predictions
    output = model(inputs)
    
    # Calculate the loss from the obtained predictions and the ground truth values
    loss = torch.sqrt(criterion(output, label))
    
    # Calculate gradients by doing one step of back propagation
    loss.backward()
    
    # Apply those gradients to the weights by doing one step of optimizer
    optimizer.step()
    
    # Add the current loss to the list of all loses (used later for prediction)
    all_loss.append(loss)
    
    # For monitoring and debugging
    if verbose and epoch % 1000 == 0:
        print(f"Epoch: {epoch}  |  Loss: {loss}")


# In[ ]:


# Test the model and compute Validation Accuracy
VAL_inp = Variable(torch.from_numpy(x_val)).float()

y_pred = model(VAL_inp).detach().numpy()

# Calculate R^2 Accuracy (used for regression where discrete values are absent)
rss = sum((y_val - y_pred)**2)       # Residual Sum of Squares
tss = sum((y_val - y_val.mean())**2) # Total Sum of Squares

r2_accuracy = (1 - rss / tss)
print(f"Validation Accuracy of the model is: {r2_accuracy.squeeze() * 100} %")


# In[ ]:


plt.plot(all_loss)


# In[ ]:


# Let's now use sklearn's Linear Regression Model
model = LinearRegression()
model.fit(x_train, y_train)


# In[ ]:


# Measure the model accuracy (the same R^2 Accuracy) using score method
model.score(x_val, y_val)


# In[ ]:


# Save the sklearn model
joblib.dump(model, "sklearn_model_college_adm.sav")


# In[ ]:




