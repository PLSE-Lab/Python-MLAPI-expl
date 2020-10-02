#!/usr/bin/env python
# coding: utf-8

# I'm sure you're all aware of the Titanic (especially since you're here), so I'll spare you much of an introduction in that regards. Anyways, this notebook uses ensemble learning with a random forest, k-nearest neighbors, linear regression, and an artificial neural network. The first three are from scikit-learn while the ANN is from PyTorch. Some other things that are used are a grid search, simple imputers, and some manual one-hot encoding (mentioned for people quickly looking for an example of them). 
# 
# This notebook got 79.425% for a top 20% placement here on Kaggle. 
# 
# If you're looking to learn something from this then I hope you do! If you have any questions or comments, I'd love to hear them. 

# Uses: scikit-learn, pandas, NumPy, PyTorch

# # 1 - Explore (A Little)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Not used as of now


# In[ ]:


train_df = pd.read_csv("../input/titanic/train.csv")
test_df = pd.read_csv("../input/titanic/test.csv")


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# Looking at the data initially, it looks like there's a few columns that I won't try to mess with. First is the name (which would be a place to come back to and grab the titles from it). There's also the passenger id which won't have any real correlation, with the ticket probably being in a similar position. Then there's the cabin, which is just missing from so many entries. Everything else we should be able to use, somehow. 

# In[ ]:


corr = train_df.corr()
corr


# Looking at the 'Survived' column, Pclass and Fare have the strongest correlations (negatively and positively, respectively). That doesn't mean everything else won't be useful, though. 

# # 2 - Data Preprocessing

# In[ ]:


# Impute, remove, encode, scale
# (Only scale for ANN (for now))


# In[ ]:


# Age, cabin, & embarked are incomplete (in the training set)
# Age, cabin, & fare are incomplete (in the test set)

# age and fare can be imputed using the mean
# embarked - most common (only missing 2)
# cabin can probably be safely dropped due to how many are missing


# In[ ]:


train_labels = train_df['Survived'] # get separate labels array
X_train = train_df.drop(['Survived'], axis=1) # drop labels out of df


# In[ ]:


X_train # Missing 'Survived' now, good


# # - Imputing

# Time to use two SimpleImputers from skklearn.impute to fill some missing data. 

# In[ ]:


# Import
from sklearn.impute import SimpleImputer


# In[ ]:


# Imputer for age and fare
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Imputer for embarked
common_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent') # decent guess as any


# In[ ]:


mean_imputer.fit(X_train[['Age', 'Fare']])
common_imputer.fit(X_train[['Embarked']])


# In[ ]:


X_train[['Age', 'Fare']] = mean_imputer.transform(X_train[['Age', 'Fare']]);
test_df[['Age', 'Fare']] = mean_imputer.transform(test_df[['Age', 'Fare']]);

X_train[['Embarked']] = common_imputer.transform(X_train[['Embarked']]);
test_df[['Embarked']] = common_imputer.transform(test_df[['Embarked']]);


# In[ ]:


X_train.info()


# # - Remove Columns

# In[ ]:


col_to_drop = ['Name', 'Ticket', 'Cabin']


# In[ ]:


X_train = X_train.drop(col_to_drop, axis=1)
test_df = test_df.drop(col_to_drop, axis=1)


# In[ ]:


X_train.info() #; test_df.info() # easier to comment out


# Looks good, no missing data points now. Still a few objects to deal with, though. 

# In[ ]:


X_train.head()


# In[ ]:


# Need to encode sex & embarked
# Drop Id


#  # - Feature Engineering

# In[ ]:


X_train['AgeRange'] = ['child' if x < 13 else 'teenager' if x < 22 
                       else 'adult' if x < 66 else 'senior' for x in X_train['Age']]
test_df['AgeRange'] = ['child' if x < 13 else 'teenager' if x < 22 
                       else 'adult' if x < 66 else 'senior' for x in test_df['Age']]


# Not the prettiest list comprehensions ever, but they get the job done. Tweaking the ages and number of categories here could potentially help performance a little bit. 

# In[ ]:


X_train.head()


# In[ ]:


# Make a new column that contains the # of family members aboard
X_train['FamNum'] = X_train['SibSp'] + X_train['Parch']
test_df['FamNum'] = test_df['SibSp'] + test_df['Parch']

# Remove SibSp and Parch as they were just used
# Remove Age as we have AgeRange now
# Remove PassengerId as it doesn't really tell us anything
X_train = X_train.drop(['SibSp', 'Parch', 'PassengerId', 'Age'], axis=1)
test_df = test_df.drop(['SibSp', 'Parch', 'PassengerId', 'Age'], axis=1)


# In[ ]:


# A quick idea of what is strongly correlated with Survived
pd.concat([X_train, train_labels], axis=1).corr()['Survived']


# In[ ]:


# Nothing super crazy, but Pclass is nice to see


# # - Encode

# Instead of using scikit-learn's LabelEncoder and OneHotEncoder classes, I'm going to use list comprehensions to encode the three columns mentioned in the next cell. It's a nice reminder that list comprehensions are a thing in Python and they are quite handy (even if doing them for all of that takes up more lines than using sci-kit-learn). 

# In[ ]:


need_encoding = ['Sex', 'Embarked', 'AgeRange'] # Marked for later removal


# In[ ]:


# Manual one-hot encoding on train set using list comprehensions
# Gender
X_train['Male'] = [1 if x == 'male' else 0 for x in X_train['Sex']]
X_train['Female'] = [1 if x == 'female' else 0 for x in X_train['Sex']]

# Embarked
X_train['S'] = [1 if x == 'S' else 0 for x in X_train['Embarked']]
X_train['C'] = [1 if x == 'C' else 0 for x in X_train['Embarked']]
X_train['Q'] = [1 if x == 'Q' else 0 for x in X_train['Embarked']]

# AgeRange
X_train['child'] = [1 if x == 'child' else 0 for x in X_train['AgeRange']]
X_train['teenager'] = [1 if x == 'teenager' else 0 for x in X_train['AgeRange']]
X_train['adult'] = [1 if x == 'adult' else 0 for x in X_train['AgeRange']]
X_train['senior'] = [1 if x == 'senior' else 0 for x in X_train['AgeRange']]


# In[ ]:


# Manual one-hot encoding on test set using list comprehensions
# Gender
test_df['Male'] = [1 if x == 'male' else 0 for x in test_df['Sex']]
test_df['Female'] = [1 if x == 'female' else 0 for x in test_df['Sex']]

# Embarked
test_df['S'] = [1 if x == 'S' else 0 for x in test_df['Embarked']]
test_df['C'] = [1 if x == 'C' else 0 for x in test_df['Embarked']]
test_df['Q'] = [1 if x == 'Q' else 0 for x in test_df['Embarked']]

# AgeRange
test_df['child'] = [1 if x == 'child' else 0 for x in test_df['AgeRange']]
test_df['teenager'] = [1 if x == 'teenager' else 0 for x in test_df['AgeRange']]
test_df['adult'] = [1 if x == 'adult' else 0 for x in test_df['AgeRange']]
test_df['senior'] = [1 if x == 'senior' else 0 for x in test_df['AgeRange']]


# In[ ]:


# Can safely drop the original columns now out of both
X_train = X_train.drop(need_encoding, axis=1)
test_df = test_df.drop(need_encoding, axis=1)


# In[ ]:


# Let's see what it looks like now
X_train.head()


# # 3 - Train the Models

# # Random Forest Classifier

# The first classifier up is a Random Forest. It'll go through a grid search to find a good set of hyperparameters and then we'll grab that best estimator for later use. 

# Training set: X_train && train_labels

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier()


# In[ ]:


# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = [{'n_estimators': [100], 
               'criterion': ['gini'],
               'min_samples_split': [2, 3],
               'min_samples_leaf': [1, 2],
               'max_features': ['auto', None],
               'max_depth': [None, 8, 10, 12]
              }]
# Get it ready
grid_search = GridSearchCV(rf_classifier, param_grid, cv=7, verbose=1)


# Quick Note: I did have more options for some of those parameters, but I removed them afterwards so it wouldn't take so long to train. Rest assured I removed the ones that never got picked. 

# In[ ]:


grid_search.fit(X_train, train_labels) # Train it (shouldn't take too long)


# In[ ]:


grid_search.best_params_ 
# Usually {gini, 8, None, 2, 2, 100}


# In[ ]:


grid_search.best_score_ 
# ~ 83.28%


# In[ ]:


# Save the best random forest
rf_classifier = grid_search.best_estimator_


# # K-NN

# After the Random Forest, it's basically the same thing except with a different classifier and different parameters for the grid search. 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier()


# In[ ]:


# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = [{'n_neighbors': [2, 3, 4], 
               'algorithm': ['ball_tree', 'kd_tree'],
               'leaf_size': [10, 20, 30]
              }]
# Get it ready
grid_search = GridSearchCV(knn_classifier, param_grid, cv=7, verbose=1)


# In[ ]:


grid_search.fit(X_train, train_labels) # Fit it


# In[ ]:


grid_search.best_params_ 
# ball_tree, 10, 3 last time


# In[ ]:


grid_search.best_score_ 
# ~ 80.36%


# In[ ]:


# Save the best K-NN
knn_classifier = grid_search.best_estimator_


# # ANN Classifier

# Definitely the hardest one here to implement, but it is quite a nice classifier. 

# In[ ]:


# This one is a bit messier


# In[ ]:


# Quick function that gets how many out of 'preds' match 'labels'
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# In[ ]:


# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# In[ ]:


# Scale the data for the ANN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# In[ ]:


# Scale the data
X_train_scaled = scaler.fit_transform(X_train)
test_ann = scaler.transform(test_df) # To be used later


# In[ ]:


# Want a train and val set now
from sklearn.model_selection import train_test_split

# Split the train set into train & val 
# Could try again w/o splitting the train set w/ a set NN architecture
X_train_part, X_val, y_train, y_val = train_test_split(X_train_scaled, train_labels, test_size=0.10)


# In[ ]:


X_train_part.shape


# In[ ]:


# Our Artificial Neural Network class
# Could play around with this a lot more
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
      
        self.fc1 = nn.Linear(in_features=12, out_features=32) # linear 1
        self.fc1_bn = nn.BatchNorm1d(num_features=32)
        self.drop1 = nn.Dropout(.1)
        
        self.fc2 = nn.Linear(in_features=32, out_features=8) # linear 2
        self.fc2_bn = nn.BatchNorm1d(num_features=8)
        
        self.out = nn.Linear(in_features=8, out_features=2) # output
    
    def forward(self, t):
        t = F.relu(self.fc1_bn(self.fc1(t)))
        t = self.drop1(t)
        t = F.relu(self.fc2_bn(self.fc2(t)))
        
        t = self.out(t)
        return t


# In[ ]:


# Gets the GPU if there is one, otherwise the cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:


batch_size = 64 # got a weird error if there was a different batch size

# Some weird TensorDataset stuff
# It wants tensors of type float, but they were numpy arrays of not all float
train_set = TensorDataset(torch.from_numpy(X_train_part.astype(float)), 
                          torch.from_numpy(y_train.as_matrix().astype(float)))
val_set = TensorDataset(torch.from_numpy(X_val.astype(float)), 
                        torch.from_numpy(y_val.as_matrix().astype(float)))

# Load up the data and shuffle away
train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=True)


# In[ ]:


lr = 0.001 # initial learning rate
epochs = 200 # number of epochs to run

network = ANN().float().to(device) # put the model on device (hopefully a GPU!)
optimizer = optim.Adam(network.parameters(), lr=lr) # Could try a different optimizer

# It wanted the X to be float and the y to be long, so I complied
for epoch in range(epochs):
    network.train() # training mode
    
    # decrease the learning rate a bit
    if epoch == 40:
        optimizer = optim.Adam(network.parameters(), lr=0.0001)
    
    # decrease the learning rate a bit more
    if epoch == 80:
        optimizer = optim.Adam(network.parameters(), lr=0.00000000001)
        
    for features, labels in train_dl:
        X, y = features.to(device), labels.to(device) # put X & y on device
        y_ = network(X.float()) # get predictions
        
        optimizer.zero_grad() # zeros out the gradients
        loss = F.cross_entropy(y_, y.long()) # computes the loss
        loss.backward() # computes the gradients
        optimizer.step() # updates weights
          
    # Evaluation with the validation set
    network.eval() # eval mode
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for features, labels in val_dl:
            X, y = features.to(device), labels.to(device) # to device
            
            preds = network(X.float()) # get predictions
            loss = F.cross_entropy(preds, y.long()) # calculate the loss
            
            val_correct += get_num_correct(preds, y.long())
            val_loss = loss.item() * batch_size
            
    # Print the loss and accuracy for the validation set
    if (epoch % 10) == 9: # prints every 10th epoch
        print("Epoch: ", epoch+1)
        print(" Val Loss: ", val_loss)
        print(" Val Acc: ", (val_correct/len(X_val))*100)

# Output can get a bit long


# # Linear Regression

# A nice and quick Linear Regression model, just using a few less columns 

# In[ ]:


# Helps a bit to reduce dimensions for linear regression (or so I remember)
X_lin = X_train.drop(['S', 'C', 'Q', 'teenager', 'adult'], axis=1)
test_lin = test_df.drop(['S', 'C', 'Q', 'teenager', 'adult'], axis=1) # need to use later


# In[ ]:


from sklearn.linear_model import LinearRegression

linearreg = LinearRegression(normalize=True, copy_X=True)


# In[ ]:


linearreg.fit(X_lin, train_labels)


# # Ensemble

# In[ ]:


# rf_classifier, network, knn_classifier, linearreg
# test_df = test set


# In[ ]:


# Get the test set ready for the ANN because it's special
test = TensorDataset(torch.from_numpy(test_ann.astype(float)) )
test_tensor = DataLoader(test, batch_size=batch_size, shuffle=False)


# In[ ]:


# Get predictions

# Predictions for the Random Forest ---------------
rf_preds = rf_classifier.predict(test_df)

# Predictions for the ANN -------------------------
ann_preds = torch.LongTensor().to(device) # Tensor for all predictions
network.eval() # safety
for batch in test_tensor:
    batch = batch[0].to(device) # just batch is a [tensor] for some reason
    predictions = network(batch.float()) # again with the float thing
    ann_preds = torch.cat((ann_preds, predictions.argmax(dim=1)), dim=0) 
# bring it back to the cpu and convert to an array
ann_preds = ann_preds.cpu().numpy()

# Predictions for the K-Nearest Neighbors ---------
knn_preds = knn_classifier.predict(test_df)

# Predictions for the Linear Regression -----------
lin_preds = linearreg.predict(test_lin) # special test set with less columns


# In[ ]:


lin_preds[:5] # Not quite how we want them


# In[ ]:


lin_preds = np.around(lin_preds, decimals=0).astype(int) # Rounds them


# In[ ]:


# Interesting to see
print(np.sum(rf_preds==ann_preds), "/", rf_preds.shape[0], " same predictions between Random Forest and ANN")
print(np.sum(rf_preds==knn_preds), "/", rf_preds.shape[0], " same predictions between Random Forest and K-NN")
print(np.sum(rf_preds==lin_preds), "/", rf_preds.shape[0], " same predictions between Random Forest and Linear Reg")
print(np.sum(ann_preds==knn_preds), "/", rf_preds.shape[0], " same predictions between ANN and K-NN")
print(np.sum(ann_preds==lin_preds), "/", rf_preds.shape[0], " same predictions between ANN and Linear Reg")
print(np.sum(knn_preds==lin_preds), "/", rf_preds.shape[0], " same predictions between K-NN and Linear Reg")


# In[ ]:


# Add them all up
agg_preds = rf_preds + ann_preds + lin_preds + knn_preds
agg_preds # values 0-4 now


# In[ ]:


values, counts = np.unique(agg_preds, return_counts=True) # sum number of 0s..., 4s
for i in range(5):
    print(values[i], " classifiers predicted 'Survive'", counts[i], " times", )


# In[ ]:


# Time to get the final predictions
final_preds = np.empty(len(agg_preds), dtype=int) # empty predictions array

# Survived if agg_preds has 4 or 3
# Didn't survive if agg_preds has 0 or 1
# Up to the Random Forest if agg_preds is split at 2
for i in range(len(agg_preds)): # go through agg_preds
    if agg_preds[i] < 2:
        final_preds[i] = 0
    elif agg_preds[i] > 2:
        final_preds[i] = 1
    else: # final call goes to random forest
        final_preds[i] = rf_preds[i]


# I left it up to the Random Forest classifier because it was one of the strongest classifiers when I tested them individually. Since then, the ANN might have closed the gap, but I imagine if anything they'd be able the same. I think the best way to settle this would probably be to introduce a fifth classifier. 

# In[ ]:


final_preds # Beautiful!


# # Write To CSV

# In[ ]:


# Read in sample csv
sample_df = pd.read_csv("../input/titanic/gender_submission.csv")


# In[ ]:


# Edit it
sample_df['Survived'] = final_preds


# In[ ]:


# Write to a new csv
sample_df.to_csv("predictions.csv", index=False) # Be sure to not include the index


# # Some Quick Things I Learned

# - This served as a nice reminder that ANNs perform better when inputs are scaled
# - The K-NN did much better after the categorical variables were one-hot encoded
# - Ensemble learning is quite nice, even for a small task like this (and the things I had to do in pandas for this were a nice refresher from way back/ learning experience)
# - List comprehensions in Python work quite nicely for quick, manual one-hot encoding
# - Also PyTorch can be really picky about a few things, like data type or batch size

# # A Few Final Remarks

# These results got 79.425% for a top 20% placement. Not too bad for what was done here. 

# In[ ]:


# 79.425 on test set (Kaggle) (2164th place when submitted)
# Top 20%


# Some possible future notebook improvements...
# - More visuals
# - Try and add more useful classifiers
# - Grab some info from the name / some more feature engineering
# - Mess with the ANN more

# Thanks again for reading and I hope you learned something! 
