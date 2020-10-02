#!/usr/bin/env python
# coding: utf-8

# ## Disclosure
# This kernel is only my humble attempts to learn new library for me. 
# <br>I am trying to use fully connected classifier and train it only on data without the CV and sentiment analysis first.
# <br>Any suggestions are more than welcome!!!
# 
# ### To DO:
# - Weights initialization???
# - Try to build custom loss function, using the weighted kappa criterion.
# - Scale all the numerical features
# 
# ### Different Versions and Tests:
# V35 - Trying to create regressor rather than predictor.

# In[ ]:


import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
print(os.listdir("../input"))


# In[ ]:


train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


def GetRawData(TestOrTrain):
    source = f'../input/{TestOrTrain}/{TestOrTrain}.csv'
    data =  pd.read_csv(source)
    return data


# In[ ]:


data = GetRawData('train')
len(set(data['Color2']))


# In[ ]:


# Scalers:
maxFee = max(data.Fee.values)
maxVideo = max(data.VideoAmt.values)
maxPhoto = max(data.PhotoAmt.values)
Scalers = [maxFee, maxVideo, maxPhoto]


# In[ ]:


def PrepareFeatures(TestOrTrain, Scalers):
    source = f'../input/{TestOrTrain}/{TestOrTrain}.csv'
    data =  pd.read_csv(source)
    
    data['hasFee'] = data.Fee>0
    data.hasFee[data['hasFee'] == True] = 1
    data.hasFee[data['hasFee'] == False] = 0
    data['hasPhoto'] = data.PhotoAmt>0
    data.hasPhoto[data['hasPhoto'] == True] = 1
    data.hasPhoto[data['hasPhoto'] == False] = 0

    encoder = OneHotEncoder(categories='auto')
    type_feat = encoder.fit_transform(data[['Type']]).toarray()
    type_feat = pd.DataFrame(type_feat, columns=['Dog', 'Cat'])
    gender_feat = encoder.fit_transform(data[['Gender']]).toarray()
    gender_feat = pd.DataFrame(gender_feat, columns=['Male', 'Female', 'Mixed'])
    
    color1_feat = encoder.fit_transform(data[['Color1']]).toarray()
    color1_feat = pd.DataFrame(color1_feat, columns=['1', '2', '3', '4', '5', '6', '7'])
    color2_feat = encoder.fit_transform(data[['Color2']]).toarray()
    color2_feat = pd.DataFrame(color2_feat, columns=['1', '2', '3', '4', '5', '6', '7'])
    size_feat = encoder.fit_transform(data[['MaturitySize']]).toarray()
    size_feat = pd.DataFrame(size_feat, columns=['S', 'M', 'L', 'XL'])
    fur_feat = encoder.fit_transform(data[['FurLength']]).toarray()
    fur_feat = pd.DataFrame(fur_feat, columns=['S', 'M', 'L'])
    
    vacc_feat = encoder.fit_transform(data[['Vaccinated']]).toarray()
    vacc_feat = pd.DataFrame(vacc_feat, columns=['Yes', 'No', 'Unknown'])
    deworm_feat = encoder.fit_transform(data[['Dewormed']]).toarray()
    deworm_feat = pd.DataFrame(deworm_feat, columns=['Yes', 'No', 'Unknown'])
    sterile_feat = encoder.fit_transform(data[['Sterilized']]).toarray()
    sterile_feat = pd.DataFrame(sterile_feat, columns=['Yes', 'No', 'Unknown'])
    health_feat = encoder.fit_transform(data[['Health']]).toarray()
    health_feat = pd.DataFrame(health_feat, columns=['Healthy', 'Minor Injury', 'Serious Injury'])
    
    #le = LabelEncoder()
    #data['StateLabel'] = le.fit_transform(data.State)
    #state_feat = encoder.fit_transform(data[['StateLabel']]).toarray()
    #state_feat = pd.DataFrame(state_feat)
    
    #features = pd.concat([type_feat, gender_feat, color1_feat, color2_feat, 
    #                      size_feat, fur_feat,
    #                      vacc_feat, deworm_feat, sterile_feat, health_feat,
    #                      data.Quantity, data.Age, data.Fee, data.VideoAmt, data.PhotoAmt], axis=1)
    
    features = pd.concat([type_feat, gender_feat,
                          size_feat, fur_feat,
                          vacc_feat, deworm_feat, sterile_feat, health_feat,
                          data.hasFee, data.hasPhoto], axis=1)
        
    features = features.values.astype(np.float32)
    if TestOrTrain=='train':
        label = data.AdoptionSpeed.values.astype(np.float32)
    else:
        label = np.zeros(features.shape[0])
    pet_ids = data.PetID
    return features, label, pet_ids

'''
features = ["Type", "Age", "Breed1", "Breed2", "Gender",
            "Color1", "Color2", "Color3", "MaturitySize", "FurLength",
            "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "Fee", "State",
            "VideoAmt", "PhotoAmt"]
'''
features = ["Type", "Age", "Gender",
            "MaturitySize", "FurLength",
            "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "Fee",
            "VideoAmt", "PhotoAmt"]


# In[ ]:


x, y, train_pet_ids = PrepareFeatures('train', Scalers)
raw_data = GetRawData('train')


# In[ ]:


raw_data.head(2)


# In[ ]:


label_std = raw_data.AdoptionSpeed.std(axis=0)
label_mean = raw_data.AdoptionSpeed.mean(axis=0)
print("Labels: mean={} and std={}".format(label_mean, label_std))


# In[ ]:


mean_label_dog_male = np.mean(raw_data.AdoptionSpeed[(raw_data['Type']==1) & (raw_data['Gender']==1)].values)
mean_label_dog_female = np.mean(raw_data.AdoptionSpeed[(raw_data['Type']==1) & (raw_data['Gender']==2)].values)

mean_label_cat_male = np.mean(raw_data.AdoptionSpeed[(raw_data['Type']==2) & (raw_data['Gender']==1)].values)
mean_label_cat_female = np.mean(raw_data.AdoptionSpeed[(raw_data['Type']==2) & (raw_data['Gender']==2)].values)

print(f'Mean adoption speed for dogs: Male={mean_label_dog_male:.2f}, Female={mean_label_dog_female:.2f}')
print(f'Mean adoption speed for cats: Male={mean_label_cat_male:.2f}, Female={mean_label_cat_female:.2f}')


# In[ ]:


# Check the Correctness of encoding and Pet IDs associations:
for i in range(3):
    cur_type = raw_data.Type[raw_data['PetID']==train_pet_ids[i]].values[0]
    cur_gender = raw_data.Gender[raw_data['PetID']==train_pet_ids[i]].values[0]
    cur_label = raw_data.AdoptionSpeed[raw_data['PetID']==train_pet_ids[i]].values[0]
    print(f"Encoded features: {x[i]} for label {y[i]}={cur_label}, pet id = {train_pet_ids[i]}: Type={cur_type}, Gender={cur_gender}")


# In[ ]:


num_entries = x.shape[0]
split_idx = int(num_entries*0.8)
train_x, val_x = x[:split_idx], x[split_idx:]
train_y, val_y = y[:split_idx], y[split_idx:]
test_x, test_y, test_pet_ids = PrepareFeatures('test', Scalers)

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))


# In[ ]:


# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 50

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


# ## Define Regression Model:

# In[ ]:


class ComplexRegressor3(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_hidden3, n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_hidden3)
        self.fc4 = nn.Linear(n_hidden3, n_output)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# ## Create the Classifier Model:

# In[ ]:


n_inputs = x.shape[1]
n_outputs = 1
print(f"Model should have {n_inputs} inputs and {n_outputs} outputs")


# In[ ]:


model = ComplexRegressor3(n_inputs, 128, 128, 128, n_outputs)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)


# ## Training the model:

# In[ ]:


n_epochs = 200
print_every = 2

min_val_loss = 1000

for epoch in range(n_epochs):
    running_loss = 0.0
    batch_i = 0
    
    # Train for one epoch:
    for inputs, labels in train_loader:
        batch_i += 1
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Test the model on validation set:
    model.eval()
    with torch.no_grad():
        test_loss, accuracy  = 0, 0       
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            batch_loss = criterion(outputs, labels)                  
            test_loss += batch_loss.item()
            
            # Calculate accuracy
    model.train()
    
    cur_train_loss = running_loss/len(train_loader)
    cur_val_loss = test_loss/len(valid_loader)
    
    if (epoch%print_every==0) | (epoch ==n_epochs-1):
        print(f"Epoch {epoch+1}/{n_epochs}")   
        print(f"Train Loss: {cur_train_loss:.4f}.."       
              f"Val Loss: {cur_val_loss:.4f}.. "
              f"Val accuracy: {accuracy/val_x.shape[0]:.4f}")
        
    if cur_val_loss < min_val_loss:
        print(f"Epoch {epoch+1}/{n_epochs}: Validation loss decreased from {min_val_loss:.4f} to {cur_val_loss:.4f}")
        state_dict = model.state_dict()
        min_val_loss = cur_val_loss


# In[ ]:


model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
model_dict.update(state_dict) 
model.load_state_dict(state_dict)
model.to(device);


# In[ ]:


predicted_labels = []
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    pred = model.forward(inputs)
    pred = torch.round(pred)
    xxx = [pred[0] for pred in pred.data.cpu().numpy().astype('int32').tolist()]
    predicted_labels.extend(xxx)
    print(f"Classes predicted in current batch: {set(xxx)}")


# In[ ]:


submission = pd.DataFrame({'PetID': test_pet_ids, 'AdoptionSpeed': predicted_labels})
submission.to_csv('submission.csv', index = False)

