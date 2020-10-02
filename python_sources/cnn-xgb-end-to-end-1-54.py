#!/usr/bin/env python
# coding: utf-8

# # Introduction
# ![CNN Schema](https://raw.githubusercontent.com/jxtrbtk/kaggle/master/aptos2019-blindness-detection/CNNSchema.png)
# 
# This kernel is a playground around the following idea. CNN are made of two part, convultions that acts like a features extractor and a fully connected neural network used for classification. What if we replace the neural network classifier by an XGBoost based one ? XGBoost is most often very efficient and makes it easy to achieve very good performances without a lot of tuning effort.
# 
# At the very beginning, I had better results with XGB, but the more optimized the process, the more similar the performance were...
# Hope you'll enjoy the journey !

# # Imports, settings and references

# In[ ]:


import os
import pandas as pd
import numpy as np 
import time

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision import transforms

import xgboost as xgb
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import quantile_transform

import pickle

DEVICE = torch.device("cuda:0")
DATA_SOURCE = os.path.join("..","input","aptos2019-blindness-detection")
MODEL_SOURCE = os.path.join("..","input","densenet161-1-18-v2-pth")
MODEL_SIZE = 224


# # Pre-processing

# In[ ]:


def crop_image(img,tol=7):
    w, h = img.shape[1],img.shape[0]
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.blur(gray_img,(5,5))
    shape = gray_img.shape 
    gray_img = gray_img.reshape(-1,1)
    quant = quantile_transform(gray_img, n_quantiles=256, random_state=0, copy=True)
    quant = (quant*256).astype(int)
    gray_img = quant.reshape(shape)
    xp = (gray_img.mean(axis=0)>tol)
    yp = (gray_img.mean(axis=1)>tol)
    x1, x2 = np.argmax(xp), w-np.argmax(np.flip(xp))
    y1, y2 = np.argmax(yp), h-np.argmax(np.flip(yp))
    if x1 >= x2 or y1 >= y2 : # something wrong with the crop
        return img # return original image
    else:
        img1=img[y1:y2,x1:x2,0]
        img2=img[y1:y2,x1:x2,1]
        img3=img[y1:y2,x1:x2,2]
        img = np.stack([img1,img2,img3],axis=-1)
    return img

def process_image(image, size=512):
    image = cv2.resize(image, (size,int(size*image.shape[0]/image.shape[1])))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        image = crop_image(image, tol=15)
    except Exception as e:
        image = image
        print( str(e) )
    return image


# *inspired by: https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping*

# # PyTorch's style data loader defintion

# In[ ]:


class RetinopathyDataset(Dataset):

    def __init__(self, transform, is_test=False):
        self.transform = transform
        self.base_transform = transforms.Resize((MODEL_SIZE, MODEL_SIZE))
        self.is_test = is_test 
        if not os.path.exists("cache"): os.mkdir("cache")
        if is_test : file = "test.csv"
        else : file = "train.csv"
        csv_file = os.path.join(DATA_SOURCE, file)
        df = pd.read_csv(csv_file)
        self.data = df.reset_index(drop=True)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_test : archive = "test_images"
        else : archive = "train_images"
        folder = os.path.join(DATA_SOURCE, archive)
        code = str(self.data.loc[idx, 'id_code'])
        file = code + ".png"
        cache_path = os.path.join("cache",code+".png")
        cached = os.path.exists(cache_path)
        if not cached : 
            path = os.path.join(folder, file)
            image = cv2.imread(path)
            image = process_image(image)
            imgpil = Image.fromarray(image)
            imgpil = self.base_transform(imgpil)
            imgpil.save(cache_path,"PNG")
        imgpil = Image.open(cache_path)
        img_tensor = self.transform(imgpil)
        if self.is_test : return {'image': img_tensor} 
        else : 
            label = self.data.loc[idx, "diagnosis"]
            return {'image': img_tensor, 'label': label}
        

    def get_df(self):
        return self.data


# *adapted from: https://www.kaggle.com/abhishek/very-simple-pytorch-training-0-59*

# # Re-train the pre-trained model

# In[ ]:


# first we will prepare the dataset and create the folds
NUM_FOLDS = 5

data_augmentation = transforms.Compose([
    transforms.RandomRotation((-15, 15)),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

DATA = RetinopathyDataset(data_augmentation)
df = DATA.get_df()
skf = StratifiedKFold(n_splits=NUM_FOLDS)
folds_generator = skf.split(df.index.values, df.diagnosis.values)
data_train, data_eval = [], [] 
for t, e in folds_generator:
    data_train.append(t)
    data_eval.append(e)


# In[ ]:


def get_dataloader_for_fold(n, data, train_data, eval_data, batch_size):
    """ return the train and eval dataloader for a fold
    """    
    train_sampler = SubsetRandomSampler(train_data[n])
    valid_sampler = SubsetRandomSampler(eval_data[n])

    data_loader_train = torch.utils.data.DataLoader(data, 
                    batch_size=batch_size, drop_last=False, 
                    sampler=train_sampler)
    data_loader_eval = torch.utils.data.DataLoader(data, 
                    batch_size=batch_size, drop_last=False, 
                    sampler=valid_sampler)
    
    return data_loader_train, data_loader_eval


# In[ ]:


class Classificator0(nn.Module):
    """ classifier layer used to retrain the CNN
    """    
    def __init__(self, size=128):
        super(Classificator0, self).__init__()
        self.size = size
        self.network = nn.Sequential(
              nn.BatchNorm1d(size),
              nn.Dropout(p=0.3),
              nn.Linear(in_features=size, out_features=5, bias=True),
        )        
    def forward(self, x):
        ## Define forward behavior
        return self.network(x)

class Classificator(nn.Module):
    """ classifier layer used to retrain the CNN
    """    
    def __init__(self, size=128):
        super(Classificator, self).__init__()
        self.size = size
        self.network = nn.Sequential(
              nn.BatchNorm1d(size),
              nn.Dropout(p=0.25),
              nn.Linear(in_features=size, out_features=size, bias=True),
              nn.ReLU(),
              nn.BatchNorm1d(size),
              nn.Dropout(p=0.5),
              nn.Linear(in_features=size, out_features=5, bias=True),
    )        
    def forward(self, x):
        ## Define forward behavior
        return self.network(x)


# In[ ]:


def get_base_model():
    """ get the pretrained model
    """    
    model = torchvision.models.densenet161(pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = Classificator0(in_features)
    model_path = os.path.join(MODEL_SOURCE, "densenet161.1.18.v2.pth")
    model.load_state_dict(torch.load(model_path))
    model.classifier = Classificator(in_features)
    model = model.to(DEVICE)
    model.eval()
    
    return model


# In[ ]:


def train_model(model, optimizer, scheduler, train_data_loader, eval_data_loader, 
                file_name, num_epochs = 50, patience = 7, prev_loss = 1000.00):
    """ train the model
    arguments : model, optimizer, scheduler, train_data_loader, eval_data_loader
        file_name: name of the file to save the best model 
        num_epochs: maximum number of epochs
        patience: number of epochs to wait if no improvements
        prev_loss: previous loss achieved, to surpass to have the model saved
    return: 
        best loss achieved (previous loss if not surpassed)
    """    
    criterion = nn.CrossEntropyLoss()
    countdown = patience
    best_loss = 1000.00
    since = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        counter = 0
        for bi, d in enumerate(train_data_loader):
            inputs = d["image"].to(DEVICE, dtype=torch.float)
            labels = d["label"].to(DEVICE, dtype=torch.long)
            # batch norm layers needs more than 1 set of data
            # this is to skip the last batch if it's only 1 image
            if inputs.shape[0] > 1 :
                counter += inputs.size(0)
                model.to(DEVICE)
                model.train()
                optimizer.zero_grad()
                outputs = model(inputs) 
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                loss_val = running_loss / counter
                print("{:7} {:.4f} {:.4f}".format(counter, loss.item()*1, loss_val), end="\r")
        epoch_loss = running_loss / ( len(train_data_loader) * train_data_loader.batch_size)
        time_elapsed = time.time() - since
        print(" T{:3}/{:3} loss: {:.4f} ({:3.0f}m {:2.0f}s)".format( 
            epoch, num_epochs - 1, epoch_loss,time_elapsed // 60, time_elapsed % 60))
        running_loss = 0.0
        counter = 0
        for bi, d in enumerate(eval_data_loader):
            inputs = d["image"].to(DEVICE, dtype=torch.float)
            counter += inputs.size(0)
            labels = d["label"].to(DEVICE, dtype=torch.long)
            model.to(DEVICE)
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            loss_val = running_loss / counter
            print("{:7} {:.4f} {:.4f}".format(counter, loss.item()*1, loss_val), end="\r")
        epoch_loss = running_loss / ( len(eval_data_loader) * eval_data_loader.batch_size)
        if epoch_loss < best_loss : 
            best_loss = epoch_loss
            if epoch_loss < prev_loss:
                torch.save(model.state_dict(), file_name)
                prev_loss = epoch_loss
                print("*", end="")
            else:
                print(".", end="")
            countdown = patience
        else:
            print("{:1}".format(countdown), end="")
            countdown -= 1
        time_elapsed = time.time() - since
        print("E{:3}/{:3} loss: {:.4f} ({:3.0f}m {:2.0f}s)".format( 
            epoch, num_epochs - 1, epoch_loss,time_elapsed // 60, time_elapsed % 60 ))
        scheduler.step() #epoch_loss

        if countdown <= 0 : break

    return prev_loss
    print("done.")
# Model training


# In[ ]:


# train the model num_round_per_fold times for each fold
# and the save the best model for each fold
batch_size = 56
num_round_per_fold = 2
for no in range(NUM_FOLDS):
    print("-"*22, "fold",no)
    bst_loss = 10000.00
    for r in range(num_round_per_fold):
        print("-"*11,"round",r)
        data_loader_train, data_loader_eval = get_dataloader_for_fold(no, 
                                    DATA, data_train, data_eval, batch_size)
        model = get_base_model()
        plist = [{"params": model.features.denseblock4.parameters(), "lr":0.0001},
                 {"params": model.classifier.parameters()}]
        optimizer = optim.Adam(plist, lr=0.001, amsgrad=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1,5], gamma=0.1, last_epoch=-1)        
        bst_loss = train_model(model, optimizer, scheduler, 
                               data_loader_train, data_loader_eval, 
                               "tmp"+str(no)+".pth", prev_loss=bst_loss, 
                               num_epochs=13, patience=3)
    print("-"*22, "best loss", bst_loss)
    print("")


# # Extract train features from CNN

# In[ ]:


def get_trained_model(no): 
    """ reload and return the retrained model for the given fold 
    """    
    extractor = torchvision.models.densenet161(pretrained=False)
    in_features = extractor.classifier.in_features
    extractor.classifier = Classificator(in_features)
    model_path = os.path.join("tmp"+str(no)+".pth") #no
    extractor.load_state_dict(torch.load(model_path))
    extractor = extractor.to(DEVICE)
    extractor.eval()
    return extractor


# In[ ]:


def get_extractor_model(no):
    """ reload and return the retrained model for the given fold
        and make last layer identity
    """    
    extractor = get_trained_model(no)
    extractor.classifier = nn.Identity()
    extractor = extractor.to(DEVICE)
    extractor.eval()
    return extractor


# In[ ]:


def get_train_features(data_loader, extractor):
    """ return 2 arrays of features extracted, and targets  
    """    
    for bi, d in enumerate(data_loader):
        print(".", end="")
        img_tensor = d["image"].to(DEVICE)
        target = d["label"].numpy()
        with torch.no_grad(): feature = extractor(img_tensor)
        feature = feature.cpu().detach().squeeze(0).numpy()
        if bi == 0 :
            features = feature 
            targets = target 
        else :
            features = np.concatenate([features, feature], axis=0)
            targets = np.concatenate([targets, target], axis=0)

    return features, targets


# # Fit the XGBoost model

# In[ ]:


XGBOOST_PARAM = {
    "random_state"      : 42,
    "n_estimators"      : 200,
    "objective"         : "multi:softmax",
    "num_class"         : 5,
    "eval_metric"       : "mlogloss",
}


# In[ ]:


# for each fold, get the data loader, extractor, 
# extract the features (loaded and process each time, but we can have data aug.) 
# calcul the weights table, create and fit the XGB model
batch_size = 64
eval_set = []
for no in range(NUM_FOLDS):
    print("-"*22, "fold",no)
    data_loader_train, data_loader_eval = get_dataloader_for_fold(no, 
                                DATA, data_train, data_eval, batch_size)
    extractor = get_extractor_model(no)

    print("...........|.............................................|")
    features_eval, targets_eval = get_train_features(data_loader_eval,
                                                     extractor)
    features_train, targets_train = get_train_features(data_loader_train,
                                                       extractor)
    print("")

    xgb_model = xgb.XGBClassifier(**XGBOOST_PARAM)
    xgb_model = xgb_model.fit(features_train,targets_train.reshape(-1),
                        eval_set=[(features_eval, targets_eval.reshape(-1))],
                        early_stopping_rounds=20,
                        verbose=False)
    print("score",xgb_model.evals_result()["validation_0"]["mlogloss"][-1])
    pickle.dump(xgb_model, open("xgb_model_"+str(no), "wb"))


# # Cross-Validation
# ## .using the full CNN models

# In[ ]:


# change the data augmentation of the dataset object
base_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
DATA.transform = base_transform


# In[ ]:


# first we make predictions using the CNN models for each fold 
# and calculate the score of all predictions
predictions, targets = np.zeros(len(DATA)), np.zeros(len(DATA))
batch_slice = (0, 0)
softmax = nn.Softmax(dim=1)
for no in range(NUM_FOLDS):
    _, data_loader_eval = get_dataloader_for_fold(no, 
                DATA, data_train, data_eval, 64)
    model = get_trained_model(no) #no
    for bi, d in enumerate(data_loader_eval):
        inputs = d["image"].to(DEVICE, dtype=torch.float)
        batch_slice = (batch_slice[1], batch_slice[1]+inputs.size(0))
        with torch.no_grad():
            outputs = model(inputs)
            outputs = softmax(outputs)
        predictions[batch_slice[0]:batch_slice[1]] =                 outputs.cpu().detach().squeeze(0).numpy().argmax(axis=1)
        targets[batch_slice[0]:batch_slice[1]] = d["label"]
        
print("Cohen Kappa quadratic score", 
      cohen_kappa_score(targets, predictions, weights="quadratic"))


# ## .using the XGB models

# In[ ]:


# now let's do it with the XGB models 
predictions, targets = np.zeros(0), np.zeros(0)
for no in range(NUM_FOLDS):
    _, data_loader_eval = get_dataloader_for_fold(no, 
                                DATA, data_train, data_eval, batch_size)
    features_eval, targets_eval = get_train_features(data_loader_eval,
                                                     extractor)
    print("")
    xgb_model = xgb.XGBClassifier()
    model_path = os.path.join("xgb_model_"+str(no))
    xgb_model = pickle.load(open(model_path, "rb"))
    prediction = xgb_model.predict(features_eval)
    predictions = np.concatenate([predictions, prediction], axis=0)
    targets = np.concatenate([targets, targets_eval], axis=0)

print("Cohen Kappa quadratic score", 
      cohen_kappa_score(targets, predictions, weights="quadratic"))


# In[ ]:


# cleaning
if os.path.exists("cache"):
    for e in os.listdir("cache"):
        os.remove(os.path.join("cache", e))
    os.rmdir("cache")


# # Extract test features using CNN model

# In[ ]:


base_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_test = RetinopathyDataset(base_transform, is_test=True)
data_loader = torch.utils.data.DataLoader(data_test, 
                            batch_size=16, shuffle=False, 
                            num_workers=0, drop_last=False)

def get_test_features(data_loader, extractor):
    """ return an array of features extracted  
    """    
    for bi, d in enumerate(data_loader):
        if bi % 4 == 0 : print(".", end="")
        img_tensor = d["image"].to(DEVICE)
        with torch.no_grad(): feature = extractor(img_tensor)
        feature = feature.cpu().detach().numpy() #.squeeze(0) for batch_size > 1
        if bi == 0 :
            features = feature 
        else :
            features = np.concatenate([features, feature], axis=0)
    return features


# # Prediction using XGB

# In[ ]:


# adding prediction for each model 
# we can loop several time to perform data augmentation (tta) 
# (note: a bit risky as we cache the image, should be done bucket by bucket
#  and clean after each bucket to avoid filling all the disk space)
print("................................ v")
predictions = np.zeros((len(data_test),5))
for tta in range(1):
    print("............tta"+str(tta)+"................")
    for no in range(NUM_FOLDS):
        extractor = get_extractor_model(no)
        features = get_test_features(data_loader, extractor)
        print("",no)
        xgb_model = xgb.XGBClassifier()
        model_path = os.path.join("xgb_model_"+str(no))
        xgb_model = pickle.load(open(model_path, "rb"))
        prediction = xgb_model.predict_proba(features)
        predictions = predictions + prediction 


# # Prediction using CNN

# In[ ]:


batch_size = 8
data_augmentation = transforms.Compose([
    transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_test = RetinopathyDataset(data_augmentation, is_test=True)
data_loader = torch.utils.data.DataLoader(data_test, 
                            batch_size=batch_size, shuffle=False, 
                            num_workers=0, drop_last=False)


# In[ ]:


softmax = nn.Softmax(dim=1)
for tta in range(4):
    print("............tta"+str(tta)+"...............")
    for no in range(NUM_FOLDS):
        model = get_trained_model(no) #no
        batch_slice = (0, 0)
        for bi, d in enumerate(data_loader):
            if bi % (64//batch_size) == 0 : print(".", end="")
            img_tensor = d["image"].to(DEVICE, dtype=torch.float)
            batch_slice = (batch_slice[1], batch_slice[1]+img_tensor.size(0))
            with torch.no_grad(): 
                feature = model(img_tensor)
                feature = softmax(feature)
                predictions[batch_slice[0]:batch_slice[1],:] +=                         feature.cpu().detach().squeeze(0).numpy()
        print("")


# # Submission

# In[ ]:


# voting
prediction_final = predictions.argmax(axis=1)
csv_file = os.path.join(DATA_SOURCE, "sample_submission.csv")
df = pd.read_csv(csv_file)
df["diagnosis"] = prediction_final
df.to_csv('submission.csv',index=False)


# In[ ]:


df


# In[ ]:


# cache cleaning
if os.path.exists("cache"):
    for e in os.listdir("cache"):
        os.remove(os.path.join("cache", e))
    os.rmdir("cache")

