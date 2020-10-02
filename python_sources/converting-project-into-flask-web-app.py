#!/usr/bin/env python
# coding: utf-8

# Hope you all are doing well and learning something new.
# So as a last part of this course I want to present a method to deploy this project as a web application using Flask.

# # What you will learn <br>
# First of all it's important to note that I am not going to tell anything about web development. Topic that will be covered here is how to use flask to deploy this competition's code and make a deep learning based web application.
# 
# <br><br>
# I am supposing that you are on your local machine. So let us see step by step approach to do so.
# 
# 

# # Step 1) create a virtual environment

# I have tested my application on python-3.7 so I am creating virtual environment for this python version .
# <br><br>
# So type following command in your terminal:<br><br>
# python3.7 -m virtualenv name_you_want_to_give_of_your_virtual_environment

# ![Screenshot%20from%202020-06-29%2011-33-33.png](attachment:Screenshot%20from%202020-06-29%2011-33-33.png)

# Above I created virtual environment named as 'pytorchh'

# # Step2) Activate that environment

# In[ ]:


cd name_you_want_to_give_of_your_virtual_environment
. bin/activate


# ![Screenshot%20from%202020-06-29%2011-33-58.png](attachment:Screenshot%20from%202020-06-29%2011-33-58.png)

# In the last line you can see (pytorchh) that means now I am in newly created pytorchh environment.<br><br>
# In this environent there will be not any module present so you have to install all required modules.

# # Step 3) Create some folders and files
# 
# Create a base folder that will be your project folder inside your virtual environment folder.<br>
# 
# ![Screenshot%20from%202020-06-29%2011-45-00.png](attachment:Screenshot%20from%202020-06-29%2011-45-00.png)
# 
# <br>
# Here you can see that 'pytorchh' is environment directory and inside it i have created a 'protien' directory that is my project folder<br><br>
#  Inside protien folder there are some essential folders and files that are required by flask that are static and templates folder and one driver python file that is app.py here.<br><br>
# 
# folders : models , static/css/main.css, static/js/main.js ,templates/home.html,  templates/result.html , uploads <br>
# files : app.py

# If you are familiar with flask then you know that static and templates folder and one driver python file are required by any flask app.<br><br>
# # What it means??<br>
# <br>
# Well! static folder serves static files such as images, css, js and templates folder serves templates that are html files. And there is one python file that is called driver file that will be given in command line to run flask app. 

# Don't worry about html ,css and js code. They will be given in the last of this tutorial.<br> Only understand the basic structure 

# 
# # Step 4)Inside app.py
# Here I have written basic code required for classification 

# # 4a) First import all important libraries

# In[ ]:


from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn


import torch.nn.functional as F


# # 4b) Define a flask app 

# In[ ]:


# Define a flask app 
app = Flask(__name__)   #always required when creating flask app
MODEL_PATH = 'models/model_resnet.pth'    #your model's path where you have stored it.


# # 4c) Define Labels

# In[ ]:


labels = {
    0: 'Mitochondria',
    1: 'Nuclear bodies',
    2: 'Nucleoli',
    3: 'Golgi apparatus',
    4: 'Nucleoplasm',
    5: 'Nucleoli fibrillar center',
    6: 'Cytosol',
    7: 'Plasma membrane',
    8: 'Centrosome',
    9: 'Nuclear speckles'
}


# # 4d) Define your model same as you have created previously during training.<br><br>
# Here I am using this notebook's model: https://www.kaggle.com/nachiket273/protein-classification-one-cycle for the privacy of my model according to competition rule.<br><br>
# So here you can store your own model.

# In[ ]:


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=1):
        super().__init__()
        self.adavgp = nn.AdaptiveAvgPool2d(sz)
        self.adamaxp = nn.AdaptiveMaxPool2d(sz)
        
    def forward(self, x):
        x = torch.cat([self.adavgp(x), self.adamaxp(x)], 1)
        x = x.view(x.size(0),-1)
        return x



class CustomClassifier(nn.Module):
    def __init__(self, in_features, intermed_bn= 512, out_features=10, dout=0.25):
        super().__init__()
        self.fc_bn0 = nn.BatchNorm1d(in_features)
        self.dropout0 = nn.Dropout(dout)
        self.fc0 = nn.Linear(in_features, intermed_bn, bias=True)
        self.fc_bn1 = nn.BatchNorm1d(intermed_bn, momentum=0.01)
        self.dropout1 = nn.Dropout(dout * 2)
        self.fc1 = nn.Linear(intermed_bn, out_features, bias=True)
        
    def forward(self, x):
        x = self.fc_bn0(x)
        x = self.dropout0(x)
        x = F.relu(self.fc0(x))
        x = self.fc_bn1(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        return x


# # 4e) Load your pretrained model that has been used for training

# In[ ]:


model = models.resnet18(pretrained=True)   #loaded pretrained model


# # 4f) Add your own layers in that pretrained model(if any) same as during training. 

# In[ ]:


model.avgpool = AdaptiveConcatPool2d()     #added own defined layer
model.fc = CustomClassifier(in_features=model.fc.in_features*2, out_features=10) #added own classifier layer according to requirement


# # 4g) Here it's important to note our model was on gpu , so for prediction on cpu map model to cpu
# 
# 
# map_location=torch.device('cpu')

# In[ ]:


#load and map model dict

# model was trained on gpu so mapped it to cpu as our application will do prediction on cpu 
# and tensors also will be on cpu
model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))


# # 4h) Now define transformation for your images same as when model was being tested 

# In[ ]:


data_transforms = transforms.Compose([
        transforms.RandomCrop(512, padding=8, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0793, 0.0530, 0.0545], std=[0.1290, 0.0886, 0.1376])
        ])


# # 4i) Define pred_single function (see in your competition code ,it must be there)
# Here one function of Image class from PIL module has been used ==>>> Image.open()<br>
# Note this point.

# In[ ]:


def pred_single(img_path, return_label=True):
    with torch.no_grad():
        model.eval()
        img = Image.open(img_path)  #NOTE : Image.open 
        img = data_transforms(img)
        bs_img = img.unsqueeze(0)
        #bs_img = bs_img.to(device)
        preds = torch.sigmoid(model(bs_img))
        prediction = preds[0]
        return prediction


# # 4j) Define a function to decode labels(it will be present in your competition code)

# In[ ]:


def decode_labels(target, thresh=0.5, return_label=True):
    result = []
    for i, tgt in enumerate(target):
        if tgt > thresh:
            result.append(str(i) + ":" + labels[i] + " ")           
    return result


# # 4k) Now here comes some basic flask concpets that can be used to make a web application
# First define a home/index route ('/') that means when you open the browser with given link it is routed here and renders a template index.html

# In[ ]:


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


# # 4l) Now here is another route ('/predict') with get and post request. 
# 
# If request is post that means you have uploaded image from upload files button from browser then it will be stored in uploads folder then it will be predicted and corresponding label will be returned.

# In[ ]:


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))  
        f.save(file_path)    # save image into uploads folder

        # Make prediction
        preds = pred_single(file_path, model)  # predict that image
        result = decode_labels(preds)          # decode that prediction
        result = result[0] + "" + result[1]    # create a string of result 
        return result                          # return that result
    return None


if __name__ == '__main__':
    app.run(debug=True,threaded=False)


# # step5) Move your .pth file into models folder
# 
# This is the only step you need to do if you are using my source code(link my source code will be given below )

# # step6) Now run your application from command line
# 
# python app.py

# A link will be appeared : https://127.0.0.1:5000

# ![Screenshot%20from%202020-06-29%2011-58-37.png](attachment:Screenshot%20from%202020-06-29%2011-58-37.png)

# Open this link and your application will start working

# # Points to take away <br>
# 
# 1. Download pretrained model and make changes in it same as you have done in your notebook.<br>
# 2. Apply transformation same as you have done in your notebook during testing/validation of model<br>
# 3. store your model's .pth file inside models folder (if using my source code's folder structure) 

# My github Repo(Source code is here ): <br>
# https://github.com/pandeynandancse/human-protein-classifier
# <br>
# <br>
# # FEEL FREE TO FORK IT

# In[ ]:




