#!/usr/bin/env python
# coding: utf-8

# # Objective:
# 
# Objective of this notebook is how to host Deep Learning Models on AWS Lambda layer. The emphasis is to follow a structured approach which could help in following a template when it comes to hosting Deep Learning models. Contrary to the belief that we need a GPU based environment for model serving, I would like to highlight that CPU based model serving does well. 
# 
# A structured approach helps in testing a model before depoying on AWS Lambda. This also saves on  debugging time. The Deep Learning model is written using python's FastAI and Pytorch libraries. 
# 
# These libraries enable to quickly write Deep Learning models. Pytorch has become quite popular in just two years after its release and FastAI is further helping by providing a framework which makes it an appropriate choice in applying deep learning techniques to industry problems.
# 
# This article also describes about how to test pieces of code in a Jupyter Notebook, before deploying to AWS Lambda.
# 

# # Background & Motivation
# 
# Chest X-ray exam is one of the most frequent and cost-effective medical imaging examination. However clinical diagnosis of chest X-ray can be challenging, and sometimes believed to be harder than diagnosis via chest CT imaging. Even some promising work have been reported in the past, and especially in recent deep learning work on Tuberculosis (TB) classification. NIH Chest X Ray Dataset (https://nihcc.app.box.com/v/ChestXray-NIHCC) provides a comprehensive dataset of X Rays which may be used for developing machine learning models which can predict diseases related to chest region. This dataset comprises of 14 labels of data, listed below:
# 
# (1, Atelectasis; 2, Cardiomegaly; 3, Effusion; 4, Infiltration; 5, Mass; 6, Nodule; 7, Pneumonia; 8, Pneumothorax; 9, Consolidation; 10, Edema; 11, Emphysema; 12, Fibrosis; 13, Pleural_Thickening; 14 Hernia)
# 
# ![image.png](attachment:image.png)
# 
# Source(http://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a)
# 
# As challenging is to train an efficient and accurate ML model, equally challenging is to provide this model for consumption. Also, we are looking to create the experience closer to reality. i.e. when an examiner looks at the X-Ray, (s)he is able to tell about a number of diseases, similarly, our model hosting also achieves this, i.e. it can output a number of labels (symptoms in an X-Ray) (**multi-label classification**).
# 
# Here in this article, we will focus on How to host a pretrained Deep Learning model on a cloud platform.

# # Prerequisites
# 
# You will need a pretrained model in trained in Pytorch framework. Alongwith the model, output labels are also required. The trained model and output classes are included in a tar file.
# 
# For this case, the trained model is named as `chestxray_resnet50_jit.pth` and prediction classes are available in `classes.txt`. At the inference time, above two files are used to respond to a query sent in form of an image.
# 
# The name of the tarfile is `chestxray.tar.gz`
# 
# Also, the language for building models, productionizing on AWS lambda is python.
# 
# **Note** : As this is an exercise on hosting a model, not much emphasis has been given to accuracy of the model, but the model is expected to do well as it is using resnet50 architecture with an optimal learning rate applied.

# # Prediction Service Hosted on AWS
# Prediction service is hosted on AWS lambda and the model is kept in a S3 bucket. The S3 bucket holds the tar file which houses model and label information and the execution is done at Lambda layer.
# 
# ![image.png](attachment:image.png)
# 
# Above gives a view of various AWS components required for model hosting. Please notice that how simple this architecture is.
# 

# # lambda_handler
# The starting point of Lambda execution is **lambda_handler** function.

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# Once the Lambda layer is invoked, the execution is done with two primary functions: **input_fn** and **predict**.
# 
# - **input_fn** - This function internally calls load_and_process_image which takes an url and returns a image tensor. With the help of url, this input_fn_core can be tested in a jupyter notebook environment. Let us see this here. Here, in order to test the functionality in a notebook, `input_fn` is used as a wrapper function, which actually calls the `load_and_process_image` which takes a parameter called url as an input.
# 
# - **predict** - This function encapsulates the call to model and applies logic for inferencing. In this article, we will also talk about a strategy to do multi-label classification, by using thresholds in a function `predict_multi_labels`.
# 

# In[ ]:


import os
import io
import json
import tarfile
import glob
import time
import logging
import boto3
import requests
import PIL
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import models, transforms


# **IMPORTANT**
# 
# Before calling the model, the important thing is to load the image, resize this and do transformations, if applicable. Sometimes, if there is a mismatch in the methodology how a image has been loaded, resized at the training time and inference time, we may get random results.
# 
# No matter, how good our model is, if this step is not right, we may not expect our model to behave rightly. There may be some loss of pixel values, because of randomness produced due to resizing of images, but generally tensors should match. Hence this is recommended to match the pixel values of the image at train time and inference time and ensure that these agree.
# 
# To facilitate this, `load_and_process_image` function is written, whose input is url to an image which needs to be examined. Lets look at the function and examine the output.

# In[ ]:


def load_and_process_image(url, size):
    # Notice the unsqueeze part in the last line,this is for having a batch size of 1.
    img_request = requests.get(url, stream=True)
    img = PIL.Image.open(io.BytesIO(img_request.content))
    img = img.convert('RGB')
    img = img.resize((size, size),PIL.Image.BILINEAR)
    res = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    res = res.view(size, size, -1).permute(2,0,1).float().div_(255)
    return(res.unsqueeze(0))


# In[ ]:


url = "https://drive.google.com/uc?export=download&id=1W-3GLMoKv00i_w1niGqEdcZjDDDjbFmw" # 00028173_003.png
load_and_process_image(url, size = 128)


# # predict
# The predict function does a few things. To help in aligning with business demands, helper functions like predict_one_label and predict_multi_labels are created and output is then sent back to caller. Following is the structure of the predict function. This returns a dictionary back to the caller.

# In[ ]:


def predict(input_object, model):
    """Predicts the class from an input image.
    Parameters
    ----------
    input_object: Tensor, required
        The tensor object containing the image pixels reshaped and normalized.
    Returns
    ------
    Response object: dict
        Returns the predicted class and confidence score.
    
    """        
    predict_values = model(input_object)
    predict_values = F.softmax(predict_values)
    response = prediction_one_label(predict_values)
    response_str = prediction_multi_labels(predict_values)
    response['multi_label'] = response_str
    return response


# * Notice that predict function is used as a wrapper, which calls rwo functions , `predict_one_label` and `predict_multi_labels`. This is an important addition where we can handle the use case of how to handle multi label prediction at inference stage.

# # predict_one_label

# In[ ]:


def prediction_one_label(predict_values):
    preds = F.softmax(predict_values, dim=1)
    conf_score, indx = torch.max(preds, dim=1)
    predict_class = classes[indx]
    response = {}
    response['single_class'] = str(predict_class)
    response['single_confidence'] = conf_score.item()
    response['predict_values'] = predict_values
    return response


# # predict_multi_labels

# In[ ]:


def prediction_multi_labels(predict_values):
    indxx = (predict_values > threshholds)
    indxx = np.array(indxx).reshape(-1)
    output = [classes[x]  for x in range (len(classes))  if  (indxx[x]==1)] 
    output = [x.decode('utf-8') for x in output ] 
    if (len(output) > 1):
        output = [x for x in output if str(x) != 'No Finding']
    retval = ';'.join(output); retval
    return retval


# # Setting up AWS Access Parameters
# 
# Before accessing AWS resources, we need to set up some parameters which will tell the program where to take the resources (model and labels).
# 
# Note: Following parameters will not work for you, you need to supply your own AWS account details. 

# In[ ]:


bucket_url = 'https://tfg-models.s3.us-east-2.amazonaws.com/chestxray.tar.gz' 
ACCESS_ID = 'AKIARM4DEL66FTOYP72M'
ACCESS_KEY = '4+5UlbGmbl48wZY2rKyaGR4RVXW7OF/oDe+dO4k2'
MODEL_BUCKET = "test-model-bucket"
MODEL_KEY = "chestxray.tar.gz"
THRESHOLD_VALUES = torch.tensor([.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20]) 


# # Loading model
# This is done once so there is no overhead to loading the model every time. Please note that threshold values are important for multi-label classification which are obtained from statistical considerations, and applicable for the use case we are trying to solve.

# In[ ]:


def load_model():
    """Loads the PyTorch model into memory from a file on S3.
    Returns
    ------
    Vision model: Module
        Returns the vision PyTorch model to use for inference.
    The developer needs to replace print call to logger.info before uploading to aws lambda.    
    
    """      
    global classes
    global threshholds
    threshholds = THRESHOLD_VALUES # Multilabel Thresholds
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_ID, aws_secret_access_key= ACCESS_KEY)
    print('Loading model from S3')
    obj = s3.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
    bytestream = io.BytesIO(obj['Body'].read())
    tar = tarfile.open(fileobj=bytestream, mode="r:gz")
    for member in tar.getmembers():
        if member.name.endswith(".txt"):
            print("Classes file is :", member.name)
            f=tar.extractfile(member)
            classes = f.read().splitlines()
            print(classes)
        if member.name.endswith(".pth"):
            print("Model file is :", member.name)
            f=tar.extractfile(member)
            print("Loading PyTorch model")
            model = torch.jit.load(io.BytesIO(f.read()), map_location=torch.device('cpu')).eval()
    return model


# In[ ]:


model = load_model()


# # Calling Model

# In[ ]:


img_url = "https://drive.google.com/uc?export=download&id=1W-3GLMoKv00i_w1niGqEdcZjDDDjbFmw"
input_object = load_and_process_image(img_url, size = 128)
start_time = time.time()
print("--- Object loading time: %s seconds ---" % (time.time() - start_time))
print(input_object.size())
print("Calling prediction")
start_time = time.time()
response = predict(input_object, model)
print("--- Inference time: %s seconds ---" % (time.time() - start_time))
print(response)


# Above code execution eastablishes that the model loading, image processing and getting inference are working as per the expectations. We have also observed how an image tensor looks like. Now it is the time to thread the pieces together and so that they are made available in a AWS Lambda function.
# 
# While deploying the model at Lambda layer, we need to do certain changes, instead of putting the various keys and values in a script, they would be passed dynamically. Following will make this clear.
# 
# ![image.png](attachment:image.png)
# 
# Now, its time to stitch the input_fn and predict functions together. Once various pieces are done, input_fn and predict become very easy.
# 
# ![image.png](attachment:image.png)

# # Deploy on AWS Lambda
# 
# You will need aws and sam utilities to do the same. 
# with `aws configure` command, you can set up `ACCESS_ID` and `ACCESS_KEY` among other parameters.
# Then execute the following command:
# 
# `sam package --output-template-file packaged.yaml --s3-bucket "YOUR_BUCKET_NAME"`
# 
# `sam deploy --template-file packaged.yaml --stack-name "YOUR_STACL_NAME" --capabilities CAPABILITY_IAM --parameter-overrides BucketName="YOUR_BUCKET_NAME" ObjectKey=chestxray.tar.gz`
# 

# ## Testing the model from Command Line
# 
# This can be done using Curl Command. Some of the urls are masked and they will be changed as per your implementation.
# 
# ![image.png](attachment:image.png)
# 
# Note: An example github project is available to you at https://github.com/sinharitesh/aws-lambda-starter . This is an uncustomized version, you need to implement various functions described above to fit to your needs.

# # Conclusion
# 
# Thats it! You can see that how a deep learning model built in Pytorch can be successfully wrapped inside a AWS lambda layer. This is useful, as deep learning models can be succesfully served on a cloud platform and this is also cost efficient and less maintenance centric. 
# 
# Another advantage is that hosting on AWS enables this to scale when loads go up and user only pays for compute time.
# 
# We have also discovered a strategy where multi-label classification can be approached. This model hosting can also be tried at other cloud platforms like Azure.
# 
# Please send your feedback through the comments section. Happy Model Hosting!
