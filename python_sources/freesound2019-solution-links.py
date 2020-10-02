#!/usr/bin/env python
# coding: utf-8

# The description of our solution is here.  
# https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/96440
# 
# 
# ### preprocessed log-mel
# - train/test https://www.kaggle.com/osciiart/mel128v3  
# - noisy  https://www.kaggle.com/osciiart/mel128v3n
# 
# ### Model #1 ResNet34, MTL, slice=512
# - fold 1 https://www.kaggle.com/osciiart/resnet34-mel-ver3-log-multi-hardaug?scriptVersionId=13887036
# - fold 2 https://www.kaggle.com/osciiart/resnet34-mel-ver3-log-multi-hardaug?scriptVersionId=13931215
# - fold 3 https://www.kaggle.com/osciiart/resnet34-mel-ver3-log-multi-hardaug?scriptVersionId=13931219
# - fold 4 https://www.kaggle.com/osciiart/resnet34-mel-ver3-log-multi-hardaug?scriptVersionId=13931223
# - fold 5 https://www.kaggle.com/osciiart/resnet34-mel-ver3-log-multi-hardaug?scriptVersionId=13931230
# - trained weights https://www.kaggle.com/osciiart/resnet34hardaug512
# 
# ### Model #2 ResNet34, MTL, SSL, slice=512
# - fold 1 https://www.kaggle.com/osciiart/resnet34multi-mixmatch?scriptVersionId=15206185
# - fold 2 https://www.kaggle.com/osciiart/resnet34multi-mixmatch?scriptVersionId=15238544
# - fold 3 https://www.kaggle.com/osciiart/resnet34multi-mixmatch?scriptVersionId=15238571
# - fold 4 https://www.kaggle.com/osciiart/resnet34multi-mixmatch?scriptVersionId=15238580
# - fold 5 https://www.kaggle.com/osciiart/resnet34multi-mixmatch?scriptVersionId=15240622
# - trained weights https://www.kaggle.com/osciiart/resnet34mix2
# 
# ### Model #3 ResNet34, MTL, slice=1024
# - fold 1 https://www.kaggle.com/osciiart/resnet34-multi1024/output?scriptVersionId=15293609
# - fold 2 https://www.kaggle.com/osciiart/resnet34-multi1024/output?scriptVersionId=15293637
# - fold 3 https://www.kaggle.com/osciiart/resnet34-multi1024/output?scriptVersionId=15293651
# - fold 4 https://www.kaggle.com/osciiart/resnet34-multi1024/output?scriptVersionId=15360273
# - fold 5 https://www.kaggle.com/osciiart/resnet34-multi1024/output?scriptVersionId=15293584
# - trained weights https://www.kaggle.com/osciiart/resnet34multi1024
# 
# ### Model #4 EnvNet-v2, multitasklearning, sigmoid, slice=133300
# - trained weights https://www.kaggle.com/junyasato/bs16bce
# 
# ### Model #5 EnvNet-v2, multitasklearning, SoftMax, slice=133300
# - trained weights https://www.kaggle.com/junyasato/envnet133300
# 
# ### Model #6 EnvNet-v2, multitasklearning, sigmoid, slice=200000
# - trained weights https://www.kaggle.com/junyasato/envnet200000
# 
# ### Inference
# Public LB 0.752 https://www.kaggle.com/osciiart/make-final-submission1?scriptVersionId=15486664  
# Public LB 0.750(final submission 1) https://www.kaggle.com/osciiart/make-final-submission1?scriptVersionId=15520633  
# Public LB 0.747(final submission 2) https://www.kaggle.com/osciiart/make-final-submission2?scriptVersionId=15516494  

# In[ ]:




