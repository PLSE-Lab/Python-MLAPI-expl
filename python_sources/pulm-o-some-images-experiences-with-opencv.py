#!/usr/bin/env python
# coding: utf-8

# <h1>Hello everyone!</h1>
# In this notebook, I will show a bit of my fun... i mean, work with the DICOM imagens and how I have analised them in order to try to extract better samples for a future training (probably a ConvNet)
# 
# It is a step by step of what I was thinking at the time, so it goes to some dead ends and not necessarily leads to the best solution, but I hope that it might be usefull to compare different lines of thought and perhaps give you a few ideas on how to approach this matter.
# (Feel free to comment on yours thoughts on it as well)
# 

# So let's start. I am using these libraries, with special mention to OpenCV

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
from matplotlib import pyplot
import cv2 
import random
import os


# Now let's set up the data and see what we are dealing with:

# In[ ]:


os.listdir('../input')


# In[ ]:


treino = pd.read_csv('../input/stage_1_train_labels.csv')
pac = pd.read_csv('../input/stage_1_detailed_class_info.csv')
img =[]
                  
for pid in treino['patientId']:
    DICOM = pydicom.read_file('../input/stage_1_train_images/{}.dcm'.format(pid))
    img.append(DICOM)                 
    


# Visualizing...

# In[ ]:


pyplot.imshow(img[random.randrange(len(img))].pixel_array)


# By the way, you can change the way the image is read including a maping option to make it looks more like a x-ray:

# In[ ]:


pyplot.imshow(img[random.randrange(len(img))].pixel_array, cmap = 'gray')


# To have a broader sample, let's get more images together.
# 
# *Obs: The images on this notebook are randomly selected, so if you want more samples, just re-run the code lines*

# In[ ]:


numIm = [4,4]
lista=[]
listaId=[]
for i in range(numIm[0]*numIm[1]):
    lista.append(img[random.randrange(len(img))].pixel_array)
    listaId.append(img[random.randrange(len(img))].PatientID)
    
graf, loc= pyplot.subplots(numIm[0],numIm[1], figsize=(20,20))
i=0
for lo in loc:
    for l in lo:
        l.imshow(lista[i])
        i =i+1


# Can't say much from just that, so let's get a positive image and see what a "pneumonia region" looks like
# 

# In[ ]:


positives = treino[treino['Target'] == 1]
rand = random.randrange(len(positives))
tempInfo = positives.iloc[rand]
for pid in img:
    if(pid.PatientID == tempInfo.patientId):
        temp = pid.pixel_array
        t1 = pid.PatientID


# In[ ]:


graf, loc= pyplot.subplots(2, figsize=(20,20))
temp2 = temp.copy()
temp2=cv2.rectangle(temp2, (int(tempInfo['x']),int(tempInfo['y'])), (int(tempInfo['x']+tempInfo['width']), int(tempInfo['y']+tempInfo['height'])) , 255, 3)
loc[0].imshow(temp2)

temp4 = temp2[  int(tempInfo['y']) : int(tempInfo['y']+tempInfo['height']+1), int(tempInfo['x']) : int(tempInfo['x']+tempInfo['width']+1) ]
loc[1].imshow(temp4)


# hmm... I can see that it is foggy, but can't really differentiate from any other "foggy part".
# 
# So, still no ideia on how to approach this.   I also don't know how doctors classify penumonia,  so I can only try to play with the information at hand 
# 

#  So let's try some transformations and see what happens. 
#  
#  
#  First, applying some blur filters (there are 3 types)
# 

# In[ ]:


rand = random.randrange(len(positives))
tempInfo = positives.iloc[rand]
for pid in img:
    if(pid.PatientID == tempInfo.patientId):
        temp = pid.pixel_array
        t1 = pid.PatientID
#temp with the marked place
tempM = temp.copy()
tempM=cv2.rectangle(tempM, (int(tempInfo['x']),int(tempInfo['y'])), (int(tempInfo['x']+tempInfo['width']), int(tempInfo['y']+tempInfo['height'])) , 255, 3)


# In[ ]:


kernel=(5,5)
temp2 = cv2.blur(temp, (kernel))
temp2 = cv2.blur(temp2, (kernel))
temp2 = cv2.blur(temp2, (kernel))
temp2 = cv2.blur(temp2, (kernel))

graf, loc= pyplot.subplots(1,2, figsize=(15,15))
loc[0].imshow(tempM)
loc[1].imshow(temp2)


# In[ ]:


kernel=(3,3)
temp2 =  cv2.GaussianBlur(temp,(kernel),0)
temp2 =  cv2.GaussianBlur(temp2,(kernel),0)
temp2 =  cv2.GaussianBlur(temp2,(kernel),0)
temp2 =  cv2.GaussianBlur(temp2,(kernel),0)

graf, loc= pyplot.subplots(1,2, figsize=(15,15))
loc[0].imshow(tempM)
loc[1].imshow(temp2)


# In[ ]:


temp2 = cv2.medianBlur(temp, 5)
temp2 = cv2.medianBlur(temp2, 5)
temp2 = cv2.medianBlur(temp2, 5)
temp2 = cv2.medianBlur(temp2, 5)

graf, loc= pyplot.subplots(1,2, figsize=(15,15))
loc[0].imshow(tempM)
loc[1].imshow(temp2)


#  Can't see anything new from that.

# Perhaps some erode/dilatations  (operations sometimes good to extract features)?

# In[ ]:


kernel =(5,5)
temp2 = cv2.erode(temp, (kernel),1)
temp2 = cv2.dilate(temp2, (kernel),1)
for i in range(50):
    temp2 = cv2.erode(temp2, (kernel),1)
for i in range(50):
    temp2 = cv2.dilate(temp2, (kernel),1)


# In[ ]:


graf, loc= pyplot.subplots(1,2, figsize=(15,15))
loc[0].imshow(tempM)
loc[1].imshow(temp2)


# It seems pretty interesting to "reduce" the bones over the lungs, leaving a higher density on places that had some "fog" previously, but I am not convinced that it will help to diferentiation between "non-pneumonia" and "pneumonia" fogs, so I will try other things

# Using some image oparations ( dilateded minus eroded image) to extract the borders

# In[ ]:


kernel =(3,3)
temp2 = cv2.erode(temp, (kernel),1)
temp2 = cv2.erode(temp2, (kernel),1)
temp3 = cv2.dilate(temp, (kernel),1)
temp3 = cv2.dilate(temp3, (kernel),1)
temp2 = temp3-temp2
pyplot.figure(figsize=(15,15))

temp2=cv2.rectangle(temp2, (int(tempInfo['x']),int(tempInfo['y'])), (int(tempInfo['x']+tempInfo['width']), int(tempInfo['y']+tempInfo['height'])) , 255, 3)
pyplot.imshow(temp2)


# Kind hard to evaluate, but at a closer look, also seems be hard to diferentiate between the diferent fogs

# Now some thresholding

# In[ ]:


temp2 =  cv2.GaussianBlur(temp,(5,5),0)
ret, temp2 = cv2.threshold(temp2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
graf, loc= pyplot.subplots(1,2, figsize=(15,15))
loc[0].imshow(tempM)
loc[1].imshow(temp2)


# In[ ]:


ret, temp2 = cv2.threshold(temp2,175,255,cv2.THRESH_BINARY)
graf, loc= pyplot.subplots(1,2, figsize=(15,15))
loc[0].imshow(tempM)
loc[1].imshow(temp2)


# In[ ]:


ret, temp2 = cv2.threshold(temp,70,1,1)
graf, loc= pyplot.subplots(1,2, figsize=(15,15))
loc[0].imshow(tempM, cmap="gray")
temp2=cv2.addWeighted(temp, 1.2, temp2, -50, 1.0)
loc[1].imshow(temp2, cmap="gray")


# In[ ]:


graf, loc= pyplot.subplots(1,3, figsize=(15,15))
ret, temp2 = cv2.threshold(temp,50,255,1)
loc[0].imshow(tempM)
temp2 = temp-temp2
loc[1].imshow(temp2)
ret, temp2 = cv2.threshold(temp2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
loc[2].imshow(temp2)


# Again, I am not convinced that it might help, specially since "fog" and the bones have a very close color and they end up together

# And some changes to the alfa e beta changes, trying to increasing the contrast

# In[ ]:


temp2= cv2.convertScaleAbs(temp, alpha=3, beta=-350)
temp3= cv2.convertScaleAbs(temp, alpha=5, beta=-700) 
temp4= cv2.convertScaleAbs(temp, alpha=3.5, beta=-500)

graf, loc= pyplot.subplots(2,2, figsize=(15,15))

loc[0,0].imshow(tempM, cmap='gray')
loc[0,1].imshow(temp2, cmap='gray')
loc[1,0].imshow(temp3, cmap='gray')
loc[1,1].imshow(temp4, cmap='gray')
loc[1,1].set_title('the selected values after some observation')


# Now it seems to be a better solution. Increasing the contrast exposes a lot more (at least visually) the "fog". 
# 
# If we start from here and try to extract the features, we might have good results. So let's define a kernel to use to train the future ml model

# First, let's check if the image has anything significant in the density distribution

# In[ ]:


rand = random.randrange(len(positives))
tempInfo = positives.iloc[rand]
for pid in img:
    if(pid.PatientID == tempInfo.patientId):
        temp = pid.pixel_array
        t1 = pid.PatientID


# In[ ]:


def density(imag, x,y, alt, larg):    
    dLin=[0]*alt
    dCol=[0]*larg
    for col in range(larg):       
        for row in range(alt):  
            dLin[row] = imag[y+row, x+col] + dLin[row]
            dCol[col] = imag[y+row, x+col] + dCol[col]  
    return({'lin':dLin , 'col':dCol})          


# In[ ]:


temp2 = cv2.convertScaleAbs(temp, alpha=3.5, beta=-500)
ret = density(temp2, int(tempInfo['x']), int(tempInfo['y']),int(tempInfo['height']), int(tempInfo['width']) )
ret2= density(temp2, 0, 0, len(temp), len(temp) )

graf, loc= pyplot.subplots(3,2, figsize=(20,20))
loc[0,0].barh(range(len(ret['lin'])),list(reversed(ret['lin'])))
loc[0,0].set_title('Vertical density (roi)')
loc[0,1].bar(range(len(ret['col'])),ret['col'])
loc[0,1].set_title('Horizontal density (roi)')
loc[2,0].barh(range(len(ret2['lin'])),list(reversed(ret2['lin'])))
loc[2,0].set_title('Vertical density (whole image)')
loc[2,1].bar(range(len(ret2['col'])),ret2['col'])
loc[2,1].set_title('Horizontal density (whole image)')
loc[1,0].imshow(temp2[  int(tempInfo['y']) : int(tempInfo['y']+tempInfo['height']+1), int(tempInfo['x']) : int(tempInfo['x']+tempInfo['width']+1) ],cmap='gray')
loc[1,0].set_title('ROI')
loc[1,1].imshow(temp2, cmap='gray')
loc[1,1].set_title('whole image')


# No ideia if the density might be significative, hehe. 
# 
# But I am starating to think that  the "pneumonia fog" have a shape that is closer to a "amoeba"  (with some "little arms" on the edges) than a random fog, what gives a insteresting pattern to work with.
# 
# All that is left is to find the a way to extract this information from the image

# In[ ]:


graf, loc= pyplot.subplots(3,2, figsize=(20,20))
temp2 = temp.copy()
temp2=cv2.rectangle(temp2, (int(tempInfo['x']),int(tempInfo['y'])), (int(tempInfo['x']+tempInfo['width']), int(tempInfo['y']+tempInfo['height'])) , 255, 3)
loc[0,0].imshow(temp2)
loc[0,0].set_title('Original')

temp2= cv2.convertScaleAbs(temp, alpha=3.5, beta=-500)
temp2=cv2.rectangle(temp2, (int(tempInfo['x']),int(tempInfo['y'])), (int(tempInfo['x']+tempInfo['width']), int(tempInfo['y']+tempInfo['height'])) , 255, 3)
loc[0,1].imshow(temp2)
loc[0,1].set_title('Alfa/Beta changes')

temp2 = cv2.medianBlur(temp, 5)
temp2 = cv2.medianBlur(temp2, 5)
temp2 = cv2.medianBlur(temp2, 5)
temp2= cv2.convertScaleAbs(temp2, alpha=3.5, beta=-500)
temp2=cv2.rectangle(temp2, (int(tempInfo['x']),int(tempInfo['y'])), (int(tempInfo['x']+tempInfo['width']), int(tempInfo['y']+tempInfo['height'])) , 255, 3)
loc[1,0].imshow(temp2)
loc[1,0].set_title('Alfa/Beta + blur')

temp3 = cv2.resize(temp, (64,64))
temp3= cv2.convertScaleAbs(temp3, alpha=3.5, beta=-500)
loc[1,1].imshow(temp3)
loc[1,1].set_title('Alfa/Beta + resize')

temp4 = temp2[  int(tempInfo['y']) : int(tempInfo['y']+tempInfo['height']+1), int(tempInfo['x']) : int(tempInfo['x']+tempInfo['width']+1) ]
loc[2,0].imshow(temp4)
loc[2,0].set_title('Alfa/Beta + blur on ROI')

temp5= cv2.resize(temp4, (32,32))
loc[2,1].imshow(temp5)
loc[2,1].set_title('Alfa/Beta + resize on ROI')


# After some attempts, I figured that not only it is possible to reduce the image size without losing too much information (in this case, the "fog" still kind of looks likes an amoeba, what is a must to gain efficiency.
# 
# So, let's compare with some random negative data to see if they are really diferent and might be good subjects to a ml model
# 

# In[ ]:


negatives = treino[treino['Target'] == 0]
rand = random.randrange(len(negatives))
tempInfoN = negatives.iloc[rand]
for pid in img:
    if(pid.PatientID == tempInfoN.patientId):
        tempN = pid.pixel_array
        t1 = pid.PatientID
        
tempN = cv2.convertScaleAbs(tempN, alpha=3.5, beta=-500)


# In[ ]:


rand = random.randrange(len(positives))
tempInfo = positives.iloc[rand]
for pid in img:
    if(pid.PatientID == tempInfo.patientId):
        temp = pid.pixel_array
        t1 = pid.PatientID
temp = cv2.convertScaleAbs(temp, alpha=3.5, beta=-500)


# In[ ]:


graf, loc= pyplot.subplots( 2 , figsize=(15,15))

temp = temp[int(tempInfo['y']) : int(tempInfo['y']+tempInfo['height']+1), int(tempInfo['x']) : int(tempInfo['x']+tempInfo['width']+1)]
temp = cv2.resize(temp, (32,32))

loc[0].imshow(temp)
loc[0].set_title('random positive sample')
tam = 1024 - 32

x= random.randrange(tam)
y= random.randrange(tam)

loc[1].imshow(tempN[y:y+32 , x:x+32] )       
loc[1].set_title('random negative sample')


# Overall, the positive samples are more "granulated" than the negatives ones.
# Seems promissing!
# 
# Now, all that is left to do is test this concept and see if we can train something from this.
# 
# <hr>
#  I hope that this might have been usefull to you.
#  Please excuse any language mistakes and feel free to share your toughts/tips on this notebook as well!

# In[ ]:




