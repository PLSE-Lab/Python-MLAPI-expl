# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # opencv library 

cam = cv2.VideoCapture(0)

#boundary conditions for green color H,S,V
lowerBound=np.array([33,80,40])
upperBound=np.array([102,255,255])

#for removing noises window size
kernelOpen = np.ones((5,5))
kernelClose = np.ones((20,20))

font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 0)

while(True):
    ret, img = cam.read()
    img = cv2.resize(img,(340,220))
    
    #convert BGR to HSV  HSV -> Hue, Saturation, Value.
    #The hue of a pixel is an angle from 0 to 359 the value of each angle decides the color of the pixel 
    #The Saturation is basically how saturated the color is, and the Value is how bright or dark the color is
    #So the range of these are as follows
    #Hue is mapped – >0º-359º as [0-179]
    #Saturation is mapped ->  0%-100% as [0-255]
    #Value is 0-255 (there is no mapping)
    
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    #create a filter or mask to filer out a specific color here we filter green color
    mask = cv2.inRange(imgHSV,lowerBound,upperBound)
    
    # removing the noises in the image. we use a window and slide it over the image whenever window find a white spot smaller than its size it covers it coz its due to noise
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    #opposite of maskopen it fill white if it finds black smaller than window size
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    
    maskFinal = maskClose
    #in the maskFinal it detect where the white space is showing and return its contour we can use this contour and draw it around object in actual image
    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img,conts,-1,(255,0,0),3)
    
    #now the contour will be shaped as detected by filter if u want to place a rectangle instead we write below code otherwise remove it
    for i in range(len(conts)):
        x,y,w,h = cv2.boundingRect(conts[i])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.putText(img,str(i+1),(x,y+h),font,fontscale, fontcolor)
 
    cv2.imshow('mask',mask)
    cv2.imshow('cam',img)
    cv2.imshow('maskopen',maskOpen)
    cv2.imshow('maskclose',maskClose)
    #cv2.waitKey(10)
    
    if(cv2.waitKey(1) == ord('q')):
        break;#ord return unicode of q we exit on pressing q
cam.release()
cv2.destroyAllWindows()