#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import cv2
import numpy as np
cap = cv2.VideoCapture('Video/face_track.mp4')


# In[ ]:



ret, frame = cap.read()


# In[ ]:


face_casc = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
face_rects = face_casc.detectMultiScale(frame)


# In[ ]:


face_x , face_y , w, h = tuple(face_rects[0])
track_window = (face_x,face_y,w,h)


# In[ ]:



roi = frame[face_y:face_y+h,
           face_x:face_x+w]


# In[ ]:


hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


# In[ ]:


roi_hist = cv2.calcHist([hsv_roi],
                       [0],
                       None,
                       [180],
                       [0,180])


# In[ ]:


cv2.normalize(roi_hist,
             roi_hist,
             0,
             255,
             cv2.NORM_MINMAX);


# In[ ]:


term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10 ,1)


# In[ ]:


# While loop
while True:

    # read capture
    ret, frame = cap.read()
    
    # if statement
    if ret == True:
    
        # Frame in HSV
        hsv = cv2.cvtColor(frame,
                          cv2.COLOR_BGR2HSV)
        
        # Calculate the base of ROI
        dest = cv2.calcBackProject([hsv],
                                  [0],
                                  roi_hist,
                                  [0,180],
                                  1)
        
        # Camshift to get the new coordinates of rectangle
        ret , track_window = cv2.CamShift(dest,
                                         track_window,
                                         term_crit)
        
        # Draw rectangle on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,
                            [pts],
                            True,
                            (0,255,0),
                            5)
        
        # Open new window and display
        cv2.imshow('Cam Shift',img2)
        
        # Close window
        if cv2.waitKey(50) & 0xFF == 27:
            break
        
    # else statement
    else:
        break
    
# Release and Destroy
cap.release()
cv2.destroyAllWindows()

