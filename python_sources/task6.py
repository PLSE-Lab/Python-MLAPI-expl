#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from dsClass.path_helper import *

# Any results you write to the current directory are saved as output.


# # Objective
# 
# Detect and recognize the faces in the following youtube video:

# In[ ]:


from IPython.display import HTML
from IPython.display import YouTubeVideo

#HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/403jzB62dAs?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')
YouTubeVideo('403jzB62dAs')


# # Imports

# In[ ]:


import os

# change directory to the dataset where our
# custom scripts are found
os.chdir("/kaggle/input/")

from dsClass.align_custom import AlignCustom
from dsClass.face_feature import FaceFeature
from dsClass.mtcnn_detect import MTCNNDetect
from dsClass.tf_graph import FaceRecGraph


# In[ ]:


import os, sys, glob
import cv2

import argparse
import sys
import json
import numpy as np
import time

import scipy
import scipy.io as sio
from scipy.io import loadmat
from datetime import datetime
import pandas as pd
import time


# # Detection and Recgnition

# Based and Inspired by:
# - **https://github.com/vudung45/FaceRec
# - Augmentation code: https://github.com/vxy10/ImageAugmentation
# - Fancy borders: https://www.codemade.io/fast-and-accurate-face-tracking-in-live-video-with-python/

# Description:
# - Images from Video Capture -> detect faces' regions -> crop those faces and align them 
# - each cropped face is categorized in 3 types: Center, Left, Right 
# - Extract 128D vectors( face features)
# - Search for matching subjects in the dataset based on the types of face positions. 
# - The preexisitng face 128D vector with the shortest distance to the 128D vector of the face on screen is most likely a match
# (Distance threshold is 0.6, percentage threshold is 70%)
#     

# # Generate Face DB

# In[ ]:


dict_faces = dict()
dict_faces["Jaime Lannister"] = ["https://s2.r29static.com//bin/entry/97f/340x408,85/1832698/image.jpg",
                                "https://upload.wikimedia.org/wikipedia/en/thumb/b/b4/Jaime_Lannister-Nikolaj_Coster-Waldau.jpg/220px-Jaime_Lannister-Nikolaj_Coster-Waldau.jpg",
                                 "https://upload.wikimedia.org/wikipedia/pt/thumb/0/06/Nikolaj-Coster-Waldau-Game-of-Thrones.jpg/220px-Nikolaj-Coster-Waldau-Game-of-Thrones.jpg",
                                 "https://purewows3.imgix.net/images/articles/2017_09/jaime-lannister-season-7-game-of-thrones-finale1.jpg?auto=format,compress&cs=strip&fit=min&w=728&h=404",
                                 "https://cdn.newsday.com/polopoly_fs/1.13944684.1502107079!/httpImage/image.jpeg_gen/derivatives/landscape_768/image.jpeg",
                                 "https://www.cheatsheet.com/wp-content/uploads/2017/08/Jaime-Lannister-Game-of-Thrones.png",
                                 "https://fsmedia.imgix.net/9c/c0/27/10/15e0/44a4/8ecb/9339993b563d/nikolaj-coster-waldau-as-jaime-lannister-in-game-of-thrones-season-7.png?rect=0%2C0%2C1159%2C580&dpr=2&auto=format%2Ccompress&w=650",
                                 "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQrIQuBKKUocAizwfWtIdhAcvfowLJatKqqDsO3ywYdh3rv-mBk"
                                ]


# In[ ]:


import cv2
import urllib
import numpy as np
import matplotlib.pyplot as plt

def read_image_from_url(url2read):
    req = urllib.request.urlopen(url2read)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1) # 'Load it as it is'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return(img)


# In[ ]:


# Check urls and print last image
for p in dict_faces.keys():
    urls = dict_faces[p]
    for url2read in urls:
        print(url2read)
        img = read_image_from_url(url2read)
plt.imshow(img)        


# # Detect Faces Using Haar Cascades

# In[ ]:


got_faces_url = "https://static.independent.co.uk/s3fs-public/thumbnails/image/2018/05/10/12/game-of-thrones-finale.jpg?w968h681"
img = read_image_from_url(got_faces_url)
plt.imshow(img)


# In[ ]:


os.chdir("/kaggle/working/")
get_ipython().system('wget https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml')
get_ipython().system('ls')


# In[ ]:


import cv2
#Create the haar cascade
face_cascade = cv2.CascadeClassifier('/kaggle/working/haarcascade_frontalface_default.xml')

def find_faces_in_image(orig_img, scaleFactor, minNeighbors, minSize, maxSize):
    orig_img_copy = orig_img
    gray = cv2.cvtColor(orig_img_copy, cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray) 
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,           
        scaleFactor=scaleFactor, 
        minNeighbors=minNeighbors,  
        minSize=minSize, 
        maxSize=maxSize 
    )
    
    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(orig_img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    plt.imshow(orig_img_copy)


scaleFactor = 1.3
minNeighbors = 5
minSize = (60, 60)   
maxSize = (70, 70)
find_faces_in_image(img.copy(), scaleFactor, minNeighbors, minSize, maxSize)


# ## Mini Assignemnt
# Change paramaters of (scaleFactor, minNeighbors, minSize, maxSize) above to find all faces in the GOT image,
# use information in: https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html

# # Face Detection in Video using Haar

# ## Mini assignment
# Complete the following find_faces_in_frame_of_video() function so we can do face derection on the video

# In[ ]:


def find_faces_in_frame_of_video(orig_img, scaleFactor, minNeighbors, minSize, maxSize):


# In[ ]:


# Fancy box drawing function by Dan Masek
# Code in: https://www.codemade.io/fast-and-accurate-face-tracking-in-live-video-with-python/
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
 
    # Top left drawing
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
 
    # Top right drawing
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
 
    # Bottom left drawing
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
 
    # Bottom right drawing
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness) 
    

def video_file_recog_haar():
    print("[INFO] Reading video file ...")
    video_file = get_file_path("Game of Thrones 7x07 - Epic Daenerys Dragonpit Entrance.mp4")
    print(video_file)
    if glob.glob(video_file):
        vs = cv2.VideoCapture(video_file); #get input from file
    else:
        print("file does not exist")
        return
    
    frame_width = int(vs.get(3))
    frame_height = int(vs.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    framerate = 10.0
    video_path = "/kaggle/working/"
    out = cv2.VideoWriter(video_path+'output_haar.mp4', fourcc, framerate, (frame_width,frame_height))
    
    frame_counter = 0
    t0 = time.time()
    while True:        
        ret,frame = vs.read();
        if ret:
            frame_counter+=1
            if frame_counter%(30/framerate)==0:
                min_face_size = 60 #min face size is set to 60x60
                rects = find_faces_in_frame_of_video(frame, 1.3, 5, (min_face_size,min_face_size), (70,70))
                print("number of faces found in frame " + str(frame_counter) + ":",len(rects))
                aligns = []
                positions = []
                for (i, rect) in enumerate(rects):
                    draw_border(frame, (rect[0],rect[1]), (rect[0] + rect[2],rect[1]+rect[3]), (255,255,255), 1, 10, 10)
                    cv2.putText(frame,"Unknown",
                                        (rect[0]-4,rect[1]-4),cv2.FONT_HERSHEY_SIMPLEX,0.35,
                                        (255,255,255),1,cv2.LINE_AA)


                out.write(frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        else:
            break
    
    elapsed_time = time.time() - t0
    print("[exp msg] elapsed time for going over the video: " + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    vs.release()
    out.release()
    cv2.destroyAllWindows()
    
    print()
    print("Done")


# In[ ]:


video_file_recog_haar()


# ## Generate Face DB Functions

# In[ ]:


def import_from_images():
    print()
    print("[INFO] Extracting data from images ...")
    data_set = dict()

    for new_name in dict_faces.keys():
        person_features = {"Left" : [], "Right": [], "Center": []};
        print("Extracting:",new_name)
        print("number of img files:",len(dict_faces[new_name]))
        person_imgs = get_person_imgs(dict_faces[new_name]) 
        if person_imgs is None:
            print("extraction of:",new_name, " failed")
            continue
        
        print("extracted person_imgs from:",new_name)

        for pos in person_imgs:
            person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
        data_set[new_name] = person_features;
        
    f = open('/kaggle/working/facerec_128D.txt', 'w+'); 
    f.write(json.dumps(data_set))
    
def augment_image(img):
    aug_images = []
    flip_img = cv2.flip(img, 1)  #https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html?highlight=flip#cv2.flip
    # for more example see https://github.com/aleju/imgaug
    aug_images.append(flip_img)
    return(aug_images)

def get_person_imgs(urls):
    person_imgs = {"Left" : [], "Right": [], "Center": []};
    person_imgs_count = {"Left" : 0, "Right": 0, "Center": 0};
    counter_break = 0
    while True:    
        for url2read in urls:  
            img = read_image_from_url(url2read) # ****** file = url2read
            if img is None:
                print("********************* image was not loaded ***********************")
                continue

            frames = [img]
            frames.extend(augment_image(img))   

            for frame in frames:
                if True:
                    rects, landmarks = face_detect.detect_face(frame, 40);  # min face size is set to 40x40
                    for (i, rect) in enumerate(rects):
                        aligned_frame, pos = aligner.align(160,frame,landmarks[i]);
                        person_imgs_count[pos]+=1
                        if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                            person_imgs[pos].append(aligned_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                else:
                    break
            
        if person_imgs_count["Left"] == 0 or person_imgs_count["Right"] == 0 or person_imgs_count["Center"] == 0:
            counter_break+=1
            if counter_break > 3:
                print(person_imgs_count)  
                return None
        else:
            break
                            
    print(person_imgs_count)    
    return(person_imgs)  


# In[ ]:


def video_file_recog():
    print("[INFO] Reading video file ...")
    video_file = "/kaggle/input/Game of Thrones 7x07 - Epic Daenerys Dragonpit Entrance.mp4"
    if glob.glob(video_file):
        vs = cv2.VideoCapture(video_file); #get input from file
    else:
        print("file does not exist")
        return
    
    frame_width = int(vs.get(3))
    frame_height = int(vs.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    framerate = 30.0
    video_path = "/kaggle/working/"
    out = cv2.VideoWriter(video_path+'output.mp4', fourcc, framerate, (frame_width,frame_height))
    
    frame_counter = 0
    unknown_counter = 0
    known_counter = 0
    t0 = time.time()
    while True:        
        ret,frame = vs.read();
        if ret:
            frame_counter+=1
            if frame_counter%(30/framerate)==0:
                min_face_size = 80 #min face size is set to 80x80
                rects, landmarks = face_detect.detect_face(frame,min_face_size);
                print("number of faces found in frame " + str(frame_counter) + ":",len(rects))
                aligns = []
                positions = []
                for (i, rect) in enumerate(rects):
                    aligned_face, face_pos = aligner.align(160,frame,landmarks[i])
                    if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                        aligns.append(aligned_face)
                        positions.append(face_pos)
                    else: 
                        print("Align face failed") #log        
                if(len(aligns) > 0):
                    features_arr = extract_feature.get_features(aligns)
                    recog_data = findPeople(features_arr,positions);              
                    print("recog_data", str(recog_data))
                    for (i,rect) in enumerate(rects):
                        shrtname = short_name(recog_data[i][0])
                        acc = round(recog_data[i][1],1)
                        if "Unknown" in recog_data[i][0]:
                            unknown_counter+=1
                            #draw bounding box for the face
                            draw_border(frame, (rect[0],rect[1]), (rect[0] + rect[2],rect[1]+rect[3]), (255,255,255), 1, 10, 10)
                            cv2.putText(frame,shrtname+"-"+str(round(recog_data[i][1],1))+"%",
                                        (rect[0]-4,rect[1]-4),cv2.FONT_HERSHEY_SIMPLEX,0.35,
                                        (255,255,255),1,cv2.LINE_AA)
                        else:
                            known_counter+=1
                            # draw a fancy border around the faces
                            draw_border(frame, (rect[0],rect[1]), (rect[0] + rect[2],rect[1]+rect[3]), (124,252,0), 1, 10, 10)  
                            #draw bounding box for the face
                            cv2.putText(frame,shrtname+"-"+str(round(recog_data[i][1],1))+"%",
                                        (rect[0]-4,rect[1]-4),cv2.FONT_HERSHEY_SIMPLEX,0.35,
                                        (124,252,0),1,cv2.LINE_AA)                        

            out.write(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        else:
            break
    
    elapsed_time = time.time() - t0
    print("[exp msg] elapsed time for going over the video: " + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    vs.release()
    out.release()
    cv2.destroyAllWindows()
    print("known_counter:",known_counter, "unknown_counter:",unknown_counter)
    
    print()
    print("Done")


# In[ ]:


def findPeople(features_arr, positions, thres = 0.6, percent_thres = 70):
    '''
    :param features_arr: a list of 128d Features of all faces on screen
    :param positions: a list of face position types of all faces on screen
    :param thres: distance threshold
    :return: person name and percentage
    '''
    f = open('/kaggle/working/facerec_128D.txt','r')
    data_set = json.loads(f.read());
    returnRes = [];
    for (i,features_128D) in enumerate(features_arr):
        result = "Unknown";
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions[i]];
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_128D)))
                if(distance < smallest):
                    smallest = distance;
                    result = person;
        percentage =  min(100, 100 * thres / smallest)
        if percentage <= percent_thres :
            result = "Unknown"
        returnRes.append((result,percentage))
    return returnRes

def short_name(name):
    name_split = name.split(" ")
    short_name = name_split[0]
    if len(name_split) > 1:
        short_name = short_name + "." + name_split[1][0]
    return(short_name)


# # RUN Network Based Detection and Recognition

# In[ ]:


model_path = '../input/model-20170512-110547.ckpt-250000' 

os.chdir("/kaggle/input/")
# initalize
FRGraph = FaceRecGraph();
aligner = AlignCustom();
extract_feature = FaceFeature(FRGraph,model_path)
face_detect = MTCNNDetect(FRGraph, scale_factor=2); #scale_factor, rescales image for faster detection


# ## Generate face database

# In[ ]:


import_from_images()


# In[ ]:


get_ipython().system('ls ../working/ -ashl')


# ## Recognition in video file

# In[ ]:


video_file_recog()


# In[ ]:


get_ipython().system('ls ../working/ -ashl')


# # Questions and Instructions
# 
# ## Haar Cascade
# - Change paramaters of (scaleFactor, minNeighbors, minSize, maxSize) to find all faces in the GOT image, using find_faces_in_image().
#     - gray is the input grayscale image.
#     - scaleFactor is the parameter specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid.
#     - minNeighbors is a parameter specifying how many neighbors each candidate rectangle should have, to retain it. A higher number gives 
#       lower false   positives.
#     - minSize is the minimum rectangle size to be considered a face.
#     - More help can be found in: https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
# - Which parameters did the best work?
# - complete function find_faces_in_frame_of_video() to run face detection using  video_file_recog_haar() on the GOT video
# -  Change paramaters of (scaleFactor, minNeighbors, minSize, maxSize) to find as many TRUE faces as possible in video using
#     video_file_recog_haar()
#     - To download output video file (output_haar.mp4): 1. commit notebook 2. goto to offline kernel page 3. in Output tab choose Download All
#     - Which parameters did the best work?
# - Check your parameters with framerate of 30 when you think it is good enough
# 
# ## MTCNN and Face Vector Search
# - RUN Network Based Detection and Recognition code
# - Download movie output.mp4 and check who was recognized and how many times?
# - Add more individulas to the database so you could recgnize more individuals in video (notice you need images with center, right, left angles)
# - Try to augment the images using the augment_image function, does that improves the accuracy?
#    - For help check opencv image manipulaions and https://github.com/aleju/imgaug
# - Try to change min_face_size and see if you can recgnize faces in more frames
# - How would you increase the accuracy of the recgnition?
# - How would you make the entire process run faster?

# In[ ]:




