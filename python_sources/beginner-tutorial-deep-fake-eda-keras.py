#!/usr/bin/env python
# coding: utf-8

# # NOTE
# ### Please DO read this markdown carefully.
# This Kernal just the starter Kernal and only for beginners who aim to find the starting path in the video analysis using Deep Learning. It is a mixture of ideas from different public Kernals but aims to give the complete solution to the beginners who wander off to find out the complex code about what is happening in the kernals that have been written by the very proficient programmers.
# 
# I can not say everything again and again in every Kernal and I would feel bad about the fact if I get to know that some beginner is still unable to understand what is going on. So before you start here, even if you have some knowledge of NN and every other thing,
# 
# ### Please do [check out this kernal](https://www.kaggle.com/deshwalmahesh/bengali-ai-complete-beginner-tutorial-95-acc) about Keras, NN, Images and much more before you start because a lot of very very useful links have been given in the links there for beginners.

# # Problem Statement
# If you haven't been to the kernal suggest, please go there and check the links before you start.
# 
# For this problem we have been provided a dataset of videos that have been altered using [Deep Fakes](https://www.youtube.com/watch?v=gLoI9hAX9dw). Deep Fakes are alterded data either audio,video or image data by using Deep Neural Networks mostly (Encoder - Decoder) Structure. So what basically is Encoder- Decoder and how it works? To give this answer, let us suppose we have 2 two players one is very advanced in skills but lacks in physical aspect and the one is just the opposite. So what if in far future we are able to mutate the people by desire? Then I hope we are able to give the features to one another. This is what exactly Features and Deep Fakes works.
# 
# To give you the best idea about Features, we take example of different fruits. Each and evey fruit has different shape, size, weight, color and so on BUT...if we have enough different Fruits, can we classify or give similarities? Yes , we can. This is where features come into play. Even if fruits are not completely related, we can still group them somwhow, say by taste, origin, continent or something else.
# 
# In terms of images or videos (videos are just images playing with your brain), they resemble Fruits and our model ```just finds out the features``` somehow. It happens by the use of Encoders and Decoders. We train out computers to extract the features from two different things with the use of Encoders say A and B. Then when there are features given by the Encoders, we use the features of A to feed it to the Decoder of B. Complex? No. It is just like the mutation. We just swapped (one side to be precise) the features in the Encoded dimensions. It just just like Messi and Ronaldo getting trained under some strict conditions so either Ronaldo gets the agility and dribbling of Messi or Messi getting the physical aspect and power of Ronaldo. 

# # Approach
# In this tutorial we just want to start the journey in finding the Fakes by using a CNN model which I'll commit on the next version. This notebook describes every part BEFORE the Training about how to go till training.
# The idea is to extract the Frames from the videos, then detect Faces (because only faces are altered) and then combine some faces to Train as Fake or Real. Simple!!!!

# In[ ]:


pip install mtcnn


# MTCNN is an implementation of the [Research Paper](https://arxiv.org/abs/1604.02878) published on using a [Convolution Neural Network](https://www.youtube.com/watch?v=FmpDIaiMIeA) to detect Faces given in an image. It is  pretrained model to detect faces with a degree of confidence such as 0.9 or 90% confidence that there is a face. It can return multiple faces with each face's coordinates in the image array where it has found a face. You can plot a rectangle aroung the box or just crop the image to see the cropped face which is later in the tutorial. It also returns the eyes and nose coordinates. More about the use of MTCNN can be learned either from the [official documentation of the library](https://pypi.org/project/mtcnn/) or from [this website](https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/). These are very good resources. Please do go through them.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
from IPython.display import HTML
from base64 import b64encode
from tqdm import tqdm
from skimage.transform import resize
from skimage.metrics import structural_similarity
from keras.layers import Dense,Dropout,Conv2D,Conv3D,LSTM,Embedding,BatchNormalization,Input,LeakyReLU,ELU,GlobalMaxPooling2D,GlobalMaxPooling3D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from tensorflow import random as tf_rnd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


INPUT_PATH = '/kaggle/input/deepfake-detection-challenge/'
TEST_PATH = 'test_videos/'
TRAIN_PATH = 'train_sample_videos/'
PAD = 50 # padding for the copped face so that a little bit of around of face can be visible
SIZE = 128
BATCH_SIZE = 32
CASCADE_PATH = cv2.data.haarcascades # with this line, you do not need to download the XML files from web. Later.


# In[ ]:


SEED = 13
np.random.seed(SEED) # set random seed to get reproducible results
tf_rnd.set_seed(SEED) # tensor flow randomness remover
plt.style.use('seaborn-whitegrid') # just some fancy un useful stuff


# # Getting Files and EDA

# In[ ]:


# iterate through the directory to get all the file names and save them as a DataFrame. No need to pay attention to
train_files = []
ext = []
for _, _, filenames in os.walk(INPUT_PATH+TRAIN_PATH): # iterate within the directory
    for filename in filenames: # get all the files inside directory
        splitted = filename.split('.') # split the files as a . such .exe, .deb, .txt, .csv
        train_files.append(splitted[0]) # first part is name of file
        ext.append(splitted[1]) # second one is extension type

files_df = pd.DataFrame({'filename':train_files, 'type':ext})
files_df.head()


# In[ ]:


files_df.shape # 401 files


# In[ ]:


files_df['type'].value_counts() # 400 mp4 files and 1 json file


# In[ ]:


meta_df = pd.read_json(INPUT_PATH+TRAIN_PATH+'metadata.json') # We have Transpose the Df
meta_df.head()


# In[ ]:


meta_df = meta_df.T
meta_df.head()


# In[ ]:


meta_df.reset_index(inplace=True) # set the index as 0,1,2....
meta_df.rename(columns={'index':'names'},inplace=True) 
# rename the column which was first index but is currently named as 'index'
meta_df.head()


# In[ ]:


meta_df.isna().sum() # 77 original files are missing


# In[ ]:


meta_df['label'].value_counts().plot(kind='pie',autopct='%1.1f%%',label='Real Vs Fake')


# 80% files are FAKE and only 20% are REAL

# # Extracting and Processing Features

# In[ ]:


class VideoFeatures():
    '''
    Class for working with features related to videos such getting frames, plotting frames, playing videos etc
    '''
    
    def get_properties(self,filepath):
        '''
        returns the properties of a video file
        args:
            filepath: path of the video file
        out:
            num_frames: total number of frames in a video
            frame_rate: frames played per second
        '''
        cap = cv2.VideoCapture(filepath)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        return num_frames, frame_rate
        
    
    def get_frames(self,filepath,first_only=False, show=False):
        '''
        method for getting the frames from a video file
        args: 
            filepath: exact path of the video file
            first_only: whether to detect the first frame only or all of the frames
        out:
            frame: first frame in form of numpy array 
        '''
    
        cap = cv2.VideoCapture(filepath) 
        # captures the video. Think of it as if life is a movie so we ask the method to focus on patricular event
        # that is our video in this case. It will concentrate on the video
        
        
        if not first_only: # whether to get all the frames or not
            all_frames = []
            while(cap.isOpened()): # as long as all the frames have been traversed
                ret, frame = cap.read()
                # capture the frame. Again, if life is a movie, this function acts as camera
                
                if ret==True:
                    all_frames.append(frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): # break in between by pressing the key given
                        break
                else:
                    break
                    
        else:
            ret, all_frames = cap.read()
            if show:
                plt.imshow(cv2.cvtColor(all_frames, cv2.COLOR_BGR2RGB))
                # plot the image but the cv2 changes thge ordering to Blue,Green,Red than RGB so it converts the 
                # metrices to proper ordering
        
        
        cap.release()
        # release whatever was held by the method for say, resources and the video itself
        return all_frames
        
        
    def play_video(self, filepath):
        '''
        Method that uses the HTML inside Python to put a code in the Kernal so that there is a HTML page
        like code where the supported video can be played
        args:
            filepath: path of the file which you want to play
        '''
        
        video = open(filepath,'rb').read() # read video file
        dec_data = "data:video/mp4;base64," + b64encode(video).decode()
        # decode the video data in form of a sting. Funny! Video is now a string
        
        return HTML("""<video width=350 controls><source src="%s" type="video/mp4"></video>""" % dec_data)
        # embed the string as <video> tag in HTML Kernal so that it can be understood by HTML and can be played 
    
    


# ## HaarCascade

# Below is what was very new to me once I saw it. It is called HaarCascade. A HaarCascade is basically a classifier which is used to detect the object for which it has been trained for, from the source. So source is our image and it detects Faces and different features of faces like smile, side profile, eyes etc. HaarCascade is basically a XML file where there is a code written inside it by very tech savvy coders so that we do not have to do it again and again. It will detect the structure from the file. Each XML is trained for a different feature. As this is the first commit, I'll just keep it simple and when tuning the model, we will have tweak lots of parameters.
# To get insight about the working of HaarCascades I recommend you [this insightful blog](http://www.willberger.org/cascade-haar-explained/)

# In[ ]:


class FrameProcessor():
    '''
    class to process the images such as resizing, changing colors, detect faces from frames etc
    '''

    def __init__(self):
        '''
        Constructor where the data from OpenCV is used directly to find the Faces. 
        '''
        self.face_cascade=cv2.CascadeClassifier(CASCADE_PATH+'haarcascade_frontalface_default.xml')
        # XML file which has code for Frontal Face
        self.eye_cascade=cv2.CascadeClassifier(CASCADE_PATH+'haarcascade_eye.xml')
        # it extracts eyes
        
    
    def detect_face_eye(self,img,scaleFactor=1.3, minNeighbors=5, minSize=(50,50),get_cropped_face=False):
        '''
        Method to detect face and eye from the image
        args:
            img: image in the form of numpy array pixels
            scaleFactor: scale the image in proportion. indicates how much the image size is 
                         reduced at each image scale. A lower value uses a smaller step for downscaling.
            minNeighbors: int, number of Neighbors to select from. You know that the pixels at eyes are correlated 
                            with surrounding with pixels around the eye but not the 1000 pixels away at feet
            minSize: tuple. Smaller the face in the image, it is best to adjust the minSize value lower
            get_zoomed_face: Bin. Wheter to return the zoomed face only rather than the full image
        out:
            image with detected faces
        '''
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # convert image to Grayscale to make better use of resources
        
        faces = self.face_cascade.detectMultiScale(gray,scaleFactor=scaleFactor,
                                                   minNeighbors=minNeighbors,
                                                  minSize=minSize)
        # Return the face rectangle from the image
        
        if get_cropped_face:
            for (x,y,w,h) in faces:
                cropped_img = img[y-PAD:y+h+PAD, x-PAD:x+w+PAD] # slice the array to-from where the face(s) have been found
            return cropped_img
            
        
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            # draw a rectangle around the face with (0,0,255= Blue) color
        
            eyes = self.eye_cascade.detectMultiScale(gray,minSize=(minSize[0]//2,minSize[1]//2),
                                                     minNeighbors=minNeighbors)
            # eyes will always be inside a front profile. So it will reduce the TruePositive of finding other eyes
            
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)
                # draw a rectangle around the eyes with Green color (0,255,0)
        
        return img
        
        
        
    def plot_frame(self,img):
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
    
    def resize_frame(self,frame,res_w=256,preserve_aspect=True,anti_aliasing=True):
        '''
        resize the images according to desired width and height
        param:
            frame: numpy image pixels array
            rew_w: resize width default to 256
            preserve_aspect: preserve the aspect ratio in the frame. If not, the output will be a square matrix
            anti_aliasing: whether to apply or not
        out:
            resized numpy array
        '''
        
        res_h = res_w
        if preserve_aspect: # protect the aspect ratio even after the resizing
            aspect_ratio = frame.shape[0]/frame.shape[1]  # get aspect ratio
            res_h = res_w*aspect_ratio # set resulting height according to ratio
            
        return resize(frame,(res_h,res_w),anti_aliasing=anti_aliasing)
        
    
    def frames_similarity(self,frames,full=True, multichannel=True):
        '''
        Find the similarity between the consecutive frames based on a common scale
        param:
            frames: list of numpy pixel arrays
            full: whether to return full  structural similarity 
            multichannel: Bool. IF the images are Grayscale or RGB
            with_resize: Bool. Default True. whether to resize the frames before finding similarity
        '''
        sim_scores = []
        for i in tqdm(range(1, len(frames))): # tqdm shows a progress bar
            curr_frame = frames[i]
            prev_frame = frames[i-1]

            if curr_frame.shape[0] != prev_frame.shape[0]: 
                # different sizes of same images will be seen as two different images so we have to deal with this
                # so just resize the bigger image as the smaller one
                if  curr_frame.shape[0] > prev_frame.shape[0]:
                    curr_frame = curr_frame[:prev_frame.shape[0], :prev_frame.shape[0], :]
                else:
                    prev_frame = prev_frame[:curr_frame.shape[0], :curr_frame.shape[0], :]


            mean_ssim,_ = structural_similarity(curr_frame, prev_frame, full=full,multichannel=multichannel)
            # get mean similarity scores of the images 
            sim_scores.append(mean_ssim)
        
        return sim_scores
        


# In[ ]:


vf =  VideoFeatures() # instantiate both the classes to use later
fp = FrameProcessor()


# In[ ]:


vf.get_properties(INPUT_PATH+TRAIN_PATH+'cwrtyzndpx.mp4') # get properties of video


# 300 frames on total with 30 FPS

# In[ ]:


vf.play_video(INPUT_PATH+TRAIN_PATH+'cwrtyzndpx.mp4') # see the magic of HTML with Python


# In[ ]:


img = vf.get_frames(INPUT_PATH+TRAIN_PATH+'cwrtyzndpx.mp4',first_only=True,show=True)
# get first frame from image and display it too


# In[ ]:


detected_face = fp.detect_face_eye(img,minNeighbors=5,scaleFactor=1.3,minSize=(50,50))
# detect the faces form the image. Tweak the parameters to get the face if ace is not found
# it is difficult to tweak the parameters for every image and this is one the reasons there is need of MTCNN
fp.plot_frame(detected_face)


# In[ ]:


frames = vf.get_frames(INPUT_PATH+TRAIN_PATH+'cwrtyzndpx.mp4') # get all the frames
fp.plot_frame(frames[54]) # plot a random frame from the frames 


# In[ ]:


# plt.plot(range(1,len(frames)),fp.frames_similarity(frames))


# There is a huge dip in the frames similarity many times. It means either the whole picture has changes or there is some other issue. You can get the desired frame to look what has happened so that you can have better understanding of what and how it is happening.

# In[ ]:


zoomed_face = fp.detect_face_eye(frames[13],get_cropped_face=True,) # get cropped image array which has pixels of face
fp.plot_frame(zoomed_face)


# ## MTCNN

# In[ ]:


class MTCNNWrapper():
    '''
    Detect and show faces using MTCNN
    '''
    
    def get_face(self,mtcnn_obj,img):
        '''
        method to get face from an image
        args:
            img: image as numpy array
        out:
            rect: coordinates of rectangle(s) for multiple face(s)
        '''
        faces = mtcnn_obj.detect_faces(img)
        # dectect_faces returns a list of dicts of all the faces
        x, y, width, height = faces[0]['box'] 
        # faces return a list of dicts so [0] means first faces out of all the faces
        return faces
    
    
    def show_faces(self,img):
        '''
        Show faces on the original image as red boxes
        args:
            img: image as numpy array
        out: 
            None: plot the original image with faces inside red boxes
        '''
        
        faces = self.get_face(img)   # get the list of faces dict
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # plot the image and next modify the image
        ax = plt.gca() # get the context for drawing boxes
        # Get the current Axes instance on the current figure matching the given keyword args, or create one

        for result in faces: # faces returns a list of dicts of all the faces
            x, y, width, height = result['box'] # get coordinates of each box found
            rect = Rectangle((x, y), width, height, fill=False, color='red') # form rectangle at the given coordinates
            ax.add_patch(rect) # add that box to the axis/ current image
        plt.show() # plot the extra rectangles
        
        
    def get_cropped(self,img,show_only=False):
        '''
        get the cropped image only from detected face
        args:
            img: numpy image array
            show_only: whether to return cropped array or just plot the image. Default False
        out:
            numpy array of cropped image at the face
        '''
        faces = self.get_face(img)
        x, y, width, height = faces[0]['box'] # first face. Will add logic later to find the most significant face
        if show_only:
            plt.imshow(cv2.cvtColor(img[y-PAD:y+height+PAD, x-PAD:x+width+PAD], cv2.COLOR_BGR2RGB))
            return None
        else:
            return img[y-PAD:y+height+PAD, x-PAD:x+width+PAD]


# In[ ]:


img = VideoFeatures().get_frames(INPUT_PATH+TRAIN_PATH+'cwrtyzndpx.mp4',first_only=True)
mt_wrapper = MTCNNWrapper()
mt_wrapper.show_faces(img)


# In[ ]:


mt_wrapper.get_cropped(img,show_only=True) # show only. it does not return anything


# # Saving Data to Disk
# We are going to extract features (frames) from the Video file one by one, Extract Faces from MTCNN and then get the cropped face to save the images in Fake, Real directory. 

# In[ ]:


#os.makedirs('train_1') # run this line of code only once.
import shutil 
# shutil.rmtree('train_1') # just in case you want to delete the directory


# In[ ]:


x = '''
mt_wrapper = MTCNNWrapper()
frames_names = []
frames_labels = []
for i in tqdm(range(meta_df.shape[0])):
    filepath = INPUT_PATH+TRAIN_PATH+meta_df['names'][i]
    label = meta_df['label'][i]
    cap = cv2.VideoCapture(filepath)
    framerate = cap.get(5) # 5 means to get the framerarate, 3 means get width, 4 for height and so on
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
    count=0
    while(cap.isOpened()):
        frame_no = cap.get(1) # get frame number of current frame
        ret,frame = cap.read()
        if ret!=True:
            break
        if count%10==0:
            filename ='train_1/' + filepath.split('/')[-1].split('.')[-2]+"_%d.jpg" % count
            #frame = mt_wrapper.get_cropped(img)
            cv2.imwrite(filename, frame)
            frames_names.append(filename)
            frames_labels.append(label)
        count+=1

images_csv = pd.DataFrame({'image_name':frames_names,'label':frames_labels})
images_csv.to_csv('video_faces.csv')
'''


# In[ ]:


images_df = pd.read_csv('video_faces.csv')
#images_df[] = images_df['label'].apply(lambda x: x.split('/')[-1])
images_df.drop('Unnamed: 0',axis=1,inplace=True)
images_df['label'] = images_df['label'].apply(lambda x: x.split('/')[-1])
images_df.head()
images_df.to_csv('video_faces.csv')


# In[ ]:





# # Implementation
# Implementation is coming soon but the idea is to apply the given method for the very basic starting point model as:
# 1. Get the videos one by one
# 2. Extract all the ```N``` frames
# 3. Extract all the ```F``` faces from the frames and discrads the remaining ```
# 4. Develop an algorithm to detect only the significant face as there is only 1 altered face and there can be more than one face in the frame
# 5. Get ```f``` faces from each set of ```F``` frames and label those as Fake or Real
# 6. Resize these faces to save computational power and time
# 6. Train a very basic CNN
# 7. Predict using the steps 1-5 in continuation
# 
# Note: You can select ```significant``` frames to select ```k``` frames to save resources that will be used to find faces for all the ```N``` frames. Discard remaining ```N-k``` frames.
