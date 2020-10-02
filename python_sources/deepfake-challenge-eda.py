#!/usr/bin/env python
# coding: utf-8

# # DeepFake Introductory EDA

# ## Introduction
# Today's technology allows us to do the incredible things such as creation of fake videos or images of real people, [deepfakes](https://en.wikipedia.org/wiki/Deepfake). Deepfakes [are going viral](https://www.creativebloq.com/features/deepfake-examples) and creating a lot of credibility and security concerns. That is why deepfake detection is a fast growing area of research (I put some of the papers related to deepfakes in the end of the notebook).
# 
# In this analysis I will try to look close at the videos from the sample dataset on Kaggle and find traits which can help us distinguish the fakes from the real videos.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import file utilities
import os
import glob

# import charting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation 
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import HTML

# import computer vision
import cv2
from skimage.measure import compare_ssim


# ## Load Data

# First of all, we need to declare the paths to train and test samples and metadata file:

# In[ ]:


TEST_PATH = '../input/deepfake-detection-challenge/test_videos/'
TRAIN_PATH = '../input/deepfake-detection-challenge/train_sample_videos/'

metadata = '../input/deepfake-detection-challenge/train_sample_videos/metadata.json'


# Look at the number of samples in test and train sets:

# In[ ]:


# load the filenames for train videos
train_fns = sorted(glob.glob(TRAIN_PATH + '*.mp4'))

# load the filenames for test videos
test_fns = sorted(glob.glob(TEST_PATH + '*.mp4'))

print('There are {} samples in the train set.'.format(len(train_fns)))
print('There are {} samples in the test set.'.format(len(test_fns)))


# And load the metadata:

# In[ ]:


meta = pd.read_json(metadata).transpose()
meta.head()


# In the metadata we have a reference to the original video, but those videos can't be found among the samples on Kaggle.
# 
# You can find the original videos if you download the whole dataset.

# Analyze the number or fake and real samples:

# In[ ]:


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'FAKE', 'REAL'
sizes = [meta[meta.label == 'FAKE'].label.count(), meta[meta.label == 'REAL'].label.count()]

fig1, ax1 = plt.subplots(figsize=(10,7))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=['#f4d53f', '#02a1d8'])
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Labels', fontsize=16)

plt.show()


# __Only 19% of samples are real videos.__ I don't know if this is the same for the whole dataset.

# ## Preview Videos and Zoom into Faces

# Let's start with looking at some frames of the videos and trying to look closer at the faces.
# I used [HAAR cascades](https://docs.opencv.org/trunk/db/d28/tutorial_cascade_classifier.html) to detect the areas containing the faces on the image.

# In[ ]:


def get_frame(filename):
    '''
    Helper function to return the 1st frame of the video by filename
    INPUT: 
        filename - the filename of the video
    OUTPUT:
        image - 1st frame of the video (RGB)
    '''
    # Playing video from file
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()

    # Our operations on the frame come here
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    return image

def get_label(filename, meta):
    '''
    Helper function to get a label from the filepath.
    INPUT:
        filename - filename of the video
        meta - dataframe containing metadata.json
    OUTPUT:
        label - label of the video 'FAKE' or 'REAL'
    '''
    video_id = filename.split('/')[-1]
    return meta.loc[video_id].label

def get_original_filename(filename, meta):
    '''
    Helper function to get the filename of the original image
    INPUT:
        filename - filename of the video
        meta - dataframe containing metadata.json
    OUTPUT:
        original_filename - name of the original video
    '''
    video_id = filename.split('/')[-1]
    original_id = meta.loc[video_id].original
    
    return original_id

def visualize_frame(filename, meta, train = True):
    '''
    Helper function to visualize the 1st frame of the video by filename and metadata
    INPUT:
        filename - video filename
        meta - dataframe containing metadata.json
        train - indicates that the video is among train samples and the label can be retrived from metadata
    '''
    # get the 1st frame of the video
    image = get_frame(filename)

    # Display the 1st frame of the video
    fig, axs = plt.subplots(1,3, figsize=(20,7))
    axs[0].imshow(image) 
    axs[0].axis('off')
    axs[0].set_title('Original frame')
    
    # Extract the face with haar cascades
    face_cascade = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_default.xml')

    # run the detector
    # the output here is an array of detections; the corners of each detection box
    # if necessary, modify these parameters until you successfully identify every face in a given image
    faces = face_cascade.detectMultiScale(image, 1.2, 3)

    # make a copy of the original image to plot detections on
    image_with_detections = image.copy()

    # loop over the detected faces, mark the image where each face is found
    for (x,y,w,h) in faces:
        # draw a rectangle around each detected face
        # you may also need to change the width of the rectangle drawn depending on image resolution
        cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)

    axs[1].imshow(image_with_detections)
    axs[1].axis('off')
    axs[1].set_title('Highlight faces')
    
    # crop out the 1st face
    crop_img = image.copy()
    for (x,y,w,h) in faces:
        crop_img = image[y:y+h, x:x+w]
        break;
        
    # plot the 1st face
    axs[2].imshow(crop_img)
    axs[2].axis('off')
    axs[2].set_title('Zoom-in face')
    
    if train:
        plt.suptitle('Image {image} label: {label}'.format(image = filename.split('/')[-1], label=get_label(filename, meta)))
    else:
        plt.suptitle('Image {image}'.format(image = filename.split('/')[-1]))
    plt.show()


# In[ ]:


visualize_frame(train_fns[0], meta)


# On this video the nose of the person is strange.

# In[ ]:


visualize_frame(train_fns[4], meta)


# The glasses of this don't look very realistic. There is also a strange rounded shape around the right eye of the lady. Strange white spot to the right of the mouth.

# In[ ]:


visualize_frame(train_fns[8], meta)


# The face of this person is so blurry.

# Let's also look at a couple of real images:

# In[ ]:


visualize_frame('../input/deepfake-detection-challenge/train_sample_videos/afoovlsmtx.mp4', meta)


# We can see that real faces have such details as:
# * actual teeth (not just one white blob);
# * glasses with reflections.

# In[ ]:


visualize_frame('../input/deepfake-detection-challenge/train_sample_videos/agrmhtjdlk.mp4', meta)


# This one will be really hard to predict! To my mind, the videos like these are garbage and should be removed from the training set.

# These fakes are really nice! Only small details tell that those are not real.

# ## Preview Multiple Frames

# Let's look at multiple frames:

# In[ ]:


def get_frames(filename):
    '''
    Get all frames from the video
    INPUT:
        filename - video filename
    OUTPUT:
        frames - the array of video frames
    '''
    frames = []
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read()
                
        if not ret:
            break;
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(image)

    cap.release()
    cv2.destroyAllWindows()
    return frames

def create_animation(filename):
    '''
    Function to plot the animation with matplotlib
    INPUT:
        filename - filename of the video
    '''
    fig = plt.figure(figsize=(10,7))
    frames = get_frames(filename)

    ims = []
    for frame in frames:
        im = plt.imshow(frame, animated=True)
        ims.append([im])

    animation = ArtistAnimation(fig, ims, interval=30, repeat_delay=1000)
    plt.show()
    return animation

def visualize_several_frames(frames, step=100, cols = 3, title=''):
    '''
    Function to visualize the frames from the video
    INPUT:
        filename - filename of the video
        step - the step between the video frames to visualize
        cols - number of columns of frame grid
    '''
    n_frames = len(range(0, len(frames), step))
    rows = n_frames // cols
    if n_frames % cols > 0:
        rows = rows + 1
    
    fig, axs = plt.subplots(rows, cols, figsize=(20,20))
    for i in range(0, n_frames):
        frame = frames[i]
        
        r = i // cols
        c = i % cols
        
        axs[r,c].imshow(frame)
        axs[r,c].axis('off')
        axs[r,c].set_title(str(i))
        
    plt.suptitle(title)
    plt.show()


# In[ ]:


frames = get_frames(train_fns[0])
visualize_several_frames(frames, step=50, cols = 2, title=train_fns[0].split('/')[-1])


# Static images don't look so bad, but if we look at the video (use the code for animation: `create_animation` function above) we see a lot of artifacts, which tell us that the video is fake.

# Now let's look closer at the person's face in motion:

# In[ ]:


def get_frames_zoomed(filename):
    '''
    Get all frames from the video zoomed into the face
    INPUT:
        filename - video filename
    OUTPUT:
        frames - the array of video frames
    '''
    frames = []
    cap = cv2.VideoCapture(filename)
    
    face_cascade = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_default.xml')

    while(cap.isOpened()):
        ret, frame = cap.read()
                
        if not ret:
            break;
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        faces = face_cascade.detectMultiScale(image, 1.2, 3)
        image_with_detections = image.copy()

        crop_img = image.copy()
        for (x,y,w,h) in faces:
            crop_img = image[y:y+h, x:x+w]
            break;
        
        frames.append(crop_img)

    cap.release()
    cv2.destroyAllWindows()
    return frames

def create_animation_zoomed(filename):
    '''
    Function to create the animated cropped faces out of the video
    INPUT:
        filename - filename of the video
    '''
    fig, ax = plt.subplots(1,1, figsize=(10,7))
    frames = get_frames_zoomed(filename)

    def update(frame_number):
        plt.axis('off')
        plt.imshow(frames[frame_number])

    animation = FuncAnimation(fig, update, interval=30, repeat=True)
    return animation


# In[ ]:


animation = create_animation_zoomed(train_fns[0])
HTML(animation.to_jshtml())


# We clearly see that the video is fake looking closer at the face! Some frames are really creepy. And there is flickering.

# In[ ]:


# visualize the zoomed in frames
frames_face = get_frames_zoomed(train_fns[0])
visualize_several_frames(frames_face, step=55, cols = 2, title=train_fns[0].split('/')[-1])


# Individual frames don't look too bad. This means that we have to build models using maximum frames, we can't just sample some frames. But we can use only frames containing faces to train the model.

# ## Explore the Similarity between Frames

# In[ ]:


def get_similarity_scores(frames):
    '''
    Get the list of similarity scores between the frames.
    '''
    scores = []
    for i in range(1, len(frames)):
        frame = frames[i]
        prev_frame = frames[i-1]
        
        if frame.shape[0] != prev_frame.shape[0]:
            if  frame.shape[0] > prev_frame.shape[0]:
                frame = frame[:prev_frame.shape[0], :prev_frame.shape[0], :]
            else:
                prev_frame = prev_frame[:frame.shape[0], :frame.shape[0], :]
        
        (score, diff) = compare_ssim(frame, prev_frame, full=True, multichannel=True)
        scores.append(score)
    return scores

def plot_scores(scores):
    '''
    Plot the similarity scores
    '''
    plt.figure(figsize=(12,7))
    plt.plot(scores)
    plt.title('Similarity Scores')
    plt.show()


# In[ ]:


scores = get_similarity_scores(frames)
plot_scores(scores)


# We can see that there are some similarity drops, let's try to look at the frames in this area:

# In[ ]:


max_dist = np.argmax(scores[1:50])
max_dist
plt.imshow(frames_face[max_dist])


# In[ ]:


plt.imshow(frames_face[max_dist+5])


# Let's compare similarity score with the original video (it is not among samples, I uploaded it in separate dataset):

# Open video and look at the first frame:

# In[ ]:


visualize_frame('../input/deepfake-utils/vudstovrck.mp4', meta, train = False)


# The difference between real and fake is quite clear. Just look at the nose.

# Let's get the frames and plot the similarity scores:

# In[ ]:


# get frames from the original video
orig_frames = get_frames('../input/deepfake-utils/vudstovrck.mp4')
# plot similarity scores
orig_scores = get_similarity_scores(orig_frames)
plot_scores(orig_scores)


# Plot similarity scores together:

# In[ ]:


plt.figure(figsize=(12,7))
plt.plot(scores, label = 'fake image', color='g')
plt.plot(orig_scores, label = 'real image', color='orange')
plt.title('Similarity Scores (Real and Fake)')
plt.show()


# The similarity scores of the original image are almost identical __if we take the whole frames__. Image similarity could still work if taking only frames containing faces. But I should fix the face detection first.

# ## Improvement
# 
# 1. Haar cascades don't seem to be very convenient tool. Setting up the parameters to match various images seems to be nearly manual. Consider using deep learning techniques, such as [MTCNN](https://github.com/ipazc/mtcnn) to detect faces.

# ## Deepfake Research Papers

# 1. [Unmasking DeepFakes with simple Features](https://arxiv.org/pdf/1911.00686v2.pdf): The method is based on a classical frequency domain analysis
# followed by a basic classifier. Compared to previous systems, which need to be fed with large amounts of labeled data, this
# approach showed very good results using only a few annotated training samples and even achieved good accuracies in fully
# unsupervised scenarios. [Github repo](https://github.com/cc-hpc-itwm/DeepFakeDetection)
# 
# 2. [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/pdf/1901.08971v3.pdf): This paper
# examines the realism of state-of- the-art image manipulations, and how difficult it is to detect them, either automatically
# or by humans. [Github repo](https://github.com/ondyari/FaceForensics)
# 
# 3. [In Ictu Oculi: Exposing AI Generated Fake Face Videos by Detecting Eye Blinking](https://arxiv.org/pdf/1806.02877v2.pdf): Method is based on detection of eye blinking in the videos,
# which is a physiological signal that is not well presented in the synthesized fake videos. Method is tested over
# benchmarks of eye-blinking detection datasets and also show promising performance on detecting videos generated with DeepFake.
# [Github repo](https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi)
# 
# 4. [USE OF A CAPSULE NETWORK TO DETECT FAKE IMAGES AND VIDEOS](https://arxiv.org/pdf/1910.12467v2.pdf): "Capsule-Forensics"
# method to detect fake images and videos. [Github repo](https://github.com/nii-yamagishilab/Capsule-Forensics-v2)
# 
# 5. [Exposing DeepFake Videos By Detecting Face Warping Artifacts](https://arxiv.org/pdf/1811.00656v3.pdf): Deep learning based
# method that can effectively distinguish AI-generated fake videos (referred to as DeepFake videos hereafter) from real videos.
# Method is based on the observations that current DeepFake algorithm can only generate images of limited resolutions, which
# need to be further warped to match the original faces in the source video. Such transforms leave distinctive artifacts in
# the resulting DeepFake videos, and we show that they can be effectively captured by convolutional neural networks (CNNs).
# [Github repo](https://github.com/danmohaha/CVPRW2019_Face_Artifacts)
# 
# 6. [Limits of Deepfake Detection: A Robust Estimation Viewpoint](https://arxiv.org/pdf/1905.03493v1.pdf): This work gives a
# generalizable statistical framework with guarantees on its reliability. In particular, we build on the information-theoretic
# study of authentication to cast deepfake detection as a hypothesis testing problem specifically for outputs of GANs,
# themselves viewed through a generalized robust statistics framework.

# ## Conclusion

# In this notebook:
# * I loaded saparate frames of the videos and sequences of video frames.
# * I created some animation of fake videos.
# * I used Haar cascades for face detection and zoomed into real and fake faces.
# * I looked at the similarity between frames.
# 
# Fake videos can be detected by:
# * Small missing details (nose, glasses, teeth),
# * Blurry contours of the face,
# * Flickering.
# 
# __Please, leave your comments and/or suggestions. I am really happy to hear from you!__
