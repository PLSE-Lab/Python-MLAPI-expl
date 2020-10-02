#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os


# In[ ]:


get_ipython().system('ls -GFlash ../input/deepfake-detection-challenge')


# In[ ]:


import cv2
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


from PIL import Image, ImageDraw


# In[ ]:


os.listdir('/kaggle/input/deepfake-detection-challenge')


# In[ ]:


INPUT_FOLDER = '../input/deepfake-detection-challenge'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
TEST_FOLDER = 'test_videos'


# In[ ]:


print(f'No of training sample videos : {len(os.listdir(os.path.join(INPUT_FOLDER, TRAIN_SAMPLE_FOLDER)))}')
print(f'No of test videos : {len(os.listdir(os.path.join(INPUT_FOLDER, TEST_FOLDER)))}')


# **Let's check the extensions**

# In[ ]:


train_files = os.listdir(os.path.join(INPUT_FOLDER, TRAIN_SAMPLE_FOLDER))
all_extensions = []
for file in train_files:
    ext = file.split('.')[1]
    if ext not in all_extensions:
        all_extensions.append(ext)


# In[ ]:


print(f'All extensions in Train folder are - {all_extensions}')


# **JSON file contains metadata. Let's explore it**

# In[ ]:


json_file = [file for file in train_files if file.endswith('.json')][0]
print(f'Metadata filename is {json_file}')


# In[ ]:


df_metadata = pd.read_json(os.path.join(INPUT_FOLDER, TRAIN_SAMPLE_FOLDER, json_file)).T


# In[ ]:


df_metadata


# In[ ]:


plt.figure(figsize = (13,8))
ax = sns.countplot(df_metadata['label'])
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}\n({p.get_height()/df_metadata.shape[0] * 100:.1f}%)', xy = (p.get_x() + p.get_width()/2., p.get_height()), ha = 'center', xytext = (-10, 5), textcoords = 'offset points')
ax.set_ylim(0, .9 * df_metadata.shape[0])
plt.xlabel('Video Type', fontsize = 14)
plt.ylabel('Count', fontsize = 14)
plt.title('Distribution of Video type', fontsize = 14)
plt.show()


# **Check Missing Data**

# In[ ]:


total = df_metadata.isnull().sum()
count = df_metadata.isnull().count()
percent_missing = (total/count * 100)
print(f'Original videos which are missing are {total.original} which is {percent_missing.original}% of the total')


# **This number is exactly same as number of REAL videos. Does it mean that all Real label videos do not have original videos?
# Let us confirm our suspicion**

# In[ ]:


len(df_metadata[df_metadata['label'] == 'REAL'])


# In[ ]:


df_metadata.original.nunique()


# **Out of total 323 original videos (which are all FAKE) only 209 are unique**

# In[ ]:


df_metadata.original.value_counts()


# In[ ]:


def grab_image_from_video(filename):
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    cv2.destroyAllWindows()
    return frame


# In[ ]:


def plot_frame(filename, type):
    frame = grab_image_from_video(filename)
    fig = plt.figure(figsize = (13,8))
    ax = fig.add_subplot(111)
    ax.imshow(frame)
    ax.axis('off')
    ax.set_title(f'Screen grab of {type} Video - {filename.split("/")[-1]}')


# **Let's plot some Fake samples**

# In[ ]:


fake_videos = list(df_metadata[df_metadata['label'] == 'FAKE'].sample(3).index)


# In[ ]:


for fake_video in fake_videos:
    plot_frame(os.path.join(INPUT_FOLDER, TRAIN_SAMPLE_FOLDER, fake_video), 'FAKE')


# **Some Real videos too**

# In[ ]:


real_videos = list(df_metadata[df_metadata['label'] == 'REAL'].sample(3).index)


# In[ ]:


for real_video in real_videos:
    plot_frame(os.path.join(INPUT_FOLDER, TRAIN_SAMPLE_FOLDER, real_video), 'REAL')


# **Detect faces using Haar Cascade classifier**

# In[ ]:


def detect_faces(image):
    face_cascade = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.2, 3)
    return faces


# In[ ]:


sample_videos = list(df_metadata.sample(3).index)


# In[ ]:


for video in sample_videos:
    screen_grab = grab_image_from_video(os.path.join(INPUT_FOLDER, TRAIN_SAMPLE_FOLDER, video))
    faces = detect_faces(screen_grab)
    for x, y, w, h in faces:
        cv2.rectangle(screen_grab, (x,y), (x+w, y+h), (255, 0, 0), 3)
    fig = plt.figure(figsize = (11, 11))
    ax = fig.add_subplot(111)
    image = cv2.cvtColor(screen_grab, cv2.COLOR_BGR2RGB)
    ax.axis('off')
    ax.imshow(image)


# **Thus for sideways looking subjects cascade classifier is not able to detect faces. Lets try another method of face detection using facial_recognition library and see if perfomance improves**

# In[ ]:


get_ipython().system('pip install face_recognition')


# **Detect face using face_recognition library and zoom on it**

# In[ ]:


import face_recognition
def detect_faces_using_face_recognition(image):
    face_locations = face_recognition.face_locations(image)
    print(f'Found {len(face_locations)} faces in image')
    fig, axs = plt.subplots(1, 2, figsize = (21, 5))
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].set_title('Original Image')
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        axs[1].imshow(face_image)
        axs[1].axis('off')
        axs[1].set_title('Detected Face')


# In[ ]:


sample_videos = list(df_metadata.sample(3).index)
for file in sample_videos:
    screen_grab = grab_image_from_video(os.path.join(INPUT_FOLDER, TRAIN_SAMPLE_FOLDER, file))
    detect_faces_using_face_recognition(screen_grab)


# In[ ]:


def draw_face_landmarks(image):
    img = Image.fromarray(image)
    d = ImageDraw.Draw(img)
    face_landmarks_list = face_recognition.face_landmarks(image)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            #print(f'{facial_feature} has points: {face_landmarks[facial_feature]}')
            d.line(face_landmarks[facial_feature], width = 3)
            
    display(img)


# In[ ]:


sample_videos = list(df_metadata.sample(3).index)
for file in sample_videos:
    screen_grab = grab_image_from_video(os.path.join(INPUT_FOLDER, TRAIN_SAMPLE_FOLDER, file))
    draw_face_landmarks(screen_grab)

