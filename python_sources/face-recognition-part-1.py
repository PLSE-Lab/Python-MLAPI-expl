#!/usr/bin/env python
# coding: utf-8

# # Face recognition with Tensorflow Object Detection API (LFW dataset)

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Visiulazation
import matplotlib.pyplot as plt
#image processing
import cv2
#extracting zippped file
import tarfile
#systems
import os
print(os.listdir("../input"))


# This is the complete code for part 1 of [my Medium article](https://medium.com/@saidakbarp/real-time-face-recognition-tflite-3fb818ac039a). The complete notebook for the part 2 of my Medium article is [here](https://www.kaggle.com/saidakbarp/face-recognition-part-2/notebook).
# 
# Face recognition relies on the dataset that has been annotated with boxes. Manually annotating faces in each images can be time consuming and for large scale training, manual annotation is impractical. For this reason we will use available face annotation tools to annotate each image with boxes. Afterwards, we can move to Object detection training part.

# In[ ]:


#example
imgg="/kaggle/input/photos/ben.jpg"
celeb=cv2.imread(imgg)


# In[ ]:


def show_image(image):
    plt.figure(figsize=(8,5))
    #Before showing image, bgr color order transformed to rgb order
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    


# In[ ]:


show_image(celeb)


# In[ ]:


# Our face detection function that uses haarcascade from OpenCV
def face_detection(img):
    face_cascade = cv2.CascadeClassifier('/kaggle/input/haarcascades/haarcascade_frontalface_alt.xml')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    print('Number of faces detected:', len(faces))
        
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #img = img[y:y+h, x:x+w] # for cropping
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv_rgb


# In[ ]:


#imgg2=cv2.imread("/kaggle/input/photos/ben.jpg")
a=face_detection(celeb)
plt.imshow(a)
plt.show() 

# as shown below, the library is not detecting this particular face angle of Ben Afflek


# In[ ]:


plt.figure(figsize=(15,18))
img=cv2.imread("../input/photos/people.jpg")
c=face_detection(img)
plt.imshow(c)
plt.show()


# Haarcascade has a con: it can not detect non-frontal face images and also boxes sometimes do not include full face, clipping chins or forehead. Let us try a better model: openCV DNN

# In[ ]:


# using openCV DNN
# load the model
#modelFile = "../input/opencv-dnn/opencv_face_detector_uint8.pb" 
#configFile = "../input/opencv-dnn/opencv_face_detector.pbtxt"
#net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

modelFile ="../input/opencvdnnfp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "../input/opencvdnnfp16/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


# In[ ]:


#function to extract box dimensions
def face_dnn(img, coord=False):
    blob = cv2.dnn.blobFromImage(img, 1, (224,224), [104, 117, 123], False, False) #
    # params: source, scale=1, size=300,300, mean RGB values (r,g,b), rgb swapping=false, crop = false
    conf_threshold=0.8 # confidence at least 60%
    frameWidth=img.shape[1] # get image width
    frameHeight=img.shape[0] # get image height
    max_confidence=0
    net.setInput(blob)
    detections = net.forward()
    detection_index=0
    bboxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            
            if max_confidence < confidence: # only show maximum confidence face
                max_confidence = confidence
                detection_index = i
    i=detection_index        
    x1 = int(detections[0, 0, i, 3] * frameWidth)
    y1 = int(detections[0, 0, i, 4] * frameHeight)
    x2 = int(detections[0, 0, i, 5] * frameWidth)
    y2 = int(detections[0, 0, i, 6] * frameHeight)
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if coord==True:
        return x1, y1, x2, y2
    return cv_rgb


# In[ ]:


#gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# multiple faces needs increasing the size of image as well as multiple detections
def nfaces_dnn(img):
    blob = cv2.dnn.blobFromImage(img, 1.2, (1200,1200), [104, 117, 123], False, False) #
    # params: source, scale=1, size=300,300, mean RGB values (r,g,b), rgb swapping=false, crop = false
    conf_threshold=0.6 # confidence at least 60%
    frameWidth=img.shape[1] # get image width
    frameHeight=img.shape[0] # get image height
    net.setInput(blob)
    detections = net.forward()
  
    bboxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
                  
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv_rgb


# In[ ]:


a=face_dnn(celeb)
plt.imshow(a)
plt.show() 


# In[ ]:


img=cv2.imread("../input/photos/people.jpg")
c=nfaces_dnn(img)
plt.figure(figsize=(15,18))
plt.imshow(c)
plt.show()


# As output shows, although 'multiple faces' function does not register some faces from a group of people, DNN is able to detect non-frontal faces without clipping chin.

# Our main purpose is to use LFW people images that have at least 20 examples ( later, obtain 100 images of top 100 celebrities from Google image search), then crop their faces (single face photos). Afterwards, we train our mobilenetV2 coco model to recognize those celebrities. Final goal is to identify celebrities from their photos.

# ## Now we extract LFW for face detection

# Each picture is centered on a single face. The typical task is called Face Verification: given a pair of two pictures, a binary classifier must predict whether the two images are from the same person.
# 
# An alternative task, Face Recognition or Face Identification is: given the picture of the face of an unknown person, identify the name of the person by referring to a gallery of previously seen pictures of identified persons.
# 
# Both Face Verification and Face Recognition are tasks that are typically performed on the output of a model trained to perform Face Detection. The most popular model for Face Detection is called Viola-Jones and is implemented in the OpenCV library. The LFW faces were extracted by this face detector from various online websites.

# In[ ]:


#os.listdir('../input/lfwpeople/')
fname='../input/lfwpeople/lfwfunneled.tgz'


# In[ ]:


def jpg_files(members): #only extract jpg files
    for tarinfo in members:
        if os.path.splitext(tarinfo.name)[1] == ".jpg":
            yield tarinfo
def untar(fname,path="LFW"): #untarring the archive
    tar = tarfile.open(fname)
    tar.extractall(path,members=jpg_files(tar))
    tar.close()
    if path is "":
        print("File Extracted in Current Directory")
    else:
        print("File Extracted in to",  path)


# In[ ]:


untar(fname,"LFW")


# In[ ]:


len(os.listdir('../working/LFW/lfw_funneled/')) # total number of folders (people)


# In[ ]:


# total number of images
total = sum([len(files) for r, d, files in os.walk('../working/LFW/lfw_funneled/')])
print(total)


# In[ ]:


count=0 
imglist=[]
# 
for r, d, files in os.walk('../working/LFW/lfw_funneled/'):   
    if len(files)>=20: 
        imglist.append(r)
        #print(count, r)
        count+=1 # counts how many folders have with at least 20 images
print(count)


# In[ ]:


#os.listdir(imglist[0])
# pick one random photo
a=np.random.randint(0,20)
b=np.random.randint(0,62)
imglist[b]


# In[ ]:


# show the random photo with face detected
img=imglist[b]+'/'+os.listdir(imglist[b])[a]
img=cv2.imread(img)
c=face_dnn(img)
plt.imshow(c)
plt.show()


# We have a list of people (62) with at least 20 images. Now, we will use this annotation of faces, and train MobileNet NN model to recognize those people.

# ## Creating training and test sets

# In[ ]:


#remove unused folders
import shutil
pathd='../working/LFW/lfw_funneled/'
#shutil.rmtree(os.path.realpath('LFW'))
for dirs in os.listdir(pathd):
    if not (pathd+dirs) in imglist:
        shutil.rmtree(os.path.realpath(pathd+dirs))


# In[ ]:


dirs=os.listdir(pathd)
dirs.sort()


# In[ ]:


#example of box coordinates
#a=np.random.randint(0,20)
b=np.random.randint(0,62)
for img in os.listdir(pathd+dirs[b])[:5]:
    #print(pathd+dirs[0]+'/'+img)
    print(dirs[b])
    img=cv2.imread(pathd+dirs[b]+'/'+img)
    x1, y1, x2, y2=face_dnn(img, True)
    #print coordinates of the detected face
    print(x1, y1, x2, y2)
    plt.imshow(img)
    plt.show()


# In[ ]:


os.listdir('../working/LFW/')
pathd
#(os.listdir(pathd))


# In[ ]:


# # creating test and train set
from numpy import random
datadir='../working/LFW/'
train=datadir+'train/'
test=datadir+'test/'

if not os.path.exists(train):
    os.mkdir(train)
if not os.path.exists(test):
    os.mkdir(test)
    
for dirs in os.listdir(pathd):
    filenames = os.listdir(pathd+dirs)
    filenames.sort()  # make sure that the filenames have a fixed order before shuffling
    random.seed(402)
    random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)
    split = int(0.85 * len(filenames))
    train_filenames = filenames[:split] # splitting filenames into two parts
    test_filenames = filenames[split:]
    for img in train_filenames:
        full_file_name = os.path.join(pathd+dirs, img)
        cur_dir=os.path.join(train+dirs)
        #print(cur_dir)
        if not os.path.exists(cur_dir): # create this current person's folder for training
            os.mkdir(cur_dir)
        shutil.copy(full_file_name, cur_dir)
    for img in test_filenames:
        full_file_name = os.path.join(pathd+dirs, img)
        cur_dir=os.path.join(test+dirs)
        if not os.path.exists(cur_dir): # create this current person's folder for testing
            os.mkdir(cur_dir)
        shutil.copy(full_file_name, cur_dir)
        #a=full_file_name+' '+test+dirs
shutil.rmtree('../working/LFW/lfw_funneled/')


# In[ ]:


# total number of images left
total = sum([len(files) for r, d, files in os.walk(datadir)])
print(total)


# In[ ]:


labeldir="Labels/" # labels dir
wdir="../working/LFW/"
lab=wdir+labeldir
if not os.path.exists(lab):
    os.mkdir(lab)


# In[ ]:


# function for creating box labels as txt file
def label_txt(pathdr, lab_dir):
    for fol in os.listdir(pathdr):
        tfile = open(lab_dir+fol+".txt","w+")
        for img in os.listdir(pathdr+fol):
            pathimg=os.path.join(pathdr+fol, img)
            #print(pathimg)
            pic=cv2.imread(pathimg)
            x1, y1, x2, y2=face_dnn(pic, True) # face detection and then saving into txt file       
            tfile.write(img+' '+str(x1)+' '+str(x2)+' '+str(y1)+' '+str(y2)+'\n')          
        tfile.close()
    print('Saved')


# In[ ]:


lab_dir=lab+'train/'
os.mkdir(lab_dir)
label_txt(train, lab_dir)

lab_dir=lab+'test/'
os.mkdir(lab_dir)
label_txt(test, lab_dir)


# In[ ]:


#let's check if txt files are correct:
f2 = open("../working/LFW/Labels/test/Arnold_Schwarzenegger.txt","r")
print(f2.read())
f2.close()


# In[ ]:


# checking Arnold_Schwarzenegger_0041.jpg
pic=cv2.imread(test+'Arnold_Schwarzenegger/Arnold_Schwarzenegger_0041.jpg')
cv2.rectangle(pic,(77 ,62),(176,194),(255,255,0),2)
plt.imshow(pic)
plt.show()


# As we can see box annotation is correct! Now, we have to save annotations as tfrecord:

# # Creating tfrecords with annotated images

# In[ ]:


def read_txt(person, photo):  
    txtfile = labels+person+".txt"
    txtfile_contents = open(txtfile, "r")
    txtlines = txtfile_contents.readlines()
    txtfile_contents.close()
    for line in txtlines:
        if photo in line:
            txtlines=line
    return txtlines


# In[ ]:


get_ipython().system('pip install object_detection >obj_dtc.txt # installing object detection')


# In[ ]:


from PIL import Image
from object_detection.utils import dataset_util

# flags = tf.app.flags
# flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
# FLAGS = flags.FLAGS
# modified from source: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
def create_tf_example(photo, person, iclass, foldr):
    # one image at a time
    img_f=os.path.join(foldr+person,photo+".jpg")
    pic = Image.open(img_f)
    height = pic.height # Image height
    width = pic.width # Image width
    filename = str.encode(photo) # Filename of the image. Empty if image is not from file
    #encoded_image_data = None # Encoded image bytes
    image_data = tf.gfile.GFile(img_f,'rb').read()
    
    image_format = b'jpeg' #None #  or b'png'
    #declare coordinates
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    
    txtlines = read_txt(person, photo)

    labels = txtlines.split()

    xmins.append(float(labels[1])/width)
    xmaxs.append(float(labels[2])/width)
    ymins.append(float(labels[3])/height)
    ymaxs.append(float(labels[4])/height)

    classes_text.append(str.encode(person))
    classes.append(iclass) #### iterator is needed
    #print(xmins, xmaxs, ymins, ymaxs, classes_text, photo, img_f) # for test purposes
    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


# In[ ]:


import tensorflow as tf
#saving tfrecords
def save_tf(folder):
    tf_file=folder.split('/')[-2] +'.tfrecord'
    writer = tf.python_io.TFRecordWriter('../working/'+tf_file)
    
    labelmap = '../working/'+'object_label.pbtxt' # for model training
    txtf = open(labelmap, "w")
    
    labels = '../working/'+'labels.txt' # for android deployment
    txtl = open(labels, "w")
    
    for ind, person in enumerate(os.listdir(folder)):
        iclass=ind+1
        txtf.write("item\n{\n  id: %s\n  name: '%s'\n}\n"%(iclass,person))
        txtl.write("%s\n"%person)
        #print(iclass, person)
        for photo in os.listdir(folder+person):
            tf_example = create_tf_example(photo.split('.')[0], person, iclass, folder) #004.jpg, arnold, 1
            #print('Folder:', pathd+fol, iclass)
            writer.write(tf_example.SerializeToString())
    txtf.close()
    writer.close()


# In[ ]:


labels='../working/LFW/Labels/train/'
save_tf(train)

labels='../working/LFW/Labels/test/'
save_tf(test)


# In[ ]:


os.stat('../working/train.tfrecord').st_size/1024/1024


# In[ ]:


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
make_tarfile('test_images.tar.gz','/kaggle/working/LFW/test')        
shutil.rmtree('/kaggle/working/LFW/')
#os.listdir('/kaggle/working/LFW/test')


# In[ ]:


os.listdir('/kaggle/working/')


# In[ ]:


# for test purposes
# pic = Image.open('../working/LFW/test/Jennifer_Capriati/Jennifer_Capriati_0007.jpg')
# height = pic.height # Image height
# width = pic.width # Image width
# pic=cv2.imread('../working/LFW/test/Jennifer_Capriati/Jennifer_Capriati_0007.jpg')
# cv2.rectangle(pic,(int(0.304*width),int(0.236*height)),(int(0.692*width),int(0.768*height)),(255,255,0),2)
# plt.imshow(pic)
# plt.show()


# Our tfrecords and labels are ready for model training. [Part 2](https://www.kaggle.com/saidakbarp/face-recognition-part-2/notebook) shows how to train an existing model and convert it to tflite directly in jupyter notebook.
