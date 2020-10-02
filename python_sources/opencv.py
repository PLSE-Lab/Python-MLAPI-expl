#!/usr/bin/env python
# coding: utf-8

# # Using OpenCV and BVLC GoogleNet to identify objects

# In this project we will try to identify the objects from a set of images and videos. For this purpose we will use BVLC GoogleNet Caffe model which is already pre-trained on 1000 classes including animals and objects. So, we don't need to train any model. Let's start.

# imutils is necessary for changing the dimension of the images to generate it. It is not present Kaggle by default.Thus, downloading it first. 

# In[ ]:


get_ipython().system('pip install imutils')


# In[ ]:


# importing all the libraries
import numpy as np 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
import imutils


# ## Adding the files

# I have used a directory : "visual-yolo-opencv" to store images, videos and the caffe model. You can create your own folder to include all the files. The list of files are mentioned in the references at the bottom.

# In[ ]:


# reading a sample image and printing it's shape
path = "../input/visual-yolo-opencv/"
img = cv2.imread(path+"vehicles.jpg")
print(img.shape)


# Here, the image contians 2584 x 4536 pixels and it has 3 colors channel which is RGB (Red Green Blue).

# Cv2 works with BGR (BLue Green Red). Thus, any image given in RGB (Red Green Blue) format will be treated as BGR. An example is shown below. The actual image is the second one.

# In[ ]:


fig,axs = plt.subplots(1,2, figsize=[15,15])
axs[0].imshow(img)
axs[0].set_title("Wrong color channel: RGB image, read in BGR", fontsize = 15)
axs[0].axis('off')

correct_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

axs[1].imshow(correct_img)
axs[1].set_title("Correct color channel: RGB image changed to BGR", fontsize = 15)
axs[1].axis('off')

plt.show()


# ## Applying the model ##

# **Background info:**
# 
# Synset or synonyms set is a set of 1000 unique classes like tractor, tiger, shark etc.
# 
# Caffe model is calculating probability for all these 1000 classes based on their index values. It means that the results obtained after feeding the image to the model will contain the proabiblity of the image belonging to each of thses 1000 classes. 
# 
# However, the proability for let's say a shark's image will be higher for class 'shark' than the probability of shark's image for class 'tractor'. 
# 
# Thus we will have to choose top 3 or 5 probabilties out of these 1000. And based on the index returned we can find the correspondng class name.  
# 
# Some synset classes contains synonym names and their id too. We are storing only the first name in a seperate list. We can always access the class name based on index thus removing the id too.
# 
# The link for Synset is provided below.
# 
# Link : [Synset words](https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt)

# In[ ]:


rows = open('../input/visual-yolo-opencv/synset_words.txt').read().strip().split("\n")
classes = [ row [ row.find(" ") + 1:].split(",")[0] for row in rows]

print("Total number of classes are: ", len(classes))


# The following markdown shows the synset before and after removing synonym names and id

# In[ ]:


print("Before : ", rows[:5])
print("After : ", classes[:5])


# The next step is to read the network model stored in the caffe model.
# 
# The prototxt contains the deep learning network architecture in Json format. These information includes the layer type (eg: RELU), weights etc information. The Caffe model is the pre-trained model. Thus, we have the model and we have it's architecture. Thus we can use it to find the probabilities.
# 
# [BVLC GoogleNet Prototxt](https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/bvlc_googlenet.prototxt)
# 
# [BVLC Caffe model](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)

# In[ ]:


google_net = cv2.dnn.readNetFromCaffe('../input/visual-yolo-opencv/bvlc_googlenet.prototxt','../input/visual-yolo-opencv/bvlc_googlenet.caffemodel')


# Let's see how we can use the model now:

# In[ ]:


# cafemodel requires image dimension to be 224x224
# blob is a 4D tensor obtained from the image
# blob from image parameters: ( image = input image , scalefactor = if scaling the image, 
# size = size of output image, mean, swapRB = swapping RGB to BGR, crop, ddepth)
blob = cv2.dnn.blobFromImage(correct_img, 1, (224,224))

# feeding the blob as input to the network
google_net.setInput(blob)

# getting the 1000 probabilities
result = google_net.forward()

# printing the result for first 10 classes
print("Length of the result: ", len(result[0]))

# finding the max 3 probabilities after sorting all of them in descending order
index = np.argsort(result[0])[::-1][:3]

print("\n\nTop 3 probabilities index : ", index)

# based on the index, retrieve the classes from synset
print("\nTop 3 probabilties of classes based on retrieved index\n")
      
for (i,id) in enumerate(index):
    print("{}. {} : Probability {:.3}%".format(i+1, classes[id], result[0][id]*100) + "\n")


# Now let's define a function to reuse the same code for using it on multiple images and videos

# In[ ]:


def display_match(image):
    
    # txt to store the results
    txt=""    
    
    blob = cv2.dnn.blobFromImage(image, 1, (224,224))
    
    google_net.setInput(blob)

    result = google_net.forward()

    index = np.argsort(result[0])[::-1][:3]
    
    for (i,id) in enumerate(index):
        txt += "{}. {} : Probability {:.3}%".format(i+1, classes[id], result[0][id]*100) + "\n"
            
    return txt        


# In[ ]:


result = display_match(correct_img)

plt.figure(figsize=[10,10])
plt.imshow(correct_img)
plt.title(result)
plt.axis('off')
plt.show()


# Applying the function on multiple images:

# In[ ]:


img_list = ["beach.jpg","cycle.jpg","dog.jpg","elephant.jpg","tiger.jpg",'laptop.jpg']

for i in range(len(img_list)):
    img_list[i] = path + img_list[i]


# In[ ]:


fig,axs = plt.subplots(3,2, figsize=[15,15])
fig.subplots_adjust(hspace=.5)

count=0
for i in range(3):    
    for j in range(2):        
        new_img = cv2.imread(img_list[count])
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        txt = display_match(new_img)
        #cv2.putText(new_img, txt, (0, 25 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        
        axs[i][j].imshow(new_img)
        axs[i][j].set_title(txt, fontsize = 12)
        axs[i][j].axis('off')
        count+=1

plt.suptitle("Top 3 predictions shown in title", fontsize = 18)
plt.show()


# We can even type the predictions on the top of the image

# In[ ]:


img = cv2.imread(img_list[4])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

txt = display_match(img)
line = txt.split("\n")

for i in range(3):
    cv2.putText(img, str(line[i]), (10, 30 + 40*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)


plt.figure(figsize=[10,10])
plt.imshow(img)
plt.title("Writing probability on the image", fontsize = 15)
plt.axis('off')

plt.show()


# We have try to predict the images. Now let's try it for videos.

# [Video source: Reddit reaction GIFs](https://www.reddit.com/r/reactiongifs/comments/2wcxt8/mrw_i_go_to_post_a_joke_my_friend_told_me_to/)

# In[ ]:


vid = cv2.VideoCapture('../input/visual-yolo-opencv/computer.mp4')

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_computer.mp4',fourcc, 20.0, (640, 640))

if(vid.isOpened==False):
    print("Can't open the video file")
    
try:
    while(True):

        ret, frame = vid.read()
        if not ret:
            vid.release()
            out.release()
            print("Completed! Read all the frames.")
            break
            
        txt = display_match(frame)
        line = txt.split("\n")
        for i in range(3):
            cv2.putText(frame, str(line[i]), (10, 30 + 40*i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,0,0), 2)

        resized_frame = cv2.resize(frame,(640,640))
        out.write(resized_frame)
        plt.imshow(resized_frame)
        plt.show()
        
        clear_output(wait=True)

except KeyboardInterrupt:
    vid.release()
    print("Error in reading frames")


# The results are uploaded on youtube and then shown here

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo('EX7ULuSPpaY', width=800, height=450)


# Applying it on another video:
# 
# Video source: [A Group Of Young People In Discussion Of A Group Project: By Pressmaster on Pexels](https://www.pexels.com/video/a-group-of-young-people-in-discussion-of-a-group-project-3209298/)
# 

# In[ ]:


vid = cv2.VideoCapture('../input/visual-yolo-opencv/video.mp4')

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_study.mp4',fourcc, 20.0, (1800, 1800))

if(vid.isOpened==False):
    print("Can't open the video file")

try:
    while(True):
        
           
        ret, frame = vid.read()
        if not ret:
            vid.release()
            out.release()
            print("Completed! Read all the frames.")
            break
            
        txt = display_match(frame)
        line = txt.split("\n")
        for i in range(3):
            cv2.putText(frame, str(line[i]), (70, 300 + 250*i), cv2.FONT_HERSHEY_SIMPLEX, 8, (255,0,0), 25)

        resized_frame = cv2.resize(frame,(1800,1800))
        out.write(resized_frame)
        #plt.imshow(resized_frame)
        #plt.show()
        
        #clear_output(wait=True)

except KeyboardInterrupt:
    vid.release()
    print("Error in reading frames")


# Again, the results are uploaded on Youtube and shown here

# In[ ]:


YouTubeVideo('liUmyzGuc70', width=800, height=450)


# ## References:##
# 
# Files required for training:
# 
# 1. [Synset words](https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt)
# 2. [BVLC GoogleNet Prototxt](https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/bvlc_googlenet.prototxt)
# 3. [BVLC Caffe model](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
# 
# Images:
# 
# 1. [Photo by Nik Shuliahin on Unsplash](https://unsplash.com/photos/pGwXiFyB7JE)
# 2. [Photo by bennett tobias on Unsplash](https://unsplash.com/photos/xOVS_XU1eeA)
# 3. [Photo by AJ Robbie on Unsplash](https://unsplash.com/photos/BuQ1RZckYW4)
# 4. [Photo by Radek Grzybowski on Unsplash](https://unsplash.com/photos/eBRTYyjwpRY)
# 5. [Photo by Nick Karvounis on Unsplash](https://unsplash.com/photos/-KNNQqX9rqY)
# 6. [Photo by Anoir Chafik on Unsplash](https://unsplash.com/photos/2_3c4dIFYFU)
# 7. [Photo by Neil Thomas on Unsplash](https://unsplash.com/photos/tMkEedbhuDo)
# 
# Videos:
# 1. [A Group Of Young People In Discussion Of A Group Project: By Pressmaster on Pexels](https://www.pexels.com/video/a-group-of-young-people-in-discussion-of-a-group-project-3209298/)
# 2. [MRW I go to post a joke my friend told me to /r/jokes and figure out that's where he got the joke from.](https://www.reddit.com/r/reactiongifs/comments/2wcxt8/mrw_i_go_to_post_a_joke_my_friend_told_me_to/)
# 

# ## Thank You ##

# In[ ]:




