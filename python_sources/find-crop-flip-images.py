#!/usr/bin/env python
# coding: utf-8

# This kernel shows:
# * How to check image flip/crop with simple way
# * Flip/crop images are not inclueded leaderboard score calculation

# In[ ]:


import numpy as np
import matplotlib.pylab as plt
import pandas as pd

#from PIL import ImageDraw, Image
import cv2

from tqdm import tqdm


# ### average train images

# In[ ]:


train_df = pd.read_csv("../input/pku-autonomous-driving/train.csv")
print(train_df.shape)

train_df.head()


# In[ ]:


test_df = pd.read_csv("../input/pku-autonomous-driving/sample_submission.csv")
print(test_df.shape)

test_df.head()


# In[ ]:


train_average_img = np.zeros([2710, 3384, 3])
train_dir = "../input/pku-autonomous-driving/train_images/"

for imgid in tqdm(train_df["ImageId"]):
    color_img = cv2.imread(train_dir + imgid + ".jpg")
    train_average_img += color_img


# In[ ]:


train_average_img_ = (train_average_img / train_df.shape[0]).astype(np.uint8)
train_average_img_ = cv2.cvtColor(train_average_img_, cv2.COLOR_BGR2RGB)


# In[ ]:


plt.figure(figsize=(13, 13))
plt.imshow(train_average_img_)


# In[ ]:


plt.imsave("train_all_average.png",train_average_img_)


# ### template matching
# EV bonnet mark as a template

# In[ ]:


train_template = train_average_img_[2600:,2200:2800,:]
plt.imshow(train_template)


# In[ ]:


imgid = train_df.iloc[0]["ImageId"]
imgid


# In[ ]:


tmp_img = cv2.imread("../input/pku-autonomous-driving/train_images/" + imgid + ".jpg")
tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)

# for 
tmp_img = tmp_img[1500:,:,:]

plt.figure(figsize=(13, 13))
plt.imshow(tmp_img)


# In[ ]:


_, w,h = train_template.shape[::-1]

result = cv2.matchTemplate(tmp_img, train_template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(f"max value: {max_val}, position: {max_loc}")

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

tmp_img_rect = cv2.rectangle(tmp_img,top_left, bottom_right, [255,0,0], 3)

plt.figure(figsize=(13, 13))
plt.imshow(tmp_img_rect)


# If a image is not cropped and/or flipped, this "EV bonnet mark" should be found here.

# In[ ]:


imgid_list = []
x_list = []
y_list = []

test_dir = "../input/pku-autonomous-driving/test_images/"

for imgid in tqdm(test_df["ImageId"]):
    filename = test_dir+imgid+".jpg"
    
    color_img = cv2.imread(filename)
    
    color_img = color_img[2000:,1500:,:] # for matching speedup
    
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    color_img = cv2.copyMakeBorder(color_img,0,55,0,0,cv2.BORDER_REPLICATE) # for robust matching
    color_img = color_img.astype(np.uint8)
    
    result = cv2.matchTemplate(color_img, train_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    imgid_list.append(imgid)
    x_list.append(max_loc[0])
    y_list.append(max_loc[1])


# In[ ]:


matched_df = pd.DataFrame({
    "ImageId":imgid_list,
    "x":x_list,
    "y":y_list,
})


# In[ ]:


print(matched_df.shape)
matched_df.head()


# In[ ]:


plt.scatter(matched_df["x"],matched_df["y"])


# In[ ]:


plt.scatter(matched_df["x"],matched_df["y"])
plt.xlim(645,760)
plt.ylim(580,620)
plt.grid()


# In[ ]:


matched_df["no_crop_no_flip"] = (
    (matched_df["x"] >= 680) &
    (matched_df["x"] <= 720) & 
    (matched_df["y"] >= 592) & 
    (matched_df["y"] <= 608)).astype("int")


# In[ ]:


matched_df.head()


# flip and do same template matching

# In[ ]:


x_list = []
y_list = []

test_dir = "../input/pku-autonomous-driving/test_images/"

for imgid in tqdm(test_df["ImageId"]):
    
    if matched_df[matched_df["ImageId"] == imgid].iloc[0]["no_crop_no_flip"] ==1: # no need to match again
        x_list.append(-1)
        y_list.append(-1)
        continue
    
    filename = test_dir+imgid+".jpg"
    
    color_img = cv2.imread(filename)
    color_img = np.fliplr(color_img).copy()
    
    color_img = color_img[2000:,1500:,:] # for matching speedup
    
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    color_img = cv2.copyMakeBorder(color_img,0,55,0,0,cv2.BORDER_REPLICATE) # for robust matching
    color_img = color_img.astype(np.uint8)
    
    result = cv2.matchTemplate(color_img, train_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    x_list.append(max_loc[0])
    y_list.append(max_loc[1])


# In[ ]:


matched_df["flip_x"] = x_list
matched_df["flip_y"] = y_list


# In[ ]:


plt.scatter(matched_df["flip_x"],matched_df["flip_y"])
plt.xlim(645,760)
plt.ylim(580,620)
plt.grid()


# In[ ]:


matched_df["no_crop_flip"] = (
    (matched_df["flip_x"] >= 680) &
    (matched_df["flip_x"] <= 720) & 
    (matched_df["flip_y"] >= 592) & 
    (matched_df["flip_y"] <= 608)).astype("int")


# In[ ]:


matched_df.head()


# In[ ]:


print((matched_df["no_crop_no_flip"] == 1).sum())
print((matched_df["no_crop_flip"] == 1).sum())
print(((matched_df["no_crop_no_flip"] != 1) & (matched_df["no_crop_flip"] != 1)).sum())


# In 2021 test images, there are 1230 no-crop and no-flip images, and 246 no-crop and h-flip images.  
# Rest(545) images are cropped.

# ### visualize

# No crop, no flip images

# In[ ]:


fig = plt.figure(figsize=(17, 17))
num=1

for idx,row in matched_df[matched_df["no_crop_no_flip"] == 1].sample(16).iterrows():
    filename = test_dir+row["ImageId"] +".jpg"
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax = fig.add_subplot(4, 4, num)
    ax.set_aspect('equal')
    
    ax.imshow(img)
    num+=1


# no crop, but flipped images

# In[ ]:


fig = plt.figure(figsize=(17, 17))
num=1

for idx,row in matched_df[matched_df["no_crop_flip"] == 1].sample(16).iterrows():
    filename = test_dir+row["ImageId"] +".jpg"
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax = fig.add_subplot(4, 4, num)
    ax.set_aspect('equal')
    
    ax.imshow(img)
    num+=1


# cropped images

# In[ ]:


fig = plt.figure(figsize=(17, 17))
num=1

for idx,row in matched_df[(matched_df["no_crop_no_flip"] != 1) & (matched_df["no_crop_flip"] != 1)].sample(16).iterrows():
    filename = test_dir+row["ImageId"] +".jpg"
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax = fig.add_subplot(4, 4, num)
    ax.set_aspect('equal')
    
    ax.imshow(img)
    num+=1


# In[ ]:


matched_df.to_csv("matched_df.csv",index=None)


# ### check submission score
# https://www.kaggle.com/hocop1/centernet-baseline?scriptVersionId=23634825  
# Original Private Score:0.029  
# Original Public Score:0.027

# In[ ]:


orig_sub_df = pd.read_csv("../input/sample-submission/predictions.csv")
print(orig_sub_df.shape)


# In[ ]:


orig_sub_df["no_crop_flip"] = matched_df["no_crop_flip"]
orig_sub_df.head()


# In[ ]:


PredictionString_list=[]
for idx,row in orig_sub_df.iterrows():
    if row["no_crop_flip"] == 0:
        PredictionString_list.append(row["PredictionString"])
    else:
        PredictionString_list.append("") # blank prediction


# In[ ]:


test_df = pd.read_csv("../input/pku-autonomous-driving/sample_submission.csv")
print(test_df.shape)


# In[ ]:


test_df["PredictionString"] = PredictionString_list


# In[ ]:


(test_df["PredictionString"] == "").sum()


# In[ ]:


test_df[18:25]


# 246/2021 images are blank predictions,  
# then Public or Private score must be worse.
# 
# But Public and Private LB score remains same.

# You can try with your submission file.  
# Note: This template matching method is not perfect, so you may encount small score change.

# In[ ]:


test_df.to_csv('submission.csv', index=False)

