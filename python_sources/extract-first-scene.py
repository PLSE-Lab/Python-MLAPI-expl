#!/usr/bin/env python
# coding: utf-8

# Thanks @Rustem Iskuzhin, for pointing out how to install Lyft sdk
# 
# 
# To get understanding of how each files are related with each other, I tried to extract a single scene.
# This way, it is also easier to load it. So some testing on the smaller dataset can be done, and then
# one can move on to the complete dataset
# 
# Prereq:
# - All required library is installed (specifically: lyft_dataset_sdk)
# - Virtual folder for images, data and lidar created pointing to train_images, train_data and train_lidar
# - All folders required on SAMP_DATA_PATH has been created
# 
# 
# 
# I have uploaded the single scene data as [sampData.zip](https://www.kaggle.com/aknirala/sampdata)

# In[ ]:


get_ipython().system('pip install lyft-dataset-sdk -q')


# In[ ]:


#First import:  
from lyft_dataset_sdk.lyftdataset import LyftDataset  #Assuming you have already installed it
import pandas as pd
import json


# In[ ]:


get_ipython().system('mkdir ./sampData')
get_ipython().system('mkdir ./sampData/train_data')
get_ipython().system('mkdir ./sampData/train_images')
get_ipython().system('mkdir ./sampData/train_lidar')
get_ipython().system('mkdir ./sampData/train_maps')


# In[ ]:


get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/test_images images')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/test_maps maps')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/test_lidar lidar')


# In[ ]:





# In[ ]:


#Some path initialization:
DATA_PATH = "/kaggle/input/3d-object-detection-for-autonomous-vehicles/"
wLoc = "./sampData/"


# In[ ]:


#Load all the data
#lyft_dataset = LyftDataset(data_path=DATA_PATH, json_path=DATA_PATH+'train_data')
lyft_dataset = LyftDataset(data_path=".", json_path=DATA_PATH+'train_data')


# In[ ]:


#We'll start from scene.json, first scene from this.
#scene.json:
f = open(wLoc +"train_data/" + "scene.json", "w")
json.dump([lyft_dataset.scene[0]], f)
f.close()

#Next 180s are in:log.json, 
#log.json: just write the one for scene[0]["map_token"] (1 entry)
f = open(wLoc +"train_data/" + "log.json", "w")
json.dump([lyft_dataset.get("log", lyft_dataset.scene[0]["log_token"])], f)
f.close()

#map.json: this is same for all so copy as it is (we can reduce but it's OK)


# In[ ]:


#sample.json  Loop from first_sample_token in scene
S = []
t = lyft_dataset.scene[0]["first_sample_token"]
while t is not None and t != "":
    S.append(lyft_dataset.get("sample", t))
    t = lyft_dataset.get("sample", t)["next"]
f = open(wLoc +"train_data/" + "sample.json", "w")
json.dump(S, f)
f.close()


# In[ ]:


#sample_data.json: from samples extracted 
#                    > from their data
#                       > extract each sample_data
SD = []
for s in S:
    for sName in s["data"]:
        SD.append(lyft_dataset.get("sample_data", s["data"][sName]))
f = open(wLoc +"train_data/" + "sample_data.json", "w")
json.dump(SD, f)
f.close()


# In[ ]:


#sample_annotation.json:create a set of sample tokens and exhaustively check.. and include
# What is the better way to do it?
sTokenSet = set()
for s in S:
    sTokenSet.add(s["token"])

SA = []
for sa in lyft_dataset.sample_annotation:
    if sa["sample_token"] in sTokenSet:
        SA.append(sa)

f = open(wLoc +"train_data/" + "sample_annotation.json", "w")
json.dump(SA, f)
f.close()


# In[ ]:


#instance.json: get instance_token from sample_annotation (need to get Set) and then extract
iSet = set()
for sa in SA:
    iSet.add(sa["instance_token"])

I = []
for t in iSet:
    I.append(lyft_dataset.get("instance", t))

f = open(wLoc +"train_data/" + "instance.json", "w")
json.dump(I, f)
f.close()


# In[ ]:


#ego_pose.json: Extract from sample_data
epSet = set()
for sd in SD:
    epSet.add(sd["ego_pose_token"])

#this reduced 1260 SDs to 632 ego poses
EP = []
for ep in epSet:
    EP.append(lyft_dataset.get("ego_pose", ep))

f = open(wLoc +"train_data/" + "ego_pose.json", "w")
json.dump(EP, f)
f.close()


# In[ ]:


#attribute.json: small copy as it is
#calibrated_sensor.json: small, copy as it is??
#category.json: copy as it is
#map.json:copy as it is as it is same for all (If we want we can prune list of 180 tokens to just one log in it)
# visibility.json: copy as it is
#sensor.json: copy as it is


# In[ ]:


#To copy imags and lidar we would rely on SD
#You might wanna modify the folder names
cpCommand = ""
for sd in SD:
    if sd["filename"][-4:] == "jpeg":
        cpCommand += "\ncp "+DATA_PATH+"train_"+sd["filename"]+ " " +wLoc + "train_images/"
    else:
        cpCommand += "\ncp "+DATA_PATH+"train_"+sd["filename"]+ " " +wLoc + "train_lidar/"

#print(cpCommand) #Copy paste it and run in suitabel folder. Then rename your folders


# In[ ]:


#Copy train_maps manually.

#For train.csv: Load all and filter thsoe which are needed
tr = pd.read_csv(DATA_PATH+"train.csv")
trSamp = tr.loc[tr["Id"].isin([s["token"] for s in S])]
trSamp.to_csv(wLoc+"train.csv")


# In[ ]:


#Create virtual links at wLoc. After stepping in wLoc do
"""
$ln -s train_images images
$ln -s train_lidar lidar
$ln -s train_maps maps
$ln -s train_data data

"""


# In[ ]:


#Now lets try to load our sample using lyft sdk
samp_lyft_dataset = LyftDataset(data_path=wLoc, json_path=wLoc+'train_data')
#This will fail as of now, coz files  have not been copied etc.,,.


# In[ ]:





# In[ ]:




