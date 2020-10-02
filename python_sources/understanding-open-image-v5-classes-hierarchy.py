#!/usr/bin/env python
# coding: utf-8

# # Exploring classes and Heirarchy 
# 
# ![class Hierarchy](https://0x0.st/zfd_.png)

# If you have been messing up your brains trying to understand the class hierarchies, or just beggining with it, I believe that this kernel is going to help you as I am sharing all my findings so far here. If you have no idea about what I am talking, visit [bbox_labels_600_hierarchy_visualizer](https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy_visualizer/circle.html) page and try to understand what exactly is going on.
# 
# If you feel that I am wrong anywhere, feel free to comment below and help improvise this kernel. 

# > These annotation files cover the 600 boxable object classes, and span the 1,743,042 training images where we annotated bounding boxes, object segmentations, and visual relationships, as well as the full validation (41,620 images) and test (125,436 images) sets.

# ### Downloading the required files
# 
# I am using the annotation files and the class_names files. So download them from the given link.

# In[ ]:


# Downloading the hierarchy json
# !wget https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json # --> old link
get_ipython().system('wget https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-label500-hierarchy.json')

# Downloading class names
get_ipython().system('wget https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv')
    
# Downlaoding class-annotations
get_ipython().system('wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv')


# In[ ]:


import pandas as pd
cls=pd.read_csv('class-descriptions-boxable.csv', header=None)
classes2name={i:j for i,j in zip(cls[0], cls[1])}
name2classes={j:i for i,j in zip(cls[0], cls[1])}
cls.tail()


# In[ ]:


trn=pd.read_csv('train-annotations-bbox.csv')
trn.head()


# Lets now print the classes from the Attached JSON given with the dataset. 
# Here you can understand the heirarchy level as per the name indentation.

# In[ ]:


import json
hier = json.load(open('challenge-2019-label500-hierarchy.json','r'))
level1=[]
level2=[]
level3=[]
for l2 in hier['Subcategory']:
    print(classes2name[l2['LabelName']])
    level3.append(classes2name[l2['LabelName']])
    try:
        for j in l2['Subcategory']:
            print('----> ',classes2name[j['LabelName']])
            level2.append(classes2name[j['LabelName']])
            try:
                for k in j['Subcategory']:
                    print('\t----> ',classes2name[k['LabelName']])
                    level1.append(classes2name[k['LabelName']])
            except:
                pass
    except:
        pass
        
level1 = set(level1)
level2 = set(level2)
level3 = set(level3)


# ### Classes count
# 
# We can see that we obtain 3 levels of classes hierarchy. 

# In[ ]:


print('Classes count in level 1 is {}'.format(len(level1)))
print('Classes count in level 2 is {}'.format(len(level2)))
print('Classes count in level 3 is {}'.format(len(level3)))
print('Total unique class counts are {}'.format(len(level1)+len(level2)+len(level3)))


# So there are 577 unique classes that we obtain from the json provided to us. But there could be classes overlap. Lets run a quick check over this.

# In[ ]:


print('level1 and level2 overlaps = {}'.format(len(level2&level1)))
print('level2 and level3 overlaps = {}'.format(len(level2&level3)))
print('level1 and level3 overlaps = {}'.format(len(level3&level1)))


# ### Classes along with counts 
# 
# The index of the classes in a list along with the value count in the training dataset.

# In[ ]:


res=trn['LabelName'].value_counts()
trn_classes=[]
for idx, (i,j) in enumerate(zip(res.index, res)):
    trn_classes.append(classes2name[i])
    print('{} \t {} \t {}'.format(idx+1, classes2name[i], j))


# ### Classes that could be missing
# 
# We have seen that we are provided with 600 classes but we obtained lesser classes from the classes hierarchy JSON. So lets try to print the classes that could be missing.

# In[ ]:


all_training_classes = set(list(cls[1]))
all_json_classes = level1.union(level2).union(level3)
print('There are {} classes from JSON file and {} classes from training file'.format(len(all_json_classes), len(all_training_classes)))
print('Thus there are {} missing classes'.format(len(all_training_classes)-len(all_json_classes)))


# So the missing classes could be these.

# In[ ]:


all_training_classes-all_json_classes


# In[ ]:



