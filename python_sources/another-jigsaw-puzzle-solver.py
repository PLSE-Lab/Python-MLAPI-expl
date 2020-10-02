#!/usr/bin/env python
# coding: utf-8

# # 1. Motivation

# As noted by [@Peter](https://www.kaggle.com/pestipeti) in this [notebook](http://www.kaggle.com/pestipeti/jigsaw-puzzle-solved), the images from train dataset are 1024x1024 squared patches of the original images. For some classes we can reconstruct the original images by solving a jigsaw puzzle. For other classes, we have some patches missing se we can't reconstruct all the original images but we can get most of them.
# 
# The mentioned notebook by @Peter doesn't work for the (3x2) and for arvalis_3 class because all the images of this class are too similar to each other. My notebook works for all the classes including (3x2) and arvalis_3.

# ### Parameters

# My approach is configurable to some extent. You can run the script with different parameters to maybe get better results.

# ### Outputs

# There is an output with the merges images and the corresponding dataset with the bounding boxes. Since version 9 of the kernel, the bounding boxes of adjacent images that have large edges in common are merged.

# ### Known issues

# 1. After a visual inspection, all the results are correct except for two images in arvalis_3 and one in ethz_1. We must remove those images manually.
# 2. For arvalis_3, arvalis_1 and ethz_1, there are patches that couldn't be linked to any group. This is normal because we don't have all the patches for those two classes. Maybe with a better tuning of the parameters other groups could be found.

# # 2. Imports and definitions

# In[ ]:


import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
import ast
import os
import random
from IPython.display import clear_output
from datetime import datetime
from tqdm.notebook import tnrange
from shutil import copyfile


# In[ ]:


INPUT_DIR = '/kaggle/input/global-wheat-detection/'
OUTPUT_DIR = '/kaggle/working/global-wheat-detection/'
TEST_DIR = INPUT_DIR+'test/'
TRAIN_DIR = INPUT_DIR+'train/'
TRAIN_LABELS_FILE = INPUT_DIR+'train.csv'
PUZZLE_IMAGES_TRAIN_DIR = OUTPUT_DIR+'train_puzzle/'
PUZZLE_IMAGES_TRAIN_LABELS_FILE = OUTPUT_DIR+'train_puzzle.csv'


# In[ ]:


def get_ax(row=1, col=1, size=16, figsize=None):
    if figsize is None:
        _, ax = plt.subplots(row,col,figsize=(size,size))
    else:
        _, ax = plt.subplots(row,col,figsize=figsize)
    return ax

def display_instance(filename, bbox, ax):
    im = Image.open(filename)
    color=["red","yellow","blue","purple","green","black","white"]
    if type(bbox)!=list:
        lbbox=[bbox]
    else:
        lbbox=bbox
    for idx, bbox_list in enumerate(lbbox):
        for str_box in bbox_list:
            if type(str_box)==np.ndarray:
                box = str_box
            else:
                box = ast.literal_eval(str_box)
            #print(box)
            x1, y1, w, h = box
            p = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                  alpha=0.7, linestyle="solid",
                                  edgecolor=color[idx%len(color)], facecolor='none')
            ax.add_patch(p)
    ax.imshow(im)
    ax.set_title(filename)

def merge_images(candidate):
    shape = (2,3) if len(candidate)==6 else (1,2) if len(candidate)==2 else (2,2)
    im = np.zeros((1024*shape[0],1024*shape[1],3),dtype='uint8')
    for rows in range(shape[0]):
        for cols in range(shape[1]):
            im[rows*1024:(rows+1)*1024, cols*1024:(cols+1)*1024] = np.array(Image.open(TRAIN_DIR+candidate[rows*shape[1]+cols]+'.jpg'))
    return im

def get_images_in_group(solved):
    linked = []
    for candidate in solved:
        shift = parameters['shift'][candidate['dir_type']]
        linked.append([images[n] for n in np.array(candidate['nodes'])[shift]])
    return linked


# # 3. Data Frames

# In[ ]:


train_labels=pd.read_csv(TRAIN_LABELS_FILE)
bboxes = [ast.literal_eval(str_box) for str_box in train_labels['bbox'].values]
train_labels['x'] = [i[0] for i in bboxes]
train_labels['y'] = [i[1] for i in bboxes]
train_labels['w'] = [i[2] for i in bboxes]
train_labels['h'] = [i[3] for i in bboxes]
del train_labels['bbox']
del train_labels['width']
del train_labels['height']
train_labels.head()


# In[ ]:


train_all_images = pd.DataFrame({'count': train_labels.groupby(['image_id','source'])['image_id'].count()}).reset_index()
train_all_images = train_all_images.join(pd.DataFrame({'image_id':[f[:-4] for f in os.listdir(TRAIN_DIR)]}).set_index('image_id'), how='outer', on='image_id').reset_index(drop=True)
train_all_images['source'] = train_all_images['source'].fillna('empty')
train_all_images['count'] = train_all_images['count'].fillna(0)
train_all_images.head()


# # 4. Jigsaw Solver

# ### Set_config

# In config we have the parameters used by each of the classes. You can modify config['params'] to try to get better results. Do not modify the rest of the parameters.
# 
# Config['params'] is a list of tuples. We will run the algorithm for each tuple in the list. Each tuple contains:
# * Threshold: to define two borders as neighbors the distance between them must be less than this value.
# * Similarity acceptance: For the moment I always use 0.2. For a candidate to be accepted, the distance between edges must be 20% lower than for the second best candidate

# In[ ]:


def set_config(group_name):
    config = {}
    if group_name == 'rres_1':
        config['params'] = [(0.1,0.2),(0.2,0.2)]
    elif group_name == 'usask_1':
        config['params'] = [(0.2,0.2)]
    elif group_name == 'ethz_1':
        config['params'] = [(0.2,0.2),(0.25,0.2)]
    elif group_name == 'inrae_1':
        config['params'] = [(0.2,0.2)]
    elif group_name == 'arvalis_3':
        config['params'] = [(0.05,0.2),(0.1,0.2),(0.15,0.2),(0.2,0.2)]
    elif group_name == 'arvalis_2':
        config['params'] = [(0.15,0.2),(0.25,0.2)]
    elif group_name == 'arvalis_1':
        config['params'] = [(0.1,0.2),(0.15,0.2)]

    if group_name in ['arvalis_1', 'rres_1']:
        config['shape'] = (2,3)
        config['list_borders'] = [[0,0,1,2,2,3],[0,1,2,2,3,0],[1,2,2,3,0,0],[2,2,3,0,0,1],[2,3,0,0,1,2],[3,0,0,1,2,2]]
        config['shortcuts'] = [[[-1],[-1],[-1],[-1],[3,1],[3,0]], [[-1],[-1],[-1],[3,0],[-1],[0,0]], [[-1],[-1],[-1],[-1],[-1],[0,0,1,2]], [[-1],[-1],[-1],[-1],[1,1],[1,0]], [[-1],[-1],[-1],[1,0],[-1],[2,0]], [[-1],[-1],[-1],[-1],[-1],[2,0,3,2]]]
        config['nb_steps'] = 5
        config['shift'] = [[3,4,5,2,1,0],[2,3,4,1,0,5],[1,2,3,0,5,4],[0,1,2,5,4,3],[5,0,1,4,3,2],[4,5,0,3,2,1]]
    elif group_name in ['arvalis_2', 'arvalis_3', 'inrae_1']:
        config['shape'] = (2,2)
        config['list_borders'] = [[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]]
        config['shortcuts'] = [[[-1],[-1],[-1],[3,0]], [[-1],[-1],[-1],[0,0]], [[-1],[-1],[-1],[1,0]], [[-1],[-1],[-1],[2,0]]]
        config['nb_steps'] = 3
        config['shift'] = [[2,3,1,0],[1,2,0,3],[0,1,3,2],[3,0,2,1]]
    elif group_name in ['usask_1', 'ethz_1']:
        config['shape'] = (1,2)
        config['list_borders'] = [[0],[2]]
        config['shortcuts'] = [[[-1],[-1]],[[-1],[-1]]]
        config['nb_steps'] = 1
        config['shift'] = [[1,0],[0,1]]
    return config


# ### Read Images

# I read all the images of a group and return the following information:
# * color_signature: np_array with the colors of each of the edges of the image. We can set nb_avg to x get the average of the x pixels near the edge.
# * images: list with the name of the images
# * cache: a dictionary that I will use to avoid repeating calculation of distance between edges
# * used: an array with the images that were already linked into a group

# In[ ]:


def read_images(df, group_name):
    nb_avg = 1
    images = df[df['source']==group_name]['image_id'].values
    color_signature = np.zeros((4,len(images),1024,3))
    for idx, image_id in enumerate(images):
        image_filename = TRAIN_DIR+image_id+'.jpg'
        im_pil = Image.open(image_filename)

        #enhancer = ImageEnhance.Brightness(im_pil)
        #enhancer = ImageEnhance.Color(im_pil)
        #im_pil = enhancer.enhance(1.6)
        #enhancer = ImageEnhance.Brightness(im_pil)
        #im_pil = enhancer.enhance(1.1)

        im = np.array(im_pil)
        color_signature[0,idx] = im[:,:nb_avg].mean(axis=1)   # left border
        color_signature[1,idx] = im[:nb_avg,:].mean(axis=0)   # top border
        color_signature[2,idx] = im[:,-nb_avg:].mean(axis=1)  # right border
        color_signature[3,idx] = im[-nb_avg:,:].mean(axis=0)  # bottom border
    cache = {'cache': np.zeros((4,len(images),len(images))), 'is_cached':np.zeros((4,len(images)))}
    used = np.zeros(len(images), dtype='uint8')
    
    return color_signature, images, cache, used


# ### Border distance

# With an image and a direction, calculate the distance to all the other images of the group. If destination_node is not null, return the distance with another image, otherwise, return an array of distances with all the other images.
# I defined two distances measures: euclidean and selection of the channels (rgb) with max difference for each point and then euclidean. Any other distance can be used, like cosine distance. You just need to add another elif for distance_type

# In[ ]:


def border_distance(image_nb, direction, destination_node=None):
    distance_type = 'euclidean'
    if cache['is_cached'][direction,image_nb]==1:
        if destination_node is None:
            return cache['cache'][direction,image_nb]
        else:
            return cache['cache'][direction,image_nb,destination_node]
    else:
        border_1, border_2 = color_signature[direction,image_nb:image_nb+1], color_signature[(direction+2)%4,:]
        if distance_type=='euclidean':
            distance = np.sqrt(np.sum((border_1[None, :, :] - border_2[:, None, :])**2,axis=(-2,-1))[:,0]/(3*1024))/256
        elif distance_type=='max_channel+euclidean':
            distance = np.sqrt(np.sum(np.max((border_1[None, :, :] - border_2[:, None, :])**2,axis=-1),axis=-1)[:,0]/1024)/256
        cache['cache'][direction,image_nb] = distance
        cache['is_cached'][direction,image_nb] = 1
        if destination_node is None:
            return distance
        else:
            return distance[destination_node]


# ### Get Neighbors

# Given a candidate, get the distance with the five neighbors closest to it if this distance is lower than the accepted threshold

# In[ ]:


def get_neighbors(candidate, param_config):
    NB_NEIGHBOAR_MAX = 5
    image_nb = candidate['nodes'][-1]
    direction = candidate['next_direction']
    distance = border_distance(image_nb, direction)
    args = np.argsort(distance)
    args = np.array([x for x in args if x not in candidate['nodes']])
    dist_threshold = np.where(np.logical_and(distance[args]<param_config[0],used[args]==0))[0][:NB_NEIGHBOAR_MAX]
    neigh_distance = distance[args][dist_threshold]
    neigh_node = args[dist_threshold]
    return [{'distance': neigh_distance[i], 'node': neigh_node[i]} for i in range(len(dist_threshold))]


# ### Check shortcuts

# Check if two images are close enough to each other. This is useful for the (3x2) images to validate that the two images in the center are close enough.

# In[ ]:


def check_shortcuts(candidate, param_config):
    shortcuts = parameters['shortcuts']
    original_node = candidate['nodes'][-1]
    candidate_pos = len(candidate['nodes'])-1
    direction = shortcuts[candidate['dir_type']][candidate_pos][0]
    if direction>=0:
        destination_node = candidate['nodes'][shortcuts[candidate['dir_type']][candidate_pos][1]]
        distance = border_distance(original_node, direction, destination_node)
        candidate_res = candidate.copy()
        candidate_res['distance_acc'] += distance
        if len(shortcuts[candidate['dir_type']][candidate_pos])==2 or distance>=param_config[0]:
            return distance<param_config[0], candidate_res
        else:
            direction = shortcuts[candidate['dir_type']][candidate_pos][2]
            destination_node = candidate['nodes'][shortcuts[candidate['dir_type']][candidate_pos][3]]
            distance = border_distance(original_node, direction, destination_node)
            candidate_res['distance_acc'] += distance
            return distance<param_config[0], candidate_res
    else:
        return True, candidate


# ### Main process

# 1. We iterate over the list of parameters (*for config in parameters['params']*)
# 2. For each config, we iterate over all the images of the group (*for idx in tnrange(....*)
# 3. We build a list of candidates with all the images with a distance lower than the threshold
# 4. If at the end we get only one candidate we take it as a final image. If there are more than one, we take the best if the distance with the second one is big enough and we move to the next image

# In[ ]:


def solve_puzzle(group_name):
    print("Processing images for group",group_name)
    start_time = datetime.now()
    global parameters
    global color_signature
    global images
    global cache
    global used
    parameters = set_config(group_name)
    color_signature, images, cache, used = read_images(train_all_images,group_name)
    process_time = (datetime.now()-start_time).total_seconds()
    print("Start solving puzzle")
    solved=[]
    candidates = []
    nb_groups_found = 0
    start_time = datetime.now()
    #steps = 0
    for config in parameters['params']:
        threshold = config[0]
        for idx in tnrange(len(images),desc="threshold "+str(threshold),leave=False):
            if used[idx]==1:
                continue
            candidates = [{'dir_type': i, 'threshold': config[0], 'steps': [], 'distance_acc': 0.0, 'nodes': [idx], 'next_direction':parameters['list_borders'][i][0]} for i in range(len(parameters['list_borders']))]
            for i in range(parameters['nb_steps']):
                #steps += 1
                cur_candidates = candidates.copy()
                candidates = []
                for cur_candidate in cur_candidates:
                    neighbors = get_neighbors(cur_candidate, config)
                    for n in neighbors:
                        next_steps = cur_candidate['steps'].copy()
                        next_steps.append(cur_candidate['next_direction'])
                        next_nodes = cur_candidate['nodes'].copy()
                        next_nodes.append(n['node'])
                        new_candidate = {
                            'dir_type': cur_candidate['dir_type'], 
                            'threshold': cur_candidate['threshold'],
                            'steps': next_steps,
                            'distance_acc': cur_candidate['distance_acc']+n['distance'],
                            'nodes': next_nodes,
                            'next_direction': parameters['list_borders'][cur_candidate['dir_type']][len(cur_candidate['nodes'])] if len(cur_candidate['nodes'])<parameters['nb_steps'] else -1
                        }
                        shortcut_ok, new_candidate = check_shortcuts(new_candidate, config)
                        if shortcut_ok:
                            candidates.append(new_candidate)
            if len(candidates)==1:
                candidates = candidates[0]
                used[candidates['nodes']] = 1
            elif len(candidates)>1:
                candidates.sort(key=lambda x:x['distance_acc'])
                if candidates[0]['distance_acc']*(1+config[1])<candidates[1]['distance_acc']:
                    candidates = candidates[0]
                    used[candidates['nodes']] = 1
                else:
                    candidates = []
            if not candidates == []:
                solved.append(candidates)
            
    group_list = get_images_in_group(solved)
    clear_output(wait=True)
    #print("Images processed in",process_time,"seconds")
    #print("Puzzle solved in",(datetime.now()-start_time).total_seconds(),"seconds")
    #print("Groups found:",len(solved))
    #print("Images linked to a group:",len(solved)*parameters['shape'][0]*parameters['shape'][1],"out of",len(images))
    
    return group_list


# ### Run the puzzle solver for all the classes

# In[ ]:


groups = {}
classes = ['rres_1','inrae_1','ethz_1','arvalis_1','arvalis_2','arvalis_3','usask_1']
for c in classes:
    groups[c] = solve_puzzle(c)


# In[ ]:


for g in groups.keys():
    print("source:  ", g, "\t  length:", len(groups[g]), "\texamples:", groups[g][:2])


# If group is 'arvalis_3' or 'ethz_1', we remove the images we know that were incorrectly linked

# In[ ]:


for group_name in groups.keys():
    incorrect_groups = []
    if group_name == 'arvalis_3':
        incorrect_groups = [['88f3e7313','47a1184e4','f8d848989','218a99bee'],['dbd433d29','d688932d4','fabaeac81','2ad7fa68e']]
    elif group_name == 'ethz_1':
        incorrect_groups = [['8a702e7da', '02d662fa8']]
    groups[group_name] = [g for g in groups[group_name] if g not in incorrect_groups]


# In[ ]:


for g in groups.keys():
    print("source:  ", g, "\t  length:", len(groups[g]))


# Add non grouped images

# In[ ]:


for g in groups.keys():
    images = train_all_images[train_all_images['source']==g]['image_id'].values
    grouped_images = [im for gg in groups[g] for im in gg]
    non_grouped_images = [im for im in images if im not in grouped_images]
    for im in non_grouped_images:
        groups[g].append([im])


# In[ ]:


for g in groups.keys():
    print("source:  ", g, "\t  length:", len(groups[g]), "\t grouped:", len([a for a in groups[g] if len(a)>1]), "\t non-grouped:", len([a for a in groups[g] if len(a)==1]))


# # 5. Save images and dataset

# In[ ]:


if not os.path.exists(PUZZLE_IMAGES_TRAIN_DIR):
    os.makedirs(PUZZLE_IMAGES_TRAIN_DIR)
IMG_SIZE=1024
new_images = []
for k2 in tnrange(len(groups.keys()),leave=False):
    k = list(groups.keys())[k2]
    for g2 in tnrange(len(groups[k]),leave=False):
        g = groups[k][g2]
        if len(g)==1:
            new_images.append((k,g,g[0],g[0],1,0,0))
            copyfile(TRAIN_DIR+g[0]+'.jpg', PUZZLE_IMAGES_TRAIN_DIR+g[0]+'.jpg')
        else:
            if len(g)==2:
                im_np = np.zeros((IMG_SIZE,IMG_SIZE*2,3),dtype='uint8')
            elif len(g)==4:
                im_np = np.zeros((IMG_SIZE*2,IMG_SIZE*2,3),dtype='uint8')
            else:
                im_np = np.zeros((IMG_SIZE*2,IMG_SIZE*3,3),dtype='uint8')
            for idx, im in enumerate(g):
                if len(g)==2:
                    start_x, start_y = 1024*idx, 0
                elif len(g)==4:
                    start_x, start_y = 1024*(idx%2), 1024*(idx//2)
                else:
                    start_x, start_y = 1024*(idx%3), 1024*(idx//3)
                im_np[start_y:start_y+1024,start_x:start_x+1024] = np.array(Image.open(TRAIN_DIR+im+'.jpg'))
                new_images.append((k,g,"".join(g),im,len(g),start_x,start_y))
            Image.fromarray(im_np).save(PUZZLE_IMAGES_TRAIN_DIR+"".join(g)+'.jpg')


# In[ ]:


os.listdir(PUZZLE_IMAGES_TRAIN_DIR)[:10]


# In[ ]:


df_puzzle = pd.DataFrame(new_images, columns = ['source', 'image_group', 'final_image_id', 'image_id', 'image_len', 'start_x', 'start_y'])
df_puzzle.head()


# In[ ]:


train_labels.head()


# In[ ]:


train_labels_big_images = pd.merge(df_puzzle, train_labels, on=['source','image_id'])
train_labels_big_images['x'] += train_labels_big_images['start_x']
train_labels_big_images['y'] += train_labels_big_images['start_y']
train_labels_big_images['bbox_source']='original'
del train_labels_big_images['start_x']
del train_labels_big_images['start_y']
train_labels_big_images.head()


# ### display some random images with boxes

# In[ ]:


ax = get_ax(10,2,figsize=(30,120))
all_images = train_labels_big_images['final_image_id'].values
for i in range(20):
    random_image = np.random.choice(all_images)
    bboxes = train_labels_big_images[train_labels_big_images['final_image_id']==random_image][['x','y','w','h']].values
    display_instance(PUZZLE_IMAGES_TRAIN_DIR+random_image+'.jpg',bboxes,ax[i//2,i%2])


# ### Merge boxes in adjacent patches

# In[ ]:


def merge_boxes(bbox_1, bbox_2, axis):
    if axis==0: 
        offset=1
    else: 
        offset=0
    
    if bbox_1[offset]>=(bbox_2[offset]+bbox_2[offset+2]):
        return None
    if (bbox_1[offset]+bbox_1[offset+2])<=bbox_2[offset]:
        return None
    intersection = min(bbox_1[offset]+bbox_1[2+offset],bbox_2[offset]+bbox_2[2+offset])-max(bbox_1[offset],bbox_2[offset])
    union = max(bbox_1[offset]+bbox_1[2+offset],bbox_2[offset]+bbox_2[2+offset])-min(bbox_1[offset],bbox_2[offset])
    if (intersection/union)<0.6:
        return None
    return [min(bbox_1[0],bbox_2[0]),min(bbox_1[1],bbox_2[1]),max(bbox_2[0]+bbox_2[2]-bbox_1[0],bbox_1[0]+bbox_1[2]-bbox_2[0]),max(bbox_2[1]+bbox_2[3]-bbox_1[1],bbox_1[1]+bbox_1[3]-bbox_2[1])]


# This is the function to merge boxes. The coding is awful with lots of embeded loops. Probably it could be improved a lot, but since the dataset is small, it takes a reasonable time to finish.

# In[ ]:


def df_merge_boxes(df,axis, img_names):
    rows = []
    for img_name in img_names:
        nb_images = df[df['final_image_id']==img_name][['image_len']].values[0][0]
        row = df[df['final_image_id']==img_name].values.tolist()
        bboxes = df[df['final_image_id']==img_name][['x','y','w','h']].values.tolist()
        merged_boxes = []
        if nb_images == 1 or (nb_images==2 and axis==1):
            rows += row
        else:
            if nb_images==6 and axis==0:
                edges = [1024,2048]
            elif nb_images==6 and axis==1:
                edges = [1024]
            elif nb_images==4:
                edges = [1024]
            elif nb_images==2:
                edges = [1024]
            for idx, edge in enumerate(edges):
                lbox_1 = [b for b in bboxes if b[axis]+b[axis+2]==edge]
                lbox_2 = [b for b in bboxes if b[axis]==edge]
                #print(axis,idx,edge,len(lbox_1),len(lbox_2))
                for box1 in lbox_1:
                    for box2 in lbox_2:
                        b = merge_boxes(box1,box2,axis)
                        if b is not None:
                            row = [r for r in row if not (box1[0]==r[5] and box1[1]==r[6] and box1[2]==r[7] and box1[3]==r[8]) and not (box2[0]==r[5] and box2[1]==r[6] and box2[2]==r[7] and box2[3]==r[8])]
                            merged_boxes.append(row[0][:3]+[None]+[row[0][4]]+b+['merged'])
            rows += row
            rows += merged_boxes
    return pd.DataFrame(rows,columns = ['source','image_group','final_image_id','image_id','image_len','x','y','w','h','bbox_source'])


# Fix boxes in vertical axis and then in horizontal axis. Save the results in a dataset.

# In[ ]:


img_names = np.unique(train_labels_big_images['final_image_id'].values)

train_labels_big_images = df_merge_boxes(train_labels_big_images,0, img_names)
train_labels_big_images = df_merge_boxes(train_labels_big_images,1, img_names)


# Display the results. Boxes in yellow are merged boxes. In red the original ones.

# In[ ]:


ax = get_ax(2,2,figsize=(60,60))
all_images = np.random.choice(np.unique(train_labels_big_images[train_labels_big_images['image_len']>1]['final_image_id'].values),4)
print(all_images)
for i in range(len(all_images)):
    random_image = all_images[i]
    bboxes_original = train_labels_big_images[(train_labels_big_images['final_image_id']==random_image) & (train_labels_big_images['bbox_source']=='original')][['x','y','w','h']].values
    bboxes_merged = train_labels_big_images[(train_labels_big_images['final_image_id']==random_image) & (train_labels_big_images['bbox_source']=='merged')][['x','y','w','h']].values
    display_instance(PUZZLE_IMAGES_TRAIN_DIR+random_image+'.jpg',[bboxes_original,bboxes_merged],ax[i//2,i%2])


# ### Export dataframe as csv

# In[ ]:


train_labels_big_images.to_csv(PUZZLE_IMAGES_TRAIN_LABELS_FILE)

