#!/usr/bin/env python
# coding: utf-8

# ### The kernel is ensemble of approaches form public lederboard and object detection based DSL
# 
# Credit: https://www.kaggle.com/adityaork/decision-tree-smart-data-augmentation

# In[ ]:


from xgboost import XGBClassifier
import pdb

import pandas as pd
import math
import sys
import cv2

import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random

from collections import defaultdict
from itertools import product
from itertools import combinations,permutations
from math import floor

import copy


data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

def plot_result(test_input, test_prediction,
                input_shape):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    fig, axs = plt.subplots(1, 2, figsize=(15,15))
    test_input = test_input.reshape(input_shape[0],input_shape[1])
    axs[0].imshow(test_input, cmap=cmap, norm=norm)
    axs[0].axis('off')
    axs[0].set_title('Actual Target')
    test_prediction = test_prediction.reshape(input_shape[0],input_shape[1])
    axs[1].imshow(test_prediction, cmap=cmap, norm=norm)
    axs[1].axis('off')
    axs[1].set_title('Model Prediction')
    plt.tight_layout()
    plt.show()
    
def plot_test(test_prediction, task_name):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    fig, axs = plt.subplots(1, 1, figsize=(15,15))
    axs.imshow(test_prediction, cmap=cmap, norm=norm)
    axs.axis('off')
    axs.set_title(f'Test Prediction {task_name}')
    plt.tight_layout()
    plt.show()
    
# https://www.kaggle.com/inversion/abstraction-and-reasoning-starter-notebook


# In[ ]:


# https://www.kaggle.com/inversion/abstraction-and-reasoning-starter-notebook
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

sample_sub1 = pd.read_csv(data_path/'sample_submission.csv')
sample_sub1 = sample_sub1.set_index('output_id')
sample_sub1.head()


# In[ ]:


from collections import defaultdict
from itertools import product
from itertools import combinations,permutations
from math import floor


def get_moore_neighbours(color, cur_row, cur_col, nrows, ncols):

    if cur_row<=0: top = -1
    else: top = color[cur_row-1][cur_col]
        
    if cur_row>=nrows-1: bottom = -1
    else: bottom = color[cur_row+1][cur_col]
        
    if cur_col<=0: left = -1
    else: left = color[cur_row][cur_col-1]
        
    if cur_col>=ncols-1: right = -1
    else: right = color[cur_row][cur_col+1]
        
    return top, bottom, left, right

def get_tl_tr(color, cur_row, cur_col, nrows, ncols):
        
    if cur_row==0:
        top_left = -1
        top_right = -1
    else:
        if cur_col==0: top_left=-1
        else: top_left = color[cur_row-1][cur_col-1]
        if cur_col==ncols-1: top_right=-1
        else: top_right = color[cur_row-1][cur_col+1]   
        
    return top_left, top_right

from itertools import product
def getAround(i,j,inp,size=1):
    #v = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
    r,c = len(inp),len(inp[0])
    v = []
    sc = [0]
    for q in range(size):
        sc.append(q+1)
        sc.append(-(q+1))
    for idx,(x,y) in enumerate(product(sc,sc)):
        ii = (i+x)
        jj = (j+y)
        v.append(-1)
        if((0<= ii < r) and (0<= jj < c)):
            v[idx] = (inp[ii][jj])
    return v

def make_features(input_color, size):
    nrows, ncols = input_color.shape
    feat = []
    cur_idx = 0
    for i in range(nrows):
        for j in range(ncols):
            feat.append([])
            feat[cur_idx].append(i+1)
            feat[cur_idx].append(j+1)
            feat[cur_idx].append(input_color[i][j])
            
            feat[cur_idx].append(input_color[i][j] in input_color[i,j+1:])
            feat[cur_idx].append(input_color[i][j] in input_color[i,:j])
            feat[cur_idx].append(input_color[i][j] in input_color[i+1:,:])
            feat[cur_idx].append(input_color[i][j] in input_color[:i,:])
            
            feat[cur_idx].append(len(np.unique(input_color[i,:])))
            feat[cur_idx].append(len(np.unique(input_color[:,j])))
            feat[cur_idx].append((i+j))
            feat[cur_idx].append((i*j))
            
            for m in range(10):
                feat[cur_idx].append(i%(m+1))
                feat[cur_idx].append(j%(m+1))
                
            feat[cur_idx].append((i+1)/(j+1))
            feat[cur_idx].append((j+1)/(i+1))
            feat[cur_idx].append(nrows)
            feat[cur_idx].append(ncols)
            
            around = getAround(i, j, input_color, size)
            feat[cur_idx].extend(around)
            feat[cur_idx].append(len(np.unique(around)))
 
            cur_idx += 1
        
    return feat

def equal(a, b):
    if a.shape != b.shape:
        return False
    if (a == b).all():
        return True
    return False

def getiorc(pair):
    inp = pair["input"]
    return pair["input"],pair["output"],len(inp),len(inp[0])

def getBkgColor(task_json):
    color_dict = defaultdict(int)
    
    for pair in task_json['train']:
        inp,oup,r,c = getiorc(pair)
        for i in range(r):
            for j in range(c):
                color_dict[inp[i][j]]+=1
    color = -1
    max_count = 0
    for col,cnt in color_dict.items():
        if(cnt > max_count):
            color = col
            max_count = cnt
    return color

def get_num_colors(inp,oup,bl_cols):
    r,c = len(inp),len(inp[0])
    return 

def replace(inp,uni,perm):
    # uni = '234' perm = ['5','7','9']
    #print(uni,perm)
    r_map = { int(c):int(s) for c,s in zip(uni,perm)}
    r,c = len(inp),len(inp[0])
    rp = np.array(inp).tolist()
    #print(rp)
    for i in range(r):
        for j in range(c):
            if(rp[i][j] in r_map):
                rp[i][j] = r_map[rp[i][j]]
    return rp
            
    
def augment(inp,oup,bl_cols):
    cols = "0123456789"
    npr_map = [1,9,72,3024,15120,60480,181440,362880,362880]
    uni = "".join([str(x) for x in np.unique(inp).tolist()])
    for c in bl_cols:
        cols=cols.replace(str(c),"")
        uni=uni.replace(str(c),"")

    exp_size = len(inp)*len(inp[0])*npr_map[len(uni)]
    
    mod = floor(exp_size/120000)
    mod = 1 if mod==0 else mod
    
    #print(exp_size,mod,len(uni))
    result = []
    count = 0
    for comb in combinations(cols,len(uni)):
        for perm in permutations(comb):
            count+=1
            if(count % mod == 0):
                result.append((replace(inp,uni,perm),replace(oup,uni,perm)))
    return result

def get_bl_cols(task_json):
    result = []
    bkg_col = getBkgColor(task_json);
    result.append(bkg_col)
    # num_input,input_cnt,num_output,output_cnt
    met_map = {}
    for i in range(10):
        met_map[i] = [0,0,0,0]
        
    total_ex = 0
    for pair in task_json['train']:
        inp,oup=pair["input"],pair["output"]
        u,uc = np.unique(inp, return_counts=True)
        inp_cnt_map = dict(zip(u,uc))
        u,uc = np.unique(oup, return_counts=True)
        oup_cnt_map = dict(zip(u,uc))
        
        for col,cnt in inp_cnt_map.items():
            met_map[col][0] = met_map[col][0] + 1
            met_map[col][1] = met_map[col][1] + cnt
        for col,cnt in oup_cnt_map.items():
            met_map[col][2] = met_map[col][2] + 1
            met_map[col][3] = met_map[col][3] + cnt
        total_ex+=1
    
    for col,met in met_map.items():
        num_input,input_cnt,num_output,output_cnt = met
        if(num_input == total_ex or num_output == total_ex):
            result.append(col)
        elif(num_input == 0 and num_output > 0):
            result.append(col)
    
    result = np.unique(result).tolist()
    if(len(result) == 10):
        result.append(bkg_col)
    return np.unique(result).tolist()


def get_flips(inp,oup):
    result = []
    n_inp = np.array(inp)
    n_oup = np.array(oup)
    result.append((np.fliplr(inp).tolist(),np.fliplr(oup).tolist()))
    result.append((np.rot90(np.fliplr(inp),1).tolist(),np.rot90(np.fliplr(oup),1).tolist()))
    result.append((np.rot90(np.fliplr(inp),2).tolist(),np.rot90(np.fliplr(oup),2).tolist()))
    result.append((np.rot90(np.fliplr(inp),3).tolist(),np.rot90(np.fliplr(oup),3).tolist()))
    result.append((np.flipud(inp).tolist(),np.flipud(oup).tolist()))
    result.append((np.rot90(np.flipud(inp),1).tolist(),np.rot90(np.flipud(oup),1).tolist()))
    result.append((np.rot90(np.flipud(inp),2).tolist(),np.rot90(np.flipud(oup),2).tolist()))
    result.append((np.rot90(np.flipud(inp),3).tolist(),np.rot90(np.flipud(oup),3).tolist()))
    result.append((np.fliplr(np.flipud(inp)).tolist(),np.fliplr(np.flipud(oup)).tolist()))
    result.append((np.flipud(np.fliplr(inp)).tolist(),np.flipud(np.fliplr(oup)).tolist()))
    return result

def get_array(x):
    return np.array(copy.deepcopy(x))

def features(task, mode='train', name=None, size=1, bl_cols=None):
    num_train_pairs = len(task[mode])
    feat, target = [], []
    
    global local_neighb
    for task_num in range(num_train_pairs):
        flag = 0
        inp = copy.deepcopy(task[mode][task_num]['input'])
        out = copy.deepcopy(task[mode][task_num]['output'])

        feat.extend(make_features(get_array(inp), size=size))
        target.extend(get_array(out).reshape(-1,))
        
        flips = get_flips(inp, out)
        for x, y in flips:
            feat.extend(make_features(get_array(x), size))
            target.extend(get_array(y).reshape(-1,))
        
        if bl_cols:
            augs = augment(inp, out, bl_cols)
            for x, y in augs:
                feat.extend(make_features(get_array(x), size))
                target.extend(get_array(y).reshape(-1,))
            
    return np.array(feat), np.array(target)

def check_in_oup_equal(task_json):
    return all([ len(pair["input"]) == len(pair["output"]) and len(pair["input"][0]) == len(pair["output"][0])
                for pair in task_json['train']])


# In[ ]:


# mode = 'eval'
mode = 'test'
if mode=='eval':
    task_path = evaluation_path
elif mode=='train':
    task_path = training_path
elif mode=='test':
    task_path = test_path

all_task_ids = sorted(os.listdir(task_path))

local_neighb = 5
valid_scores = {}

model_accuracies = {'ens': []}
pred_taskids = []

for task_id in all_task_ids:

    task_file = str(task_path / task_id)
    with open(task_file, 'r') as f:
        task = json.load(f)
        
    if not check_in_oup_equal(task):
        print("Ignoring", task_file)
        continue
    
    bl_cols = get_bl_cols(task)
    
    feat1, target1 = features(task, name=task_file, size=1, bl_cols=bl_cols)
    feat2, target2 = features(task, name=task_file, size=3, bl_cols=bl_cols)
    feat3, target3 = features(task, name=task_file, size=5, bl_cols=bl_cols)

    xgb1 =  XGBClassifier(n_estimators=100, n_jobs=-1)
    xgb2 =  XGBClassifier(n_estimators=100, n_jobs=-1)
    xgb3 =  XGBClassifier(n_estimators=100, n_jobs=-1)
    
    xgb1.fit(feat1, target1, verbose=-1)
    xgb2.fit(feat2, target2, verbose=-1)
    xgb3.fit(feat3, target3, verbose=-1)


    
    num_test_pairs = len(task['test'])
    for task_num in range(num_test_pairs):
        cur_idx = 0
        input_color = np.array(task['test'][task_num]['input'])
        nrows, ncols = len(input_color), len(
            input_color[0])
        feat1 = make_features(input_color, 1)
        feat2 = make_features(input_color, 3)
        feat3 = make_features(input_color, 5)

        print('Made predictions for ', task_id[:-5])

        preds1 = xgb1.predict(np.array(feat1)).reshape(nrows,ncols)
        preds2 = xgb2.predict(np.array(feat2)).reshape(nrows,ncols)
        preds3 = xgb3.predict(np.array(feat3)).reshape(nrows,ncols)
#         preds1 = preds2
#         preds3 = preds2
        if (mode=='train') or (mode=='eval'):                
            ens_acc = max((np.array(task['test'][task_num]['output'])==preds1).sum()/(nrows*ncols),
                         (np.array(task['test'][task_num]['output'])==preds2).sum()/(nrows*ncols),
                         (np.array(task['test'][task_num]['output'])==preds3).sum()/(nrows*ncols))

            model_accuracies['ens'].append(ens_acc)

            pred_taskids.append(f'{task_id[:-5]}_{task_num}')

            print('ensemble accuracy',ens_acc)

          
        preds1 = preds1.astype(int).tolist()
        preds2 = preds2.astype(int).tolist()
        preds3 = preds3.astype(int).tolist()
#         plot_test(preds1, task_id)
#         plot_test(preds2, task_id)
#         plot_test(preds3, task_id)
        sample_sub1.loc[f'{task_id[:-5]}_{task_num}',
                       'output'] = flattener(preds1) +' '+ flattener(preds2) +' '+ flattener(preds3) + ' '


# In[ ]:


data_path = '/kaggle/input/abstraction-and-reasoning-challenge/'
submission = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'), index_col='output_id')
submission.output = ''
submission.head(2)


# In[ ]:


def read_json(name, type_='train'):
    if type_ == 'train':
        task_file = os.path.join(training_path, name)
    elif type_ == 'test':
        task_file = os.path.join(test_path, name)
    else:
        task_file = os.path.join(val_path, name)

    with open(task_file, 'r') as f:
        task = json.load(f)

    return task

def get_size(box):
    return (len(box), len(box[0]))

def unique(x):
    return list(set(x))

def nunique(x):
    return len(list(set(x)))

def equal(a, b):
    if a.shape != b.shape:
        return False
    if (a == b).all():
        return True
    return False

def compare_kernel(a, b):
    if a.shape != b.shape:
        return -1, -2
    
    if equal(a, b):
        return 1, 101
    
    if equal(a[:, ::-1], b):
        return 1, 106

    if equal(a[::-1, :], b):
        return 1, 107
    
    if equal(a.T, b):
        return 1, 105
    
    if equal(a[::-1, ::-1], b):
        return 1, 108
    
    if equal(np.rot90(a), b):
        return 1, 102
    
    if equal(np.rot90(np.rot90(a)), b):
        return 1, 103
    
    if equal(np.rot90(np.rot90(np.rot90(a))), b):
        return 1, 104
    
    if len(np.unique(b)) == 1:
        return 0, np.unique(b)[0]
    
    else:
        return -1, -1
    
    
def perform_rotation(k, f):
    if f == 101:
        return k
    if f == 102:
        return np.rot90(k)
    if f == 103:
        return np.rot90(np.rot90(k))
    if f == 104:
        return np.rot90(np.rot90(np.rot90(k)))
    if f == 105:
        return k.T
    if f == 106:
        return k[:, ::-1]
    if f == 107:
        return k[::-1, :]
    if f == 108:
        return k[::-1, ::-1]

from collections import Counter
def get_sq_expansion_ratio(task):
    x = [(int(get_size(t['output'])[0] / get_size(t['input'])[0]),      int(get_size(t['output'])[1] / get_size(t['input'])[1])) for t in task['train']]
    
    num_test = len(task['test'])
    
    if nunique(x) == 1:
        res = []
        while len(res) < num_test:
            res.append(unique(x)[0])
        return True, res
    else:
#         for i, train in enumerate(task['train']):
            
            
        return False, x
    
def check_edge(t):
    img1 = np.array(t["input"], dtype='uint8')
    img2 = np.array(t["output"], dtype='uint8')
    
    img1 = cv2.resize(img1, (0, 0), fx=30, fy=30, interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(img2, (0, 0), fx=30, fy=30, interpolation = cv2.INTER_AREA)
    
    edge1 = cv2.Canny(img1, 0, 1)
#     plt.imshow(edge1, cmap=cmap, norm=norm)
#     plt.show()
    
    edge2 = cv2.Canny(img2, 0, 1)
#     plt.imshow(edge2, cmap=cmap, norm=norm)
#     plt.show()
    
    return (edge1 == edge2).all()


# In[ ]:


def flattener123(pred):
    
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


# In[ ]:


def predict(case, task, task_name):
    prediction = []
    test_id = int(task_name.split('_')[1])
    
    for i, train in enumerate(task['train']):
        prediction.append(get_prediction(test_id, task, case, i))    
    
    prediction = list(set([flattener(pred) if (type(pred) == type([1, 2, 3])) else flattener(pred.astype('int').tolist()) for pred in prediction]))
    
    while len(prediction) < 3:
        prediction.append(prediction[-1])
    
    if len(prediction) > 3:
        prediction = prediction[:3]
        
    prediction = ' '.join(prediction)
#     prediction = prediction[0] + ' '
    
    submission.loc[task_name, 'output'] = prediction

#     if 'output' in task['test'][test_id].keys():
#         print("Validation for ", task_name, [(x == test['output']).all() for x in prediction[i]])
        

    
def get_prediction(test_id, task, case, train_index):
    '''
    t - train task on whose basis prediction is made
    '''
    test_inp = task['test'][test_id]['input']
#     print(case)
    if case == 'case BA04':
        return case_BA04(test_inp, task, case, train_index)
    
    if case == 'case BA05':
        return case_BA05(test_inp, task, case, train_index)
    
    if case == 'case BA06':
        return case_BA06(test_inp, task, case, train_index)
    
    else:
        return test_inp
        
        


# In[ ]:


def plot_task(task):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    num_rows = int(math.ceil((len(task['train']) + len(task['test'])) / 2))
    fig, axs = plt.subplots(num_rows, 4, figsize=(15,15))
    i = 0
    while i < len(task['train']):
        axs[int(i/2)][(i%2)*2].imshow(task['train'][i]['input'], cmap=cmap, norm=norm)
        axs[int(i/2)][(i%2)*2].axis('off')
        axs[int(i/2)][(i%2)*2].set_title('Train Input')
        axs[int(i/2)][1 + (i%2)*2].imshow(task['train'][i]['output'], cmap=cmap, norm=norm)
        axs[int(i/2)][1 + (i%2)*2].axis('off')
        axs[int(i/2)][1 + (i%2)*2].set_title('Train Output')
        i += 1
    j = 0
    while j < len(task['test']):
        axs[int(i/2)][(i%2)*2].imshow(task['test'][j]['input'], cmap=cmap, norm=norm)
        axs[int(i/2)][(i%2)*2].axis('off')
        axs[int(i/2)][(i%2)*2].set_title('Test Input')
#         axs[int(i/2)][1 + (i%2)*2].imshow(task['test'][j]['output'], cmap=cmap, norm=norm)
#         axs[int(i/2)][1 + (i%2)*2].axis('off')
#         axs[int(i/2)][1 + (i%2)*2].set_title('Test Output')
        i += 1 
        j += 1
        
    plt.tight_layout()
    plt.show()
# plot_task(read_json('00d62c1b.json'))


# In[ ]:


def compare_dict(a, b):
    a_key = list(a.keys())
    b_key = list(b.keys())
    a_key.sort()
    b_key.sort()
    if a_key != b_key:
        return False
    for x in a:
        if a[x] != b[x]:
            return False
    return True

def get_counter(a):
    return Counter([y for x in a for y in x])


def get_rectified_kernel(inp):
    i = 0
    j = 0
    k = 0
    mapping = {}
    while i < len(inp):
        j = 0
        while j < len(inp[0]):
            if inp[i][j] not in mapping:
                mapping[inp[i][j]] = k
                inp[i][j] = k
                k += 1
            else:
                inp[i][j] = mapping[inp[i][j]]
            j += 1
        i += 1
    
    return inp, mapping

def Defensive_Copy(A): 
    n = len(A)
    k = len(A[0])
    L = np.zeros((n,k), dtype = int)
    for i in range(n):
        for j in range(k):
            L[i,j] = 0 + A[i][j]
    return L.tolist()


# In[ ]:


def case_three(task_name):
    print(task_name)
    global flag123
#     flag123 = True
    task = read_json(row.name.split('_')[0]+'.json', 'test')


    if ([(get_size(t['output'])[0] % get_size(t['input'])[0] == 0) and         (get_size(t['output'])[1] % get_size(t['input'])[1] != 0) for t in task['train']]):
        # case CD01
        print(task_name, "case CD01")
#             plot_task(read_json(task_name))
#         flag123 = True
#             pass
    else:
#             print(task_name, "fffff")
        pass


# In[ ]:


import copy
def check_in(t):
    inp = copy.deepcopy(np.array(t['input']))
    out = copy.deepcopy(np.array(t['output']))
    out_size = out.shape
    i, j = 0, 0
    while i < len(inp):
        j = 0
        while j < len(inp[0]):
            if equal(out, inp[i:i+out_size[0], j:j+out_size[1]]):
                return True
            j += 1
        i += 1
            
    return False
    


# In[ ]:


import random
def unique_location(t):
    inp = copy.deepcopy(np.array(t['input']))
    out = copy.deepcopy(np.array(t['output']))
    out_size = out.shape
    i, j = 0, 0
    res = []
    while i < len(inp):
        j = 0
        while j < len(inp[0]):
            if equal(out, inp[i:i+out_size[0], j:j+out_size[1]]):
                res.append((i, j, i+out_size[0], j+out_size[1]))
            j += 1
        i += 1
    
    return res
#     if len(res) == 1:
#     return (res[0][0], res[0][1], res[0][0]+out_size[0], res[0][1]+out_size[1])
#     else:
#         return (random.randint(1, 4000), random.randint(1, 4000), random.randint(1, 4000), random.randint(1, 4000))
def find_common(lists):
    index = 0
    i = 1
    for i in range(len(lists)):
        if len(lists[i]) > len(lists[index]):
            index = i
    
    res = []
    for te in lists[index]:
        if all([te in lists[i] for i in range(len(lists))]):
            print(te)
            res.append(te)
            
    if len(res) == 1:
        return True
    else:
        return False

    
def self_repeating(t):
    inp = copy.deepcopy(np.array(t['input']))
    out = copy.deepcopy(np.array(t['output']))
    
    if inp.shape[0] > inp.shape[1]:
        #vertical
        key = 2
        while 2*key <= inp.shape[0]:
            if inp.shape[0] % key != 0:
                key += 1
                continue
            
            if equal(inp[0:key, :], inp[key:2*key, :]):
                return True
            key += 1
        return False
            
    elif inp.shape[0] < inp.shape[1]:
        #horizontal
        key = 2
        while 2*key <= inp.shape[1]:
            if inp.shape[1] % key != 0:
                key += 1
                continue
            
            if equal(inp[:, 0:key], inp[:, key:2*key]):
                return True
            key += 1
        return False
        
    else:
        return False


# In[ ]:


def groupByColor(pixmap):
    nb_colors = int(pixmap.max()) + 1
    splited = [(pixmap == i) * i for i in range(1, nb_colors)]
    return [x for x in splited if np.any(x)]

def cropToContent(pixmap):
    true_points = np.argwhere(pixmap)
    if len(true_points) == 0:
        return []

    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    pixmap = pixmap[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    return pixmap

def unique_box(t):
    inp = copy.deepcopy(np.array(t['input']))
    out = copy.deepcopy(np.array(t['output']))
    
    color_frames = groupByColor(inp)
    for ori_frame in color_frames:
        t = np.unique(ori_frame)
        frame = copy.deepcopy(ori_frame)
        frame = np.array(frame, dtype='uint8') * 10
        ret, frame = cv2.threshold(frame,5,255,0)

        contours,hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        frame = frame * 0
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                temp = ori_frame[y:y+h, x:x+w]
                if equal(inp[y:y+h, x:x+w], out):
                    return True
                elif equal(inp[y+1:y+h-1, x+1:x+w-1], out):
                    return True
    return False


# In[ ]:


def case_BA04(test_inp, task, case, train_index): # Done
    decision = []
    for t in task['train']:
        inp = copy.deepcopy(np.array(t['input']))
        out = copy.deepcopy(np.array(t['output']))

        color_frames = groupByColor(inp)
        flag = False
        res = []
        area = []
        color_val = []
        cou_count = []
        for ori_frame in color_frames:
            t = np.unique(ori_frame)
            frame = copy.deepcopy(ori_frame)
            uni_val = [x for x in np.unique(ori_frame) if x != 0]
            
            frame = np.array(frame, dtype='uint8') * 10
            ret, frame = cv2.threshold(frame,5,255,0)

            contours,hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            frame = frame * 0
            
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                if len(approx) == 4:
                    (x, y, w, h) = cv2.boundingRect(approx)
                    if w*h != ori_frame.shape[0]*ori_frame.shape[1]:
                        temp = ori_frame[y:y+h, x:x+w]
                        sum_ver = np.sum(temp, axis=0) / uni_val[0]
                        sum_hor = np.sum(temp, axis=1) / uni_val[0]
                        if (sum_ver[0] == temp.shape[0]) and (sum_ver[-1] == temp.shape[0]) and                                 (sum_hor[0] == temp.shape[1]) and (sum_hor[-1] == temp.shape[1]):
                            if equal(inp[y:y+h, x:x+w], out):
                                res.append(((x, y, w, h), True, "with_border", w*h, uni_val[0], np.sum(ori_frame)==np.sum(temp), np.sum(sum_hor)))
                            elif equal(inp[y+1:y+h-1, x+1:x+w-1], out):
                                res.append(((x, y, w, h), True, "without_border", w*h, uni_val[0], np.sum(ori_frame)==np.sum(temp), np.sum(sum_hor)))
                            else:
                                res.append(((x, y, w, h), False, "", w*h, uni_val[0], np.sum(ori_frame)==np.sum(temp), np.sum(sum_hor)))

        res = list(set(res))
        print("frame", res)
        area = [x[3] for x in res]
        color_val = [x[4] for x in res]
        cou_count = [x[5] for x in res]
        cou_sum = [x[6] for x in res]
        for i in range(len(res)):
            if res[i][1] == True:
                if len(area) == 1:
                    print("smalllargest")
                    decision.append((res[i], "smallestlargest", color_val[i], cou_count[i]))
                    
                elif (max(area) == area[i]) and (len([1 for x in area if max(area)==x]) == 1):
                    print("largest")
                    decision.append((res[i], "largest", color_val[i], cou_count[i]))
                    
                elif (max(area) == area[i]) and (len([1 for x in area if max(area)==x]) != 1):
                    if (max(cou_sum) == cou_sum[i]): 
                        print("largest_max")
                        decision.append((res[i], "largest_max", color_val[i], cou_count[i]))
                    elif (min(cou_sum) == cou_sum[i]): 
                        print("largest_min")
                        decision.append((res[i], "largest_min", color_val[i], cou_count[i]))
                    else: 
                        print("largest_no")
                        decision.append((res[i], "largest_no", color_val[i], cou_count[i]))
                
                elif (min(area) == area[i]) and (len([1 for x in area if min(area)==x]) == 1):
                    print("smallest")
                    decision.append((res[i], "smallest", color_val[i], cou_count[i]))
                    
                elif (min(area) == area[i]) and (len([1 for x in area if min(area)==x]) != 1):
                    if (max(cou_sum) == cou_sum[i]): 
                        print("smallest_max")
                        decision.append((res[i], "smallest_max", color_val[i], cou_count[i]))
                    elif (min(cou_sum) == cou_sum[i]): 
                        print("smallest_min")
                        decision.append((res[i], "smallest_min", color_val[i], cou_count[i]))
                    else: 
                        print("smallest_no")
                        decision.append((res[i], "smallest_no", color_val[i], cou_count[i]))
                
                else:
                    decision.append((res[i], "uni", color_val[i], cou_count[i]))
                    
#     predict
    inp = copy.deepcopy(np.array(test_inp))
    color_frames = groupByColor(inp)
    res = []
    area = []
    color_val = []
    cou_count = [] 
    for ori_frame in color_frames:
        t = np.unique(ori_frame)
        frame = copy.deepcopy(ori_frame)
        uni_val = [x for x in np.unique(ori_frame) if x != 0]
        frame = np.array(frame, dtype='uint8') * 10
        ret, frame = cv2.threshold(frame,5,255,0)

        contours,hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        frame = frame * 0

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                if w*h != ori_frame.shape[0]*ori_frame.shape[1]:
                    temp = ori_frame[y:y+h, x:x+w]
                    sum_ver = np.sum(temp, axis=0) / uni_val[0]
                    sum_hor = np.sum(temp, axis=1) / uni_val[0]
                    if (sum_ver[0] == temp.shape[0]) and (sum_ver[-1] == temp.shape[0]) and                             (sum_hor[0] == temp.shape[1]) and (sum_hor[-1] == temp.shape[1]):

                        res.append(((x, y, w, h), w*h, uni_val[0], np.sum(ori_frame)==np.sum(temp), np.sum(sum_hor)))
    
    res = list(set(res))
    print("-->", res)
    print(decision)
    info = (None, None)
    
    if nunique([x[2] for x in decision]) == 1:
        print("color")
        for x in res:
            if x[2] == unique([y[2] for y in decision])[0]:
                info = (x[0], decision[0][0][2])
                
    elif all([True if "largest" in x[1] else False for x in decision]):
        if all([True if "max" in x[1] else False for x in decision]):
            print("lar size max")
            area = [x[1] for x in res]
            max_sum = [x[4] if x[1] == max(area) else 0 for x in res ]
            info = (res[max_sum.index(max(max_sum))][0], decision[0][0][2])
            
        elif all([True if "min" in x[1] else False for x in decision]):
            print("lar size min")
            area = [x[1] for x in res]
            min_sum = [x[4] if x[1] == max(area) else 9999999 for x in res ]
            info = (res[min_sum.index(min(min_sum))][0], decision[0][0][2])
        else:
            print("lar size")
            area = [x[1] for x in res]
            info = (res[area.index(max(area))][0], decision[0][0][2])
    
    elif all([True if "smallest" in x[1] else False for x in decision]):
        if all([True if "max" in x[1] else False for x in decision]):
            print("small size max")
            area = [x[1] for x in res]
            max_sum = [x[4] if x[1] == min(area) else 0 for x in res ]
            info = (res[max_sum.index(max(max_sum))][0], decision[0][0][2])
            
        elif all([True if "min" in x[1] else False for x in decision]):
            print("lar size min")
            area = [x[1] for x in res]
            min_sum = [x[4] if x[1] == min(area) else 9999999 for x in res ]
            info = (res[min_sum.index(min(min_sum))][0], decision[0][0][2])
        else:
            print("small size")
            area = [x[1] for x in res]
            info = (res[area.index(min(area))][0], decision[0][0][2])
    
    elif (nunique([x[3] for x in decision]) == 1) and (unique([x[3] for x in decision])[0] == True):
        print("unique")
        for x in res:
            if x[3] == True:
                info = (x[0], decision[0][0][2])
    else:
        print("fuck", )
    if info[0]:
        (x, y, w, h) = info[0]
        print(info)
        if info[1] == "with_border":
            return inp[y:y+h, x:x+w]
        else:
            return inp[y+1:y+h-1, x+1:x+w-1]
                     
    else:
        info = (res[min(train_index, len(res)-1)][0], decision[0][0][2])
        (x, y, w, h) = info[0]
        if info[1] == "with_border":
            return inp[y:y+h, x:x+w]
        else:
            return inp[y+1:y+h-1, x+1:x+w-1]


# In[ ]:


def unique_color(t):
    inp = copy.deepcopy(np.array(t['input']))
    out = copy.deepcopy(np.array(t['output']))
    
    color_frames = groupByColor(inp)
    flag = False
    for ori_frame in color_frames:
        t = np.unique(ori_frame)
        frame = copy.deepcopy(ori_frame)
        
        true_points = np.argwhere(frame)
        top_left = true_points.min(axis=0)
        bottom_right = true_points.max(axis=0)
        
        if equal(inp[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1], out):
            if np.sum(ori_frame[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]) ==                 np.sum(ori_frame):
                return True
    return False


def is_in(t):
    inp = copy.deepcopy(np.array(t['input']))
    out = copy.deepcopy(np.array(t['output']))
    skeleton = (inp != 0).astype('int')

    frame = copy.deepcopy(skeleton)
    frame = np.array(frame, dtype='uint8') * 10
    ret, frame = cv2.threshold(frame,5,255,0)
    contours,hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
#         print((x, y, w, h))
#         tep = np.zeros(frame.shape)
#         cv2.drawContours(tep, [c], -1, (100, 100, 100), 1)
#         plt.imshow(tep)
#         plt.show()

        is_equal, _ = compare_kernel(inp[y:y+h, x:x+w], out)
#         print(is_equal)
#         print(inp[y:y+h, x:x+w])
#         print(out)
        if is_equal == 1:
            return True
    return False


# In[ ]:


def show(im):
    plt.imshow(im)
    plt.show()
    
def case_BA05(test_inp, task, case, train_task): # Done
    decision = []
    for t in task['train']:
        inp = copy.deepcopy(np.array(t['input']))
        out = copy.deepcopy(np.array(t['output']))
        color_frames = groupByColor(inp)
        
        count = {}
        for ori_frame in color_frames:
            skeleton = copy.deepcopy(ori_frame)
            uni_val = [x for x in np.unique(ori_frame) if x != 0][0]
            
            skeleton = (skeleton != 0).astype('int')
            frame = copy.deepcopy(skeleton)

            frame = np.array(frame, dtype='uint8') * 10
            ret, frame = cv2.threshold(frame,5,255,0)
            contours,hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            count[uni_val] = []
            for c, hi in zip(contours, hierarchy[0]):
                x, y, w, h = cv2.boundingRect(c)
#                 print(x,y,w,h, "-->", hi, "col", uni_val)
                if hi[3] == -1:
                    tep = np.zeros(frame.shape)
                    cv2.drawContours(tep, [c], -1, (100, 100, 100), 1)
#                     show(tep)
                
                    if equal(inp[y:y+h, x:x+w], out) and (np.sum(ori_frame[y:y+h, x:x+w]) == np.sum(ori_frame)):
                        count[uni_val].append(((x, y, w, h), True, np.sum(ori_frame)/uni_val))
                    else:
                        count[uni_val].append(((x, y, w, h), False, np.sum(ori_frame)/uni_val))
            
        decision.append(count)
    
#     predict
    test_count = {}
    inp = copy.deepcopy(np.array(test_inp))
    color_frames = groupByColor(inp)

    for ori_frame in color_frames:
        skeleton = copy.deepcopy(ori_frame)
        uni_val = [x for x in np.unique(ori_frame) if x != 0][0]

        skeleton = (skeleton != 0).astype('int')
        frame = copy.deepcopy(skeleton)

        frame = np.array(frame, dtype='uint8') * 10
        ret, frame = cv2.threshold(frame,5,255,0)
        contours,hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        test_count[uni_val] = []
        for c, hi in zip(contours, hierarchy[0]):
            x, y, w, h = cv2.boundingRect(c)
#             print(x,y,w,h, "-->", hi, "col", uni_val)
            if hi[3] == -1:
                tep = np.zeros(frame.shape)
                cv2.drawContours(tep, [c], -1, (100, 100, 100), 1)
#                 show(tep)

                if equal(inp[y:y+h, x:x+w], out) and (np.sum(ori_frame[y:y+h, x:x+w]) == np.sum(ori_frame)):
                    test_count[uni_val].append(((x, y, w, h), True, np.sum(ori_frame)/uni_val))
                else:
                    test_count[uni_val].append(((x, y, w, h), False, np.sum(ori_frame)/uni_val))
#     inp = copy.deepcopy(np.array(test_inp))
#     # single contour in color
#     truth_feature = []
#     for i, frame in enumerate(decision):
#         truth_feature.append(False)
#         for col in frame:
#             for val in frame[col]:
#                 if val[1] == True:
#                     if len(frame[col]) == 1:
#                         truth_feature[i] = True
#                 else:
#                     if len(frame[col]) == 1:
#                         truth_feature[i] = False
                        
#     print(truth_feature)
#     if all(truth_feature):
#         res = []
#         for col in test_count:
#             if len(test_count[col]) == 1:
#                 res.append(test_count[col][0])
#         if len(res) == 1:
#             x, y, w, h = res[0][0]
#             return inp[y:y+h, x:x+w]
#         else:
#             x, y, w, h = res[min(len(res)-1, train_task)][0]
#             return inp[y:y+h, x:x+w]
        
#     inp = copy.deepcopy(np.array(test_inp))
    # min area in color
    truth_feature = []
    for i, frame in enumerate(decision):
        truth_feature.append(False)
        frame_area = []
        for col in frame:
            frame_area.append(frame[col][0][2])
            
        for col in frame:
            for val in frame[col]:
                if val[1] == True:
                    if val[2] == min(frame_area):
                        truth_feature[i] = True
                        
    print(truth_feature)
    if all(truth_feature):
        frame_area = []
        for col in test_count:
            frame_area.append([col, test_count[col][0][2]])
            
        frame_area.sort(key=(lambda x:x[1]))
        
        min_area_frame_col = frame_area[0][0]
        res_skeleton = (test_inp == min_area_frame_col).astype('int')

        true_points = np.argwhere(res_skeleton)
        top_left = true_points.min(axis=0)
        bottom_right = true_points.max(axis=0)
        
        return inp[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    inp = copy.deepcopy(np.array(test_inp))
    # max area in color
    truth_feature = []
    for i, frame in enumerate(decision):
        truth_feature.append(False)
        frame_area = []
        for col in frame:
            frame_area.append(frame[col][0][2])
            
        for col in frame:
            for val in frame[col]:
                if val[1] == True:
                    if val[2] == max(frame_area):
                        truth_feature[i] = True
                        
    print(truth_feature)
    if all(truth_feature):
        frame_area = []
        for col in test_count:
            frame_area.append([col, test_count[col][0][2]])
            
        frame_area.sort(key=(lambda x:x[1]))
        
        min_area_frame_col = frame_area[-1][0]
        res_skeleton = (test_inp == min_area_frame_col).astype('int')

        true_points = np.argwhere(res_skeleton)
        top_left = true_points.min(axis=0)
        bottom_right = true_points.max(axis=0)
        
        return inp[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

#     inp = copy.deepcopy(np.array(test_inp))
    #failsafe
    colors = list(test_count.keys())
    colors.sort(reverse=True)
#     colors.sort()
    lucky_frame_col = colors[min(len(colors)-1, train_task)]
    res_skeleton = (test_inp == lucky_frame_col).astype('int')

    true_points = np.argwhere(res_skeleton)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    show(inp[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1])
    return inp[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

        
    


# In[ ]:


def get_col_profile(d):
    return Counter(list(d.reshape(-1)))

def case_BA06(test_inp, task, case, train_task): # Done
    decision = []
    for t in task['train']:
        inp = copy.deepcopy(np.array(t['input']))
        out = copy.deepcopy(np.array(t['output']))

        skeleton = (inp != 0).astype('int')

        frame = copy.deepcopy(skeleton)
        frame = np.array(frame, dtype='uint8') * 10
        ret, frame = cv2.threshold(frame,5,255,0)
        contours,hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        res = []
        t = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
#             print((x, y, w, h))
#             tep = np.zeros(frame.shape)
#             cv2.drawContours(tep, [c], -1, (100, 100, 100), 1)
#             plt.imshow(tep)
#             plt.show()
#             if (x, y, w, h) not in t:
#                 t.append((x, y, w, h))
            is_equal, transformation = compare_kernel(inp[y:y+h, x:x+w], out)
            if is_equal == 1:
                res.append((inp[y:y+h, x:x+w], True, transformation))
            else:
                res.append((inp[y:y+h, x:x+w], False, -1))
        
        decision.append(res)
        
    
    # test_res
    test_res = []
    test_col = []
    inp = copy.deepcopy(np.array(test_inp))
    skeleton = (inp != 0).astype('int')

    frame = copy.deepcopy(skeleton)
    frame = np.array(frame, dtype='uint8') * 10
    ret, frame = cv2.threshold(frame,5,255,0)
    contours,hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    t = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
#         print((x, y, w, h))
#         tep = np.zeros(frame.shape)
#         cv2.drawContours(tep, [c], -1, (100, 100, 100), 1)
#         plt.imshow(tep)
#         plt.show()
        if (x, y, w, h) not in t:
            t.append((x, y, w, h))
            test_res.append(inp[y:y+h, x:x+w])
    
#     print(decision)
    # count
    findings = []
    findings_col = []
    inp = copy.deepcopy(np.array(test_inp))
    for frame in decision:
        frame_res = []
        frame_count = []
        frame_area = []
        frame_flag = []
        frame_color = []
        for con in frame:
            frame_res.append(con[0])
            frame_flag.append(con[1])
            frame_area.append(con[0].shape[0]*con[0].shape[1])
            frame_color.append(get_col_profile(con[0]))
        for i in frame_res:
            temp_count = 0
            for j in frame_res:
                is_equal, _ = compare_kernel(i, j)
                if is_equal == 1:
                    temp_count += 1
            frame_count.append(temp_count)
        true_index = frame_flag.index(True)

        if nunique(frame_count) != 1 and frame_count[true_index] == min(frame_count):
            findings.append('min')
        elif nunique(frame_count) != 1 and frame_count[true_index] == max(frame_count):
            findings.append('max')
#         elif nunique(frame_area) != 1 and frame_area[true_index] == min(frame_area):
#             findings.append('minarea')
#         elif nunique(frame_area) != 1 and frame_area[true_index] == max(frame_area):
#             findings.append('maxarea')
        else:
            findings.append('---')
            
        if all([list(frame_color[0].keys()) == k for k in [list(ccc.keys()) for ccc in frame_color]]):
#             findings.append('---')
            color_ress = []
            print("in res")
            for col in list(frame_color[0].keys()):
                temp_125 = []
                for i in range(len(frame_color)):
                    temp_125.append(frame_color[i][col])
                if temp_125[true_index] == min(temp_125) and temp_125.count(min(temp_125)) == 1:
                    color_ress.append('min_'+str(col))
                    
                elif temp_125[true_index] == max(temp_125) and temp_125.count(max(temp_125)) == 1:
                    color_ress.append('max_'+str(col))
            findings_col.append(color_ress)
        
    
#     predict
    print(len(test_res))
    frame_count = []
    for i in test_res:
        temp_count = 0
        for j in test_res:
            is_equal, _ = compare_kernel(i, j)
            if is_equal == 1:
                temp_count += 1
        frame_count.append(temp_count)
#     print("-->", frame_count)
#     print(findings)
    if nunique(findings) == 1 and unique(findings)[0] == 'min':
        return test_res[frame_count.index(min(frame_count))]
        
    if nunique(findings) == 1 and unique(findings)[0] == 'max':
        return test_res[frame_count.index(max(frame_count))]
      
    count_col = Counter([c for f in findings_col for c in f])
    fin_des = []
#     print("22", count_col)
    for col in count_col:
        if count_col[col] == 3:
            fin_des.append(col)
    if len(fin_des) == 1:
        print("------------------------------->", fin_des)
        frame_col = []
        for co in test_res:
            frame_col.append(get_col_profile(co))
#         print(frame_col)
        #         col_res = []
        for col in list(frame_col[0].keys()):
            if str(col) in fin_des[0]:
                temp_125 = []
                for i in range(len(frame_color)):
                    temp_125.append(frame_col[i][col])
#                 print(temp_125)
                if "min" in fin_des[0]:
                    return test_res[temp_125.index(min(temp_125))]
                elif "max" in fin_des[0]:
                    return test_res[temp_125.index(max(temp_125))]
        
    
    if findings[min(len(findings)-1, train_task)] == 'min':
        return test_res[frame_count.index(min(frame_count))]
    
    if findings[min(len(findings)-1, train_task)] == 'max':
        return test_res[frame_count.index(max(frame_count))]
    
    
    return test_res[min(len(test_res)-1, train_task)]


# In[ ]:


def case_two(task_name):
    print(task_name)
    task = read_json(task_name.split('_')[0]+'.json', 'test')
   # case 1
    if all([check_in(t) for t in task['train']]):
#         plot_task(task)
        if all([get_size(t['output'])==(1, 1) for t in task['train']]):
            print("case BA01", task_name)
            plot_task(task)
        elif find_common([unique_location(t) for t in task['train']]):
            print("case BA02", task_name)
#             print([unique_location(t) for t in task['train']])
            plot_task(task)
        elif all([self_repeating(t) for t in task['train']]):
            print("case BA03", task_name)
        elif all([unique_box(t) for t in task['train']]):
            print("case BA04", task_name)
            plot_task(task)
            predict("case BA04", task, task_name)
        elif all([unique_color(t) for t in task['train']]):
            print("case BA05", task_name)
            plot_task(task)
            predict("case BA05", task, task_name)

        elif all([is_in(t) for t in task['train']]):
            print("case BA06", task_name)
            
            predict("case BA06", task, task_name)
    else:
        print("case ----", task_name)
#         plot_task(task)
#         print("case ----", task_name)


# In[ ]:


# case_two('1a6449f1_0')


# In[ ]:


# case_two('2c0b0aff_0')
# task = read_json('2c0b0aff.json', 'test')
# plot_task(task)
# predict("case BA06", task, 'taskname_0')


# In[ ]:


for i, row in submission.iterrows():
    task = read_json(row.name.split('_')[0]+'.json', 'test')
#     plot_task(task)
    if all([(get_size(t['input']) == get_size(t['output'])) for t in task['train']]):
        pass
        
    elif all([(get_size(t['input']) > get_size(t['output'])) for t in task['train']]):
        case_two(row.name)
        pass
        
    elif all([(get_size(t['input']) < get_size(t['output'])) for t in task['train']]):
#         case_three(row.name)
        pass
    else:
        pass
    

# submission.to_csv('submission.csv')


# In[ ]:


sample_sub1 = sample_sub1.reset_index()
sample_sub1 = sample_sub1.sort_values(by="output_id")

sample_sub2 = submission.sort_values(by="output_id")
out1 = sample_sub1["output"].astype(str).values
out2 = sample_sub2["output"].astype(str).values

merge_output = []
for o1, o2 in zip(out1, out2):
    if o2 == '':
        o = o1
    else:
        o = o2
#         o = o1.strip().split(" ")[:1] + o2.strip().split(" ")[:2]
#         o = " ".join(o[:3])
    merge_output.append(o)
sample_sub1["output"] = merge_output
sample_sub1["output"] = sample_sub1["output"].astype(str)
sample_sub1.to_csv("submission.csv", index=False)


# In[ ]:





# In[ ]:




