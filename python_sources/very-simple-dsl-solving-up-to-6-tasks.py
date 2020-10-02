#!/usr/bin/env python
# coding: utf-8

# # DSL for Object Detection

# In[ ]:


import os
import json
from pathlib import Path
from time import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors

np.random.seed(seed = 440)

cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge')
submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='output_id')
#display(submission.head())


# In[ ]:


get_ipython().system('cp /kaggle/input/abstraction-and-reasoning-challenge/sample_submission.csv ./submission.csv')


# This DSL focuses on object detection.
# 
# Object detection = output is in input. 

# In[ ]:


def check_sub(a,b):
    arrx = []
    if(a.shape[0] < b.shape[0]):
        for i in range(b.shape[0] + 1 - a.shape[0]):
            for t in range(b.shape[1] + 1 - a.shape[1]):
                arrx.append((a == b[i : i + a.shape[0], t : t + a.shape[1]]).all())                
    return np.array(arrx).any()


# In[ ]:


arr = []
test_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/test')
test_tasks = sorted(os.listdir(test_path))
arr = []
for s in range(len(test_tasks)):
    task_file = str(test_path / test_tasks[s])
    #print(task_file)
    
    with open(task_file, 'r') as f:
        task = json.load(f)
        for i in task["train"]:
            inp = np.array(i["input"])
            out = np.array(i["output"])

            arr.append([s, inp.shape[0], inp.shape[1], out.shape[0], out.shape[1], int(check_sub(out, inp)), (np.unique(out) != 0).sum(), int((out == out[:, ::-1]).all())])


# In[ ]:


def plot_one(ax, i,train_or_test,input_or_output):
   
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + ' '+input_or_output)
    

def plot_task(task):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """    
    num_train = len(task['train'])
    fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))
    for i in range(num_train):     
        plot_one(axs[0,i],i,'train','input')
        plot_one(axs[1,i],i,'train','output')        
    plt.tight_layout()
    plt.show()        


# In[ ]:


df = pd.DataFrame(arr)
dfz = df.groupby(0, as_index = False).mean()


# ## Examples from Test Set

# In[ ]:


for s in (dfz[dfz[5] == 1][0].values[:]):
    print(test_tasks[s])
    task_file = str(test_path / test_tasks[s])
    
    with open(task_file, 'r') as f:
        task = json.load(f)
        
    #print(i)
    #print(training_tasks[i])
    plot_task(task)


# In[ ]:


def check_in_array(arri, testx):
    if(len(arri) > 0):
        arrs = []
        for i in range(len(arri)):
            arrs.append(np.array_equal(arri[i],testx))
        return(~np.array(arrs).any())
    else:
        return True


# In[ ]:


def checkList(lst): 
  
    ele = lst[0] 
    chk = True
    for item in lst: 
        if ele != item: 
            chk = False
            break; 
              
    if (chk == True):
        return True
    else: 
        return False


# # Assign Properties

# I will introduce this example object detection to explain some of the functions, the task is detect the red square:

# #### Input

# In[ ]:


sample = np.zeros(shape = (8, 8))
sample[2:6, 2:6] = 2

plt.figure()
im = plt.imshow(sample, cmap, norm)

ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, sample.shape[0], 1));
ax.set_yticks(np.arange(0, sample.shape[0], 1));

# Labels for major ticks
ax.set_xticklabels(np.arange(1, sample.shape[1], 1));
ax.set_yticklabels(np.arange(1, sample.shape[1], 1));

# Minor ticks
ax.set_xticks(np.arange(-.5, sample.shape[0], 1), minor=True);
ax.set_yticks(np.arange(-.5, sample.shape[0], 1), minor=True);

# Gridlines based on minor ticks
ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=1)


# #### Output

# In[ ]:


sample = np.ones(shape = (4, 4))*2
#sample[2:6, 2:6] = 2

plt.figure()
im = plt.imshow(sample, cmap, norm)

ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, sample.shape[0], 1));
ax.set_yticks(np.arange(0, sample.shape[0], 1));

# Labels for major ticks
ax.set_xticklabels(np.arange(1, sample.shape[1], 1));
ax.set_yticklabels(np.arange(1, sample.shape[1], 1));

# Minor ticks
ax.set_xticks(np.arange(-.5, sample.shape[0], 1), minor=True);
ax.set_yticks(np.arange(-.5, sample.shape[0], 1), minor=True);

# Gridlines based on minor ticks
ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=1)


# The concept of this pipeline:

# 1. Assign properties to the train outputs (like output has a single color)

# 2. Random crop the test input
# 

# In[ ]:


sample = np.zeros(shape = (8, 8))
sample[2:6, 2:6] = 2
sample = sample[3:8, 3:8]

plt.figure()
im = plt.imshow(sample, cmap, norm)

ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, sample.shape[0], 1));
ax.set_yticks(np.arange(0, sample.shape[0], 1));

# Labels for major ticks
ax.set_xticklabels(np.arange(1, sample.shape[1], 1));
ax.set_yticklabels(np.arange(1, sample.shape[1], 1));

# Minor ticks
ax.set_xticks(np.arange(-.5, sample.shape[0], 1), minor=True);
ax.set_yticks(np.arange(-.5, sample.shape[0], 1), minor=True);

# Gridlines based on minor ticks
ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=1)


# 3. Check if properties of the crop are the same as the assigned properties
# 
# if yes: 
#      append as solution     
#  if no:
#      repeat from step 2

# ## Definitions

# #### outer border

# In[ ]:


sample = np.zeros(shape = (8, 8))
sample[2:6, 2:6] = 2
sample[1, 2:6] = 3
sample[6, 2:6] = 3
sample[2:6, 1] = 3
sample[2:6, 6] = 3

plt.figure()
im = plt.imshow(sample, cmap, norm)

ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, sample.shape[0], 1));
ax.set_yticks(np.arange(0, sample.shape[0], 1));

# Labels for major ticks
ax.set_xticklabels(np.arange(1, sample.shape[1], 1));
ax.set_yticklabels(np.arange(1, sample.shape[1], 1));

# Minor ticks
ax.set_xticks(np.arange(-.5, sample.shape[0], 1), minor=True);
ax.set_yticks(np.arange(-.5, sample.shape[0], 1), minor=True);

# Gridlines based on minor ticks
ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=1)


# #### inner border

# In[ ]:


sample = np.zeros(shape = (8, 8))
sample[2:6, 2:6] = 2
sample[2, 2:6] = 3
sample[5, 3:5] = 3
sample[2:6, 2] = 3
sample[2:6, 5] = 3

plt.figure()
im = plt.imshow(sample, cmap, norm)

ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, sample.shape[0], 1));
ax.set_yticks(np.arange(0, sample.shape[0], 1));

# Labels for major ticks
ax.set_xticklabels(np.arange(1, sample.shape[1], 1));
ax.set_yticklabels(np.arange(1, sample.shape[1], 1));

# Minor ticks
ax.set_xticks(np.arange(-.5, sample.shape[0], 1), minor=True);
ax.set_yticks(np.arange(-.5, sample.shape[0], 1), minor=True);

# Gridlines based on minor ticks
ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=1)


# # Property functions

# Visualised what the functions assess as properties:

# ## all outer borders edges individually have one color

# In[ ]:


sample = np.zeros(shape = (8, 8))
sample[2:6, 2:6] = 2
sample[1, 2:6] = 3
sample[6, 2:6] = 4
sample[2:6, 1] = 5
sample[2:6, 6] = 6

plt.figure()
im = plt.imshow(sample, cmap, norm)

ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, sample.shape[0], 1));
ax.set_yticks(np.arange(0, sample.shape[0], 1));

# Labels for major ticks
ax.set_xticklabels(np.arange(1, sample.shape[1], 1));
ax.set_yticklabels(np.arange(1, sample.shape[1], 1));

# Minor ticks
ax.set_xticks(np.arange(-.5, sample.shape[0], 1), minor=True);
ax.set_yticks(np.arange(-.5, sample.shape[0], 1), minor=True);

# Gridlines based on minor ticks
ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=1)


# In[ ]:


def check_outer_border(a,b):
    arrx = []
    if(a.shape[0] < b.shape[0]):
        for i in range(b.shape[0] - a.shape[0] + 1):
            for t in range(b.shape[1] - a.shape[1] + 1):
                if((b[i:i+a.shape[0], t:t+a.shape[1]] == a).all()):
                    arr = []
                    if((i > 0)):
                        arr.append(np.unique(b[i-1, t:t+a.shape[1]]).tolist())
                    else:
                        arr.append([0])
                    if((i + a.shape[0] < b.shape[0])):
                        arr.append(np.unique(b[i+a.shape[0], t:t+a.shape[1]]).tolist())
                    else:
                        arr.append([0])
                    if((t > 0)):
                        arr.append(np.unique(b[i:i+a.shape[0], t-1]).tolist())
                    else:
                        arr.append([0])
                    if((t + a.shape[1] < b.shape[1])):
                        arr.append(np.unique(b[i:i+a.shape[0], t+a.shape[1]]).tolist())
                    else:
                        arr.append([0])
                        
                    if((len(arr[0]) == 1)& (len(arr[1]) == 1)& (len(arr[2]) == 1)& (len(arr[3]) == 1)):
                        arrx.append(True)
                    else:
                        arrx.append(False)
    if(np.array(arrx).any()):
        return True
    else:
        return False


# ## all individual outer border have the same color

# In[ ]:


sample = np.zeros(shape = (8, 8))
sample[2:6, 2:6] = 2
sample[1, 2:6] = 4
sample[6, 2:6] = 4
sample[2:6, 1] = 4
sample[2:6, 6] = 4

plt.figure()
im = plt.imshow(sample, cmap, norm)

ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, sample.shape[0], 1));
ax.set_yticks(np.arange(0, sample.shape[0], 1));

# Labels for major ticks
ax.set_xticklabels(np.arange(1, sample.shape[1], 1));
ax.set_yticklabels(np.arange(1, sample.shape[1], 1));

# Minor ticks
ax.set_xticks(np.arange(-.5, sample.shape[0], 1), minor=True);
ax.set_yticks(np.arange(-.5, sample.shape[0], 1), minor=True);

# Gridlines based on minor ticks
ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=1)


# In[ ]:


def check_outer_border2(a,b):
    arrx = []
    if(a.shape[0] < b.shape[0]):
        for i in range(b.shape[0] - a.shape[0] + 1):
            for t in range(b.shape[1] - a.shape[1] + 1):
                if((b[i:i+a.shape[0], t:t+a.shape[1]] == a).all()):
                    arr = []
                    if((i > 0)):
                        arr.append(np.unique(b[i-1, t:t+a.shape[1]]).tolist())
                    else:
                        arr.append([0])
                    if((i + a.shape[0] < b.shape[0])):
                        arr.append(np.unique(b[i+a.shape[0], t:t+a.shape[1]]).tolist())
                    else:
                        arr.append([0])
                    if((t > 0)):
                        arr.append(np.unique(b[i:i+a.shape[0], t-1]).tolist())
                    else:
                        arr.append([0])
                    if((t + a.shape[1] < b.shape[1])):
                        arr.append(np.unique(b[i:i+a.shape[0], t+a.shape[1]]).tolist())
                    else:
                        arr.append([0])
                        
                    if((len(np.unique(arr)) == 1) & (len(arr[0]) == 1)):
                        arrx.append(True)
                    else:
                        arrx.append(False)
    if(np.array(arrx).any()):
        return True
    else:
        return False


# ## all individual outer border are black

# In[ ]:


sample = np.zeros(shape = (8, 8))
sample[2:6, 2:6] = 2

plt.figure()
im = plt.imshow(sample, cmap, norm)

ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, sample.shape[0], 1));
ax.set_yticks(np.arange(0, sample.shape[0], 1));

# Labels for major ticks
ax.set_xticklabels(np.arange(1, sample.shape[1], 1));
ax.set_yticklabels(np.arange(1, sample.shape[1], 1));

# Minor ticks
ax.set_xticks(np.arange(-.5, sample.shape[0], 1), minor=True);
ax.set_yticks(np.arange(-.5, sample.shape[0], 1), minor=True);

# Gridlines based on minor ticks
ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=1)


# In[ ]:


def check_outer_border3(a,b):
    arrx = []
    if(a.shape[0] < b.shape[0]):
        for i in range(b.shape[0] - a.shape[0] + 1):
            for t in range(b.shape[1] - a.shape[1] + 1):
                if((b[i:i+a.shape[0], t:t+a.shape[1]] == a).all()):
                    arr = []
                    if((i > 0)):
                        arr.append(np.unique(b[i-1, t:t+a.shape[1]]).tolist())
                    else:
                        arr.append([0])
                    if((i + a.shape[0] < b.shape[0])):
                        arr.append(np.unique(b[i+a.shape[0], t:t+a.shape[1]]).tolist())
                    else:
                        arr.append([0])
                    if((t > 0)):
                        arr.append(np.unique(b[i:i+a.shape[0], t-1]).tolist())
                    else:
                        arr.append([0])
                    if((t + a.shape[1] < b.shape[1])):
                        arr.append(np.unique(b[i:i+a.shape[0], t+a.shape[1]]).tolist())
                    else:
                        arr.append([0])
                        
                    if((len(np.unique(arr)) == 1) & (len(arr[0]) == 1)):
                        if(arr[0][0] == 0):
                            arrx.append(True)
                        else:
                            arrx.append(False)
                    else:
                        arrx.append(False)
    if(np.array(arrx).any()):
        return True
    else:
        return False


# ## all individual outer border have more than 1 color

# In[ ]:


sample = np.zeros(shape = (8, 8))
sample[2:6, 2:6] = 2
sample[1, 2:6] = 3
sample[1, 2:3] = 1
sample[1, 4:5] = 5
sample[6, 2:6] = 4
sample[6, 5:6] = 1
sample[2:6, 1] = 5
sample[2:3, 1] = 4
sample[2:6, 6] = 6
sample[4:6, 6] = 8

plt.figure()
im = plt.imshow(sample, cmap, norm)

ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, sample.shape[0], 1));
ax.set_yticks(np.arange(0, sample.shape[0], 1));

# Labels for major ticks
ax.set_xticklabels(np.arange(1, sample.shape[1], 1));
ax.set_yticklabels(np.arange(1, sample.shape[1], 1));

# Minor ticks
ax.set_xticks(np.arange(-.5, sample.shape[0], 1), minor=True);
ax.set_yticks(np.arange(-.5, sample.shape[0], 1), minor=True);

# Gridlines based on minor ticks
ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=1)


# In[ ]:


def check_outer_border4(a,b):
    arrx = []
    if(a.shape[0] < b.shape[0]):
        for i in range(b.shape[0] - a.shape[0] + 1):
            for t in range(b.shape[1] - a.shape[1] + 1):
                if((b[i:i+a.shape[0], t:t+a.shape[1]] == a).all()):
                    arr = []
                    if((i > 0)):
                        arr.append(np.unique(b[i-1, t:t+a.shape[1]]).tolist())
                    else:
                        arr.append([0])
                    if((i + a.shape[0] < b.shape[0])):
                        arr.append(np.unique(b[i+a.shape[0], t:t+a.shape[1]]).tolist())
                    else:
                        arr.append([0])
                    if((t > 0)):
                        arr.append(np.unique(b[i:i+a.shape[0], t-1]).tolist())
                    else:
                        arr.append([0])
                    if((t + a.shape[1] < b.shape[1])):
                        arr.append(np.unique(b[i:i+a.shape[0], t+a.shape[1]]).tolist())
                    else:
                        arr.append([0])
                        
                    if((len(arr[0]) != 1) & (len(arr[1]) != 1) & (len(arr[2]) != 1) & (len(arr[3]) != 1)):
                        arrx.append(True)
                    else:
                        arrx.append(False)
    if(np.array(arrx).any()):
        return True
    else:
        return False


# In[ ]:


def all_same(items):
    return all(x == items[0] for x in items), items[0]


# In[ ]:


def all_same_length(items):
    return all(len(x) == len(items[0]) for x in items), len(items[0])


# ## Output multiple times in input

# In[ ]:


def multiple_times_in_input(a,b):   
    if(b.shape[0] < a.shape[0]):
        count = 0
        for y in range(0, (a.shape[0] - b.shape[0]) + 1):
            for x in range(0, (a.shape[1] - b.shape[1]) + 1):
                selection = a[y:y+b.shape[0], x:x+b.shape[1]]
                if((selection == b).all()):
                    count += 1
        if(count > 1):
            return True
        else:
            return False
    else:
        return False


# ### Output multiple times in Input with flips

# In[ ]:


def multiple_times_in_input2(a,b):   
    if(b.shape[0] < a.shape[0]):
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for y in range(0, (a.shape[0] - b.shape[0]) + 1):
            for x in range(0, (a.shape[1] - b.shape[1]) + 1):
                selection = a[y:y+b.shape[0], x:x+b.shape[1]]
                if((selection == b).all()):
                    count1 += 1
                    
                if((selection[:, ::-1] == b).all()):
                    count2 += 1
                    
                if((selection[:, ::-1] == b).all()):
                    count3 += 1
                    
                if((selection[:, ::-1] == b).all()):
                    count4 += 1
        if(((np.array([count1, count2, count3, count4]) > 1).sum()) > 1):
            return True
        else:
            return False
    else:
        return False


# # Crop and Check

# Get all properties mentioned above and some additional:
#     1. vertical + horizontal symetry
#     2. output is square shaped
#     3. No zeros in inner border
#     4. Same colors in output
#     5. Same number of colors in output
#     6. no zeros in output

# In[ ]:


factor1 = 0
factor2 = 0
test_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/test')
test_tasks = sorted(os.listdir(test_path))
arr = []
arrf = []
finalpropertyarr = []
countingarr = []

for s in (dfz[dfz[5] == 1][0].values[:]):
    print(s)
    task_file = str(test_path / test_tasks[s])
    
    with open(task_file, 'r') as f:
        task = json.load(f)
        arri = []
        arrx = []
        allcolors = []
        allcolorsinnerborder = []
        
        for i in task["train"]:
            
            #assign more properties
            
            inp = np.array(i["input"])
            out = np.array(i["output"])
            
            cib = (out[:, 0] != 0).all() & (out[:, -1] != 0).all() & (out[0, :] != 0).all() & (out[-1, :] != 0).all()
            
            cvsym = (out == out[:, ::-1]).all()
            chsym = (out == out[::-1, :]).all()
            
            cob_kanten_individuell_nur_eine_farbe = check_outer_border(out, inp)
            cob_kanten_alle_nur_eine_farbe = check_outer_border2(out, inp)
            cob_kanten_alle_nur_zero = check_outer_border3(out, inp)
            cob_kanten_nichtnureinefarbe = check_outer_border4(out, inp)
            
            out_multiple_times_in_input = multiple_times_in_input(inp, out)
            out_multiple_times_in_input2 = multiple_times_in_input2(inp, out)
            
            nozero = (out != 0).all()
            nozeroinnerborder = ((out[:, 0] != 0).all() & (out[:, -1] != 0).all() & (out[0, :] != 0).all() & (out[-1, :] != 0).all())
            
            onecolor = len(np.unique(out[out!= 0])) == 1
            onecolorinnerborder_individuell = (len(np.unique(out[:, 0])) == 1) & (len(np.unique(out[0, :])) == 1) & (len(np.unique(out[:, -1])) == 1) & (len(np.unique(out[-1, :])) == 1)
            
            onecolorinnerborder_alle = all_same([np.unique(out[:, 0]).tolist(),np.unique(out[0, :]).tolist(),np.unique(out[:, -1]).tolist(), np.unique(out[-1, :]).tolist()])[0] & onecolorinnerborder_individuell
            
            if(onecolorinnerborder_alle):
                onecolorinnerborder_alle_value = np.unique(out[:, 0])
            else:
                onecolorinnerborder_alle_value = 20
                
            outputcolors = np.unique(out)
            
            is_square = (out.shape[0] == out.shape[1])
            
            arrx.append([onecolor, cib, cob_kanten_individuell_nur_eine_farbe, cob_kanten_alle_nur_eine_farbe, 
                         cvsym, chsym, nozero, nozeroinnerborder, onecolorinnerborder_individuell, cob_kanten_alle_nur_zero, 
                         cob_kanten_nichtnureinefarbe, onecolorinnerborder_alle, is_square, out_multiple_times_in_input, out_multiple_times_in_input2])
            
            allcolors.append(outputcolors.tolist())
            allcolorsinnerborder.append(onecolorinnerborder_alle_value)
            
        dfy = pd.DataFrame(arrx)

        plt.imshow(inp, cmap, norm)
        plt.show()
        plt.imshow(out, cmap, norm)
        plt.show()

        ccolor, ccolorvalue = all_same(allcolors)
        ccolorlength, ccolorlengthvalue = all_same_length(allcolors)
        
        for idx_o, o in enumerate(task["test"]):
            inp = np.array(o["input"])
            plt.imshow(inp, cmap, norm)
            plt.show()
            
            properties = dfy.mean(axis = 0).values
            finalpropertyarr.append(properties)
            
            for f in range(800000):
                
                #random crop
                
                rand1 = np.random.randint(0, inp.shape[0]-2)
                rand2 = np.random.randint(0, inp.shape[1]-2)
                
                rand3 = np.random.randint(2, inp.shape[0]-rand1 + 1)
                rand4 = np.random.randint(2, inp.shape[1]-rand2 + 1)
                
                testx = inp[rand1:rand3+rand1, rand2:rand4+rand2]
                
                #check if properties are the same for the crop
                
                test_onecolor = len(np.unique(testx[testx!= 0])) == 1
                test_cib = (testx[:, 0] != 0).all() & (testx[:, -1] != 0).all() & (testx[0, :] != 0).all() & (testx[-1, :] != 0).all()
                test_nozeroinnerborder = ((testx[:, 0] != 0).all() & (testx[:, -1] != 0).all() & (testx[0, :] != 0).all() & (testx[-1, :] != 0).all())
                test_onecolorinnerborder_individuell =  (len(np.unique(testx[:, 0])) == 1) & (len(np.unique(testx[:, 0])) == 1) & (len(np.unique(testx[:, -1])) == 1) & (len(np.unique(testx[-1, :])) == 1)
                test_onecolorinnerborder_alle = all_same([np.unique(testx[:, 0]).tolist(),np.unique(testx[0, :]).tolist(),np.unique(testx[:, -1]).tolist(), np.unique(testx[-1, :]).tolist()])[0] & test_onecolorinnerborder_individuell  
                if(((properties[12] == 1) & (testx.shape[0] == testx.shape[1])) | ((properties[12] < 1))):
                    if(((properties[11] == 1) & (test_onecolorinnerborder_alle)) | ((properties[11] < 1))):
                        if((ccolorlength & (ccolorlengthvalue == len(np.unique(testx)))) | (ccolorlength == False)):
                            if((ccolor & (ccolorvalue == np.unique(testx).tolist())) | (ccolor == False)):
                                if(((properties[8] == 1) & (test_onecolorinnerborder_individuell)) | ((properties[8] < 1))):
                                    if(((properties[7] == 1) & (test_nozeroinnerborder)) | ((properties[7] < 1))):
                                        if(((properties[6] == 1) & (testx != 0).all()) | ((properties[6] < 1))):
                                            if(((properties[5] == 1) & (testx == testx[::-1, :]).all()) | ((properties[5] < 1))):
                                                if(((properties[4] == 1) & (testx == testx[:, ::-1]).all()) | ((properties[4] < 1))):
                                                    if((testx.shape[0] > 2) & (testx.shape[1] > 2)):
                                                        if(((properties[0] == 1) & (test_onecolor)) | ((properties[0] < 1))):# | ((dfy.mean(axis = 0)[0] == 0))):
                                                            if(((properties[0] == 0) & (test_onecolor == False)) | ((properties[0] > 0))):
                                                                if(((properties[1] == 1) & (test_cib)) | ((properties[1] < 1))):
                                                                    test_cob_kanten_individuell_nur_eine_farbe = check_outer_border(testx, inp)
                                                                    test_cob_kanten_alle_nur_eine_farbe = check_outer_border2(testx, inp)
                                                                    test_cob_kanten_alle_nur_zero = check_outer_border3(testx, inp)
                                                                    test_cob_kanten_nichtnureinefarbe = check_outer_border4(testx, inp)
                                                                    if(((properties[10] == 1) & (test_cob_kanten_nichtnureinefarbe)) | ((properties[10] <1))):
                                                                        if(((properties[9] == 1) & (test_cob_kanten_alle_nur_zero)) | ((properties[9] <1))):
                                                                            if(((properties[2] == 1) & (test_cob_kanten_individuell_nur_eine_farbe)) | ((properties[2] <1))):
                                                                                if(((properties[3] == 1) & (test_cob_kanten_alle_nur_eine_farbe)) | ((properties[3] <1 ))):
                                                                                    if(check_in_array(arri, testx)):
                                                                                        test_multiple_times_in_output = multiple_times_in_input(inp, testx)
                                                                                        test_multiple_times_in_output2 = multiple_times_in_input2(inp, testx)
                                                                                        if(((properties[14] == 1) & (test_multiple_times_in_output2)) | ((properties[14] < 1))):
                                                                                            if(((properties[13] == 1) & (test_multiple_times_in_output)) | ((properties[13] < 1))):
                                                                                                if(((properties[13] == 0) & (test_multiple_times_in_output == False)) | ((properties[13] > 0))):
                                                                                                    arri.append(testx)
                                                                                                    plt.imshow(testx, cmap, norm)
                                                                                                    plt.show()
                                                                                                    print("number of iterations:", f)
                      
                    
                #remove random properties if no solution is found
                
                if(f == 500000):
                    properties[np.random.choice([1,2,3,4,5,7,8,13,12,11])] = 0.5
                if(f == 600000):
                    properties[np.random.choice([1,2,3,4,5,7,8,13,12,11])] = 0.5
                if(f == 700000):
                    properties[np.random.choice([1,2,3,4,5,7,8,13,12,11])] = 0.5
                    
                if(len(arri) >= 3):
                    break
            countingarr.append(f)
            arrf.append([submission.index[s], test_tasks[s][:-5] + "_" + str(idx_o), test_tasks[s], arri])


# In[ ]:


def flattener(pred):
    str_pred = str([list(row) for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


# In[ ]:


selectedvalues = np.random.choice(np.array([0,3,4,5]), factor2)


# In[ ]:


submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='output_id')
display(submission.head())

g = 0
for idx, i in enumerate(arrf):
    if(~(idx == selectedvalues).any()):
        pred = ""
        for k in i[3]:
            pred += flattener(k) + " "
            g += 1
        submission.loc[i[1], 'output'] = pred
        
        task_file = str(test_path) + "/" + i[2]
        
        with open(task_file, 'r') as f:
            task = json.load(f)
            for i in task["train"]:
                inp = np.array(i["input"])
                out = np.array(i["output"])
            plt.imshow(inp)
            plt.show()
    else:
        continue


# In[ ]:


submission.to_csv('submission.csv')

