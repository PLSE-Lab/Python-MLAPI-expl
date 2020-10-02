#!/usr/bin/env python
# coding: utf-8

# # The idea:
# 
# Study a context free algorithm that knows nothing about shapes, numbers, lines or planar geometry. 
# However it assumes that:
# 
# 1, All the input pictures have the same size (n,k)
# 
# 2, All the output pictures have the same size (a,b)
# 
# 3, The color of Output_Picture at pixel (p,q) is given by the color of Input_Picture at pixel (i,j).
# 
# The goal is to find the mapping from (p,q) to the correct (i,j). The algorithm is quite simple:
# 
# 1, At the training time for given (p,q) it collects all the pairs (i,j) that are possible candidates. 
# 
# 2, At the prediction time it uses majority rule. (computes the colors of Test_Picture at the candidate places).
# 

# # Scores:
# 
# Training: 11/400
# 
# Evaluation: 17/400
# 
# Test: 0
# 
# It seems there is no free lunch. But still, perhaps somebody can use this as a building block.
# 
# 
# 
# 

# In[ ]:


import numpy as np
import pandas as pd

import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from pathlib import Path

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)


# # Getting the data

# In[ ]:


data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'
training_tasks = sorted(os.listdir(training_path))
eval_tasks = sorted(os.listdir(evaluation_path))


# In[ ]:


T = training_tasks
Trains = []
for i in range(400):
    task_file = str(training_path / T[i])
    task = json.load(open(task_file, 'r'))
    Trains.append(task)
    
E = eval_tasks
Evals= []
for i in range(400):
    task_file = str(evaluation_path / E[i])
    task = json.load(open(task_file, 'r'))
    Evals.append(task)


# # Helper Functions

# In[ ]:


cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
# 0:black, 1:blue, 2:red, 3:greed, 4:yellow,
# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
plt.figure(figsize=(5, 2), dpi=200)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.show()

def plot_task(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    
    plt.tight_layout()
    plt.show()


# In[ ]:


def plot_picture(x):
    plt.imshow(np.array(x), cmap = cmap, norm = norm)
    plt.show()


# In[ ]:


def Defensive_Copy(A): 
    n = len(A)
    k = len(A[0])
    L = np.zeros((n,k), dtype = int)
    for i in range(n):
        for j in range(k):
            L[i,j] = 0 + A[i][j]
    return L.tolist()


# In[ ]:


def Create(task, task_id = 0):
    n = len(task['train'])
    Input = [Defensive_Copy(task['train'][i]['input']) for i in range(n)]
    Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]
    Input.append(Defensive_Copy(task['test'][task_id]['input']))
    return Input, Output


# # The algorithm:
#  

# In[ ]:


def Recolor(task):
    Input = task[0]
    Output = task[1]
    Test_Picture = Input[-1]
    Input = Input[:-1]
    N = len(Input)

    x0 = Input[0]
    y0 = Output[0]
    n = len(x0)
    k = len(x0[0])
    a = len(y0)
    b = len(y0[0])
    for x in Input+[Test_Picture]:
        if len(x) != n or len(x[0]) != k:
            return -1
    for y in Output:
        if len(y) != a or len(y[0]) != b:
            return -1
    List1 = {}
    List2 = {}
    
    for i in range(n):
        for j in range(k):
            seq = []
            for x in Input:
                seq.append(x[i][j])
            List1[(i,j)] = seq
            
    for p in range(a):
        for q in range(b):
            seq1 = []
            for y in Output:
                seq1.append(y[p][q])
           
            places = []
            for key in List1:
                if List1[key] == seq1:
                    places.append(key) 
                    
            List2[(p,q)] = places
            if len(places) == 0:
                return -1
                
    answer = np.zeros((a,b), dtype = int)
   
    for p in range(a):
        for q in range(b):
            palette = [0,0,0,0,0,0,0,0,0,0]
            for i, j in List2[(p,q)]:
                color = Test_Picture[i][j]
                palette[color]+=1
            answer[p,q] =  np.argmax(palette)
            
    return answer.tolist()


# # Results on Training set

# In[ ]:


training_examples = []
for i in range(400):
    task = Trains[i]
    basic_task = Create(task,0)
    a = Recolor(basic_task)
  
    if a != -1 and task['test'][0]['output'] == a:
        plot_picture(a)
        plot_task(task)
        print(i)
        training_examples.append(i)      


# In[ ]:


print(len(training_examples))
print(training_examples)


# # Results on the evaluation set

# In[ ]:


evaluation_examples = []


for i in range(400):
    task = Evals[i]
    basic_task = Create(task,0)
    a = Recolor(basic_task)
    
    if a != -1 and task['test'][0]['output'] == a:
        plot_picture(a)
        plot_task(task)
        evaluation_examples.append(i)        


# In[ ]:


print(len(evaluation_examples))
print(evaluation_examples)


# As remarked elsewhere there are some subtle differences between the Evaluation tasks and the Testing tasks and
# this approach doesn't give a hit on the leaderboard.  

# In[ ]:


submission = pd.read_csv(data_path/ 'sample_submission.csv')
submission.head()


# In[ ]:


def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


# In[ ]:


example_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
display(example_grid)
print(flattener(example_grid))


# In[ ]:


Solved = []
Problems = submission['output_id'].values
Proposed_Answers = []
for i in  range(len(Problems)):
    output_id = Problems[i]
    task_id = output_id.split('_')[0]
    pair_id = int(output_id.split('_')[1])
    f = str(test_path / str(task_id + '.json'))
   
    with open(f, 'r') as read_file:
        task = json.load(read_file)
    
    n = len(task['train'])
    Input = [Defensive_Copy(task['train'][j]['input']) for j in range(n)]
    Output = [Defensive_Copy(task['train'][j]['output']) for j in range(n)]
    Input.append(Defensive_Copy(task['test'][pair_id]['input']))
    
    solution = Recolor([Input, Output])
   
    
    pred = ''
        
    if solution != -1:
        Solved.append(i)
        pred1 = flattener(solution)
        pred = pred+pred1+' '
        
    if pred == '':
        pred = flattener(example_grid)
        
    Proposed_Answers.append(pred)
    
submission['output'] = Proposed_Answers
submission.to_csv('submission.csv', index = False)


# In[ ]:


print(Solved)


# In[ ]:


submission2 = pd.read_csv('submission.csv')
submission2.tail(25)


# In[ ]:


submission2.head()

