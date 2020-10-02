#!/usr/bin/env python
# coding: utf-8

# ## This is ensemble of what I found public and decided to use. 
# Please upvote kernels used in the final ensemble in order to give the bright authors the credits their deserve:  
# - (1) [Decision Tree + Smart data augmentation](https://www.kaggle.com/adityaork/decision-tree-smart-data-augmentation)
# - (2) [Color And Counting Modulo Q](https://www.kaggle.com/szabo7zoltan/colorandcountingmoduloq)
# - (3) [Using Decision Trees for ARC](https://www.kaggle.com/meaninglesslives/using-decision-trees-for-arc) -> look 4
# - (4) [Grid Search with XGBoost and CV](https://www.kaggle.com/nxrprime/grid-search-with-xgboost-and-cv)   -> look 3
# - (5) [ARC C++ approach](https://www.kaggle.com/zaharch/arc-c-approach)
# 
# GNN is not included (but deserves upvote):    
# - (6') [ARC and Graph Neural Networks](https://www.kaggle.com/mandubian/arc-and-graph-neural-networks)

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pathlib import Path
from collections import defaultdict
from itertools import product
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations,permutations
from sklearn.tree import *
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
import random
from math import floor

data_path = Path("/kaggle/input/abstraction-and-reasoning-challenge")
train_path = data_path/'training'
test_path = data_path/'test'

def plot_result(inp,eoup,oup):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    fig, axs = plt.subplots(1, 3, figsize=(15,15))
    
    axs[0].imshow(inp, cmap=cmap, norm=norm)
    axs[0].axis('off')
    axs[0].set_title('Input')

    axs[1].imshow(eoup, cmap=cmap, norm=norm)
    axs[1].axis('off')
    axs[1].set_title('Output')
    
    axs[2].imshow(oup, cmap=cmap, norm=norm)
    axs[2].axis('off')
    axs[2].set_title('Model prediction')
    
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_mats(mats):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    fig, axs = plt.subplots(1, len(mats), figsize=(15,15))
    
    for i in range(len(mats)):
        axs[i].imshow(mats[i], cmap=cmap, norm=norm)
        axs[i].axis('off')
        axs[i].set_title('Fig: '+str(i))
    
    plt.rc('grid', linestyle="-", color='white')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def getiorc(pair):
    inp = pair["input"]
    return pair["input"],pair["output"],len(inp),len(inp[0])
    
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

def getDiagonal(i,j,r,c):
    return
        
    
def getX(inp,i,j,size):
    z = []
    n_inp = np.array(inp)
    z.append(i)
    z.append(j)
    r,c = len(inp),len(inp[0])
    for m in range(5):
        z.append(i%(m+1))
        z.append(j%(m+1))
    z.append(i+j)
    z.append(i*j)
#     z.append(i%j)
#     z.append(j%i)
    z.append((i+1)/(j+1))
    z.append((j+1)/(i+1))
    z.append(r)
    z.append(c)
    z.append(len(np.unique(n_inp[i,:])))
    z.append(len(np.unique(n_inp[:,j])))
    arnd = getAround(i,j,inp,size)
    z.append(len(np.unique(arnd)))
    z.extend(arnd)
    return z

def getXy(inp,oup,size):
    x = []
    y = []
    r,c = len(inp),len(inp[0])
    for i in range(r):
        for j in range(c):
            x.append(getX(inp,i,j,size))
            y.append(oup[i][j])
    return x,y
    
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
    
def gettaskxy(task_json,aug,around_size,bl_cols,flip=True):    
    X = []
    Y = []
    for pair in task_json['train']:
        inp,oup=pair["input"],pair["output"]
        tx,ty = getXy(inp,oup,around_size)
        X.extend(tx)
        Y.extend(ty)
        if(flip):
            for ainp,aoup in get_flips(inp,oup):
                tx,ty = getXy(ainp,aoup,around_size)
                X.extend(tx)
                Y.extend(ty)
                if(aug):
                    augs = augment(ainp,aoup,bl_cols)
                    for ainp,aoup in augs:
                        tx,ty = getXy(ainp,aoup,around_size)
                        X.extend(tx)
                        Y.extend(ty)
        if(aug):
            augs = augment(inp,oup,bl_cols)
            for ainp,aoup in augs:
                tx,ty = getXy(ainp,aoup,around_size)
                X.extend(tx)
                Y.extend(ty)
    return X,Y

def test_predict(task_json,model,size):
    inp = task_json['test'][0]['input']
    eoup = task_json['test'][0]['output']
    r,c = len(inp),len(inp[0])
    oup = predict(inp,model,size)
    return inp,eoup,oup

def predict(inp,model,size):
    r,c = len(inp),len(inp[0])
    oup = np.zeros([r,c],dtype=int)
    for i in range(r):
        for j in range(c):
            x = getX(inp,i,j,size)
            o = int(model.predict([x]))
            o = 0 if o<0 else o
            oup[i][j]=o
    return oup

def submit_predict(task_json,model,size):
    pred_map = {}
    idx=0
    for pair in task_json['test']:
        inp = pair["input"]
        oup = predict(inp,model,size)
        pred_map[idx] = oup.tolist()
        idx+=1
        plot_result(inp,oup,oup)
    return pred_map

def dumb_predict(task_json):
    pred_map = {}
    idx=0
    for pair in task_json['test']:
        inp = pair["input"]
        pred_map[idx] = [[0,0],[0,0]]
        idx+=1
    return pred_map

def get_loss(model,task_json,size):
    total = 0
    for pair in task_json['train']:
        inp,oup=pair["input"],pair["output"]
        eoup = predict(inp,model,size)
        total+= np.sum((np.array(oup) != np.array(eoup)))
    return total

def get_test_loss(model,task_json,size):
    total = 0
    for pair in task_json['test']:
        inp,oup=pair["input"],pair["output"]
        eoup = predict(inp,model,size)
        total+= np.sum((np.array(oup) != np.array(eoup)))
    return total

def get_a_size(task_json):
    return 4;

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

def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

def combine_preds(tid,pm1,pm3,pm5):
    result = []
    for i in range(len(pm1)):
        tk_s = tid+"_"+str(i)
        str_pred = flattener(pm1[i])+" "+flattener(pm3[i])+" "+flattener(pm5[i])
        #print(tk_s,str_pred)
        result.append([tk_s,str_pred])
    return result

def inp_oup_dim_same(task_json):
    return all([ len(pair["input"]) == len(pair["output"]) and len(pair["input"][0]) == len(pair["output"][0])
                for pair in task_json['train']])
    

solved_task = 0
total_task = 0
task_ids = []
task_preds = []
for task_path in test_path.glob("*.json"):
    task_json = json.load(open(task_path))
    tk_id = str(task_path).split("/")[-1].split(".")[0]
    print(tk_id)
    if(inp_oup_dim_same(task_json)):
        a_size = get_a_size(task_json)
        bl_cols = get_bl_cols(task_json)
        
        isflip = False
        X1,Y1 = gettaskxy(task_json,True,1,bl_cols,isflip)
        X3,Y3 = gettaskxy(task_json,True,3,bl_cols,isflip)
        X5,Y5 = gettaskxy(task_json,True,5,bl_cols,isflip)
        
        model_1 = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100).fit(X1, Y1)
        model_3 = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100).fit(X3, Y3)
        model_5 = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100).fit(X5, Y5)
        
        pred_map_1 = submit_predict(task_json,model_1,1)
        pred_map_3 = submit_predict(task_json,model_3,3)
        pred_map_5 = submit_predict(task_json,model_5,5)
        
        for tks,str_pred in combine_preds(tk_id,pred_map_1,pred_map_3,pred_map_5):
            task_ids.append(tks)
            task_preds.append(str_pred)
            #print(tks,str_pred)
        solved_task+=1
        #break
    else:
        pred_map_1 = dumb_predict(task_json)
        pred_map_3 = dumb_predict(task_json)
        pred_map_5 = dumb_predict(task_json)
        
        for tks,str_pred in combine_preds(tk_id,pred_map_1,pred_map_3,pred_map_5):
            task_ids.append(tks)
            task_preds.append(str_pred)
            #print(tks,str_pred)
        
    total_task+=1
    
sample_sub1 = pd.DataFrame({"output_id":task_ids,'output':task_preds})
sample_sub1.to_csv("submission_1.csv", index=None)


# In[ ]:


import os
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'
training_tasks = sorted(os.listdir(training_path))
eval_tasks = sorted(os.listdir(evaluation_path))


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
    

def plot_picture(x):
    plt.imshow(np.array(x), cmap = cmap, norm = norm)
    plt.show()
    
    
def Defensive_Copy(A): 
    n = len(A)
    k = len(A[0])
    L = np.zeros((n,k), dtype = int)
    for i in range(n):
        for j in range(k):
            L[i,j] = 0 + A[i][j]
    return L.tolist()


def Create(task, task_id = 0):
    n = len(task['train'])
    Input = [Defensive_Copy(task['train'][i]['input']) for i in range(n)]
    Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]
    Input.append(Defensive_Copy(task['test'][task_id]['input']))
    return Input, Output


def Recolor(task):
    Input = task[0]
    Output = task[1]
    Test_Picture = Input[-1]
    Input = Input[:-1]
    N = len(Input)
    
    for x, y in zip(Input, Output):
        if len(x) != len(y) or len(x[0]) != len(y[0]):
            return -1
        
    Best_Dict = -1
    Best_Q1 = -1
    Best_Q2 = -1
    Best_v = -1
    # v ranges from 0 to 3. This gives an extra flexibility of measuring distance from any of the 4 corners
    Pairs = []
    for t in range(15):
        for Q1 in range(1,8):
            for Q2 in range(1,8):
                if Q1+Q2 == t:
                    Pairs.append((Q1,Q2))
                    
    for Q1, Q2 in Pairs:
        for v in range(4):
    
  
            if Best_Dict != -1:
                continue
            possible = True
            Dict = {}
                      
            for x, y in zip(Input, Output):
                n = len(x)
                k = len(x[0])
                for i in range(n):
                    for j in range(k):
                        if v == 0 or v ==2:
                            p1 = i%Q1
                        else:
                            p1 = (n-1-i)%Q1
                        if v == 0 or v ==3:
                            p2 = j%Q2
                        else :
                            p2 = (k-1-j)%Q2
                        color1 = x[i][j]
                        color2 = y[i][j]
                        if color1 != color2:
                            rule = (p1, p2, color1)
                            if rule not in Dict:
                                Dict[rule] = color2
                            elif Dict[rule] != color2:
                                possible = False
            if possible:
                
                # Let's see if we actually solve the problem
                for x, y in zip(Input, Output):
                    n = len(x)
                    k = len(x[0])
                    for i in range(n):
                        for j in range(k):
                            if v == 0 or v ==2:
                                p1 = i%Q1
                            else:
                                p1 = (n-1-i)%Q1
                            if v == 0 or v ==3:
                                p2 = j%Q2
                            else :
                                p2 = (k-1-j)%Q2
                           
                            color1 = x[i][j]
                            rule = (p1,p2,color1)
                            
                            if rule in Dict:
                                color2 = 0 + Dict[rule]
                            else:
                                color2 = 0 + y[i][j]
                            if color2 != y[i][j]:
                                possible = False 
                if possible:
                    Best_Dict = Dict
                    Best_Q1 = Q1
                    Best_Q2 = Q2
                    Best_v = v
                
                
    if Best_Dict == -1:
        return -1 #meaning that we didn't find a rule that works for the traning cases
    
    #Otherwise there is a rule: so let's use it:
    n = len(Test_Picture)
    k = len(Test_Picture[0])
    
    answer = np.zeros((n,k), dtype = int)
   
    for i in range(n):
        for j in range(k):
            if Best_v == 0 or Best_v ==2:
                p1 = i%Best_Q1
            else:
                p1 = (n-1-i)%Best_Q1
            if Best_v == 0 or Best_v ==3:
                p2 = j%Best_Q2
            else :
                p2 = (k-1-j)%Best_Q2
           
            color1 = Test_Picture[i][j]
            rule = (p1, p2, color1)
            if (p1, p2, color1) in Best_Dict:
                answer[i][j] = 0 + Best_Dict[rule]
            else:
                answer[i][j] = 0 + color1
                                    
           
            
    return answer.tolist()


Function = Recolor

training_examples = []
for i in range(400):
    task = Trains[i]
    basic_task = Create(task,0)
    a = Function(basic_task)
  
    if  a != -1 and task['test'][0]['output'] == a:
        plot_picture(a)
        plot_task(task)
        print(i)
        training_examples.append(i)
        
        
evaluation_examples = []


for i in range(400):
    task = Evals[i]
    basic_task = Create(task,0)
    a = Function(basic_task)
    
    if a != -1 and task['test'][0]['output'] == a:
       
        plot_picture(a)
        plot_task(task)
        print(i)
        evaluation_examples.append(i)
        
        
sample_sub2 = pd.read_csv(data_path/ 'sample_submission.csv')
sample_sub2.head()


def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


example_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
display(example_grid)
print(flattener(example_grid))

Solved = []
Problems = sample_sub2['output_id'].values
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
    
sample_sub2['output'] = Proposed_Answers
sample_sub2.to_csv('submission2.csv', index = False)


# In[ ]:


from xgboost import XGBClassifier
import pdb

### I added this from 
### https://www.kaggle.com/backaggle/ensemble-from-public-kernels

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
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


sample_sub3 = pd.read_csv(data_path/'sample_submission.csv')
sample_sub3 = sample_sub1.set_index('output_id')
sample_sub3.head()

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

def make_features(input_color, nfeat):
    nrows, ncols = input_color.shape
    feat = np.zeros((nrows*ncols,nfeat))
    cur_idx = 0
    for i in range(nrows):
        for j in range(ncols):
            feat[cur_idx,0] = i
            feat[cur_idx,1] = j
            feat[cur_idx,2] = input_color[i][j]
            feat[cur_idx,3:7] = get_moore_neighbours(input_color, i, j, nrows, ncols)
            feat[cur_idx,7:9] = get_tl_tr(input_color, i, j, nrows, ncols)
            feat[cur_idx,9] = len(np.unique(input_color[i,:]))
            feat[cur_idx,10] = len(np.unique(input_color[:,j]))
            feat[cur_idx,11] = (i+j)
            feat[cur_idx,12] = len(np.unique(input_color[i-local_neighb:i+local_neighb,
                                                         j-local_neighb:j+local_neighb]))

            cur_idx += 1
        
    return feat

def features(task, mode='train'):
    num_train_pairs = len(task[mode])
    feat, target = [], []
    
    global local_neighb
    for task_num in range(num_train_pairs):
        input_color = np.array(task[mode][task_num]['input'])
        #print(input_color)
        target_color = task[mode][task_num]['output']
        #print(target_color)
        nrows, ncols = len(task[mode][task_num]['input']), len(task[mode][task_num]['input'][0])

        target_rows, target_cols = len(task[mode][task_num]['output']), len(task[mode][task_num]['output'][0])
        
        if (target_rows!=nrows) or (target_cols!=ncols):
            print('Number of input rows:',nrows,'cols:',ncols)
            print('Number of target rows:',target_rows,'cols:',target_cols)
            not_valid=1
            return None, None, 1

        imsize = nrows*ncols
        #offset = imsize*task_num*3 #since we are using three types of aug
        feat.extend(make_features(input_color, nfeat))
        target.extend(np.array(target_color).reshape(-1,))
            
    return np.array(feat), np.array(target), 0

# mode = 'eval'
mode = 'test'
if mode=='eval':
    task_path = evaluation_path
elif mode=='train':
    task_path = training_path
elif mode=='test':
    task_path = test_path

all_task_ids = sorted(os.listdir(task_path))

nfeat = 13
local_neighb = 5
valid_scores = {}

model_accuracies = {'ens': []}
pred_taskids = []

for task_id in all_task_ids:

    task_file = str(task_path / task_id)
    with open(task_file, 'r') as f:
        task = json.load(f)

    feat, target, not_valid = features(task)
    if not_valid:
        print('ignoring task', task_file)
        print()
        not_valid = 0
        continue

    xgb =  XGBClassifier(n_estimators=10, n_jobs=-1)
    xgb.fit(feat, target, verbose=-1)


#     training on input pairs is done.
#     test predictions begins here

    num_test_pairs = len(task['test'])
    for task_num in range(num_test_pairs):
        cur_idx = 0
        input_color = np.array(task['test'][task_num]['input'])
        nrows, ncols = len(task['test'][task_num]['input']), len(
            task['test'][task_num]['input'][0])
        feat = make_features(input_color, nfeat)

        print('Made predictions for ', task_id[:-5])

        preds = xgb.predict(feat).reshape(nrows,ncols)
        
        if (mode=='train') or (mode=='eval'):
            ens_acc = (np.array(task['test'][task_num]['output'])==preds).sum()/(nrows*ncols)

            model_accuracies['ens'].append(ens_acc)

            pred_taskids.append(f'{task_id[:-5]}_{task_num}')

#             print('ensemble accuracy',(np.array(task['test'][task_num]['output'])==preds).sum()/(nrows*ncols))
#             print()

        preds = preds.astype(int).tolist()
#         plot_test(preds, task_id)
        sample_sub3.loc[f'{task_id[:-5]}_{task_num}',
                       'output'] = flattener(preds)
        


if (mode=='train') or (mode=='eval'):
    df = pd.DataFrame(model_accuracies, index=pred_taskids)
    print(df.head(10))

    print(df.describe())
    for c in df.columns:
        print(f'for {c} no. of complete tasks is', (df.loc[:, c]==1).sum())

    df.to_csv('ens_acc.csv')



sample_sub3.head()

sample_sub3.to_csv('submission3.csv')


# In[ ]:


#import os
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from matplotlib import colors
#import json
#import pdb

from pathlib import Path

#from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

## plotting functions
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
    

# For flattening 2D numpy arrays
# https://www.kaggle.com/inversion/abstraction-and-reasoning-starter-notebook
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

sample_sub4 = pd.read_csv(data_path/'sample_submission.csv')
sample_sub4 = sample_sub4.set_index('output_id')
sample_sub4.head()


## Extract neibourhood features
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

## Make features for each train sample
def features(task, mode='train'):
    cur_idx = 0
    num_train_pairs = len(task[mode])
    total_inputs = sum([len(task[mode][i]['input'])*len(task[mode][i]['input'][0]) for i in range(num_train_pairs)])
    feat = np.zeros((total_inputs,nfeat))
    target = np.zeros((total_inputs,), dtype=np.int)
    
    global local_neighb
    for task_num in range(num_train_pairs):
        input_color = np.array(task[mode][task_num]['input'])
        target_color = task[mode][task_num]['output']
        nrows, ncols = len(task[mode][task_num]['input']), len(task[mode][task_num]['input'][0])

        target_rows, target_cols = len(task[mode][task_num]['output']), len(task[mode][task_num]['output'][0])
        
        if (target_rows!=nrows) or (target_cols!=ncols):
            print('Number of input rows:',nrows,'cols:',ncols)
            print('Number of target rows:',target_rows,'cols:',target_cols)
            not_valid=1
            return None, None, 1

        for i in range(nrows):
            for j in range(ncols):
                feat[cur_idx,0] = i
                feat[cur_idx,1] = j
                feat[cur_idx,2] = input_color[i][j]
                feat[cur_idx,3:7] = get_moore_neighbours(input_color, i, j, nrows, ncols)
                feat[cur_idx,7:9] = get_tl_tr(input_color, i, j, nrows, ncols)
                feat[cur_idx,9] = len(np.unique(input_color[i,:]))
                feat[cur_idx,10] = len(np.unique(input_color[:,j]))
                feat[cur_idx,11] = (i+j)
                feat[cur_idx,12] = len(np.unique(input_color[i-local_neighb:i+local_neighb,
                                                             j-local_neighb:j+local_neighb]))
        
                target[cur_idx] = target_color[i][j]
                cur_idx += 1
            
    return feat, target, 0

## Train and prediction
param_grid = {
    "xgb__n_estimators": [10],
    "xgb__learning_rate": [0.1],
    "xgb__early_stopping_rounds": np.array((50, 1000))
}

all_task_ids = sorted(os.listdir(test_path))

nfeat = 13
local_neighb = 5
valid_scores = {}
for task_id in all_task_ids:

    task_file = str(test_path / task_id)
    with open(task_file, 'r') as f:
        task = json.load(f)

    feat, target, not_valid = features(task)
    if not_valid:
        print('ignoring task', task_file)
        print()
        not_valid = 0
        continue

    nrows, ncols = len(task['train'][-1]['input']
                       ), len(task['train'][-1]['input'][0])
    # use the last train sample for validation
    val_idx = len(feat) - nrows*ncols

    train_feat = feat[:val_idx]
    val_feat = feat[val_idx:, :]

    train_target = target[:val_idx]
    val_target = target[val_idx:]

    #     check if validation set has a new color
    #     if so make the mapping color independant
    if len(set(val_target) - set(train_target)):
        print('set(val_target)', set(val_target))
        print('set(train_target)', set(train_target))
        print('Number of colors are not same')
        print('cant handle new colors. skipping')
        continue

    xgb = XGBClassifier(n_estimators=100, n_jobs=-1)
   # hgb_pipe = make_pipeline([('xgb', xgb)])


    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 1001)
    hgb_grid = GridSearchCV(xgb, param_grid, n_jobs=8, 
         cv=skf, verbose=2, refit=True)
    hgb_grid.fit(feat, target)
#     training on input pairs is done.
#     test predictions begins here

    num_test_pairs = len(task['test'])
    for task_num in range(num_test_pairs):
        cur_idx = 0
        input_color = np.array(task['test'][task_num]['input'])
        nrows, ncols = len(task['test'][task_num]['input']), len(
            task['test'][task_num]['input'][0])
        feat = np.zeros((nrows*ncols, nfeat))
        unique_col = {col: i for i, col in enumerate(sorted(np.unique(input_color)))}

        for i in range(nrows):
            for j in range(ncols):
                feat[cur_idx, 0] = i
                feat[cur_idx, 1] = j
                feat[cur_idx, 2] = input_color[i][j]
                feat[cur_idx, 3:7] = get_moore_neighbours(
                    input_color, i, j, nrows, ncols)
                feat[cur_idx, 7:9] = get_tl_tr(
                    input_color, i, j, nrows, ncols)
                feat[cur_idx, 9] = len(np.unique(input_color[i, :]))
                feat[cur_idx, 10] = len(np.unique(input_color[:, j]))
                feat[cur_idx, 11] = (i+j)
                feat[cur_idx, 12] = len(np.unique(input_color[i-local_neighb:i+local_neighb,
                                                              j-local_neighb:j+local_neighb]))

                cur_idx += 1

        print('Made predictions for ', task_id[:-5])
        preds = hgb_grid.predict(feat).reshape(nrows, ncols)
        preds = preds.astype(int).tolist()
        plot_test(preds, task_id)
        sample_sub4.loc[f'{task_id[:-5]}_{task_num}',
                       'output'] = flattener(preds)
        
        
print('\n Best hyperparameters:')
print(hgb_grid.best_params_)

sample_sub4.head()

sample_sub4.to_csv('submission4.csv')


# # C++
# 

# In[ ]:


import os
import time
from pathlib import Path

KAGGLE = Path('/kaggle/working').is_dir()


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'mycplusplus.cpp', '\n#include <iostream>\n#include <fstream>\n#include <string>\n#include <vector>\n#include <queue>\n#include <utility>\n#include <stdio.h>\n#include <sys/types.h>\n#include <sys/stat.h>\n#include <algorithm>\n#include <numeric>\n#include <map>\n#include <dirent.h>\n\nusing namespace std;\n\n\ntemplate <typename T>\nvector<size_t> sort_indexes(const vector<T> &v) {\n\n  vector<size_t> idx(v.size());\n  iota(idx.begin(), idx.end(), 0);\n\n  stable_sort(idx.begin(), idx.end(),\n       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});\n\n  return idx;\n}\n\ntemplate <typename A, typename B>\nvoid zip(\n    const std::vector<A> &a, \n    const std::vector<B> &b, \n    std::vector<std::pair<A,B>> &zipped)\n{\n    for(size_t i=0; i<a.size(); ++i)\n    {\n        zipped.push_back(std::make_pair(a[i], b[i]));\n    }\n}\n\ntemplate <typename A, typename B>\nvoid unzip(\n    const std::vector<std::pair<A, B>> &zipped, \n    std::vector<A> &a, \n    std::vector<B> &b)\n{\n    for(size_t i=0; i<a.size(); i++)\n    {\n        a[i] = zipped[i].first;\n        b[i] = zipped[i].second;\n    }\n}\n\nbool IsPathExist(const std::string &s)\n{\n  struct stat buffer;\n  return (stat (s.c_str(), &buffer) == 0);\n}\n\nclass Grid {\n    public:\n        int nrows = 0;\n        int ncols = 0;\n        int **mat = NULL;\n        \n        Grid() {}\n    \n        Grid(const Grid& grid) {\n            gridCopy(grid);\n        }\n    \n        void gridCopy(const Grid& grid) {\n            nrows = grid.nrows;\n            ncols = grid.ncols;\n            \n            mat = new int*[nrows];\n            for (int i = 0; i < nrows; ++i) {\n                mat[i] = new int[ncols];\n                for (int j = 0; j < ncols; ++j)\n                    mat[i][j] = grid.mat[i][j];\n            }\n        }\n        \n        Grid(int nr, int nc) {\n            gridBySize(nr,nc);\n        }\n    \n        void gridBySize(int nr, int nc) {\n            nrows = nr;\n            ncols = nc;\n            \n            mat = new int*[nrows];\n            for (int i = 0; i < nrows; ++i) {\n                mat[i] = new int[ncols];\n                for (int j = 0; j < ncols; ++j)\n                    mat[i][j] = -1;\n            }\n        }\n        \n        Grid(std::vector<std::vector<int>> &rows) {\n            nrows = rows.size();\n            ncols = rows[0].size();\n            \n            mat = new int*[nrows];\n            for (int i = 0; i < nrows; ++i) {\n                mat[i] = new int[ncols];\n                for (int j = 0; j < ncols; ++j)\n                    mat[i][j] = rows[i][j];\n            }\n        }\n        \n        void print() {\n            std::cout << "\\n";\n            for (int i = 0; i < nrows; ++i) {\n                for (int j = 0; j < ncols; ++j) {\n                    if (mat[i][j] == 0)\n                        std::cout << \'.\';\n                    else\n                        std::cout << mat[i][j];\n                }\n                std::cout << "\\n";\n            }\n        }\n        \n        ~Grid() {\n            clear();\n        }\n    \n        virtual void clear() {\n            if (mat) {\n                for (int i = 0; i < nrows; ++i)\n                    delete mat[i];\n                delete mat;\n            }\n            mat = NULL;\n        }\n        \n        bool operator== (const Grid &ref) const\n        {\n            if ((nrows != ref.nrows) | (ncols != ref.ncols))\n                return false;\n            for (int i = 0; i < nrows; ++i)\n                for (int j = 0; j < ncols; ++j)\n                    if (mat[i][j] != ref.mat[i][j])\n                        return false;\n            return true;\n        }\n        \n        std::string flatten() {\n            std::string out;\n            for (int i = 0; i < nrows; ++i) {\n                out.append("|");\n                for (int j = 0; j < ncols; ++j)\n                    out.append(std::to_string(mat[i][j]));\n            }\n            out.append("|");\n            return out;\n        }\n};\n\nclass Pair {\n    public:\n        Grid *input = NULL;\n        Grid *output = NULL;\n        \n        Pair(Grid *inp, Grid *out) {\n            input = inp;\n            output = out;\n        }\n    \n        Pair() {}\n        \n        void print() {\n            input->print();\n            output->print();\n        }\n    \n        void set_input(Grid *inp) {\n            input = inp;\n        }\n    \n        void set_output(Grid *out) {\n            output = out;\n        }\n        \n        ~Pair() {\n            if (input)\n                delete input;\n            if (output)\n                delete output;\n        }\n};\n\nclass Task {\n    public:\n        std::vector<Pair*> train;\n        std::vector<Pair*> test;\n        \n        void add_train(Pair* pair) {\n            train.push_back(pair);\n        }\n    \n        void add_test(Pair* pair) {\n            test.push_back(pair);\n        }\n    \n        ~Task() {\n            for (Pair* pair : train)\n                delete pair;\n            for (Pair* pair : test)\n                delete pair;\n        }\n};\n\nclass Property : public Grid {\n    public:\n        Property() {}\n        Property(const Grid& grid) : Grid(grid) {}\n        std::string name;\n        virtual void populate(Grid *grid) {}\n        virtual void pre(vector<Property*> props) {}\n        virtual void post(vector<Property*> props) {}\n\n        void calcBackground(Grid* grid) {\n\n            int cnts[10] = {};\n            for (int i = 0; i < grid->nrows; ++i)\n                for (int j = 0; j < grid->ncols; ++j)\n                    cnts[grid->mat[i][j]]++;\n\n            int max_cnt = -1;\n            for (int c = 0; c < 10; ++c)\n                if (cnts[c] > max_cnt) {\n                    max_cnt = cnts[c];\n                    bg[0] = c;\n                }\n        }\n\n        void copyBG(int* bg_in) {\n            for (int i=0; i<10; i++)\n                bg[i] = bg_in[i];\n        }\n\n        bool operator== (const Property &ref) const {\n            return Grid::operator==((const Grid&)ref);\n        }\n\n        int bg[10] {};\n\n        vector<pair<int,int>> dirs = {make_pair(-1,-1),make_pair(-1,0),make_pair(-1,1),make_pair(0,-1),\n                                      make_pair(0,1),make_pair(1,-1),make_pair(1,0),make_pair(1,1)};\n        vector<pair<int,int>> dirs_rook = {make_pair(-1,0),make_pair(0,1),make_pair(1,0),make_pair(0,-1)};\n        vector<pair<int,int>> dirs_row = {make_pair(0,1),make_pair(0,-1)};\n        vector<pair<int,int>> dirs_col = {make_pair(1,0),make_pair(-1,0)};\n        vector<pair<int,int>> dirs_diag1 = {make_pair(1,1),make_pair(-1,-1)};\n        vector<pair<int,int>> dirs_diag2 = {make_pair(1,-1),make_pair(-1,1)};\n};\n\nclass PropBorderType : public Property {\n    public:\n    \n        PropBorderType() {\n            name = "PropBorderType";\n        }\n        \n        virtual void populate(Grid *grid) {\n\n            gridBySize(grid->nrows, grid->ncols);\n\n            markCell(grid,0,0,1);\n            markCell(grid,0,ncols-1,2);\n            markCell(grid,nrows-1,0,3);\n            markCell(grid,nrows-1,ncols-1,4);\n            \n            for (int i = 1; i < (nrows-1); ++i) {\n                markCell(grid,i,0,5);\n                markCell(grid,i,ncols-1,6);\n            }\n            for (int j = 1; j < (ncols-1); ++j) {\n                markCell(grid,0,j,7);\n                markCell(grid,nrows-1,j,8);\n            }\n\n            for (int i = 0; i < nrows; ++i)\n                for (int j = 0; j < ncols; ++j)\n                    if (mat[i][j] < 0) mat[i][j] = 0;\n        }\n    \n    private:\n        \n        void markCell(Grid *grid, int i, int j, int col) {\n            if (mat[i][j] >= 0) \n                return;\n            mat[i][j] = col;\n            int c = grid->mat[i][j];\n            for (pair<int,int> p : dirs_rook) {\n                int x = i + p.first;\n                int y = j + p.second;\n                if ((x < 0) || (x >= nrows) || (y < 0) || (y >= ncols))\n                    continue;\n                if (grid->mat[x][y] != c)\n                    continue;\n                markCell(grid, x, y, col);\n            }\n        }\n};\n\n\nclass PropBorderDistance : public Property {\n    public:\n    \n        PropBorderDistance(int max_dist = 100, bool onBG = true) : max_dist{max_dist}, onBG{onBG} {\n            name = "PropBorderDistance";\n\n            name += to_string(max_dist);\n            if (onBG)\n                name += "G";\n        }\n        \n        virtual void populate(Grid *grid) {\n\n            if (bg[0] == -1)\n                calcBackground(grid);\n\n            nrows = grid->nrows;\n            ncols = grid->ncols;\n            \n            mat = new int*[nrows];\n            for (int i = 0; i < nrows; ++i) {\n                mat[i] = new int[ncols];\n                for (int j = 0; j < ncols; ++j)\n                    mat[i][j] = 0;\n            }\n            \n            for (int i = 0; i < nrows; ++i) {\n                markCell(grid,i,0,1);\n                markCell(grid,i,ncols-1,1);\n            }\n            for (int j = 1; j < (ncols-1); ++j) {\n                markCell(grid,0,j,1);\n                markCell(grid,nrows-1,j,1);\n            }\n            \n            while(search.size() > 0) {\n                std::pair<int,int> p = search.front();\n                search.pop();\n                int depth = mat[p.first][p.second] + 1;\n                if (depth > max_dist) depth = max_dist;\n                markCell(grid,p.first-1,p.second,depth);\n                markCell(grid,p.first+1,p.second,depth);\n                markCell(grid,p.first,p.second-1,depth);\n                markCell(grid,p.first,p.second+1,depth);\n            }\n        }\n    \n    private:\n    \n        std::queue<std::pair<int,int>> search;\n        \n        void markCell(Grid *grid, int i, int j, int depth) {\n            if ((i < 0) | (i >= nrows) | (j < 0) | (j >= ncols))\n                return;\n            if ((!onBG || (grid->mat[i][j] == bg[0])) && (mat[i][j] == 0)) {\n                mat[i][j] = depth;\n                std::pair<int,int> p(i,j);\n                search.push(p);\n            }\n        }\n\n        int max_dist = 0;\n        bool onBG = true;\n};\n\nclass PropHoles : public PropBorderDistance {\n    public:\n        \n        PropHoles() {\n            name = "PropHoles";\n        }\n    \n        virtual void populate(Grid *grid) {\n            PropBorderDistance::populate(grid);\n            for (int i = 0; i < nrows; ++i)\n                for (int j = 0; j < ncols; ++j)\n                    if ((mat[i][j] == 0) && (grid->mat[i][j] == 0))\n                        mat[i][j] = 1;\n                    else\n                        mat[i][j] = 0;\n        }\n};\n\nclass PropColor : public Property {\n    public:\n    \n        PropColor () {\n            name = "PropColor";\n        }\n    \n        virtual void populate(Grid *grid) {\n            gridCopy(*grid);\n        }\n};\n    \nclass PropModulo : public Property {\n    public:\n    \n        PropModulo(int mod, bool is_row, bool is_col, bool is_rev) : mod{mod}, is_row{is_row}, is_col{is_col}, is_rev(is_rev) {\n            name = "PropModulo";\n            name += to_string(mod);\n            if (is_row)\n                name += "T";\n            else\n                name += "F";\n            if (is_col)\n                name += "T";\n            else\n                name += "F";\n            if (is_rev)\n                name += "R";\n        }\n    \n        virtual void populate(Grid *grid) {\n            gridCopy(*grid);\n            \n            for (int i = 0; i < nrows; ++i)\n                for (int j = 0; j < ncols; ++j) {\n                    mat[i][j] = 0;\n                    int x = i;\n                    int y = j;\n                    if (is_rev) {\n                        x = nrows-1-x;\n                        y = ncols-1-y;\n                    }\n                    if (is_row)\n                        mat[i][j] += mod*(x%mod);\n                    if (is_col)\n                        mat[i][j] += (y%mod);\n                }\n        }\n\n    private:\n        int mod = 2;\n        bool is_row = true;\n        bool is_col = true;\n        bool is_rev = false;\n};\n\n\n\nclass Action {\n    public:\n        std::string name;\n        \n        vector<vector<Grid*>> test_out;\n        vector<string> names_out;\n\n        virtual bool train(Task* task) {}\n        virtual Grid* generate(Grid* grid, int test_id) {}\n};\n\n\nclass Coloring {\n    public:\n        Coloring(std::vector<Property*> props, bool old_post=false) : props {props}, old_post{old_post} {\n            len = props.size();\n            constructor_init();\n        }\n        \n        Coloring(bool old_post=false, int len=1) : old_post{old_post}, len{len} {\n            constructor_init();\n        }\n\n        void constructor_init() {\n            if (len < 2) m_separate = false;\n            if (m_separate)\n                for (int c1=0; c1<10; ++c1) {\n                    for (int c2=0; c2<10; ++c2) {\n                        sep_row[c1][c2] = -1;\n                    }\n                }\n        }\n\n        void init_props(std::vector<Property*> props_in) {\n            props = props_in;\n            init_props_low();\n        }\n\n        void init_props(Grid* grid) {\n            for (Property* prop : props) {\n                prop->clear();\n                prop->populate(grid);\n            }\n            \n            init_props_low();\n        }\n\n        void init_props_low() {\n            for (int i=0; i<3; i++)\n                if (i < len)\n                    mx[i] = 10;\n                else\n                    mx[i] = 1;\n\n            if (train_id == 0) {\n                name = "";\n                for (Property* prop : props)\n                    name += prop->name + ";";\n            }\n        }\n        \n        int getVal(int i=0, int j=0, int z=0) {\n            return (cmap[i][j][z] - 1);\n        }\n    \n        int getVal(int* idx) {\n            return getVal(idx[0],idx[1],idx[2]);\n        }\n    \n        void setVal(int i=0, int j=0, int z=0, int val=0) {\n            cmap[i][j][z] = val + 1;\n        }\n        \n        void finalizeTrain(int id) {\n            if (m_separate && (train_id == 0)) {\n                for (int c3 = 0; c3 < mx[2]; ++c3) {\n                    for (int c2 = 0; c2 < mx[1]; ++c2) {\n                        int found_color = -1;\n                        for (int c1 = 0; c1 < mx[0]; ++c1) {\n                            if (getVal(c1,c2,c3) >= 0) {\n                                if ((found_color >= 0)) {\n                                    m_separate = false;\n                                }\n                                sep_row[c2][c3] = c1;\n                                found_color = c1;\n                            }\n                        }\n                    }\n                }\n            }\n        }\n        \n        int inferenceSeparate(int i, int j) {\n            int idx0 = props[0]->mat[i][j];\n            int idx1 = props[1]->mat[i][j];\n            int idx2 = 0;\n            if (len == 3)\n                idx2 = props[2]->mat[i][j];\n            \n            int cm = sep_map[idx0];\n            if (cm == -1) return -1;\n            cm = getVal(cm, idx1, idx2);\n            if (cm == -1) return -1;\n            cm = sep_map_rev[cm];\n            \n            return cm;\n        }\n    \n        bool trainSeparate(int i, int j, int c_out) {\n            \n            if (train_id == 0) {\n                bool ret = trainTogether(i,j,c_out);\n                return ret;\n            } else {\n                int cm = inferenceSeparate(i,j);\n                if (cm != c_out) return false;\n                return true;\n            }\n        }\n        \n        bool trainTogether(int i, int j, int c_out) {\n            int cm = get_color_base(i,j);\n            \n            if (cm >= 0) {\n                if (cm != c_out)\n                    return false;\n            } else\n                set_color(i,j,c_out);\n            \n            return true;\n        }\n        \n        bool train(int i, int j, int c_out) {\n            if (m_separate) {\n                bool ret_separate = trainSeparate(i,j,c_out);\n                m_separate = m_separate && ret_separate;\n            }\n            if (m_together) {\n                bool ret_together = trainTogether(i,j,c_out);\n                m_together = m_together && ret_together;\n            }\n            return (m_together || m_separate);\n        }\n        \n        void setTrainId(int id) {\n            train_id = id;\n            if (m_separate) {\n                for (int c=0; c<10; ++c) {\n                    sep_map[c] = -1;\n                    sep_map_rev[c] = -1;\n                }\n\n                if (train_id != 0) {\n                    for (int i=0; i<props[0]->nrows; i++)\n                        for (int j=0; j<props[0]->ncols; j++) {\n                            int idx0 = props[0]->mat[i][j];\n                            int idx1 = props[1]->mat[i][j];\n                            int idx2 = 0;\n                            if (len == 3)\n                                idx2 = props[2]->mat[i][j];\n                            sep_map[idx0] = sep_row[idx1][idx2];\n                        }\n                    \n                    for (int c=0; c<10; ++c) {\n                        if (sep_map[c] >= 0)\n                            sep_map_rev[sep_map[c]] = c;\n                    }\n                }\n            }\n\n        }\n        \n        void printCMap() {\n            if (len <= 2) {\n                for (int c1 = 0; c1 < 10; ++c1) {\n                    for (int c2 = 0; c2 < 10; ++c2) {\n                        int c = getVal(c1,c2);\n                        if (c >= 0)\n                            std::cout << getVal(c1,c2);\n                        else\n                            std::cout << ".";\n                    }\n                    std::cout << "\\n";\n                }\n            } else {\n                for (int c1 = 0; c1 < 10; ++c1) {\n                    std::cout << "first prop " << c1 << "\\n";\n                    for (int c2 = 0; c2 < 10; ++c2) {\n                        for (int c3 = 0; c3 < 10; ++c3) {\n                            int c = getVal(c1,c2,c3);\n                            if (c >= 0)\n                                std::cout << c;\n                            else\n                                std::cout << ".";\n                        }\n                        std::cout << "\\n";\n                    }\n                    std::cout << "\\n";\n                }\n            }\n\n        }\n\n        void post_process() {\n            int cm;\n\n            if (m_separate) {\n                name += "SEP;";\n            }\n            \n            int cols[3][10][2][10] = {};\n            int diag[3][10] = {};\n\n            for (int i = 0; i < 3; ++i) {\n                for (int c = 0; c < mx[i]; ++c) {\n                    for (int c1 = 0; c1 < mx[dims[i][0]]; ++c1) {\n                        for (int c2 = 0; c2 < mx[dims[i][1]]; ++c2) {\n                            int idx[3];\n                            idx[i] = c;\n                            idx[dims[i][0]] = c1;\n                            idx[dims[i][1]] = c2;\n                            cm = getVal(idx);\n                            if (cm >= 0) {\n                                cols[i][c][0][c1]++;\n                                cols[i][c][1][c2]++;\n                                if (c1 == c2)\n                                    diag[i][c]++;\n                            }\n                        }\n                    }\n                }\n            }\n            \n            while (true) {\n\n                int max_cnt = 0;\n                for (int i = 0; i < 3; ++i) {\n                    for (int c = 0; c < mx[i]; ++c) {\n                        for (int j = 0; j < 2; ++j) {\n                            for (int c1 = 0; c1 < 10; ++c1) {\n                                int val = cols[i][c][j][c1];\n                                if ((val > max_cnt) && (val < 10)) max_cnt = val;\n                            }\n                        }\n                        int val = diag[i][c];\n                        if ((val > max_cnt) && (val < 10)) max_cnt = val;\n                    }\n                }\n\n                int vals[10][3];\n                bool found = false;\n\n                for (int i = 0; !found && (i < 3); ++i) {\n                    for (int c = 0; !found && (c < mx[i]); ++c) {\n                        for (int j = 0; !found && (j < 2); ++j) {\n                            for (int c1 = 0; !found && (c1 < 10); ++c1) {\n                                int val = cols[i][c][j][c1];\n                                if (val == max_cnt) {\n                                    int idx[3];\n                                    idx[0] = i;\n                                    idx[1] = dims[i][j];\n                                    idx[2] = dims[i][1-j];\n                                    for (int c2 = 0; c2 < 10; ++c2) {\n                                        vals[c2][idx[0]] = c;\n                                        vals[c2][idx[1]] = c1;\n                                        vals[c2][idx[2]] = c2;\n                                    }\n                                    found = true;\n                                }\n                            }\n                        }\n                        int val = diag[i][c];\n                        if (val == max_cnt) {\n                            int idx[3];\n                            idx[0] = i;\n                            idx[1] = dims[i][0];\n                            idx[2] = dims[i][1];\n                            for (int c2 = 0; c2 < 10; ++c2) {\n                                vals[c2][idx[0]] = c;\n                                vals[c2][idx[1]] = c2;\n                                vals[c2][idx[2]] = c2;\n                            }\n                            found = true;\n                        }\n                    }\n                }\n\n                if (!found) break;\n\n                bool any_colored = false;\n\n                int selfColors[2] = {};\n                float cnts[10] = {};\n                for (int i = 0; i < 10; ++i) {\n                    cm = getVal(vals[i][0],vals[i][1],vals[i][2]);\n                    if (cm >= 0) {\n                        selfColors[int(cm == i)]++;\n                        if (vals[i][1] == vals[i][2])\n                            cnts[cm] += 1.0;\n                        else\n                            cnts[cm] += 1.05;\n                    }\n                }\n                int numColors = 0;\n                int freqColor = 0;\n                for (int i = 0; i < 10; ++i) {\n                    if (cnts[i] > cnts[freqColor])\n                        freqColor = i;\n                    numColors += int(cnts[i] > 0);\n                }\n                bool selfColoring = ((selfColors[0] == 0) && (selfColors[1] >= 2)) ||\n                                    ((selfColors[0] == 1) && (selfColors[1] >= 3));\n                for (int i = 0; i < 10; ++i) {\n                    if (getVal(vals[i][0],vals[i][1],vals[i][2]) < 0) {\n                        int col = -1;\n                        if (selfColoring)\n                            col = i;\n                        else if ((numColors <= 2) && (cnts[freqColor] >= 2.0))\n                            col = freqColor;\n                        \n                        if (col >= 0) {\n                            setVal(vals[i][0],vals[i][1],vals[i][2],col);\n                            any_colored = true;\n\n                            for (int j = 0; j < 3; ++j) {\n                                cols[j][vals[i][j]][0][vals[i][dims[j][0]]]++;\n                                cols[j][vals[i][j]][1][vals[i][dims[j][1]]]++;\n                                if (vals[i][dims[j][0]] == vals[i][dims[j][1]])\n                                    diag[j][vals[i][j]]++;\n                            }\n                        }\n                    }\n                }\n\n                if (!any_colored) break;\n            }\n        }\n        \n        void set_color(int i, int j, int val) {\n            int idx0 = props[0]->mat[i][j];\n            int idx1 = 0;\n            int idx2 = 0;\n            if (len >= 2)\n                idx1 = props[1]->mat[i][j];\n            if (len >= 3)\n                idx2 = props[2]->mat[i][j];\n            setVal(idx0,idx1,idx2,val);\n        }\n\n        int get_color(int i, int j) {\n            if (m_separate)\n                return inferenceSeparate(i,j);\n            else\n                return get_color_base(i,j);\n        }\n\n        int get_color_base(int i, int j) {\n            int idx0 = props[0]->mat[i][j];\n            int idx1 = 0;\n            int idx2 = 0;\n            if (len >= 2)\n                idx1 = props[1]->mat[i][j];\n            if (len >= 3)\n                idx2 = props[2]->mat[i][j];\n            return getVal(idx0,idx1,idx2);\n        }\n\n        string name;\n        bool m_separate = true;\n\n    private:\n        int cmap[10][10][10] = {};\n        int mx[3] = {};\n        int dims[3][2] = {{1,2}, {0,2}, {0,1}};\n        int len = 0;\n        int train_id = 0;\n        bool m_together = true;\n        int sep_map[10];\n        int sep_map_rev[10];\n        int sep_row[10][10];\n        bool old_post;\n        std::vector<Property*> props;    \n};\n\n\nTask* parse_file(std::string filepath) {\n    \n    std::ifstream ifs;\n    ifs.open(filepath.c_str(), std::ifstream::in);\n    \n    int cnt = 0;\n    bool char_t = false;\n    bool char_n = false;\n    bool is_train = false;\n    bool is_test = false;\n    bool is_input = false;\n    bool is_output = false;\n    int opened = 0;\n    int opened_curl = 0;\n    \n    std::vector<int> row;\n    std::vector<std::vector<int>> rows;\n    \n    std::vector<Pair> pairs;\n    Pair *pair = NULL;\n    Task *task = new Task();\n    \n    char c = ifs.get();\n    while (ifs.good()) {\n        \n        if ((c == \'e\') & char_t)\n            is_test = true;\n        if ((c == \'r\') & char_t)\n            is_train = true;\n        \n        if ((c == \'p\') & char_n)\n            is_input = true;\n        if ((c == \'p\') & char_t)\n            is_output = true;\n        \n        char_t = (c == \'t\');\n        char_n = (c == \'n\');\n        \n        if (c == \'{\') {\n            opened_curl++;\n            if (opened_curl == 2)\n                pair = new Pair();\n        }\n        if (c == \'}\') {\n            opened_curl--;\n        }\n        \n        if (c == \'[\') {\n            opened++;\n        }\n        if (c == \']\') {\n            opened--;\n            if (opened == 0) {\n                is_test = false;\n                is_train = false;\n            }\n            if (opened == 1) {\n                Grid *grid = new Grid(rows);\n                \n                if (is_input) {\n                    pair->set_input(grid);\n                \n                    if (is_train)\n                        task->add_train(pair);\n                    if (is_test)\n                        task->add_test(pair);\n                }\n                if (is_output) {\n                    pair->set_output(grid);\n                }\n                \n                rows.clear();\n                \n                is_input = false;\n                is_output = false;\n            }\n            if (opened == 2) {\n                rows.push_back(row);\n                row.clear();\n            }\n        }\n        \n        if ((c >= \'0\') & (c <= \'9\')) {\n            row.push_back(int(c) - int(\'0\'));\n        }\n        \n        c = ifs.get();\n        cnt++;\n    }\n\n    ifs.close();\n    \n    return task;\n}\n\nclass PropSet {\n    public:\n        PropSet(vector<Property*> &props) : props{props} {}\n\n        Property* operator[](int index) { \n            return props[index]; \n        }\n\n        void post() {\n\n            int cnts[10] {};\n            for (Property* p : props) {\n                for (int i=0; i<p->nrows; ++i)\n                    for (int j=0; j<p->ncols; ++j)\n                        cnts[p->mat[i][j]]++;\n            }\n\n            num_classes = 0;\n            for (int c=0; c<10; ++c)\n                if (cnts[c] > 0)\n                    num_classes++;\n        }\n\n        vector<Property*> props;\n        int num_classes = 0;\n};\n\nclass PropManager {\n    public:\n\n        PropManager(Task *task) : task{task} {\n            bg = calcBackground(task);\n        }\n\n        void calcBackgroundCount(Grid* grid, int* cnts) {\n            for (int i = 0; i < grid->nrows; ++i)\n                for (int j = 0; j < grid->ncols; ++j)\n                    cnts[grid->mat[i][j]]++;\n        }\n\n        int* calcBackground(Task* task) {\n            int cnts[10] = {};\n\n            for (Pair* pair : task->train)\n                calcBackgroundCount(pair->input, cnts);\n            \n            for (Pair* pair : task->test)\n                calcBackgroundCount(pair->input, cnts);\n\n            int* bg = new int[10] {};\n            \n            std::vector<int> cnts_vec;\n            cnts_vec.assign(cnts, cnts + 10);\n            int k=0;\n            for (auto i: sort_indexes(cnts_vec)) {\n                bg[k] = i;\n                k++;\n            }\n\n            return bg;\n        }\n\n        template<class PType, typename ... Args> void addProperty(Args ... args) {\n\n            vector<Property*> task_props;\n\n            int train_id = 0;\n            for (Pair* pair : task->train) {\n                Property* prop = (Property*) new PType(args...);\n                prop->copyBG(bg);\n                prop->populate(pair->input);\n                task_props.push_back(prop);\n                train_id++;\n            }\n\n            vector<Property*> task_props_test;\n\n            for (Pair* pair : task->test) {\n                Property* prop = (Property*) new PType(args...);\n                prop->copyBG(bg);\n                prop->populate(pair->input);\n                task_props_test.push_back(prop);\n            }\n\n            vector<Property*> task_props_all;\n            for (Property* p : task_props)\n                task_props_all.push_back(p);\n            for (Property* p : task_props_test)\n                task_props_all.push_back(p);\n\n            task_props_all[0]->post(task_props_all);\n\n            if (false) {\n                for (Property* prop : task_props) {\n                    cout << "DEBUG: " << prop->name << "\\n";\n                    prop->print();\n                }\n                for (Property* prop : task_props_test) {\n                    cout << "DEBUG TEST: " << prop->name << "\\n";\n                    prop->print();\n                }\n            }\n\n            PropSet prop_set(task_props);\n            prop_set.post();\n            if (prop_set.num_classes == 1) {\n                return;\n            }\n\n            bool different = true;\n            int k=0;\n            for (PropSet saved_props : props) {\n                different = false;\n                int i = 0;\n                for (Property* saved_prop : saved_props.props) {\n                    if (!((*saved_prop) == (*task_props[i]))) {\n                        different = true;\n                        break;\n                    }\n                    i++;\n                }\n                i = 0;\n                for (Property* saved_prop : props_test[k].props) {\n                    if (!((*saved_prop) == (*task_props_test[i]))) {\n                        different = true;\n                        break;\n                    }\n                    i++;\n                }\n                k++;\n                if (!different) {\n                    break;\n                }\n            }\n\n            if (!different) return;\n\n            props.push_back(prop_set);\n            names.push_back(task_props[0]->name);\n            props_test.push_back(PropSet(task_props_test));\n        }\n\n        void print() {\n            for (string name : names)\n                cout << name << "\\n";\n        }\n\n        vector<PropSet> props;\n        vector<PropSet> props_test;\n        vector<string> names;\n        Task* task;\n        int* bg = NULL;\n};\n\n\nclass MultiColoring {\n    public:\n        MultiColoring(PropManager *mngr) : mngr{mngr} {\n        }\n        \n        ~MultiColoring() {\n            for (vector<Coloring*> vcol : colorings)\n                for (Coloring* col : vcol)\n                    delete col;\n            \n            for (int i=0; i<3; i++)\n                delete failed[i];\n        }\n\n        void init(int level_in) {\n            \n            level = level_in;\n            vector<Coloring*> colorings_lvl;\n            vector<int> scores_lvl;\n\n            if (level == 0) {\n                for (PropSet vp : mngr->props) {\n                    Coloring* coloring = new Coloring(false, 1);\n                    colorings_lvl.push_back(coloring);\n                    scores_lvl.push_back(vp.num_classes);\n                }\n            } else if (level == 1) {\n                int i=0;\n                for (PropSet vp1 : mngr->props) {\n                    int k=0;\n                    for (PropSet vp2 : mngr->props) {\n                        if (k > i) {\n                            Coloring* coloring = new Coloring(false, 2);\n                            colorings_lvl.push_back(coloring);\n                            scores_lvl.push_back(vp1.num_classes + vp2.num_classes);\n                        }\n                        k++;\n                    }\n                    i++;\n                }\n            } else {\n                int p=0;\n                for (PropSet vp0 : mngr->props) {\n                    int i=0;\n                    for (PropSet vp1 : mngr->props) {\n                        int k=0;\n                        for (PropSet vp2 : mngr->props) {\n                            if ((k > i) && (i > p)) {\n                                Coloring* coloring = new Coloring(false, 3);\n                                colorings_lvl.push_back(coloring);\n                                scores_lvl.push_back(vp0.num_classes + vp1.num_classes + vp2.num_classes);\n                            }\n                            k++;\n                        }\n                        i++;\n                    }\n                    p++;\n                }\n            }\n\n            colorings.push_back(colorings_lvl);\n            scores.push_back(scores_lvl);\n\n            failed[level] = new bool[colorings[level].size()] {};\n        }\n\n        void setId(vector<PropSet> props_in, int id) {\n            if (level == 0) {\n                int i = 0;\n                for (PropSet vp : props_in) {\n                    std::vector<Property*> props;\n                    props.push_back(vp[id]);\n                    colorings[level][i]->init_props(props);\n                    i++;\n                }\n            } else if (level == 1) {\n                int num = 0;\n                int i = 0;\n                for (PropSet vp1 : props_in) {\n                    int k=0;\n                    for (PropSet vp2 : props_in) {\n                        if (k > i) {\n                            std::vector<Property*> props;\n                            props.push_back(vp1[id]);\n                            props.push_back(vp2[id]);\n                            colorings[level][num]->init_props(props);\n                            num++;\n                        }\n                        k++;\n                    }\n                    i++;\n                }\n            } else {\n                int num = 0;\n                int p=0;\n                for (PropSet vp0 : props_in) {\n                    int i = 0;\n                    for (PropSet vp1 : props_in) {\n                        int k=0;\n                        for (PropSet vp2 : props_in) {\n                            if ((k > i) && (i > p)) {\n                                std::vector<Property*> props;\n                                props.push_back(vp0[id]);\n                                props.push_back(vp1[id]);\n                                props.push_back(vp2[id]);\n                                colorings[level][num]->init_props(props);\n                                num++;\n                            }\n                            k++;\n                        }\n                        i++;\n                    }\n                    p++;\n                }\n\n            }\n        }\n\n        void setTrainId(int train_id) {\n            setId(mngr->props, train_id);\n            for (Coloring* col : colorings[level])\n                col->setTrainId(train_id);\n        }\n\n        void setTestId(int test_id) {\n            setId(mngr->props_test, test_id);\n            for (Coloring* col : colorings[level])\n                col->setTrainId(-1);\n        }\n\n        bool train(int i, int j, int c) {\n            int k = -1;\n            bool any_success = false;\n            for (Coloring* colors : colorings[level]) {\n                k++;\n                if (failed[level][k])\n                    continue;\n                bool ret = colors->train(i, j, c);\n                if (!ret) {\n                    failed[level][k] = true;\n                } else any_success = true;\n            }\n            return any_success;\n        }\n\n        void finalizeTrain(int id) {\n            for (Coloring* colors : colorings[level]) {\n                colors->finalizeTrain(id);\n            }\n        }\n\n        bool post_process(Task* task) {\n            vector<int> selected_scores;\n            for (int i=0; i<colorings[level].size(); i++)\n                if (!failed[level][i]) {\n                    Coloring* selected_col = colorings[level][i];\n                    selected_col->post_process();\n                    selected_cols.push_back(selected_col);\n                    selected_scores.push_back(scores[level][i]);\n                }\n            \n            if (selected_cols.size() > 0) {\n                std::vector<std::pair<Coloring*,int>> zipped;\n                zip(selected_cols, selected_scores, zipped);\n\n                std::sort(std::begin(zipped), std::end(zipped), \n                    [&](const auto& a, const auto& b)\n                    {\n                        return a.second < b.second;\n                    });\n\n                unzip(zipped, selected_cols, selected_scores);\n            }\n\n            return (selected_cols.size() > 0);\n        }\n\n        vector<Coloring*> selected_cols;\n\n    private:\n        \n        vector<vector<Coloring*>> colorings;\n        vector<vector<int>> scores;\n        bool* failed[3];\n        int level = 0;\n        \n        PropManager *mngr = NULL;\n};\n\n\nclass ActMultiProps : public Action {\n    public:\n        ActMultiProps(PropManager *mngr) : mngr {mngr} {\n            name = "ActMultiProps";\n            colors = new MultiColoring(mngr);\n        }\n        \n        ~ActMultiProps() {\n            delete colors;\n        }\n        \n        bool train(Task* task) {\n\n            for (int level=0; level<3; level++) {\n                \n                colors->init(level);\n                int train_id = 0;\n\n                bool failed = false;\n                for (Pair* pair : task->train) {\n                    if (level == 0)\n                        if ((pair->input->nrows != pair->output->nrows) || (pair->input->ncols != pair->output->ncols))\n                            return false;\n                    \n                    colors->setTrainId(train_id);\n                    \n                    for (int i = 0; i < pair->input->nrows; ++i) {\n                        for (int j = 0; j < pair->input->ncols; ++j) {\n                            int c = pair->output->mat[i][j];\n                            bool ret = colors->train(i, j, c);\n                            if (ret == false) {\n                                failed = true;\n                                break;\n                            }\n                        }\n                        if (failed) break;\n                    }\n                    if (failed) break;\n                    \n                    colors->finalizeTrain(train_id);\n                    \n                    train_id++;\n                }\n                \n                if (!failed) {\n                    bool success = colors->post_process(task);\n                    \n                    if (!success)\n                        continue;\n\n                    success = false;\n                    for (Coloring *col: colors->selected_cols) {\n                        int test_id = 0;\n                        vector<Grid*> out_vec;\n                        for (Pair* pair : task->test) {\n                            Grid* out = generateTest(pair->input, test_id, col);\n                            if (out == NULL) {\n                                for (Grid* gg : out_vec)\n                                    delete gg;\n                                out_vec.clear();\n                                break;\n                            }\n                            out_vec.push_back(out);\n                            test_id++;\n                        }\n                        if (out_vec.size() > 0) {\n                            success = true;\n                            test_out.push_back(out_vec);\n                            names_out.push_back(col->name);\n                        }\n                    }\n\n                    if (success) return true;\n                    else colors->selected_cols.clear();\n                }\n            }\n            \n            return false;\n        }\n\n        Grid* generateTest(Grid* grid, int test_id, Coloring* col) {\n            Grid* out = new Grid(*grid);\n            \n            colors->setTestId(test_id);\n            \n            for (int i = 0; i < grid->nrows; ++i)\n                for (int j = 0; j < grid->ncols; ++j) {\n                    int c = col->get_color(i,j);\n                    if (c < 0) {\n                        delete out;\n                        return NULL;\n                    }\n                    out->mat[i][j] = c;\n                }\n            \n            return out;\n        }\n        \n        Grid* generate(Grid* grid, int test_id) {\n            return NULL;\n        }\n\n    private:\n        PropManager *mngr = NULL;\n        MultiColoring *colors = NULL;\n};\n\n\nint main(int argc, char *argv[]) {\n    \n    bool KAGGLE = IsPathExist("/kaggle/working/");\n    \n    std::string current_exec_name = argv[0];\n    std::vector<std::string> all_args;\n    \n    std::string folder = std::string("test");\n    if (argc > 1) {\n        all_args.assign(argv + 1, argv + argc);\n        folder = all_args[0];\n    } else {\n        if (!KAGGLE) {\n            folder = std::string("training");\n            //folder = std::string("evaluation");\n        }\n    }\n    \n    std::string PATH = "C:\\\\StudioProjects\\\\ARC\\\\";\n    std::string PATH_WORK = "C:\\\\StudioProjects\\\\ARC\\\\";\n    std::string SPLITTER = "\\\\";\n    if (KAGGLE) {\n        PATH = "/kaggle/input/abstraction-and-reasoning-challenge/";\n        PATH_WORK = "/kaggle/working/";\n        SPLITTER = "/";\n    }\n    \n    std::cout << "Start" << "\\n";\n    if (KAGGLE)\n        std::cout << "Running Kaggle" << "\\n";\n    else\n        std::cout << "Running Local, folder " << folder << "\\n";\n    \n    std::string path = PATH + folder;\n    std::string sub_path = PATH_WORK + std::string("submission5.csv");\n    std::string succ_path = PATH_WORK + std::string("success.csv");\n    \n    std::ofstream ifs_sub;\n    ifs_sub.open(sub_path.c_str(), std::ofstream::out);\n    ifs_sub << "output_id,output\\n";\n\n    std::ifstream ifs_succ;\n    ifs_succ.open(succ_path.c_str(), std::ifstream::in);\n    vector<string> succ_ids;\n    std::string line;\n    while (getline(ifs_succ, line)) {\n        succ_ids.push_back(line);\n    }\n    ifs_succ.close();\n\n    std::ofstream ofs_succ;\n    ofs_succ.open(succ_path.c_str(), ios_base::app | std::ofstream::out);\n\n    std::map<std::string,std::string> description;\n    description[std::string("3bdb4ada")]=std::string("LATER: grid in grid");\n    description[std::string("63613498")]=std::string("HARD: color sharing between shapes");\n    description[std::string("a5f85a15")]=std::string("TODO: modulo2 column");\n    description[std::string("bda2d7a6")]=std::string("LATER: separate coloring harder case");\n    description[std::string("0692e18c")]=std::string("TODO: zoom-in with reversed colors");\n    description[std::string("1da012fc")]=std::string("HARD: color sharing between shapes");\n    description[std::string("45737921")]=std::string("TODO: property of adjacent color");\n    description[std::string("62ab2642")]=std::string("TODO: smallest and biggest shape property");\n    description[std::string("1caeab9d")]=std::string("LATER: shapes moving");\n    description[std::string("ba97ae07")]=std::string("LATER: need 3 props?");\n    description[std::string("d037b0a7")]=std::string("TODO: neighbor up one prop");\n    description[std::string("f823c43c")]=std::string("TODO: denoising");\n    description[std::string("aedd82e4")]=std::string("TODO: single point shape");\n    description[std::string("0a2355a6")]=std::string("TODO: count holes inside");\n    description[std::string("694f12f3")]=std::string("TODO: shape size, ordered per grid");\n    description[std::string("b230c067")]=std::string("TODO: frequency of appearance of a shape");\n    description[std::string("9565186b")]=std::string("LATER: lvl1 is not enough, need to go into lvl2");\n    description[std::string("150deff5")]=std::string("HARD: fit two shapes into a complex shape");\n    description[std::string("1e0a9b12")]=std::string("LATER: cells moving down");\n    description[std::string("22eb0ac0")]=std::string("TODO: color to the left/right");\n    description[std::string("25ff71a9")]=std::string("LATER: shapes moving one down");\n\n\n    DIR *dir;\n    struct dirent *ent;\n    int n_task_ids = 0;\n\n    if ((dir = opendir(path.c_str())) != NULL) {\n        int score[3] = {}; // correct, error, skipped\n        \n        bool first = true;\n        \n        int* res_train = NULL;\n        int* res_test = NULL;\n        int sz = -1;\n        \n        while ((ent = readdir(dir)) != NULL) {\n            std::string fn = std::string(ent->d_name);\n            if ((fn == std::string(".")) | (fn == std::string("..")))\n                continue;\n            //if (fn != std::string("9def23fe.json")) continue;\n            std::string filepath = PATH + folder + SPLITTER + fn;\n            //std::cout << filepath << "\\n";\n            Task *task = parse_file(filepath);\n\n            PropManager pm(task);\n            pm.addProperty<PropColor>();\n            pm.addProperty<PropHoles>();\n            pm.addProperty<PropModulo>(2, true, true, false);\n            pm.addProperty<PropBorderType>();            \n            \n            std::vector<Action*> acts {\n                new ActMultiProps(&pm),\n            };\n\n            if (res_train == NULL) res_train = new int[acts.size()] {};\n            if (res_test == NULL) res_test = new int[acts.size()] {};\n            if (sz == -1) sz = acts.size();\n\n            bool* success_train = new bool[acts.size()];\n            bool any_success_train = false;\n            \n            if (first) {\n                std::cout << "\\n";\n                for (int a=0; a<acts.size(); ++a)\n                    std::cout << a << ". " << acts[a]->name << "\\n";\n                std::cout << "\\n";\n            }\n            \n            for (int a=0; a<acts.size(); ++a) {\n                success_train[a] = acts[a]->train(task);\n                any_success_train = any_success_train || success_train[a];\n            }\n            \n            int test_id = 0;\n            for (Pair* pair : task->test) {\n                \n                vector<Grid*> outputs;\n\n                bool* success_test = new bool[acts.size()] {};\n                bool any_success_test = false;\n\n                n_task_ids++;\n                std::string id_name = fn.substr(0,8) + "_" + std::to_string(test_id);\n\n                Grid* out = NULL;\n                int printed = 0;\n                if (pair->output == NULL)\n                    ifs_sub << id_name << ",";\n\n                for (int a=0; a<acts.size(); ++a) {\n\n                    int p=0;\n                    for (vector<Grid*> out_vec : acts[a]->test_out) {\n                        if (printed < 3) {\n\n                            out = out_vec[test_id];\n                            bool identical = false;\n                            for (Grid* o : outputs)\n                                if (*o == *out) {\n                                    identical = true;\n                                }\n                            if (identical) {\n                                p++;\n                                continue;\n                            }\n\n                            outputs.push_back(out);\n\n                            cout << acts[a]->names_out[p] << "\\n";\n                            if (pair->output != NULL) {\n                                bool result = ((*out) == (*pair->output));\n                                if (result && !success_test[a]) {\n                                    success_test[a] = true;\n                                    any_success_test = true;\n                                }\n                            } else {\n                                if (printed > 0) {\n                                    ifs_sub << " ";\n                                    cout << id_name << "\\n";\n                                }\n                                ifs_sub << out->flatten();\n                            }\n                            printed++;\n                            p++;\n                        }\n                    }\n                }\n\n                any_success_train = false;\n                for (int a=0; a<acts.size(); ++a)\n                    any_success_train = any_success_train || success_train[a];\n                \n                if (pair->output == NULL) {\n                    if (printed == 0)\n                        ifs_sub << pair->input->flatten();\n                    ifs_sub << "\\n";\n                }\n\n                if (any_success_test) {\n                    if (std::find(succ_ids.begin(), succ_ids.end(), id_name) == succ_ids.end())\n                        ofs_succ << id_name << "\\n";\n                } else {\n                    if (std::find(succ_ids.begin(), succ_ids.end(), id_name) != succ_ids.end())\n                        cout << "achtung: " << id_name << "\\n";\n                }\n                \n                if (any_success_train) {\n                    std::cout << id_name << " ";\n                    for (int a=0; a<acts.size(); ++a)\n                        std::cout << success_train[a] + success_test[a];\n                    if (any_success_test)\n                        std::cout << " > PASS    ";\n                    else {\n                        std::cout << " > FAILED  ";\n                    \n                        map<string,string>::iterator it = description.find(fn.substr(0,8));\n                        if (it != description.end())\n                            cout << ":" << it->second;\n                    }\n\n                    std::cout << "\\n";\n                }\n                \n                if (any_success_test && any_success_train)\n                    score[0]++;\n                if (!any_success_test && any_success_train)\n                    score[1]++;\n                if (!any_success_test && !any_success_train)\n                    score[2]++;\n                \n                for (int a=0; a<acts.size(); ++a) {\n                    res_train[a] += success_train[a];\n                    res_test[a] += success_test[a];\n                }\n\n                test_id++;\n            }\n            \n            for (int a=0; a<acts.size(); ++a) {\n                delete acts[a];\n            }\n            delete success_train;\n            delete task;\n            \n            first = false;\n        }\n        \n        char buffer [50];\n        cout << "\\n";\n        cout << "         ";\n        for (int a=0; a<sz; ++a) {\n            sprintf(buffer,"%3d",a);\n            cout << buffer;\n        }\n        cout << "\\n";\n        cout << "train    ";\n        for (int a=0; a<sz; ++a) {\n            sprintf(buffer,"%3d",res_train[a]);\n            cout << buffer;\n        }\n        cout << "\\n";\n        cout << "test     ";\n        for (int a=0; a<sz; ++a) {\n            sprintf(buffer,"%3d",res_test[a]);\n            cout << buffer;\n        }\n        cout << "\\n";\n\n        delete res_train;\n        delete res_test;\n\n        std::cout << "correct: " << score[0] << " error: " << score[1] << " skipped: " << score[2] << "\\n";\n        \n        closedir(dir);\n    } else {\n        perror("failed to read files");\n        return 1;\n    }\n    \n    ifs_sub.close();\n    ofs_succ.close();\n\n    cout << "n_task_ids: " << n_task_ids << "\\n";\n    if ((n_task_ids != 102) && (n_task_ids != 104)) {\n        remove(sub_path.c_str());\n    }\n\n    return 0;\n}')


# ## Compile

# In[ ]:


get_ipython().system('g++ -shared -fPIC -o mycplusplus.dll mycplusplus.cpp')
#!g++ -shared -o mycplusplus.dll mycplusplus.cpp #against symbol `_ZNSt6vectorIiSaIiEEC1Ev' can not be used when making a shared object; recompile with -fPIC


# In[ ]:


get_ipython().system('ls ../working/')


# In[ ]:


from ctypes import cdll
import _ctypes

lib = cdll.LoadLibrary('../working/mycplusplus.dll')
try:
    print(lib)
except:
    print("Function not found")
#unload so we can rebuild it again
_ctypes.dlclose(lib._handle)


# In[ ]:


os.system("g++ mycplusplus.cpp && ./a.out")


# In[ ]:


#!du -hs submission5.csv


# In[ ]:


## Compilation and running
st = time.time()
if KAGGLE:
    print(os.system("g++ -pthread -lpthread -O3 -std=c++17 -o main mycplusplus.cpp 2> error.log"))
else:
    print(os.system("g++ -pthread -lpthread -g -std=c++17 -o main mycplusplus.cpp 2> error.log"))
print("running time:", time.time()-st)


# In[ ]:


st = time.time()
if KAGGLE:
    get_ipython().system('./main training')
else:
    get_ipython().system('main.exe training')
print("running time:", time.time()-st)


# In[ ]:


st = time.time()
if KAGGLE:
    get_ipython().system('./main evaluation')
else:
    get_ipython().system('main.exe evaluation')
print("running time:", time.time()-st)


# In[ ]:


st = time.time()
if KAGGLE:
    get_ipython().system('./main test')
else:
    get_ipython().system('main.exe test')
print("running time:", time.time()-st)


# In[ ]:


subCpath = '/kaggle/working/'

get_ipython().system('ls /kaggle/working/')


# In[ ]:


## re-reading from submission file
sample_sub5 = pd.read_csv(subCpath+'/'+'submission5.csv')
sample_sub5


# ## Merge all

# In[ ]:


sample_sub1 = sample_sub1.reset_index()
sample_sub1 = sample_sub1.sort_values(by="output_id")

sample_sub2 = sample_sub2.sort_values(by="output_id")
sample_sub3 = sample_sub3.sort_values(by="output_id")
sample_sub4 = sample_sub4.sort_values(by="output_id")
sample_sub5 = sample_sub5.sort_values(by="output_id")

out1 = sample_sub1["output"].astype(str).values
out2 = sample_sub2["output"].astype(str).values
out3 = sample_sub3["output"].astype(str).values
out4 = sample_sub4["output"].astype(str).values
out5 = sample_sub5["output"].astype(str).values

# Use all:
merge_output = []
for o1, o2, o3, o4, o5 in zip(out1, out2, out3, out4, out5):
    o = o1.strip().split(" ")[:1] + o2.strip().split(" ")[:2] + o3.strip().split(" ")[:3] + o4.strip().split(" ")[:4] + o5.strip().split(" ")[:5]
    o = " ".join(o[:6])
    merge_output.append(o)

# Use only two:
#for o1, o2 in zip(out1, out2):
#    o = o1.strip().split(" ")[:1] + o2.strip().split(" ")[:2]
#    o = " ".join(o[:3])
#    merge_output.append(o)
    
sample_sub1["output"] = merge_output
sample_sub1["output"] = sample_sub1["output"].astype(str)
sample_sub1 = sample_sub1.drop(['index'], axis=1)
sample_sub1.to_csv("submission.csv", index=False)

