#!/usr/bin/env python
# coding: utf-8

# # Cellular Automata with ColorFreqEncoder

# Part of 9th place solution

# Based on @teddykoker "Training Cellular Automata Part II: Learning Tasks" kernel: https://www.kaggle.com/teddykoker/training-cellular-automata-part-ii-learning-tasks.

# Changes comparing to original kernel: using 3 random seeds, encoding colors by frequency (useful for tasks when colors change from example to example).

# In[ ]:


import os
import json
import numpy as np
from pathlib import Path
import random
from collections import Counter
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[ ]:


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[ ]:


data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge')
train_path = data_path / 'training'
valid_path = data_path / 'evaluation'
test_path = data_path / 'test'


# In[ ]:


SEEDS = [0,1,2]


# In[ ]:


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)


# In[ ]:


set_seeds(0) # i set seed here just in case. really seads are set in predict_tasks function. 


# In[ ]:


paths = {'train': train_path, 'eval': valid_path, 'test': test_path}

def get_tasks(dataset='train'):
    path = paths[dataset]
    fns = sorted(os.listdir(path))
    tasks = {}
    for idx, fn in enumerate(fns):
        fp = path / fn
        with open(fp, 'r') as f:
            task = json.load(f)
            tasks[fn.split('.')[0]] = task
    return tasks


test_tasks = get_tasks('test')


# In[ ]:


def input_output_shape_is_same(task):
    return all([np.array(el['input']).shape == np.array(el['output']).shape for el in task['train']])


# In[ ]:


class Encoder():
    def __init__(self, task):
        pass
    
    def encode(self, inp):
        inp = np.array(inp)
        img = np.full((10, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
        for i in range(10):
            img[i] = (inp==i)
        return img
    
    def encode_y(self, inp):
        inp = np.array(inp)
        return inp
    
    
    def decode(self,img, num_states):
        return img[:, :num_states, :, :].argmax(1).squeeze().cpu().numpy()


# In[ ]:


def encode_colors(array, color2num):

    new_array = np.empty_like(array)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            new_array[i, j] = color2num[array[i, j]]
    return new_array


# In[ ]:


class ColorFreqEncoder():
    def __init__(self, sample, task):
        train = task['train']
        sample0 = train[0]
        input0 = sample0['input']
        output0 = sample0['output']
        input0_colors_cnt = Counter(np.array(input0).flatten().tolist())
        output0_colors_cnt = Counter(np.array(output0).flatten().tolist())
        sorted_input0_colors = [color for color,_ in input0_colors_cnt.most_common()]
        sorted_output0_colors =  [color for color,_ in output0_colors_cnt.most_common()
                                 if color not in sorted_input0_colors]
        input_colors_cnt = Counter(np.array(sample['input']).flatten().tolist())
        sorted_input_colors = [color for color,_ in input_colors_cnt.most_common()]
        task_colors = sorted_input_colors + sorted_output0_colors
        self.color2num = {color:num for num, color in enumerate(task_colors)}
        self.num2color = task_colors
        self.num_states = len(task_colors)

        self.encoder = Encoder(sample)
    
    def encode(self, array):
        array = np.array(array)
        array = encode_colors(array, self.color2num)
        img = self.encoder.encode(array)
        return img
    def encode_y(self, array):
        array = np.array(array)
        array = encode_colors(array, self.color2num)
        return self.encoder.encode_y(array)
    
    def decode(self, img):
        out = self.encoder.decode(img, self.num_states)
        return encode_colors(out, self.num2color)


# In[ ]:


class CAModel(nn.Module):
    def __init__(self, num_states):
        super(CAModel, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(num_states,128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, num_states, kernel_size=1)
        
        )
        
    def forward(self, x, steps=1):
        for _ in range(steps):
            x = nn.functional.pad(x, (1,1,1,1), 'constant', 1)
            x = self.transition(torch.softmax(x, dim=1))
        return x


# In[ ]:


@torch.no_grad()
def predict(model, task, num_steps=100, mode='train'):
    task_ = task
    task = task[mode]
    model.eval()
    predictions = []
    for sample in task:
        encoder = ColorFreqEncoder(sample, task_)
        x = torch.from_numpy(encoder.encode(sample["input"])).unsqueeze(0).float().to(device)
        pred = encoder.decode(model(x, num_steps))
        predictions.append(pred)
    return predictions


def solve_task(task, max_steps=10):
    
    task_ = task
    task = task['train']
    model = CAModel(10).to(device)
    model.train()
    num_epochs = 100
    num_epochs_2 = 1
    criterion = nn.CrossEntropyLoss()
    losses = np.zeros((max_steps - 1) * num_epochs * num_epochs_2)
    for ep2 in range(num_epochs_2):
        for num_steps in range(ep2 +1, max_steps):
            optimizer = torch.optim.Adam(model.parameters(), lr=(0.1 / (num_steps * 2)))

            for e in range(num_epochs):
                optimizer.zero_grad()
                loss = 0.0

                for sample in task:
                    encoder= ColorFreqEncoder(sample, task_)
                    x = torch.from_numpy(encoder.encode(sample["input"])).unsqueeze(0).float().to(device)
                    y = torch.tensor(encoder.encode_y(sample["output"])).long().unsqueeze(0).to(device)
                    y_pred = model(x, num_steps)
                    loss += criterion(y_pred, y)

                    y_in = torch.from_numpy(encoder.encode(sample["output"])).unsqueeze(0).float().to(device)
                    y_pred = model(y_in, 1) 
                    loss +=  criterion(y_pred, y)


                loss.backward()
                optimizer.step()
                losses[ep2 * num_epochs * (num_steps - 1) + (num_steps - 1) * num_epochs + e] = loss.item()
                random.shuffle(task)
            
    return model, num_steps, losses


# In[ ]:


def color_n(array):
    array = np.array(array)
    return len(set(array.flatten().tolist()))


# In[ ]:


def get_colors(array):
    array = np.array(array)
    return set(array.flatten().tolist())


# In[ ]:


def same_color_number(task):
    train = task['train']
    test = task['test']
    inputs = [sample['input'] for sample in train + test]
    outputs = [sample['output'] for sample in train]
    input_color_n = color_n(inputs[0])
    output_color_n =color_n(outputs[0])
    only_output_colors = get_colors(outputs[0]) - get_colors(inputs[0])
    for input in inputs:
        if color_n(input) != input_color_n:
            return False
    for output in outputs:
        if color_n(output) != output_color_n:
            return False
    for input,output in zip(inputs, outputs):
        if get_colors(output) - get_colors(input) != only_output_colors:
            return False
    return True


# In[ ]:


def predict_tasks(tasks):
    predictions = {}
    k = 0
    for idx, task in tqdm(tasks.items()):

        if input_output_shape_is_same(task) and same_color_number(task): 
            preds = [[]] * len(task['test'])
            for i in SEEDS:
                set_seeds(i)
                model, num_steps, _ = solve_task(task)
                pred = predict(model, task, mode='test')
                for j in range(len(task['test'])):
                    preds[j].append(pred[j])
        else:
            preds = []
        k += 1
        predictions[idx] = preds

    return predictions


# In[ ]:


predictions = predict_tasks(test_tasks)


# In[ ]:


def get_string(pred):
    str_pred = str([list(row) for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


# In[ ]:


def submit():
    submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='output_id')
    submission['output'] = ''
    test_fns = sorted(os.listdir(test_path))
    count = 0
    for fn in test_fns:
        fp = test_path / fn
        with open(fp, 'r') as f:
            task_idx = fn.split('.')[0]
            all_input_preds = predictions[task_idx]
            if all_input_preds:
                count += 1

                for i, preds in enumerate(all_input_preds):
                    output_id = str(fn.split('.')[-2]) + '_' + str(i)
                    string_preds = [get_string(pred) for pred in preds[:3]]
                    pred = ' '.join(string_preds)
                    submission.loc[output_id, 'output'] = pred
    print(count)
    submission.to_csv('submission.csv')

submit()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




