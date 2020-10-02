#!/usr/bin/env python
# coding: utf-8

# 1. [Francois chollet paper: "On the Measure of Intelligence"](https://arxiv.org/abs/1911.01547)
# 2. [Cellular automata on this competition](https://www.kaggle.com/arsenynerinovsky/cellular-automata-as-a-language-for-reasoning), and idea to solve thinks with somehow "basic operations"
# 3. A possible solution to mix basic operations: [genetic algorithm](https://www.kaggle.com/zenol42/dsl-and-genetic-algorithm-applied-to-arc)
# 4. [This video, an introduction to this challenge](https://www.youtube.com/watch?v=K5KDZLHsr1o&fbclid=IwAR2erffIK1pJhMHTSQlCsczIpWXab87pVJftQNK1dKWTuYcmvDCLXtqol0k), you can watch from min 26 because before it's just the explantion of the paper
# 5. Search maybe for a baseline or starter notebook in order to obtain basic input and ploting functions or a first idea to approach whith if you liked it (just type "starter" or "basline" in the search bar in the [competition site](https://www.kaggle.com/c/abstraction-and-reasoning-challenge/notebooks)

# In[ ]:


import numpy as np
import os
import matplotlib.pyplot as plt
from pprint import pprint as pp
import seaborn as sns
import json
import networkx as nx
from itertools import product

in_kaggle = True 


# In[ ]:


data_dir = './data/' if not in_kaggle else '/kaggle/input/abstraction-and-reasoning-challenge/'
training_path, evaluation_path, test_path = f'{data_dir}training', f'{data_dir}evaluation', f'{data_dir}test'
training_tasks, evaluation_tasks, test_tasks = sorted(os.listdir(training_path)), sorted(os.listdir(evaluation_path)), sorted(os.listdir(test_path))
colors = { cname: c for c, cname in enumerate(['black', 'blue', 'red', 'green', 'yellow', 'gray', 'magenta', 'orange', 'sky', 'brown']) }


# In[ ]:


createGridDict = lambda m: { (i,j): { } for i in range(m.shape[0]) for j in range(m.shape[1]) }
cellColor = lambda x, i, j: x[i, j]

def loadJsonFromPath(path):
  j = None
  with open(path, 'r') as f:
    j = json.load(f)
  return j

def getGeneralInfoFromTask(task_path):
  j = loadJsonFromPath(task_path)
  j_train, j_test = j['train'], j['test']
  n_patterns, n_test = len(j_train), len(j_test)
  return {
    'task_json': j, 
    'train': j_train, 
    'test': j_test, 
    'n_patterns': n_patterns,
    'n_test': n_test
  }


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib import colors as cls
from pprint import pprint as pp
import numpy as np

cmap = cls.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00','#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = cls.Normalize(vmin=0, vmax=9)
    
def plotResults(task_samples, predictions):
  for sample, prediction in zip(task_samples, predictions):
    t_in, t_out, prediction = np.array(sample["input"]), np.array(sample["output"]), np.array(prediction)
    titles = [f'Input {t_in.shape}', f'Output {t_out.shape}', f'Predicted {prediction.shape}']
    figures = [t_in, t_out, prediction]
    fig, axs = plt.subplots(1, 3, figsize=(2*3, 32))
    for i, (figure, title) in enumerate(zip(figures, titles)):
      plotOne(axs[i], figure, title)
    plt.show()

def showTotalColors():
  # 0:black, 1:blue, 2:red, 3:greed, 4:yellow,
  # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
  plt.figure(figsize=(5, 2), dpi=100)
  plt.imshow([list(range(10))], cmap=cmap, norm=norm)
  plt.xticks(list(range(10)))
  plt.yticks([])
  plt.show()

def plotOne(ax, task_in_out, title):
  ax.imshow(task_in_out, cmap=cmap, norm=norm)
  ax.set_title(title)
  #ax.axis('off')
  ax.set_yticks([x-0.5 for x in range(1+task_in_out.shape[0])])
  ax.set_xticks([x-0.5 for x in range(1+task_in_out.shape[1])])  
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  ax.grid(True, which='both', color='lightgrey', linewidth=0.5) 

def plot_task(task_info, n, name, test_task=False):
  # 0:black, 1:blue, 2:red, 3:greed, 4:yellow,
  # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
  n_pairs = task_info['n_patterns'] + task_info['n_test']
  train_info, test_info = task_info['train'], task_info['test']
  
  plt.subplots_adjust(wspace=0, hspace=0)
  fig, axs = plt.subplots(2, n_pairs, figsize=(4*n_pairs,8),  dpi=50)
  fig_num = 0
  
  for i, t in enumerate(task_info['train']):
    t_in, t_out = np.array(t["input"]), np.array(t["output"])
    if (t_in > 9).any() or (t_out > 9).any(): print(f"Number Out of color range ({np.max(t_in)}, {np.max(t_out)}")
    plotOne(axs[0, fig_num], t_in, f'{n}: Train Input {i} - {t_in.shape} - {name}')
    plotOne(axs[1, fig_num], t_out, f'{n}: Train Output {i} - {t_out.shape} - {name}')
    fig_num += 1
  for i, t in enumerate(task_info['test']):
    t_in, t_out = np.array(t["input"]), np.array(t["output"]) if not test_task else None
    if (t_in > 9).any() or (t_out > 9).any(): print(f"Number Out of color range ({np.max(t_in)}, {np.max(t_out)}")
    plotOne(axs[0, fig_num], t_in, f'{n}: Test Input {i} - {t_in.shape} - {name}')
    if not test_task: 
      plotOne(axs[1, fig_num], t_out, f'Test Output {i} - {t_out.shape} - {name}')
    else:
      axs[1, fig_num].axis('off')
    fig_num += 1

  plt.tight_layout()
  plt.show()


# In[ ]:


color_info = {
  **{ cname: lambda x,i,j,color: x[i,j] == color for cname, cvalue in colors.items() } # inputs(x, *cell, color)
}

caracterize = {
  **color_info,  # inputs(x, *cell, color)
}

def caracterizeIO(m, caracterize, characteristics):
  """Convert a matrix in a dict where keys are position in matrices and values are characteristics"""
  m_d = createGridDict(m)
  for name, f in caracterize.items():
    for cell, _ in m_d.items():
      is_same_color = lambda fn,x,i,j: (np.unique(fn(x,i,j))).shape[0] == 1
      if name in characteristics:
        if name in colors:
          m_d[ cell ][ name ] = f(m, *cell, colors[name])
        else:
          m_d[ cell ][ name ] = is_same_color(f, m, *cell)
  return m_d

def convertToDimentionalMatrix(m, m_d, caracterize):
  """From caracterized matrix (dict of { (cell): characteristics, ... }), get a matrix for each caracteristic"""
  n_caracteristics = len(caracterize)
  dim_m = np.empty((n_caracteristics, *m.shape))
  for cell, char_cell in m_d.items():
    color_count, others_count = 0, 0
    for j, cname in enumerate(caracterize):
      x,y = cell
      # from 0 to ncolors, let it be first places, afterwards the others
      if cname in colors:
        dim_m[color_count, x, y] = int(char_cell[cname]) if type(char_cell[cname]) == bool else char_cell[cname]
        color_count += 1
      else:
        dim_m[others_count+len(colors), x, y] = int(char_cell[cname]) if type(char_cell[cname]) == bool else char_cell[cname]
        others_count += 1
  return dim_m

def transformCaracterize(m, chars):
  x = np.array(m)
  x_d = caracterizeIO(x, caracterize, chars)
  return convertToDimentionalMatrix(x, x_d, chars)


# In[ ]:


import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.nn.functional as F
import torch.optim as optim
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[ ]:


class SimpleInputOutputNet(nn.Module):  
  def __init__(self, init_dict):
    """input_dict.keys() == ['in_ch', 'in_out_ch', 'in_ker', 'out_ch']"""
    super(SimpleInputOutputNet, self).__init__()
    in_ch, in_out_ch, in_ker = init_dict['in_ch'], init_dict['in_out_ch'], init_dict['in_ker']
    out_ch = init_dict['out_ch']
    self.conv1 = nn.Conv2d(in_ch, in_out_ch, in_ker, padding=1)
    self.conv2 = nn.Conv2d(in_out_ch, out_ch, kernel_size=1)
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.conv2(x)
    return x
  def is_automata(self): return False


# In[ ]:


def isSameInputOutputShape(task):
    return all([np.array(el['input']).shape == np.array(el['output']).shape for el in task['train']])
  
def getClassyOutput(task): 
  """get simply the output as a class"""
  return task['output']

def getCharacterizedBinaryOutput(task, output_characteristics):
  """get output different dimensions with matrices of 0 and 1"""
  return transformCaracterize(task['output'], output_characteristics)

def getOutput(task, output_characteristics, is_class=True):
  return getClassyOutput(task) if is_class else getCharacterizedBinaryOutput(task, output_characteristics)

def getInput(task, input_characteristics):
  return transformCaracterize(task['input'], input_characteristics)

def solveTask(epochs, task_samples, net, criterion, input_char, output_char, optim_args={'lr':0.1}, steps=1, is_class=True, train_outToOut=False, show_prints=False):
  all_losses, step = [], 0
  for step in range(0, steps):
    optimizer = optim.Adam(net.parameters(), lr=(optim_args['lr']/(step+1)))
    #optimizer = optim.Adam(net.parameters(), **optim_args)
    is_automata = net.is_automata()
    losses = []
    for epoch in range(epochs):
      optimizer.zero_grad()
      running_loss = 0.0
      for t, task_sample in enumerate(task_samples):
        # -- Get Input and Output --
        inp, out = getInput(task_sample, input_char), getOutput(task_sample, output_char, is_class=is_class)
        inp = tensor(inp).unsqueeze(0).float().to(device) # adds_a dimension to the shape, and convert to float
        out = tensor(out).unsqueeze(0).long().to(device) # adds_a dimension to the shape, and convert to long == int  
        #inp = tensor(inp2img(sample["input"])).unsqueeze(0).float().to(device) 
        #out = tensor(sample['output']).long().unsqueeze(0).to(device)
        if show_prints: print(f'(input, output) shape ({inp.shape}, {out.shape})')

        predicted = net(inp) if not is_automata else net(inp, step+1)
        running_loss += criterion(predicted, out)
        if show_prints: print(f'(predicted) shape ({predicted.shape})')

        if train_outToOut:
          out_inp = getOutput(task_sample, input_char, False)
          out_inp = tensor(out_inp).unsqueeze(0).float().to(device) 
          #out_inp = tensor(inp2img(sample["output"])).unsqueeze(0).float().to(device) 
          if show_prints: print(f'(output as input) shape {out_inp.shape}')
          predicted = net(out_inp) if not is_automata else net(out_inp, 1)
          running_loss += criterion(predicted, out)
      running_loss.backward()
      optimizer.step()
      losses.append(running_loss.item())
    all_losses += losses
  return net, all_losses, step 


# In[ ]:


@torch.no_grad()
def predictOutputTenDimColors(net, task, input_char, steps=1 ):
  predictions = []
  for sample in task:
    inp = getInput(sample, input_char)
    inp = tensor(inp).unsqueeze(0).float().to(device)
    is_automata = net.is_automata()
    pred = net(inp) if not is_automata else net(inp, (steps+1))
    pred = pred.argmax(1) # (shape: [1, 10, _,_]) argmax-Returns the indices of the maximum value of all elements in the input tensor. dim: dimension to reduce
    pred = pred.squeeze().cpu().numpy() #squeeze returns tesnor with dimensions of 1 removed, so the first dimension in this case
    predictions.append(pred)
  return predictions

def getTaskSuccess(task_samples, predictions):
  is_same_array = lambda a,b,cell: np.array_equal(cellColor(a, *cell), cellColor(b, *cell))
  success = []
  for sample, prediction in zip(task_samples, predictions):
    output, prediction = np.array(sample['output']), np.array(prediction)
    out_d, pred_d = createGridDict(output), createGridDict(prediction)
    same = np.array([ is_same_array(output, prediction, cell) for cell, _ in out_d.items()])
    success += [ np.all(same) ]
  good = sum([int(v) for v in success])
  return success, good, len(success) - good     
  
def fillReport(report, task_path, train, test):
  train_success, train_good, train_bad = train
  test_success, test_good, test_bad = test
  if train_bad == 0: report['trained_success'] += 1
  if train_bad == 0 and test_bad > 0: report['task_train_success'] += [task_path]
  if test_bad == 0: report['success'] += 1
  if train_bad == 0 and test_bad == 0: report['task_success'] += [task_path]
  if train_good > 0 and train_bad > 0: report['half_sucess'] +=1
  if train_good == 0: report['fails'] += 1


# In[ ]:


# --- RUN For Simple Convolutional Layer
def runSimpleNet(task_paths, init_params, in_out_char, model, criterion=nn.CrossEntropyLoss(), optim_args={'lr':0.1}, is_class=True, train_outToOut=False, steps=1, epochs=100,
                 show_prints=False, show_plots=False, show_results=True, grabbing_net=False, is_test=False):
  report = { 'trained_success': 0, 'half_sucess': 0, 'fails': 0, 'success': 0, 'task_train_success': [], 'task_success': [] }
  nets_info = { }
  showed = False
  folder_path = training_path if not is_test else test_path
  if grabbing_net: net = model(init_params).to(device)
  for task_path in task_paths:
    task_info = getGeneralInfoFromTask(f'{folder_path}/{task_path}')
    if not isSameInputOutputShape(task_info): continue
    input_char, output_char = in_out_char['input_char'], in_out_char['output_char']
    samples = task_info['train']
    if not grabbing_net: net = model(init_params).to(device)
    if not showed and show_results: 
      print(net.eval())
      showed = True
    net, losses, step = solveTask(
      epochs=epochs, task_samples=samples,
      net=net, 
      criterion=criterion, #nn.CrossEntropyLoss() OR nn.MSELoss()
      optim_args=optim_args,
      input_char=input_char, output_char=output_char,
      steps=steps,
      is_class=is_class,
      train_outToOut=train_outToOut,
      show_prints=show_prints
    )
    if show_plots: plt.plot(losses)
    predictions = predictOutputTenDimColors(net, samples, input_char, step)
    success, good, bad = getTaskSuccess(samples, predictions)
    if (not show_prints and show_results) or show_prints: print(f'--- {good} out of {len(success)} were ok ---') if bad != 0 else print('---- Successful training!!! :) ----')
    if show_plots: plotResults(samples, predictions)
    test_success, test_good, test_bad = [], 0, 1
    if bad == 0 and not is_test:
      test_predictions = predictOutputTenDimColors(net, task_info['test'], input_char, step)
      test_success, test_good, test_bad = getTaskSuccess(task_info['test'], test_predictions)
      if show_results: print(f'--- {test_good} out of {len(test_success)} were ok ---') if test_bad != 0 else print('---- Successful ALL!!! :) ----')
    if is_test:
      test_predictions = predictOutputTenDimColors(net, task_info['test'], input_char, step)
      nets_info[task_path] = { 'predictions': test_predictions, 'input': [ sample['input'] for sample in task_info['test'] ] }
    else:
      fillReport(report, task_path, (success, good, bad), (test_success, test_good, test_bad))
  return report, nets_info


# In[ ]:


input_char, output_char = color_info.keys(), color_info.keys()
print(f'Characteristics for (input - {len(input_char)}, output - {len(output_char)}): \ninp -> {input_char}\nout -> {output_char} ')
report, net = runSimpleNet(
  show_plots=True,
  show_prints=False,
  show_results=True,
  model= SimpleInputOutputNet,
  optim_args={'lr':0.2},
  task_paths= ['00d62c1b.json'], 
  in_out_char={ 'input_char': input_char, 'output_char': output_char },
  init_params={ 'in_out_ch': 128, 'in_ker':3, 'in_ch': len(input_char), 'out_ch': len(output_char) } 
)
pp(report)


# In[ ]:


output_char, input_char = color_info.keys(), color_info.keys()
report, net = runSimpleNet( show_plots=False, show_prints=False, show_results=False, model=SimpleInputOutputNet,
    steps=1,
    epochs=50,
    train_outToOut=True,
    optim_args={'lr': 0.05},
    task_paths= training_tasks, # ['00d62c1b.json'], # 
    in_out_char={ 'input_char': input_char, 'output_char': output_char },
    init_params={ 'in_out_ch': 128, 'in_ker':3, 'in_ch': len(input_char), 'out_ch': len(output_char) } 
  )
pp(report)


# In[ ]:


import pandas as pd
submission = pd.read_csv('/kaggle/input/abstraction-and-reasoning-challenge/sample_submission.csv', index_col='output_id')
display(submission.head())


# In[ ]:


output_char, input_char = color_info.keys(), color_info.keys()
report, nets_info = runSimpleNet( show_plots=False, show_prints=False, show_results=False, model=SimpleInputOutputNet,
    steps=1,
    epochs=50,
    train_outToOut=True,
    optim_args={'lr': 0.05},
    task_paths= test_tasks, 
    in_out_char={ 'input_char': input_char, 'output_char': output_char },
    init_params={ 'in_out_ch': 128, 'in_ker':3, 'in_ch': len(input_char), 'out_ch': len(output_char) } ,
    is_test=True
  )
pp(report)


# In[ ]:


def getNGuessesFromExemple(submission_exo, output_id, n=1):
  out = submission_exo.loc[output_id].output
  vals = out.split(' ')
  return vals[:n]
  
def getSubmissionDF(nets_info, submission):
  submission_dict = {}
  for task_name in test_tasks:
    if not task_name in submission_dict: submission_dict[task_name] = { 'indexes': [] }
  
  for task_name, info in submission_dict.items():
    if not task_name in nets_info: continue
    net_info = nets_info[task_name]
    predictions, inputs = net_info['predictions'], net_info['input']
    sub_like_pred, sub_like_inp = '|', '|'
    submission_dict[task_name]['strings'], submission_dict[task_name]['inputs'] = [], []
    for inp in inputs:
      #sub_like_inp += ' |' if sub_like_inp != '' else '|'
      for row in inp:
        sub_like_inp += ''.join([ str(v) for v in row]) + '|'
      sub_like_inp += ' '
      submission_dict[task_name]['inputs'] += [sub_like_inp]
    for prediction in predictions:
      #sub_like_pred += ' |' if sub_like_pred != '' else '|'
      for row in prediction:
        sub_like_pred += ''.join([ str(v) for v in row]) + '|'
      sub_like_pred += ' '
      submission_dict[task_name]['strings'] += [sub_like_pred]
      

  to_submit = {}
  for task_name, info in submission_dict.items():
    if not task_name in nets_info: continue
    task, _ = task_name.split('.')
    net_info = nets_info[task_name]
    predictions = net_info['predictions']
    for index, prediction in enumerate(predictions):
      output_id = f'{task}_{index}'
      if not 'strings' in info: 
        to_submit[output_id] = getNGuessesFromExemple(submission, output_id)[0] + ' '
      else :
        to_submit[output_id] = info['strings'][index]
        
  sub_df = pd.DataFrame(to_submit.values(), index=to_submit.keys(), columns=['output'])
  sub_df.index.name = 'output_id'
  return sub_df

submission = pd.read_csv('/kaggle/input/abstraction-and-reasoning-challenge/sample_submission.csv')
df = getSubmissionDF(nets_info, submission)
display(df.head())
display(df.info())
df.to_csv('submission.csv')


# In[ ]:




