#!/usr/bin/env python
# coding: utf-8

# # CyberLuke's Tensor Operations 101
# 
# ### how to not shoot yourself in the foot
# 
# A starter project for PyTorch & Tensor operations
# - Section 1 - How to work with Tensor seamlessly on CPU, GPU and TPU
# - Section 2 - Pretty print & plot your 1D Tensor
# - Section 3 - Advanced custom printing technique
# - Section 4 - Plot multi-dimensional Tensor with Plotly
# - Section 5 - Tensor Operation API limits & solutions

# ### [TPU] Uncomment this line to test on TPU (tested on Kaggle, should work on Google Collab)

# In[ ]:


# Experimental Kaggle TPU support for PyTorch (should work also on Google Collab)!
#!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
#!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev


# ### Imports

# In[ ]:


# Import torch and other required modules
import torch
import random
import os
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt


# ### Pre-seed, so we get reproducible results

# In[ ]:


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# In[ ]:


seed = 2020
seed_everything(seed)


# ## Section 1 - How to work with Tensor seamlessly on CPU, GPU and TPU
# 
# In the past working with PyTorch on both CPU and GPU required you to rewrite the code. Nowadays it is still not perfect, but I will show you how it could be done including a few corner cases.

# In[ ]:


# Lazily set the current device to reuse in whole project (important!)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# If GPU, prints: 'cuda:0', where 0 is GPU id (for multi GPU system)
# If CPU, prints: 'cpu'
# If TPU, prints: 'cpu' as well
print(device)

# Additional GPU nice to have info
if torch.cuda.is_available():
    # How to get current GPU
    print(torch.cuda.get_device_name())
    print(torch.cuda.get_device_properties(torch.cuda.current_device()))
    
    # How to navigate on multi GPU system
    for i in range(0, torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i))


# If GPU, prints: 'cuda:0', where 0 is GPU id (for multi GPU system)
# 
# If CPU, prints: 'cpu'
# 
# If TPU, prints: 'cpu' as well

# In[ ]:


# Now we initialise Tensor, it is important to supply device parameter everywhere in order to make your code portable!
tensor = torch.tensor(range(15), device=device)


# The example above will initialise Tensor on any device we have currently set by default. This is an important step to make it portable! Lets call it a best practice until we have a better solution in PyTorch.

# In[ ]:


# Now if we try to print Tensor as a Numpy array, it will break on GPU
print("Print from Numpy directly:")
print(tensor.numpy())
print()
# The solution is to move it to CPU memory
print("A better portable solution for multi-platform development (GPU):")
print(tensor.cpu().numpy())


# The first command `tensor.numpy()` will break on GPU:
# > TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

# Remember always initialize your device and pass the device parameter, where possible. For built-in Python functions and external libraries like Numpy or plotting functions, always add cpu() call just to be safe (best practice at the times of writing)

# ## Section 2 - Pretty print & plot your Tensor
# 
# torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None) [[SOURCE]](https://pytorch.org/docs/stable/_modules/torch/_tensor_str.html#set_printoptions)

# In[ ]:


# Example 1 - Print 1D Tensor
tensor_1d = torch.tensor([1.1, 2., 3., 4., 5.], device=device)

torch.set_printoptions(profile='default', sci_mode=False)

print('Python default:')
print(tensor_1d)
print()
print('PyTorch Scientific Mode:')
torch.set_printoptions(profile='short', sci_mode=True)
print(tensor_1d)
torch.set_printoptions(profile='default', sci_mode=False)


# Explanation about example

# In[ ]:


# Example 2 - Convert 1D Tensor to 2D Tensor
tensor_2d = tensor_1d.view(tensor_1d.size()[0], 1)

print(tabulate(tensor_2d))


# In[ ]:


# Example 3 - breaking it
# for multi-dimensional tensors we can use charting library to plot them
plt.imshow(tensor_2d)

plt.imshow(tensor_2d.cpu()) # do not forget for cpu() method call as described in Section 1


# This will break on GPU: 
# > TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

# ## Section 3 - Advanced custom printing technique
# 
# Note: you may need to hit "trust notebook" at the top to allow it to inject HTML.

# In[ ]:


# Example 1 - working
import numpy as np
import IPython.core.display

def _html_repr_helper(contents, index, is_horz):
    dims_left = contents.ndim - len(index)
    if dims_left == 0:
        s = contents[index]
    else:
        s = '<span class="numpy-array-comma">,</span>'.join(
            _html_repr_helper(contents, index + (i,), is_horz) for i in range(contents.shape[len(index)])
        )
        s = ('<span class="numpy-array-bracket numpy-array-bracket-open">[</span>'
            '{}'
            '<span class="numpy-array-bracket numpy-array-bracket-close">]</span>'.format(s))
        
    # apply some classes for styling
    classes = []
    classes.append('numpy-array-slice')
    classes.append('numpy-array-ndim-{}'.format(len(index)))
    classes.append('numpy-array-ndim-m{}'.format(dims_left))
    if is_horz(contents, len(index)):
        classes.append('numpy-array-horizontal')
    else:
        classes.append('numpy-array-vertical')
    
    hover_text = '[{}]'.format(','.join('{}'.format(i) for i in (index + (':',) * dims_left)))

    return "<span class='{}' title='{}'>{}</span>".format(
        ' '.join(classes), hover_text, s,
    )

basic_css = """
    .numpy-array {
        display: inline-block;
    }
    .numpy-array .numpy-array-slice {
        border: 1px solid #cfcfcf;
        border-radius: 4px;
        margin: 1px;
        padding: 1px;
        display: flex;
        flex: 1;
        text-align: right;
        position: relative;
    }
    .numpy-array .numpy-array-slice:hover {
        border: 1px solid #66BB6A;
    }
    .numpy-array .numpy-array-slice.numpy-array-vertical {
        flex-direction: column;
    }
    .numpy-array .numpy-array-slice.numpy-array-horizontal {
        flex-direction: row;
    }
    .numpy-array .numpy-array-ndim-m0 {
        padding: 0 0.5ex;
    }
    
    /* Hide the comma and square bracket characters which exist to help with copy paste */
    .numpy-array .numpy-array-bracket {
        font-size: 0;
        position: absolute;
    }
    .numpy-array span .numpy-array-comma {
        font-size: 0;
        height: 0;
    }
"""

show_brackets_css = """
    .numpy-array.show-brackets .numpy-array-slice {
        border-radius: 0;
    }
    .numpy-array.show-brackets .numpy-array-bracket {
        border: 1px solid black; 
        border-radius: 0;  /* looks better without... */
    }
    .numpy-array.show-brackets .numpy-array-horizontal > .numpy-array-bracket-open {
        top: -1px;
        bottom: -1px;
        left: -1px;
        width: 10px;
        border-right: none;
        border-top-right-radius: 0;
        border-bottom-right-radius: 0;
    }
    .numpy-array.show-brackets .numpy-array-horizontal > .numpy-array-bracket-close {
        top: -1px;
        bottom: -1px;
        right: -1px;
        width: 10px;
        border-left: none;
        border-top-left-radius: 0;
        border-bottom-left-radius: 0;
    }
    .numpy-array.show-brackets .numpy-array-vertical > .numpy-array-bracket-open {
        top: -1px;
        right: -1px;
        left: -1px;
        height: 10px;
        border-bottom: none;
        border-bottom-right-radius: 0;
        border-bottom-left-radius: 0;
    }
    .numpy-array.show-brackets .numpy-array-vertical > .numpy-array-bracket-close {
        left: -1px;
        bottom: -1px;
        right: -1px;
        height: 10px;
        border-top: none;
        border-top-right-radius: 0;
        border-top-left-radius: 0;
    }
"""

def make_pretty(self, show_brackets=False, is_horz=lambda arr, ax: ax == arr.ndim - 1):

    classes = ['numpy-array']
    css = basic_css
    if show_brackets:
        classes += ['show-brackets']
        css += show_brackets_css
    return IPython.core.display.HTML(
        """<style>{}</style><div class='{}'>{}</div>""".format(
            css,
            ' '.join(classes),
            _html_repr_helper(self, (), is_horz))
    )


# In[ ]:


tensor_3d = torch.rand(10, 1, 3)

print(tabulate(tensor_3d, showindex='always', tablefmt='pretty'))


# In[ ]:


# Example 2 - working
make_pretty(tensor_3d) # make sure this call goes after print()


# ## Section 4 - Plot multi-dimensional Tensor with Plotly
# 
# Now I show you how you can plot 3 dimension data of temperature, humidity & pressure as 3d mesh. Notice that we plot the data in order of z, y, x. So that temperature (x axis from tensor) represents z axis on our plot, which is more practical. Mostly the temperature is the most significant data.

# In[ ]:


# Example 1 - working
import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

# Temperature: -40 to +85gradC
# Humidity: 0-100%
# Pressure: 300-1100 hPa
    
temperature = [-10, 0, 5, 15, 30]
humidity = [20, 40, 40, 60, 70]
pressure = [400, 500, 600, 700, 800]

weather_tensor = torch.tensor([temperature, pressure, humidity], device=device)

# temperature on x axis to represent height in graph
z, y, x = weather_tensor

fig = go.Figure(data=[
    go.Mesh3d(
        x=x,
        y=y,
        z=z,
        colorbar_title='Temperature',
        colorscale=[[0, 'gold'],
                    [0.5, 'mediumturquoise'],
                    [1, 'magenta']],
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=z,
        showscale=True
    )
])

fig.update_layout(
    title="Weather Mesh Example",
    font=dict(
        family="Courier New, monospace",
        size=16,
        color="#7f7f7f"
    )
)

camera = dict(
    eye=dict(x=-2, y=-2, z=0.1)
)

fig.update_layout(scene_camera=camera)

fig.show()


# ## Section 5 - Tensor Operation API limits & solutions
# 
# For machine learning you will need to scale data from 0.0 to 1.0. This might come handy also when you need to plot and index normalized data. This is not possible to do with built-in functions, so you can use something as MinMaxScaler
# 
# Lets see how we can normalize our temperature data from previous example (-10 to 30 degrees of Celsius)

# In[ ]:


# Example 1 - normalizing data
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

# first reshape (-1, 1) to change list dimension from 1D to 2D
temperature_normalized = min_max_scaler.fit_transform(weather_tensor[0].numpy().reshape(-1, 1))

# now reshape the list again to single values
temperature_normalized = temperature_normalized.reshape(-1)

print(tabulate([weather_tensor[0].numpy(), temperature_normalized], showindex='always', tablefmt='pretty'))


# You might want to filter tensor data, which will take only positive nonzero values and create a new tensor that will return the indexes of these values from original tensor (indices).
# 
# It works by defining a mask, which acts as a condition. So you can filter any data you want:

# In[ ]:


# Example 2 - filter or mask tensor data
tensor_x = torch.tensor([0.1, 0.5, -1.0, 0, 1.2, 0])

print(tensor_x)

mask = tensor_x >= 0 # This is the important step, where you define filtering condition

print(mask)

indices = torch.nonzero(mask)

print(tensor_x[indices]) # mapping index example

print(indices) # index from original tensor


# If you need to get particular index out of tensor, you can use indices matrix and gather() function:

# In[ ]:


# Example 3 - basic gather example with matrix
t = torch.tensor([[1,2],[3,4]])
torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))


# Split sequence into blocks according to its content. Imagine you have a sequence of L events stored in a tensor of shape L x W, where for each event its last feature is a timestamp which indicates when the event appears. Now you would like to split this sequence into blocks according to their timestamp and group by 1 second.
# 
#  tensor([
#     [n1, 0],
#     [n2, 0.25],
#     [n3, 0.75],
#     [n4, 1],
#     [n5, 1.5],
#     [n6, 2.1],
#     ...
# ])
# 
# to
# 
# [
#     tensor([
#         [n1, 0],
#         [n2, 0.25],
#         [n3, 0.75],
#     ]),
#     tensor([
#         [n4, 1],
#         [n5, 1.5]
#     ]),
#     tensor([
#         [n6, 2.1]
#     ]),
#     ...
# ]

# In[ ]:


def get_splits(x):
    bins = np.arange(0, np.ceil(x[:,1].max())+1)
    d = torch.from_numpy(np.digitize(x.numpy()[:, 1], bins))
    _, counts = torch.unique(d, return_counts=True)
    return torch.split(x, counts.tolist())

# create tensor
N = 50
c0 = torch.randn(N)
c1 = torch.arange(N) + 0.1 * torch.randn(N)
x = torch.stack((c0, c1), dim=1)

print(*get_splits(x), sep='\n')

print(tabulate(get_splits(x), showindex='always', tablefmt='pretty'))


# ## Conclusion
# 
# Here I wanted to show you how to read & present tensor data, which I find the most important step when learning anything. It comes also handy when you need to quickly debug some data. Also I show you how to overcome some limitations of PyTorch Tensor class with Numpy and SciKit (SKLearn). One would think that this features will be built-in, but as you dig deep into Tensor documentation, you find out that you spend extra hours just to find out how to do basic stuff before you can move on to more serious machine learning.

# ## Reference Links
# 
# * Official documentation for `torch.Tensor`: https://pytorch.org/docs/stable/tensors.html
# * Plotly: https://plotly.com/python/
# * Python Tabulate package for pretty print: https://pypi.org/project/tabulate/

# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


import jovian


# In[ ]:


jovian.commit(project="01-tensor-operations", environment=None)


# In[ ]:




