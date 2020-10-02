#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage.color import rgb2gray
from skimage.util import montage as montage2d
from tqdm import tqdm_notebook
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})


# ### Unit-Testing Code
# Here we import doctest and create an autotest tool so we can perform simple tests on our point matching functions to ensure they work correctly before applying them to large datasets

# In[ ]:


import doctest
import copy
import functools
# tests help notebooks stay managable
def autotest(func):
    globs = copy.copy(globals())
    globs.update({func.__name__: func})
    doctest.run_docstring_examples(
        func, globs, verbose=True, name=func.__name__)
    return func


# In[ ]:


@autotest
def flatten_dict(nested_dict, # type: Dict[Any, Any]
                 name_mapper=None # type: Callable[Tuple[T, List[T]], T]
                ):
    """Flattens nested dict.
    
    Usually from a json 
    :params nested_dict: a nested hierarchical dictionary
    :params name_mapper: a tool for mapping indices to names see xy_mapper
    >>> flatten_dict({'a': 1, 'b': [1, 2]})
    {'a': 1, 'b_0': 1, 'b_1': 2}
    >>> flatten_dict({'a': 1, 'b': {'d': 0, 'e': 1}})
    {'a': 1, 'b_d': 0, 'b_e': 1}
    >>> xy_mapper = lambda arg, arg_space: 'xy'[arg] if len(arg_space)==2 else arg
    >>> flatten_dict({'a': 1, 'b': [1, 2]}, xy_mapper)
    {'a': 1, 'b_x': 1, 'b_y': 2}
    """
    flat_dict = {}
    if name_mapper is None:
        name_mapper = lambda arg, arg_space: arg
    
    def _flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                _flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                _flatten(a, name + str(name_mapper(i, x)) + '_')
                i += 1
        else:
            flat_dict[name[:-1]] = x

    _flatten(nested_dict)
    return flat_dict


# # Load Data

# In[ ]:


bubbles_df = pd.read_json('../input/aluminum-preprocessing/tracked_bubbles.json')
print(bubbles_df.shape, 'bubbles loaded')
bubbles_df.sample(3)


# ## Preview Frames

# In[ ]:


big_bubbles_df = bubbles_df.query('area>50').sort_values('time_ms')
# keep 10 frames with bubbles
many_bubble_frames = big_bubbles_df.    groupby('time_ms').    agg({'label': 'count'}).    reset_index().    query('label>10')['time_ms'].values.tolist()
many_bubbles_df = big_bubbles_df[big_bubbles_df['time_ms'].isin(many_bubble_frames)]
many_bubbles_df.head(3)


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
for frame, (ts, c_bubbles) in enumerate(many_bubbles_df.groupby('time_ms')):
    ax1.plot(c_bubbles['x'], c_bubbles['y'], 's', label='{:1.2f} ms'.format(ts/1000))
    if frame>10:
        break
ax1.legend()


# ## Two Good Frames

# In[ ]:


frame_1 = big_bubbles_df[big_bubbles_df['time_ms'].isin(many_bubble_frames[0:1])]
frame_2 = big_bubbles_df[big_bubbles_df['time_ms'].isin(many_bubble_frames[1:2])]
fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
ax1.plot(frame_1['x'], frame_1['y'], 's', label='Frame 1')
ax1.plot(frame_2['x'], frame_2['y'], 's', label='Frame 2')
print(frame_1.shape, '->', frame_2.shape)


# # Nearest Neighbor Tracking Routine
# 

# In[ ]:


from scipy.spatial import distance_matrix
from string import ascii_lowercase
from itertools import product
xyzs_mapper = lambda arg, arg_space: 'xyzs'[arg] if len(arg_space)==4 else arg
@autotest
def match_nearest_points(
    last_row, # type: Dict[str, List[float]]
    cur_row, # type: Dict[str, List[float]]
    max_dist = 1e-3,
    include_last_row = True,
    as_list = False,
    add_status = False,
    max_dim = 3,
    letter_basis = ascii_lowercase
):
    # type: (...) -> Dict[str, List[float]]
    """Matches points from previous to current row.
    
    Simple nearest neighbor matching code
    
    >>> a = {'x': [0, 0, 0]}
    >>> match_nearest_points({}, a, as_list=True)
    {'a': [0, 0, 0]}
    >>> match_nearest_points(a, a, as_list=True)
    {'x': [0, 0, 0]}
    >>> match_nearest_points(a, {'x': [1, 1, 1]}, as_list=True)
    {'x': [0, 0, 0], 'b': [1, 1, 1]}
    >>> b = {'y': [0, 1, 0], 'z': [0, 0, 0.5]}
    >>> match_nearest_points(a, b, max_dist=0.5, as_list=True)
    {'x': [0.0, 0.0, 0.5], 'b': [0.0, 1.0, 0.0]}
    >>> match_nearest_points(b, a, max_dist=0.5, as_list=True)
    {'y': [0.0, 1.0, 0.0], 'z': [0, 0, 0]}
    >>> c = {'w': [0, 0, 0.25], 'v': [0, 0, 0.25]}
    >>> match_nearest_points(b, c, max_dist=0.5, as_list=True)
    {'y': [0.0, 1.0, 0.0], 'c': [0.0, 0.0, 0.25], 'z': [0.0, 0.0, 0.25]}
    >>> match_nearest_points(b, c, max_dist=0.5, add_status=True)
    {'y': [0.0, 1.0, 0.0, -1], 'c': [0.0, 0.0, 0.25, 1], 'z': [0.0, 0.0, 0.25, 0]}
    >>> match_nearest_points({}, {'1': [2078, -2911, 95], '0': [1935, -3020, 94]}, add_status=True) 
    {'a': [2078, -2911, 95, 1], 'b': [1935, -3020, 94, 1]}
    >>> td1 = {'a': [2078, -2911, 95], 'b': [1935, -3020, 94]} 
    >>> td2 = {'1': [2078, -2911, 96], '0': [1935, -3020, 97]} 
    >>> match_nearest_points(td1, td2, add_status=True) 
    {'a': [2078, -2911, 95, -1], 'c': [2078, -2911, 96, 1], 'b': [1935, -3020, 94, -1], 'd': [1935, -3020, 97, 1]}
    """
    if add_status:
        """
        add status means a status point is added to the end of every point
        0 means it was found in the sequence before
        1 means it is a new point in the current row
        -1 means it is a left-over point from the last row
        ideally all points are 0s but 0
        when markers disappear we can interpolate between the -1 and the 1
        """
        as_list = True
    out_dict = {} # type: Dict[str, List[float]]
    # dicts aren't sorted
    last_keys = list(last_row.keys())
    cur_keys = list(cur_row.keys())
    # make Mx3 vectors for each
    if len(last_keys)>0:
        last_vec = np.stack([last_row[k] for k in last_keys], 0)
    else:
        last_vec = np.zeros((0, max_dim))
    if len(cur_keys)>0:
        cur_vec = np.stack([cur_row[k] for k in cur_keys], 0)
    else:
        cur_vec = np.zeros((0, max_dim))
    # only keep valid dimensions
    last_vec = last_vec[:, :max_dim]
    cur_vec = cur_vec[:, :max_dim]
    d_mat = distance_matrix(cur_vec, last_vec)
    if np.prod(d_mat.shape)>0:
        for k in range(d_mat.shape[0]): # along cur_vec
            for best_idx in np.argsort(d_mat[k, :]): # sort for the most promising match
                if d_mat[k, best_idx]<=max_dist: # if match is acceptable
                    if last_keys[best_idx] is not None: # if the point is available
                        out_val = cur_vec[k, :].tolist() if as_list else cur_vec[k, :]
                        if add_status:
                            out_val.append(0)
                        out_dict[last_keys[best_idx]] = out_val
                        last_keys[best_idx] = None # so it cannot be reused
                        cur_keys[k] = None # current point is done
                        break
    # add the last keys (that haven't been used)
    if include_last_row:
        for c_key, c_vec in zip(last_keys, last_vec):
            if c_key is not None:
                out_val = c_vec.tolist() if as_list else c_vec
                if add_status:
                    
                    out_val.append(-1)
                out_dict[c_key] = out_val
    # process the leftovers from the current row
    letter_idx = len(out_dict)
    for c_key, c_vec in zip(cur_keys, cur_vec):
        if c_key is not None:
            out_val = c_vec.tolist() if as_list else c_vec
            if add_status:
                out_val.append(1)
            out_dict[letter_basis[letter_idx]] = out_val
            letter_idx+=1
            
    return out_dict


# In[ ]:


def rows_to_dict(in_rows):
    return {'T{time_ms:2.0f}_L{label:d}'.format(**c_row): 
     [c_row['x'], c_row['y']] for _, c_row in in_rows.iterrows()}
rows_to_dict(frame_1)


# ## Track frame 1 $\rightarrow$ frame 2
# Choosing a simple example to make sure the tracking works well before we apply it to all of the frames in our series

# In[ ]:


match_nearest_points(rows_to_dict(frame_1), 
                     rows_to_dict(frame_2), 
                     max_dist=50, 
                     max_dim=2,
                     add_status=True)


# In[ ]:


longer_ascii = sorted([''.join(a) for a in product(*([ascii_lowercase]*2))])
def cum_match_points(in_df):
    last_point = {}
    out_points = []
    for ts, c_rows in in_df.sort_values('time_ms').groupby('time_ms'):
        last_point = match_nearest_points({k: v[:3] for k, v in last_point.items()}, 
                                          rows_to_dict(c_rows), 
                                          add_status=True,
                                          max_dim=2,
                                         max_dist=50,
                                        letter_basis=longer_ascii
                                         )
        out_points.append(dict(time=ts, **last_point))
    return out_points


# In[ ]:


match_points = cum_match_points(many_bubbles_df)


# In[ ]:


xys_mapper = lambda arg, arg_space: 'xys'[arg] if len(arg_space)==3 else arg
match_xy_df = pd.DataFrame([flatten_dict(c_row, name_mapper=xys_mapper) for c_row in match_points])

match_xy_df.sample(3)


# ## Show Tracked Points
# Here we show the tracked points until they are removed (`status=-1`) we can see in this plot the jumps where the points move together and the general trajectories

# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
for i in np.unique(['_'.join(x.split('_')[:-1]) for x in match_xy_df.columns]):
    if len(i)>0:
        c_rows = match_xy_df[match_xy_df['{}_{}'.format(i, 's')]>=0]
        if c_rows.shape[0]>20: # tracked for more than 20 frames
            ax1.plot(
                c_rows['{}_{}'.format(i, 'x')],
                c_rows['{}_{}'.format(i, 'y')],
                '.-',
                label='{}'.format(i)
            )
            ax2.plot(
                c_rows['time'],
                c_rows['{}_{}'.format(i, 'x')],
                '-',
                label='{}'.format(i)
            )
            ax2.set_title('x vs Time')
            ax3.plot(
                c_rows['time'],
                c_rows['{}_{}'.format(i, 'y')],
                '-',
                label='{}'.format(i)
            )
            ax3.set_title('y vs Time')


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
for i in np.unique(['_'.join(x.split('_')[:-1]) for x in match_xy_df.columns]):
    if len(i)>0:
        c_rows = match_xy_df#[match_xy_df['{}_{}'.format(i, 's')]>=0]
        if c_rows.shape[0]>20: # tracked for more than 20 frames
            ax1.plot(
                c_rows['{}_{}'.format(i, 'x')],
                c_rows['{}_{}'.format(i, 'y')],
                '.-',
                label='{}'.format(i)
            )
            ax2.plot(
                c_rows['time'].values,
                c_rows['{}_{}'.format(i, 'x')].diff().values,
                '.',
                label='{}'.format(i)
            )
            ax2.set_title('$\Delta x$ vs Time')
            ax3.plot(
                c_rows['time'].values,
                c_rows['{}_{}'.format(i, 'y')].diff().values,
                '-',
                label='{}'.format(i)
            )
            ax3.set_title('$\Delta Y$ vs Time')


# In[ ]:


vec_xy_df = match_xy_df.copy()
for i in np.unique(['_'.join(x.split('_')[:-1]) for x in match_xy_df.columns]):
    if len(i)>0:
        vec_xy_df['{}_{}'.format(i, 'dx')] = vec_xy_df['{}_{}'.format(i, 'x')].diff()
        vec_xy_df['{}_{}'.format(i, 'dy')] = vec_xy_df['{}_{}'.format(i, 'y')].diff()


# In[ ]:



vec_xy_df.head(10)


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
for (_, c_row) in vec_xy_df.iterrows():
    dx_list = []
    dy_list = []
    for i in np.unique(['_'.join(x.split('_')[:-1]) for x in match_xy_df.columns]):
        if len(i)>0:
            dx_list += [c_row['{}_{}'.format(i, 'dx')]]
            dy_list += [c_row['{}_{}'.format(i, 'dy')]]
    bins = np.linspace(-10, 10, 50)
    ax1.hist(dx_list, bins, label='{time:2.0f}'.format(**c_row), alpha=0.5)
    
    ax2.hist(dx_list, bins, label='{time:2.0f}'.format(**c_row), alpha=0.5)
ax1.set_title('$\Delta x$ values')
ax2.set_title('$\Delta y$ values')
ax1.set_yscale('log')


# In[ ]:




