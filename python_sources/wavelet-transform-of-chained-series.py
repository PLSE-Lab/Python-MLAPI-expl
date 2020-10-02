#!/usr/bin/env python
# coding: utf-8

# This kernel is a fork of Markus [Missing Link](https://www.kaggle.com/friedchips/the-missing-link) that chaines series.
# 
# The continuous wavelet transform of chained series for each surface are shown bellow as scaleograms. These views allow to see both frequency related and time related features. Interresting patterns shows up at scales larger than 128 samples.

# Scaleo

# # Cnaining of series

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# First, let's get our raw data:

# In[ ]:


train_X = pd.read_csv('../input/X_train.csv')
columns = train_X.columns
train_X = train_X.iloc[:,3:].values.reshape(-1,128,10)
test_X  = pd.read_csv('../input/X_test.csv' ).iloc[:,3:].values.reshape(-1,128,10)
print('train_X shape:', train_X.shape, ', test_X shape:', test_X.shape)


# ...and y / group data:

# In[ ]:


df_train_y = pd.read_csv('../input/y_train.csv')

# build a dict to convert surface names into numbers
surface_names = df_train_y['surface'].unique()
num_surfaces = len(surface_names)
surface_to_numeric = dict(zip(surface_names, range(num_surfaces)))
print('Convert to numbers: ', surface_to_numeric)

# y and group data as numeric values:
train_y = df_train_y['surface'].replace(surface_to_numeric).values
train_group = df_train_y['group_id'].values


# Let's plot the 4 orientation channels of a random group in series:

# In[ ]:


fig, axes = plt.subplots(1,4)
fig.set_size_inches(20,3)

for i in range(4):
    axes[i].plot(train_X[train_group == 17][:,:,i].reshape(-1))
    axes[i].grid(True)


# Well, that certainly looks like a jigsaw puzzle to me. And that leads to an idea: the euclidean distance in the 4-dimensional "orientation space" between, for example, the right edge of one sample and the left edge of its true neighbor should be a minimum, *ideally even among all samples*, not only the samples in its group. Same for left/right. This should enable us to stitch the runs together again. All we have to do is link samples together which are *each other's* closest neighbors. Let's code:

# In[ ]:


def sq_dist(a,b):
    ''' the squared euclidean distance between two samples '''
    
    return np.sum((a-b)**2, axis=1)


def find_run_edges(data, edge):
    ''' examine links between samples. left/right run edges are those samples which do not have a link on that side. '''

    if edge == 'left':
        border1 = 0
        border2 = -1
    elif edge == 'right':
        border1 = -1
        border2 = 0
    else:
        return False
    
    edge_list = []
    linked_list = []
    
    for i in range(len(data)):
        dist_list = sq_dist(data[i, border1, :4], data[:, border2, :4]) # distances to rest of samples
        min_dist = np.min(dist_list)
        closest_i   = np.argmin(dist_list) # this is i's closest neighbor
        if closest_i == i: # this might happen and it's definitely wrong
            print('Sample', i, 'linked with itself. Next closest sample used instead.')
            closest_i = np.argsort(dist_list)[1]
        dist_list = sq_dist(data[closest_i, border2, :4], data[:, border1, :4]) # now find closest_i's closest neighbor
        rev_dist = np.min(dist_list)
        closest_rev = np.argmin(dist_list) # here it is
        if closest_rev == closest_i: # again a check
            print('Sample', i, '(back-)linked with itself. Next closest sample used instead.')
            closest_rev = np.argsort(dist_list)[1]
        if (i != closest_rev): # we found an edge
            edge_list.append(i)
        else:
            linked_list.append([i, closest_i, min_dist])
            
    return edge_list, linked_list


def find_runs(data, left_edges, right_edges):
    ''' go through the list of samples & link the closest neighbors into a single run '''
    
    data_runs = []

    for start_point in left_edges:
        i = start_point
        run_list = [i]
        while i not in right_edges:
            tmp = np.argmin(sq_dist(data[i, -1, :4], data[:, 0, :4]))
            if tmp == i: # self-linked sample
                tmp = np.argsort(sq_dist(data[i, -1, :4], data[:, 0, :4]))[1]
            i = tmp
            run_list.append(i)
        data_runs.append(np.array(run_list))
    
    return data_runs


# Let's go:

# In[ ]:


train_left_edges, train_left_linked  = find_run_edges(train_X, edge='left')
train_right_edges, train_right_linked = find_run_edges(train_X, edge='right')
print('Found', len(train_left_edges), 'left edges and', len(train_right_edges), 'right edges.')


# Well, that certainly looks promising. Found 76 runs, similar number than the number of groups. Build the runs:

# In[ ]:


train_runs = find_runs(train_X, train_left_edges, train_right_edges)


# Have we found all samples? Have we used any sample twice? The answer is yes, and no. Perfect.

# In[ ]:


flat_list = [series_id for run in train_runs for series_id in run]
print(len(flat_list), len(np.unique(flat_list)))


# Now for the real test. How many different surfaces are in each run? *Only 4 runs have more than one surface* (and if you look at them, you can easily split them by hand). This actually works!

# In[ ]:


print([ len(np.unique(train_y[run])) for run in train_runs ])


# Interesting. Some runs contain  2, 3 and even 4 groups. So several groups were cut from one run:

# In[ ]:


print([ len(np.unique(train_group[run])) for run in train_runs ])


# Let's plot all 10 channels for one run.  Beautiful.

# In[ ]:


fig, axes = plt.subplots(10,1, sharex=True)
fig.set_size_inches(20,15)
fig.subplots_adjust(hspace=0)

for i in range(10):
    axes[i].plot(train_X[train_runs[0]][:,:,i].reshape(-1))
    axes[i].grid(True)


# Let's add our new knowledge to train_y. Now you can use this info to train your models to even greater perfection. Enjoy!

# In[ ]:


df_train_y['run_id'] = 0
df_train_y['run_pos'] = 0

for run_id in range(len(train_runs)):
    for run_pos in range(len(train_runs[run_id])):
        series_id = train_runs[run_id][run_pos]
        df_train_y.at[ series_id, 'run_id'  ] = run_id
        df_train_y.at[ series_id, 'run_pos' ] = run_pos

df_train_y.to_csv('y_train_with_runs.csv', index=False)
df_train_y.tail()


# ...But wait. Might this also work with the test data?

# In[ ]:


test_left_edges, test_left_linked  = find_run_edges(test_X, edge='left')
test_right_edges, test_right_linked = find_run_edges(test_X, edge='right')
print('Found', len(test_left_edges), 'left edges and', len(test_right_edges), 'right edges.')


# Oh yeah!

# In[ ]:


test_runs = find_runs(test_X, test_left_edges, test_right_edges)


# Again no samples are used twice, but we have lost some.

# In[ ]:


flat_list = [series_id for run in test_runs for series_id in run]
print(len(flat_list), len(np.unique(flat_list)))


# [](http://) 3816 - 3790 = 26 series are not in any run and aren't edges. They must form a closed ring -> another run. find it:

# In[ ]:


lost_samples = np.array([ i for i in range(len(test_X)) if i not in np.concatenate(test_runs) ])
print(lost_samples)
print(len(lost_samples))


# In[ ]:


find_run_edges(test_X[lost_samples], edge='left')[1][0]


# In[ ]:


lost_run = np.array(lost_samples[find_runs(test_X[lost_samples], [0], [5])[0]])
test_runs.append(lost_run)


# Perfect. Now we also have test runs. A nice plot to prove it:

# In[ ]:


fig, axes = plt.subplots(10,1, sharex=True)
fig.set_size_inches(20,15)
fig.subplots_adjust(hspace=0)

for i in range(10):
    axes[i].plot(test_X[test_runs[1]][:,:,i].reshape(-1))
    axes[i].grid(True)


# I'll leave you with a caution and an exercise:
# First, there are certainly some errors in the test runs. Two runs with different surfaces might have been stitched into one (as happened 4 times with the train data) by chance.
# Second, might it even be possible to link *across train and test*? Well, see for yourself...

# In[ ]:


print(f"There is {len(test_runs)} runs of series containing one or several groups of surfaces")


# # Wavelet transforms

# In[ ]:


import scaleogram as scg
periods         = np.logspace(np.log10(1), np.log10(1024), 50)
wavelet_scales  = scg.periods2scales( periods )

def plot_cwt_run(id_run):
    id_series = train_runs[id_run]
    surfaces = df_train_y[df_train_y.series_id.isin(id_series)].surface.values
    print(f"surfaces in run={id_run}: ", surfaces)
    (fig1, axes1) = plt.subplots(6, 1, figsize=(10,4))
    (fig2, axes2) = plt.subplots(6, 1, figsize=(11,9))
    for i in range(4,10):
        title   = f'surface={surfaces[0]} movement sensors train_grp={id_run} '
        signal = train_X[id_series][:,:,i].ravel()[0:1024*4]
        signal -= signal.mean()
        axes1[i-4].plot(signal)
        axes1[i-4].set_xlim(0,1024*4)
        plt.xlim(0, len(signal))
        scg.cws(signal, scales=wavelet_scales, ax=axes2[i-4], yscale='log',
               title="", clim=(0, 0.2) if i < 7 else (0, 4), 
               xlim=(0, 1024*4))
    axes1[0].set_title(title)
    axes2[0].set_title(title)
    fig1.subplots_adjust(hspace=0)
    fig2.subplots_adjust(hspace=0)
    #fig2.tight_layout()
        


# # Surfaces by run group
# 
# note: a run group is the series_id which are grouped together
# 

# In[ ]:


for (irun, run) in enumerate(train_runs):
    surface = df_train_y[df_train_y.series_id==run[0]].surface.values[0]
    print(f"run={irun} surface={surface}")


# # Concrete

# In[ ]:


plot_cwt_run(2)


# # Fine concrete
# 

# In[ ]:


plot_cwt_run(10)


# # Hard tiles

# In[ ]:


plot_cwt_run(27)


# # Hard tiles large space

# In[ ]:


plot_cwt_run(51)


# # Soft PVC

# In[ ]:


plot_cwt_run(23)


# # Soft tiles

# In[ ]:


plot_cwt_run(0)


# # Tiled

# In[ ]:


plot_cwt_run(26)


# # Wood

# In[ ]:


plot_cwt_run(3)


# In[ ]:





# In[ ]:




