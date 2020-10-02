#!/usr/bin/env python
# coding: utf-8

# # Overvew
# The notebook shows how to load the data and visualize the results easily

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 14
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})
top_row_dict = lambda in_df: list(in_df.head(1).T.to_dict().values())[0]
base_dir = os.path.join('..', 'input', 'quickdraw_simplified')


# In[ ]:


obj_files = glob(os.path.join(base_dir, '*.ndjson'))
print(len(obj_files), 'categories found!', 'first is:', obj_files[0])


# In[ ]:


c_json = pd.read_json(obj_files[0], lines = True, chunksize = 1)
f_row = next(c_json)
f_dict = top_row_dict(f_row)
f_row


# In[ ]:


def draw_dict(in_dict, in_ax, legend = True):
    for i, (x_coord, y_coord) in enumerate(in_dict['drawing']):
        in_ax.plot(x_coord, y_coord, '.-', label = 'stroke {}'.format(i))
    if legend:
        in_ax.legend()
    in_ax.set_title('A {word} from {countrycode}\n Guessed Correctly: {recognized}'.format(**in_dict))
    in_ax.axis('off')
fig, ax1 = plt.subplots(1, 1, figsize = (8,8))
draw_dict(f_dict, ax1)


# # All Classes
# Here we make a nice simple round-robin generator to give us examples from each category so we can start to possibly build models

# In[ ]:


def multi_ndjson_gen(in_paths, shuffle = True):
    json_readers = [pd.read_json(c_path, lines = True, chunksize = 1) for c_path in in_paths]
    while True:
        if shuffle:
            np.random.shuffle(json_readers)
        for c_reader in json_readers:
            yield top_row_dict(next(c_reader))
nd_gen = multi_ndjson_gen(obj_files)


# In[ ]:


fig, m_axs = plt.subplots(3, 3, figsize = (20,20))
for f_dict, c_ax in zip(nd_gen, m_axs.flatten()):
    draw_dict(f_dict, c_ax)


# # Generating Images
# For some tasks (offline classification), the strokes will need to be rasterized into an image for classification. We show a few basic examples here
# 

# In[ ]:


from PIL import Image
def strokes_to_mat(in_strokes, out_dims = (256, 256), rescale_dims = None, resample_points = 500):
    base_img = np.zeros(out_dims, dtype = np.float32)
    # TODO: 1d interpolation is a bad strategy here, should be improved to something that makes more sense
    rs_points = lambda x_pts, out_dim: np.interp(np.linspace(0, 1, resample_points),
                                        np.linspace(0, 1, len(x_pts)), 
                                        np.array(x_pts)/256.0*out_dim).astype(int)
    for (x_coord, y_coord) in in_strokes:
        rx_coord = rs_points(x_coord, out_dims[1])
        ry_coord = out_dims[0]-1-rs_points(y_coord, out_dims[0])
        base_img[ry_coord, rx_coord] = 1.0
    if rescale_dims is not None:
        base_img = np.array(Image.fromarray(base_img).resize(rescale_dims, 
                                                             resample = Image.BICUBIC))
    return base_img

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12,8))
draw_dict(f_dict, ax1)
ax2.imshow(strokes_to_mat(f_dict['drawing']))
ax2.set_title('High Resolution')
ax3.matshow(strokes_to_mat(f_dict['drawing'], rescale_dims = (28, 28)), vmin = 0, cmap = 'bone')
ax3.set_title('Low Resolution')


# In[ ]:


fig, m_axs = plt.subplots(9, 9, figsize = (20,20))
for f_dict, c_ax in zip(nd_gen, m_axs.flatten()):
    c_ax.imshow(strokes_to_mat(f_dict['drawing'], rescale_dims = (28, 28)))
    c_ax.set_title(f_dict['word'])
    c_ax.axis('off')
fig.savefig('tiles.jpg', figdpi = 300)


# # Nice Figures
# Here we make a nice overview figure for the dataset description

# In[ ]:


fig, m_axs = plt.subplots(9, 9, figsize = (20,20))
for f_dict, c_ax in zip(nd_gen, m_axs.flatten()):
    draw_dict(f_dict, c_ax, legend = False)
    c_ax.set_title('')
fig.savefig('overview.jpg', figdpi = 300)


# # Standardization and Export
# NDJSON requires a fair amount of preprocessing to feed into models, let's try to preprocess it a bit here so we can build models later. Here we just pack all the strokes together into a single fixed sized array and add stroke id to keep them seperate. We can also include a low resolution rendering and the category / country for reference.
# 
# ## Statistics
# We calculate some statistics first to determine the appropriate size and dimensions for the data. If the number of strokes is <256 we can use an 8-bit unsigned integer for the index, and the total length is the zero-padded size of the output

# In[ ]:


n_strokes = []
stroke_length = []
total_length = []
for f_dict, _ in zip(nd_gen, range(512)):
    n_strokes += [len(f_dict['drawing'])]
    stroke_length += [len(x) for x,y in f_dict['drawing']]
    total_length += [sum([len(x) for x,y in f_dict['drawing']])]


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 4))
ax1.hist(n_strokes, np.arange(20))
ax1.set_title('Number of Strokes')
ax2.hist(stroke_length, np.arange(64))
ax2.set_title('Stroke Length')
ax3.hist(total_length, np.arange(200))
ax3.set_title('Total Length')
print('Max Strokes', np.max(n_strokes))
print('Max Total Length', np.max(total_length))
print('Total Length (99.59 percentile)', np.percentile(total_length, 99.5))


# In[ ]:


def drawing_to_array(in_drawing, max_length = 100):
    out_arr = np.zeros((max_length, 3), dtype = np.uint8) # x, y, indicator if it is a new stroke
    c_idx = 0
    for seg_label, (x_coord, y_coord) in enumerate(in_drawing):
        last_idx = min(c_idx + len(x_coord), max_length)
        seq_len = last_idx - c_idx
        out_arr[c_idx:last_idx, 0] = x_coord[:seq_len]
        out_arr[c_idx:last_idx, 1] = y_coord[:seq_len]
        out_arr[c_idx, 2] = 1 # indicate a new stroke
        c_idx = last_idx
        if last_idx>=max_length:
            break
    out_arr[:last_idx, 2] += 1
    return out_arr


# In[ ]:


test_arr = drawing_to_array(f_dict['drawing'])
test_arr = test_arr[test_arr[:,2]>0, :] # only keep valid points
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 5))
draw_dict(f_dict, ax1, legend = False)
lab_idx = np.cumsum(test_arr[:,2]-1)
for i in np.unique(lab_idx):
    ax2.plot(test_arr[lab_idx==i,0], 
                test_arr[lab_idx==i,1], '.-')
ax2.axis('off')
ax2.set_title('Single Array Reconstruction');


# In[ ]:


get_ipython().system('wc ../input/*.ndjson')


# In[ ]:


from tqdm import tqdm_notebook
out_blocks = []
for c_path in tqdm_notebook(obj_files, desc = 'File Progress'):
    for c_block in pd.read_json(c_path, lines = True, chunksize = 2100): # only take 1000
        # export as NHWC
        c_block['thumbnail'] = c_block['drawing'].map(lambda x: np.expand_dims(strokes_to_mat(x, rescale_dims = (28, 28)), -1))
        c_block['strokes'] = c_block['drawing'].map(lambda x: drawing_to_array(x))
        c_block.drop('drawing', inplace = True, axis = 1)
        out_blocks += [c_block]
        break


# In[ ]:


# we don't want the blocks in order otherwise batches (KerasHDF5 loader) will be very boring
np.random.shuffle(out_blocks)
big_df = pd.concat(out_blocks, ignore_index=True) #index seems to c
del out_blocks
print(big_df.shape[0], 'rows')
big_df.sample(2)


# # Export to HDF5
# While TFRecords are great for somethings, HDF5 is a lot easier for every tool but tensorflow. In addition we can open HDF5 in R or ImageJ in case there is some analysis that makes more sense in those tools

# In[ ]:


import h5py
from tqdm import tqdm
def write_df_as_hdf(out_path,
                    out_df,
                    compression='gzip'):
    with h5py.File(out_path, 'w') as h:
        for k, arr_dict in tqdm(out_df.to_dict().items()):
            try:
                s_data = np.stack(arr_dict.values(), 0)
                try:
                    h.create_dataset(k, data=s_data, compression=
                    compression)
                except TypeError as e:
                    try:
                        h.create_dataset(k, data=s_data.astype(np.string_),
                                         compression=compression)
                    except TypeError as e2:
                        print('%s could not be added to hdf5, %s' % (
                            k, repr(e), repr(e2)))
            except ValueError as e:
                print('%s could not be created, %s' % (k, repr(e)))
                all_shape = [np.shape(x) for x in arr_dict.values()]
                warn('Input shapes: {}'.format(all_shape))


# In[ ]:


from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(big_df, random_state = 2018, test_size = 0.20, stratify = big_df['word'])
del big_df
train_df, valid_df = train_test_split(train_df, random_state = 2018, test_size = 0.33, stratify = train_df['word'])
print('Training', train_df.shape[0], 
      'Valid', valid_df.shape[0], 
      'Test', test_df.shape[0])


# In[ ]:


write_df_as_hdf('quickdraw_train.h5', train_df)
write_df_as_hdf('quickdraw_valid.h5', valid_df)
write_df_as_hdf('quickdraw_test.h5', test_df)


# In[ ]:


# show what is inside
with h5py.File('quickdraw_train.h5', 'r') as h5_data:
    for c_key in h5_data.keys():
        print(c_key, h5_data[c_key].shape, h5_data[c_key].dtype)


# In[ ]:


get_ipython().system('ls -lh *.h5')


# In[ ]:




