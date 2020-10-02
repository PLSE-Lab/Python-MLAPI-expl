#!/usr/bin/env python
# coding: utf-8

# # Google PAIR Facets
# #### https://github.com/PAIR-code/facets
# 
# This notebooks uses `Facets` to visualize the 33,321 images in dataset. Basics steps to do this:
#     1. Create dataframe with all image paths and any interesting metadata
#     2. Feed this dataframe to the `Atlasmaker` tool to create a montage of all the images
#     3. Use example Jupyter Notebook snippet to display HTML of visualization (https://colab.research.google.com/github/PAIR-code/facets/blob/master/colab_facets.ipynb)
#     
# Ideas for additional faceting visualizations:
#     - Visuals groups of incorrectly labelled images in validation. Could scatter by distance from threshold.
#     - Draw bounding boxes to verify generalization of segmentation model
#     - Use bounding box data to add fluke size and then use this to scatter images across an axis
#     - Add B&W / RGB column
#     - Add image ratio, image sizes to scatter on or group by
#     - Add corner cases column with labels like "heavily occluded", "image with text", etc.
#     - If doing metric learning reduce dimensionality with PCA / tSNE and plot the images in that space
#     
# It takes a minute to load completely (~100MB), but you can view this full screen here: https://davidwagnerkc.github.io/
# 
# Glad I finally got to try Facets out. It needs a pip installable package with a one liner to get from DataFrame to notebook output. 

# In[ ]:


from IPython.core.display import display, HTML
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw


# In[ ]:


# Bounding boxes from this kernel (@suicaokhoailang ran Martin Piotte's model on the current competition dataset)
# https://www.kaggle.com/suicaokhoailang/generating-whale-bounding-boxes
bb_df = pd.read_csv('../input/boundingbox/bounding_boxes.csv')


# In[ ]:


DATA_DIR = Path('/kaggle/input/humpback-whale-identification/')
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR =  DATA_DIR / 'test'

train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df = pd.read_csv(DATA_DIR / 'sample_submission.csv')


# In[ ]:


w, h = (bb_df.x1 - bb_df.x0), (bb_df.y1 - bb_df.y0)
bb_df['crop_size'] = w * h
bb_df['crop_ratio'] = w / h


# In[ ]:


bb_df.head()


# In[ ]:


train_df['freq'] = train_df.groupby('Id')['Id'].transform('count')
train_df['set'] = 'train'
train_df = train_df.sort_values('freq', ascending=False)


# In[ ]:


test_df['Id'] = 'unknown'
test_df['freq'] = 1
test_df['set'] = 'test'


# In[ ]:


df = pd.concat([train_df, test_df]).reset_index(drop=True)


# In[ ]:


df = pd.merge(df, bb_df, on='Image')


# In[ ]:


# Add image ratio data
def ratio(row):
    im_path = TRAIN_DIR / row.Image if 'train' in row.set else TEST_DIR / row.Image
    im = Image.open(im_path)
    return im.width / im.height

def total_size(row):
    im_path = TRAIN_DIR / row.Image if 'train' in row.set else TEST_DIR / row.Image
    im = Image.open(im_path)
    return im.width * im.height


# In[ ]:


def draw_bb(row):
    im_path = TRAIN_DIR / row.Image if 'train' in row.set else TEST_DIR / row.Image
    im = Image.open(im_path)
    bb = row.x0, row.y0, row.x1, row.y1
    draw = ImageDraw.Draw(im) 
    draw.rectangle(bb, outline=255)
    im.save(Path('/kaggle/working/draw_crops/') / row.Image)
    return True


# In[ ]:


p = Pool()


# In[ ]:


get_ipython().run_cell_magic('time', '', "df['ratio'] = p.map(ratio, [x[1] for x in list(df.iterrows())]) #df.apply(ratio, axis=1)\ndf['total_size'] = p.map(ratio, [x[1] for x in list(df.iterrows())]) #df.apply(total_size, axis=1)")


# In[ ]:


df['crop_perc'] = df.crop_size / df.total_size


# In[ ]:


df[::3000]


# In[ ]:


# !mkdir draw_crops


# In[ ]:


# %%time
# p.map(draw_bb, [x[1] for x in list(df.iterrows())])


# In[ ]:


df = df.drop(['x0', 'x1', 'y0', 'y1'], axis=1)


# In[ ]:


df['x_rand'] = np.random.random(len(df))
df['y_rand'] = np.random.random(len(df))


# In[ ]:


get_ipython().system('git clone https://github.com/PAIR-code/facets.git')


# In[ ]:


# If anybody is interested in building Facets themselves this might be useful. Turns out I didn't need to build Atlasmaker since it is just three Python modules.

# !pip install -r facets/facets_atlasmaker/requirements.txt
# !apt-get install -y pkg-config zip g++ zlib1g-dev unzip python
# !curl -LOk https://github.com/bazelbuild/bazel/releases/download/0.21.0/bazel-0.21.0-installer-linux-x86_64.sh
# !chmod +x bazel-0.21.0-installer-linux-x86_64.sh
# !bash bazel-0.21.0-installer-linux-x86_64.sh

# cd /kaggle/working/facets/facets_atlasmaker/
# %%time
# !bazel build :atlasmaker

# cd /kaggle/working/facets/bazel-bin/facets_atlasmaker/


# In[ ]:


cd /kaggle/working/facets/facets_atlasmaker/


# In[ ]:


# Does anybody use Python 2 anymore?
get_ipython().system("sed -i 's/from urlparse import urlparse/from urllib.parse import urlparse/g' atlasmaker_io.py")
# Let's pretend tensorflow isn't available to avoid another Python 2 problem 
get_ipython().system("sed -i 's/import tensorflow as tf/import tensorflop/g' atlasmaker_io.py")


# In[ ]:


#df.apply(lambda x: str(Path('/kaggle/working/draw_crops/') / x.Image), axis=1).to_csv('absolute_paths.csv', index=False)
df.apply(lambda x: str(TRAIN_DIR / x.Image) if 'train' in x.set else str(TEST_DIR / x.Image), axis=1).to_csv('absolute_paths.csv', index=False)


# In[ ]:


df.ratio.mean()


# In[ ]:


get_ipython().run_cell_magic('time', '', '!python atlasmaker.py --sourcelist=absolute_paths.csv --image_width=58 --image_height=29 --output_dir=/kaggle/working/')


# In[ ]:


#Image.open('/kaggle/working/spriteatlas.png').convert('L').save('/kaggle/working/spriteatlas.png', optimize=True)


# In[ ]:


cd /kaggle/working/


# # View in notebook

# In[ ]:


sprite_width, sprite_height = 58, 29
atlas_path = 'spriteatlas.png'
jsonstr = df.to_json(orient='records')
html = f"""<link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html">
           <facets-dive atlas-url="{atlas_path}" fit-grid-aspect-ratio-to-viewport="true" sprite-image-width="{sprite_width}" sprite-image-height="{sprite_height}" height="800" id="elem"></facets-dive>
           <script>document.querySelector("#elem").data = {jsonstr};</script>"""
display(HTML(html))


# # View full screen from Kaggle kernel

# In[ ]:


html = f"""<link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html">
           <facets-dive atlas-url="{atlas_path}" fit-grid-aspect-ratio-to-viewport="true" cross-origin="anonymous" sprite-image-width="{sprite_width}" sprite-image-height="{sprite_height}" id="elem"></facets-dive>
           <script>document.querySelector("#elem").data = {jsonstr};</script>"""


# In[ ]:


with open('facets_static.html', 'w') as out_file:
    out_file.write(html)


# In[ ]:


get_ipython().system('(jupyter notebook list | grep http | awk \'{printf $1}\'; printf "files/facets_static.html") | sed "s/http:\\/\\/localhost:8888/https:\\/\\/www\\.kaggleusercontent\\.com/"')


# # To host locally
# 
# 1. Download facets_static.html and spriteatlas.png and make a folder structure like this:
#         facets_server/
#                 facets_static.html
#                 spriteatlas.png
# 2. cd to facets_server/ and run this command `python -m http.server`
# 3. Access locally @ `localhost:8000`

# In[ ]:


get_ipython().system('rm -rf facets/')
get_ipython().system('rm -rf draw_crops/')
get_ipython().system('rm im_paths.csv')
get_ipython().system('rm mani')

