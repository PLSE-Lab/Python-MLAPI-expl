#!/usr/bin/env python
# coding: utf-8

# # Data visualization with *google/PAIR-code/Facets*
# 
# 
# In this notebooks I try a tool from PAIR-code called [facets](https://github.com/pair-code/facets) to explore this dataset of favicons. There is a perfect presentation of this data exploration tool [here](https://pair-code.github.io/facets/). In brief, facets looks like:
# 
# Facets - Overview | Facets - Dive
# --- | ---
# ![facets_overview](https://raw.githubusercontent.com/PAIR-code/facets/master/img/overview-census.png) | ![facets_dive](https://raw.githubusercontent.com/PAIR-code/facets/master/img/dive-census.png)
# 
# **Important**: 
# - **Facets work only in Chrome browser. Issues related with that: [link](https://github.com/PAIR-code/facets/issues/26), [link](https://github.com/PAIR-code/facets/issues/9).** 
# - **Facets is `html/javascript` extension, therefore data processing and visualization happens in the browser and freeze it.**
# 
# 
# Facets contains two visualization modules:
# - [Facets Overview](https://github.com/PAIR-code/facets#facets-overview) to explore metadata
# - [Facets Dive](https://github.com/PAIR-code/facets#facets-dive) to dive into data (for example images)
# 
# ## Notebook content
# 
# - [Setup facets](#Setup-facets)
# - [Data access](#Data-access)
# - [Data exploration](#Let's-explore-data-with-facets)
# 
# ### Several screenshots with Favicons dataset:
# 
# Facets - Overview 1 | Facets - Overview 2
# --- | ---
# ![facets_overview1](https://gist.githubusercontent.com/vfdev-5/4687223bdef5912527587d5023a2c9d7/raw/d032dd6d2abbf89ce8576a0d469c54461c556447/facets_overview1.png) | ![facets_overview2](https://gist.githubusercontent.com/vfdev-5/4687223bdef5912527587d5023a2c9d7/raw/d032dd6d2abbf89ce8576a0d469c54461c556447/facets_overview2.png)
# Facets - Dive 1 | Facets - Dive 2
# --- | ---
# ![facets_dive1](https://gist.githubusercontent.com/vfdev-5/4687223bdef5912527587d5023a2c9d7/raw/d032dd6d2abbf89ce8576a0d469c54461c556447/facets_dive1.png) | ![facets_dive2](https://gist.githubusercontent.com/vfdev-5/4687223bdef5912527587d5023a2c9d7/raw/d032dd6d2abbf89ce8576a0d469c54461c556447/facets_dive2.png)
# 

# In[ ]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# ## Setup facets
# 
# Following this [guide](https://github.com/PAIR-code/facets#enabling-usage-in-jupyter-notebooks) on how to setup *facets*:
# 
# - clone the code
# - setup as nbextension
# - update `sys.path`

# But as it is said in the guide :
# > You do not need to run any follow-up jupyter nbextension enable command for this extension.

# In[ ]:


import os
import sys 

facets_path = os.path.dirname('.')
facets_path = os.path.abspath(os.path.join(facets_path, 'facets', 'facets_overview', 'python'))
                              
if not facets_path in sys.path:
    sys.path.append(facets_path)


# ## Data access
# 
# Here I copy useful methods to access metadata and images from the dataset. Code taken from dataset author's [welcome kernel](https://www.kaggle.com/colinmorris/favicon-helper-functions)
# 
# 

# In[ ]:


import PIL.Image
import pandas as pd
import numpy as np
import itertools
import math
import zipfile
from matplotlib import pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## Helpers for loading/viewing image data

# Background color for figures/subplots when showing favicons. Use a faint grey
# instead of the default white to make it clear which sections are transparent.
BG = '.95'
def show(df, scale=1, titles=None):
    """Show the favicons in the given dataframe, arranged in a grid.
    scale is a multiplier on the size of the drawn icons.
    """
    n = len(df)
    cols = int(min(n, max(4, 8//scale)))
    rows = math.ceil(n / cols)
    row_height = 1 * scale
    col_width = 1 * scale
    fs = (cols*col_width, rows*row_height)
    fig, axes = plt.subplots(rows, cols, figsize=fs, facecolor=BG)
    if rows == cols == 1:
        axes = np.array([axes])
    for i, (row, ax) in enumerate(
        itertools.zip_longest(df.itertuples(index=False), axes.flatten())
    ):
        if row is None:
            ax.axis('off')
        else:
            try:
                img = load_favicon(row.fname, row.split_index)
                _show_img(img, ax)
                if titles is not None:
                    ax.set_title(titles[i])
            except CorruptFaviconException:
                ax.axis('off')
                
def _show_img(img, ax=None):
    if ax is None:
        _fig, ax = plt.subplots(facecolor=BG)
    ax.tick_params(which='both', 
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False, labelleft=False, labelright=False,
                  )
    ax.grid(False, which='both')
    plt.setp(list(ax.spines.values()), color='0.8', linewidth=1, linestyle='-')
    ax.set_facecolor(BG)
    cmap = None
    if img.mode in ('L', 'LA'):
        cmap = 'gray'
    ax.imshow(img, cmap=cmap, aspect='equal', interpolation='none')
                
class CorruptFaviconException(Exception): pass

_ZIP_LOOKUP = {}
def load_favicon(fname, split_ix):
    if split_ix not in _ZIP_LOOKUP:
        zip_fname = '../input/full-{}.z'.format(split_ix)
        _ZIP_LOOKUP[split_ix] = zipfile.ZipFile(zip_fname)
    archive = _ZIP_LOOKUP[split_ix]
    fp = archive.open(fname)
    try:
        fav = PIL.Image.open(fp)
    except (ValueError, OSError):
        raise CorruptFaviconException
    if fav.format == 'ICO' and len(fav.ico.entry) > 1:
        pil_ico_hack(fav)
    return fav

def pil_ico_hack(img):
    """When loading an ICO file containing multiple images, PIL chooses the
    largest. We want whichever one is listed first."""
    ico = img.ico
    ico.entry.sort(key = lambda d: d['offset'])
    first = ico.frame(0)
    first.load()
    img.im = first.im
    img.mode = first.mode
    img.size = first.size


# In[ ]:


def load_metadata_df():
    """Return a dataframe with a row of metadata for each favicon in the dataset."""
    csvpath = '../input/favicon_metadata.csv'
    return pd.read_csv(csvpath)


# In[ ]:


METADATA_CSV = load_metadata_df()
METADATA_CSV.head()


# In[ ]:


example_icons = METADATA_CSV.sample(6, random_state=123)
show(example_icons)

# That cat is adorable. Let's see a bigger version.
show(example_icons.iloc[5:6], scale=2)


# ## Let's explore data with facets
# 
# ### Metadata overview
# 
# Whole metadata table is huge, contains 360610 entries. I believe that facets can proceed all the data, but not having patience, I reduce metadata table size in order to avoid browser freezing.
# 
# Facets overview provides feature statistics by type. In our case, feature types are `int`, `string`
# 
# 

# In[ ]:


import numpy as np
np.random.seed(2017)

indices = METADATA_CSV.index
indices = np.random.choice(indices, size=100000)
_metadata_csv = METADATA_CSV.loc[indices, :]


# In[ ]:


sys.path.append('/opt/facets/facets_overview/python')


# In[ ]:


from generic_feature_statistics_generator import GenericFeatureStatisticsGenerator
proto = GenericFeatureStatisticsGenerator().ProtoFromDataFrames([{'name': 'Metadata', 'table': _metadata_csv}])


# In[ ]:


from IPython.core.display import display, HTML
import base64
protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")
HTML_TEMPLATE = """<link rel="import" href="/nbextensions/facets-jupyter.html" >
        <facets-overview id="elem"></facets-overview>
        <script>
          document.querySelector("#elem").protoInput = "{protostr}";
        </script>"""
html = HTML_TEMPLATE.format(protostr=protostr)
display(HTML(html))


# ### Data dive
# 
# At first we need to create an atlas image containing all our data. Details can be found [here](https://github.com/PAIR-code/facets/tree/master/facets_dive#providing-sprites-for-dive-to-render).
# Idea is to create a single image of data image tiles (sprits):
# ```
# +---------+---------+---------+- - - - -+---------+
# |         |         |         |         |         |
# |    0    |    1    |    2    |   ...   |    99   |
# |         |         |         |         |         |
# +---------+---------+---------+- - - - -+---------+
# |         |         |         |         |         |
# |   100   |   101   |   102   |   ...   |   199   |
# |         |         |         |         |         |
# +---------+---------+---------+- - - - -+---------+
# |         |         |         |         |         |
# |   200   |   201   |   202   |   ...   |   299   |
# |         |         |         |         |         |
# +---------+---------+---------+- - - - -+---------+
# |         |         |         |         |         |
#      .         .         .        .          .
# |    .    |    .    |    .    |    .    |    .    |
#      .         .         .          .        .
# ```
# 
# Note that all tiles (or sprits) should have the same size.
# 
# **Important**: total image size should not be too large (< 32767 x 32767) to be load by Chrome. [link](https://stackoverflow.com/questions/6081483/maximum-size-of-a-canvas-element)
# 
# For example, if we choose tile size 32x32 and number of horizontal tiles ~100 then we obtain total atlas image size : 

# In[ ]:


sprit_image_size = (32, 32)
n = 300
ids = _metadata_csv[['fname', 'split_index']].values
m = int(np.ceil(len(ids) * 1.0 / n))
(m*sprit_image_size[0], n*sprit_image_size[1])


# Let's create atlas image and store it locally:

# In[ ]:


atlas_image_path = "complete_atlas_image.png"

sprit_image_size = (32, 32)
if not os.path.exists(atlas_image_path):    
    ids = _metadata_csv[['fname', 'split_index']].values
    n = 300
    m = int(np.ceil(len(ids) * 1.0 / n))
    complete_image = PIL.Image.new('RGBA', (n*sprit_image_size[0], m*sprit_image_size[1]))
    counter = 0
    for i in range(m):
        print("-- %i / %i" % (counter, len(ids)))
        ys = i*sprit_image_size[1]
        ye = ys + sprit_image_size[1]
        for j in range(n):
            xs = j*sprit_image_size[0]
            xe = xs + sprit_image_size[0]
            if counter == len(ids):
                break
            image_id = ids[counter]; counter+=1
            try:
                img = load_favicon(*image_id)
                if img.size != sprit_image_size:
                    img = img.resize(size=sprit_image_size, resample=PIL.Image.BICUBIC)
                complete_image.paste(img.convert(mode='RGBA'), (xs, ys))
            except Exception:
                pass            
        if counter == len(ids):
            break        
    
    complete_image.save(atlas_image_path)
    del complete_image


# Now we can display all this images:

# In[ ]:


atlas_url = atlas_image_path


# Next cell's execution and rendering can take some time and freeze browser

# In[ ]:


# Display the Dive visualization for this data
from IPython.core.display import display, HTML

HTML_TEMPLATE = """<link rel="import" href="/nbextensions/facets-jupyter.html">
        <facets-dive 
            id="elem" 
            height="750"
            cross-origin="anonymous"
            sprite-image-width="32"
            sprite-image-height="32">
        </facets-dive>
        <script>
          var data = {jsonstr};
          var atlas_url = "{atlas_url}";
          document.querySelector("#elem").data = data;
          document.querySelector("#elem").atlasUrl = atlas_url;
        </script>"""
html = HTML_TEMPLATE.format(jsonstr=_metadata_csv.to_json(orient='records'), atlas_url=atlas_url)
display(HTML(html))


# Congrats, if you managed to see stacked icons! Otherwise, as facets is `html/javascript` package, you can open browser debugging console. I recall that facets works only with Chrome browser. Clearing all cells output also can help. In the end of the notebook, I provide some code that should run if facets is correctly setup.

# Choosing various options in `faceting` can split data for better visualization. Try the following configurations for *Row-Based* and *Column-Based* facetings:
# - width vs height
# - color_mode vs format
# - ...
# 
# As tool developer says about the motivation of facets:
# > Best summing up the motivation would be the motto: "Debug your data".
# 

# ## References:
# - [facets github source](https://github.com/pair-code/facets)
# - [Installation guide to Jupyter](https://github.com/pair-code/facets#enabling-usage-in-jupyter-notebooks)

# ### Appendix
# 
# For debugging purposes, if you run the following two cells, you should be able to see to circles

# In[1]:


jsonstr=[{
  "name": "apple",
  "category": "fruit",
  "calories": 95
},{
  "name": "broccoli",
  "category": "vegetable",
  "calories": 50
}]


# In[ ]:


# Display the Dive visualization for this data
from IPython.core.display import display, HTML

HTML_TEMPLATE = """<link rel="import" href="/nbextensions/facets-jupyter.html">
        <facets-dive id="elem" height="750"></facets-dive>
        <script>
          var data = {jsonstr};
          document.querySelector("#elem").data = data;
        </script>"""
html = HTML_TEMPLATE.format(jsonstr=jsonstr)
display(HTML(html))


# In[ ]:




