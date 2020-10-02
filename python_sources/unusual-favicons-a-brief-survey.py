#!/usr/bin/env python
# coding: utf-8

# Last year I scraped about 360,000 favicons (the little images browsers use to represent websites in tabs or the URL bar). This kernel focuses on some of the extremes in the dataset - in terms of size, aspect ratio, and color depth.
# 
# (Scroll down a page to skip ahead to the pictures.)

# In[ ]:


import PIL.Image
import pandas as pd
import numpy as np
import itertools
import math
import zipfile
from matplotlib import pyplot as plt
import matplotlib as mpl
plt.style.use('ggplot')


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
    row_height = 1 * scale * (1 if titles is None else 1.33)
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
    
def sample_nontrivial(df, n, random_state=2017):
    # shuffle
    df = df.sample(frac=1, random_state=random_state)
    indices = []
    for row in df.itertuples():
        try:
            img = load_favicon(row.fname, row.split_index)
            assert (np.array(img) != 0).any()
            indices.append(row.Index)
        except (CorruptFaviconException, AssertionError):
            continue
        if len(indices) >= n:
            return df.loc[indices]
    assert False, "Failed to find {} non-trivial/corrupt imgs".format(n)    


# In[ ]:


def load_metadata_df():
    """Return a dataframe with a row of metadata for each favicon in the dataset."""
    csvpath = '../input/favicon_metadata.csv'
    return pd.read_csv(csvpath)


# In[ ]:


df = load_metadata_df()
df.head(5)


# # Favicons big and small
# 
# Let's look at a random sample of favicons from the dataset.

# In[ ]:


sample =  df.sample(32, random_state=2017)
show(sample)


# I've scaled them to be the same size here, but the dataset has favicons of a variety of dimensions...

# In[ ]:


size_cols = ['width', 'height']
top_sizes = df.groupby(size_cols).size().sort_values(ascending=False).head(8).sort_values()
top_sizes.plot.barh(title='Most common favicon dimensions');


# In[ ]:


# Let's scatter the width and height of 1000 random favicons
fig, ax = plt.subplots(figsize=(8, 6))
eg = df.sample(1000, random_state=2000)
x, y = eg['width'].values, eg['height'].values
ax.scatter(x, y, alpha=.2);
ticks = [1, 16, 32, 48, 64, 100, 128, 256]
ax.set_xticks(ticks); ax.set_yticks(ticks)
ax.set_xlabel('width'); ax.set_ylabel('height');
ax.grid(False)
ax.set_title('width and height of 1000 favicons');


# ## Non-square favicons?
# 
# Most favicons fall under one of a small set of standard square sizes (16x16, 32x32, 48x48...).
# 
# But there are a few dots in the plot above that fall outside the main diagonal, representing favicons with  non-square aspect ratios. Let's see some examples.

# In[ ]:


nonsquare = df[df['width'] != df['height']]
print("{} out of {} favicons ({:.1%}) have non-square aspect ratios".format(
    len(nonsquare), len(df), len(nonsquare)/len(df),
))
print("For example...")
examples = nonsquare.sample(10, random_state=54321)
sizes = list(map(tuple, examples[size_cols].values))
show(examples, scale=1.5, titles=sizes)


# ## Tiny favicons
# 
# A surprising number of favicons are 1x1!

# In[ ]:


is_onepx = (df['width'] == 1) & (df['height'] == 1)
print("{} out of {} favicons ({:.1%}) are 1x1".format(
    is_onepx.sum(), len(df), is_onepx.sum()/len(df),
))
examples = df[is_onepx].sample(8, random_state=2017)
show(examples)


# Naturally, these aren't very interesting to look at, so let's filter them out and look at some of the smallest non-trivial favicons in the dataset.

# In[ ]:


orig_df = df
df = df[~is_onepx]


# In[ ]:


is_tiny = (df['width'] < 16) & (df['height'] < 16)
is_nano = (df['width'] < 10) & (df['height'] < 10)
tiny = df[is_tiny]
nano = df[is_nano]
print("{} favicons are bigger than 1x1 but smaller than 16x16.".format(len(tiny)))
examples = tiny.sample(8, random_state=12345)
sizes = list(map(tuple, examples[size_cols].values))
show(examples)
plt.show()
print("{} are smaller than 10x10".format(len(nano)))
examples = sample_nontrivial(nano, 16)
show(examples)


# They're kind of minimalistically beautiful, aren't they?
# 
# ## Huge favicons

# In[ ]:


is_big = (df['width'] > 256) & (df['height'] > 256)
big = df[is_big]
print("{} favicons are bigger than 256x256".format(len(big)))
examples = big.sample(8, random_state=123)
sizes = list(map(tuple, examples[size_cols].values))
show(examples, scale=4, titles=sizes)


# # Colorless favicons

# In[ ]:


ax = df['color_mode'].value_counts().plot.barh(title='Favicon color modes')


# A fair number of favicons are grayscale.

# In[ ]:


grayscale_modes = {'Grayscale', 'GrayscaleAlpha'}
is_grayscale = df['color_mode'].isin(grayscale_modes)
print("{:,} grayscale favicons".format(is_grayscale.sum()))
examples = df[is_grayscale].sample(24, random_state=123)
show(examples)


# Many use the even more extreme 'bilevel' color mode, which is limited to just black and white.

# In[ ]:


is_bilevel = df['color_mode'] == 'Bilevel'
print("{:,} bilevel favicons".format(is_bilevel.sum()))
examples = df[is_bilevel].sample(24, random_state=1234)
show(examples)


# Some of these are able to effectively cheat out shades of gray using the alpha channel.

# # Simple favicons
# 
# Many favicons in the dataset are stored in a compressed format. Simple ones should compress more efficiently.

# In[ ]:


pd.options.mode.chained_assignment = None  # default='warn'
df['pixels'] = df['width'] * df['height']
df['bpp'] = (df['file_size'] * 8) / df['pixels']


# In[ ]:


# Fix a compression method, color mode, and size so we're comparing apples to apples.
zp = df[
    (df['compression']=='Zip') & (df['color_mode']=='PaletteAlpha')
    & (df['width']==32) & (df['height']==32)    
]
print("\"Simple\" favicons (less than 2 bits per pixel)...")
sample = zp[zp['bpp'] < 2].sample(16, random_state=123)
show(sample)
plt.show()

print("More than 30 bits per pixel...")
sample = zp[zp['bpp'] > 30].sample(16, random_state=123)
show(sample)


# # Big and small palettes
# 
# The metadata df also includes a 'depth' column, which has the [color depth](https://en.wikipedia.org/wiki/Color_depth) inferred by ImageMagick.
# 
# Most favicons use [indexed color](https://en.wikipedia.org/wiki/Indexed_color) (corresponding to `color_mode in {'Palette', 'PaletteAlpha'}`). An image using indexed color with depth = n can use up to 2^n distinct colors. (These files store a lookup table of colors, and represent each pixel by a pointer into that table.)

# In[ ]:


# Workaround for ImageMagick bug: https://github.com/ImageMagick/ImageMagick/issues/683
real_depth = df.loc[df['color_mode']=='PaletteAlpha', 'depth']    .apply(lambda d: min(d, 8))    .value_counts().sort_index(ascending=False)
real_depth.plot.barh(title='Distribution of favicon color depth');


# ## 256 colors
# 
# Indexed color images with `depth = 8` can use 256 distinct colors.

# In[ ]:


deep = df[(df['color_mode']=='PaletteAlpha') & (df['depth']>=8)]
print("{:,} favicons have depth 8".format(len(deep)))
sample = deep.sample(24, random_state=1234)
show(sample)


# ## Shallow color
# 
# Images with `depth = 4` have to make do with just 16 distinct colors.

# In[ ]:


shallow = df[(df['color_mode']=='PaletteAlpha') & (df['depth']==4)]
print("{:,} favicons have depth = 4".format(len(shallow)))
sample = shallow.sample(24, random_state=1234)
show(sample)


# Many of these are clearly designed for a limited palette, but some (such as the car) try to pull off something more complex and end up with some pretty ugly [dithering](https://en.wikipedia.org/wiki/Dither) artifacts.
# 
# ## 1-bit color

# In[ ]:


shallow = df[(df['color_mode']=='PaletteAlpha') & (df['depth']==1)]
print("{} favicons have depth = 1".format(len(shallow)))
sample = shallow.sample(16, random_state=123)
show(sample)


# These look like some sublime pieces of glitch art. But actually, this does seem to be a glitch in PIL.

# In[ ]:


sample.iloc[:5]['url'].values


# Most of these sites seem not to have changed their favicons since they were scraped. For example, Firefox and Chrome render the favicon for [privoxy.org](https://www.privoxy.org/) (the second image above) as a white 'P' over a blue circle, which seems a lot more natural than... whatever it is PIL drew.
