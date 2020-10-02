#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Step 1: Install of the packages
get_ipython().system('pip install mercantile')


# In[ ]:


# The first step is to define the min/max (lat,lng)
# Go to google maps and select Whats here to see the top left and bottom right sets.
# Set the zoom level (resolution level)
lat_lng = [40.7231212,-74.0173587]
delta = 0.13
tl = [lat_lng[0]+delta, lat_lng[1]-delta]
br = [lat_lng[0]-delta, lat_lng[1]+delta]
z = 13 # Set the resolution


# In[ ]:


# find the tile set IDs (x/y) for the top left and bottom right at the zoom level
import mercantile
tl_tiles = mercantile.tile(tl[1],tl[0],z)
br_tiles = mercantile.tile(br[1],br[0],z)

x_tile_range = [tl_tiles.x,br_tiles.x];print(x_tile_range)
y_tile_range = [tl_tiles.y,br_tiles.y];print(y_tile_range)


# In[ ]:


# Make the folders
get_ipython().system('mkdir ./satellite_images')
get_ipython().system('rm -rf ./satellite_images/*')
get_ipython().system('mkdir ./elevation_images')
get_ipython().system('rm -rf ./elevation_images/*')


# In[ ]:


# Loop over the ranges and extract the images from mapbox for both satellite and elevation at @2x resolution (512x512)
import requests
import shutil
for i,x in enumerate(range(x_tile_range[0],x_tile_range[1]+1)):
    for j,y in enumerate(range(y_tile_range[0],y_tile_range[1]+1)):
        print(x,y)
        r = requests.get('https://api.mapbox.com/v4/mapbox.terrain-rgb/'+str(z)+'/'+str(x)+'/'+str(y)+'@2x.pngraw?access_token=pk.eyJ1Ijoia2FwYXN0b3IiLCJhIjoiY2p3eTg3eWJoMG1jZjQ4bzZmcGg5c3F3cSJ9.vhyCyD9xDDGP9EQnhB9xtA', stream=True)
        if r.status_code == 200:
            with open('./elevation_images/' + str(i) + '.' + str(j) + '.png', 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)  
        
        r = requests.get('https://api.mapbox.com/v4/mapbox.satellite/'+str(z)+'/'+str(x)+'/'+str(y)+'@2x.png?access_token=pk.eyJ1Ijoia2FwYXN0b3IiLCJhIjoiY2p3eTg3eWJoMG1jZjQ4bzZmcGg5c3F3cSJ9.vhyCyD9xDDGP9EQnhB9xtA', stream=True)
        if r.status_code == 200:
            with open('./satellite_images/' + str(i) + '.' + str(j) + '.png', 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)


# In[ ]:


# Combine the tiles into a single large image
get_ipython().system('mkdir ./composite_images')

import PIL
import math
from os import listdir
from os.path import isfile, join

for img_name in ['elevation','satellite']:
    image_files = ['./'+img_name+'_images/' + f for f in listdir('./'+img_name+'_images/')]
    images = [PIL.Image.open(x) for x in image_files]

    edge_length_x = x_tile_range[1] - x_tile_range[0]
    edge_length_y = y_tile_range[1] - y_tile_range[0]
    edge_length_x = max(1,edge_length_x)
    edge_length_y = max(1,edge_length_y)
    width, height = images[0].size

    total_width = width*edge_length_x
    total_height = height*edge_length_y

    composite = PIL.Image.new('RGB', (total_width, total_height))

    anim_idx = 0
    y_offset = 0
    for i in range(0,edge_length_x):
        x_offset = 0
        for j in range(0,edge_length_y):
            tmp_img = PIL.Image.open('./'+img_name+'_images/' + str(i) + '.' + str(j) + '.png')
            composite.paste(tmp_img, (y_offset,x_offset))
            x_offset += width

            
        y_offset += height

    composite.save('./composite_images/'+img_name+'.png')



# In[ ]:


get_ipython().system('apt-get update')
get_ipython().system('apt install -y ffmpeg')
get_ipython().system('rm output.mp4')
get_ipython().system('ffmpeg  -i ./animate/%04d.jpg -c:v libx264 -c:a aac -ar 44100  -pix_fmt yuv420p output.mp4')
        
#         ffmpeg -ss 30 -t 3 -i input.mp4 -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif


# In[ ]:


import PIL
elevation_raw = PIL.Image.open('./composite_images/elevation.png')
rgb_elevation = elevation_raw.convert('RGBA')

# Loop over the image and save the data in a list:
elevation_data = []
# texture_data = []
for h in range(rgb_elevation.height):
    elev_row = []
    for w in range(rgb_elevation.width):
        R, G, B, A = rgb_elevation.getpixel((w, h))
        height = -10000 + ((R * 256 * 256 + G * 256 + B) * 0.1)
        elev_row.append(height)
    elevation_data.append(elev_row)
import json
with open('./elevation.json', 'w') as outfile:
    json.dump(elevation_data, outfile)


# In[ ]:


# Use the elevation data to create an image mask to modify the pixels of blue water overlayed on the surface
# First we will make anytning look blue
import numpy as np
get_ipython().system('mkdir ./depth')

for i, level in enumerate(np.arange(0,100,0.25)):
    print(i,level)
    im = PIL.Image.open('./composite_images/satellite.png').convert('RGBA')
    overlay = PIL.Image.new('RGBA', im.size,(4,22,37,255))
    # overlay = PIL.Image.new('RGBA', im.size,(255,0,0,255))

    ni = np.array(overlay)



    e=np.array([np.array(xi) for xi in elevation_data])
    



    depth = level - e
    print(depth.min(),depth.max())
    # Now we have the depths  we want anything > 0 to be zero alpha
    alpha_mask = np.copy(depth)
    alpha_mask = alpha_mask*255/alpha_mask.max()
    alpha_mask  = np.where(alpha_mask<0, 0, alpha_mask)
#     alpha_mask  = np.where(alpha_mask>200, 255, alpha_mask)
    alpha_mask = alpha_mask**.2 
    alpha_mask = alpha_mask*255/alpha_mask.max()
    print(alpha_mask.min(),alpha_mask.max(),alpha_mask.mean())




    ni[...,3] = alpha_mask[...]
    
    # ni[...,3] = 255 - ni[...,0]
    # ni[:,:,(0,1,2)] = 255
#     print(e.shape)


    W = PIL.Image.fromarray(ni)

    im.paste(W , (0,0),W)
    im.save('./depth/'+ str(i).zfill(4) +'.png')

# img = PIL.Image.new('RGBA', (s_raw.width, s_raw.height))
# img_pixels = img.load()

# for i in range(s_raw.size[0]):
#     for j in range(s_raw.size[1]):
#         R, G, B, A = s_raw.getpixel((i, j))
#         img_pixels[w,h] = (0,0,255,255)
        
# img.save('./test.png')


# In[ ]:


get_ipython().system('apt-get update')
get_ipython().system('apt install -y ffmpeg')
get_ipython().system('rm output.mp4')
get_ipython().system('ffmpeg  -i ./depth/%04d.png -c:v libx264 -c:a aac -ar 44100 -filter "minterpolate=\'fps=30\'" -pix_fmt yuv420p output.mp4')
get_ipython().system('ffmpeg  -i ./depth/%04d.png -c:v libx264 -c:a aac -ar 44100 -filter "minterpolate=\'fps=30\'" -pix_fmt yuv420p output.mp4')

