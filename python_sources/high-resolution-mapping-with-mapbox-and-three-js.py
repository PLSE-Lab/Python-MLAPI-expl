#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Step 1: Install of the packages
get_ipython().system('pip install mercantile')


# In[ ]:


# The first step is to define the min/max (lat,lng)
# Go to google maps and select Whats here to see the top left and bottom right sets.
# Set the zoom level (resolution level)
lat_lng = [43.640918, -79.371478]
delta = 0.05
tl = [lat_lng[0]+delta, lat_lng[1]-delta]
br = [lat_lng[0]-delta, lat_lng[1]+delta]
z = 15 # Set the resolution


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
get_ipython().system('mkdir ./animate')

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
    print(total_width,total_height)

    anim_idx = 0
    y_offset = 0
    for i in range(0,edge_length_x):
        x_offset = 0
        for j in range(0,edge_length_y):
            tmp_img = PIL.Image.open('./'+img_name+'_images/' + str(i) + '.' + str(j) + '.png')
            composite.paste(tmp_img, (y_offset,x_offset))
            x_offset += width
            composite.save('./animate/'+str(anim_idx).zfill(4)+'.jpg',optimize=True,quality=95)
            anim_idx += 1
            print(anim_idx)

            
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
    for w in range(rgb_elevation.width):
        R, G, B, A = rgb_elevation.getpixel((w, h))
        height = -10000 + ((R * 256 * 256 + G * 256 + B) * 0.1)
        elevation_data.append(height)

import json
with open('./elevation.json', 'w') as outfile:
    json.dump(elevation_data, outfile)

