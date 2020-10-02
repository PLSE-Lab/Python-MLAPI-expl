#!/usr/bin/env python
# coding: utf-8

# ```python
# # convert your RLE to images, saved in a folder
# # Run as: python rle_to_image.py
# # Parameter: specify filename in the function
# # mkdir folder sept15_out before running
# 
# from PIL import Image
# mp_0 = [(0,0,0)]*(1280*1918)
# 
# def get_pixels(filename='train_masks.csv'):
#              f = open(filename, "r") # add folder name if required
#              lines = f.readlines()
#              for line in lines[1:]:
#                  [fname, line] = line.split(',')
#                  lst = line.split(' ')
#                  n = len(lst)
#                  
#                  mp = mp_0
#                  for i in range(0, n, 2):
#                      p = int(lst[i]) - 1
# 
#                      v = int(lst[i+1])
#                      for j in range(v):
#                        	mp[p + j] = (255, 255, 255)
#                  #return mp
#                  im = Image.new('RGB', (1918, 1280))
#                  im.putdata(mp)
#                  im.save('sept15_out/' + fname) # mkdir folder sept15_out before running
# get_pixels()
# ```

# Convert your RLE to images.
