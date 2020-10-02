#!/usr/bin/env python
# coding: utf-8

# ```python
# # Compare your predicted output against ground truth (training masks)
# # Run as: python mask_over_mask.py
# # Change folder name of your predicted images (to use as red mask)
# # train masks are green in color.
# # create output folder 'compare' before running
# # (in case you just have RLE, use this to get 
# #  images first: https://www.kaggle.com/mkagenius/rle-to-images )
# 
# import os
# from PIL import Image
# R, G, B = 0,1,2
# 
# flag = False
# predicted_masks_folder = '15sept_out'
# for fname in os.listdir(predicted_masks_folder):
#           pred_img = Image.open(predicted_masks_folder+'/'+fname)
# 
#           source = pred_img.split()
#           m1 = source[R].point(lambda i: i > 60 and 255)
#        	  m4 = source[R].point(lambda i: i > 60 and 122)
#        	  if not flag:
#        	       	flag = True
#                 m2 = source[R].point(lambda i: 0)
#                 m3 = source[R].point(lambda i: 0)
#           im_with_trans = Image.merge('RGBA', [m1, m2, m3, m4])
#         
#         # Using ground truth now 
#        	  fname_train_mask = fname.split('.')[0] + "_mask.gif" # change extension if req.
#           im_train = Image.open('train_masks/'+fname_train_mask) # change folder name if req.
#        	  im_train = im_train.convert('RGB')
#        	  source = im_train.split()
# 
#        	  m1 = source[G].point(lambda i: i )
#        	  m4 = source[G].point(lambda i: 122)
# 
# 
#        	  im_train_with_trans = Image.merge('RGBA', [m2, m1, m3, m4])
#           im_train_with_trans.paste(im_with_trans, (0,0), im_with_trans)
#           im_train_with_trans.save('compare/'+fname_train_mask) # create folder before running
#           
# ```   

# This can be used to see what errors are done by your model. Later you are think of strategies to improve your score.
