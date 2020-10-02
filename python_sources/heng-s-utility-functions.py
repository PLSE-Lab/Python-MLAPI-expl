import numpy as np
import pandas as pd
import os

from timeit import default_timer as timer
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch.utils.data as data
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
import torch

PI = np.pi
IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]
DEFECT_COLOR = [(0,0,0),(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

# AUGMENTATIONS
# All the augmentations implemented from scratch
def do_random_crop(image, mask, w, h):
    height, width = image.shape[:2]
    x,y=0,0
    if width>w:
        x = np.random.choice(width-w)
    if height>h:
        y = np.random.choice(height-h)
    image = image[y:y+h,x:x+w]
    mask  = mask [:,y:y+h,x:x+w]
    return image, mask

def do_random_crop_rescale(image, mask, w, h):
    height, width = image.shape[:2]
    x,y=0,0
    if width>w:
        x = np.random.choice(width-w)
    if height>h:
        y = np.random.choice(height-h)
    image = image[y:y+h,x:x+w]
    mask  = mask [:,y:y+h,x:x+w]

    #---
    if (w,h)!=(width,height):
        image = cv2.resize( image, dsize=(width,height), interpolation=cv2.INTER_LINEAR)

        mask = mask.transpose(1,2,0)
        mask = cv2.resize( mask,  dsize=(width,height), interpolation=cv2.INTER_NEAREST)
        mask = mask.transpose(2,0,1)

    return image, mask




def do_flip_lr(image, mask):
    image = cv2.flip(image, 1)
    mask  = mask[:,:,::-1]
    return image, mask

def do_flip_ud(image, mask):
    image = cv2.flip(image, 0)
    mask  = mask[:,::-1,:]
    return image, mask




def do_random_scale_rotate(image, mask, w, h):
    H,W = image.shape[:2]

    #dangle = np.random.uniform(-2.5, 2.5)
    #dscale = np.random.uniform(-0.10,0.10,2)
    dangle = np.random.uniform(-5, 5)
    dscale = np.random.uniform(-0.15,0.15,2)
    dshift = np.random.uniform(0,1,2)
    cos = np.cos(dangle/180*PI)
    sin = np.sin(dangle/180*PI)
    sx,sy = 1 + dscale #1,1 #
    tx,ty = dshift

    src = np.array([[-w/2,-h/2],[ w/2,-h/2],[ w/2, h/2],[-w/2, h/2]], np.float32)
    src = src*[sx,sy]
    x = (src*[cos,-sin]).sum(1)
    y = (src*[sin, cos]).sum(1)
    x = x-x.min()
    y = y-y.min()
    x = x + (W-x.max())*tx
    y = y + (H-y.max())*ty

    if 0:
        overlay=image.copy()
        for i in range(4):
            cv2.line(overlay, int_tuple([x[i],y[i]]), int_tuple([x[(i+1)%4],y[(i+1)%4]]), (0,0,255),5)
        image_show('overlay',overlay)
        cv2.waitKey(0)


    src = np.column_stack([x,y])
    dst = np.array([[0,0],[w,0],[w,h],[0,h]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s,d)

    image = cv2.warpPerspective( image, transform, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    mask = mask.transpose(1,2,0)
    mask = cv2.warpPerspective( mask, transform, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    mask = mask.transpose(2,0,1)
    mask = (mask>0.5).astype(np.float32)

    return image, mask


def do_random_crop_rotate_rescale(image, mask, w, h):
    H,W = image.shape[:2]

    #dangle = np.random.uniform(-2.5, 2.5)
    #dscale = np.random.uniform(-0.10,0.10,2)
    dangle = np.random.uniform(-8, 8)
    dshift = np.random.uniform(-0.1,0.1,2)

    dscale_x = np.random.uniform(-0.00075,0.00075)
    dscale_y = np.random.uniform(-0.25,0.25)

    cos = np.cos(dangle/180*PI)
    sin = np.sin(dangle/180*PI)
    sx,sy = 1 + dscale_x, 1+ dscale_y #1,1 #
    tx,ty = dshift*min(H,W)

    src = np.array([[-w/2,-h/2],[ w/2,-h/2],[ w/2, h/2],[-w/2, h/2]], np.float32)
    src = src*[sx,sy]
    x = (src*[cos,-sin]).sum(1)+W/2
    y = (src*[sin, cos]).sum(1)+H/2
    # x = x-x.min()
    # y = y-y.min()
    # x = x + (W-x.max())*tx
    # y = y + (H-y.max())*ty

    if 0:
        overlay=image.copy()
        for i in range(4):
            cv2.line(overlay, int_tuple([x[i],y[i]]), int_tuple([x[(i+1)%4],y[(i+1)%4]]), (0,0,255),5)
        image_show('overlay',overlay)
        cv2.waitKey(0)


    src = np.column_stack([x,y])
    dst = np.array([[0,0],[w,0],[w,h],[0,h]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s,d)

    image = cv2.warpPerspective( image, transform, (W, H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    mask = mask.transpose(1,2,0)
    mask = cv2.warpPerspective( mask, transform, (W, H),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    mask = mask.transpose(2,0,1)


    return image, mask
def do_random_log_contast(image):
    gain = np.random.uniform(0.70,1.30,1)
    inverse = np.random.choice(2,1)

    image = image.astype(np.float32)/255
    if inverse==0:
        image = gain*np.log(image+1)
    else:
        image = gain*(2**image-1)

    image = np.clip(image*255,0,255).astype(np.uint8)
    return image


def do_noise(image, mask, noise=8):
    H,W = image.shape[:2]
    image = image + np.random.uniform(-1,1,(H,W,1))*noise
    image = np.clip(image,0,255).astype(np.uint8)
    return image, mask


def train_augment(image, mask, infor):
    u=np.random.choice(3)
    if u==0:
        pass
    elif u==1:
        image, mask = do_random_crop_rescale(image, mask, 1600-(256-224), 224)
    elif u==2:
        image, mask = do_random_crop_rotate_rescale(image, mask, 1600-(256-224), 224)

    if np.random.rand()>0.5:
        image = do_random_log_contast(image)

    if np.random.rand()>0.5:
        image, mask = do_flip_lr(image, mask)

    if np.random.rand()>0.5:
        image, mask = do_flip_ud(image, mask)

    if np.random.rand()>0.5:
        image, mask = do_noise(image, mask)
    return image, mask, infor


def valid_augment(image, mask, infor):
    return image, mask, infor


# TIME
def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError

def df_loc_by_list(df, key, values):
    df = df.loc[df[key].isin(values)]
    df = df.assign(sort = pd.Categorical(df[key], categories=values, ordered=True))
    df = df.sort_values('sort')
    #df = df.reset_index()
    df = df.drop('sort', axis=1)
    return  df


# Image and resnet input adjustment
def image_to_input(image,rbg_mean,rbg_std):#, rbg_mean=[0,0,0], rbg_std=[1,1,1]):
    input = image.astype(np.float32)
    input = input[...,::-1]/255
    input = input.transpose(0,3,1,2)
    input[:,0] = (input[:,0]-rbg_mean[0])/rbg_std[0]
    input[:,1] = (input[:,1]-rbg_mean[1])/rbg_std[1]
    input[:,2] = (input[:,2]-rbg_mean[2])/rbg_std[2]
    return input


def input_to_image(input,rbg_mean,rbg_std):#, rbg_mean=[0,0,0], rbg_std=[1,1,1]):
    input = input.data.cpu().numpy()
    input[:,0] = (input[:,0]*rbg_std[0]+rbg_mean[0])
    input[:,1] = (input[:,1]*rbg_std[1]+rbg_mean[1])
    input[:,2] = (input[:,2]*rbg_std[2]+rbg_mean[2])
    input = input.transpose(0,2,3,1)
    input = input[...,::-1]
    image = (input*255).astype(np.uint8)
    return image

# Drawing functions
def mask_to_inner_contour(mask):
    mask = mask>0.5
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour


def draw_contour_overlay(image, mask, color=(0,0,255), thickness=1):
    contour =  mask_to_inner_contour(mask)
    if thickness==1:
        image[contour] = color
    else:
        for y,x in np.stack(np.where(contour)).T:
            cv2.circle(image, (x,y), thickness//2, color, lineType=cv2.LINE_4 )
    return image

def draw_mask_overlay(image, mask, color=(0,0,255), alpha=0.5):
    H,W,C = image.shape
    mask = (mask*alpha).reshape(H,W,1)
    overlay = image.astype(np.float32)
    overlay = np.maximum( overlay, mask*color )
    overlay = np.clip(overlay,0,255)
    overlay = overlay.astype(np.uint8)
    return overlay

def draw_grid(image, grid_size=[32,32], color=[64,64,64], thickness=1):
    H,W,C = image.shape
    dx,dy = grid_size

    for x in range(0,W,dx):
        cv2.line(image,(x,0),(x,H),color, thickness=thickness)
    for y in range(0,H,dy):
        cv2.line(image,(0,y),(W,y),color, thickness=thickness)
    return image


def draw_predict_result(image, truth_mask, truth_label, probability_mask, stack='horizontal', scale=-1):
    color = DEFECT_COLOR

    if scale >0:
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    H,W,C   = image.shape
    overlay = image.copy()
    result  = []
    for c in range(4):
        r = np.zeros((H,W,3),np.uint8)

        if scale >0:
            t = cv2.resize(truth_mask[c], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            p = cv2.resize(probability_mask[c], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            t = truth_mask[c]
            p = probability_mask[c]

        #r = draw_mask_overlay(r, p, color[c+1], alpha=1)
        r = draw_mask_overlay(r, p, (255,255,255), alpha=1)
        r = draw_contour_overlay(r, t, color[c+1], thickness=2)
        draw_shadow_text(r,'predict%d'%(c+1),(5,30),1,color[c+1],2)
        overlay = draw_contour_overlay(overlay, t, color[c+1], thickness=6)
        result.append(r)

    draw_shadow_text(overlay,'truth',(5,30),1,[255,255,255],2)
    result = [image,overlay,] + result
    if stack=='horizontal':
        result = np.hstack(result)
    if stack=='vertical':
        result = np.vstack(result)

    result = draw_grid(result, grid_size=[W,H], color=[255,255,255], thickness=1)
    return result

def draw_predict_result_single(image, truth_mask, truth_label, probability_mask, stack='horizontal', scale=-1):
    color = DEFECT_COLOR


    if scale >0:
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        p = cv2.resize(probability_mask[0], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        p = probability_mask[0]

    H,W,C   = image.shape
    r = np.zeros((H,W,3),np.uint8)
    r = draw_mask_overlay(r, p, (255,255,255), alpha=1)

    overlay = image.copy()
    for c in range(4):
        if scale >0:
            t = cv2.resize(truth_mask[c], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            t = truth_mask[c]

        r = draw_contour_overlay(r, t, color[c+1], thickness=4)
        overlay = draw_contour_overlay(overlay, t, color[c+1], thickness=4)


    draw_shadow_text(r,'predict(all)',(5,30),1,(255,255,255),2)
    draw_shadow_text(overlay,'truth',(5,30),1,[255,255,255],2)
    result = [image,overlay,r]
    if stack=='horizontal':
        result = np.hstack(result)
    if stack=='vertical':
        result = np.vstack(result)

    result = draw_grid(result, grid_size=[W,H], color=[255,255,255], thickness=1)
    return result

def draw_predict_result_32x32(image, truth_mask, truth_label, probability_label):
    color = DEFECT_COLOR
    H,W,C = image.shape

    result  = []
    overlay = image.copy()
    for c in range(4):
        overlay = draw_contour_overlay(overlay, truth_mask[c], color[c+1], thickness=2)

        t = truth_label[c][...,np.newaxis]*color[c+1]
        p = probability_label[c][...,np.newaxis]*[255,255,255]
        t = t.astype(np.uint8)
        p = p.astype(np.uint8)
        r = np.hstack([t,p])

        result.append(r)

    result = np.vstack(result)
    result = cv2.resize(result, dsize=None, fx=32,fy=32, interpolation=cv2.INTER_NEAREST)
    assert(result.shape==(4*H,2*W,3))

    result  = draw_grid(result, grid_size=[32,32], color=[64,64,64], thickness=1)
    overlay = draw_grid(overlay, grid_size=[32,32], color=[255,255,255], thickness=1)


    result = np.vstack([
        np.hstack([overlay, image]),
        result
    ])
    result = draw_grid(result, grid_size=[W,H], color=[255,255,255], thickness=3)
    return result


def draw_predict_result_label(image, truth_mask, truth_label, probability_label, stack='horizontal', scale=-1):
    color = DEFECT_COLOR

    if scale >0:
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    H,W,C   = image.shape
    overlay = image.copy()
    for c in range(4):
        if scale >0:
            t = cv2.resize(truth_mask[c], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            t = truth_mask[c]
#         overlay = draw_contour_overlay(overlay, t, color[c+1], thickness=4)

    for c in range(4):
        draw_shadow_text(overlay,'pos%d %0.2f (%d)'%(c+1,probability_label[c],truth_label[c]),(5,(c+1)*24),0.75,color[c+1],1)


    draw_shadow_text(overlay,'truth',(5,30),1,[255,255,255],2)
    result = [image,overlay]
    if stack=='horizontal':
        result = np.hstack(result)
    if stack=='vertical':
        result = np.vstack(result)

    result = draw_grid(result, grid_size=[W,H], color=[255,255,255], thickness=1)
    return result

def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)
    
def image_show(name, image, resize=1):
    H,W = image.shape[0:2]
#     cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)  #WINDOW_NORMAL
    #cv2.namedWindow(name, cv2.WINDOW_GUI_EXPANDED)  #WINDOW_GUI_EXPANDED
    for a in range(2):
        plt.subplots(a,1,figsize=(40,10))
        plt.imshow(image[a*256:(a+1)*256,:])
#     cv2.resizeWindow(name, round(resize*W), round(resize*H))
