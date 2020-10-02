#!/usr/bin/env python
# coding: utf-8

# # Updated Data exploration notebook  
# *We need to know the data we work with.*
# 
# Train data contains:
# 
#  - csv file with train images and labels 
#  - images without markers
#  - image with markers
#  - MismatchedTrainImages.txt 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pylab as plt

def plt_st(l1,l2):
    plt.figure(figsize=(l1,l2))
    
def display_img(img_data, roi=None, no_colorbar=True, **kwargs):
    if roi is not None:
        # roi is [minx, miny, maxx, maxy]
        x,y,xw,yh = roi
        img_data = img_data[y:yh,x:xw,:] if len(img_data.shape) == 3 else img_data[y:yh,x:xw]

    plt.imshow(img_data, **kwargs)
    if not no_colorbar:
        plt.colorbar(orientation='horizontal')    


# In[ ]:


get_ipython().system('ls ../input/Train/*.jpg | wc -l')
get_ipython().system('ls ../input/TrainDotted/*.jpg | wc -l')
get_ipython().system('ls ../input/Test/*.jpg | wc -l')


# In[ ]:


import os
DATA_PATH = "../input"
TRAIN_PATH = os.path.join(DATA_PATH, 'Train')
TRAIN_DOTTED_PATH = os.path.join(DATA_PATH, 'TrainDotted')
TEST_PATH = os.path.join(DATA_PATH, 'Test')
TRAIN_CSV_DATA = pd.read_csv(os.path.join(TRAIN_PATH, 'train.csv'))

def get_filename(image_id, image_type=''):
    ext = 'jpg'
    if image_type == '':
        data_path = TRAIN_PATH
    elif image_type == 'dotted':
        data_path = TRAIN_DOTTED_PATH
    elif image_type == 'test':
        data_path = TEST_PATH
    else:
        raise Exception("Image type '%s' is not recognized" % image_type) 
        
    if not os.path.exists(data_path):
        os.makedirs(data_path)
   
    return os.path.join(data_path, "{}.{}".format(image_id, ext))


import cv2
from PIL import Image

def get_image_data(image_id, image_type=''):
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# In[ ]:


IMAGE_IDS = get_ipython().getoutput('ls {TRAIN_PATH}/*.jpg')
IMAGE_IDS = [image_id[len(TRAIN_PATH)+1:-4] for image_id in IMAGE_IDS]

TEST_IMAGE_IDS = get_ipython().getoutput('ls {TEST_PATH}/*.jpg')
TEST_IMAGE_IDS = [image_id[len(TEST_PATH)+1:-4] for image_id in TEST_IMAGE_IDS]


# ## What a single image looks like

# In[ ]:


plt_st(10, 10)
plt.subplot(121)
plt.title("Train image")
plt.imshow(get_image_data(IMAGE_IDS[0]))
plt.subplot(122)
plt.title("Train image dotted")
_ = plt.imshow(get_image_data(IMAGE_IDS[0],'dotted'))


# ## What all available train images look like

# In[ ]:


tile_size = (256, 256)
n = 5
m = int(np.ceil(len(IMAGE_IDS) * 1.0 / n))
complete_train_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
counter = 0
for i in range(m):
    ys = i*(tile_size[1] + 2)
    ye = ys + tile_size[1]
    for j in range(n):
        xs = j*(tile_size[0] + 2)
        xe = xs + tile_size[0]
        if counter == len(IMAGE_IDS):
            break
        image_id = IMAGE_IDS[counter]; counter+=1
        img = get_image_data(image_id)
        img = cv2.resize(img, dsize=tile_size)
        img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=3)
        complete_train_image[ys:ye, xs:xe, :] = img[:,:,:]
    if counter == len(IMAGE_IDS):
        break


# In[ ]:


plt_st(15, 15)
plt.imshow(complete_train_image)
plt.title("Training dataset")


# In[ ]:


tile_size = (256, 256)
n = 5
m = int(np.ceil(len(IMAGE_IDS) * 1.0 / n))
complete_train_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
counter = 0
for i in range(m):
    ys = i*(tile_size[1] + 2)
    ye = ys + tile_size[1]
    for j in range(n):
        xs = j*(tile_size[0] + 2)
        xe = xs + tile_size[0]
        if counter == len(IMAGE_IDS):
            break
        image_id = IMAGE_IDS[counter]; counter+=1
        img = get_image_data(image_id, 'dotted')
        img = cv2.resize(img, dsize=tile_size)
        img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=3)
        complete_train_image[ys:ye, xs:xe, :] = img[:,:,:]
    if counter == len(IMAGE_IDS):
        break


# In[ ]:


plt_st(15, 15)
plt.imshow(complete_train_image)
plt.title("Training dotted dataset")


# ## and all available test images 

# In[ ]:


tile_size = (256, 256)
n = 3
m = int(np.ceil(len(TEST_IMAGE_IDS) * 1.0 / n))
complete_train_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
counter = 0
for i in range(m):
    ys = i*(tile_size[1] + 2)
    ye = ys + tile_size[1]
    for j in range(n):
        xs = j*(tile_size[0] + 2)
        xe = xs + tile_size[0]
        if counter == len(TEST_IMAGE_IDS):
            break
        image_id = TEST_IMAGE_IDS[counter]; counter+=1
        img = get_image_data(image_id, 'test')
        img = cv2.resize(img, dsize=tile_size)
        img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=3)
        complete_train_image[ys:ye, xs:xe, :] = img[:,:,:]
    if counter == len(TEST_IMAGE_IDS):
        break


# In[ ]:


plt_st(15, 15)
plt.imshow(complete_train_image)
plt.title("Test dataset")


# In[ ]:





# ## Some images of TrainDotted are NOT exact copies with annotations
# 
# These are : 
# 
# ```
# train_id
# 3, 7, 9, 21, 30 ,34 ,71 ,81 ,89 ,97 ,151 ,184 ,215 ,234 ,242 ,268 ,290 ,311 ,331 ,344 ,380 ,384 ,406 ,421 ,469 ,475 ,490 ,499 ,507 ,530 ,531 ,605 ,607 ,614 ,621 ,638 ,644 ,687 ,712 ,721 ,767 ,779 ,781 ,794 ,800 ,811 ,839 ,840 ,869 ,882 ,901 ,903 ,905 ,909 ,913 ,927 ,946
# ```

# In[ ]:


roi = [1500, 1760, 2000, 2260]
for image_id in [IMAGE_IDS[4], IMAGE_IDS[8], IMAGE_IDS[10]]:
    img1 = get_image_data(image_id)
    img2 = get_image_data(image_id, 'dotted')
    plt_st(12, 10)
    plt.subplot(121)
    display_img(img1, roi=roi)
    plt.subplot(122)
    display_img(img2, roi=roi)
    plt.suptitle("Original VS Dotted, image id = %s" % image_id)


# In[ ]:





# ## Animal marker detection
# 
# Inspired by this [kernel](https://www.kaggle.com/asymptote/noaa-fisheries-steller-sea-lion-population-count/initial-exploration)

# In[ ]:


def get_n_animals(image_id):
    df = TRAIN_CSV_DATA[TRAIN_CSV_DATA['train_id'] == int(image_id)]
    df = df.drop('train_id', axis=1)
    return df.sum(axis=1).values[0]


# In[ ]:


def create_train_mask(image_id):
    img1 = get_image_data(image_id)
    img2 = get_image_data(image_id, 'dotted')
    black_mask = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    black_mask = cv2.medianBlur(black_mask, 5)
    black_mask = (black_mask > 10).astype(np.uint8)    
    img11 = img1 * np.expand_dims(black_mask, axis=2)    
    diff = cv2.absdiff(img2, img11)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)    
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, 
                            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    ret,mask = cv2.threshold(gray, 10, 255,cv2.THRESH_BINARY)
    return mask, img1, img2

def filter_by_area(cnts, min_area=2.0**2):
    out = []
    for c in cnts:
        if cv2.contourArea(c) > min_area:
            out.append(c)
    return out


# In[ ]:


# red: adult males
# magenta: subadult males
# brown: adult females
# blue: juveniles
# green: pups

MARKER_COLORS = {
    (255, 0, 0): 'adult males - red',
    (255, 0, 255): 'subadult males - magenta',
    (127, 69, 0): 'adult females - brown',
    (0, 0, 255): 'juveniles - blue',
    (0, 255, 0): 'pups - green'    
}
    


# In[ ]:


def find_markers(image_id):
    out = []
    mask, img, img_dotted  = create_train_mask(image_id)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = filter_by_area(cnts)
    for c in cnts:
        out.append(cv2.minEnclosingCircle(c))
    return out, img, img_dotted


def find_closest_marker_color(p, colors):
    colors = np.array(colors)
    p = np.array(p)
    err = np.sum(np.abs(colors - p[None, :]), axis=1)
    return colors[np.argmin(err)]


# In[ ]:


image_id = IMAGE_IDS[0] 
markers, img, img_dotted = find_markers(image_id)

print("{} markers found".format(len(markers)))
print("{} sea lions".format(get_n_animals(image_id)))

for i, marker in enumerate(markers):
    (x, y), r = marker
    x = int(x)
    y = int(y)    
    p = np.mean(img_dotted[y-1:y+1, x-1:x+1, :],axis=(0, 1))
    
    roi = [x - 50, y - 50, x + 50, y + 50]
    plt_st(12, 6)
    plt.subplot(121)
    display_img(img, roi=roi)
    plt.subplot(122)
    display_img(img_dotted, roi=roi)

    c = find_closest_marker_color(p, list(MARKER_COLORS.keys() ) ) 
    plt.suptitle("{}".format(MARKER_COLORS[tuple(c)]) )
    
    if i == 10:
        break


# In[ ]:


image_id = IMAGE_IDS[1] 
markers, img, img_dotted = find_markers(image_id)

print("{} markers found".format(len(markers)))
print("{} sea lions".format(get_n_animals(image_id)))

for i, marker in enumerate(markers):
    (x, y), r = marker
    x = int(x)
    y = int(y)    
    p = np.mean(img_dotted[y-1:y+1, x-1:x+1, :],axis=(0, 1))
    
    roi = [x - 50, y - 50, x + 50, y + 50]
    plt_st(12, 6)
    plt.subplot(121)
    display_img(img, roi=roi)
    plt.subplot(122)
    display_img(img_dotted, roi=roi)

    c = find_closest_marker_color(p, list(MARKER_COLORS.keys() ) ) 
    plt.suptitle("{}".format(MARKER_COLORS[tuple(c)]) )
    
    if i == 10:
        break


# In[ ]:


help(cv2.Canny)


# In[ ]:


def compute_edges(img):    
    edges = cv2.Canny(img, 170, 210)
    return edges


# In[ ]:


image_id = IMAGE_IDS[1] 
img = get_image_data(image_id)
edges = compute_edges(img)


# In[ ]:


help(cv2.Laplacian)


# In[ ]:


def get_roi(img, x, y, size):
    img_roi = img[y-size:y+size, x-size:x+size, :] if len(img.shape) == 3 else img[y-size:y+size, x-size:x+size]
    return img_roi

proc = img
proc = cv2.Laplacian(proc, cv2.CV_8U, ksize=1)
proc = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)

x = 3619
y = 3609
roi_size = 50
img_roi = get_roi(proc , x, y, roi_size)
shape = img_roi.shape
mask = np.zeros((shape[0] + 2, shape[1] + 2), dtype=np.uint8)
mask[1:-1,1:-1] = get_roi(edges, x, y, roi_size)


_, img2, mask2, _ = cv2.floodFill(img_roi, mask, 
                                  seedPoint=(roi_size-1, roi_size-1), 
                                  newVal=255,
                                  loDiff=(50, 50, 50), 
                                  upDiff=(50, 50, 50), 
                                  flags= 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | 255 << 8)


# In[ ]:


plt_st(12, 6)
plt.subplot(121)
display_img(img_roi)
plt.subplot(122)
display_img(mask2)


# In[ ]:


def segment_sea_lion(img, seed_point):
    pass


# In[ ]:


image_id = IMAGE_IDS[1] 
markers, img, img_dotted = find_markers(image_id)

print("{} markers found".format(len(markers)))
print("{} sea lions".format(get_n_animals(image_id)))

for i, marker in enumerate(markers):
    (x, y), r = marker
    x = int(x)
    y = int(y)    
    p = np.mean(img_dotted[y-1:y+1, x-1:x+1, :],axis=(0, 1))
    
    edges = compute_edges(img)
    
    roi = [x - 50, y - 50, x + 50, y + 50]
    print(roi, (x, y))
    plt_st(12, 6)
    plt.subplot(131)
    display_img(img, roi=roi)
    plt.subplot(132)
    display_img(img_dotted, roi=roi)
    plt.subplot(133)
    display_img(img + 255 * edges[:,:,None], roi=roi)

    c = find_closest_marker_color(p, list(MARKER_COLORS.keys() ) ) 
    plt.suptitle("{}".format(MARKER_COLORS[tuple(c)]) )
    break
    if i == 10:
        break


# In[ ]:





# In[ ]:




