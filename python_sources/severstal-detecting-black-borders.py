#!/usr/bin/env python
# coding: utf-8

# **About this kernel:**
# --------------
# * This kernel was build to investigate wheter a model can succesfully detect and remove black borders in an image
# * **Step 1:** Create a method to detect black borders in an image
# * **Step 2:** Test the method on our dataset
# * **Step 3:** Compare the predictions to the predictions of the kernel [Severstal: fast.ai 256x256 crops (sub)](https://www.kaggle.com/iafoss/severstal-fast-ai-256x256-crops-sub) of [@iafoss](https://www.kaggle.com/iafoss)
# ---------------------

# The method I use was published in this paper: https://onlinelibrary.wiley.com/doi/pdf/10.1002/srin.201600068?casa_token=oz-tLM6kIB8AAAAA:-WkgQwagLClCpo__XMx2MgbUZNJyeBOgC3840cOYTgQO3lNnOEF2rCmrCPXovO-qxgzzzC41lhY72OA
# 
# **The main steps are:**
# * Detect wheter the image contains a black border or not as well as the side it is on, comparing the mean of the right and left side of the image to a threshold
# * Build three lines at 1/4, 1/2 and 3/4 of the image height and use a sliding window to find the x-axis value of the border for each line
# * Build a first order polynomial through the closest x-axis values
# * Based on the side of the border and the polynomial, remove all pixels that belong to a black border
# 
# **Changes:**
# * I compare only the mean of 1/8 of the right and 1/8 of the left side of the image to a threshold, since a border can only appear on one of the sides
# * Since three lines seemed to be pretty unstable, I use the sum of all lines to 1/4 of the image height for the first line, all lines form 1/4 to 3/4 for the second and all lines from 3/4 to the end for the third line.
# 
# **Notes:**
# * I have tried different thresholds and window sizes and t=15, w=100 seem to work well. The threshold is now roughly 1.5 standard deviations away from the mean.
# * ** Feel free to use this method on your models, share your results or give me feedback for this kernel**
# -----------------

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'import cv2\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom tqdm import tqdm')


# **Methods:**
# ------------

# In[ ]:


# taken from https://www.kaggle.com/xhlulu/severstal-simple-keras-u-net-boilerplate
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(mask_rle, shape=(256,1600)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def build_masks(rles, input_shape):
    depth = len(rles)
    height, width = input_shape
    masks = np.zeros((height, width, depth))
    
    for i, rle in enumerate(rles):
        if type(rle) is str:
            masks[:, :, i] = rle2mask(rle, (width, height))
    
    return masks

def build_rles(masks):
    width, height, depth = masks.shape
    
    rles = [mask2rle(masks[:, :, i])
            for i in range(depth)]
    
    return rles


# In[ ]:


def boundary_detection(image, old_mask, t=15, window_size=100):
    gl = np.mean(image[:,:int(image.shape[1]/8)])
    gr = np.mean(image[:,int(image.shape[1]*7/8):])
    if gl<=t and gr>t:
        mode = 'left'
    elif gl>t and gr<=t:
        mode = 'right'
    elif gl>t and gr>t:
        return old_mask
    elif gl<=t and gr<=t:
        return np.zeros(image.shape)
    
    #img_row_sum = np.sum(image, axis=0)
    l0 = np.sum(image[:int(image.shape[0]/4)], axis=0)
    l1 = np.sum(image[int(image.shape[0]/4):int(image.shape[0]/4*2)], axis=0)
    l2 = np.sum(image[int(image.shape[0]/4/2):int(image.shape[0]/4*3)], axis=0)
    
    b0 = np.argmax([abs(sum(l0[i-window_size:i])-sum(l0[i+1:window_size+i+1])) for i in range(window_size,int((len(l0)-window_size*2)))]) + window_size
    b1 = np.argmax([abs(sum(l1[i-window_size:i])-sum(l1[i+1:window_size+i+1])) for i in range(window_size,int((len(l1)-window_size*2)))]) + window_size
    b2 = np.argmax([abs(sum(l2[i-window_size:i])-sum(l2[i+1:window_size+i+1])) for i in range(window_size,int((len(l2)-window_size*2)))]) + window_size
    d01 = np.linalg.norm(np.array([b0,int(image.shape[0]/4)])-np.array([b1,int(image.shape[0]/4*2)]))
    d12 = np.linalg.norm(np.array([b1,int(image.shape[0]/4*2)])-np.array([b2,int(image.shape[0]/4*3)]))
    
    if np.argmin([d01,d12]) == 0:
        coefficients = np.polyfit(np.array([int(image.shape[0]/4),int(image.shape[0]/4*2)]),np.array([b0,b1]), 1)
    elif np.argmin([d01,d12]) == 1:
        coefficients = np.polyfit(np.array([int(image.shape[0]/4*2),int(image.shape[0]/4*3)]),np.array([b1,b2]), 1)
    polynomial = np.poly1d(coefficients)
    
    x_axis = [int(polynomial[0]+polynomial[1]*i) for i in range(image.shape[0])]
    mask = []
    if mode == 'left':
        for i in range(image.shape[0]):
            mask.append([old_mask[i][j] if x_axis[i]<=j else 0 for j in range(image.shape[1])])
    elif mode == 'right':
        for i in range(image.shape[0]):
            mask.append([old_mask[i][j] if x_axis[i]>=j else 0 for j in range(image.shape[1])])
    return mask


# -----------------

# **Evaluation:**
# ----------------
# * Read the original dataset
# * Detect boundaries using the method above
# * Visualize detected boundaries

# In[ ]:


submission_df = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')
submission_df['ImageId'] = submission_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
submission_df['ClassId'] = submission_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
submission_df = submission_df.fillna('')
print(submission_df.shape)
submission_df.head()


# In[ ]:


path = '../input/severstal-steel-defect-detection/test_images/'
encoded = []
for i, filename in enumerate(tqdm(submission_df['ImageId'].unique()[0:100])):
    img_path = f"{path}/{filename}"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = boundary_detection(img,np.ones(img.shape),15,100)
    rle = mask2rle(np.array(mask))
    encoded.append(rle)


# In[ ]:


for i, filename in enumerate(submission_df['ImageId'].unique()[0:100]):
    img_path = f"{path}/{filename}"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_mask = rle2mask(encoded[i],(1600,256)).T
    plt.figure(figsize=(25, 2))
    plt.subplot(131)
    plt.imshow(img, 'gray',vmin=0,vmax=255)
    img_row_sum = np.sum(img,axis=0).tolist()
    plt.subplot(132)
    plt.plot(img_row_sum)
    plt.subplot(133)
    plt.imshow(img, 'gray')
    plt.imshow(np.array(new_mask)*255, 'brg', alpha=0.25, vmin=0,vmax=255)
    plt.show()


# -----------------

# **Inspect a real model:**
# ---------------
# * Load predictions of the kernel [Severstal: fast.ai 256x256 crops (sub)](https://www.kaggle.com/iafoss/severstal-fast-ai-256x256-crops-sub) of [@iafoss](https://www.kaggle.com/iafoss)
# * Detect differences between the predictions of [@iafoss] and our method
# * Visualize the differences

# 

# In[ ]:


submission_df = pd.read_csv('../input/severstal-fast-ai-256x256-crops-sub/submission.csv')
submission_df['ImageId'] = submission_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
submission_df['ClassId'] = submission_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
submission_df = submission_df.fillna('')
print(submission_df.shape)
submission_df.head()


# In[ ]:


path = '../input/severstal-steel-defect-detection/test_images/'
encoded = []
for i, filename in enumerate(tqdm(submission_df['ImageId'])):
    if submission_df.iloc[i]['EncodedPixels']!='':
        img_path = f"{path}/{filename}"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        old_mask = rle2mask(submission_df.iloc[i]['EncodedPixels'],(1600,256)).T
        mask = boundary_detection(img,old_mask,15,100)
        rle = mask2rle(np.array(mask))
        encoded.append(rle)
    else:
        encoded.append('')


# In[ ]:


submission_df['EncodedPixels_New'] = encoded
diff_df = submission_df[submission_df['EncodedPixels_New']!=submission_df['EncodedPixels']]


# In[ ]:


for i, filename in enumerate(diff_df['ImageId'].unique()):
    img_path = f"{path}/{filename}"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    old_mask = rle2mask(diff_df.iloc[i]['EncodedPixels'],(1600,256)).T
    new_mask = rle2mask(diff_df.iloc[i]['EncodedPixels_New'],(1600,256)).T
    plt.figure(figsize=(25, 2))
    plt.subplot(151)
    plt.imshow(img, 'gray',vmin=0,vmax=255)
    img_row_sum = np.sum(img,axis=0).tolist()
    plt.subplot(152)
    plt.plot(img_row_sum)
    plt.subplot(153)
    plt.imshow(img, 'gray')
    plt.imshow(np.array(old_mask)*255, 'brg', alpha=0.25)
    plt.subplot(154)
    plt.imshow(img, 'gray')
    plt.imshow(np.array(new_mask)*255, 'brg', alpha=0.25)
    plt.subplot(155)
    plt.imshow(img, 'gray')
    plt.imshow(np.array(old_mask)*255-np.array(new_mask)*255, 'brg', alpha=0.25)
    plt.show()


# ------------------

# **Insights:**
# ---------------
# * The implemented method can successfully detect black borders in the dataset
# * The model of [@iafoss](https://www.kaggle.com/iafoss) has no problem detecting black borders
