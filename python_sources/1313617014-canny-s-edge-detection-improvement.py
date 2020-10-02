#!/usr/bin/env python
# coding: utf-8

# # Canny's Edge Detection Improvement
# John F. Canny's edge detection algorithm has been handy to detect edges in images. But, there is still some room to improve it.
# 
# _Muhammad Aufi Rayesa Frandhana
# OSK110 - Computer Science of Jakarta State University
# 1313617014_

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.lib.stride_tricks import as_strided


# Convolution function we made in class was the fastest. Although the fourier transform on continous convolution is proven faster, I was not able to comprehend how it was done. So, I'm still using the current convolution function.

# In[ ]:


def get_all_window(M, w):
    M = np.pad(M, w//2, 'symmetric')
    sub_shape = (w, w)
    view_shape = tuple(np.subtract(M.shape, sub_shape) + 1) + sub_shape
    arr_view = as_strided(M, view_shape, M.strides * 2)
    arr_view = arr_view.reshape((-1,) + sub_shape)
    return arr_view

def fastest_convolution_2d(im, K):
    w, _ = K.shape
    m,n = im.shape
    im_all_subw = get_all_window(im, w)
    X = np.sum(np.sum(im_all_subw * K, 1), 1)
    return X.reshape(m,n)


# Also some of the gradient kernels

# In[ ]:


grad_operators = {
    'prewitt':(
        np.array([[-1,0,1],
                  [-1,0,2],
                  [-1,0,1]]),
        np.array([[1,1,1],
                  [0,0,0],
                  [-1,-1,-1]])
    ),
    'robertcross':(
        np.array([[1,0],
                  [0,1]]),
        np.array([[0,-1],
                  [-1,0]]) 
    ),
    'sobelfeldman':(
        np.array([[-3,0,3],
                  [-10,0,10],
                  [-3,0,3]]),
        np.array([[3,10,3],
                  [0,0,0],
                  [-3,-10,-3]])
    ),
    'scharr':(
        np.array([[47,0,-47],
                  [162,0,-162],
                  [47,0,-47]]),
        np.array([[47,162,47],
                  [0,0,0],
                  [-47,-162,-47]])
    ),
    'sobel':(
        np.array([[-1,0,1],
                  [-2,0,2],
                  [-1,0,1]]),
        np.array([[1,2,1],
                  [0,0,0],
                  [-1,-2,-1]])
    )
}


# Also lest we forget we need these bad bois for the gradient thingy

# In[ ]:


def join_gradient_euclidean(Gx, Gy):
    return np.sqrt((Gx**2)+(Gy**2))

def join_gradient_manhattan(Gx, Gy):
    return np.abs(Gx)+np.abs(Gy)


# ## Preprocessing
# All images can be processed whatever the value (0..1 or 0..255) or color (Binary, Grayscale, RGB, or RGBA). But for the sake of the exeriment time, I decided to keep the preprocessing goes as usual. The image is quanitified to 0..255 grayscale. Although I think increasing contrast would also improve it, i didn't do it as it would have sharpen the image, thus more noise and made the gaussian blurring obsolete.

# In[ ]:


path = '../input/meanshiftimgs/mean_shift_image.png'
if path[-4:]=='.png':
    im = np.round(mpimg.imread(path)*255).astype(int)
elif path[-4:]=='.jpg':
    im = mpimg.imread(path)
w = 11
sigma = 1.4
imgrey = np.round(0.3 * im[:,:,0] + 0.59 * im[:,:,1] + 0.11 * im[:,:,2]).astype(int)

f = plt.figure(figsize=(18,15))
f.add_subplot(121).imshow(im)
f.add_subplot(122).imshow(imgrey, cmap='gray')


# No changes done in this part. I tried to edit `w` param. Instead of previously `11`, I made it to `15` because adding more context to the kernel would be better, but with `sigma` of `1.4`, `11` is just a good number. Changing the `sigma` won't cause any better result as higher value would erase and widen the actual edge, and lower would cause noise to keep appearing.

# ## Smoothening
# It was said that adaptive filter works better as mentioned in a paper made by Bing Wang and Shaosheng Fan. The only tradeoff is we are doing n iterations and using euclidean distance formula to two gradient kernels (which means the gradient kernel is implemented twice. To be precise, n+1 times).
# 
# Let's try with conventional gaussian filter first

# In[ ]:


def gaussian_kernel2d(w, sigma):
    w = w + (w % 2 == 0)
    F = np.zeros([w,w])
    mid = w//2 
    k = np.arange(w) - mid
    denom = 2*np.pi*sigma**2
    for i in k:
        for j in k:
            par = (i**2 + j**2)/(2*sigma**2)
            F[i + mid,j + mid] = np.exp(-par)/denom
    return F


# In[ ]:


gaussk = gaussian_kernel2d(w, sigma)
im_gauss = fastest_convolution_2d(imgrey, gaussk)

f = plt.figure(figsize=(18,15))
f.add_subplot(221, title='Original').imshow(imgrey, cmap='gray')
f.add_subplot(222, title='Result').imshow(im_gauss, cmap='gray')
f.add_subplot(153, title='Kernel').imshow(gaussk, cmap='hot')


# Now, the adaptive filter. For the adaptive filter, i'll use the Sobel operator and `n` = `5`.

# In[ ]:


def adaptive_weight(Gx, Gy, h):
    return np.exp(np.sqrt(join_gradient_euclidean(Gx, Gy) / (2 * (h ** 2))))

def adaptive_convolution_2d(im, weight):
    w = 3
    m,n = im.shape
    im_all_subw = get_all_window(im, w)
    weight_all_subw = get_all_window(weight, w)
    X = np.sum(np.sum(im_all_subw * weight_all_subw, 1), 1) / np.sum(np.sum(weight_all_subw, 1), 1)
    return X.reshape(m,n)

def adaptive_filter(im, n=5, h=1.5, operator='sobel'):
    # 1. K = 1, set the iteration n and the coefficient of the amplitude of the edge h.
    #K = 1
    #h = 1.5
    weights = []
    iteration_result = [im]
    
    # Iterative section
    for i in range(n):
        # 2. Calculate the gradient value Gx and Gy
        op = grad_operators[operator]
        Gx = fastest_convolution_2d(im,op[0])
        Gy = fastest_convolution_2d(im,op[1])

        # 3. Calculate the weight
        weight = adaptive_weight(Gx, Gy, h)
        weights.append(weight)

        # 4. Convolve
        im = adaptive_convolution_2d(im, weight)
        iteration_result.append(im)
        
    return im, weights, iteration_result


# In[ ]:


n_adaptive = 5
im_adaptive, adaptive_weights, adaptive_iteration_result = adaptive_filter(imgrey, n_adaptive)

f = plt.figure(figsize=(18,15))
f.add_subplot(221,title='Iteration 0').imshow(imgrey, cmap='gray')
f.add_subplot(222,title='Final').imshow(im_adaptive, cmap='gray')
for i in range(1, n_adaptive-1):
    f.add_subplot(4,n_adaptive-2,i+(n_adaptive-2)*2,title='Iteration ' + str(i)).imshow(adaptive_iteration_result[i], cmap='gray')


# It seems like the adaptive filter did its job pretty well -- It clears the noise but keep the actual edges preserved.

# ## Gradient Magnitudes
# I was experimenting with other operators than sobel, but since the adaptive filtering did its job, i'm going to keep sobel for these sections, because the point of changing operators is to keep relevant edges from noises. However, we'll still compare all operators (except for robert-cross operator, i don't understand on how to convolute even-sized kernels which is 2x2 in this case).

# In[ ]:


def find_gradient(im, operator='sobel'):
    Kx, Ky = grad_operators[operator]
    m,n = im.shape
    w = 3
    
    im_all_subw = get_all_window(im, w)
    Gx = np.sum(np.sum(im_all_subw * Kx, 1), 1).reshape(m,n)
    Gy = np.sum(np.sum(im_all_subw * Ky, 1), 1).reshape(m,n)
    theta = np.arctan2(np.abs(Gy), Gx)
    theta = theta*180/np.pi
    return Gx, Gy, theta


# In[ ]:


Gx, Gy, theta = find_gradient(im_gauss)
G = join_gradient_euclidean(Gx, Gy) # Euclidean because kaggle has free RAM and GPU yay

f = plt.figure(figsize=(18,15))
f.add_subplot(221,title='Gx').imshow(Gx, cmap='gray')
f.add_subplot(222,title='Gy').imshow(Gy, cmap='gray')
f.add_subplot(223,title='G').imshow(G, cmap='gray')
f.add_subplot(224,title='Theta').imshow(theta, cmap='viridis')


# Now let's see how other operators are preforming.

# In[ ]:


f = plt.figure(figsize=(18,15))

Gxt, Gyt, _ = find_gradient(im_gauss, 'prewitt')
f.add_subplot(221,title='Prewitt operator').imshow(join_gradient_euclidean(Gxt, Gyt), cmap='gray')

Gxt, Gyt, _ = find_gradient(im_gauss, 'sobelfeldman')
f.add_subplot(222,title='Sobel-Feldman operator').imshow(join_gradient_euclidean(Gxt, Gyt), cmap='gray')

Gxt, Gyt, _ = find_gradient(im_gauss, 'scharr')
f.add_subplot(212,title='Scharr operator').imshow(join_gradient_euclidean(Gxt, Gyt), cmap='gray')

f.show()


# ## Non-Maximum Suppression
# The only thing changed here is the _"round thetas to nearest 45 degree, cap on 135 degree"_. From this:
# ```python
# tn = np.array([0, 45, 90, 135, 180])
# fn = np.array([0, 45, 90, 135, 0])
# m,n = theta.shape
# ntheta = np.zeros([5,m,n])
# for i in range(len(tn)):
#     ntheta[i] = np.abs(theta - tn[i])
# t_id = np.argmin(ntheta, 0)
# ftheta = np.zeros([m,n])
# for i in range(len(tn)):
#     ftheta += (t_id==i).astype(int)*fn[i]
# ```
# To this:
# ```python
# (np.round(x / 45) * 45) % 180
# ```
# Lmao

# In[ ]:


def round_to_closest_n(x, n, clock=0):
    out = np.round(x / n) * n
    if clock:
        out %= clock
    return out

def non_maximum_suppression(im,theta):
    ntheta = round_to_closest_n(theta, 45, 180)
    thetafilters = np.array([
        [[0,0,0],[-1,2,-1],[0,0,0]],
        [[-1,0,0],[0,2,0],[0,0,-1]],
        [[0,-1,0],[0,2,0],[0,-1,0]],
        [[0,0,-1],[0,2,0],[-1,0,0]]])
    
    per_angle_res = [fastest_convolution_2d(im, thetafilters[i]) * (ntheta==(45*i)) for i in range(4)]
    return np.sum(per_angle_res,0), per_angle_res, ntheta


# In[ ]:


im_nms, in_nms_perkernel, ntheta = non_maximum_suppression(G,theta)
im_nms_pos = im_nms * (im_nms >= 0)

f = plt.figure(figsize=(18,24))
f.add_subplot(321,title='Degree 0').imshow(in_nms_perkernel[0], cmap='gray')
f.add_subplot(322,title='Degree 45').imshow(in_nms_perkernel[1], cmap='gray')
f.add_subplot(323,title='Degree 90').imshow(in_nms_perkernel[2], cmap='gray')
f.add_subplot(324,title='Degree 135').imshow(in_nms_perkernel[3], cmap='gray')
f.add_subplot(325,title='Result image').imshow(im_nms_pos, cmap='gray')
f.add_subplot(326,title='Quantified Theta').imshow(ntheta, cmap='viridis')


# ## Double Tresholding
# Double tresholding is one of the most easiest part to implement, but the trickiest onto finding the optimal range. While it is originally told the strong pixels are pixels greater than 80 pixel strength and ones are between 20 to 80 inclusive are weak ones, which implies the 1..255 range. The problem is, our pixel value will not have value between 0..255 range due to the previous sobel operation despite we enforced 0..255 scale in the beginning. See for yourself:

# In[ ]:


im_nms_pos_classes = np.unique(im_nms_pos)
print(f'{im_nms_pos_classes[0]}..{im_nms_pos_classes[-1]}')


# To overcome this, I decided instead of having `255` as the scale cap, I'll have `np.max(image)` as the scale cap, and convert the tresholds by dividing them by `255` and multiply it with `np.max(image)`. But it doesn't seem to do anything better even if i change them since optimal value differs on every image. This time, instead of using tresholds 20 and 80, i'm goig to use 10 and 60.

# In[ ]:


def double_tresholding(im, hi=80, lo=20):
    strong = im > hi
    weak = (im >= hi) == (im <= lo)
    return strong, weak


# In[ ]:


maxcap = np.max(im_nms_pos)
strong_a, weak_a = double_tresholding(im_nms_pos,(60*maxcap)/255,(10*maxcap)/255)

m,n = imgrey.shape
ta = np.zeros((m,n,3))
ta[:,:,0] = strong_a
ta[:,:,1] = (strong_a + weak_a) > 0
ta[:,:,2] = weak_a
plt.figure(figsize=(18,15)).add_subplot(111, title='Yellows are strong edges').imshow(ta,cmap='gray')


# Well, that worked. But, i think Otsu's method would help us finding the optimal values by looking at how the pixel strength spreads on the histogram. First we find the strong edge from Otsu's method, then the weak edge by simply multipying strong treshold by quarter to one.

# In[ ]:


def histogram(img):
    img = img.astype(np.float)
    row, col = img.shape
    maxstr = np.int(np.floor(np.max(img))+1)
    y = np.array([np.sum(np.bitwise_and(img>=pixstr-0.5,img<pixstr+0.5)) for pixstr in range(maxstr+1)])
    return y

def get_tresholds(hist, separate=False):
    pixels = np.sum(hist)
    mx = hist.size
    tres = []
    
    for i in range(1, mx):
        le = hist[0:i]
        ri = hist[i:mx]
        
        vb = np.var(le)
        wb = np.sum(le) / pixels
        mb = np.mean(le)
        
        vf= np.var(ri)
        wf = np.sum(ri) / pixels
        mf = np.mean(ri)
        
        V2w = wb * (vb) + wf * (vf)
        V2b = wb * wf * (mb - mf)**2
        
        if not np.isnan(V2w): tres.append((i,V2w))
            
    if separate:
        return list(zip(*tres))
    return tres


# In[ ]:


hist = histogram(im_nms_pos)
pixstr, pixtres = get_tresholds(hist, True)
maxstr = np.int(np.floor(maxcap)+1)

up_tres = pixstr[pixtres.index(max(pixtres))]
lo_tres = up_tres * 0.75

print(f'Upper treshold = {up_tres} , Lower treshold = {lo_tres}')

f = plt.figure(figsize=(18,6))
f.add_subplot(121, title='Histogram').bar(np.arange(0,maxstr+1), hist, color='b', width=5, align='center', alpha=0.25)
f.add_subplot(122, title='Inter-classgroup deviation').plot(pixstr,pixtres,up_tres,max(pixtres),'ro')
f.show()


# In[ ]:


strong_b, weak_b = double_tresholding(im_nms_pos,up_tres,lo_tres)

tb = np.zeros((m,n,3))
tb[:,:,0] = strong_b
tb[:,:,1] = (strong_b + weak_b) > 0
tb[:,:,2] = weak_b
plt.figure(figsize=(18,15)).add_subplot(111, title='Yellows are strong edges').imshow(tb,cmap='gray')


# ## Hysteresis
# Though it is not well-defined in the paper, it still explain the goal well -- All weak edges that directly connects to strong edges are kept into the final result, whilst the other weak edges are removed from existence. They mentioned the BLOB thing which being said would help us in this part. At first, I imagined the algorithm is just convoluting the strong edges then mask it with the union of strong and weak edges and then it became a new strong edge, then repeat until the convolution does not produce a different strong edge than before.

# In[ ]:


def hysteresis(strong, weak):
    # basically we're just repeating 8-neighbor convolutions
    K = np.array([
            [1,1,1],
            [1,0,1],
            [1,1,1]
        ])
    union = strong + weak
    blob = strong
    blobbefore = np.ones(union.shape)-blob
    #diff = np.ones(union.shape)
    while not np.all(blob==blobbefore):
        blobbefore = blob
        blob = np.bitwise_and((fastest_convolution_2d(blob,K)+strong)>0,union)
    return blob


# In[ ]:


blob_b = hysteresis(strong_b, weak_b)

edctb = np.zeros((m,n,3))
edctb[:,:,0] = blob_b
edctb[:,:,1] = strong_b
edctb[:,:,2] = ((strong_b.astype('u8') + weak_b.astype('u8')) - blob_b.astype('u8')) + strong_b.astype('u8')
plt.figure(figsize=(18,15)).add_subplot(111, title='Reds are grabbed as strong edges, blue are defective weak edges.').imshow(edctb,cmap='gray')


# In[ ]:


plt.figure(figsize=(18,15)).add_subplot(111, title='Final result').imshow(blob_b,cmap='gray')


# It seems using the double tresholding using Otsu's method is kind of noisy despite the procesing using loops for this image is surprisingly fast and all important or relevant edges are scanned. Let's try with the conventional double tresholding.

# In[ ]:


blob_a = hysteresis(strong_a, weak_a)

edcta = np.zeros((m,n,3))
edcta[:,:,0] = blob_a
edcta[:,:,1] = strong_a
edcta[:,:,2] = ((strong_a.astype('u8') + weak_a.astype('u8')) - blob_a.astype('u8')) + strong_a.astype('u8')
plt.figure(figsize=(18,15)).add_subplot(111, title='Reds are grabbed as strong edges, blue are defective weak edges.').imshow(edcta,cmap='gray')


# In[ ]:


plt.figure(figsize=(18,15)).add_subplot(111, title='Final result').imshow(blob_a,cmap='gray')


# It takes a really long time to do it with loops for the image given by conventional double tresholding because it has to aggregate in a really long path, and also there is some edges that is lost in translation. The only good thing is using this image gives us more relevant strong edges though not all of them.
# 
# There is still one more alternative, we modify our kernel to have negative value in the center, so the unconnected lines should be yeeted away easily by filtering it with `arr > 0`. And also, only one convolution so it should be fast. The tradeoff is, even unconnected weak edges could have been considered connected using this method.
# 
# We'll keep using the conventional double tresholding result as an input image.

# In[ ]:


def hysteresis_fast(strong, weak):
    union = (strong + weak) > 0
    K = np.array([
            [1,1,1],
            [1,-4,1],
            [1,1,1]
        ])
    return np.bitwise_and(fastest_convolution_2d(union,K)>=0,union)


# In[ ]:


fin_a = hysteresis_fast(strong_a, weak_a)

edctfa = np.zeros((m,n,3))
edctfa[:,:,0] = fin_a
edctfa[:,:,1] = strong_a
edctfa[:,:,2] = ((strong_a.astype('u8') + weak_a.astype('u8')) - fin_a.astype('u8')) + strong_a.astype('u8')
plt.figure(figsize=(18,15)).add_subplot(111, title='Reds are grabbed as strong edges, blue are defective weak edges.').imshow(edctfa,cmap='gray')


# In[ ]:


plt.figure(figsize=(18,15)).add_subplot(111, title='Final result').imshow(fin_a,cmap='gray')


# As what have we witnessed from the demonstration, the runtime is incredibly fast, but with a tradeoff that unconnected weak edge can also be considered strong.

# # Conclusion
# There are several possible modifications on the Canny edge detection algorithm.
# 
# ### Preprocessing
# * Grayscale image is already a good representation of RGB image and also more memory-safe since we're only working on one channel instead of three channels. Using RGB means we're also convolving in 3D, which we had no references. Although we may just process each channels separately, it will be genuinely confusing when we're about to mix them together.
# * Prior to grayscaling, contrasting the image would be a choice to improve because it emphasizes the difference between two pixels that have different strengths. However, this comes with a trade that noise may appear in the most unexpected place like in the middle of a pool of pixels with the same strengths as contrasting the image has the same implication as sharpening the image.
# 
# ### Smoothening
# * We used Gaussian filter for our blurring process to erase noise. This comes with a price that our edge may get wider and even irrelevant because the more we blur it the more smaller the difference between pixels are.
# * The problem found when using Gaussian filter can be overcomed by by using Adaptive filter that blurs powerfully the place where the strength difference to their neighbors are small, but blurs weakly the place where the strength difference to their neighbors are humongous. However, this may cause unintentional lines appear as contours around the actual edges.
# * Both filters can also be modified to have bigger size. Adaptive filter has no size constraint. Gaussian filter, however, if we make a filter that big but the sigma is pretty much small, it won't do anything unless you also make the sigma bigger.
# 
# ### Gradient Magnitudes
# * There are a lot of variations of gradient operators, and Sobel is one of the most common one.
#   * Prewitt operator is the most basic operators that does not add more weight to the adjacent neighbors (North, East, South, West), so no discrimination where the source of elevation/depression comes from.
#   * Sobel operator is the operator that values the adjacent neighbors. 3x3 is the most popular, though this can be resized to the bigger size (e.g. 5x5 or 7x7) for better noise-evading. Sobel was chosen because this yields one of the most reasonable result ever.
#   * Sobel-Feldman is the operator with more optmum number in detecting edges. But also this means this detect noises too.
#   * Scharr takes sobel operator way too far, but at least he said it is the most optimum, ever. The tradeoff is the pixel strength of the result image became unnecessarily enormous.
#   * Instead of considering the north-south and east-west neighbor, Robert-Cross takes it northeast-southwest and northwest-southeast.
# * Using Euclidean Distance to merge `Gx` and `Gy` is more preferable if you have sufficient computing power (or patience) than using Manhattan Distance because it represents actual gradient mathematically and properly.
# 
# ### Non-Maximum Suppression
# * Instead of using arrays and certain loops, theta rounding to certain degrees can be done with a single yet simple mathematical formula, which is dividing the theta by the multiple, round it, multiply it back with the multiple, then get the remainder from the cap.
# 
# ### Double Tresholding
# * The conventional double tresholding assumes the image value is between 0..255 inclusive. The original paper gave us 80 and 20 as recommended high and low pixel strength tresholds respectively. However, this doesn't seem to work for images with values bigger than the assumed range due to the result of the previous non-maximum suppression convolution. To rescale the recommended high and low pixel strength tresholds, we could just simply divide the current treshold by 255 then multiply it with the maximum value of current image.
# * But that won't work all times because the strong edge will be no bigger than a small clump of pixels and there is a lot of edges lost in translation, so trying to play outside of the recommended values would give a change. I tried 60 and 10 and i feel this is the optimum value for current experiment.
# * however, different images can have different range or spread of values, so we might need an algorithm that will adapt by itself, which is Otsu's method. Imagine a histogram of images divided into two parts along the class frequency axes. Iterate the two part's separator from the first inter-class space to the last one, and we count the deviation between the two groups. The pixel strength where the separator belongs that produce the largest deviation would be higher treshold. The smaller treshold will just have the half of the high treshold, but i used a quarter to one times of the high treshold.
# * Conventional double tresholding is better if we know how the image input's nature and we want to evade noise. The Otsu's method didn't have that, but it is better at adapting various ranges of the image.
# 
# ### Hysteresis
# * Of course, convolving strong edges with 8-neighbor kernel with zero weight in the middle then masking it with the union of strong and weak edge so it became a new strong, and doing all of them all over again using iterations, which looks like a virus spreading from strong edges infecting weak edges and turning them to strong edges but only a pixel far at an iteration, is gruesome, but it works perfect as intended.
# * To evade such iterations, we actually can just convolve once using 8-neighbor kernel with negative weight in the middle. After removing negative values on the result, we'd get the final image. However, this is risky as weak edges that does not even connect, not just directly but even gapped far away, to strong edges has a chance to become strong edges.
# * I myself prefer using iterations because it works as intended and computing power at kaggle is termendous and free.
