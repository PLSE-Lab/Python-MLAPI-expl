#!/usr/bin/env python
# coding: utf-8

# ## Exploring the generating of pseudo random numbers from a video stream

# I somehow came across a Kaggle contest from 6 years ago named "[The Random Number Grand Challenge](https://www.kaggle.com/c/random-number-grand-challenge)". I started looking at the dataset when I remembered a video I had seen a few years back named "[The Lava Lamps That Help Keep The Internet Secure](https://www.youtube.com/watch?v=1cUUfMeOijg)". That video describes how Cloudflare, in San Francisco, uses a video stream of a wall of lava lamps, called "the Entropy Wall", to feed an algorithm which generates random numbers to provide SSL encryption for websites. I wondered back then about what algorithm could transform the random motion of the lamps into numbers... well, not just any numbers, they must be within a certain range, they must be uniformly distributed, and there must be no identifiable relationship between a generated number and any that preceded it. To sumarize, the random numbers should have the three properties:
# 1. be in the half-open interval \[0.0, 1.0\)
# 2. be uniformly distributed in the interval
# 3. appear truly random to an adversary
# 
# Disclaimer: I don't know much about random numbers, at least not to the degree that some mathematicians and cryptographers understand it. I just got highly motivated to see what I could learn on my own at midnight when I found I couldn't sleep :)
# 
# Wondering how well I could do with just any ol' video stream, I decided to use, as my entropy source, a video I had on my system. I still didn't know how to turn each 8-bit, RGB image into a single number. Finally I decided to flatten each image, comparing the current flattened image with the previous flattened image. I knew, from a while back at work, that one way to compare two 1D distributions is with "The first Wasserstein distance" and that SciPy has an implementation in its [stats module](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html).
# 
# I extraced a 2 minute clip from [KOS](https://www.netflix.com/title/70301578/) to use as my input (since I don't have a live video camera handy). I used Scikit-Video to extract the images from the video.

# Below is a plot of the Wasserstein Distances (WD) between successive frames of two minute clip (at 24 fps).

# ![wd_between_frames.png](attachment:wd_between_frames.png)

# The very large values, in the plot are when the scene changes. For example, two people might be talking and the camera angle does not change so that very little scene content changes, maybe the speaker's mouth moves slightly, but when the camera angle changes to show another person, then there is a great change in pixel values for most, or all of the image. Below is an example of such a scene change.

# ![scene_change_01.png](attachment:scene_change_01.png)

# Here is another.

# ![scene_change_3.png](attachment:scene_change_3.png)

# This might indeed yield some unpredictable values, maybe even random, but the WD value does not fulfill the 3 requirements for a random number. It might be said to fulfill number 3, but it certainly does not meet requirements 1 and 2. How then can we make this number be in the desired interval and fall within a uniform, temporal distribution?

# The line below seems to do the job pretty well.  
# ```rnd = modf(log10(1/wd))[0] if wd > 0 else 0```  
# Let's examine the different pieces. Firstly, its a Python ternary conditional expression of the form ```'true' if True else 'false'``` , so if the condition on the left side of the IF is true, then the variable rnd is equal to modf(log10(1/wd))\[0\], otherwise, rnd is set equal to the right side of the IF, which in this case is zero. The variable wd is the result of the Wasserstein Distance calculation plotted above. The Python math.modf function returns the fractional and integer parts of a floating-point number in a list. Notice that the \[0\] after the modf() means that we keep only the fractional part. 

# Below is a program which I ran on my own system (not this notebook because of a missing scikit-video dependency).

# In[ ]:


from math import modf, log10
import numpy as np
from scipy.stats import wasserstein_distance
import skvideo.io
from skimage.transform import resize
import custom_plots  

infile = "./data/clip_2_min.mp4"
#infile = "./data/downtown.mp4"
videodata = skvideo.io.vread(infile)
print(videodata.shape, videodata.dtype) 
N_frames, rows, cols, channels = videodata.shape
rr, cc = (rows//2, cols//2)
FIRST_FRAME = True
wass_list = []
rand_list = []
#for iframe in range(N_frames):
for iframe in range(N_frames):
    imgnumb = str(iframe).zfill(4)
    Img = videodata[iframe,:,:,:]

    # reduce the size of the image to increase compute speed
    Img = resize(Img, (rr,cc))
    
    # uncomment to display each frame
    #custom_plots.show_img(Img, imgnumb, pause=False)

    if not FIRST_FRAME:
        # Compute the first Wasserstein distance between two 1D distributions.
        u_img = Img.reshape(rr*cc*channels) # distribution 1
        v_img = Img_prev.reshape(rr*cc*channels) # distribution 2
        wd = wasserstein_distance(u_img, v_img)
        # generate a random float in the half-open interval [0.0, 1.0) based on wd
        rnd = modf(log10(1/wd))[0] if wd > 0 else 0
        if rnd > 0:
            rand_list.append(rnd)
            wass_list.append(wd)
            print(imgnumb, wd, rnd)

    Img_prev = Img
    FIRST_FRAME = False


# Histogram - are these numbers uniformly distributed? 
import matplotlib.pyplot as plt
num_bins = 100
fig, ax = plt.subplots(figsize=(6.5,5))
x = np.array(rand_list)
n, bins, patches = ax.hist(x, num_bins, color='darkgreen')
ax.set_xlabel('number')
ax.set_ylabel('# of instances')
ax.set_title('RNG distribution', fontname='Hack')
#plt.show()
plt.savefig('RNG distribution', bbox_inches='tight')


#Wasserstein Distance vs. Frame Number
x = np.arange(len(wass_list))
fig, ax = plt.subplots()
ax.plot(x, rand_list, '.', color='darkorange', label = 'number based on W.D.')
ax.plot(x, wass_list, '-', color='darkblue', label = 'Wasserstein distance')
ax.set_xlabel('frame number')
ax.set_ylabel('value')
legend = ax.legend(loc='upper right', shadow=True)
plt.title('Uniform-ish, Pseudo-Random Numbers from Video Feed')
#plt.show()


# It's commented out in the code above, but the "custom_plots" module is useful for displaying each frame in the main loop. It would be too slow to use much, but is useful for seeing the WD values change with the changes in the image content. The code for this module is given below (commented out so it doesn't run in the notebook).

# ```
# import matplotlib.pyplot as plt
# import warnings
# 
# warnings.filterwarnings("ignore", category=UserWarning)
# plt.style.use('dark_background')
# # https://matplotlib.org/examples/color/named_colors.html
# colors = {1:'deepskyblue', 2:'lightskyblue', 3:'limegreen', 4:'orangered',
#           5:'red', 6:'gold', 7:'snow', 8:'black'}
# 
# spines = ['top','bottom','left','right']
# fc='black'
# cm = 'gray'
# theme_col = colors[8]
# ticks_col = colors[6]
# title_col = colors[6]
# 
# fig, ax = plt.subplots(1, 1, figsize=(6.25,5)) # display single image
# 
# fig.patch.set_facecolor(fc)
#     
#     
# def show_img(img, name, cm=cm, pause=True):
#     plt.ion()
#     plt.imshow(img, cmap=cm)
#     plt.show()
#     plt.yticks(fontname="Hack")
#     plt.xticks(fontname="Hack")
#     for sp in spines:
#         ax.spines[sp].set_color(theme_col)
#         #ax.spines[sp].set_linewidth(2)
#     ax.tick_params(axis='x', colors=ticks_col)
#     ax.tick_params(axis='y', colors=ticks_col)
#     ax.yaxis.label.set_color(theme_col)
#     ax.xaxis.label.set_color(theme_col)
#     if type(name)==int:
#         ax.set_title('frame '+str(name), fontname='Hack')
#     else:
#         ax.set_title(name, fontname='Hack')
#     ax.title.set_color(title_col)
#     
#     plt.tight_layout()
#     plt.draw()
#     if pause:
#         input('')
#     plt.pause(0.0001)
#     plt.cla()
# ```

# A 5-minute clip was input to the program above and the two plots output The first plot is a 100-bins histogram of the "hopefully" randomly generated numbers. The second plot is these values vs. frame number.  

# ![5min_RNG%20distribution.png](attachment:5min_RNG%20distribution.png)

# Hmmmm, not exactly uniform looking. If it were truly uniform, then each bin would be roughly the same height. This skew from uniform must reflect, somehow, the large-scale, changing content of the images. THB, it doesn't look *that* bad though.  

# ![wd_and_rand.png](attachment:wd_and_rand.png)

# Looks mostly uniform but there is some bunching-up of points which worries me a bit. Is this good enough to use for encryption? How can one tell? I'd love to see such a plot for Cloudfare's RNG just to know how far off I am.
# 
# One easy test I can do is to use numbers from this generator in a monte carlo estimation, such as [this one](https://www.geeksforgeeks.org/estimating-value-pi-using-monte-carlo/) where they are used to estimate the digits of pi. 
# 
# Does anyone reading this have any knowledge in this domain that they'd like to share? Please let me know. In the meantime, I'd better try to get some sleep!

# In[ ]:




