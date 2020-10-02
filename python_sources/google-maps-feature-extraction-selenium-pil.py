#!/usr/bin/env python
# coding: utf-8

# I am a running and wanted to find the best route in my area.  I my mind the best route is the most beautiful.  This means lots of greenspace and water. The issue is that there is not real map data that will let me quantify if an area is nice or not.  I could go to Google Maps and determine if a route might look ok, but again this is just a visual check.  To solve this I decided to extract features from Google Maps using Python, Selenium and Pillow (image lib).
# 
# As a part one I just want to see if I can automatically get a snapshot of Google Maps and extract feature information.  Feature information in this case is a rough approximation of proximity to parks and water. (More on this later)
# 
# The heavy lifting is done by Selenium which is a package that can mimic someone actually browsing a website.  
# 
# First we need to install a browser (Firefox) and required drivers needed for using Selenium.  That in addition to some packages we need.

# In[ ]:


# Install

get_ipython().system('mkdir "../working/firefox"')
get_ipython().system('cp -a "../input/firefox-63.0.3.tar.bz2/firefox/." "../working/firefox"')
get_ipython().system('chmod -R 777 "../working/firefox"')
get_ipython().system('pip install webdriverdownloader')
from webdriverdownloader import GeckoDriverDownloader
gdd = GeckoDriverDownloader()
gdd.download_and_install("v0.23.0")
get_ipython().system('pip install selenium')
get_ipython().system('apt-get install -y libgtk-3-0 libdbus-glib-1-2 xvfb')
get_ipython().system('export DISPLAY=:99')


# Now that everything is installed we can set up an instance of the browser which will live as a background process.

# In[ ]:


# Setup for browser
from selenium import webdriver as selenium_webdriver
from selenium.webdriver.firefox.options import Options as selenium_options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities as selenium_DesiredCapabilities
browser_options = selenium_options()
browser_options.add_argument("--headless")
browser_options.add_argument("--window-size=1920,1080")
# browser_options.add_argument("user-agent=whatever you want")
capabilities_argument = selenium_DesiredCapabilities().FIREFOX
capabilities_argument["marionette"] = True
browser = selenium_webdriver.Firefox(
    options=browser_options,
    firefox_binary="../working/firefox/firefox",
    capabilities=capabilities_argument
)


# As my first step lets set up the location I would like to work on.  How about the Toronto Waterfront?
# 
# 

# In[ ]:


# Get the screenshot from google for a given location
lat = 43.640722
lng = -79.3811892
z = 17 # Height
import numpy


# Next  I will set up my URL to google maps and "browse"

# In[ ]:


url = 'https://www.google.com/maps/@' + str(lat) + ',' + str(lng) + ',' + str(z) + 'z'
print(url)
browser.set_window_size(1024,512)
browser.get(url)
browser.save_screenshot("before.png")


# I noticed that there is an element called the omnibox which overlays the map.  So lets get rid of that!

# In[ ]:


# Get the scaling in meters and pixels
foot2meter = 0.3048;
scale_in_feet = float(browser.find_element_by_id('widget-scale-label').text.replace(' ft',''))
scale_in_meters = scale_in_feet*foot2meter

pixel_length = float(browser.find_element_by_class_name('widget-scale-ruler').value_of_css_property("width").replace('px',''))
print(pixel_length)

MetersPerPixel = scale_in_meters/pixel_length
print(MetersPerPixel)

js_string = "var element = document.getElementById(\"omnibox-container\");element.remove();"
browser.execute_script(js_string)
js_string = "var element = document.getElementById(\"vasquette\");element.remove();"
browser.execute_script(js_string)

js_string = "var element = document.getElementsByClassName(\"app-viewcard-strip\");element[0].remove();"
browser.execute_script(js_string)
js_string = "var element = document.getElementsByClassName(\"scene-footer-container\");element[0].remove();"
browser.execute_script(js_string)


# Now we can save the image!

# In[ ]:


browser.save_screenshot("waterfront.png")


# Yay! We have an image of the waterfront.  Now we will use Pillow to load the image and extract the pixels

# In[ ]:


from PIL import Image,ImageDraw,ImageFont

import matplotlib.pyplot as plt

img = Image.open('/kaggle/working/waterfront.png')
# Convert to RGBA
img = img.convert('RGBA')
# Load the pixels
pixels = img.load()
get_ipython().system('mkdir "/kaggle/working/frames"')


# In[ ]:



#define the find pixels that calculates number of pixels with that selection
def find_pixels(img,pixels,colour_set,slack, size):
    num_px = []
    # Set the value you want for these variables
    r_min = colour_set[0]-slack
    r_max = colour_set[0]+slack
    g_min = colour_set[1]-slack
    g_max = colour_set[1]+slack
    b_min = colour_set[2]-slack
    b_max = colour_set[2]+slack
    for x in range(size[0][0]):
        num_px_col_count = 0
        for y in range(size[0][1]):
            r, g, b,a = pixels[x,y]
            pixels[x,y]=(int(255), int(255), int(255),int(0))

            if r >= r_min and r <= r_max and b >= b_min and b <= b_max and g >= g_min and g <= g_max:
                num_px_col_count = num_px_col_count + 1;
                pixels[x,y]=(colour_set[0], colour_set[1], colour_set[2],int(255))
#         print(x)
        num_px.append(num_px_col_count)
        if x % 5 == 0:
            img.save('/kaggle/working/frames/' + str(x)+ '.png')

    return num_px


# In[ ]:


# Set the park colour
park_colour = [197,232,197];park_slack = 2;

num_park = find_pixels(img,pixels,park_colour,park_slack,[img.size])


# In[ ]:


get_ipython().system('mkdir "/kaggle/working/matplot/"')
fig=plt.figure(figsize=(1024/100,438/100))
ax = fig.add_subplot(111)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
fig.patch.set_alpha(0.0)
ax.set_xticklabels(labels = [''], fontsize = 18,alpha = .7,minor=False)
ax.set_yticklabels(labels = [''], fontsize = 18,alpha = .7,minor=False)


ax.set_xlim([-10,1024])
ax.set_ylim([-10,438])


for spine in ax.spines.values():
    spine.set_edgecolor('white')
plt.grid(False)
plt.axis('off')

for i in range(0,1024,10):
#     print(i)
    ax.plot(num_park[:i],color='#377e4d',linewidth=3,antialiased=True)
    plt.savefig('/kaggle/working/matplot/'+str(i)+'.png', transparent=True,dpi=130)


# Calculate the percentage and total area of parkland

# In[ ]:


print(img.size)
total_area = img.size[0]*img.size[1]*MetersPerPixel/(1000*1000)
print(total_area)


# In[ ]:


# Make a gif by combning the matplot and images to form a nice graphic!
title_font = ImageFont.truetype("/kaggle/input/antonfont/Anton-Regular.ttf", 30)
small_font = ImageFont.truetype("/kaggle/input/antonfont/Anton-Regular.ttf", 10)

# Loop over the frames in 10s
frames = []
for i in range(0,1024,10):
    # Load the other images we will overlay
    googlemap = Image.open('/kaggle/working/frames/'+ str(i)+ ".png")
    matplot = Image.open('/kaggle/working/matplot/'+str(i)+'.png')
    
    # Create a new image to start
    final = Image.new(mode='RGBA',size=(googlemap.size[0],googlemap.size[1]),color=(255,255,255,0))
    # Create the total composite image
    googlemap.paste(matplot, (-175, -52),matplot)
    final.paste(googlemap, (0,0),googlemap)

    # Do resampling to get the smoothing effect
    final = final.resize(final.size, resample=Image.ANTIALIAS)

    
    frames.append(final)




frames[0].save('t.gif', format='GIF', append_images=frames[1::], save_all=True, duration=1, loop=0)


# ## This notebook shows how to use headless Firefox browser + selenium library in Python to scrape data live in real-time from within a Kaggle notebook

# I was curious if I could make web scraping via a headless Firefox browser and selenium work in a Kaggle kernel. While this curiosity led to much frustration and hair tugging, I learned alot and am happy to report it works!

# ### **Pre-requisite steps:**
# 
# 1) First, manually download firefox for linux locally somewhere:
# 
# http://ftp.mozilla.org/pub/firefox/releases/63.0.3/linux-x86_64/en-US/firefox-63.0.3.tar.bz2
# 
# 2) then upload as new private "dataset" to kaggle account, making it selectable across kernels
# 
# *note: when uploading, make sure to choose "Keep tabular files in original format" *

# In[ ]:


# # SEE WHAT'S UNDER THE HOOD HERE
# !cat /etc/os-release


# In[ ]:


# # WHERE ARE WE RIGHT NOW?
# !ls -l .


# In[ ]:


# # BUT, WHERE, REALLY, ARE WE RIGHT NOW? (p.s. IT LOOKS LIKE WE'RE INSIDE /kaggle/working FOLDER, INSIDE AN ISOLATE DOCKER CONTAINER/IMAGE)
# !echo "ls -l /kaggle"
# !ls -l /kaggle

# !echo "\nls -l /kaggle/working"
# !ls -l /kaggle/working


# ### **Part 1: ** installing portable Firefox binary, geckodriver, and selenium library

# 1) Manually "+Add data | Your Datasets | firefox-63.0.3.tar.bz2"
# 
# note: referencing uploaded binary files as "datasets" automatically places them into "../input" folder
# 
# 2) Under Settings section, set Internet = "Internet Conneted"
