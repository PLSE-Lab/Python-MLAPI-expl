#!/usr/bin/env python
# coding: utf-8

# Boilerplate firefox install and instancing

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


# In[ ]:


# Setup for browser
from selenium import webdriver as selenium_webdriver
from selenium.webdriver.firefox.options import Options as selenium_options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities as selenium_DesiredCapabilities
browser_options = selenium_options()
browser_options.add_argument("--headless")
browser_options.add_argument("--window-size=3840,2160")
# browser_options.add_argument("user-agent=whatever you want")
capabilities_argument = selenium_DesiredCapabilities().FIREFOX
capabilities_argument["marionette"] = True


# In[ ]:


class element_has_css_class(object):
  """An expectation for checking that an element has a particular css class.

  locator - used to find the element
  returns the WebElement once it has the particular css class
  """
  def __init__(self, locator, css_class):
    self.locator = locator
    self.css_class = css_class

  def __call__(self, driver):
    element = driver.find_element(*self.locator)   # Finding the referenced element
    if self.css_class in element.get_attribute("class"):
        return element
    else:
        return False


# In[ ]:


get_ipython().system("mkdir '/kaggle/working/frames/'")
#         browser.set_window_size(1024,1024)
get_ipython().system('rm -fr ./frames/*')

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
from datetime import date, timedelta

sdate = date(2020, 1, 1)   # start date
edate = date(2020, 5, 13)   # end date
delta = edate - sdate       # as timedelta
counter = 0


for i in range(0,delta.days + 1,1):
    day = sdate + timedelta(days=i)
    for h in range(0,24,3):
        print(day,h,counter)
        browser = selenium_webdriver.Firefox(
            options=browser_options,
            firefox_binary="../working/firefox/firefox",
            capabilities=capabilities_argument
        )

#         url = 'https://earth.nullschool.net/#'+str(day.year)+'/'+str(day.month)+'/'+str(day.day)+'/'+str(0)+'00Z/wind/surface/level/anim=off/overlay=temp/orthographic=-89.00,31.64,319'
#         url = 'https://earth.nullschool.net/#'+str(day.year)+'/'+str(day.month)+'/'+str(day.day)+'/'+str(h)+'00Z/chem/surface/level/anim=off/overlay=cosc/orthographic=116.76,39.69,449'
#         url = 'https://earth.nullschool.net/#'+str(day.year)+'/'+str(day.month)+'/'+str(day.day)+'/'+str(h)+'00Z/particulates/surface/level/anim=off/overlay=pm10/orthographic=116.76,39.69,' + str(450+i)
        url = 'https://earth.nullschool.net/#'+str(day.year)+'/'+str(day.month)+'/'+str(day.day)+'/'+str(h)+'00Z/wind/surface/level/anim=off/overlay=misery_index/equirectangular=11.33,0.00,233/loc=148.023,-90.395'
        browser.get(url)
        js_string = "var element = document.getElementById(\"toggle-hd\");element.click();"
        browser.execute_script(js_string)
        wait = WebDriverWait(browser, 10)
        element = wait.until(element_has_css_class((By.ID, 'progress'), "hidden"))
        js_string = "var element = document.getElementById(\"sponsor\");element.remove();"
        browser.execute_script(js_string)       
        js_string = "var element = document.getElementById(\"details\");element.remove();"
        browser.execute_script(js_string)     
        browser.save_screenshot('/kaggle/working/frames/' + str(counter).zfill(4) + ".png")
        browser.close()
        counter = counter + 1


# In[ ]:


# frames=[]
# !pip install aggdraw
# import aggdraw

# from PIL import Image,ImageDraw,ImageFont
# for i in range(0,delta.days+1,1):
#     for h in range(0,24,3):
#         day = sdate + timedelta(days=i)
#         print(i,h)
#         # Load the other images we will overlay
#         final = Image.open('/kaggle/working/frames/' + str(i) + "." + str(h) + ".png")
        
# #         draw = aggdraw.Draw(final)
# #         # note that the color is specified in the font constructor in aggdraw
# #         font = aggdraw.Font((255,255,255), "/kaggle/input/m2pfont/mplus-2p-light.ttf", size=30, opacity=255)
# #         draw.text((760,30),str(day.year) + "." + str(day.month).zfill(2) + "." + str(day.day).zfill(2) + ':H' + str(h).zfill(2), font) # no color here
# # #         draw.text((30,30)," Beijing ", font)
# #         draw.text((30,860)," Sulpher Dioxide", font)

# #         draw.flush() # don't forget this to update the underlying PIL image!
        
        
#         final = final.resize((int(final.size[0]),int(final.size[1])), resample=Image.ANTIALIAS)
#         frames.append(final)
        
# frames = frames + frames[::-1]
# frames[0].save('tmp.gif', format='GIF', append_images=frames[1::], save_all=True, fps=30 , loop=0)



# In[ ]:


get_ipython().system('apt-get update')
get_ipython().system('apt install -y ffmpeg')
get_ipython().system('rm output.mp4')
get_ipython().system('ffmpeg  -i ./frames/%04d.png -c:v libx264 -c:a aac -ar 44100 -filter "minterpolate=\'fps=30\'" -pix_fmt yuv420p output.mp4')
# -vb 20M -r 30

