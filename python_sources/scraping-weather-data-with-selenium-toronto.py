#!/usr/bin/env python
# coding: utf-8

# ### **Pre-requisite steps:**
# 
# 1) First, manually download firefox for linux locally somewhere:
# 
# http://ftp.mozilla.org/pub/firefox/releases/63.0.3/linux-x86_64/en-US/firefox-63.0.3.tar.bz2
# 
# 2) then upload as new private "dataset" to kaggle account, making it selectable across kernels
# 
# *note: when uploading, make sure to choose "Keep tabular files in original format" *

# ### **Part 1: ** installing portable Firefox binary, geckodriver, and selenium library

# 1) Manually "+Add data | Your Datasets | firefox-63.0.3.tar.bz2"
# 
# note: referencing uploaded binary files as "datasets" automatically places them into "../input" folder
# 
# 2) Under Settings section, set Internet = "Internet Conneted"

# In[ ]:


# WE WILL MAKE NEW SUBFOLDER IN WORKING FOLDER (WHICH ISN'T READ-ONLY)
get_ipython().system('mkdir "../working/firefox"')
get_ipython().system('cp -a "../input/firefox-63.0.3/firefox/." "../working/firefox"')
get_ipython().system('chmod -R 777 "../working/firefox"')
get_ipython().system('pip install webdriverdownloader')
from webdriverdownloader import GeckoDriverDownloader
gdd = GeckoDriverDownloader()
gdd.download_and_install("v0.23.0")
get_ipython().system('pip install selenium')
get_ipython().system('apt-get install -y libgtk-3-0 libdbus-glib-1-2 xvfb')
get_ipython().system('export DISPLAY=:99')


# ### **Part 2:** Scrape Data

# In[ ]:


# MORE MODULES TO IMPORT
import pandas as pd
import seaborn as sns

from selenium.webdriver.common.by  import By as selenium_By
from selenium.webdriver.support.ui import Select as selenium_Select
from selenium.webdriver.support.ui import WebDriverWait as selenium_WebDriverWait
from selenium.webdriver.support    import expected_conditions as selenium_ec
from selenium.webdriver.firefox.options import Options as selenium_options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities as selenium_DesiredCapabilities
from selenium import webdriver as selenium_webdriver
browser_options = selenium_options()

browser_options.add_argument("--headless")
browser_options.add_argument("--window-size=1920,1080")

capabilities_argument = selenium_DesiredCapabilities().FIREFOX
capabilities_argument["marionette"] = True

browser = selenium_webdriver.Firefox(
    options=browser_options,
    firefox_binary="../working/firefox/firefox",
    capabilities=capabilities_argument
)


# In[ ]:


import time
from datetime import datetime
formatted_data = []

for y in range(2014,2021):
    for m in range(1,13):
        print(y,m)
        # Now we will extract the data from the website via JS return of global variable
        browser.get('https://www.timeanddate.com/weather/canada/toronto/historic?month='+str(m)+'&year='+str(y))
        data = browser.execute_script("return data;")

        # Loop over the results and save results
        for e in data['temp']:
            # Get the date and temp data 
            clean_day = time.strftime('%m',  time.gmtime(e['date']/1000.))
            clean_hour = time.strftime('%H',  time.gmtime(e['date']/1000.))
            formatted_data.append([y,m,int(clean_day),int(clean_hour),e['temp'],(e['temp']-32)*5/9])

df = pd.DataFrame(formatted_data,columns=['year','month','day','hour','DegreeF','DegreeC'])
df.to_csv('Temperature_Toronto.csv',index=False)



# In[ ]:




