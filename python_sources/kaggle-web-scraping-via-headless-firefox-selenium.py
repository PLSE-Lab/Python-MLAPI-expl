#!/usr/bin/env python
# coding: utf-8

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


# SEE WHAT'S UNDER THE HOOD HERE
get_ipython().system('cat /etc/os-release')


# In[ ]:


# WHERE ARE WE RIGHT NOW?
get_ipython().system('ls -l .')


# In[ ]:


# BUT, WHERE, REALLY, ARE WE RIGHT NOW? (p.s. IT LOOKS LIKE WE'RE INSIDE /kaggle/working FOLDER, INSIDE AN ISOLATE DOCKER CONTAINER/IMAGE)
get_ipython().system('echo "ls -l /kaggle"')
get_ipython().system('ls -l /kaggle')

get_ipython().system('echo "\\nls -l /kaggle/working"')
get_ipython().system('ls -l /kaggle/working')


# ### **Part 1: ** installing portable Firefox binary, geckodriver, and selenium library

# 1) Manually "+Add data | Your Datasets | firefox-63.0.3.tar.bz2"
# 
# note: referencing uploaded binary files as "datasets" automatically places them into "../input" folder
# 
# 2) Under Settings section, set Internet = "Internet Conneted"

# In[ ]:


# LOOK AT INPUT FOLDER, WE SHOULD SEE "firefox-63.0.3" FOLDER ALREADY THERE
get_ipython().system('ls -l "../input"')


# In[ ]:


# WE WILL MAKE NEW SUBFOLDER IN WORKING FOLDER (WHICH ISN'T READ-ONLY)
get_ipython().system('mkdir "../working/firefox"')
get_ipython().system('ls -l "../working"')


# In[ ]:


# COPY OVER FIREFOX FOLDER INTO NEW SUBFOLDER JUST CREATED
get_ipython().system('cp -a "../input/firefox-63.0.3/firefox/." "../working/firefox"')
get_ipython().system('ls -l "../working/firefox"')


# In[ ]:


# ADD READ/WRITE/EXECUTE CAPABILITES
get_ipython().system('chmod -R 777 "../working/firefox"')
get_ipython().system('ls -l "../working/firefox"')


# In[ ]:


# INSTALL PYTHON MODULE FOR AUTOMATIC HANDLING OF DOWNLOADING AND INSTALLING THE GeckoDriver WEB DRIVER WE NEED
get_ipython().system('pip install webdriverdownloader')


# In[ ]:


# INSTALL LATEST VERSION OF THE WEB DRIVER
from webdriverdownloader import GeckoDriverDownloader
gdd = GeckoDriverDownloader()
gdd.download_and_install("v0.23.0")


# In[ ]:


# INSTALL SELENIUM MODULE FOR AUTOMATING THINGS
get_ipython().system('pip install selenium')


# In[ ]:


# LAUNCHING FIREFOX, EVEN INVISIBLY, HAS SOME DEPENDENCIES ON SOME SCREEN-BASED LIBARIES
get_ipython().system('apt-get install -y libgtk-3-0 libdbus-glib-1-2 xvfb')


# In[ ]:


# SETUP A VIRTUAL "SCREEN" FOR FIREFOX TO USe
get_ipython().system('export DISPLAY=:99')


# ### **Part 2:** automate Firefox

# In[ ]:


# PYTHON MODULES TO USE
from selenium import webdriver as selenium_webdriver
from selenium.webdriver.firefox.options import Options as selenium_options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities as selenium_DesiredCapabilities


# In[ ]:


# FIRE UP A HEADLESS BROWSER SESSION WITH A "SCREEN SIZE" OF 1920x1080

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


# SHOW LIST OF RUNNING PROCESSES; SHOULD SEE firefox AND geckodriver
get_ipython().system('ps -A')


# In[ ]:


# PERFORM A WEB SEARCH (SEE HOW WE CAN EVEN ARBITRARILY CHANGE BROWSER WINDOW SIZE ON-THE-FLY "MOSTLY" AS WE PLEASE, IF <= BROWSER_OPTION ABOVE)
browser.set_window_size(1366, 768)
browser.get("https://duckduckgo.com/")
browser.find_element_by_id('search_form_input_homepage').send_keys("Kaggle Rocks!")
browser.find_element_by_id("search_button_homepage").click()
print(browser.current_url)


# In[ ]:


# WE CAN EVEN TAKE A "SCREENSHOT"!
browser.save_screenshot("screenshot.png")

get_ipython().system('ls -l .')


# In[ ]:


# LET'S LOOK AT IT!
from IPython.display import Image
Image("screenshot.png", width=800, height=500)


# In[ ]:


# CLOSE FIREFOX BROWSER
browser.quit()

get_ipython().system('ps -A')


# ### Part 3: pull some data from somewhere out there down into a pandas data frame for analysis

# In[ ]:


# MORE MODULES TO IMPORT
import pandas as pd
import seaborn as sns

from selenium.webdriver.common.by  import By as selenium_By
from selenium.webdriver.support.ui import Select as selenium_Select
from selenium.webdriver.support.ui import WebDriverWait as selenium_WebDriverWait
from selenium.webdriver.support    import expected_conditions as selenium_ec


# In[ ]:


# FIRE UP A HEADLESS BROWSER SESSION WITH A "SCREEN SIZE" OF 1920x1080

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


# NAVIGATE TO NBA.COM TEAM STATS AND TAKE A PICTURE TO "PROVE" TO OURSELVES WHERE WE'RE AT
browser.get("https://stats.nba.com/teams/traditional/?sort=W_PCT&dir=-1")
print(browser.current_url)
browser.save_screenshot("screenshot.png")
Image("screenshot.png", width=800, height=500)


# In[ ]:


# INSTEAD OF "Per Game" TEAM STATS, LET'S LOOK AT "Per 100 Possessions"
sel_obj = selenium_Select(browser.find_element_by_name("PerMode"))
sel_obj.select_by_visible_text("Per 100 Poss")
                          
# WE NEED TO WAIT FOR DYNAMIC CONTENT TO REFRESH; WE WILL GIVE IT UP TO 10 SECONDS
wait = selenium_WebDriverWait(browser, 10)
wait.until(selenium_ec.visibility_of_element_located((selenium_By.XPATH, '//div[@class="nba-stat-table__overflow"]')))

# WHILE AT IT, LET'S SORT BY PTS IN DESCENDING ORDER, JUST FOR FUN
browser.find_element_by_xpath('//th[@data-field="PTS"]').click()

# SEE WHERE WE ARE NOW
browser.save_screenshot("screenshot.png")
Image("screenshot.png", width=800, height=500)


# In[ ]:


# OK, LET'S GRAB THE HTML OF THIS TABLE NOW AND TAKE A PEAK AT FIRST 1500 CHARACTERS TO SEE IF IT SEEMS RIGHT
html = browser.find_element_by_xpath('//div[@class="nba-stat-table__overflow"]').get_attribute("outerHTML")
html[0:1500]


# In[ ]:


# LET'S LOAD INTO PANDAS AS SEE WHAT WE REALLY HAVE
df = pd.read_html(html)[0]
df.head()


# In[ ]:


# LET'S DO SOME CLEANUP HERE (WE ONLY CARE ABOUT COLUMNS 2-29)
df = df.iloc[:, 1:28]
df.head()


# In[ ]:


# DO SOME BASIC STATS
df.describe()


# In[ ]:


# WHAT'S "MOST IMPORTANT" IN TERMS OF TEAM OFFENSIVE SCORING PER 100 POSSESSIONS AMONG SOME MAIN STAT CATEGORIES?
# NOTE: FG% APPEARS MOST IMPORTANT - JUST LIKE "Basketball on Paper", BY DEAN OLIVER SUGGESTS!
tmp = df[["PTS","FG%","3P%","FT%","OREB","AST","TOV"]]
g = sns.pairplot(tmp, kind="reg")


# In[ ]:


# WE CAN CLOSE FIREFOX NOW (REALLY COULD HAVE AFTER WE SNAGGED THE HTML WE NEEDED)
browser.quit()

get_ipython().system('ps -A')


# ### CONCLUSION

# I hope this helps somebody out there. If not, it wil be a useful reference for myself later!

# In[ ]:




