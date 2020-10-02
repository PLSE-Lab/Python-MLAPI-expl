#!/usr/bin/env python
# coding: utf-8

# ![](https://cdn-images-1.medium.com/max/2000/1*f2-zeAOSNB4RGlqH9emTlQ.jpeg)

# # Goal of this kernel is to **AUTOMATICALLY** surf the website [Yahoo finance](https://finance.yahoo.com), extract the information in **REAL** time and send it back via E-mail to desired adress
# 
# Information will be the biggest daily gainers, i.e. the stock that gained the most value that day. Or the biggest losers, depending on the choice of the user.
# 

# ## NOTES:
# * Code **WILL NOT FUNCTION** on kaggle for multiple reasons, the biggest ones are problems with including the path and not beeing able to import web_driver package. It is no problem. Reader can download the ipython notebook change some aspects of it that are compliant to one owns specifications and it should run with no problem. (dependent on the operative system, path, receiving and sending email adress etc...)
# 
# * I will also try to comment the code generated so that the reader can slightly update it and extract some other **dynamical** infomation from the website automatically himself.
# 
# 
# *  There are certain guidelines when doing web-scraping. Please do check [Safe-scraping](https://www.scrapehero.com/how-to-prevent-getting-blacklisted-while-scraping/) and be responsible. Regarding Yahoo Finance it says [in summary](https://finance.yahoo.com/robots.txt) tread carefully, do not burden the system!

# WE will need selenium to access the web-site but one of the problems is that importing specific libraries on kaggle is still buggy according to admins. Hence it is not possible to run it on kaggle directly. I assure you downloading this notebook with some changes where I indicated them works like a charm on one owns PC.

# In[ ]:


get_ipython().system('pip install selenium')


# In[ ]:


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


# In[ ]:


import pandas as pd


# In[ ]:


import time
import datetime


# In[ ]:


import smtplib
from email.mime.multipart import MIMEMultipart


# In[ ]:


import sys
sys.path.append(r"C:\Users\Noah\Desktop\zzz\chromedriver.exe")


# In[ ]:


driver = webdriver.Chrome(executable_path=r"C:\Users\Noah\Desktop\zzz\chromedriver.exe")


# We have to add chromedriver to the PATH in order to open it and surf it automatically. 2 cells are user-specific and should be changed.

# After we can access the web searcher of our choice, let us find the xpath of the desired web-elements. In other words in order to access all of the elements of the desired web-page we need to direct it in a way. Underneath, the defined function replaces that for us. We just need to inspect the web element and copy the xpath of the html tag as follows

# In[ ]:


from IPython.display import Image
Image("../input/1.png")


# In[ ]:


stocks_gainers='//*[@id="SecondaryNav-0-SecondaryNav-Proxy"]/div/ul/li[5]/a'
stocks_losers='//*[@id="SecondaryNav-0-SecondaryNav-Proxy"]/div/ul/li[6]/a'


# In[ ]:


# clicking on the losers,gainers
def market_chooser(market_product):
    
    try:
        market_product_type = driver.find_element_by_xpath(market_product)
        market_product_type.click()
    except Exception as e:
        pass


# Now we define a data frame to store these values, which is going to be executed with the **compile_data()** function.

# In[ ]:


df = pd.DataFrame()
def compile_data():
    global df
    global top_of_the_list
    
    top_of_the_list = driver.find_elements_by_xpath('//*[@id="scr-res-table"]/div[1]/table/tbody/tr[1]/td[2]')
    top_of_the_list_list = [value.text for value in top_of_the_list]

   

    for i in range(len(top_of_the_list_list)):
        try:
            df.loc["Characteristics", i] = top_of_the_list_list[i]
        except Exception as e:
            pass
    print('Excel Sheet Created!')


# ### Now email characteristics. Again reader should change the details himself

# In[ ]:


username = 'noahweber@gmail.com'
password = 'xxxxxxxxxxxxxxxxxxx'


# Possibly using hotmail, or comapny email adress

# In[ ]:


def connect_mail(username, password):
    global server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(username, password)


# Massage to be sent:

# In[ ]:


def create_msg():
    global msg
    msg = '\nInformation: {}\n'.format(current_values)


# In[ ]:


def send_email(msg):
    global message
    message = MIMEMultipart()
    message['Subject'] = 'Yahoo Finance'
    message['From'] = 'noahweber@gmail.com'
    message['to'] = 'noahweber@gmail.com'

    server.sendmail('noahweber@gmail.com', 'noahweber@gmail.com', msg)


# Finally use the time library to specify how long we are going to run the script. Since yahoo page is renewed every few minutes we should scrape it every few minutes. And than wait for 60 minutes until running the for loop again and calling the functions again and sending a new email.

# In[ ]:


for i in range(8):    
    link = 'https://finance.yahoo.com/'
    driver.get(link)
    #wait for the page to load
    time.sleep(10)

    markets_button_yfinance = driver.find_element_by_xpath('//*[@id="Nav-0-DesktopNav"]/div/div[3]/div/div[1]/ul/li[4]/a')
    markets_button_yfinance.click()
    market_chooser(stocks_gainers)
    
    
    compile_data()
    
    #save values for email
    current_values = df.iloc[0]
    
    print('Number of iterations: {}'.format(i))
    create_msg()
    connect_mail(username,password)
    send_email(msg)
    print('Email sent')
    
    df.to_excel('best_worst_stock.xlsx')
    
    time.sleep(60)


# In[ ]:





# 

# 
