#!/usr/bin/env python
# coding: utf-8

# <h1> Welcome to my kernel. I have processed some columns to get better meaning of them including the user_agent strings and I have created new csv file containing the parsed UA as it took time in parsing so I did not want others to waste their time on parsing it :D</h1>
# <h1>Please do upvote if you find this kernel helpful or else just upvote :D</h1>
# 
# <h1><a href="https://www.kaggle.com/omegaji/bots-ua-parsed">THE NEW DATA CREATED CAN BE FOUND HERE YOU CAN USE IT IN YOUR KERNELS!!  </a></h1>

# <h1>List of Contents</h1>
# <ul>
#     <li><a href="#os_ext">Os Name Extraction and shortening/Grouping</a></li>
#         <li><a href="#views_visits">Plotting User views and visits</a></li>
#         <li><a href="#what_ua">What are user agents?</a></li>
#     <li><a href="#dd">Parsing User Agent String with DeviceDetector</a></li>
#         <li><a href="#first_half">Parsing The first half sample</a></li>
#         <li><a href="#is_bot">Detecting Bot or not from user agent</a></li>
#         <li><a href="#plot_is_bot">we plot how many bots are from which os</a></li>
#             <li><a href="#is_bot_browser">we plot how many bots are from which browser</a></li>
#         <li><a href="#tocsv1">Convert the first half of the data to csv </a></li>
#             <li><a href="#secondhalf">Convert the second half </a></li>
#         <li><a href="#big">Concatenate both and final dataframe savse as csv  </a></li>
#             <li><a href="#conclusion">Future work?  </a></li>
# 
#  
# 
# 
# 
#   
#     
# </ul>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df=pd.read_csv("/kaggle/input/bot-detection/ibm_data.csv")
df


# <h3>checking for nulls in operating systems column and replacing them with notgiven, we then find out the unique values of this column</h3>

# In[ ]:


print(df.operating_sys.isnull().sum())
df.operating_sys.fillna("NotGiven",inplace=True)
print(df.operating_sys.unique())


# <h3 id="os_ext"> I create a function that will classify all different variants of each os as the same os for example MICROSOFT_WINDOWS8.1
# 
# MICROSOFT_WINXP are to be renamed to MICROSOFT PC</h3>
#     <h3>I do the same for the other types of os including mobile phones, I classify the ones not falling into these main types as others </h3>

# In[ ]:


def shortenos(x):
    #print(x)
    if "microsoft" in x.lower().split("_")[0]:
        x="MICROSOFT PC"
        return x
    elif "windowsphone" in x.lower().split("_")[0]:
        x="WINDOWS MOBILE"
        return x
    elif "windowsmobile" in x.lower().split("_")[0]:
        x="WINDOWS MOBILE"
        return x
    elif "macintosh" in x.lower().split("_")[0]:
        x="MACOS PC"
        return x
    elif "ios" in x.lower().split("_")[0]:
        x="IOS PHONE"
        return x
    elif "android" in x.lower().split("_")[0]:
        x="ANDROID"
        return x
    elif "linux" in x.lower().split("_")[0]:
        x="LINUX"
        return x
    elif x.lower()=="notgiven":
        x="NotGiven"
        return x
    else:
        x="OTHER"
        return x
df["os"]=df.operating_sys.apply(shortenos)
  


# <h3>Now lets groupby the new os column we created and sum the values for VIEWS and VISIT</h3>
# <h3>This will help us in analysing which of these users are belonging to which OS</h3>

# In[ ]:


os_df=df.groupby(["os"]).sum().reset_index()
os_df


# <h3 id="views_visits">Lets PLOT! I plotted barplot on the visit and views count you can hover for the value</h3>

# In[ ]:



import altair as alt
import altair_render_script
alt.data_transformers.disable_max_rows()
base=alt.Chart(os_df).mark_bar().encode(
x="os",
y="VISIT",
tooltip=["VISIT"]
)

base2=alt.Chart(os_df).mark_bar().encode(
x="os",
y="VIEWS",tooltip=["VIEWS"]
)
alt.hconcat(base,base2)


# <h1> As we saw microsoft leads in both VISIT and VIEWS as expected of their huge userbase</h1>
# <h3> now comes the hard part...well for me because I did not understand the user agent part until some googling </h3>

# <h1 id="what_ua">What are user agents?</h1>
# <p>Every time your web browser makes a request to a website, it sends a HTTP Header called the "User Agent". The User Agent string contains information about your web browser name, operating system, device type and lots of other useful bits of information.
# 
# But every browser sends its user agent in a different format, so decoding them can be very tricky.</p>
# <p>you have many apis and libarries that decode them for you and even tell you if the are<b>a bot or not!</b></p>
# <ul>
#     <li>user-agents from pypi <a href="https://pypi.org/project/user-agents/">  click here</a></li>
#         <li>device-detector(we will use this as it is faster) <a href="https://pypi.org/project/device-detector/">click here</a></li>
#         <li>what is my browser provides apis for this too well it is not free but there are some trial plans which might interest you <a href="https://developers.whatismybrowser.com/useragents/parse/">  click here</a></li></ul>

# In[ ]:


df.user_agent.dropna(inplace=True)


# <h1> Installing device detector</h1>

# In[ ]:


get_ipython().system('pip install device_detector')


# <h1 id="dd"> you can check the docs over here https://pypi.org/project/device-detector/</h1>
# device = SoftwareDetector(ua).parse()
# 
# device.client_name()        # >>> Chrome Mobile
# device.client_short_name()  # >>> CM
# device.client_type()        # >>> browser
# device.client_version()     # >>> 58.0.3029.83
# 
# device.os_name()     # >>> Android
# device.os_version()  # >>> 6.0
# device.engine()      # >>> WebKit
# 
# device.device_brand_name()  # >>> ''
# device.device_brand()       # >>> ''
# device.device_model()       # >>> ''
# device.device_type()        # >>> ''
# 
# 
# <h1> WHAT AM I EXTRACTING?</h1>
# <ul>
#     <li>The Browser used</li>
#     <li>The os(although we already have "os" column but I did for the sake of experimenting</li>
#     <li>THE MAIN THING IS IT A BOT????(TRUE AND FALSE)</li>
# </ul>

# In[ ]:


from device_detector import SoftwareDetector


# <h1> I create some functions which will be used in apply one will give the client name[browser name] and the other will give out the os name</h1>
# <h1 id="first_half">NOTE:I will be trying these functions on sample data of 400000 as the original dataset takes too much time if you have any recommendation please do tell :D</h1>
# <h1> NOTE: I will then convert this to a csv filer and save it and do the same process on the remaining data and convert that too a csv file too, so 2 csv files will be created and  I will make it public so that we can easily load it without worrying about the user agents parsing </h1>

# In[ ]:



def parse_family(x):
    
    return SoftwareDetector(x).parse().client_name()
def parse_os(x):
  
    return SoftwareDetector(x).parse().os_name()


    


# <h1> here as I stated Ill be taking the 400000 as my sample dataset</h1>

# In[ ]:


sample_df=df[:400000]
sample_df.user_agent.dropna(inplace=True)


# <h1> The parse family function will parse the browser</h1>
# 

# In[ ]:


x=sample_df["user_agent"].apply(parse_family)


# <h1> dropping some null values and the parse os will parse the os</h1>

# In[ ]:


sample_df["user_browser"]=x
sample_df.user_agent.dropna(inplace=True)
sample_df["user_os"]=sample_df.user_agent.apply(parse_os)


# In[ ]:


sample_df


# <h1 id="is_bot"> Note we import Device Detector below which could have been imported above as well instead of SoftwareDetector,
# but SoftwareDetector detects the software specifically and it faster than device_detector which is said in the docs :D, but device detector below will be needed to classify as bots or not
# </h1>

# In[ ]:


from device_detector import DeviceDetector
def parse_is_bot(x):
    return DeviceDetector(x).parse().is_bot()
sample_df.user_agent.dropna(inplace=True)
sample_df["is_bot"]=sample_df["user_agent"].apply(parse_is_bot)


# <h1>Some oses were empty because it couldnt be detected so I added unknown string to it</h1>
# <h2>I then group the sample_df by users_os(the one from the User Agent string) and is bot column and use the Count() on each column</h2>

# In[ ]:


def replace_empty_user(x):
    if x=="":
       return "Unknown"
    else:
        return x
sample_df["user_os"]=sample_df["user_os"].apply(replace_empty_user)
user_os_df=sample_df.groupby(["user_os","is_bot"]).count().reset_index()


# <h1>Renaming the column unamed to count</h1>

# In[ ]:


user_os_df.rename(columns={"Unnamed: 0": "count"},inplace=True)
user_os_df


# <h1 id="plot_is_bot"> Plotting 2 barplots one for bots detected and one for not bots hover for more info</h1>

# In[ ]:


base=alt.Chart(user_os_df[user_os_df.is_bot==True]).mark_bar().encode(
x="user_os",
y="count",
    tooltip=["count"]
).properties(title="BOTS")

base1=alt.Chart(user_os_df[user_os_df.is_bot==False]).mark_bar().encode(
x="user_os",
y="count",
     tooltip=["count"]
).properties(title="Not_BOTS")
alt.hconcat(base,base1)


# <h1 id="is_bot_browser"> We do the same but instead of user os we count for browser</h1>
# <h1>NOTE: There are some browser names parsed by the parse like _CT_OBJ...etc I do not have much information about them If you have any knowledge or info about them please do comment :D</h1>

# In[ ]:



browser_df=sample_df.groupby(["user_browser","is_bot"]).count().reset_index()
browser_df.rename(columns={"Unnamed: 0": "count"},inplace=True)

base=alt.Chart(browser_df[browser_df.is_bot==True]).mark_bar().encode(
x="user_browser",
y="count",
    tooltip=["count"]
).properties(title="BOTS")

base1=alt.Chart(browser_df[browser_df.is_bot==False]).mark_bar().encode(
x="user_browser",
y="count",
     tooltip=["count"]
).properties(title="Not_BOTS")
alt.hconcat(base,base1)


# <h2 id="tocsv1"> I convert this dataframe to first half csv which I will be planning to use in other notebooks </h2>

# In[ ]:


sample_df.to_csv("bots_firsthalf.csv",index=False)


# <h2> Now the same procedure for the rest of the data</h2>

# In[ ]:


sample_df2=df[400000:]
sample_df2.user_agent.dropna(inplace=True)
x=sample_df2["user_agent"].apply(parse_family)
sample_df2["user_browser"]=x
sample_df2.user_agent.dropna(inplace=True)
sample_df2["user_os"]=sample_df2.user_agent.apply(parse_os)


# In[ ]:


sample_df2.user_agent.dropna(inplace=True)
sample_df2["is_bot"]=sample_df2["user_agent"].apply(parse_is_bot)


# <h1 id="secondhalf">The secondHalf of the cvs</h1>

# In[ ]:


sample_df.to_csv("bots_secondhalf.csv",index=False)


# * <h1 id="big"> concatenate both dataframe to the original big dataframe and then save that dataframe as a csv</h1>

# In[ ]:


the_big_df=pd.concat([sample_df,sample_df2])


# In[ ]:


the_big_df.to_csv("bots_full.csv",index=False)


# <h1 id="conclusion"> TO BE CONTINUED!!! I WILL WORK ON A NEW NOTEBOOK OTHER THAN THIS STAY TUNED!!</h1>

# In[ ]:





# In[ ]:




