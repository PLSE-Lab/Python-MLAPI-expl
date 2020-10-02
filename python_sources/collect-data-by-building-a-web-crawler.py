#!/usr/bin/env python
# coding: utf-8

# ### Motivation Behind This Series

# Just like everyone else, once I had completed my *Introduction to Python* on an MOOC, I was eager to apply what I had learned on a real-world machine learning problem. However, when it actually came time to build something useful, I realized that the toy projects and purely conceptual lessons that I had learned in the Python course did not help much. I have come to realize that this is because, while most of the tutorials for Python and its application on YouTube and other sources are great for learning stand-alone aspects of a project, they seldom give a complete picture of how all the pieces fit together. Since then, through reading, trial and experimentation, and a lot of sleepless nights of debugging, I started to learn how different blocks can be put together to form a coherent project and began realizing that with a little foresight, programming can be used to perfrom anything that you need. These series of tutorials titled, **Truly End-to-End Machine Learning** are meant as a way for me to document my approach to solving a problem and also serve as a guide for you to learn new skills and apply them in a truly end-to-end ML project
# 

# Some of the thing that I want to point out so that you can get the best out these tutorials are:
# 1. You must have a beginner to an intermediate level understanding of Python to follow along. I do not cover the basics since there are those who are much more qualified than me to teach you about it. [YouTube](https://www.youtube.com/watch?v=YYXdXT2l-Gg&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU) and [Medium](https://medium.com/topic/programming) are great to get you started.
# 2. In the case of libraries, I will refer to their documentation, GitHub posts on any specific issues that I faced while using them and also any tutorials which I found helpful to gain a good understanding of what I needed to get this project going. Please refer to them if you need help understanding the methods and attributes in a library.
# 3. Although I encourage you to follow along with this tutorial, my suggestion is that you read through it first to understand how to think through the problem, and then use the skills that you learn here to solve a problem that **you find interesting**.
# 4. I will refer to other channels, pages, blogs, etc whenever possible. To learn the skills you need to solve a large problem, there will be a lot of Googling and downtime involved to learn the necessary concepts and tools.
# 5. Everything that I have written here is only meant as a template on how this particular problem can be approached. There are numerous thing which could be done to my code to improve its speed and readability. Since I want to put up tutorials on a weekly basis, I will not be refactoring my code here, but you can always find the latest version of this crawler on my [GitHub](https://github.com/vigvisw) page.
# 
# **If you are an experienced developer and you see any mistakes in my code or see places where I can make improvements**, I request you to please point them out to me.
# 

# ### A Note On Colaboratory

# Most of this tutorial will be written and can be completed by you in Google's amazing tool [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb#scrollTo=xitplqMNk_Hc). Please visit the attached link to learn more about Colab and how to use its numerous features. I also want to share my personal reasons for using it and hope that I can convince you to explore it further.
# 
# 1. Colab requires absolutely no setup to get you started. You can just login within your Google account and get started right away.
# 2. You have access to a [Jupyuter Notebook](https://jupyter.org/). Need I say more? But in all seriousness, I think it is a great way for beginners and even advanced users to prototype code and take notes using [Markdown](https://colab.research.google.com/notebooks/markdown_guide.ipynb#scrollTo=JtBxirFReX5n).
# 3. There are **LOT** of libraries pre-built into your virtual environment, and hence there is minimal setup required to get you started. 
# 4. Cloud functionality is great when switching between devices. You can code on your laptop and instantly switch over to another machine at work and connect to a local runtime, if needed.
# 5. Finally, any potato device that can run a browser has the ability to access a powerful virtual machine. I split my time between my beloved Chromebook Pro and my aging Lenovo Y50-70. Colab can be seamlessly run on both these machines.
# 
# Before you start using Colab, I **highly suggest** that you go under *Tools*, click on *Preferences* and check **'Show line numbers'**. This makes it much easier to debug your code when (**not if**) it throws an exception. You might also want to use the **dark mode** to ease the strain on your eyes for a long coding session and change any other settings as you see fit. Also check out [this](https://www.kdnuggets.com/2018/02/essential-google-colaboratory-tips-tricks.html) article from KDNuggets to learn a few more of Colab's features
# 
# **NOTE** :
# Despite all its merits, Colab caused me one major issue when writing this tutorial. I was unable to keep a hosted Colab runtime alive for more than two hours when using the crawler. With over 9000+ links to crawl across, we are looking at a crawl time in excess of three hours. I ended up having to connect to a local Jupyter Notebook runtime to get it to run for the full length of time needed. The code can be definitely be refactored to get over this issue and if you know of any method to keep Colabs hosted runtime for longer, please let me know.

# ### The Big Picture

# ![A typical Machine Learning project (Courtesy: Western Digitial)](https://2s7gjr373w3x22jf92z99mgm5w-wpengine.netdna-ssl.com/wp-content/uploads/2018/09/WD_3.png)

# The above pictograph (Courtesy: Western Digital) illustrates what is considered the typical pipeline for an ML project. This version of the pipeline works great when presented with well-structured SQL databases, CSV files or JSON data. However, in the real world, data is hard to collect, messy and presents a number of challenges. Your problem statement might be to build a model that takes in information collected using sensors (accelerometers, gyroscopes, and thermometers for example) and make predictions about a dependent variable. If data does not already exist for your project, it can be a challenge to devise your own methods to collect such data. 
# 
# My first suggestion is that you take a look at sites like [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php), [Kaggle](https://www.kaggle.com/) and, [Google Dataset Search](https://toolbox.google.com/datasetsearch). Many a time, you can find the data that you need through a simple search and save a lot of effort. Failing that try checking if the data that you need can be accessed using an API. May interesting APIs exist, which can make your job a lot simpler. I would like to point out a few interesting ones for those who want to build a dataset using an existing API.
# 
# 1. [Materials Project](https://materialsproject.org/): "Provides open web-based access to computed information on known and predicted materials as well as powerful analysis tools to inspire and design novel materials". If anyone is interested, I will be happy to make a tutorial on how to build a materials dataset using the Python wrapper for the API in the future. 
# 2. [PRAW: The Python Reddit API Wrapper](https://praw.readthedocs.io/en/latest/) and [Pushshift](https://pushshift.io/): Easy to use APIs used for accessing data from Reddit. 
# 
# The worst case and the most probable scenario is that that data you want does not exist and you have to collect it yourself. Depending on your problem, the approach and the skill set you need to learn will differ. In the context of a computer vision problem involving collecting and labeling an image data set, you may have to set up an imaging system and use computer vision tools such as [OpenCV](https://opencv.org/) to get the task done. In other cases, you may have to crawl through a website and extract the data that you need. This leads to the infamous [80/20 split](https://www.infoworld.com/article/3228245/the-80-20-data-science-dilemma.html) where up to 80% of your time in an ML project can be sunk into finding, cleaning and restructuring data.
# 
# The closest I have come to seeing a truly end-to-end ML project is [A Complete Machine Learning Project Walk-Through in Python](https://towardsdatascience.com/a-complete-machine-learning-walk-through-in-python-part-one-c62152f39420) by [Will Koehrsen](https://towardsdatascience.com/@williamkoehrsen). Even this incredibly detailed tutorial starts with a pre-assembled dataset which is often **not indicative** of the real world. The point I am trying to get across is that the most mission critical aspect of an ML project, which more often than not, is the data itself can be hard to find and it is always a good idea to set aside the time to learn the skills require to collect it. 

# In this tutorial, we will work on actually gathering the data before passing it on for processing process which will be covered in next week's tutorial. Our aim here is to develop a web crawler, which can traverse the website [GSMArena](https://www.gsmarena.com/) and gather the spec sheets for all devices stored by them in over 9000+ webpages.
# 
# **Disclaimer**
# 1. All the information that we gather here is made available through the hard work of the GSMArena team and all credits go to them for it. 
# 2. **ALWAYS** check the ['robot.txt'](https://en.wikipedia.org/wiki/Robots_exclusion_standard) of the website that you are trying to crawl. GSMArena does not forbid crawling for the information that we collecting here, but I request that you do not strain their servers by creating unnecessary requests.
# 3. I am breaking my own rule. A quick search reveals that [others](hhttps://www.google.com/search?rlz=1CAWOMZ_enUS810&ei=VN2EXJP7EOSN5wKx-q-oCQ&q=gsm+arena+dataset&oq=gsm+arena+data&gs_l=psy-ab.1.0.35i39.893.2883..5425...0.0..0.99.1026.13......0....1..gws-wiz.......0i71j0j0i22i30j0i22i10i30j0i131i10j0i20i263j0i131i20i263.ICV5YZv6D1I) have already assembled an almost identical dataset, but I still decided to go ahead with it any way because learning how to build a web crawler is an indespensable tool to have in my arsenal and I could not find any acessible tutorials online that went as deep as we are going to in a building a crawler. As a personal rule, I refrained from taking a look at these projects project and the dataset they collected, until I was done building this crawler since I wanted to give the problem a shot without an *answer key* already given to me.

# ### Framing The Problem

# I refer to the process of building the web crawler and everything else to follow as a problem because, in a realistic setting, a problem is exactly what it will turn out to be. But I do not use 'problem' in the negative context of the word. A problem is and should always be treated as an opportunity to learn from your mistakes and acquire new skills. At the end of the problem-solving process, seeing results in front of you can be one of the most exhilarating feelings in the world.
# 
# With philosophy out of the way, imagine yourself in the shoes of an aspiring data scientist who is working for a battery manufacturer. Let's say that your employer wants to study the distribution of battery capacity in commercial electronic products in the hopes of gaining some insight which could be used to gain a competitive advantage. This is a huge project, which depending on the project manager, could be split up into any number of smaller projects. For the sake of simplicity, let us assume that your project manager decided to split commercial electronics with batteries into segments such as  phones, IoT devices, kitchen devices, etc. It just so happens that you are given the task of analyzing the capacity of batteries in phones to gain any potential insight.
# 
# This is one of many reasons for why you would want to build a crawler assembling this dataset. The spec sheet for a device on GSMArena is incredibly rich and goes far beyond battery capacity and the above scenario is just one of many examples of what we can do with it. 

# ### Defining the Data

# We want to collect all the data we can about a phone from GSMArena. Cleaning and making sense of the data is the next step and will be covered in detail in the next tutorial.
# 
# To understand what we are dealing with, let us look at an example of the webpage (i.e the spec sheet) which we will be crawling. We will take a look at the GSMArena entry for a phone. The example that we will use for the tutorial is the newly released [Samsung Galaxy S10](https://www.gsmarena.com/samsung_galaxy_s10-9536.php). 

# ![Galaxy S10 spec sheet ](https://i.imgur.com/DT6KEQp.jpg )

# Looking through this example, it is very apparent what type of information we want to grab. We pretty much want the crawler to scrape whatever is available in the banner (the green region in the above image) and the specifications (specs) sheet (the red region in the above image). Along with this data, we also want to grab the *Total user opinions* (*see image below*) at the bottom of the webpage.

# ![Total user opinions](https://i.imgur.com/m54SzpW.jpg)

# Keep in mind that in this world where data is the new gold, you can always find data for interesting analysis wherever you look. As an example, along with or instead of scrapping just the specs, a motivated reader of this tutorial could add the functionality to scrape all the user opinions (i.e comments) for a given phone and perform sentiment analysis on it to test whether the public opinion of a phone is positive or negative and compare it with comments from YouTube video of the phone. However, for the purposes of this project, we will only be analyzing the specs and leave the comments alone. Always remember that the possibilities are endless.

# **A NOTE ON BIAS**

# Go through a few more examples until you feel comfortable with the information that you have to collect. But, do not look at more than a few examples. We are all human and get a little lost in exploring data which interests us. But,  **observation of the data biases the result**. It is easy enough to follow best practices such as the creation of test, train and development sets of the data such that you only see the development set when you have well-structured data (such as one from a CSV file). However, when web crawling, you inevitably have to take a close look at at least a few examples of your data to understand the best way to parse it from the webpage. This is a very deep topic that could take an entire module to cover and I urge you to do you own reserach about it. To understand more about bias in ML models and how to avoid it, check out [this](https://towardsdatascience.com/preventing-machine-learning-bias-d01adfe9f1fa) article.
# 
# In order to avoid observing the specifc data points in too much detail, we will use the S10 example for building most of the crawlers features.

# ### First Steps Towards Building A Web Crawler

# **Find the Seed Page**

# The seed page is the page that we will be providing to the crawler. Ideally, you only need to provide the crawler with the link for the seed page in order to start the crawler off on its merry way. But depending on your needs, you may have to provide a list, dictionary, DataFrame, etc. In such cases, all the same ideas and concepts apply and all you have to do is modify your approach to meet the problem.
# 
# The seed page is usually the home page of the website. Each page on the website is stored as a directory or file within that home page. In our case '/samsung_galaxy_s10-9536.php' is a webpage within the "directory" 'www.gsmarena.com'.  **However**, I used GSMArena's [list of all mobile brands](https://www.gsmarena.com/makers.php3) as my seed page since I found it to be a lot easier as the starting point. You are more than welcome to write a crawler which uses a different seed page for this same problem. If you are using this tutorial as a template for crawling a different site, look around until you get the right seed page which can link your crawler to all the other pages which may be needed.

# **Examine the Speed Page**

# Examining the maker page, we can see that the list of all brands are arranged alphabetically on a single webpage. In most cases (as we will see), when a page has a number of items (say phones from a particular maker or search results for a product on Amazon), they are split and distributed across multiple webpages and stored as separate links, which can be found on a *nag page* tab. The crawler will have to be able to find all such *nav page links*. We will not worry about it in the case of the maker page, since all the information we want is listed on a single page.
# 
# Just like an explorer in an unknown terrain, we need to assess the HTML structure of the weppage before we attempt to traverse it. I will assume that you are using Chrome (highly recommended) for this and all other tutorials, but I am sure that a simple Google search can help you figure out how to do the inspect the elements of the page for your browser.
# 
# On the list of all makers page linked above, use **Crtl + Shift + I** if you are using Chrome on Windows or ChromeOS and **Option + Cmd + I** if you are on a Mac. Check out [this](https://www.wikihow.com/Inspect-Element-on-Chrome) Wikihow link for more information on how to inspect the elements of your webpage.
# 
# Using the keyboard shortcuts should prompt your browser to go into this cool split screen mode. To make things unabmiguous, I will refer to the normally displayed human readable half of the screen as Screen 1 and the screen which is displaying the html elements as Screen 2.

# There are a few important things to that I want to point out here which I wish I had paid attention to when I started.
# 1. Learn the basics of HTML. You do not need frontend web developer levels skills to develop your first crawler. But a cursory understanding of basic HTML elements like * tags *  and *attributes* will go a long when in helping you to find the data that you need when you are a dozen HTML tags deep. After only about an hour of studying HTML, it felt as though a fog has been lifted and ideas immediately started pouring in on how to work with BeautifulSoup elements. Please refer to the [W3 School HTML Tutorial](https://www.w3schools.com/html/default.asp) for all your HTML needs. They explain everything in noob friendly language and I suggest that **at the very least** you refer to pages on [HTML Introduction](https://www.w3schools.com/html/html_intro.asp), [HTML Basic Examples](https://www.w3schools.com/html/html_basic.asp), [HTML Elements](https://www.w3schools.com/html/html_elements.asp) and [HTML Attributes](https://www.w3schools.com/html/html_attributes.asp). If time permits take a look at [HTML Links](https://www.w3schools.com/html/html_links.asp), [HTML Tables](https://www.w3schools.com/html/html_tables.asp) and [HTML Lists](https://www.w3schools.com/html/html_lists.asp), since we will be using these tags A LOT to find the information we want from webpages. This was enough to get me started but we will be revisiting this topic in future modules.
# 
# - **USE** the *Inspect Element* tool. When this toggle is turned on, and you hover the mouse over an element that you want to inspect on Screen 1, the parent tag corresponding to that element will be automatically opened on the HTML tree displayed on Screen 2. This mode can be toggled by clicking the first icon (the one which looks like an arrow in front of a screen) on the top right of Screen 2. You can also toggle it using the keyboard shortcut **Crtl + Shift  + C**. I cannot emphasize how much I wish I had learned to use this tool sooner for finding the HTML structure of the element that I want from Screen 1.
# 
# - Using the *toggle device toolbar*, if needed. The toggle device toolbar allows us to change the view between desktop mode and mobile device. Throughout this tutorial, I will be using the desktop view, but your needs may change if the webpage that you are looking to crawl for information supports certain mobile only functions. This mode can be toggled by clicking the second icon (the one which looks like a mobile phone and tablet) on the top right of Screen 2. You can also toggle it using the keyboard shortcut **Crtl + Shift  + M**.
# 
# **NOTE:** From this point forward, I will assume that you are referring back to the Webpage HTML using *Inspect Elements* as and when needed to understand the structure of the elements.

# **Step 1: Get the list of makers from the seed page**

# ![Seed page insepct element](https://i.imgur.com/r67Gy60.jpg)

# ![Maker structure](https://i.imgur.com/GzcUk3X.jpg)

# Taking a look at the maker page on Screen 2, we see that the information regarding the maker name and the link to their page (example of [Samsung's](https://www.gsmarena.com/samsung-phones-9.php) page) is stored as a table under a *div-tag* with a *class* name 'st-text'. There are three pieces of information that we can grab from each row in the table (*tr-tag*).
# 1. **maker_name** (Samsung)
# 2. **num_devices** (1174 devices)
# 3. **maker_link**('samsung-phones-9.php') | visit [this](https://doepud.co.uk/blog/anatomy-of-a-url) to get an understanding of how an url is structured
# 
# To get these three values we need to iterate over the rows of the table. The maker link is stored as the value of the attribute *href* of the "*dictionary*" *a-tag*. The maker name is stored as the text of the *a-tag*. Finally, the number of devices are stored as the text of a *span-tag*.
# 
# We first make an HTTP request to get the maker page (seed page) using the **urllib.request** module and then use **BeautifulSoup** to find the tags where the information we want is stored. To parse and extract specific information from a tag's text, we use the Python Regular Expression (RegEx) module **re**. Since there have already been great tutorials on using all these libraries, I direct you to their documentation and also to tutorials that I found useful to gain an understanding of their attributes and methods.
# 
# - Regular Expression
#   - [re Documentation](https://docs.python.org/3/library/re.html)
#   - A very through [tutorial](https://www.youtube.com/watch?v=sa-TUpSx1JA) tutorial on RegEx in Python by Corey Schafer. All his tutorials are great and I highly recommend that you check his content out.
# - BeautifulSoup 
#   - [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#)
#   - Corey's [video](https://www.youtube.com/watch?v=ng2o98k983k) on BeautifulSoup
# - urllib
#   - [urllib.request Documentation](https://docs.python.org/3/library/urllib.request.html#module-urllib.request)
#   - [Video] on urllib by Socratica
#   
#   
# If you do not have familiarity with the above modules, my suggestion is to watch the videos at 1.5x speed and then refer back to this tutorial. I know that this is a lot of prerequisite  to cover, but learning this stuff is well worth it. In particular the BeautifulSoup documentation is very accessible and it will be worth your time to go through it even though we will only be using a handful of its methods and features. **Note** that **re**, [**os**](https://www.youtube.com/watch?v=tJxcKyFMTGo&t=301s), [**numpy**](http://www.numpy.org/) and [**time**](https://docs.python.org/3/library/time.html) are not very important for this tutorial, but we will be using them a lot for the upcoming tutorials and I suggest that you take the opportunity to learn about them.

# **A NOTE ON BUILDING IN PARTS**

# When it comes to a large project like this and what will follow, there will be a lot of lines of code involved. In the beginning, it was really difficult for me to wrap my head around all the functions that the program would have to perform. This is an area where Jupyter Notebooks became very useful. You can build up your program function by function and line by line without having to execute everything from scratch. Especially in the case of a web crawler, sending out HTTP requests and getting back responses can be the most time-consuming block of the code. In such cases, using the *cells* of a Jupyter Notebook becomes very handy. 
# 
# We want to build up our program in pieces. If we put everything inside a **for** loop, we will eventually find it very difficult to understand what is going on and debugging the code becomes a nightmare. The strategy I use is to split up the problem into sensible blocks (1. get a list of all makers, 2. get a list of all nav pages for a maker, etc..) and then define functions which perform the vebose tasks that needs to be performed on the blocks. This makes the process of debugging, refactoring and repurposing your code much easier. The next step would be to create a Python **class** that incorporates all these functions. To keep the length and scope of this tutorial manageable, we will not be defining a class for the crawler. I leave it up to the motivated reader to do so. Note that I will, however, be following up on this over at my [GitHub](https://github.com/vigvisw) page in the near future. 
# 
# A few useful videos to get you started are:
# 1. Functions
#   - [Python Tutorial for Beginners 8: Functions](https://www.youtube.com/watch?v=9Os0o3wzS_I) by Corey Schafer
#   - [Python Functions](https://www.youtube.com/watch?v=NE97ylAnrz4) by Socratica
# 2. Classes
#   - [Python OOP Tutorial 1: Classes and Instances](https://www.youtube.com/watch?v=ZDa-Z5JzLYM) by Corey Schafer
#   - [Python Classes and Objects](https://www.youtube.com/watch?v=apACNr7DC_s) by Socratica

# In[ ]:


# import the require libraries 
from bs4 import BeautifulSoup
from urllib import request
import re
import numpy as np
import time
import os

# uncomment if you are using this on Colab
# from google.colab import files


# the domain name of the website we are crawling is a global variable used by the crawler
domain_name = 'https://www.gsmarena.com/'


# In[ ]:


# if using Google Colab, I use a list to collect the log of any issues
# you give give a shot at refactoring this function to use the logging module
debug_collector = []

def collect_debug(error):
  '''A function for logging any unexpected behaviour'''
  global debug_collector
#   print(error)
  debug_collector.append(error)


# In[ ]:


# STEP 1

# get meta data and links to all the makers in GSMArena  
def get_maker_links(url):
  '''A function for getting links to all makers from the GSMArena makers list.
 
     Takes in the url of the list of makers and returns a list of lists of the form
     [[maker_name_1, maker_name_1, num_devices_1, maker_link_1],.....]
  '''
  # get the maker page and read it
  page = request.urlopen(url)
  html = page.read()
  
   # create the BeautifulSoup(bs) object 
  bs = BeautifulSoup(html, 'html.parser')

  # find the div-tag which contains the table
  table = bs.findChild('div', class_='st-text').table

  # if there is no table in the seed page, log it 
  if not table:
    error = 'Maker Page Error: has no brands table| function_name: {}| url: {}'.format(function_name, url)
    collect_debug(error)
  # if the table is present then get the the maker information from it
  else:
    # inside each table, the data in is maker is stored under td-tag which we collect using a list
    rows = table.findChildren('td')
    rows_collector = []
    # takes in data as [index, maker_name, link, #phones]
    for maker_id, row in enumerate(rows):
    # get the maker name and maker link. if there is no a-tag, collect a log
      row_a_tag = row.a
      if not row_a_tag:
        error = 'Maker Page Error: no row_a_tag| function_name: {}| url: {}| row_num: {}'.format(function_name, url, n)
        collect_debug(error)
      else:
        maker_link = domain_name + row_a_tag['href']
        # use the stripped_strings generator to get a tuple of the maker name and num of devices 
        # if you are wondering what is going in the line above, please check out comprehensions in Python
        maker_name, num_devices = (item for item in row_a_tag.stripped_strings)

        # extract the numerical portion of num_devices and convert it into an integer
        num_devices = re.findall(re.compile('\d+'), num_devices)[0]
        num_devices = int(num_devices)
        # append all of the maker data to the rows_collector and return it
        rows_collector.append([maker_id, maker_name, num_devices, maker_link])
  return rows_collector


# The function above represents one of the blocks of code that we want the crawler to execute. It takes in the url for the seed page and returns a list of all the makers in GSMArena.  

# **A NOTE ON THE CODE**

# 
# 1. Always test out a function before passing it along to something else. It can become extremely difficult to debug when you are a few functions deep.
# 2. The code I use in the tutorials might not represent the optimal solution since the goal here is the get the task done and then worry about improving the speed and readability. I encourage you to recycle and refactor what is written here for your own projects.

# In[ ]:


seed_path = 'makers.php3'
seed_url = domain_name + seed_path 

# test out the function that we just created
maker_list = get_maker_links(seed_url)
print(maker_list)


# When it comes to crawling webpages for data, the name of the game is iteration. If we can figure out how to crawl across one of the maker's in **maker_list**, we should be able to iterate through every maker in the list. We will use the maker [**Samsung**](https://www.gsmarena.com/samsung-phones-9.php) for this example. (*No they are not sponsoring me. I just really like the S10's design and hence chose it for this example*). 
# 
# Let us quickly design a few useful functions. There are many ways to expand on the functionality of the crawler and modularize certain functions. For the sake of keeping the tutorial at a reasonable length, I leave it up to you to experiment further.

# In[ ]:


# define a function for giving us the nav_page_num 1 of a maker given their name
def get_makers_link(maker_name):
  '''A function for getting the link to a maker given the maker's name.
  
     Takes in a maker's name, say 'Samsung' returns 'https://www.gsmarena.com/samsung-phones-9.php'
     Maker name is case insenstive.
  '''
  # go through the maker_list and find list_item[1], i.e maker_name
  global maker_list
  if not maker_list:
      error = 'No maker_list!'
      collect_debug(error)
  else:
    for list_item in maker_list:
      if maker_name.lower() == list_item[1].lower():
        print('maker_link called for {} \n{}'.format(maker_name, list_item[-1]))
        
  
# test it out
maker_name = 'Samsung'
get_makers_link(maker_name)

maker_name = 'samSung'
get_makers_link(maker_name)


# In[ ]:


# since we call a webpage and get the bs object of the page a lot, we can define a function to make it easier  
def get_bs(url, parser='html.parser'):
  '''A function for returning the BeautifulSoup object of a webpage given its url
  
     Uses 'html.parser' by defualt and can be modified using the optional argument 'parser'
  '''
  # return the bs onject for a given webpage
  page = request.urlopen(url)
  html = page.read()
  bs = BeautifulSoup(html, parser)
  return bs


# ### Expanding The Capability Of The Web Crawler

# ** A NOTE ON THE RENDERED PAGE**

# 1. Some of the "phones" listed in GSMArena are smart-watches, tablets, and other devices. Our crawler will be device agnostic and grab data on all the devices.
# 2. Another convenient reason for why I chose Samsung is that it has the most number of devices in **maker_list**. Below we will define a function to check and verify this, but is is not necessary for the crawler to work.
# 3. The 1174 devices listed under Samsung are spread across 14 pages. We can see this at the very bottom of the webpage (*see image below*). The crawler we make has to traverse each of those pages. Ideally, you have to account for a clickable JavaScript element to get a list of all the pages. Since Samsung has, by far, more devices than other manufacturers, and consequently the maximum number of such **nav_pages** all listed on its **maker_page**, we do not have to worry about this. A quick examination using *Inspect Elements* shows us that all the **nav_page** links are clearly structured in the page HTML and hence are easy to extract.

# ![Samsung page](https://i.imgur.com/r6e3BiR.jpg)

# ![Nav links html](https://i.imgur.com/dU3F1pq.jpg)

# **A NOTE ON THE WEBPAGE HTML**

# **Phone**
# 
# 1. Inside a maker's webpage, the data about the device is stored inside a *div-tag* with the class name 'makers' in the form on an unordered list. 
# 
# 2. Each device on the page is a list item with an *a-tag* with the link (which we want) to the device. The *a-tag* has two child tags. 
# 
# 3. The thumbnail for the device is stored inside an *img-tag*. The *img-tag* has two attributes *'scr'* and *'title'*, which we want.
#  
# 4. The text of the **strong-tab** is name of the device, which we want.
# 
# 
# **Nav Pages**
# 1. All the information we want about the **nav_pages** is stored inside a *div-tag* with the class name 'nav-pages'.
# 2. The landing page for the maker is 'Page 1' and hence has no hyperlink.
# 3. All the other **nav_pages** for a given maker are stored inside the *div-tag* as child *a-tags* with the link accessible as the value of the *'href'* attribute, which we want.
# 
# Thinking your way through the HTML tree of the webpages that you are trying to crawl across and gaining familiarity with the layout of the information on the page is very important and I advise that you spend a few minutes on this before you write any code.
# 
# My approach to developing this crawler was to first visit a given maker in the **maker_list** and then create a dictionary with all the **nav_page** numbers and their links, including the maker's landing page. We will then use this dictionary to visit all the **nav_pages** and collect the information that we want about the phones. You are more than welcome to add your own twist to this. 
# 
# We will prototype this portion of the crawler using Samsung and then iterate through all the makers in **maker_list**

# In[ ]:


# since Samsung is the maker we want from maker_ist, we write a function to get the largest maker
def get_largest_maker():
  '''A function that returns the maker data corresponding to largest maker from the maker_list'''
  global maker_list
  # compare and set the num_devices under each maker against this variable if num_devices > is_largest
  is_largest = 0
  maker_id = None
  # iterate through all the maker's in maker_list and return the largest maker
  for maker in maker_list:
    num_devices = maker[2]
    if num_devices > is_largest:
      is_largest = num_devices
      maker_id = maker[0]
  return maker_list[maker_id]

# test it out
maker = get_largest_maker()
print(maker)


# In[ ]:


# STEP 2

# iterate through each maker in the maker list and apply this function over the maker to get the nav_page_links
def get_nav_page_links(maker):
  '''A function for getting all the nav pages under a given maker
  
    This function takes in a maker_list item of the form [maker_id, maker_name, num_devices, maker_link]
    Returns a dict of the form {nav_page_num:nav_page_link} for the maker
  '''
  # unpack the items in the list
  maker_id, maker_name, num_devices, maker_link = maker
  # a dictionary that will be used to collect all the nav_pages for a given maker
  maker_nav_pages = {}
  # first add the landing page as nav_page_num = 1
  maker_nav_pages[1] = maker_link
  # get the maker's page
  bs = get_bs(maker_link)

  # find the div-tag containing the nav_pages
  nav_pages = bs.findChild('div', class_='nav-pages')
  # if the maker has no nav_pages, which is possible, collect it for logging
  if not nav_pages:
    error = '{} does not have nav_pages| maker_link: {}'.format(maker_name, maker_link)
    collect_debug(error)
  # otherwise we can get a list of all the nav_pages 
  else:
    # insde this div tag, the pages we want are inside a-tags
    nav_pages = nav_pages.findChildren('a', recursive=False)
    for nav_page_num, nav_page in enumerate(nav_pages):
        # nav_page_num needs to be offset by 2 before using as a key to add the nav page link
        maker_nav_pages[nav_page_num + 2] = domain_name + nav_page['href']
  return maker_nav_pages

# test it out
nav_page_links = get_nav_page_links(maker)  
print(nav_page_links)  


# Once the **nav_pages** for a maker (Samsung in this case) have been collected, we can collect the device info and most importantly the **device_link** for every device by that maker.

# In[ ]:


# STEP 3

# get the information about the devices present in a nav_page by iterating through the all the devices on that page
# all the devices by a given maker are collected as elements in dictionary of the form
# {Samsung:[device_1_data, device_2_data,.......], Acer:[device_1_data, device_2_data,....],....}

# we also need to define a couple of global variables, which are results from the earlier functions
devices_collector = {}
maker_name = maker[1]
maker_link = maker[-1]

# for nav_page in nav_pages, we will iterate through this function
nav_page_links = get_nav_page_links(maker) 

# collect all devices by the makers in the devices_collector dict using maker_name's as the key
devices_collector[maker_name] = [] 
def get_device_links(nav_page_link, devices_collector, maker_name):
  '''A function to get the device links and device info for all devices in a nav page
     This function will be called for every device in GSMArena when used with the crawler
  '''
  # unpack the 
#   global devices_collector, maker_name
  # get the nav_page
  bs = get_bs(nav_page_link)

  # get the list items under the div-tag with the class name 'makers'
  devices = bs.findChild('div', class_='makers').ul
  devices = devices.findChildren('li', recrusive=False)

  # iterate through each device and collect the device_name, device_info, device_img_link, device_link
  page_device_collector = []
  for device_num, device in enumerate(devices):
    device_name = device.get_text()
    # we cannot collect the link for a device if it does not have an a-tag
    if not device.a:
        error = "{} does not have a link : nav_page: {}| maker_name {}: ".format(device_name, nav_page_link , maker_name)
        collect_debug.append(error)
    else:
      device_link = domain_name + device.a['href']
      # img_link, and title are stored in the img tag
      img_tag = device.a.findChild('img')
      if not img_tag:
        error = "{} does not have a img_tag| nav_page: {}| maker_name: {}".format(device_name, nav_page_link , maker_name)
        collect_debug.append(error)
      else:
        device_img_link = img_tag['src']
        device_info = img_tag['title']
    page_device_collector.append([device_name, device_info, device_img_link, device_link]) 
  # concat the device info from this nav page onto what is already present on the list
  devices_collector[maker_name] += page_device_collector

# test it out 
for nav_page_num, nav_page_link in nav_page_links.items():
  get_device_links(nav_page_link, devices_collector, maker_name)
  
print(devices_collector.keys())
print(devices_collector['Samsung'])
print(devices_collector['Samsung'].__len__())


# ### Gathering Data From the Endpoint

# Now that we have the functionality to find all the device information as well the **device_links** for the devices under a maker, the last component that we need is for the crawler to be able to visit a device and collect all the information for a device from its **banner** and **spec_sheet**. First, let us use *Inspect Elements* to examine the device page for the [Samsung Galaxy S10](https://www.gsmarena.com/samsung_galaxy_s10-9536.php)

# ![S10 html](https://i.imgur.com/q1ixL1Q.jpg)

# Use the same procedure that I described in the above section to dig deeper into the HTML tags to find the information that you need. 
# 
# The device data stored in the banner is mostly a repeat of what is in the spec sheet, but I will grab it regardless because calling the webpage itself is the most time-consuming process. The **banner** also contains the **web_hits** and the **popularity** of given device. I did not find the 'Become a Fan' attribute interesting or rich enough for analysis, so I did not grab it. 
# 
# Use *Inspect Elements* and the comments in the code to help you break down the HTML structure for the **banner** and the **spec_sheet**.

# In[ ]:


# get the data for Samsung Galaxy S10 (devices[3])so that we can build a sample crawler for a device 
devices = devices_collector['Samsung']
device = devices[3]

# to get the device information, the functions take in each of these device info as attributes
device_link = device[-1]
device_name = device[0]

# a collector dict which consolidates all features, including the banner for a device
specs_collector = {}


# In[ ]:


# get the spec_sheet for a device from the device_link
def get_device_specs(bs, specs_collector, device_name, device_link):
  '''A function for findinf the specs tabele of a device given a bs object of the device webpage'''
  # the specs are stored inside individual tables, so find them all
  specs_tables = bs.findChildren('table')
  if not specs_tables:
    error = '{} has no specs_tables| device_link: {}'.format(device_name, device_link)
    collect_debug.append(error)
  # if the phone does have a spec-list
  else:
    # get the spec category like 'Network', 'Launch', 'Memory', 'Battery', ...
    for table in specs_tables:
      # find all the rows in the table 
      table_rows = table.findChildren('tr', recursive=False)
      # each table will only hacve one child th-tag, i.e a header
      # this header of the table is the name of the spec
      table_header = table.findChild('th').get_text(strip=True)

      # for row in tables: if the class = 'ttl' or 'nfo', it is a column in the table
      # 'ttl' tags correspond to a potential feature that we could extract such as Dimension, Weight, Date Announced, etc..
      # 'nfo' coresponds to a actual data point corresponding to the 'ttl' feature
      ttl_collector = {}
      for row_num, row in enumerate(table_rows):
        ttl_tag = row.findChild('td', class_='ttl')
        nfo_tag = row.findChild('td', class_='nfo')

        # if neither the ttl_tag or nfo tag are present, we want to log it
        if (not ttl_tag) or (not nfo_tag):
          error = '{} has ttl-tag OR nfo-tag| device_link:{}'.format(device_name, device_link)
          collect_debug(error)
          # we also want to set the text to NaN if a column is empty so that it can later be processed easily 
          ttl_tag_text = np.NaN
          nfo_tag_text = np.NaN
        # if either the ttl_tag or nfo tag are present, we want to collect them and log any missing values
        else:
          if not ttl_tag:
            error = '{} has no ttl-tag| device_link:{}'.format(device_name, device_link)
            collect_debug(error)
          else:
            ttl_tag_text = ttl_tag.get_text(strip=True)
            if ttl_tag_text == '\xa0' or ttl_tag_text == '':
              ttl_tag_text = np.NaN

          if not nfo_tag:
            error = 'No nfo-tag: {}: {}: {}'.format(n, link, row)
            collect_debug(error_mess)
          else:
            nfo_tag_text = nfo_tag.get_text(strip=True)
            if nfo_tag_text == '\xa0' or nfo_tag_text == '':
              nfo_tag_text = np.NaN
        # add the values of the ttl-tag and nfo-tag as key value pairs
        ttl_collector.setdefault(ttl_tag_text, nfo_tag_text)
      # add the table header and the collected attribute value pairs to the specs_collector
      specs_collector.setdefault(table_header, ttl_collector)
      
      
# test it out
bs = get_bs(device_link)
get_device_specs(bs, specs_collector, device_name, device_link)
for key, value in specs_collector.items():
  print('{} : {}'.format(key, value))


# In[ ]:


# ge the device banner data and add it to the specs_collector with the key 'Banner'
def get_device_banner(bs, specs_collector, device_name, device_link):
  '''A function to scrape data from the banner of a a device'''
  # get the unordered list with the class name 'specs-spotlight-features'
  banner = bs.findChild('ul', class_='specs-spotlight-features')
  # if a banner is not present, collect the information for dbugging
  if not banner:
    error = '{} has no banner| device_link:{}'.format(device_name, device_link)
    collect_debug(error)
  # else get all the list items and find the data stored in the banner
  else:
    banner_items = banner.findChildren('li')
    banner_specs_collector = {}
    for list_item in banner_items:
      # find all the items in the list falling into the data-spec category, such as battery-hl, screen-hl, etc...
      banner_specs = list_item.findChildren(['span', 'div'], {'data-spec':re.compile('.*')})
      # for each spec in the banner iterate through the key value pairs and add it to banner_spec_collector
      for banner_spec in banner_specs:
        banner_spec_name = banner_spec['data-spec']
        if banner_spec_name:
          # setting strip = True removes any white space space characters
          banner_spec_value = banner_spec.get_text(strip=True)
          if banner_spec_value:
            banner_specs_collector[banner_spec_name] = banner_spec_value

      # we now need to find the device popularity and hits from the webpage
      if 'help-popularity' in list_item['class']:
        # get information about the device's popularity and collect debug if it does not have the attribute
        device_popularity = list_item.findChild('strong')
        if not device_popularity:
          error = '{} has no device_popularity| device_link:{}'.format(device_name, device_link)
          collect_debug(error)
        else:
          device_popularity = device_popularity.get_text()
          # do no capture the Unicode white space character '\xa0'
          if device_popularity == '\xa0' or device_popularity == '' :
            device_popularity = None        
        # collect information about the device's popularity and collect debug if it does not have the attribute          
        device_hits = list_item.findChild('span')
        if not device_hits:
          error = '{} has no device_hits| device_link:{}'.format(device_name, device_link)
          collect_debug(error)
        else:
          device_hits = device_hits.get_text()
          if device_hits == '\xa0'or device_hits == '':
            device_hits = None

        # add device_popularity and divice_hits to the banner_specs_collector if they are present
        if device_popularity:
          banner_specs_collector['device_popularity'] = device_popularity
        if device_hits:
          banner_specs_collector['device_hits'] = device_hits
    specs_collector['Banner'] = banner_specs_collector

# test it out
get_device_banner(bs, specs_collector, device_name, device_link)
for key, value in specs_collector.items():
  print('{} : {}'.format(key, value))
  
# we can see that the Banner has now been successfully added to the specs_collector


# In[ ]:


# the last thing we want to grab from the device page is the 'Total user opinions' at the bottom of the page
def get_device_opinions(bs,specs_collector, device_name, device_link):
  '''A function to get the 'Total user opionions' for a device form the devie pages bs object'''
  opinions = bs.findChild('div', id='opinions-total')
  if not opinions:
    error = '{} has no Total user opinions| device_link:{}'.format(devie_name, device_link)
    collect_debug(error)
  else:
    num_opinions = opinions.b.get_text(strip=True)
    specs_collector.setdefault('Opinions', num_opinions)
  

# test it out
get_device_opinions(bs, specs_collector, device_name, device_link)
get_device_banner(bs, specs_collector, device_name, device_link)
for key, value in specs_collector.items():
  print('{} : {}'.format(key, value))


# In[ ]:


# put everything we have made so far for collecting the specs of a device into a single function
def get_device_data(device_link):
  '''A function to get the banner data and spec-sheet from a device on GSMArena
  
     Takes in a device's url and returns a dict with all the specs
  '''
  specs_collector = {}
  # get the devie bs object
  bs = get_bs(device_link)
  device_name = bs.findChild('h1', class_='specs-phone-name-title').get_text()
  # get te device_specs
  specs = get_device_specs(bs,specs_collector, device_name, device_link)
  # get the banner using the get_device_banner method, defined below
  banner = get_device_banner(bs, specs_collector, device_name, device_link)
  # get the user opinions for the device
  opinions = get_device_opinions(bs,specs_collector, device_name, device_link)
  # get the banner spec_sheet using the get_device_specs method, defined below 
  return specs_collector

    
# test it out
device_link = 'https://www.gsmarena.com/samsung_galaxy_s10-9536.php'
get_device_data(device_link)


# ###Putting it All Together

# We now have all the components required to build the final crawler. Two things to decide before we can put everything together are:
# 1. **Crawl Strategy**: This is how we actually want to proceed with the crawl. We do not want to go full Inception on the crawler and go a link within a link within a link. This will eventually lead to us getting lost in nested loops with no way to debug our code. My preferred crawl strategy for this particular crawler is as follows.
# 
# > Get the **maker_list** from the seed page.
# 
# >For each **maker** in the **maker_list**.
# 
# 
# >> For each **device** in a **maker**. 
# 
# >>> Get all the device information on a device.
# 
# >> Return a **dict** with the device info and **device_link** to all devices on GSMArena using the **maker_name** as key.
# 
# > For **all** devices in dict above find the spec sheet, banner, and opinions.
# 
# >> Append the spec sheet to the device info in the **dict** from above.
# 
# > Return the dict as **devices_collector**.
# 
# 2. **Data Storage**: This is how the information about all the phones will be stored. You are welcome to develop your own strategy for this, but my preferred method is to use the structure of the form **{maker_1_name: [device info, spec_collector], ...], maker_2_name:[.......]., ....}**. This will also allow for easy conversion of the data we scrape into any format we want such as a JSON object.

# I also defined two functions which helps us get a list of devices under a particular maker because the code was taking too long to run as descriibed earlier. You can try to crawl for all the sites, but this might fail for you on the two hour mark if your using Colab. Give it a shot and let me know how it does.
# 
# You can also download this file as an [IPython](https://ipython.org/) notebook by using *File > Download .ipynb*. You can then run this notebook on you local Jupyter Notebook environment or try connecting Colab to a [local runtime](https://research.google.com/colaboratory/local-runtimes.html).
# 

# In[ ]:


# helper function that allows returns the maker_id of of maker
def get_maker_id(name_of_maker, maker_list):
  '''A function for returning the maker_id of a maker given the maker_name.
     
     This function is case insensitive.
  '''
  for maker in maker_list:
    name = maker[1]
    maker_id = maker[0]
    if name_of_maker.lower() == name.lower():
      return maker_id
  # if a name is not found in the maker list, we want to throw an exception and collect it for log
  raise NameError('GSMArena has no maker \'{}\''.format(maker_name))
  
  
def switch(maker_id, name_of_maker, maker_list):
  '''A function for returning a bool which tells the crawler which maker(s) to scrape for data.'''
  # if no name_of_maker is given, return true in all cases
  if name_of_maker is None:
    return True
  # else get maker_id for the given maker name and return True only when current maker_id == given maker_id
  else:
    given_maker_id = get_maker_id(name_of_maker, maker_list)
    if given_maker_id == maker_id:
      return True
    else:
      return False


# In[ ]:


# assemble the functions we built earlier in the right format in order to get the functionality we want

def GSMCrawler(seed_url, name_of_maker=None):
  '''A crawler to return device data from GSMArena
      
     Takes in the seed_url 'https://www.gsmarena.com/makers.php3'.
     If name_of_maker is specified, device info for will be collected only for that maker.
  '''
  # we want to measure how long the crawling took to excecute
  start_time = time.time()
# STEP 1: get the links to all the makers in GSMArena
  print('Starting GSMArena Crawler...\n')
  maker_list = get_maker_links(seed_url)
#   maker_list = get_maker_links(seed_url)
  print('Successfully retrived maker_list!\n')
  
  # tell us if we the crawl is being done for a single maker or all makers
  if name_of_maker is None:
    print('Crawling for devices by ALL makers...\n')
  else:
    print('Crawling for devices by {}...\n'.format(name_of_maker))
  
# STEP 2: iterate trough each maker and get the device links and device info from all the nav pages
  devices_collector = {}
  for maker_id, maker in enumerate(maker_list):
    if switch(maker_id, name_of_maker, maker_list):
      maker_link = maker[-1]
      maker_name = maker[1]

# STEP 3: the first thing we want to do on the makers page is to get a list of all nav links
      nav_pages_links = get_nav_page_links(maker)
      # for each nav page in a maker's nav_pages, get the device info for all devices by that maker
      print('Getting nav_page_links for {}...\n'.format(maker_name))
      devices_collector[maker_name] = []
      for nav_page_num, nav_page_link in nav_pages_links.items():
        get_device_links(nav_page_link, devices_collector, maker_name)
      print('Successfully collected all device info for {}!\n'.format(maker_name))
      
  # notify us of how many devices were collected in total 
  total_num_devices = 0
  for maker, devices_info in devices_collector.items():
    total_num_devices += devices_info.__len__()
  print('Successfully collected info for all devices! {} devices were collected\n'.format(total_num_devices))

# STEP 4: go through each each device_link in the devices_collector and pass it onto get_device_data
  print('Collecting spec sheets for all devices. This could take a while. Sit back and relax...\n')
# WARNING: This loop will scrape the spec sheet of every device in GSM Arena.
 # it is good practice to put this under a try block; in case some thing we want to collect some debug info
  try:
    for maker, devices_info in devices_collector.items():
      print('Getting spec sheets for {} devices by {}\...n'.format(devices_info.__len__(), maker))
      for device_num, device in enumerate(devices_info):
        device_link = device[-1]
        # get the device data using the get_device_data function we defined earlier
        device_specs =  get_device_data(device_link)
        device.append(device_specs)
# WARNING END
      print('Successfully scraped info for all devices by {}\n!'.format(maker))
  
  except Exception as e :
      error = 'Device crawl exception: {}| device_name: {}| device_links:{}\n'.format(e, device_link)
      collect_debug(error)
  # if nothing went wrong, let us know that all has gone well
  else:
      end_time = time.time()
      print('GSMCrawler has completed excecuting! All credits for this data goes to the GSMArena team\n')
      print('Time time required to excecute for {}: {} seconds'.format(end_time - start_time))
      print('Time time per : {} seconds'.format(end_time - start_time))
      print('='*50)
  finally:
    # finally return the data_collector
    return devices_collector


# In[ ]:


# try out our newly built crawler

seed_path = 'makers.php3'
seed_url = domain_name + seed_path 

# due to Colab's limitations, I will run the crawler only for Samsung
# you can find the data the full set of devices on my GitHub page under the name devices_data.json
devices_collector = GSMCrawler(seed_url,'Samsung')

# uncomment the code below to run the crawler for the full site
# devices_collector = GSMCrawler(seed_url)

print(devices_collector.keys())
print(devices_collector['Samsung'])


# And just like that, our crawler is complete!
# 
# The last step is to convert the data in **devices_collector** into a JSON object and save or download it (if you are on Colab)

# In[ ]:


# we want to convert the data that we just collected into a JSON oject to interact with later 
def make_devices_json(devices_collector, save_json=False):
  '''A function for coverting devices_collector text into a JSON obj and optionally saving the file
     If save_json is True, a file called devices_data.txt will be made in your current working directory
  '''
  json_dict = {}
  for maker, devices_info in devices_collector.items():
    maker_dict = {}
    for device_id, device in enumerate(devices_info):
      device_dict = {}
      device_name, device_info, device_img_link, device_link, device_specs = device

      # start adding data as key value pairs into the device_dict
      device_dict['device_name'] = device_name
      device_dict['device_info'] = device_info
      device_dict['device_img_link'] = device_img_link
      device_dict['device_link'] = device_link
      device_dict['device_specs'] = device_specs


      # use the device_id as key to to set the device
      maker_dict[device_id] = device_dict
    # set the maker id to the json_dict with the maker name as key
    json_dict[maker] = maker_dict
    
  # if save json is true, then save the devices collected by the crawler in the working directory as a json file
  if save_json:
    cwd = os.getcwd()
    save_file_name = cwd + '/devices_data.txt'
    with open(save_file_name, 'w', encoding='utf-8') as file:
      json.dump(json_dict, file, ensure_ascii=False)
    # notify us where the file was saved
    print('Successfully saved device data as a JSON file at {}'.format(save_file_name))
    
  return json.dumps(json_dict, ensure_ascii=False)
  
# test it out
devices_json = make_devices_json(devices_collector, save_json=True)
devices_json


# In[ ]:


# verify that the newly created json file is present in you local directory
get_ipython().system('ls')


# In[ ]:


# if you are using Colab and want to download the file we just created
files.download('devices_data.txt')


# If you are interested in reading in **devices_data** form a JSON file on your device, use the following function.

# You can download the full list of specs that I downloaded for all the makers [here](https://drive.google.com/open?id=1rpefi8CrQMUgs14U_H5rPEpkeYNqCvio).

# In[ ]:


def read_devices_json(file_path):
  '''A function for reading in a JSON obj of the devices data i.e devices.txt
     
     Takes in the string file_path
  '''
  with open(file_path, 'r', encoding='utf-8') as file:
    return json.load(file)

# test it out
cwd = os.getcwd()
file_path = cwd + '/devices_data.txt'
json_dict = read_devices_json(file_path)

# double check that we have a dictionary 
json_dict.keys()


# ### Concluding Remarks

# The way things are set up now, there is a lot of scope for expanding the functionality of the crawler. Building all the base functions required for the crawler to work took up more time than I had anticipated. But I think it was well worth it because there is a lot of potential in this rich dataset. One obvious next step is to implement **classes** for makers and devices. I will follow up on this soon, but we have work to do before that.
# 
# Next week, it will be function city as we will be writing a **LOT** of functions to parse the data we just collected to extract meaningful information from it. The **re** module will be invaluable for this process, so please read up on it if you are following along. We will also be using [**pandas**](https://pandas.pydata.org/) to handle and manipulate the data and [**matplotlib**](https://matplotlib.org/) to visualize it before moving on to [exploratory data analysis](https://towardsdatascience.com/a-gentle-introduction-to-exploratory-data-analysis-f11d843b8184) and hardcore ML in subsequent weeks.
# 
# The last thing I have for you are a few lessons which I learned along the way. I am leaving these here in the hopes that they will "[be a light to you in dark places, when all other lights go out"](https://www.goodreads.com/quotes/140704-may-it-be-a-light-to-you-in-dark-places).

# **Lessons Learned**

# 1. Do not try achieve too much too fast. Always try to create a prototype of the function first before attempting to expand your code's functionality to multiple items.
# 
# 2. Do not change too much too fast. Making too many changes before you run a test of the code can make it difficult to track down the culprit when errors will occur.
# 
# 3. Errors **WILL** inevitably occur. While it can be frustrating, it is still a problem that you can think your way through.
# 
# 5. There will be times when your program will not work the way you want it to. If you have hit a wall during the debugging process, sometimes walking away is the best strategy. Leaving the mind to its own devices and then getting to the block of code has been doing wonders for me and I suggest that you try this out.
# 
# 3. Endless improvements are possible over time, but always try to develop the minimum viable product (MVP) first.
# 
# 4. Do not shy away from learning new things. I do not come from a background in Computer Science. My expertise are in Nanotechnology and Materials Science. Your background can be in anything, but programming is a tool just like any other tool you can learn to use with mastery. However, it is something that can be used to build things faster than most tools due to its accessibility. 
# 
# 6. Programming is not magic. When observing from a third party's perspective any new field can seem overwhelming and hard to break into. Keep chipping away at it until one day, it finally cracks.
# 
# 7. Split time between learning and practicing. I have often been given the advice that practice is the best way to learn how to code. While I agree with this, I think that it is equally important to continuously learn new concepts and ideas just for the sake of it. Being knowledgeable about multiple fields helps you to come up with novel, never before seen, solutions to hard problems.

# This was a fun project to do, and remember that we are only just getting started with this series. Documenting the process certainly changed the way I approached the problem and I had a lot of fun (and sleepless nights) doing it. Regardless of how much attention this tutorial gets, I will be documenting the entire **Truly End-to-End Machine Learning** series and you will be able to find them on my [GitHub](https://github.com/vigvisw) page, as soon as I can finish writing them. I acknowledge that this was a long read, but I wanted to share as much as I could to help any beginners' who are the same shoes that I was in. If you have any questions, queries, or ideas, please feel free to reach out to me at vigvisw@gmail.com.
# 
# Thank you for reading up until the very end and I hope you use the information provided here to build something incredible.
