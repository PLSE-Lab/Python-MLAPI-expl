#!/usr/bin/env python
# coding: utf-8

# 

# The purpose of this kernal is to be able to request it to plot any S & P 500 stock. Because this was written by a novice for a novice (my self) its heavily annotated for my own future reference. I hope that anyone who is also new finds it helpful. All notes refer to the cell below it.
# 
# Note that I am not a programmer by profession. I am a mechanical engineer who is just interested in learning about neural networks. I am familiar with using data for making choices and the first step is analyzing data, so that is what I do here. It seemed a good way to get my feet wet with programing python. As I am familiar with stocks and the information presented with them I decided to use the S&P 500 data set for my first analsys.
# 
# This kernal its heavily annotated to help others learn and for my own future reference. I hope that anyone who is also new finds it a useful guide as i tried to go in to more detail than I had seen in other explains and tutorials. It took me many hours of reading lots of examples and tutorials on kaggle, github, and stackoverflow to make this. I would cite everywhere I read to make this but there's just too many, and I don't remember them all. I would be embarresed to admit how many hours it took to make such a simple program but we all start as newbies and I do not have any formal education in this, all self tought. 
# 
# please feel free to fork this kernal or leave questions and comments on it. I want to learn from those that know and want to help others learn what i have in a hopefully easier manner.

# First step is to load all the modules we will use. modules are chunks of code made by others to make things easy.
# 
# To learn what a module does you just got to search around. Some have great websites that detail how to use them, others... not so much. I often felt frustrated by the lack of clear documentation on a module and its commands. It could be that they where written for advanced users and my knowlege is simply too low to understand what they are saying. Another question you might as is "how do I find a list of modules, or a module that does X/Y/Z? the best answer i can give is just look up how to do X/Y/Z and see what modules people are using. Not a great answer, but thats how I did it.

# In[ ]:


import pandas as pd  #pandas does things with matrixes
import numpy as np #used for sorting a matrix
import matplotlib.pyplot as plt #matplotlib is used for plotting data
import matplotlib.ticker as ticker #used for changing tick spacing
import datetime as dt #used for dates
import matplotlib.dates as mdates #used for dates, in a different way


# we need data to analyize so lets load the S&P 500 data
# 
# the '../input/' is how you call the location of the data on kaggle that you picked

# In[ ]:


allstock = pd.read_csv('../input/all_stocks_5yr.csv') #reads the file


# I then view the columns of the file. I do this for two reasons, one: to make sure it loaded right, two: to see what catagories we have for data.

# In[ ]:


allstock.columns #prints just the columns of the matrix


# always good to see its working and what our columns are. But what if we don't know what all stocks are in the S&P 500? we will want to see them all! this cell pulls all the unique names from the column 'Name' and puts them in a matrix. we then sort the matrix so its alphabetical, and display that matrix.

# In[ ]:


stocknames=allstock.Name.unique() #pulls all unique names from column 'name'
stocknames=np.sort(stocknames,kind='quicksort') #sorts them alphabetically
print(stocknames) #displays the matrix of the names


# I wanted to know if there where cells with missing information. a zero thrown in to the wrong equation could break any future analysis.  we would want to get rid of those few lines. Normally throwing out data is bad but daily information over 5 years is a ton of data points so we can safely toss any bad points if there isn't too many of them. first lets see how many there are. as an aside, it's not needed to sterilize the data just to plot it. but, its a good habit to do it for future analysis.
# 
# The 'total =' line will sum up all the cells that have a 'null' value in them.  the ascending = false is another way of saying put them descending. 
# 
# for percent it divides the number of null cells by the total then multiplied by 100 as its labeled as a percent.
# 
# missing_data is a matrix. 'pd.' means its calling up pandas to make it. that sounds weird, I know but keep in mind Python was named after Monty Pythons Flying Circus, so expect things to get weird. the concat part of it tells it to concentrate the panda elements. so if you had two matrixies it makes them one. we are concentrating total and percentage. axis = 1 tells them to be next to each other and keys is the titles at the top.
# 
# here's where i have to admit some ignorance. this works great and does what we want but i'm not sure why it does it. to be more specific, i don't know what the "total=" line sums them up by column. just reading the line it sounds like it should give the grand total of null cells. more research is needed here. perhaps thats just how it works, which is great for us.
# 
# by the way, if you are ever curious about what a line of code or part of a line of code does or why its there, just delete it or change it and re-run it! I learned a lot by modifying other peoples code this way. keep in mind sometimes you need to restart the entire kernal and then run all as otherwise it will remember variables you commented out.

# In[ ]:


total = allstock.isnull().sum().sort_values(ascending=False) #counts all null cells in a row
percent = ((allstock.isnull().sum()/allstock.isnull().count()).sort_values(ascending=False)*100) #sees what percent of the data is null
missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent']) #combines the two matrixies
missing_data #this displays the matrix


# it found some missing data, oh no! but its okay, its a small amount of data, less than a tenth of a percent is missing. we can toss that and not even feel bad about it. interestingly, when i looked through the data real quick I couldn't even find the null cells so i'm not sure where its missing or why. If I really cared about every data point I would review it manually and come up with a way to fill in the missing bits. The code to drop it all is really simple.
# 
# normally you would have to make a line of code for each catagory (volume, open, low, high, etc) and drop each one. however, I only had to drop Volume and Open. When I first wrote this I would write a line for dropping data, then re-run the above cell (copied to below the drop code) and check that I got it all. I found after doing only Volume and Open that 100% of the null data was removed. Its a happy coincence that shows these data points must have been missing together, ie entirely blank rows. at least thats what I assume as I did not manual check the data sheet to find these null cells. I wonder, is there a way to pull what line in a matrix a null cell was found? I'll have to look in to that in the future, maybe I'll make a v2 of this later.
# 
# For code explanation, the .drop I use with .isnull drops all rows that have a null cell for a particular column.

# In[ ]:


allstock = allstock.drop(allstock.loc[allstock['Volume'].isnull()].index) #drops rows with a null cell in the Volume column
allstock = allstock.drop(allstock.loc[allstock['Open'].isnull()].index) #drops rows with a null cell in the Open column


# lets re-run the matrix showing us how many null cells there where. we are double checking my work. I sure don't trust my work, you shouldn't either. so we prove to our selfs that it worked now.

# In[ ]:


total = allstock.isnull().sum().sort_values(ascending=False)
percent = ((allstock.isnull().sum()/allstock.isnull().count()).sort_values(ascending=False)*100)
missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data


# with the data all cleaned up we are ready to make some plots. I like to set up the figure size first. it gives nice consistancy for the plots you make when you start adjusting how much data you're using, font size, etc. 
# 
# theres some unnessisary code here, but its again part of checking my work. first it pulls up the current figure size and prints it. then it changes the size and saves it, then it prints the size again.
# 
# .you can change the X and Y size values to anything you want! I tried to make it easy to find and modify them in my notations next to the code.
# 
# remember how I talked about sometimes needing to restart all of the kernal to see changes? run the below cell twice and you will see that the second time you run it that the old size is the already modified new size. Thats because it remembers!

# In[ ]:


fig_size = plt.rcParams["figure.figsize"] #loads current figure size
print('old size:',fig_size) #prints the size
fig_size[0] = 15 #sets the X size to 15
fig_size[1] = 8 #sets the Y size to 8
plt.rcParams["figure.figsize"] = fig_size #sets this numbers to the new size
fig_size = plt.rcParams["figure.figsize"] #loads the figure size for checking
print ('new size:',fig_size) #prints the figure size


# now the moment you have been waiting for, lets actually plot some stuff!
# 
# the very first line sets up what stock we want to look at. its also the line to edit if you want to see other stocks. I picked Amazon (AMZN).
# 
# next we set the variable called category. this is which column we want to plot. in this case i'm looking at the high for the day. you can edit this and change it to Low, Open, Close, or Volume. see the allstock.columns cell above for everything you can put here.
# 
# allstocksingle is a matrix that is made only of data points from the desired stock. the line looks at all the rows where the column of 'Name' matches your desired stock and puts them in the allstocksingle matrix. it may be worth noting that if you delete these first 3 lines of code it will plot ALL the stocks. which, if you code on a little raspberry pi3 like I do, it takes a while to plot.
# 
# for x we use the datetime module to convert the date in the file to something python can work with. I'm really not sure whats it's doing to be honest. but it works. maybe its just telling python how to read it?
# 
# y is straight forward, it looks at the value of our choice. 
# 
# we then tell it what the ticks are displayed as for the x axis. it tells it to display month/day/year. with out this code it only displays the year, which is not so helpful.
# 
# The yaxis locator line below it sets the number between yaxis ticks. you can change that to what ever you'd like.
# 
# one thing I had serious problems with is there is a limit to how many ticks python will display, I think its 2000. we have over 2000 days (and therefor ticks) so if it tried to display them all it would fail. which is maybe good as you wouldn't be able to read that anyway. thats what the next line (ends with 'interval = 60') is for. it tells it to only write the date tick every 60 days. you can change the 60 to what ever you want. 
# 
# the next few lines actually make the plot and add all the ticks and labels. well plt.plot(x,y) is what makes the plot, the rest add the details to make it readable. 
# 
# at the end the commented out part, starts with '#', saves the plot to disk. It's usefull if you are running the code on a machine instead of on kaggle. You can then use that picture and upload it to facebook or linked in and impress all your friends and colleagues with the code you "wrote". Hey, I don't judge. 
# 
# interestingly you can remove the # and run it on kaggle and it does work. I just don't know what it does. does it save the picture somewhere? where does it save it if so? or does it just ignore it and move on? i have no idea. if you know, please tell me!

# In[ ]:


stock = 'AMZN' #edit the stock name to view other stocks. see the cell above with a list of stocknames
catagory = 'High' #edit this to change which value is plotted (see allstock.columns cell for options)
allstocksingle = allstock[allstock['Name'] == stock] #makes matrix with only the stock info

x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in allstocksingle['Date']] #convert date to something python understands
y = allstocksingle[catagory] #plots which ever catagory you entered above

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y')) #display the date properly
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60)) #x axis tick every 60 days
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(100)) # sets y axis tick spacing to 100

plt.plot(x,y) #plots the x and y
plt.grid(True) #turns on axis grid
plt.ylim(0) #sets the y axis min to zero
plt.xticks(rotation=90,fontsize = 10) #rotates the x axis ticks 90 degress and font size 10
plt.title(stock) #prints the title on the top
plt.ylabel('Stock Price For '+ catagory) #labels y axis
plt.xlabel('Date') #labels x axis

#plt.savefig(stock+catagory+'.png')


# Well thats pretty great. but what if I don't want to view all 5 years? instead I just want a specific time frame? thats a good question! lets do it.
# 
# good news is it's very simple. we modify the 'plt.ylim' we used before. first change y to x. then add another value. the first value is the min and the second value is the max. I set those up as the first two lines of code as variables so that its easy to change and understand.
# 
# other than that its reusing a bunch of code we used above. speaking of reusing, the plt.plot function defaults the x and y min and max based of the entire dataset. so if you only look at a early year, such as '13-'14, you would find theres a lot of unused chart space. it looks bad and makes it hard to read. I edited the ylim command and added a lower maximum.
# 
# finally, changed the title of the chart to say that its a plot of a specific time.

# In[ ]:


startdate = ('2013-01-01') #enter the start date here, it must be YYYY-MM-DD
enddate = ('2014-01-01') #enter the end date here, it must be YYYY-MM-DD

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y')) #display the date
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20)) #x axis tick every 20 days
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(50)) # sets y axis tick spacing to 50

plt.plot(x,y) #plots the x and y
plt.grid(True)
plt.xlim(startdate,enddate) #this is the new line of code that sets the start and end limits on the x axis
plt.ylim(0, 500) #sets the y axis min to zero and y max to 500
plt.xticks(rotation=90,fontsize = 10) #rotates the x axis ticks 90 degress and font size 10
plt.title(stock+ '  '+ startdate + ' to ' + enddate) #prints the title on the top

plt.ylabel('Stock Price For '+ catagory) #labels y axis
plt.xlabel('Date') #labels x axis


# There you go! that should be all you need to know to pull data from a .csv file, reading intresting bits out of it, and plot what you want to see. I hope this helps people out as much as it helped me to write it. There where a few commands I thought I understood until I wrote this out, did a bit more research about what the line did, and realized I was wrong. have any questions or comments? let me know! especially if I said something that was wrong, I'm still learning.

# In[ ]:




