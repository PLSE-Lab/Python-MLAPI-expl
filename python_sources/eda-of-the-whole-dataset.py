#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importation
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gc


# In[2]:


# Functions
def quantiles (x):
    for i in range(0, 11):
        y = i / 10
        print("{0:.0f}".format(y*100), "quantile :", "{0:.4f}".format(x.quantile(y)))


# **One day targeting :**
# 
# As your know, the file is really heavy and our data are spread between 3 days. I really want to analyse a whole day, but I'm not sure about analysing **all** of those days. So, in this second part, we will make a focus on a single day at take a deeper look at our different variables. 
# Firstly, we need to convert the click into a time format, then to use it as an index.

# ** I updated importation for a faster version :**

# In[3]:


# Rows importation
df = pd.read_csv('../input/train.csv', skiprows = 9308568, nrows = 59633310)

# Header importation
header = pd.read_csv('../input/train.csv', nrows = 0) 
df.columns = header.columns
df

# Cleaning
del header
gc.collect()

# And check his size        
print("The created dataframe contains", df.shape[0], "rows.")    


# Our dataframe is pretty big. On previous versions of this kernel, I had some problems with the RAM management. That's why, I add some optimization of variables types here. I think we could free a lot of memory with more appropriated datatypes. Thanks for this kernel : https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65/notebook

# In[4]:


total_before_opti = sum(df.memory_usage())

# Type's conversions
def conversion (var):
    if df[var].dtype != object:
        maxi = df[var].max()
        if maxi < 255:
            df[var] = df[var].astype(np.uint8)
            print(var,"converted to uint8")
        elif maxi < 65535:
            df[var] = df[var].astype(np.uint16)
            print(var,"converted to uint16")
        elif maxi < 4294967295:
            df[var] = df[var].astype(np.uint32)
            print(var,"converted to uint32")
        else:
            df[var] = df[var].astype(np.uint64)
            print(var,"converted to uint64")
    
for v in ['ip', 'app', 'device','os', 'channel', 'is_attributed'] :
    conversion(v)

# Results :    
print("Memory usage before optimization :", str(round(total_before_opti/1000000000,2))+'GB')
print("Memory usage after optimization :", str(round(sum(df.memory_usage())/1000000000,2))+'GB')
print("We reduced the dataframe size by",str(100-round(sum(df.memory_usage())/total_before_opti *100,2))+'%')


# **How many different values does our categorial variables take ?**

# In[ ]:


print("Number of different values :")
print("IP :",len(df['ip'].unique()))
print("App :", len(df['app'].unique()))
print("Device :", len(df['device'].unique()))
print("OS :", len(df['os'].unique()))
print("Channel :",len(df['channel'].unique()))


# **What proportion of click generate downloads ?**

# In[ ]:


print("Proportion of click which generate downloads :")
print(df['is_attributed'].value_counts())

plt.figure(figsize=(8,8))
plt.title('Proportion of click which generate downloads \n', fontsize =15)
ax =(df['is_attributed'].value_counts(normalize=True)*100).plot(kind='bar')
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))


# As we can see, only a very little proportion of clicks generate downloads.

# **Zoom on this IP :**

# In[ ]:


# We create a dataframe with all IP, and the number of click from this IP
IP = df['ip'].value_counts()

# We can now take a first look at those IP
plt.figure(figsize = [10,5])
sns.boxplot(IP)
plt.title('Number of click by IP', fontsize =15)


# Most of the IP have generate only few clicks, but few IP generated **a ton** of clicks. Let's check this in details :

# In[ ]:


IP.describe()


# I really want more informations about those suspicious IP. We're now going to zoom on the most suspicious IP (I arbitrarily set the threshold at 300 clicks).

# In[ ]:


suspicious_IP = IP[IP > 300]
print("Number of rows selected :",suspicious_IP.shape[0])


# In[ ]:


plt.figure(figsize=(10,10))
sns.distplot(suspicious_IP, hist = False)
plt.title('Number of clicks by IP (<300 only)', fontsize = 20)
plt.xlabel('Number of clicks', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)

# Cleaning
del suspicious_IP
gc.collect()


# Ok, so even with a threshold of 300 clicks, we can identify some "big clickers" and few "very big clickers". I want to go a little bit deeper and reproduce the same operation with a threshold set at 30000.

# In[ ]:


very_suspicious_IP = IP[IP > 30000]
print("Number of rows selected :",very_suspicious_IP.shape[0])


# In[ ]:


plt.figure(figsize=(10,10))
sns.distplot(very_suspicious_IP, hist = False)
plt.title('Number of clicks by IP (<30 000 only)', fontsize = 20)
plt.xlabel('Number of clicks', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)

# Cleaning
del very_suspicious_IP
gc.collect()


# Great ! We've now a list of 36 789 suspicious IP, and another of 66 **very** suspicious ! 
# Now it's time to leave the IP level and to go back to the click level to see if those suspicious IP really are.

# In[ ]:


# We prepare our IP list to the merge
IP_ready_to_merge = pd.DataFrame(IP).reset_index()
IP_ready_to_merge.columns=['ip', 'freq_ip']

# Cleaning
del IP
gc.collect()

# Creation of clicker categories (we will use it later)
IP_ready_to_merge['clicker_type'] = ''
IP_ready_to_merge.loc[IP_ready_to_merge['freq_ip'] <= 10, 'clicker_type'] = "very_little_clicker"
IP_ready_to_merge.loc[(IP_ready_to_merge['freq_ip'] >= 10) & (IP_ready_to_merge['freq_ip'] < 300), 'clicker_type'] = "little_clicker"
IP_ready_to_merge.loc[(IP_ready_to_merge['freq_ip'] >= 300) & (IP_ready_to_merge['freq_ip'] < 30000), 'clicker_type'] = "big_clicker"
IP_ready_to_merge.loc[IP_ready_to_merge['freq_ip'] >= 30000, 'clicker_type'] = "huge_clicker"

# Now we can add our IP frequency to our main dataframe
df = pd.merge(df, IP_ready_to_merge, on ='ip')


# **Relation between number of click by IP and downloading the app :**
# 
# Great ! We've our number of clicks variable, then we can do our test.

# In[ ]:


# Frequencies
print('Minimum number of clicks needed to download an app :', df.freq_ip[df['is_attributed']==1].min())
print("How many IP do we have in each category ?\n", IP_ready_to_merge['clicker_type'].value_counts())
print("How many clicks, clickers of each caterogy have generate ?\n",df['clicker_type'].value_counts())


# In[ ]:


plt.figure(figsize = (7,7))
df['clicker_type'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.title("Proportion of clicks generated by each categories of clickers", fontsize =15)
plt.ylabel("")


# Thereforce, big clickers represents 36 764 IPs (over 145 693) and generate 85% of the clicks and huge clickers represents only 66 IPs but generated 8% of the clicks. Consequently, arround 93% of the clicks are generate by a sub-population of suspicious IPs.
# This statistical remember what Talking Data said in the overview "3 billion clicks per day, of which 90% are potentially fraudulent".
# 
# Finaly, it looks like watching at the number of clicks by IPs is a great way of identify bots.

# **What proportion of IP download the app ?**

# In[33]:


DL_by_IP = df.groupby('ip').is_attributed.sum()

plt.figure(figsize=(8,8))
plt.title('Proportion of IP which generate downloads at least once \n', fontsize =15)
ax =((DL_by_IP > 0).value_counts(normalize=True)*100).plot(kind='bar')

for p in ax.patches:
    ax.annotate('{:.2f}%'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))


# **Does bots download the app ?**

# In[32]:


print(DL_by_IP.describe(), '\n Quantile 99% :',DL_by_IP.quantile(0.99),       '\n Quantile 99,9% :',DL_by_IP.quantile(0.999),       '\n Quantile 99,999% :',DL_by_IP.quantile(0.9999))


# In[31]:


data_to_plot = DL_by_IP.nlargest(10).reset_index()
data_to_plot.columns=('IP', 'Downloads')
data_to_plot.sort_values('Downloads', ascending = False)
plt.figure(figsize = (8,5))
sns.barplot(x = data_to_plot['Downloads'], y = data_to_plot['IP'], orient = 'h')
plt.title('Top 10 bigest downloader', fontsize = 15)

# Cleaning
del data_to_plot
gc.collect()


# **Does standard users download the app ?**

# In[ ]:


# We need a DataFrame
data_to_plot2 = pd.DataFrame(DL_by_IP).reset_index()

# We create some categories to plot
data_to_plot2["cat_DL"] = ''
data_to_plot2.loc[data_to_plot2['is_attributed'] == 0, "cat_DL"] = "No"
data_to_plot2.loc[data_to_plot2['is_attributed'] == 1, "cat_DL"] = "Yes, once"
data_to_plot2.loc[data_to_plot2['is_attributed'] > 1, "cat_DL"] = "Yes, multiple times"

# We can plot it
plt.figure(figsize=(8,8))
data_to_plot2["cat_DL"].value_counts().plot(kind = 'pie',autopct='%1.0f%%')
plt.title('Does users download the app ?', fontsize=15)
plt.ytitle=''


# In[ ]:


data_to_plot3 = data_to_plot2[data_to_plot2['is_attributed'] > 1]
data_to_plot4 = data_to_plot3[data_to_plot3['is_attributed'] <= 15]

fig, ax = plt.subplots(1,2, figsize =(15,4))
ax[0].title.set_text("Number of downloads by IP (>1)")
sns.violinplot(x = data_to_plot3['is_attributed'], ax = ax[0] )
ax[1].title.set_text("Number of downloads by IP (1 to 15)")
sns.violinplot(x = data_to_plot4['is_attributed'], ax = ax[1])

# Cleaning
del data_to_plot3, data_to_plot4
gc.collect()


# **How many times, each categories of clickers download the app ?**

# In[ ]:


ip_level = pd.merge(IP_ready_to_merge, data_to_plot2, on='ip')
cross_tab = pd.crosstab(ip_level['cat_DL'], ip_level['clicker_type'], normalize='columns')

# We need to sort the index (because 'Yes, one time' was beofre 'Yes, multiple times')
cross_tab.index = pd.CategoricalIndex(cross_tab.index, categories = ['Yes, multiple times', 'Yes, once', 'No'])
cross_tab = cross_tab.sort_index(ascending = False)

# Same thing for columns
cross_tab = cross_tab[['huge_clicker', 'big_clicker', 'little_clicker', 'very_little_clicker']]

# Ok we can make our graph now
plt.figure(figsize = (8,5))
plt.title('How many times, each categories of clickers download the app ? \n', fontsize=15) 
sns.heatmap(cross_tab,annot=True, fmt='.0%')
plt.xlabel('Categories of clickers ', fontsize = 10)
plt.ylabel('Categories of downloaders', fontsize = 10)

# Some cleaning
del IP_ready_to_merge, data_to_plot2
gc.collect()


# There is some very interesting information in the graph above : little, and very little clickers (humans) don't download the app, or download it only one time. More they click, more their probability of downloading the app seems high. 
# 
# On  another side, big clickers have 38% of chance to download the app multiple times and huge clickers 100%. I guess that my categories are not perfect, the lower bound of 'big clicker' should probably be higher, because, IMO, some standard users (humans) are in this category, and so, have a standard behaviour.
# 
# At this point, I think we should keep in mind that the aim of the competition is to predict if the current click will download or not the app, not the user (IP). Identify suspicious IP is great, but we really need to go deeper.

# **Download by click ratio :**
# 
# I think this indicator could help us to understand what kind of clickers download the app the most, in poportion of there amount of clicks. 

# In[ ]:


# Ratio computation
ip_level['DL_by_click_ratio'] = ip_level['is_attributed']/ip_level['freq_ip']

# Ratio global analysis
print(ip_level['DL_by_click_ratio'].describe())

plt.figure(figsize = (6,4))
sns.violinplot(ip_level['DL_by_click_ratio'])
plt.title('Ratio : Download by click', fontsize=15)
plt.xlabel('Download by click')


# Unsurprised, most of IPs have a low ratio (50% lower than 0.0025 and 75% lower than 0.2). I guess it would be more interesting to check by category of clicker.

# In[ ]:


plt.figure(figsize = (6,4))
sns.boxplot(ip_level['DL_by_click_ratio'], ip_level['clicker_type'])
plt.title('Ratio : Download by click', fontsize=15)
plt.xlabel('Download by click')
plt.ylabel('Category of clicker')

# Cleaning
del ip_level
gc.collect()


# The relation looks obvious : biger the number of clicks is, lower the ratio is. In other words, bots spam clicks, download the app multiples times, but way less clicks from them lead to downloads.
# 
# I think we've enough informations about the relation click / download right now. It's time to look at the 'attributed_time' column which is the time of the downloading click.

# **Attributed time analysis :**

# In[ ]:


not_missing = df[df['attributed_time'].isna() == False]
not_missing['gap'] = pd.to_datetime(not_missing['attributed_time']).sub(pd.to_datetime(not_missing['click_time']))

for i in range(0, 11):
        y = i / 10
        print("{0:.0f}".format(y*100), "quantile :", not_missing['gap'].quantile(y))


# In[ ]:


# By clicker type
not_missing.groupby('clicker_type').gap.describe()

# Cleaning
del not_missing, df['attributed_time'], df['clicker_type'], df['freq_ip']
gc.collect()


# I don't really see anything interesting here.

# **Back to the click time**
# 
# We've currently analyze the amount of clicks by IP adress but not the click time. Maybe we could find some interesting information here.

# In[6]:


df.set_index(pd.to_datetime(df['click_time']), inplace = True)
by_hour = df.resample('H').ip.count()

plt.figure(figsize = (10,5))
by_hour.plot()
plt.title('Number of clicks over the day', fontsize = 15)
plt.xlabel('Time')
plt.ylabel('Number of clicks')

# Cleaning
del by_hour
gc.collect()


# Strangly, the number of clicks decrease during between 3PM and 11PM. We should analyze another day to check this tendancy.

# **Download rate by hour :**

# In[9]:


plt.figure(figsize = (10,5))
df.resample('H').is_attributed.mean().plot()
plt.title('Download rate evolution over the day', fontsize = 15)
plt.xlabel('Time')
plt.ylabel('Download rate')


# **It's time to analyze the device : number of clicks by device**
# 
# We already know we have  2265 different devices.

# In[ ]:


clicks_by_device = df.groupby('device').is_attributed.count()
quantiles(clicks_by_device)


# In[ ]:


print("Number of devices with a number of clicks greater than 100 :", len(clicks_by_device[clicks_by_device >= 100 ]))
print("Number of devices with a number of clicks greater than 200 :", len(clicks_by_device[clicks_by_device >= 200 ]))
print("Number of devices with a number of clicks greater than 300 :", len(clicks_by_device[clicks_by_device >= 300 ]))
print("Number of devices with a number of clicks greater than 1000 :", len(clicks_by_device[clicks_by_device >= 1000 ]))


# **Same thing for apps : **

# In[ ]:


clicks_by_app = df.groupby('app').is_attributed.count()
quantiles(clicks_by_app)


# **And OS :**

# In[ ]:


clicks_by_os = df.groupby('os').is_attributed.count()
quantiles(clicks_by_os)


# **Last click from the IP analysis :**

# In[ ]:


df['id'] = range(1, len(df) + 1)
last_clicks = df.groupby('ip').id.last().reset_index()
last_clicks['last_click'] = 1
last_clicks.drop('ip', axis=1, inplace = True)

df = pd.merge(df, last_clicks, on ='id', how = 'left').set_index(pd.to_datetime(df['click_time']))
df['last_click'].fillna(0, inplace = True)
conversion('last_click')

# Cleaning
del df['id'], df['click_time'], last_clicks
gc.collect()


# **Do the last click generate more download ?**

# In[ ]:


print("Download rate for last clicks :", str(round(df[df['last_click'] == 1].is_attributed.mean()*100,2))+'%')
print("Download rate for all clicks :", str(round(df.is_attributed.mean()*100,2))+'%')


# Great ! I would include this variable in my features engineering for sure.

# **Number of clicks from the IP during the last minute**

# In[ ]:


# We firstly need to sort our data in the right order
df['click_time'] = df.index
df.sort_values(['click_time', 'ip'], inplace = True)

# We can now compute the number of clicks during the last minute
clicks_minute = pd.DataFrame(df.groupby('ip')['app'].rolling('min').count())

# We can't use a pd.merge because it takes to much memory
clicks_minute.reset_index(inplace = True)
clicks_minute.sort_values(['click_time', 'ip'], inplace = True)

# Like our two dataset are in the same order, we can make a simple concatenation
del clicks_minute['click_time'], clicks_minute['ip'], df['click_time']
gc.collect()
df['clicks_minute'] = clicks_minute.values

# Conversion
conversion('clicks_minute')

# Cleaning
del clicks_minute
gc.collect()


# In[ ]:


df['temp_cats'] = ''
df.loc[df['clicks_minute'] <= 10, 'temp_cats'] = 'less than 10'
df.loc[(df['clicks_minute'] > 10) & (df['clicks_minute'] < 100), 'temp_cats'] = 'between 10 and 100'
df.loc[df['clicks_minute'] >= 100, 'temp_cats'] = 'more than 100'

y = df.groupby('temp_cats').is_attributed.mean()
y = y[['less than 10', 'between 10 and 100', 'more than 100']]

plt.figure(figsize =(6,4))
plt.title('Download rate by number of clicks during the last minute\n', fontsize = 15)
y.plot(kind = 'bar', width = 0.8)
plt.xlabel('')

