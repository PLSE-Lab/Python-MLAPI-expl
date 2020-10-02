#!/usr/bin/env python
# coding: utf-8

# ### Table of Contents
# 
# <ul>
#     <li> Introduction </li>
#     <li> Import all libraries </li>
#     <li> Load Dataset </li>
#     <li> Data Pre-Processing Steps </li>
#          <ul>
#              <li>Check Missing values</li>
#              <li>Remove Duplicates rows</li>
#              <li>Convert app size from Bytes to MB</li>
#         </ul>
#     <li>Find Different Insights </li>
#         <ul>
#             <li> Find different kind of Genres </li>
#             <li> Find Total Free and Paid apps </li>
#             <li> Find Free and Paid apps in all Genres </li>
#             <li> Find Most Popluar apps in all Genres ratingwise </li>
#             <li> Find Total Free and Paid apps in the list of popular apps (rating 4 to 5) </li>
#             <li> Find Most Popluar Genres in Free apps and Paid apps </li>
#             <li> Find Most High rating apps in all Genres</li>
#             <li> Find Most High rating Free apps in all Genres</li>
#             <li> Find Most High rating Paid apps in all Genres</li>
#             <li> Find Most Supported devices apps </li>
#             <li> Find Total apps in Most Supported devices </li>
#             <li> Find Total apps in Most Supported devices genrewise </li>
#             <li> Find Total Free apps in Most Supported devices </li>
#             <li> Find Total Paid apps in Most Supported devices </li>
#             <li> Total number of Games that are available in a multiple languages</li>
#             <li> Maximum languges support application</li>
#         </ul>
#     <li>Pearson Correlation </li>
#     <li> Feature engineering </li>
# </ul>
# 

# ### <div id="intro">Introduction </div>

#  #### Dataset Name : Mobile App Statistics (Apple iOS app store)
# #### Contents :
# 
#  "id" : App ID <br>
#  
#  "track_name": App Name <br>
#  
#  "size_bytes": Size (in Bytes) <br>
#  
#  "currency": Currency Type <br>
#  
#  "price": Price amount <br>
#  
#  "rating_count_tot": User Rating counts (for all version) <br>
#  
#  "rating_count_ver": User Rating counts (for current version) <br>
#  
#  "user_rating" : Average User Rating value (for all version) <br>
#  
#  "user_rating_ver": Average User Rating value (for current version) <br>
#  
#  "ver" : Latest version code <br>
#  
#  "cont_rating": Content Rating <br>
#  
#  "prime_genre": Primary Genre <br>
#  
#  "sup_devices.num": Number of supporting devices <br>
#  
#  "ipadSc_urls.num": Number of screenshots showed for display <br>
#  
#  "lang.num": Number of supported languages <br>
#  
#  "vpp_lic": Vpp Device Based Licensing Enabled <br>
# 

# ### <div id="libs">Import all libraries </div>

# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify # pip install squarify (algorithm for treemap)
import os
import math 

get_ipython().run_line_magic('matplotlib', 'inline')


# ### <div id="dataset">Load Dataset</div>

# In[ ]:


# Load app store file
apps = pd.read_csv("../input/app-store-apple-data-set-10k-apps/AppleStore.csv", index_col=[0])
apps.head()


# In[ ]:


#Overview of dataset
apps.info()


# ### <div id="dataprep">Data Pre-Processing Steps</div>

# ####  <div id="missing"> --- Check Missing values --- </div>
# 
# This is almost the first step to check whether dataset has any null or empty value or not.

# In[ ]:


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data(apps)


# As we can see that there is no missing values in entire dataset

# 1. #### <div id="bytes"> --- Convert app size from Bytes to MB --- </div>
# 
# This step is  necessary as it become easy for further analysis

# In[ ]:


apps['size_bytes'] = apps['size_bytes'] / 1000000
apps.rename(columns={'size_bytes':'size_mb'}, inplace=True)
apps.head(5)


# #### <div id="dup">--- Remove Duplicates rows --- </div>

# In[ ]:


apps.duplicated(subset=None, keep=False)
apps.head(5)


# ## <div id="insight"> Find Different Insights </div>
# 

# #### <div id="un_genres"> --- Find different kind of genres --- </div>
# 

# In[ ]:


genres = apps['prime_genre'].unique()
print("Total genres : {}".format(len(genres)))
print(genres)


# #### <div id="total_free_paid"> --- Find Total free and paid apps --- </div>
# 

# Let's count total free apps and total paid apps in this dataset

# In[ ]:


freeapps = apps[apps.price == 0.0]
paidapps = apps[apps.price != 0.0]

print("Free apps : ",len(freeapps))
print("Paid apps : ",len(paidapps))


# Now, visualize on pie chart 

# In[ ]:


app_pricedf= pd.DataFrame( [len(freeapps),len(paidapps)] , index=['free','paid'])
app_pricedf.plot(kind='pie', subplots=True, figsize=(16,8), autopct='%1.1f%%')


#  #### <div id="total_free_paid_genres"> --- Find Free and paid apps in all genres --- </div>
# 

# In[ ]:


# Return the numbers of free app in each genres
def genreFree(gen):
    return len(apps[(apps['price'] == 0.0) & (apps['prime_genre']== gen)])


# In[ ]:


# Return the numbers of paid app in each genres
def genrePaid(gen):
    return len(apps[(apps['price'] != 0.0) & (apps['prime_genre']== gen)])


# In[ ]:


# Make list of each genre , its free app, paid app and total app . then merge it into one dataframe
genre_list = list()
genreFree_list = list()
genrePaid_list = list()
genreTotal_list = list()


# In[ ]:


# append all details in respective list
for gen in genres:  
    free_gen = genreFree(gen)
    paid_gen = genrePaid(gen)
    totalapp_gen = free_gen + paid_gen
    genre_list.append(gen)
    genreFree_list.append(free_gen)
    genrePaid_list.append(paid_gen)
    genreTotal_list.append(totalapp_gen)


# In[ ]:


# Let's make a dataframe of it
genre_df = pd.DataFrame({
    "genre_name" : genre_list,
    "genre_freeApp" : genreFree_list,
    "genre_paidApp" : genrePaid_list,
    "genre_totalApp" : genreTotal_list
},columns=['genre_name','genre_freeApp','genre_paidApp','genre_totalApp'])

#sorting into descending order
genre_df.sort_values('genre_totalApp', ascending=False, inplace=True)

genre_df.head(10)


# In[ ]:


# remove duplicate genre 
genre_df.drop_duplicates('genre_name',keep= False,inplace=True)
genre_df.head()


# In[ ]:


def groupedGraph(start,end):
    # set width of bar
    barWidth = 0.20

    # set height of bar
    bars1 = genre_df['genre_freeApp'][start:end]
    bars2 = genre_df['genre_paidApp'][start:end]
    bars3 = genre_df['genre_totalApp'][start:end]

    # Set position of bar on X axis
    r1 = np.arange(bars1.size)
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bars1, color='#36688D', width=barWidth, edgecolor='white', label='Free apps')
    plt.bar(r2, bars2, color='#F3CD05', width=barWidth, edgecolor='white', label='Paid apps')
    plt.bar(r3, bars3, color='#F49F05', width=barWidth, edgecolor='white', label='Total apps')

    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth for r in range(len(bars1))], genre_df['genre_name'][start:end])

    # Create legend & Show graphic
    plt.legend()


# In[ ]:


#Let's visualize this dataframe into the Grouped barplot

fig = plt.figure(figsize=(25,15))

plt.subplot(311)
groupedGraph(0,1)

plt.subplot(312)
groupedGraph(1,12)

plt.subplot(313)
groupedGraph(12,23)


# #### <div id="popular_app_genres"> --- Find Most Popluar apps in all Genres --- </div>
# 

# Here, I have sorted user_rating and rating_count_tot fields in a descending order to get highest rating apps for all versions in order to find most popular apps

# In[ ]:


popular_apps = apps.sort_values(['user_rating','rating_count_tot'], ascending=False)
popular_apps.head()


# In[ ]:


#Let's visualize top 10 higher rating applications in bar plot

fig = plt.figure(figsize=(20,8))

ax = sns.barplot(popular_apps['track_name'][0:20], (popular_apps['rating_count_tot']/popular_apps['user_rating'])[0:20])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")


# #### <div id="total_free_paid_popularapp"> --- Find Total Free and Paid apps in the list of popular apps (rating 4 to 5) --- </div>

# Let's segregate apps from rating 4 to 5 only

# In[ ]:


# All higher rating applications 
ratingapp = popular_apps[(popular_apps['user_rating'] == 4.0) | (popular_apps['user_rating'] == 5.0) | (popular_apps['user_rating']==4.5)]
ratingapp.head(5)


# Divide ratingapp dataframe into free_ratingapp and paid_ratingapp

# In[ ]:


#Only free higher rating applications
free_ratingapp = ratingapp[ratingapp['price'] == 0.0]

#Only paid higher rating applications
paid_ratingapp = ratingapp[ratingapp['price'] != 0.0]

print("All higher rating applications :", len(ratingapp))
print("Free higher rating applications : ",len(free_ratingapp))
print("Paid higher rating applications : ",len(paid_ratingapp))


# In[ ]:


#let's visualize popular free and paid rating apps (4 to 5 rating)
fig = plt.figure(figsize=(20,8))

plt.subplot(411)
plt.title("Top 10 highest free rating apps (4 to 5 rating)")
ax = sns.barplot(free_ratingapp['track_name'][0:9] + '---' + free_ratingapp['prime_genre'][0:9],(free_ratingapp['rating_count_tot']/free_ratingapp['user_rating'])[0:9], color="red")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

plt.subplot(414)
plt.title("Top 10 highest paid rating apps (4 to 5 rating)")
ax = sns.barplot(paid_ratingapp['track_name'][0:9] + '---' + paid_ratingapp['prime_genre'][0:9],(paid_ratingapp['rating_count_tot']/paid_ratingapp['user_rating'])[0:9], color="blue")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")


# #### <div id="popular_genres_both"> --- Find Most Popluar Genres in Free apps and Paid apps --- </div>
# 

# In[ ]:


# create the empty lists for storing total highratingapps in their respective genres
free_highratingapp = list()
paid_highratingapp = list()
total_highratingapp = list()


# In[ ]:


# appending free and paid popular apps for each genres
for g in ratingapp['prime_genre'].unique():
    free_highratingapp.append(len(free_ratingapp[free_ratingapp['prime_genre']== g]))
    paid_highratingapp.append(len(paid_ratingapp[paid_ratingapp['prime_genre']== g]))
    total_highratingapp.append(len(free_ratingapp[free_ratingapp['prime_genre']== g]) + len(paid_ratingapp[paid_ratingapp['prime_genre']== g]))
    


# In[ ]:


# Make dataframe of total free and paid apps genreswise
rating_df = pd.DataFrame({
    'genre' : genre_list,
    'free_higherRating' : free_highratingapp,
    'paid_higherRating' : paid_highratingapp,
    'total_higherRating' : total_highratingapp
},columns=['genre','free_higherRating','paid_higherRating','total_higherRating'])

rating_df.sort_values('total_higherRating',ascending =False, inplace=True)
rating_df.head()


# In[ ]:


#remove duplicates if any
rating_df.drop_duplicates('genre',keep=False,inplace=True)
rating_df.head()


# Here, visualize it in circle graph

# In[ ]:


fig = plt.figure(figsize=(15,20))

plt.subplot(321)
# Create a circle for the center of the plot
circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(rating_df['free_higherRating'][0:10], labels= rating_df['genre'][0:10])
p=plt.gcf()
p.gca().add_artist(circle)
plt.title("Top 10 Free popular applications genres")

plt.subplot(322)
# Create a circle for the center of the plot
circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(rating_df['paid_higherRating'][0:10], labels= rating_df['genre'][0:10])
p=plt.gcf()
p.gca().add_artist(circle)
plt.title("Top 10 Paid popular applications genres")


# #### <div id="popular_app_rating"> --- Find Most High rating apps in all Genres --- </div>
# 

# In[ ]:


def dountChart(gen,title):  
    # Create a circle for the center of the plot
    circle=plt.Circle( (0,0), 0.7, color='white')
    
    # just keep on user rating as name not overlapping while pie chart plotting
    plt.pie(ratingapp['user_rating'][ratingapp['prime_genre']==gen][0:10], labels= ratingapp['track_name'][ratingapp['prime_genre']==gen][0:10])
    p=plt.gcf() #gcf = get current figure
    p.gca().add_artist(circle)
    plt.title(title , fontname="arial black")


# In[ ]:


gens = ['Games','Shopping','Social Networking','Music','Food & Drink', 'Photo & Video','Sports','Finance']

fig = plt.figure(figsize=(25,30))

plt.subplot(421)
dountChart(gens[0],'Top Higher rating '+gens[0]+' apps')  

plt.subplot(422)
dountChart(gens[1],'Top Higher rating '+gens[1]+' apps')

plt.subplot(423)
dountChart(gens[2],'Top Higher rating '+gens[2]+' apps')

plt.subplot(424)
dountChart(gens[3],'Top Higher rating '+gens[3]+' apps')

plt.subplot(425)
dountChart(gens[4],'Top Higher rating '+gens[4]+' apps')

plt.subplot(426)
dountChart(gens[5],'Top Higher rating '+gens[5]+' apps')

plt.subplot(427)
dountChart(gens[6],'Top Higher rating '+gens[6]+' apps')

plt.subplot(428)
dountChart(gens[7],'Top Higher rating '+gens[7]+' apps')


# #### <div id="popular_free_app_rating"> --- Find Most High rating Free apps in all Genres --- </div>
# 

# In[ ]:


def squatifyChart(gen,title):
    squarify.plot(free_ratingapp['user_rating'][free_ratingapp['prime_genre']==gen][0:5], 
              label=free_ratingapp['track_name'][free_ratingapp['prime_genre']==gen][0:5],
              alpha=.5,
             norm_x=50)
    plt.title(title)
    plt.axis('off')


# In[ ]:


gens = ['Games','Shopping','Social Networking','Music','Food & Drink', 'Photo & Video','Sports','Finance']

fig = plt.figure(figsize=(25,30))

plt.subplot(421)
squatifyChart(gens[0],'Top Free Higher rating '+gens[0]+' apps')  

plt.subplot(422)
squatifyChart(gens[1],'Top Free Higher rating '+gens[1]+' apps')

plt.subplot(423)
squatifyChart(gens[2],'Top Free Higher rating '+gens[2]+' apps')

plt.subplot(424)
squatifyChart(gens[3],'Top Free Higher rating '+gens[3]+' apps')

plt.subplot(425)
squatifyChart(gens[4],'Top Free Higher rating '+gens[4]+' apps')

plt.subplot(426)
squatifyChart(gens[5],'Top Free Higher rating '+gens[5]+' apps')

plt.subplot(427)
squatifyChart(gens[6],'Top Free Higher rating '+gens[6]+' apps')

plt.subplot(428)
squatifyChart(gens[7],'Top Free Higher rating '+gens[7]+' apps')


# #### <div id="popular_paid_app_rating"> --- Find Most High rating Paid apps in all Genres --- </div>
# 
# 

# In[ ]:


def squatifyChart_paid(gen,title):
    squarify.plot(paid_ratingapp['user_rating'][paid_ratingapp['prime_genre']==gen][0:5], 
              label=paid_ratingapp['track_name'][paid_ratingapp['prime_genre']==gen][0:5],
              alpha=.5,
              color=["pink","green","blue", "grey"],
             norm_x=50)
    plt.title(title)
    plt.axis('off')


# In[ ]:


gens = ['Games','Shopping','Social Networking','Music','Food & Drink', 'Photo & Video','Sports','Finance']

fig = plt.figure(figsize=(25,30))

plt.subplot(421)
squatifyChart_paid(gens[0],'Top Paid Higher rating '+gens[0]+' apps')  

plt.subplot(422)
squatifyChart_paid(gens[1],'Top Paid Higher rating '+gens[1]+' apps')

plt.subplot(423)
squatifyChart_paid(gens[2],'Top Paid Higher rating '+gens[2]+' apps')

plt.subplot(424)
squatifyChart_paid(gens[3],'Top Paid Higher rating '+gens[3]+' apps')

plt.subplot(425)
squatifyChart_paid(gens[4],'Top Paid Higher rating '+gens[4]+' apps')

plt.subplot(426)
squatifyChart_paid(gens[5],'Top Paid Higher rating '+gens[5]+' apps')

plt.subplot(427)
squatifyChart_paid(gens[6],'Top Paid Higher rating '+gens[6]+' apps')

plt.subplot(428)
squatifyChart_paid(gens[7],'Top Paid Higher rating '+gens[7]+' apps')


# #### <div id="sup_device"> --- Find Most Supported devices apps --- </div>

# In[ ]:


apps.sort_values(["sup_devices.num","user_rating"],ascending=False).head()


# #### <div id="total_sup_device"> --- Find Total apps in Most Supported devices ---</div>
# 

# In[ ]:


sup_devices_apps = pd.DataFrame({
    'number_of_devices' :apps["sup_devices.num"].value_counts().index,
    'total_number_of_apps' : apps["sup_devices.num"].value_counts()
},columns=['number_of_devices','total_number_of_apps'])

sup_devices_apps.head()


# #### <div id="total_sup_device_genres"> --- Find Total apps in Most Supported devices genrewise ---</div>
# 

# In[ ]:


def sup_device_genre(genre):
    genre_apps = apps.groupby("prime_genre").get_group(genre)
    return pd.DataFrame({
              genre : genre_apps["sup_devices.num"].value_counts(),
                },columns=[genre])


# In[ ]:


for g in genres:
    sup_devices_apps[g] = sup_device_genre(g)
  
    
sup_devices_apps.fillna(0, inplace=True)
sup_devices_apps.sort_values('number_of_devices', ascending= False, inplace=True)
sup_devices_apps.head()


# #### <div id="total_sup_device_free"> --- Find Total Free apps in Most Supported devices ---</div>
# 

# In[ ]:


sup_devices_free_apps = pd.DataFrame({
    'number_of_devices' :freeapps["sup_devices.num"].value_counts().index,
    'total_number_of_free_apps' : freeapps["sup_devices.num"].value_counts()
},columns=['number_of_devices','total_number_of_free_apps'])

sup_devices_free_apps.head()


# #### <div id="total_sup_device_paid"> --- Find Total Paid apps in Most Supported devices ---</div>
# 

# In[ ]:


sup_devices_paid_apps = pd.DataFrame({
    'number_of_devices' : paidapps["sup_devices.num"].value_counts().index,
    'total_number_of_paid_apps' : paidapps["sup_devices.num"].value_counts()
},columns=['number_of_devices','total_number_of_paid_apps'])

sup_devices_paid_apps.head()


# ### <div id="pearson"> Pearson Correlation  </div>

# Remove columns that are not much useful but how will you decide whether it is important for you or not.<br>
# well, there are many ways : <br>
# 1. It depends on what is your purpose for making model.
# 2. various correlation techniques such a pearson correction.
# 
# Here, the link : https://www.yashpatel.tech/online-tutorials/pearson-correlation-for-feature-selection-in-ml/
# 
# For now I am deleting currency and version fields as I feel that it is not much useful

# In[ ]:


plt.figure(figsize=(12,10))
cor = apps.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# - A value closer to 0 implies a weaker correlation (exact 0 implying no correlation)
# - A value closer to 1 implies a stronger positive correlation
# - A value closer to -1 implies a stronger negative correlation

# In[ ]:


new_apps = apps

del new_apps['currency']
del new_apps['ver']

new_apps.head()


#  ### <div id="feature_engineer"> Feature engineering </div>

# Let's Create a new features 

# In[ ]:


new_apps['total_users_rating'] = new_apps['rating_count_tot'] / new_apps['user_rating']
new_apps['total_users_rating_cur'] = new_apps['rating_count_ver'] / new_apps['user_rating_ver']

del new_apps['rating_count_tot']
del new_apps['user_rating']
del new_apps['rating_count_ver']
del new_apps['user_rating_ver']

new_apps['total_users_rating'].fillna(0, inplace=True)
new_apps['total_users_rating_cur'].fillna(0, inplace=True)

new_apps['total_users_rating'] = new_apps['total_users_rating'].apply(lambda x : math.ceil(x))
new_apps['total_users_rating_cur'] = new_apps['total_users_rating_cur'].apply(lambda x : math.ceil(x))
new_apps.head()


# Check whether any error in support language field...

# In[ ]:


len(new_apps[new_apps['lang.num'] == 0])


# Here, this is an error as it is not possible that game has 0 language <br>
# So, Let's replace 0 to 1 

# In[ ]:


new_apps['lang.num'].replace(0,1, inplace= True)
len(new_apps[new_apps['lang.num'] == 0])


#  <div id="games">Total number of Games that are available in a multiple languages</div>

# In[ ]:


print("Total number of apps that are available in a multiple languages : {} ".format(len(new_apps[new_apps['lang.num'] > 1])))
print("Number of apps that are available in only one language languages : {} ".format(len(new_apps[new_apps['lang.num'] == 1])))
print("Number of apps that are available in the range of 5 to 20 languages : {} ".format(len(new_apps[(new_apps['lang.num'] >= 5) & (new_apps['lang.num'] <= 20)])))
print("Number of apps that are available in the range of 20 to 50 languages : {} ".format(len(new_apps[(new_apps['lang.num'] >= 20) & (new_apps['lang.num'] <= 50)])))
print("Number of apps that are available in more than 50 languages : {} ".format(len(new_apps[new_apps['lang.num'] >= 50])))


#  <div id="maxlangapp">Maximum languges support application</div>

# In[ ]:


print("Maximum languges support application : ")
new_apps[['track_name','lang.num','prime_genre']][new_apps['lang.num'] == max(new_apps['lang.num'])]


# 
