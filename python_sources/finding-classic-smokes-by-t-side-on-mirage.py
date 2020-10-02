#!/usr/bin/env python
# coding: utf-8

# # Finding Classic Smokes by T-side on mirage
# 
# Hello!  Here is a kernel that uses K means to find a few common nades on mirage.

# In[25]:


import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# In[26]:


csgo = pd.read_csv('../input/mm_grenades_demos.csv')
index=[(csgo['nade']=='Smoke') & 
       (csgo['map']=='de_mirage') & 
       (csgo['att_side'] == 'Terrorist')][0]
csgo_sm=csgo[index]
csgo_sm_small=csgo_sm[[ 'nade_land_x',
                       'nade_land_y',
                       'att_rank', 
                       'att_pos_x', 
                       'att_pos_y']]
csgo_sm_small.dropna()


# Here is a function to convert x,y in game coordinates to coordinates on an image.
# 

# In[27]:


def pointx_to_resolutionx(xinput,startX=-3217,endX=1912,resX=1024):
    sizeX=endX-startX
    if startX < 0:
        xinput += startX *(-1.0)
    else:
        xinput += startX
    xoutput = float((xinput / abs(sizeX)) * resX);
    return xoutput

def pointy_to_resolutiony(yinput,startY=-3401,endY=1682,resY=1024):
    sizeY=endY-startY
    if startY < 0:
        yinput += startY *(-1.0)
    else:
        yinput += startY
    youtput = float((yinput / abs(sizeY)) * resY);
    return resY-youtput


# In[28]:


csgo_sm_small['thrower_xpos']=csgo_sm_small['att_pos_x'].apply(pointx_to_resolutionx)
csgo_sm_small['thrower_ypos']=csgo_sm_small['att_pos_y'].apply(pointy_to_resolutiony)
csgo_sm_small['nade_ypos']=csgo_sm_small['nade_land_y'].apply(pointy_to_resolutiony)
csgo_sm_small['nade_xpos']=csgo_sm_small['nade_land_x'].apply(pointx_to_resolutionx)


# In[29]:


csgo_sm_small.head() #looks like it worked fine...


# lets see all of the thrower and nade data

# In[30]:


im = plt.imread('../input/de_mirage.png')
plt.figure(figsize=(11,11))
t = plt.imshow(im)
t = plt.scatter(csgo_sm_small['nade_xpos'], csgo_sm_small['nade_ypos'],alpha=0.05,c='blue')
t = plt.scatter(csgo_sm_small['thrower_xpos'], csgo_sm_small['thrower_ypos'],alpha=0.05,c='red')


# In[31]:


#Drop old raw columns so we don't get confused
csgo_sm_small.drop(['nade_land_x','nade_land_y','att_pos_x','att_pos_y'],inplace=True,axis=1)


# In[32]:


csgo_sm_small.columns


# In[33]:


#rename my frame css cause its too long
css=csgo_sm_small


# calculate point density (probably inefficient, and square density not the best, but this is a rough thing anyway)

# In[34]:


def calc_N_Nearby(x,y,xarr,yarr,dt=15):
    index=[(xarr < (x + dt)) & (xarr > (x - dt)) & (yarr < (y + dt)) & (yarr > (y - dt))][0]
    return len(xarr[index])

zarr15=[]
#zarr10=[] #I already know I want density 15 but others could be useful...
#zarr5=[]
#takes a hot minute
for i in range(len(css['thrower_xpos'])):
    zarr15.append(calc_N_Nearby(css['thrower_xpos'].iloc[i],css['thrower_ypos'].iloc[i],
              css['thrower_xpos'],css['thrower_ypos'],dt=15))
    #zarr10.append(calc_N_Nearby(css['thrower_xpos'].iloc[i],css['thrower_ypos'].iloc[i],
    #          css['thrower_xpos'],css['thrower_ypos'],dt=10))
    #zarr5.append(calc_N_Nearby(css['thrower_xpos'].iloc[i],css['thrower_ypos'].iloc[i],
    #          css['thrower_xpos'],css['thrower_ypos'],dt=5))
    #print(z)
    


# In[35]:


css['density15']=zarr15


# In[36]:


x=plt.hist(zarr15,bins=75)


# In[38]:


index=[(css['density15']>100)][0]#I use this notation cause I'm used to a language called IDL and it works like this
im = plt.imread('../input/de_mirage.png')
plt.figure(figsize=(12,12))
implot = plt.imshow(im)
implot = plt.scatter(css['thrower_xpos'][index], css['thrower_ypos'][index],alpha=0.15
                     ,c='red',s=5) 
implot = plt.scatter(css['nade_xpos'][index], css['nade_ypos'][index],alpha=0.15
                     ,c='blue',s=5) 


# A ton more exploration could be done changing the density threshold and whatnot, but this looks fine for now

# In[40]:


index=[(css['density15']>100)][0]


# In[41]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=25) #n clusters required??
kmeans.fit(css[["thrower_xpos","thrower_ypos"]][index])
c_centers=pd.DataFrame(kmeans.cluster_centers_)


# In[44]:


im = plt.imread('../input/de_mirage.png')
plt.figure(figsize=(12,12))
implot = plt.imshow(im)
implot = plt.scatter(css['thrower_xpos'][index], css['thrower_ypos'][index],alpha=0.25
                     ,c='red',s=5) 
implot = plt.scatter(css['nade_xpos'][index], css['nade_ypos'][index],alpha=0.25
                     ,c='blue',s=5) 

implot = plt.scatter(c_centers[0],c_centers[1],alpha=1.0
                     ,c='green',s=35) 


# In[45]:


#perhaps the worst way to do this?
css['kmeanslabel']=np.zeros(len(index))
css['kmeanslabel'][index]=kmeans.labels_


# In[50]:


#for i in range(0,27): #causes max figure opened error/warning...
for i in range(1,3):
    index=[css['kmeanslabel'] == i][0]
    im = plt.imread('../input/de_mirage.png')
    plt.figure(figsize=(12,12))
    implot = plt.imshow(im)
    implot = plt.scatter(css['thrower_xpos'][index], css['thrower_ypos'][index],alpha=0.25
                         ,c='red',s=5) 
    implot = plt.scatter(css['nade_xpos'][index], css['nade_ypos'][index],alpha=0.25
                         ,c='blue',s=5) 
    #implot = plt.savefig("picked_out_nades_"+str(i)+".png")


# # Full results are here [https://imgur.com/a/56QMh](https://imgur.com/a/56QMh)

# Clearly much more can be done with this data.  You could answer questions like
# * Do high rank players throw different nades than low ranked players?
# * Can this be used to figure out if a smoke was throw correctly/incorrectly? 
# * What about CT smokes? 
# * What are some smokes thrown after the bomb is planted?
# * What about other maps?
# * What about flashes/nades/mollies?
# 
# I can think of many more questions.  However, I want to try and find a more automated cluster finding algorithm.  
# 
# I think DBSCAN or some other hierarchial clustering algorithms may be the way but I need to do more research on them 
# 

# In[ ]:





# In[ ]:





# In[ ]:




