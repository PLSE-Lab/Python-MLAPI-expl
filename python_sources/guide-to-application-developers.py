#!/usr/bin/env python
# coding: utf-8

# **The rate at which mobile applications are being released into the market (Appstore or playstore)  is greater than ever before. To survive in a such  intense market conditions, an application developer should develop a significant understanding of the factors influencing the count of ratings and average user rating. This kernel provides an overview of the factors influencing the ratings. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


Data=pd.read_csv("../input/AppleStore.csv")
Data=Data.iloc[0:,2:]
#print(Data.shape)
Data.head(6)
#Converted the size in MB
Data["size_bytes"]=Data["size_bytes"]*0.000001


# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(y=Data["prime_genre"])
plt.xlabel("Number of applications", fontsize=12, color='blue')
plt.ylabel("Genre",fontsize=12, color='blue')
plt.title("Figure 1: Number of applications present in each genre by the end of 2018",fontsize=14, color='blue')


# **The above figure suggests that the magnitude of the game genre in the app store is very high in comparision with any other genre. Following the game genre, entertainment, education, Photo and video and utilities genres can be ranked 2,3,4 and 5th in terms of number of applications in the app store. The initial scope of this kernel is restricted to only these five genres. **

# In[ ]:



Index=["Games","Education","Entertainment","Utilities","Photo & Video"]
X=["Games","Education","Entertainment","Utilities","Photo & Video"]
for i in range (0,5):
   X[i]=Data[Data["prime_genre"]==Index[i]]

Y=["Games_Notrated","Education_Notrated","Entertainment_Notrated","Utilities_Notrated","Photo_Video_Notrated"]
for i in range (0,5): 
    Y[i]=X[i][X[i]["rating_count_tot"]==0]

Z=["Games_rated","Education_rated","Entertainment_rated","Utilities_rated","Photo_Video_rated"]
for i in range (0,5): 
    Z[i]=X[i][X[i]["rating_count_tot"]>0]
K=[]
for i in range (0,5):
    K.append(len(Y[i]))

J=[]
for i in range (0,5):
    J.append(len(Z[i]))

A=[]
B=[]
C=[]
D=[]
E=[]
L=[A,B,C,D,E]

for i in range (0,5):
    L[i].append(K[i])
    L[i].append(J[i])
    
Labels=["Not rated",'Rated']  
fig=plt.figure(figsize=(12,12))
for i in range (1,6):
    ax = fig.add_subplot(3,2,i)
    plt.pie(L[i-1],labels=Labels,autopct='%.2f%%',explode=(0,0))
    plt.title(Index[i-1])
    plt.axis('equal')
fig.show()


# **Except for the photo and video genre, the remaining four genres have more than 10% of applications that have received  user ratings counts and the average rating of the application is zero. The following part of the kernel tries to understand what are the factors that promoting the ratings and average user rating value in the remaining 90% of applications. 
# 
# **Best way to identify the factors influencing the ratings is correlation plots. Correlation plots for the five genres indicated above are generated and presented. Following correlation plots, scatter plots are generated to identify the how each factory is influencing the user ratings counts and averate user rating value. Plots corresponding to total count of ratings and averate user rating value are plotted in blue and red respectively
# 
# 

# In[ ]:


#Eliminate the outliers.
Z[0]=Z[0][Z[0]["rating_count_tot"]<1000000]
Z[1]=Z[1][Z[1]["rating_count_tot"]<25000]
Z[2]=Z[2][Z[2]["rating_count_tot"]<150000]
Z[3]=Z[3][Z[3]["rating_count_tot"]<150000]
Z[4]=Z[4][Z[4]["rating_count_tot"]<500000]

#Generating correlation plots
T=["Corr1","Corr2","Corr3","Corr4","Corr5"]
for i in range (0,5):
    T[i]=Z[i].corr()
for i in range (1,6):
    plt.figure(figsize=(8,8))
    sns.heatmap(T[i-1], annot=True, fmt=".1f")
    plt.title(Index[i-1])
    plt.show()
    


# In[ ]:


fig4=plt.figure(figsize=(20,15))
plt.subplot(221)
plt.scatter(Z[0]["rating_count_tot"],Z[0]["price"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Price",fontsize=16)
plt.title("Games",fontsize=16)
plt.subplot(222)
plt.scatter(Z[0]["rating_count_tot"],Z[0]["lang.num"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Number of languages supported", fontsize=16)
plt.title("Games",fontsize=16)
plt.subplot(223)
plt.scatter(Z[0]["user_rating"],Z[0]["lang.num"],color='r')
plt.xlabel("Average user rating value", fontsize=16)
plt.ylabel("Number of languages supported",fontsize=16)
plt.title("Games", fontsize=16)


# In[ ]:


fig4=plt.figure(figsize=(20,28))
plt.subplot(421)
plt.scatter(Z[1]["rating_count_tot"],Z[1]["size_bytes"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Size in MB",fontsize=16)
plt.title("Education",fontsize=16)
plt.subplot(422)
plt.scatter(Z[1]["rating_count_tot"],Z[1]["lang.num"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Education",fontsize=16)
plt.subplot(423)
plt.scatter(Z[1]["rating_count_tot"],Z[1]["sup_devices.num"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Number of supporting devices",fontsize=16)
plt.title("Education",fontsize=16)
plt.subplot(424)
plt.scatter(Z[1]["user_rating"],Z[1]["size_bytes"], color='r')
plt.xlabel("Average user rating value",fontsize=16)
plt.ylabel("Size in MB",fontsize=16)
plt.title("Education",fontsize=16)
plt.subplot(425)
plt.scatter(Z[1]["user_rating"],Z[1]["sup_devices.num"],color='r')
plt.xlabel("Average user rating value",fontsize=16)
plt.ylabel("Number of supporting devices",fontsize=16)
plt.title("Education",fontsize=16)
plt.subplot(426)
plt.scatter(Z[1]["user_rating"],Z[1]["ipadSc_urls.num"],color='r')
plt.xlabel("Average user rating value",fontsize=16)
plt.ylabel("Number of screenshots showed for display",fontsize=16)
plt.title("Education",fontsize=16)
plt.subplot(427)
plt.scatter(Z[1]["user_rating"],Z[1]["lang.num"],color='r')
plt.xlabel("Average user rating value",fontsize=16)
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Education",fontsize=16)


# 

# In[ ]:


fig45=plt.figure(figsize=(20,25))
plt.subplot(421)
plt.scatter(Z[2]["rating_count_tot"],Z[2]["price"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Price",fontsize=16)
plt.title("Entertainment",fontsize=16)
plt.subplot(422)
plt.scatter(Z[2]["rating_count_tot"],Z[2]["lang.num"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Entertainment",fontsize=16)
plt.subplot(423)
plt.scatter(Z[2]["rating_count_tot"],Z[2]["ipadSc_urls.num"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Number of screenshots showed for display",fontsize=16)
plt.title("Entertainment",fontsize=16)
plt.subplot(424)
plt.scatter(Z[2]["rating_count_tot"],Z[2]["sup_devices.num"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Number of supporting devices",fontsize=16)
plt.title("Entertainment",fontsize=16)
plt.subplot(425)
plt.scatter(Z[2]["rating_count_tot"],Z[2]["vpp_lic"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Vpp Device Based Licensing Enabled",fontsize=16)
plt.title("Entertainment",fontsize=16)
plt.subplot(426)
plt.scatter(Z[2]["user_rating"],Z[2]["lang.num"],color='r')
plt.xlabel("Average User Rating value ",fontsize=16)
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Entertainment",fontsize=16)
plt.subplot(427)
plt.scatter(Z[2]["user_rating"],Z[2]["size_bytes"],color='r')
plt.xlabel("Average User Rating value ",fontsize=16)
plt.ylabel("Size in MB",fontsize=16)
plt.title("Entertainment",fontsize=16)


# In[ ]:



fig45=plt.figure(figsize=(20,35))
plt.subplot(521)
plt.scatter(Z[3]["rating_count_tot"],Z[3]["price"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Price",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(522)
plt.scatter(Z[3]["rating_count_tot"],Z[3]["ipadSc_urls.num"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Number of screenshots showed for display",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(523)
plt.scatter(Z[3]["rating_count_tot"],Z[3]["lang.num"])
plt.xlabel("User rating counts")
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(524)
plt.scatter(Z[3]["rating_count_tot"],Z[3]["size_bytes"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Size in MB",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(525)
plt.scatter(Z[3]["user_rating"],Z[3]["size_bytes"], color='r')
plt.xlabel("Average User Rating value ",fontsize=16)
plt.ylabel("Size in MB",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(526)
plt.scatter(Z[3]["user_rating"],Z[3]["price"], color='r')
plt.xlabel("Average User Rating value ",fontsize=16)
plt.ylabel("Price",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(527)
plt.scatter(Z[3]["user_rating"],Z[3]["sup_devices.num"], color='r')
plt.xlabel("Average User Rating value",fontsize=16)
plt.ylabel("Number of supporting devices",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(528)
plt.scatter(Z[3]["user_rating"],Z[3]["ipadSc_urls.num"], color='r')
plt.xlabel("Average User Rating value",fontsize=16)
plt.ylabel(" Number of screenshots showed for display",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(529)
plt.scatter(Z[3]["user_rating"],Z[3]["lang.num"], color='r')
plt.xlabel("Average User Rating value ",fontsize=16)
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Utilities",fontsize=16)


# In[ ]:


X[4]=X[4][X[4]["rating_count_tot"]<150000]
fig6=plt.figure(figsize=(20,20))
plt.subplot(321)
plt.scatter(Z[4]["rating_count_tot"],Z[4]["price"])
plt.xlabel("User Rating counts",fontsize=16)
plt.ylabel("Price",fontsize=16)
plt.title("Photo & Video",fontsize=16)
plt.subplot(322)
plt.scatter(Z[4]["rating_count_tot"],Z[4]["size_bytes"])
plt.xlabel("User Rating counts",fontsize=16)
plt.ylabel("Size in MB",fontsize=16)
plt.title("Photo & Video",fontsize=16)
plt.subplot(323)
plt.scatter(Z[4]["rating_count_tot"],Z[4]["lang.num"])
plt.xlabel("User Rating counts",fontsize=16)
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Photo & Video",fontsize=16)
plt.subplot(324)
plt.scatter(Z[4]["user_rating"],Z[4]["ipadSc_urls.num"], color='r')
plt.xlabel("Average User Rating value",fontsize=16)
plt.ylabel("Number of screenshots showed for display",fontsize=16)
plt.title("Photo & Video",fontsize=16)
plt.subplot(325)
plt.scatter(Z[4]["user_rating"],Z[4]["size_bytes"], color='r')
plt.xlabel("Average User Rating value",fontsize=16)
plt.ylabel("Size in MB",fontsize=16)
plt.title("Photo & Video",fontsize=16)
plt.subplot(326)
plt.scatter(Z[4]["user_rating"],Z[4]["lang.num"], color='r')
plt.xlabel("Average User Rating value",fontsize=16)
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Photo & Video",fontsize=16)


# In[ ]:




