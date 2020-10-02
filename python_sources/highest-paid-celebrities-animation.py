#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
plt.style.use("seaborn-pastel")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Load Data

# The dataset contains salary/earning information of top 100 celebrities published by Forbes in the form of "Celebrity 100" list.
# 
# Columns of the table:
# * **Name**: Celebrity name
# * **Pay (USD millions)**: Earning of the celebrity for the given year
# * **Year**: Year of the earning (YYYY format)
# * **Category**: Category of fame such as athlete, actor, actress etc.

# In[ ]:


df = pd.read_csv("/kaggle/input/forbes-celebrity-100-since-2005/forbes_celebrity_100.csv")
df.head()


# The data consist of 4 columns and 1547 rows. Each row represents a celebrity's for given year. The dataset is ordered by year column.

# In[ ]:


df.info()


# ## Exploratory Data Analysis

# Even though it is an annual Top 100 list, yearly number of celebrities in the list is changed since 2014.

# In[ ]:


df.groupby("Year").agg(Celebrity_Count=("Name","count"))


# ### Breakdown of the Category column
# 
# * Celebrities are represented in 13 different categories
# * Athletes and musicians are almost half of the celebrities (47%)
# * Actors, actresses (including TV) constitutes of 25% of the names

# In[ ]:


df.groupby("Name").first()["Category"].value_counts().plot(kind="pie",autopct="%.0f%%",pctdistance=0.8,
                                                          wedgeprops=dict(width=0.4),figsize=(8,8))
plt.ylabel(None)
plt.show()


# There are 432 unique names in the data

# In[ ]:


len(df.Name.unique())


# ## Earning of the Categories
# * Musicians and Athletes make tops the sum of earning list as they are represented with larger counts
# * Directors/Producers makes the most average annual earnings amongst the other categories

# In[ ]:


earning_by_category = df.groupby("Category").agg(celebrity_count=("Name","count"),
                          total_earning=("Pay (USD millions)","sum"),
                          mean_earning=("Pay (USD millions)","mean")).sort_values("total_earning",ascending=False)
earning_by_category


# In[ ]:


plt.figure(figsize=(12,7))
sns.barplot(data=earning_by_category,x="total_earning",y=earning_by_category.index,orient="h")
plt.title("Sum of Earnings by Category (2005-2019)",fontweight="bold",fontsize=16)
plt.xlabel("Earning (in USD millions)")
for i,count in enumerate(earning_by_category.celebrity_count):
    use = earning_by_category.total_earning[i]/1000
    plt.text(-1500,i,count,va="center")
    plt.text(earning_by_category.total_earning[i],i,
             f"{int(use)}Bn" if use>1 else f"{round(use,2)}Bn")
plt.xlim(left=-2500)
plt.text(-2000,-0.6,"Celebrity\nCount",fontweight="bold")
plt.show()


# In[ ]:


plt.figure(figsize=(12,7))  
sns.barplot(data=earning_by_category,x="mean_earning",y=earning_by_category.index,orient="h",
            order=earning_by_category.sort_values("mean_earning",ascending=False).index)
plt.title("Average Yearly Earnings by Category",fontweight="bold",fontsize=16)
plt.xlabel("Earning (in USD millions)")
for i,earn in enumerate(earning_by_category.sort_values("mean_earning",ascending=False).mean_earning):
    plt.text(earn,i,f"{round(earn,1)}M")

plt.tight_layout()
plt.show()


# The above graphs can also be shown without using groupby functions:

# In[ ]:


fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,7))
sns.barplot(data=df,x="Pay (USD millions)",y="Category",orient="h",estimator=np.sum,ax=ax1)
sns.barplot(data=df,x="Pay (USD millions)",y="Category",orient="h",ax=ax2)
plt.tight_layout()
plt.show()


# ### Highest Paid Celebrities

# In[ ]:


top_paid = df.groupby("Name").agg(pay=("Pay (USD millions)","sum"),
                      category=("Category","first"),av_pay=("Pay (USD millions)","mean")).sort_values("pay",ascending=False).head(15)
plt.figure(figsize=(12,7))
sns.barplot(data=top_paid,x="pay",y=top_paid.index,orient="h")
sns.barplot(data=top_paid,x="av_pay",y=top_paid.index,orient="h",color="sandybrown")
plt.xlabel("Earning (USD Millions)")
plt.title("Top Earning Celebrities (2005-2019)",fontweight="bold",fontsize=16)
plt.text(0,-0.8,"Average Annual Pay",fontweight="bold")
for i,pay in enumerate(top_paid.pay):
    plt.text(pay,i,f"{int(pay)}M",va="center")
    plt.text(top_paid.av_pay[i],i,f"{int(top_paid.av_pay[i])}M",va="center")
    
plt.ylabel(None)
plt.show()


# ### Top Earning Athletes

# In[ ]:


top_paid = df[df.Category == "Athletes"].groupby("Name").agg(pay=("Pay (USD millions)","sum"),
                      category=("Category","first"),av_pay=("Pay (USD millions)","mean")).sort_values("pay",ascending=False).head(15)
plt.figure(figsize=(12,7))
sns.barplot(data=top_paid,x="pay",y=top_paid.index,orient="h")
sns.barplot(data=top_paid,x="av_pay",y=top_paid.index,orient="h",color="sandybrown")
plt.xlabel("Earning (USD Millions)")
plt.title("Top Earning Athletes (2005-2019)",fontweight="bold",fontsize=16)
plt.text(0,-0.8,"Average Annual Pay",fontweight="bold")
for i,pay in enumerate(top_paid.pay):
    plt.text(pay,i,f"{int(pay)}M",va="center")
    plt.text(top_paid.av_pay[i],i,f"{int(top_paid.av_pay[i])}M",va="center")
    
plt.ylabel(None)
plt.show()


# ### Top Earning Actors

# In[ ]:


top_paid = df[df.Category == "Actors"].groupby("Name").agg(pay=("Pay (USD millions)","sum"),
                      category=("Category","first"),av_pay=("Pay (USD millions)","mean")).sort_values("pay",ascending=False).head(15)
plt.figure(figsize=(12,7))
sns.barplot(data=top_paid,x="pay",y=top_paid.index,orient="h")
sns.barplot(data=top_paid,x="av_pay",y=top_paid.index,orient="h",color="sandybrown")
plt.xlabel("Earning (USD Millions)")
plt.title("Top Earning Actors (2005-2019)",fontweight="bold",fontsize=16)
plt.text(0,-0.8,"Average Annual Pay",fontweight="bold")
for i,pay in enumerate(top_paid.pay):
    plt.text(pay,i,f"{int(pay)}M",va="center")
    plt.text(top_paid.av_pay[i],i,f"{int(top_paid.av_pay[i])}M",va="center")
    
plt.ylabel(None)
plt.show()


# ### Animate cumulative pays of Actors (2005-2019)

# In[ ]:


from matplotlib.animation import FuncAnimation, FFMpegWriter

data = df[df.Category=="Actors"].pivot_table(values="Pay (USD millions)",index="Year",columns="Name").cumsum().fillna(method="ffill")
data.index = pd.to_datetime(data.index,format="%Y")
data = data.resample("D").interpolate(method="linear")[::7]

#select top paid actors only
data = data[top_paid.index]

fig,ax = plt.subplots(figsize=(12,7))
fig.subplots_adjust(left=0.15)
no_of_frames = data.shape[0] #Number of frames

#initiate the barplot with the first rows of the dataframe
bars = sns.barplot(y=data.columns,x=data.iloc[0,:],orient="h",ax=ax)
ax.set_xlim(0,1000)
txts = [ax.text(0,i,0,va="center") for i in range(data.shape[1])]
title_txt = ax.text(500,-1,"Date: ",fontsize=12)
ax.set_xlabel("Pay (Millions USD)")
ax.set_ylabel(None)

def animate(i):
    print("{0}/{1}".format(i,no_of_frames-1),end="\r")
    #get i'th row of data 
    y = data.iloc[i,:]
    
    #update title of the barplot axis
    title_txt.set_text(f"Date: {str(data.index[i].date())}")
    
    #update elements in both plots
    for j, b, in enumerate(bars.patches):
        #update each bar's height
        b.set_width(y[j])
        
        #update text for each bar (optional)
        txts[j].set_text((y[j].astype(int)))
        txts[j].set_x(y[j])

print("Creating Animation")
anim=FuncAnimation(fig,animate,repeat=True,frames=no_of_frames,interval=1,blit=False)
print("Saving into .gif file")
anim.save('actors.gif', writer='imagemagick',fps=30)
plt.close(fig)


# Animation saved in a .gif file which looks like below:

# ![](./actors.gif)

# ![](./messi2.gif)
