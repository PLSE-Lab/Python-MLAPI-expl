#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
plt.style.use("seaborn-pastel")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load the excel file

# In[ ]:


df = pd.read_excel("/kaggle/input/yearly-highest-paid-100-athletes-forbes-20122019/Forbes Athlete List 2012-2019.xlsx")
df.head(10)


# It contains 7 columns, 795 rows:
# * **Rank**: Yearly rank based on pay
# * **Name**: Name of the athlete
# * **Pay**: Salary and endorsement pays altogether
# * **Salary/Winnings**: Salary of the athlete
# * **Endorsements**: Earnings from commercials, social media, sponsors etc.
# * **Sport**: Type of the sport of the athlete
# * **Year**: Year of the pay

# In[ ]:


df.info()


# ## Data preprocessing
# 
# Some of the columns are inconsistent in the dataset as Forbes changed their mind wheter to put **"#"** before the rank value over time. Lets fix this one as well as taking out **"dollar signs"** and **"M"**. Let's also change "Soccer" to "Football" and "Football" to "American Football"

# In[ ]:


df.Rank = df.Rank.apply(lambda x: int(x.split("#")[1]) if type(x) == np.str else x)
df.Pay = df.Pay.apply(lambda x: float(x.split(" ")[0].split("$")[1]))
df.Endorsements = df.Endorsements.apply(lambda x: float(x.split(" ")[0].split("$")[1]))
df["Salary/Winnings"].replace("-",'$nan M',inplace=True)
df["Salary/Winnings"] = df["Salary/Winnings"].apply(lambda x: float(x.split(" ")[0].split("$")[1]))
df.Sport.replace({"Soccer":"Football",
                  "Football":"American Football",
                 "Mixed Martial Arts":"MMA",
                 "Auto racing":"Racing",
                  "Auto Racing":"Racing",
                  "Basketbal":"Basketball",
                 },inplace=True)

df.columns = ['Rank', 'Name', 'Pay', 'Salary_Winnings', 'Endorsements', 'Sport', 'Year']
df.head(10)


# # Exploratory Data Analysis
# 
# Let's see the progress of Lionel Messi's pay over the years

# In[ ]:


messi = df[df.Name == "Lionel Messi"].sort_values("Year")
messi


# In[ ]:


sns.barplot(data=messi,x="Pay",y="Year",orient="h")
plt.title("Messi's Pay Progress")
plt.show()


# Let's see the distribution of athletes in the dataset based on their sport type

# In[ ]:


df.groupby("Name").first()["Sport"].value_counts().plot(kind="pie",autopct="%.0f%%",figsize=(8,8),wedgeprops=dict(width=0.4),pctdistance=0.8)
plt.ylabel(None)
plt.title("Breakdown of Athletes by Sport",fontweight="bold")
plt.show()


# Total earnings by sport type:
# * Even though basketball has the highest total pay it is due to frequency of players
# * Yearly average pays (in million USD) of fighters are much higher as compared to basketball player averages

# In[ ]:


sports = df.groupby("Sport").agg(
    total_pay = ("Pay","sum"),
    no_of_players = ("Name","count")
)

sports["pay_per_player"] = sports.total_pay/sports.no_of_players
sports.sort_values("pay_per_player",ascending=False)


# ## Racing bar for the cumulative earnings of players
# Let's visualise cummulative pays of the athletes in a racing bar animation. First we convert the year column into datetime object:
# 

# In[ ]:


df.Year = pd.to_datetime(df.Year,format="%Y")
df.dtypes


# Then prepare a pivot table where columns are athletes and index is years:
# * The table have 8 rows (year) against 300 athletes

# In[ ]:


racing_bar_data = df.pivot_table(values="Pay",index="Year",columns="Name")
racing_bar_data.cumsum()


# Following athletes are the only ones which are consistently involved in the Top100 list for each year since 2012. The rest of the athletes have NaN values. We will first interpolate NaNs linearly and use fill remaining NaNs with backward filling:

# In[ ]:


racing_bar_data.columns[racing_bar_data.isnull().sum() == 0]


# In[ ]:


racing_bar_filled = racing_bar_data.interpolate(method="linear").fillna(method="bfill")
racing_bar_filled


# Now convert the data into cumulative sum of pays over years:

# In[ ]:


racing_bar_filled = racing_bar_filled.cumsum()
racing_bar_filled


# Now let's oversample the dataset with interpolation (linear) for a smooth transitions in frames in the animation. Rows will represent days now.

# In[ ]:


racing_bar_filled = racing_bar_filled.resample("1D").interpolate(method="linear")[::7]


# This is how Messi's cumulative pay looks over time after resampling and interpolations

# In[ ]:


racing_bar_filled["Lionel Messi"].plot(marker=".",figsize=(12,4))
plt.ylabel("Cumulative Pay (Million USD)")
plt.show()


# Import necessary packages for creating animation and saving them, and initiate the plots and their elements (lines, bars, texts etc.). Following codes will generate an animation for the Top 10 highest paid athletes between 2012 and 2019:

# In[ ]:


from matplotlib.animation import FuncAnimation, FFMpegWriter

selected  = racing_bar_filled.iloc[-1,:].sort_values(ascending=False)[:20].index
data = racing_bar_filled[selected].round()

fig,ax = plt.subplots(figsize=(9.3,7))
fig.subplots_adjust(left=0.18)
no_of_frames = data.shape[0] #Number of frames

#initiate the barplot with the first rows of the dataframe
bars = sns.barplot(y=data.columns,x=data.iloc[0,:],orient="h",ax=ax)
ax.set_xlim(0,1500)
txts = [ax.text(0,i,0,va="center") for i in range(data.shape[1])]
title_txt = ax.text(650,-1,"Date: ",fontsize=12)
ax.set_xlabel("Pay (Millions USD)")
ax.set_ylabel(None)

def animate(i):
#     print(f"i={i}/{no_of_frames}")
    #get i'th row of data 
    y = data.iloc[i,:]
    
    #update title of the barplot axis
    title_txt.set_text(f"Date: {str(data.index[i].date())}")
    
    #update elements in both plots
    for j, b, in enumerate(bars.patches):
        #update each bar's height
        b.set_width(y[j])
        
        #update text for each bar (optional)
        txts[j].set_text(f"${y[j].astype(int)}M")
        txts[j].set_x(y[j])

anim=FuncAnimation(fig,animate,repeat=False,frames=no_of_frames,interval=1,blit=False)
anim.save('athletes.gif', writer='imagemagick', fps=120)
plt.close(fig)


# ![](./athletes.gif)
