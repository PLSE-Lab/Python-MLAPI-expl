#!/usr/bin/env python
# coding: utf-8

# <img src=https://www.rossmann-cdn.de/dam/jcr:0d0bb06a-8527-4d7c-ac70-7cb60e7d0007/cor-presse-download-logos-rossmann-drogeriemarkt.2015-12-07-16-37-10.jpg width="700">

# <h2><center> By Stefano Zakher</center></h2>

# <h2><center> EDA and Forecasting with
#     RandomForest Regressor </center></h2>
# 
# 
#  **Objective:** 
# 
# - Data Wrangling and Exploration (treat outliers, handle missing values etc.).
# - Correlation analysis with Store dataset joined.
# - Training Model with RF 
# - Visualize and evaluate model
# - Choose the best performing one and predict the next 6 weeks of sales

# The following dataset created by Rossman on its Store Sales and Information of its different drug stores:
# 
# <table>
# <tr>
#     <th><center>Columns</center></th>
#     <th><center>Descriptions</center></th>
# </tr>
# <tr>
#     <td><center>Id</center></td>
#     <td><center> An Id that represents a (Store, Date) duple within the test set</center></td>
# </tr>
# <tr>
#     <td><center>Store</center></td>
#     <td><center>A unique Id for each store</center></td>
# </tr>
# <tr>
#     <td><center>Sales</center></td>
#     <td><center>The turnover on a given day (our target variable) </center></td>
# </tr>
# <tr>
#     <td><center>Customers</center></td>
#     <td><center>The number of customers on a given day</center></td>
# </tr>
# <tr>
#     <td><center>Open</center></td> 
#     <td><center>open: 0 = the store is closed , 1 = the store is open</center></td>
# </tr>
# <tr>
#     <td><center>StateHoliday</center></td>
#     <td><center>Indicates a state holiday. a = public holiday, b = Easter holiday, c = Christmas, 0 = None</center></td>
# </tr>
# <tr>
#     <td><center>SchoolHoliday</center></td>
#     <td><center> Store on this Date was affected or not by the closure of public schools</center></td>
# </tr>
# <tr>
#     <td><center>StoreType</center></td>
#     <td><center>4 different stores:a,b,c,d </center></td>
# </tr>
# <tr>
#     <td><center>Assortment </center></td>
#     <td><center>Assortment level: a = basic, b = extra, c = extended</center></td>
# </tr>
# <tr>
#     <td><center>CompetitionDistance</center></td>
#     <td><center>Distance in meters to the nearest competitor store</center></td>
# </tr>
# <tr>
#     <td><center>CompetitionOpenSince[Month/Year]</center></td>
#     <td><center>gives the approximate year and month of the time the nearest competitor was opened</center></td>
# </tr>
# <tr>
#     <td><center>Promo</center></td>
#     <td><center>Promo or not on that day</center></td>
# </tr>
# <tr>
#     <td><center>Promo2</center></td>
#     <td><center>Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating</center></td>
# </tr>
# <tr>
#     <td><center>Promo2Since[Year/Week]</center></td>
#     <td><center>describes the year and calendar week when the store started participating in Promo2</center></td>
# </tr>
# <tr>
#     <td><center>PromoInterval</center></td>
#     <td><center>describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store</center></td>
# </tr>
# </table>

# **Approach:**

# Having strong interest for Tree-Based models; this notebook is meant to show the performance of a robust ensemble methods ; Random Forest (parallel tree creation)  and evaluate the best performing one for this case.
# 
# I first do the habitual data treatment and cleansing.
# 
# In order to understand better the patterns of the data, i will make use of libraries like matplotlib and seaborn to deep dive cases in the dataset and give better visibility on what is happening with the different types of Rossman drug stores.
# 
# This Exploratory analysis will help me move forward with the correlation analysis and feature engineering part of the project.
# 
# Then i train my model using scikit-learn with Random Forest Regressor and evaluate it on a validation set in order to analyse and understand which model works best for this scenario. 
# 
# 

# ------------------------------------------------------------------------------------------

# **Libraries to import:**

# In[2]:


import warnings
warnings.filterwarnings("ignore")
#Data Manipulation and Treatment
import numpy as np
import pandas as pd
from datetime import datetime
#Plotting and Visualizations
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats
import itertools
#Scikit-Learn for Modeling
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[3]:


def str_to_date(date):
    return datetime.strptime(date, '%Y-%m-%d').date()


# **Load the datasets:**

# In[4]:


#The training Set
df_train = pd.read_csv("../input/train.csv",sep=',', parse_dates=['Date']
                       , date_parser=str_to_date,
                       low_memory = False)


#Additional Information on those stores 
df_store = pd.read_csv("../input/store.csv"
                       , low_memory = False)


# **A quick glimpse at the data on hand:**

# In[5]:


df_train.head() 


# In[6]:


df_train.tail()


# In[7]:


df_train.dtypes,print ("The Train dataset has {} Rows and {} Variables".format(str(df_train.shape[0]),str(df_train.shape[1])))


# In[8]:


df_store.head()


# In[9]:


df_store.tail()


# In[10]:


df_store.dtypes ,print ("The Store dataset has {} Rows (which means unique Shops) and {} Variables".format(str(df_store.shape[0]),str(df_store.shape[1]))) 


# <h2>Data Wrangling and Exploration:</h2>

# How many missing fields each variable has:

# In[11]:


df_train.count(0)/df_train.shape[0] * 100


# - We can see that the the columns have got a good fill rate. We don't need to do any change for the train set for now at least.

# <h3>A closer look at the Train set:</h3>

# <h4>Deep Dive on Stores Closed (which means 0 customers and 0 sales) on Certain days:</h4>

# In[12]:


print ()
print ("-Over those two years, {} is the number of times that different stores closed on given days.".format(df_train[(df_train.Open == 0)].count()[0]))
print ()
print ("-From those closed events, {} times occured because there was a school holiday. " .format(df_train[(df_train.Open == 0) & (df_train.SchoolHoliday == 1)&(df_train.StateHoliday == '0') ].count()[0]))
print ()
print ("-And {} times it occured because of either a bank holiday or easter or christmas.".format(df_train[(df_train.Open == 0) &
         ((df_train.StateHoliday == 'a') |
          (df_train.StateHoliday == 'b') | 
          (df_train.StateHoliday == 'c'))].count()[0]))
print ()
print ("-But interestingly enough, {} times those shops closed on days for no apparent reason when no holiday was announced. In fact, those closings were done with no pattern whatsoever and in this case from 2013 to 2015 at almost any month and any day.".format(df_train[(df_train.Open == 0) &
         (df_train.StateHoliday == "0")
         &(df_train.SchoolHoliday == 0)].count()[0]))
print ()


# <h4>What should be done?</h4>

# - After reading the descrition of the this task, Rossman clearly stated that they were undergoing refurbishments sometimes and had to close. Most probably those were the times this event was happening.
# 
# - And since we don't want to bias our decision tree models to consider those exceptions, the best solution here is to get rid of closed stores and prevent the models to train on them and get false guidance.
# 
# - In this case we will analyse only open stores since a close  store yield a profit of 0.

# In[13]:


df_train=df_train.drop(df_train[(df_train.Open == 0) & (df_train.Sales == 0)].index)


# In[14]:


df_train = df_train.reset_index(drop=True) #making sure the indexes are back to [0,1,2,3 etc.] 


# In[15]:


print ("Our new training set has now {} rows ".format(df_train.shape[0]))


# **What about the distribution of Sales and Customers in the train set? Any outliers?**

# **1) Sales:**

# In[16]:


df_train.Sales.describe() 
#we see here a minimum of 0 which means some stores even opened got 0 sales on some days.


# In[17]:


df_train=df_train.drop(df_train[(df_train.Open == 1) & (df_train.Sales == 0)].index)
df_train = df_train.reset_index(drop=True) 


# In[18]:



fig, axes = plt.subplots(1, 2, figsize=(17,3.5))
axes[0].boxplot(df_train.Sales, showmeans=True,vert=False)
axes[0].set_xlim(0,max(df_train["Sales"]+1000))
axes[0].set_title('Boxplot For Sales Values')
axes[1].hist(df_train.Sales, cumulative=False, bins=20)
axes[1].set_title("Sales histogram")
axes[1].set_xlim((min(df_train.Sales), max(df_train.Sales)))

{"Mean":np.mean(df_train.Sales),"Median":np.median(df_train.Sales)}



# In[19]:


print ("{0:.2f}% of the time Rossman are actually having big sales day (considered outliers).".format(df_train[df_train.Sales>14000].count()[0]/df_train.shape[0]*100))
print ("{0:.2f}% of the time Rossman are actually having no sales at all.".format(df_train[df_train.Sales==0].count()[0]/df_train.shape[0]*100))


# **Findings:**

# - Some exceptions (the outliers) in the boxplot had to be checked to see if it's wrong inputted data but it turns out this big amount of sales on certain days is explained by either promotional purposes,the type of the store being big and popular or just not having near enough competition and being the monopoly in its region. (Charts will come in the analysis section of the train and store dataset when merged).
# 
# - Concerning the 0 of the time having 0 sales.it represented before removing them a tiny amount of the train set(0.01%), those values can affect further calculation of metrics and bias and are not to be taken into account. Those cases could happen for some shops, probably due to external events affecting it.( an incident, a manifestation etc.)
# 
# - An important metric to always check when looking at a distribution is how the mean compares to the median and how close are they from each other. As we see here a mean of 6955 versus 6369 in median is a very good sign that there are no extravagant values affecting the general distribution of Sales.

# In[20]:


df_train.Customers.describe()    


# In[21]:



fig, axes = plt.subplots(1, 2, figsize=(17,3.5))
axes[0].boxplot(df_train.Customers, showmeans=True,vert=False)
axes[0].set_xlim(0,max(df_train["Customers"]+100))
axes[0].set_title('Boxplot For Customer Values')
axes[1].hist(df_train.Customers, cumulative=False, bins=20)
axes[1].set_title("Customers histogram")
axes[1].set_xlim((min(df_train.Customers), max(df_train.Customers)))

{"Mean":np.mean(df_train.Customers),"Median":np.median(df_train.Customers)}


# In[22]:


print ("{0:.2f}% of the time Rossman are actually having customers more than usual (considered outliers).".format(df_train[df_train.Customers>1500].count()[0]/df_train.shape[0]*100))
print ("{0:.2f}% of the time Rossman are actually having no customers at all.".format(df_train[df_train.Customers==0].count()[0]/df_train.shape[0]*100))


# In[23]:


df_train[df_train.Customers>7000]


# In[24]:


stats.pearsonr(df_train.Customers, df_train.Sales)[0]


# **Findings:**

# - We can see similair patterns with the customers column and the Sales column, in fact our pearson correlation factor of 0.82 explains that there is a strong positive correlation between Sales and Customers. In general, the more customers you have in a store, the higher your sales for the day.
# 
# - We see that on a specific day there was a huge amount of customers in a store,this was due to a big promotion going on. Those specific values are affecting the mean which concludes the difference between a mean of 762 and a median of 676.
# 
# - We observe a right skewness in both distributions because of the low number of outliers but the high representation of each outlier alone which pushes the distribution to the lefta as seen in both histograms.This typically occurs when the mean is higher than the median.

# <h3>A closer look at the Store Dataset:</h3>

# In[25]:


df_store.count(0)/df_store.shape[0] * 100


# **Findings:**
# 
# - The `Promo2SinceWeek`,`Promo2SinceYear` and `PromoInterval` variables has 51% fill rate since they are actually NULL values because there are no continuous promotion for those stores. 
# 
# - Instead for `CompetitionOpenSinceMonth` and `CompetitionOpenSinceYear`, it's basically missing data that we're dealing with here (68.25% fill rate), this means that we have the nearest distance of the competitor but miss the date information on when did he actually opened next to the Rossman store.

# **Let's start the cleansing process by order:**

# 1) `CompetitionDistance`:

# In[26]:


df_store[pd.isnull(df_store.CompetitionDistance)] 
#rows with missing values for Competition Distance, only 3 rows with null which makes sense since 99.73% is filled


# - Before deciding how to treat this,we know there are infinite ways of filling missing values.
# - The most common and simplistic approach is to fill it with either the mean or the median of this variable.
# - Let's quickly have a look at those metrics.

# In[27]:


df_store_check_distribution=df_store.drop(df_store[pd.isnull(df_store.CompetitionDistance)].index)
fig, axes = plt.subplots(1, 2, figsize=(17,3.5))
axes[0].boxplot(df_store_check_distribution.CompetitionDistance, showmeans=True,vert=False,)
axes[0].set_xlim(0,max(df_store_check_distribution.CompetitionDistance+1000))
axes[0].set_title('Boxplot For Closest Competition')
axes[1].hist(df_store_check_distribution.CompetitionDistance, cumulative=False, bins=30)
axes[1].set_title("Closest Competition histogram")
axes[1].set_xlim((min(df_store_check_distribution.CompetitionDistance), max(df_store_check_distribution.CompetitionDistance)))
{"Mean":np.nanmean(df_store.CompetitionDistance),"Median":np.nanmedian(df_store.CompetitionDistance),"Standard Dev":np.nanstd(df_store.CompetitionDistance)}#That's what i thought, very different values, let's see why 


# We see a highly right skewed distribution for this variable with a significant difference between the mean and the median. This being caused by the amount of disperness in the data with a standard deviation of 7659, higher than the mean and the median.

# **Solution:**

# - It is realistically better to input the median value to the three Nan stores then the mean since the mean is biased by those outliers.

# In[28]:


df_store['CompetitionDistance'].fillna(df_store['CompetitionDistance'].median(), inplace = True)


# 2) `CompetitionOpenSinceMonth` and `CompetitionOpenSinceYear`?

# - Since we have no information whatsoever on those missing values and no accurate way of filling those values.
# - A creative way could be to apply a multilabel classification algorithm and train on the non Nan fields and then predict what could be most probably the month and year for those fields. But this approach is computationally too long.
# - So for this purpose those fields are going to be assigned to 0 .
# 

# In[29]:


df_store.CompetitionOpenSinceMonth.fillna(0, inplace = True)
df_store.CompetitionOpenSinceYear.fillna(0,inplace=True)


# 3) `Promo2SinceWeek`, `Promo2SinceYear` and `PromoInterval` ?

# In[30]:


#df_store[pd.isnull(df_store.Promo2SinceWeek)]
#df_store[pd.isnull(df_store.Promo2SinceWeek)& (df_store.Promo2==0)]


# **Findings:**
# - This case is pretty straighforward, all the missing values comes from fields where `Promo2`=0 which means there are no continuous promotional activities for those stores.
# - Having no promotion means those fields have to be 0 as well since they are linked to Promo2.

# In[31]:


df_store.Promo2SinceWeek.fillna(0,inplace=True)
df_store.Promo2SinceYear.fillna(0,inplace=True)
df_store.PromoInterval.fillna(0,inplace=True)


# In[32]:


df_store.count(0)/df_store.shape[0] * 100


# Now that we are done with clearing missing values, let's merge the two datasets.

# In[33]:


#Left-join the train to the store dataset since .Why?
#Because you want to make sure you have all events even if some of them don't have their store information ( which shouldn't happen)
df_train_store = pd.merge(df_train, df_store, how = 'left', on = 'Store')
df_train_store.head() 
print ("The Train_Store dataset has {} Rows and {} Variables".format(str(df_train_store.shape[0]),str(df_train_store.shape[1]))) 


# <h3>Store Type Analysis:</h3>

# The best way to asses the performance of a store type is to see what is the sales per customer so that we normalize everything and we get the store that makes its customers spend the most on average.
# 

# Let's compare first the total sales of each store type, its average sales and then see how it changes when we add the customers to the equation:

# In[34]:


df_train_store['SalesperCustomer']=df_train_store['Sales']/df_train_store['Customers']


# In[35]:


df_train_store.head()


# In[36]:


fig, axes = plt.subplots(2, 3,figsize=(17,10) )
palette = itertools.cycle(sns.color_palette(n_colors=4))
plt.subplots_adjust(hspace = 0.28)
#axes[1].df_train_store.groupby(by="StoreType").count().Store.plot(kind='bar')
axes[0,0].bar(df_store.groupby(by="StoreType").count().Store.index,df_store.groupby(by="StoreType").count().Store,color=[next(palette),next(palette),next(palette),next(palette)])
axes[0,0].set_title("Number of Stores per Store Type \n Fig 1.1")
axes[0,1].bar(df_train_store.groupby(by="StoreType").sum().Sales.index,df_train_store.groupby(by="StoreType").sum().Sales/1e9,color=[next(palette),next(palette),next(palette),next(palette)])
axes[0,1].set_title("Total Sales per Store Type (in Billions) \n Fig 1.2")
axes[0,2].bar(df_train_store.groupby(by="StoreType").sum().Customers.index,df_train_store.groupby(by="StoreType").sum().Customers/1e6,color=[next(palette),next(palette),next(palette),next(palette)])
axes[0,2].set_title("Total Number of Customers per Store Type (in Millions) \n Fig 1.3")
axes[1,0].bar(df_train_store.groupby(by="StoreType").sum().Customers.index,df_train_store.groupby(by="StoreType").Sales.mean(),color=[next(palette),next(palette),next(palette),next(palette)])
axes[1,0].set_title("Average Sales per Store Type \n Fig 1.4")
axes[1,1].bar(df_train_store.groupby(by="StoreType").sum().Customers.index,df_train_store.groupby(by="StoreType").Customers.mean(),color=[next(palette),next(palette),next(palette),next(palette)])
axes[1,1].set_title("Average Number of Customers per Store Type \n Fig 1.5")
axes[1,2].bar(df_train_store.groupby(by="StoreType").sum().Sales.index,df_train_store.groupby(by="StoreType").SalesperCustomer.mean(),color=[next(palette),next(palette),next(palette),next(palette)])
axes[1,2].set_title("Average Spending per Customer in each Store Type \n Fig 1.6")
plt.show()


# **Findings:**
# - From this training set we can see that Storetype A has the highest number of branches,sales and customers from the 4 different storetypes. But this doesn't mean it's the best performing Storetype.
# 
# - When looking at the average sales and number of customers, we see that actually it is Storetype B who was the highest average Sales and highest average Number of Customers. One assumption could be that if B has only 17 stores but such a high amount of average sales and customers that it is likely hyper Rossman branches whereas A would be smaller in size but much more present.
# 
# - Surprisingly it is StoreType D who has the highest average spending per Customer, this is probably explained by an average competition distance higher than ther rest which means each customer will buy more since he knows there isn't a lot of similair shops around.
# 
# - What would help us understand better what's happening is to look for other variables explaining this behaviour like Assortments, Comeptition and Promotions.

# <h3>Assortments:</h3>

# As we cited in the description, assortments have three types and each store has a defined type and assortment type: 
# - `a` means basic things
# - `b` means extra things
# - `c` means extended things so the highest variety of products.
# 
# What could be interesting is to see the relationship between a store type and its respective assortment type.

# In[37]:


StoretypeXAssortment = sns.countplot(x="StoreType",hue="Assortment",order=["a","b","c","d"], data=df_store,palette=sns.color_palette("Set2", n_colors=3)).set_title("Number of Different Assortments per Store Type")
df_store.groupby(by=["StoreType","Assortment"]).Assortment.count()


# **Findings:**
# - We can clearly see here that most of the stores have either `a` assortment type or `c` assortment type.
# - Interestingly enough StoreType d which has the highest Sales per customer average actually has mostly `c` assortment type, this is most probably the reason for having this high average in Sales per customer.Having variery in stores always increases the customers spending pattern.
# - Another important factor here is the fact that store type b is the only one who has the b assortment type and a lot of them actually which stands for "extra" and by looking at fig 1.4 and 1.5 he's the one who has the highest number of customers and sales. Probably this formula of extra is the right middlepoint for customers between not too much variety like C assortment and not too basic like A assortment and this is what is driving the high traffic in this store.

# **Promotion:**

# Let's see how Promotion affect the overall sales of Rossman by looking at when there is and when there isn't promotion over those 3 years. This allow us first to see the impact of promotion and as well to see the evolution of sales over specific years (so trends in a given year) and the gradual increase in sales from 2013 to 2015:

# In[38]:


df_train_store['Month']=df_train_store.Date.dt.month
df_train_store['Year']=df_train_store.Date.dt.year


# In[39]:



sns.factorplot(data = df_train_store, x ="Month", y = "Sales", 
               col = 'Promo', # per store type in cols
               hue = 'Promo2',
               row = "Year"
              ,sharex=False)


# In[40]:


sns.factorplot(data = df_train_store, x ="Month", y = "SalesperCustomer", 
               col = 'Promo', # per store type in cols
               hue = 'Promo2',
               row = "Year"
              ,sharex=False)


# **Findings:**
# - We see the dramatic change when we compare having promotion `Promo`=1 to not having promotion `Promo`=0 and can conclude that a store that have promotion on a given day changes its amount of sales considerably.
# - But Surprisingly, when we check more granularly at the `Promo2` variable (indicating a contunious promotion blue vs orange) we see that in general when there is no consecutive promotion stores tend to sell more then with consecutive promotion. This is probably a solution they're putting in place to treat stores with very low sales in the first place. And indeed when checking the Sales per Customer over promotion we understand that initially those stores suffer from low sales and those continuous promotion shows a tremending increase in the buying power of customers.
# - If we look over the years,there is a slight increase Year over Year but we don't see any major change from 2013 to 2015 and we actually see a very similair pattern in the months over the years with major spikes first around Easter period in March and April then in Summer in May,June and July and then finally around the Christmas period in November and December.

# What if we go more granular to look at days in a week for promotion impact?

# In[41]:


sns.factorplot(data = df_train_store, x ="DayOfWeek", y = "Sales",
                hue='Promo'
              ,sharex=False)


# In[42]:


#33 Stores are opened on Sundays
print ("Number of Stores opened on Sundays:{}" .format(df_train_store[(df_train_store.Open == 1) & (df_train_store.DayOfWeek == 7)]['Store'].unique().shape[0]))


# **Findings:**
# - We see already a big difference again even on a week level (from Monday to Friday) when we seperate promotion and no promotion.We see there are no promotions during the weekend.
# 
# - For Sunday to have such a high peak is understandable, since a very few stores opens on Sundays (only 33);if anyone needs anything urgently and don't have the time to get it during the week, he will have to do some distance to get to the open ones even if it's not close to his house. This means that those 33 open stores on Sunday actually accounts for the potential demand if all Rossman Stores were closed on Sundays. This clearly shows us how important it is for stores to be opened on Sundays.
# 
# - After attempting to look at Sales behaviour in a week over the Years and over the months, i concluded that the pattern doesn't change, which means all the time there is a peak on Mondays with promotions, a tiny peak on Friday before the weekend and a big peak on Sunday because of closed stores.

# **Competition Distance:**

# What i find also interesting to plot is the effect of the closest competition distance on Sales, to see whether the one with very far competition actually make more sales then the one with close competition.

# Since `CompetitionDistance` is a continuous variable, we need to first convert it into a categorical variable with 5 different bins (i chose this number by looking at the distribution and to keep aesthetic).

# In[43]:


df_train_store['CompetitionDist_Cat']=pd.cut(df_train_store['CompetitionDistance'], 5)


# In[44]:


df_train_store.head()


# In[45]:


df_train_store.groupby(by="CompetitionDist_Cat").Sales.mean(),df_train_store.groupby(by="CompetitionDist_Cat").Customers.mean()


# **Findings:**
# - As we can see here, like i thought, the stores that are the furthest have the highest average sales and number of customers.
# - This doesn't mean automatically that the furthest the better, but it does shed light on the fact that when there are no competition nearby, stores tend to sell more and have more customers because there are almost a monopoly in this region. We could think of it as McDonalds on highways where there are no other restaurants around, people who are hungry are forced to go there to eat.

# <h2>Preliminary Step for Correlation Analysis:</h2>

# Since we need numerical variables for both our correlation Analysis and to feed the decision tree based models, we need to transform what is not numerical to a numerical representation while keeping the logic behind it present.

# Since we used `CompetitionDist_Cat` just to show the variation of the variable `CompetitionDistance` we can go ahead and remove it now.

# In[46]:


del df_train_store["CompetitionDist_Cat"]


# Let's get Days from Date and delete Date since we already have its Year and Month:

# In[47]:


df_train_store['Day']=df_train_store.Date.dt.day


# In[48]:


del df_train_store["Date"]


# We still have StoreType,Assortment and StateHoliday as Obejcts we need to convert them to numerical categories:
# 
# But first we need to make sure we don't have Nan before doing those transformations otherwise Nan will be equal to -1

# In[49]:


df_train_store['StoreType'].isnull().any(),df_train_store['Assortment'].isnull().any(),df_train_store['StateHoliday'].isnull().any()
#No Null values we can proceed with the transformation


# In[50]:


df_train_store["StoreType"].value_counts(),df_train_store["Assortment"].value_counts(),df_train_store["StateHoliday"].value_counts()


# In[51]:


df_train_store['StateHoliday'] = df_train_store['StateHoliday'].astype('category')
df_train_store['Assortment'] = df_train_store['Assortment'].astype('category')
df_train_store['StoreType'] = df_train_store['StoreType'].astype('category')
df_train_store['PromoInterval']= df_train_store['PromoInterval'].astype('category')


# In[52]:


df_train_store['StateHoliday_cat'] = df_train_store['StateHoliday'].cat.codes
df_train_store['Assortment_cat'] = df_train_store['Assortment'].cat.codes
df_train_store['StoreType_cat'] = df_train_store['StoreType'].cat.codes
df_train_store['PromoInterval_cat'] = df_train_store['PromoInterval'].cat.codes


# In[53]:


df_train_store['StateHoliday_cat'] = df_train_store['StateHoliday_cat'].astype('float')
df_train_store['Assortment_cat'] = df_train_store['Assortment_cat'].astype('float')
df_train_store['StoreType_cat'] = df_train_store['StoreType_cat'].astype('float')
df_train_store['PromoInterval_cat'] = df_train_store['PromoInterval_cat'].astype('float')


# Since associating 0,1,2,3 to categorical variables like StoreType,Assortment,StateHoliday affect the bias of the algorithm (0 would account less then 3 when actually StoreType a and StoreType c should be treated equally).
# I will just convert it to categorical now for the purpose of the correlation analysis and then use the get_dummies function to encode them binarly.

# In[54]:



#df_train_store[['StateHoliday', 'StoreType', 'Assortment']] = df_train_store[['StateHoliday', 'StoreType', 'Assortment']].apply(lambda x: x.cat.codes)


# In[55]:


df_train_store.dtypes


# <h3>Correlation Analysis:</h3>

# In[56]:


df_correlation=df_train_store[['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo',
        'SchoolHoliday',
       'CompetitionDistance', 'CompetitionOpenSinceMonth',
       'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
       'Promo2SinceYear', 'SalesperCustomer', 'Month', 'Year',
       'Day', 'StateHoliday_cat', 'Assortment_cat', 'StoreType_cat',
       'PromoInterval_cat']]


# In[57]:


df_correlation=df_correlation.drop('Open', axis = 1)


# In[58]:


upper_triangle = np.zeros_like(df_correlation.corr(), dtype = np.bool)
upper_triangle[np.triu_indices_from(upper_triangle)] = True #make sure we don't show half of the other triangle
f, ax = plt.subplots(figsize = (15, 10))
sns.heatmap(df_correlation.corr(),ax=ax,mask=upper_triangle,annot=True, fmt='.2f',linewidths=0.5,cmap=sns.diverging_palette(10, 133, as_cmap=True))


# **Interpretation:**
# - We can first see the 0.82 between Customers and sales which suggests that they are positively correlated like we stated above in the analysis.
# - It's interesting to see that Sales per Customer and Promo (0.28) actually correlate positively, since running a promotion increases that number .
# - Sales per Customer also correlates with Competition Distance(0.21), in a positive manner, like i said up the higher the competitionn distance the more sales per customer we do, which makes sense , the further our competition, the more monopolization Rossman can achieve in the region.
# - Additionally, the effect of promo2 to Sales per Customer like we said above as well(0.22), it  did provoke a change in the buying pattern and increased it when continuous promotion were applied.
# - Finally, we can see that StoreType does play a major role with Sales per Customer (0.44), this is probably due to my encoding of the store type variable which suggests that the high categories like d which is equal 4 has higher weight, but if not then it makes sense that the last categories like d does explain the increase in Sales per Customer like fig 1.6 shows.

# <h3>Conclusion of Exploratory Analysis:</h3>

# At this stage, we got a solid understanding of the distributions, the statistical properties and the relationships of our variables.
# The next step is to identify what variables to feed XGboost and Random Forest for training and to work on the modeling part of the project

# <h2>Training with RandomForest Regressor:</h2>

# **Definition:**
# 
# - RandomForest is a machine learning alogrithm used for classification and regression that is bestly used with structured and tabular data.

# **Its advantages:**
#     
# - Random forest runtimes are quite fast, and they are able to deal with unbalanced and missing data.
# - The process of averaging or combining the results of different decision trees helps to overcome the problem of overfitting.
# - They also do not require preparation of the input data. You do not have to scale the data.
# 
# **Its Disadvantages:**
# - The main drawback of Random Forests is the model size. You could easily end up with a forest that takes hundreds of megabytes of memory and is slow to evaluate.
# - They get a bit harder to interpret than regular deicison trees, since we are constructing of forest of more than 50 decision trees and more using grid search.
# 

# **Further Feature Engineering before Training:**

# -Since the competition variables `CompetitionOpenSinceYear` and `CompeitionOpenSinceMonth` have the same underlying meaning, merging them into one variable that we call `CompetitionOpenSince` makes easier for the algorithm to understand the pattern and creates less branches and thus complex trees.

# In[59]:


df_train_store.columns


# In[60]:


df_train_store['CompetitionOpenSince'] = np.where((df_train_store['CompetitionOpenSinceMonth']==0) & (df_train_store['CompetitionOpenSinceYear']==0) , 0,(df_train_store.Month - df_train_store.CompetitionOpenSinceMonth) + 
                                       (12 * (df_train_store.Year - df_train_store.CompetitionOpenSinceYear)) )


# In[61]:


#now that CompetitionOpenSince is created 
#we can get rid of `CompetitionOpenSinceYear` and `CompeitionOpenSinceMonth`
del df_train_store['CompetitionOpenSinceYear']
del df_train_store['CompetitionOpenSinceMonth']


# -The `StateHoliday` is not very important to distinguish (what type of holiday) and can be merged in a binary variable called `is_holiday_state`.

# In[62]:


df_train_store["is_holiday_state"] = df_train_store['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})


# In[63]:


del df_train_store['StateHoliday_cat']


# -I think it's always better when working with decision tree based models to have dummy variables instead of categorical with different levels, because this alters the bias of the algorithm who will favor a higher weight to the categories like 4 and deprioritize levels like 1. And this problem could rise in the variables `Assortment` and `StoreType` and `PromoInterval`.
# So far those are the codes we have for each variable:
# 
# -That's why i use the get_dummies function to instead do a binary encoding and prevent this.

# In[64]:


df_train_store=pd.get_dummies(df_train_store, columns=["Assortment", "StoreType","PromoInterval"], prefix=["is_Assortment", "is_StoreType","is_PromoInteval"])


# In[65]:


del df_train_store['Assortment_cat']
del df_train_store['StoreType_cat']


# In[66]:


del df_train_store['PromoInterval_cat']


# In[67]:


df_train_store.columns


# - The Train set is ready to be fed to the RandomForest Algorithm.
# - Let's do the same transformation for the test set now.

# <h3>Test Set Adaptation:</h3>

# In[68]:


df_test = pd.read_csv("../input/test.csv",sep=',', parse_dates=['Date']
                       , date_parser=str_to_date,
                       low_memory = False)
print ("The Test dataset has {} Rows and {} Variables".format(str(df_test.shape[0]),str(df_test.shape[1])))


# In[69]:


df_test.fillna(1, inplace = True) #11rows with Nans decided to leave them open since its one store 622 which is 
#usually open
#Left-join the train to the store dataset since .Why?
#Because you want to make sure you have all events even if some of them don't have their store information ( which shouldn't happen)
df_test_store = pd.merge(df_test, df_store, how = 'left', on = 'Store')
print ("The Test_Store dataset has {} Rows and {} Variables".format(str(df_test_store.shape[0]),str(df_test_store.shape[1]))) 
df_test_store['Month']=df_test_store.Date.dt.month
df_test_store['Year']=df_test_store.Date.dt.year
df_test_store['Day']=df_test_store.Date.dt.day

df_test_store['StateHoliday'] = df_test_store['StateHoliday'].astype('category')
df_test_store['Assortment'] = df_test_store['Assortment'].astype('category')
df_test_store['StoreType'] = df_test_store['StoreType'].astype('category')
df_test_store['PromoInterval']= df_test_store['PromoInterval'].astype('category')
df_test_store['StateHoliday_cat'] = df_test_store['StateHoliday'].cat.codes
df_test_store['Assortment_cat'] = df_test_store['Assortment'].cat.codes
df_test_store['StoreType_cat'] = df_test_store['StoreType'].cat.codes
df_test_store['PromoInterval_cat'] = df_test_store['PromoInterval'].cat.codes
df_test_store['StateHoliday_cat'] = df_test_store['StateHoliday_cat'].astype('float')
df_test_store['Assortment_cat'] = df_test_store['Assortment_cat'].astype('float')
df_test_store['StoreType_cat'] = df_test_store['StoreType_cat'].astype('float')
df_test_store['PromoInterval_cat'] = df_test_store['PromoInterval_cat'].astype('float')
df_test_store['CompetitionOpenSince'] = np.where((df_test_store['CompetitionOpenSinceMonth']==0) & (df_test_store['CompetitionOpenSinceYear']==0) , 0,(df_test_store.Month - df_test_store.CompetitionOpenSinceMonth) + 
                                       (12 * (df_test_store.Year - df_test_store.CompetitionOpenSinceYear)) )



df_test_store["is_holiday_state"] = df_test_store['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})




df_test_store=pd.get_dummies(df_test_store, columns=["Assortment", "StoreType","PromoInterval"], prefix=["is_Assortment", "is_StoreType","is_PromoInteval"])


# In[70]:


del df_test_store["Date"]
del df_test_store['CompetitionOpenSinceYear']
del df_test_store['CompetitionOpenSinceMonth']


# In[71]:


del df_test_store['StateHoliday_cat']


# In[72]:


del df_test_store['Assortment_cat']
del df_test_store['StoreType_cat']
del df_test_store['PromoInterval_cat']


# In[73]:


del df_test_store['StateHoliday']


# In[74]:


del df_train_store['StateHoliday']


# -Now that my Test Set is ready i can go back and proceed with the training phase

# ----
# 
# ## Developing The Model
# In this section of the project, I will develop a randomforestregressor training and fitting using GridSearch for hyperparameter optimization to make a prediction.
# Then ,I will make accurate evaluations of each model's performance (RandomForestRegressor vs XGBoostRegressor) through the use of the sklearn library which will help me evaluate which model is more suitable.

# ### Developing The Model: Define a Performance Metric
# It is difficult to measure the quality of a given model without quantifying its performance over training and testing. This is typically done using some type of performance metric. For this project, I will use the rmspe(Root Mean Square Percentage Error) score provided as an evaluation metric for the competition. You can find the formula inserted in a function down below.If the result is lower then 10% this signifies a very good quality of predicition and would be our goal in this project.
# 

# In[75]:


def rmspe(y, yhat):
    rmspe = np.sqrt(np.mean( (y - yhat)**2 ))
    return rmspe


# In[76]:


features = df_train_store.drop(['Customers', 'Sales', 'SalesperCustomer'], axis = 1) 
#a rule of thumb is to transform my target value to log if i see the values are very dispersed which is the case
#and then of course revert them with np.exp to their real values
targets=np.log(df_train_store.Sales)


# ### Developing The Model: Shuffle and Split Data with Cross Validation, Grid Search and Fitting the model:

# A crucial Step in Machine Learning is to make sure your model is robust by testing it on a small part of your dataset we call here train_test set which is usually divided 80% training and 20% validation.

# In[77]:


X_train, X_train_test, y_train, y_train_test = model_selection.train_test_split(features, targets, test_size=0.20, random_state=15)
print ("Training and testing split was successful.")


# Instead of doing a function, i decided to loop over different combination of hyperparameters and see what would be the optimal combination and at least over the loop i could be monitor how is the training performing.
# 
# But First let's setup the RandomForestRegressor object.

# In[78]:


rfr = RandomForestRegressor(n_estimators=10, 
                             criterion='mse', 
                             max_depth=5, 
                             min_samples_split=2, 
                             min_samples_leaf=1, 
                             min_weight_fraction_leaf=0.0, 
                             max_features='auto', 
                             max_leaf_nodes=None, 
                             min_impurity_decrease=0.0, 
                             min_impurity_split=None, 
                             bootstrap=True, 
                             oob_score=False,
                             n_jobs=4,
                             random_state=31, 
                             verbose=0, 
                             warm_start=False)
rfr.fit(X_train, y_train)


# The choice of the combinations of my hyperparameters are based on my past experience with Machine learning projects i worked on,most of the time the number of estimators should not exceed the 100 trees since the training set is big enough and the computational ressources needed are very big and in this case we're only using a local computer with 16GB of Ram. 
# As of the min_samples_split being the minimum number of samples required to split an internal node, i am giving it full freedom to not limit itself only requiring minimum 2 but making it more conservative but trying 10 for instance which could alter the default way it splits its nodes in each tree.

# In[79]:



'''
params = {'max_depth':(4,6,8,10,12,14,16,20),
         'n_estimators':(4,8,16,24,48,72,96,128),
         'min_samples_split':(2,4,6,8,10)}
#scoring_fnc = metrics.make_scorer(rmspe)
#the dimensionality is high, the number of combinations we have to search is enormous, using RandomizedSearchCV 
# is a better option then GridSearchCV
grid = model_selection.RandomizedSearchCV(estimator=rfr,param_distributions=params,cv=10) 
#choosing 10 K-Folds makes sure i went through all of the data and didn't miss any pattern.(takes time to run but is worth doing it)
grid.fit(X_train, y_train)
'''
#I AM NOT GOING TO RUN THIS CHUNK TO BE ABLE TO COMMIT AND RUN MY KERNEL ON KAGGLE


# In[80]:


#This is the best combination i got from what i propose to try out with a (mse) score of 0.855 which is quite good
#grid.best_params_,grid.best_score_
#MY BEST PARAMS ARE :n_estimators=128,max_depth=20,min_samples_split=10


# ## Test our RF on the validation set:

# In[81]:


#with the optimal parameters i got let's see how it behaves with the validation set
rfr_val=RandomForestRegressor(n_estimators=128, 
                             criterion='mse', 
                             max_depth=20, 
                             min_samples_split=10, 
                             min_samples_leaf=1, 
                             min_weight_fraction_leaf=0.0, 
                             max_features='auto', 
                             max_leaf_nodes=None, 
                             min_impurity_decrease=0.0, 
                             min_impurity_split=None, 
                             bootstrap=True, 
                             oob_score=False,
                             n_jobs=4, #setting n_jobs to 4 makes sure you're using the full potential of the machine you're running the training on
                             random_state=35, 
                             verbose=0, 
                             warm_start=False)
model_RF_test=rfr_val.fit(X_train,y_train)


# In[82]:


yhat=model_RF_test.predict(X_train_test)


# In[83]:


plt.hist(yhat)


# In[84]:


error=rmspe(y_train_test,yhat)
error


# Well 0.16 on RMSPE is a very good measure that usually indicates the quality of my predictions are very good.
# With the right amoung of comupational ressources and a bit more time i could manage to lower this score to reach the 0.10

# **Results:**

# At this stage with an RMSPE of 0.16 we're quite accurate when it comes to testing our validation set.
# A lot of approaches can be taken here in order to improve this score:
# 
# -Either allocating more training sample when splitting so that the algorithm has more data to train on.
# 
# -Proposing more variation in the hyperparameter to reach a better optimal combination.
# 
# -Use another approach then RandomForest like polynomial linear regression,XGboost,Neural Network etc.

# **In order to understand better what happened when we ran our randomforest regressor, here is a chart that represents, the importance and role that each variable that i decided to include played in this learning process:**

# In[85]:


importances = rfr_val.feature_importances_
std = np.std([rfr_val.feature_importances_ for tree in rfr_val.estimators_],
             axis=0)
indices = np.argsort(importances)
palette1 = itertools.cycle(sns.color_palette())
# Store the feature ranking
features_ranked=[]
for f in range(X_train.shape[1]):
    features_ranked.append(X_train.columns[indices[f]])
# Plot the feature importances of the forest

plt.figure(figsize=(10,15))
plt.title("Feature importances")
plt.barh(range(X_train.shape[1]), importances[indices],
            color=[next(palette1)], align="center")
plt.yticks(range(X_train.shape[1]), features_ranked)
plt.ylabel('Features')
plt.ylim([-1, X_train.shape[1]])
plt.show()


# **Findings:**
# - Our top 5 most important variables are:
# 
#  1-Competitor Distance: This indeed impacts a lot the sales of a store like we saw previously in our EDA,when competition is very far stores tend to sell a lot more.
#  
#  2-Promo: Promotion is primordial for a store to increase its sales, it allows price breaking and thus more customers intersted in buying.
#  
#  3-Store: The Store itself represents a unique identificator for the algorithm to recognise which store has what attributes and indeed better accounts for the forecasting of those same stores in a future timeline.
#  
#  4-CompetitionOpenSince: The merging of this variable paid out and allowed us to give more accurate predicitions of the sales based on the time of opening of those competitors.
#  
#  5-DayofWeek: Like we said, during a week , the pattern varies a lot if it's a sunday or a monday (like we saw in our EDA) for instance and each day in the week has his own attributes and properties that allow to know how much are we going to sell.
# 

# <h3>Kaggle Submission:</h3>

# In[86]:


df_test_store1=df_test_store.drop(['Id'],axis=1)
kaggle_yhat= model_RF_test.predict(df_test_store1)

kaggle_preds= pd.DataFrame({'Id': df_test_store['Id'], 
                          'Sales': np.exp(kaggle_yhat)})
kaggle_preds.to_csv("Stefano_Zakher_RF_Rossman_Kaggle_submission.csv", index = False)


# <h2>Conclusion of this project:</h2>
# -  We can understand from this project the flexibility and robustness of a decision tree based model like RandomForest which helped us predict the Store Sales of Rossman based on attributes that defines each store and its surroundings.
# - As we can see, it always delivers a good predicition score while not having a lot of modifications and difficulties capturing the patterns hidden in the data. Fortunately we had a train set that was large enough for it to converge but in general RandomForest performs not so bad on small sets since its resampling method (bagging) and its random production of trees allow the bias to remain not so high and in this case always performs good on unseen data where as XGboost has tendency to overfit if not gently and smartly tuned.
# - I believe using hyperparameter optimization techniques like Gridsearch and RandomizedSearch is crucial to any Machine Learning problem since it allows the algorithm to not just limit itself on its defaulted parameters but to discover new opportunities of combining those parameters to reach a local optima while training on the data.

# Thank you again for taking the time to go through my work! i hope you enjoyed!
# 
# For the full version please check my 
# 
# GitHub:
# https://github.com/stefanozakher/Kaggle_Rossman_Sales_Predicitions/blob/master/Rossman%20Sales%20Prediciton%20_%20Stefano%20Zakher.ipynb

# Please leave an upvote if you think this was benefitial for your understanding of the problem and follow me for more exciting projects upcoming!
