#!/usr/bin/env python
# coding: utf-8

# <h1>  Exploratory Data Analyses of Kickstarter dataset using Tableau visualizations</h1> <br>

# <h1>Fund raisers for start ups from across the globe </h1>
# 
# 
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/World.png)
# <br>
# It can be seen that a majority of the projects seeking funds in Kickstarter are from the United states with highest success rate in successfully raising funds.
# 
# Map credits : OpenStreetMap Contributors (Tableau)

# <h1>Category-wise distribution</h1>
# ![Category-wise distribution](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/CategoryWiseCampaigns.png) <br>
# 
# Looks like a vast majority of the fund seekers want to make ***Film and Videos***
#  
# Lets look at the top 50 sub categories 
#  
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/Top50SubCategories.png) 
# 
# Product design tops the list, followed by Documentary, Music, Tabletop Games etc.,

# <h1>Sum of the funds recieved from the backers by category with sub category that are successful</h1>
# 
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/SumCategoryAndSubCategorywise.png)
# 
# It's a beautiful pattern in which Product design appears stand out followed by Tabletop Games, Video games, Documentary etc., Technology comes after all this.
# <br>This image might be a little too biased towards Product Design and games where one or two companies could get a larger portion of the funds while rest gets lesser support.
# <br>That's is because we are looking at the ***SUM***. Now lets look at the ***Average***
# 
# <h1>Average funds recieved from the backers by category and sub category that are successful</h1>
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/AvgCatSubCat.png)
# <br>
# This appears to be a better picture. Now, the average funds received by each technology startup seems to consistent across various sub categies. <br>
# Please note that the Product design comes far later when the average funds are considered. 
# <br>
# <br>
# Now, lets look at the average funds by main category.
# 

# <h1> Average Funds raised by Main-category </h1>
# 
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/AvgFundsByCategory.png)
# 
# Although the number of Technology based start ups are about half the number of Film & Videos category , the average pledged USD seems to be highest for the Technology Kickstarters.
# <br>
# Technology, Design and Games appear to be the top trending main categories that are raising good funds.
# <br>
# Now lets take a closer look at each of these main categories.
# 
# <br>
# <h1> Top 5 sub categories by number of records in Technology, Design and Games startups
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/Top5inTDG.png)

# <h1> Fund Raising statuses in Percentage of total Fund seekers </h1>
# <img src="https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/TotalShare.png" alt="Total share of the campaign statuses" width="800" height="800"> 
# 
# It seems that* **a little more than a third*** of the Kickstarter Projects are successful in the campaign 

# <h1> Average length of campaigns that have seen different fates </h1>
# 
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/Average+days+bn.png)
# 
# It apprears that, on an average a campaign that has been successful has about ***32.16 days*** to achieve the feat

# 

# <h1> Top 15 Fund raisers in the campaign</h1>
# 
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/Top15.png)
# 
# <br>
# It can be seen that the Product design category is attracting most funds with 7 companies in the top 15 being from Product Design and 3 in top 5 belonging to Product design space.

# <h1>Top 50 projects by highest backers</h1>
# 
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/ProjectsBySumBackers.png)
# 
# <br>
# It can be observed that the Games category domnates here followed by Design, Tech and only a few Film & Video entries.

# <h1>Average days to reach Average goal funding - Category-wise</h1>
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/DaysBnVsAvgGoal.png)
# 
# On an average crafts related start ups seems to reach the average Goal faster than any other category. <br>
# However, the average USD required by this community is also less than all other categories.<br>
# Start ups in Gaming space seems to be reaching the average goal faster than most other categories.

# 
# <h1>Category-wise Success/Failure</h1>
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/CategorywiseSuccessFailurePercentage.png)
# 
# <br>
# 
# It can be seen that a major portion of Kick starters seeking funding from ***Music, Theater, Comics and Dance*** categories are successful in the campaign. <br> <br>
# In the next picture it looks like the fund seekers from Theater, Comics and Dance have surpassed goal funding. <br><br>
# 
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/PledgedMorethanGoal.png)

# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/CurrencyVSAvgPledged.png)
# 
# <br>
# It looks like the Swiss Franc (CHF) is leading the inventment in Kickstarter Projects. (Please note all the currencies are USD equivalents)

# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/daysDiffDist.png)
# 
# This looks like a clear sky. Isn't it ? <br>
# This plot hardly says anyhting. <br>
# Well, wrong. This graph captures the most critical information in data. ***OUTLIERS*** <br>
# 
# It looks like there is at least one Kickstarter Project who's campaign has extended over 16,000 days.  <br>
# Few outperformers pull the sample mean and standard deviation towards themselves.<br>
# Let's exclude the outliers and look at the distribution.
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/DaysDiffDistribution.png)
# 
# 
# <br>
# <br>
# There are spikes in the histogram at the multiples of weeks (7 days, 14 days, 21 days 28 days ) or in multiples of 5 days (5 days, 10 days, 15 days etc.,) or in multiples of months (30 days, 31 days, 60 days, 90 days).<br>
# This suggests that a large portion of campaigns have run in terms of weeks and months and multilples of 5 days.
# <br>

# <h1> Average funds raised per backer in different categories </h1>
# 
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/AvgUSDperBacker.png)
# 
# Technology seems to lead again.

# <h1>Average Backers v Average Goal for Successful/failed/cancelled campaigns</h1>
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/BackerAndAvg+Goal.png)
# 
# <br>
# This suggests that majority of the backers look for investing in projects that have smaller average USD as a goal.<br> 
# May be backers think Projects with lower goal are less riskier. <br>
# On an average these projects also tend to be successful in campaigns.

# <h3> Lets have a look at the Box plots for categories to get better idea of the funds raised by each category.</h3>
# 
# <h1> Box plots funds raised - category-wise </h1>
# 
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/BoxSmall.png)
# 
# <br>
# Well, the plot is supposed to be box plot. But there are a big number of outliers making the box look like a tiny line. <br>
# So, box plot is probably not a good idea when the data is huge and there are way many outliers. <br>
# Let's look closer by magnifying  the y axis (funds raised)
# 
# <h1>  Box plots funds raised - category-wise (magnified) </h1>
# 
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/BoxBig.png)
# 
# Design, Games and Technology have relatively larger IQRs while Crafts, Journalism, Art and Photography have smaller IQRs. <br>
# This is not much helpful to get any specific information. However, a general idea about the distribution can be infered.

# Let's look at the Success and failure rates category-wise
# 
# <h1> Success rate of projects in different category-wise </h1>
# 
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/percSuccessCat.png)
# 
# About Two thirds (65.44%) of the Dance projects seeking funds  are succeessful in raising funds while only about a quarter (23.79%) of technology are successful in raising funds.

# <h2> Now let's see the trend of Fund seekers </h2>

# <h1> Distribution of kickstarter fund seekers year-wise  </h1>
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/Year_DistPerc.png)
# 
# <br>
# Well, this uncovers a lot of trends. Some of the prominent observations are as below. <br>
# 1. 2014 and 2015 has seen a good surge in the number of Startups that seek funding from Kickstarter. 
# 2. For some reason there has been a steep decline in 2016 from 2015.
# 3. The number of startups as a percent of the number seeking funds in particular year is steadily increasing for design and fashion categories while increasing at a much higher rate for Technology and gaming spaces. However, tech has seen a small dip in the last couple of years.
# 4. Similarly, such a trend seen to be declining in Film & Videos categories and Music.
# 
# 
# <h3> Let's see the funding success rate of these categories </h3>

# <h1> Success rate of the Project categories, Year-wise </h1>
# 
# ![](https://s3.us-east-2.amazonaws.com/kaggleimages/Kickstarter/yearSuccessPercentage.png)
# <br>
# Again, Music and Film & videos are seeing a decline in being successful in campaign, while technology, Games and Design are increasingly successful in raising funds over years.
# 
# This could possibly explain the increased interest in the backers for Games, Technology and Design based startups

# In[ ]:


df=pd.read_csv('../input/ks-projects-201801.csv')


# In[ ]:





# In[ ]:


df.launched=df.launched.str.split(' ',expand=True)[0]
df.launched=df.launched.astype('datetime64[ns]')
df.deadline=df.deadline.astype('datetime64[ns]')


# In[ ]:


df['days'] = (df.deadline-df.launched).dt.days


# In[ ]:


fs = 15
plt.figure(figsize=(10,10))
df.days.hist(bins=30,xlabelsize=fs)
plt.xlabel('Number of days between Campaign start date and deadline',fontsize=fs)
plt.ylabel('Number of instances ',fontsize=fs)


# In[ ]:




