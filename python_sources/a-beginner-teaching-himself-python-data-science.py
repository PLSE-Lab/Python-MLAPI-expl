#!/usr/bin/env python
# coding: utf-8

# Hi I'm Don! While I do have a Computer Science degree, this will be my first time working on anything related to programming, data science, etc. in over 7 years, and also my first time ever using Python. I will be investigating this dataset containing reviews of games released in 2011-2019 scraped from metacritic.com by Andrea Cadeddu at https://www.kaggle.com/skateddu/metacritic-games-stats-20112019. 
# 
# I have given myself 10 practice sub-projects to look at the data in ways I think could be interesting (ordered by estimated difficulty to implement). Progress rating at the end of each sub-project: 0/5 indicates no progress beyond ideation, 1/5 indicates a cell has been created, 2/5 indicates some working code has been run, 3/5 indicates presentable data, 4/5 indicates some polish, 5/5 indicates completion:
# 
# 1. create a visualization showing the percent of total positive/neutral/negative critic scores and user scores (5/5)
# 2. create a visualization showing number of games in each genre (4/5)
# 3. create a visualization of the correlation between total critic reviews vs. metascore (3/5)
# 4. create a visualization of both metascore and user score vs. "popularity", i.e., total user reviews (4/5)
# 5. analyze any correlation between rating vs. both metascore and user score (3/5)
# 6. create a visualization showing the number of critic/user reviews by year, look at the breakdown by positive/neutral/negative (3/5)
# 7. create a visualization showing the market share of each platform per year, really curious how PC vs. everything else has fared, and also what the life cycle of a console looks like- how many games are released in the 1st, 2nd, 3rd, etc. year for consoles and how well-received are those games? (1/5)
# 8. analyze any games that critics hated but users loved, and vice versa (3/5)
# 9. determine correlation between metascore/user_score across genres, dates, and popularity buckets, create visualization of any interesting findings. (2/5)
# 10. analyze polarizing games (lots of positive/negative with little neutral) vs. non-polarizing games for both critics/users. Investigate correlation with popularity and year (2/5)
# 11. Stretch goals: 11A) web scrape Steam data, 11B) compare Steam reviews to metacritic reviews, 11C) compare metacritic reviews of games not on Steam to metacritic reviews of games on Steam
# 
# Overall progress estimate: 80% complete

# In[ ]:


#import required packages
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import plotly
import numpy as np
import plotly.graph_objs as go
import seaborn as sns
import pandas as pd
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

#read the data file
df1 = pd.read_csv('../input/metacritic_games.csv', delimiter=',')

#add total critics and total users
df1['total_critics'] = df1['positive_critics'] + df1['neutral_critics'] + df1['negative_critics'] 
df1['total_users'] = df1['positive_users'] + df1['neutral_users'] + df1['negative_users']

#make release date a datetime
df1['release_date'] = pd.to_datetime(df1['release_date'])

#quick checkup that the data looks good
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns. Below is a sample of 5 random rows.')
df1.sample(5)


# A few quick notes about the data. Many games are on multiple platforms, each with their own data entry (e.g., Stardew Valley is on PC, PS4, Switch, and XONE, and thus occupies 4 different rows). The counts of user reviews (e.g., the positive_users columns) refer to how many written reviews there are, not the number of ratings which the user_score is based off of. Unfortunately, the data does not contain the number of user ratings. Also for some reason metacritic ranges critic scores from 0 to 100 and user scores from 0 to 10- thus the user score is multiplied by 10 in this data. 
# 
# Finally, for qualitative analysis, it's important to keep in mind that metacritic is a much smaller review site than Steam. Of course, Steam only contains reviews for games on its own platform. I'd be very interested in comparing metacritic reviews for games that are on Steam vs. not on Steam, but that is a stretch goal as it would require web scraping Steam reviews. At any rate, the goal of this project is for me to learn some basic Python and data science using this simple dataset rather than form robust conclusions about the gaming industry :)

# In[ ]:


#SUB-PROJECT 1: create a visualization showing the total percentage of positive/neutral/negative critic scores and user scores

#get the percentage of each review type
percent_positive_reviews = [df1['positive_critics'].sum()/df1['total_critics'].sum()*100, df1['positive_users'].sum()/df1['total_users'].sum()*100]
percent_neutral_reviews = [df1['neutral_critics'].sum()/df1['total_critics'].sum()*100, df1['neutral_users'].sum()/df1['total_users'].sum()*100]
percent_negative_reviews = [df1['negative_critics'].sum()/df1['total_critics'].sum()*100, df1['negative_users'].sum()/df1['total_users'].sum()*100]

#display data
x_shape = [0,1]
x_names = ('Critics','Users')
plt.bar(x_shape, percent_positive_reviews, color='g', edgecolor='white', label = 'Postive reviews')
plt.bar(x_shape, percent_neutral_reviews, bottom=percent_positive_reviews, color='y', edgecolor='white', label = 'Neutral reviews')
plt.bar(x_shape, percent_negative_reviews, bottom=np.add(percent_positive_reviews, percent_neutral_reviews), color='r', edgecolor='white', label = 'Negative reviews')
plt.xticks(x_shape, x_names)
plt.legend(loc='right', bbox_to_anchor=(1.45, 0.5))
plt.show()


# The percentage of positive reviews is actually pretty close. For bad but not super bad experiences, i.e., roughly the bottom quartile excluding the bottom 5%, critics are more likely to be neutral whereas users are negative. Makes sense- as an gamer, if I'm investing money and time in a bottom quartile gaming experience, I'm generally not going to be happy. A reviewer may be more forgiving- additionally, I also suspect reviewers have some incentive to avoid negative reviews.

# In[ ]:


#SUB-PROJECT 2: create a visualization showing number of games in each genre and platform
#Also decide if this looks better with the filter at over 300 or over 100 for the "Other" category in the Genre Breakdown

#count occurences of each genre and platform
genre_series = df1['genre'].value_counts().sort_values(ascending=True)
platform_series = df1['platform'].value_counts().sort_values(ascending=True)

#filter out genres with less than 300 occurences and group them in an "Other category" ("Misc" has 282)
large_genres = genre_series[genre_series > 300] 
other_count = 0
for x in genre_series:
    if x < 300:
        other_count = other_count + x
large_genres.set_value('Other', other_count)

#display data
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.pie(large_genres.values, labels=large_genres.index, autopct='%1.0f%%')
ax1.set_title("Genre Breakdown")
ax2.pie(platform_series.values, labels=platform_series.index, autopct='%1.0f%%')
ax2.set_title("Platform Breakdown")
plt.show()


# Top genres in "Other" include Sports, Racing, Simulation, and Puzzle, as well as games simply listed as Miscellaneous. The platform breakdown is quite interesting/surprising to me, although it doesn't answer many questions on its own- would be interested in doing further research (for example, why does Nintendo never release its flagship games such as Mario on PC?). Nintendo is represented through 3 different consoles between Switch, 3DS, and WIIU, while Sony is represented through both PS4 and VITA. 

# In[ ]:


#SUB-PROJECT 3: create a visualization of the correlation between total critic reviews vs metascore

#display data
df1.plot(kind='scatter', x='total_critics', y='metascore')
plt.show()


# A positive correlation here is expected, of course. Even so, it is interesting to see the extent that having a number of critic reviews past a certain threshold (i.e., AAA games) guarantees a score higher than some threshold. I will look into learning how to quantify that extent more than simply eyeballing it on a scatter plot soon! But for now, we can see on the graph that games with 60+ critics all have metascores of 50+, and games with 100+ critics all have metascores of 70+. 

# In[ ]:


#SUB-PROJECT 4: create a visualization of game score vs. "popularity", i.e., user reviews. Need to study additional data to determine how sales/active players correlate to user reviews

#manually marking notable games on the graph to appear red
notable_points_df = pd.DataFrame(df1[(df1['game'] == 'Diablo III') | (df1['game'] == 'Infestation: Survivor Stories (The War Z)') | (df1['game'] == 'Star Wars Battlefront II')])

#display data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
df1.plot(kind='scatter', x='total_users', y='user_score', ax = ax1)
notable_points_df.plot(kind='scatter', x='total_users', y='user_score', ax = ax1, color = 'r')
df1.plot(kind='scatter', x='total_users', y='metascore', ax = ax2)
notable_points_df.plot(kind='scatter', x='total_users', y='metascore', ax = ax2, color = 'r')

plt.show()


# Some data points that drew my attention on the graph (which I went back and marked in red):
# 1. Diablo III, with 4682 user reviews. 
# 2. Infestation: Survivor Stories with 1601 user reviews but a metascore of only 20. 
# 3. The only game with over 1000 reviews and a lower user score than I:SS's 17 is Star Wars Battlefront II on the PS4 with 2331 users and a user score of 11. It also has an entry for the PC and XONE versions.
# 
# All 3 of these games were "review bombed" to protest their inclusion of ways to gain gameplay-impacting advantages with real money purchases, i.e., microtransactions. 
# 1. Top ranked players in Diablo II often bought and sold items through eBay; thus, its successor, Diablo III included a real money auction house within the game. It was removed after community backlash with little long-term damage done, as evidenced by Reaper of Souls (the expansion) selling 2.7 million copies in its first week. 
# 2. It's hard to tell how much of Infestation: Survivor Stories's rating is due to review bombing as it's widely regarded as a bad game anyway, but many reviews do complain about gameplay-impacting microtransactions.
# 3. Star Wars Battlefront II locked playable characters through playtime, a fairly common practice in video games- however, the playtime requirements were unprecedently steep (40 hours for Darth Vader). Alternatively, players could unlock them with real money, which the gaming community viewed as an unethical cash grab. Notably, an EA employee discussing microtransactions in SWBCII holds the most downvoted commented on reddit with roughly 30x more downvotes than second place. Following the controversy, costs of unlocking characters were reduced by 75%, but sales of the game were still significantly affected- some sources report that SWBCII only sold 50% of what analysts expected!
# 
# Further investigation suggests that review bombing is much more common than I had anticipated, and thus my original hypothesis that the number of user reviews is a good indicator of game's sales and active players is incorrect. 
# 
# Interestingly, the games with the two highest user scores are God of War and The Witcher 3 (with 2500+ reviews each across all platforms). Both clearly merit high marks on gameplay alone; however, perhaps it isn't coincidental that the both of the development teams have been outspoken against microtransactions!

# In[ ]:


#SUB-PROJECT 5: visualization with rating on the x-axis, and metascore + user score on the y-axis. 
#perhaps I can figure out how to put these both on the same graph maybe with a violinplot where each "violin" is split down the middle?
#Also having trouble putting these side by side. the method I used above doesn't seem to work with seaborn?
#investigate whether difference in metascore for M and T games is statistically significant

#remove RP and AO due to low sample sizes and investigate correlation between genre and score UNFINISHED
#clean_ratings = df1[(df1['rating'] == 'E10+') | (df1['rating'] == 'M') | (df1['rating'] == 'T') | (df1['rating'] == 'E')]

#display data
sns.catplot(x='rating',y='metascore', data = df1, kind="boxen")
sns.catplot(x='rating',y='user_score', data = df1, kind="boxen")
plt.show()


# There are only 19 games listed as RP (Rating Pending); while the sample size is very low, I would expect this category to have a lower average score since it includes unfinished or cancelled games such as Clockwork Empires.
# 
# There are only two games rated "AO" released within the time frame of the data (2011-2019): Hatred for violence, and Seduce Me for sexual content. 
# 
# Otherwise, it seems that M games average the highest metascore while T games average the lowest. I will be doing more investigation to see if the difference is statistically significant (as per usual, once I figure out how), especially since the user ratings seem very consistent between M and T games.

# In[ ]:


#SUB-PROJECT 6: activity on metacritic by year

#display data
df1['release_date'].groupby(df1['release_date'].dt.year).hist()
plt.show()


# Metacritic is growing, perhaps alongside the growing video game industry (although metacritic's growth rate seems to be slowing down). An interesting discovery here is that most video games release near the end of the year- additional research seems to indicate this is primarily due to Christmas.

# In[ ]:


#SUB-PROJECT 7. market share of each platform by year. VERY UNFINISHED
#alright wait, why is there a missing slot in 2014. huh. I don't think I understand how map lambda works...
df1['release_date'].map(lambda d: d.year).plot(kind='hist')
plt.show()


# OK SERIOUSLY WHERE IS 2014

# In[ ]:


#SUB-PROJECT 8. games that critics hated while users loved, and vice versa

#Went back to check the average metascore and user score due to such a large discrepancy in "games users hate" vs. "games users love" at the +/- 25 rating threshold
print("Correlation between user score and metascore: " + str(df1['metascore'].corr(df1['user_score']))[:4])
df1['metascore'].plot(kind = 'kde', xlim = (0, 100), label = ('metascore (mean = ' + (str(df1['metascore'].mean())[:4])) + ')')
df1['user_score'].plot(kind = 'kde', xlim = (0, 100), label = ('user score (mean = ' + (str(df1['user_score'].mean())[:4])) + ')')
plt.xlabel('score')
plt.legend()

#Due to aforementioned discrepancy, we use a threshold of +50 for games users hate
users_hate_df = pd.DataFrame(df1[(df1['metascore'] - df1['user_score'] > 50) & (df1['total_users'] > 5)]) 
users_love_df = pd.DataFrame(df1[(df1['metascore'] - df1['user_score'] < -25) & (df1['total_users'] > 5)])

#display data
plt.show()
print("Games with much higher critic reviews")
display(users_hate_df[['game','developer','platform','genre','release_date','total_critics','total_users','metascore', 'user_score']])
print("Games with much higher user reviews")
display(users_love_df[['game','developer','platform','genre','release_date','total_critics','total_users','metascore', 'user_score']])


# I decided to manually look at the outliers for this data. There are 13 games where the metascore is lower than the user score by 25, whereas there are 227 games (17x!) where the metascore is higher than the user score by the same value of 25 (and at least 5 user reviews). I was surprised at such a huge discrepancy here, so I went back to take a quick look at the correlation and distribution between average metascore vs. user score overall for all 5699 games in the data.
# 
# 227 games is of course too many, so I looked at games where the metascore is higher than the user score by a whopping 50 points, for which there were 21 games. Looking at these first, I quickly recognized all of them except for Out of the Park Baseball 17. Fortnite is objectively one of the most successful, if not the most successful, game of all time. The rest of these are all AAA titles, and every single one of them was a very successful game with the exception of the aforementioned Star Wars Battlefront II and Artifact, both faltering due to their monetization models rather than gameplay faults. Thus, I have to conclude that most of these games were review bombed. Note how many are EA games- a company that is often despised within the more serious gaming community due to the way they implement microtransactions despite consistently producing best-sellers. Also note that the trend of review bombing and seems to have started in 2017, around the time when companies started to push the envelope with microtransactions.
# 
# For games that users rated higher than critics, 4/13 games are fanservice/sex/dating themed: Leisure Suit Larry: Reloaded, Dead or Alive Xtreme 3: Fortune, Senra Kagura Reflexions, and Super Seducer: How to Talk to Girls. Another 4/13 are horror themed: The Haunted: Hell's Reach, Knock-knock, Crystal Rift, and loosely Immortal: Unchained. Note that sample sizes for these games are much lower; thus, I investigated some expanded data and it definitely seems that a disproportionately high number of games in these two categories are rated low by critics but high by users. Amusingly, Left Alive, released very recently in March of 2019, appears to be the only game I've ever seen that has clearly been reverse-review-bombed; it has a 8.6/10 on metacritic with an extremely high number of ratings (1382), despite a 18% user rating for the PC version on Steam with 347 total reviews. 

# In[ ]:


#SUB-PROJECT 9. correlation between user score and metascore across popularity buckets, genres, dates. VERY UNFINISHED
print("Correlation between user score and metascore for:")
print("Overall: " + str(df1['metascore'].corr(df1['user_score']))[:4])

#alright, replace this trash-tier code with a loop when you get the chance. should be pretty easy. Then get the correlations for genres, ratings, dates, etc.
print("PC games: " + str(df1[(df1['platform'] == 'PC')].metascore.corr(df1.user_score))[:4])
print("PS4 games: " + str(df1[(df1['platform'] == 'PS4')].metascore.corr(df1.user_score))[:4])
print("XONE games: " + str(df1[(df1['platform'] == 'XONE')].metascore.corr(df1.user_score))[:4])
print("Switch games: " + str(df1[(df1['platform'] == 'Switch')].metascore.corr(df1.user_score))[:4])


# NINTENDO GAMERS ARE SHEEP (jk)

# In[ ]:


#SUB-PROJECT 10 Analyze polarizing games (lots of positive and negative reviews) vs non-polarizing games (lots of middling reviews)

#metacritic considers a score "neutral" if it is between 50 and 75. I manually toyed with numbers to achieve a reasonably-sized result here
polarizing_critics_df = pd.DataFrame(df1[(df1['neutral_critics'] / df1['total_critics'] < .2) & (df1['metascore'] > 50) & (df1['metascore'] < 75) & (df1['total_users'] > 5)]) 
polarizing_users_df = pd.DataFrame(df1[(df1['neutral_users'] == 0) & (df1['user_score'] > 50) & (df1['user_score'] < 75) & (df1['total_users'] > 13)]) 

#display data
print("Games with polarized critic reviews")
display(polarizing_critics_df[['game','developer','platform','genre','release_date','total_critics','total_users', 'metascore', 'user_score']])
print("Games with polarized user reviews")
display(polarizing_users_df[['game','developer','platform','genre','release_date','total_critics','total_users', 'metascore', 'user_score']])


# I wasn't sure what parameters to use here to best represent polarizing games for critics and users- the data presented here is not that great as it has serious sample size issues. As discussed earlier, users leave neutral reviews much more rarely than critics- thus the two tables were created differently. Unsurprisingly, very unique games show up in both the critics and users sections, e.g., Fortix 2 and Cloud Chamber, respectively. "Hack and slash" games appear to be overrepresented in the set of games with polarizing critic reviews. Some possible reasons for polarizing user reviews include: too high expectations (BioShock: The Collection), too buggy (Out of the Park Baseball 15), or too difficult (Mark McMorris Infinite Air). Some gamers tend to be more accepting of these things while others are not. 

# Most interesting overall conclusions for me as a gamer:
# 1. Users leave about half as many neutral reviews as critics, opting for negative reviews instead.
# 2. PC games account for 38% of games listed on metacritic, and console games account for the rest.
# 3. Critics never rate a AAA title poorly. Ever.
# 4. Users especially love leaving negative reviews to protest microstransactions.
# 5. M-rated games score the highest, T-rated games score the lowest (needs verification for statistical sigificance).
# 6. Most games are released near the end of the year in order to capitalize on buyers celebrating the birth of Christ.
# 7. 
# 8. Once again, users love leaving extremely negative users to protest microtransactions. Games in the horror and dating genres seem prone to be underrated by critics.
# 9. 
# 10. Weak data, but games in the hack and slash genre seem to be polarizing to critics. 
