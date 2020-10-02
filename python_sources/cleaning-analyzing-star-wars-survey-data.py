#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This project aims to clean, explore and analyze the dataset resulting from a FiveThirtyEight survey about Star Wars, where they collected responses from fans before Star Wars: The Force Awakens came out. 
# 
# Available at https://github.com/fivethirtyeight/data/tree/master/star-wars-survey.

# # Read in the data

# In[ ]:


import pandas as pd

star_wars = pd.read_csv("../input/star-wars-survey-data/star_wars.csv", encoding="ISO-8859-1")
star_wars.head(10)


# In[ ]:


star_wars.columns


# # Cleaning
# 
# To start out cleaning the data, let's remove invalud rows, starting by dropping any row whose RespondentID column is NaN, since that value is supposed to be a unique ID for each respondent.

# In[ ]:


star_wars = star_wars[pd.notnull(star_wars['RespondentID'])]


# Next, mapping Yes/No responses to True/False in the first two columns that contain fan answers to a question will make them easier to work with. 
# 
# These columns are:
# 
# - 'Have you seen any of the 6 films in the Star Wars franchise?'
# - 'Do you consider yourself to be a fan of the Star Wars film franchise?'

# In[ ]:


#Before cleaning
print(star_wars['Have you seen any of the 6 films in the Star Wars franchise?'].value_counts())
print(star_wars['Do you consider yourself to be a fan of the Star Wars film franchise?'].value_counts())


# In[ ]:


yes_no={
    'Yes': True,
    'No': False
}

yes_no_cols = ['Have you seen any of the 6 films in the Star Wars franchise?', 'Do you consider yourself to be a fan of the Star Wars film franchise?']

star_wars['Have you seen any of the 6 films in the Star Wars franchise?'] = star_wars['Have you seen any of the 6 films in the Star Wars franchise?'].map(yes_no)
star_wars['Do you consider yourself to be a fan of the Star Wars film franchise?'] = star_wars['Do you consider yourself to be a fan of the Star Wars film franchise?'].map(yes_no)


# In[ ]:


#After cleaning
print(star_wars['Have you seen any of the 6 films in the Star Wars franchise?'].value_counts())
print(star_wars['Do you consider yourself to be a fan of the Star Wars film franchise?'].value_counts())


# ---
# Next, columns that collected responses to whether or not fans watched each of the Star Wars movies look pretty confusing, as well as they're content. 
# 
# These include:
# 
# - 'Which of the following Star Wars films have you seen? Please select all that apply.' (Referring to Episode I).
# - 'Unnamed: 4' (Referring to Episode II).
# - 'Unnamed: 5' (Referring to Episode III).
# - 'Unnamed: 6' (Referring to Episode IV).
# - 'Unnamed: 7' (Referring to Episode V).
# - 'Unnamed: 8' (Referring to Episode VI).
# 
# Firstly, changing each column name to 'seen_1', 'seen_2' and so on until 'seen_6' (last episode at the time of the survey) will start solving the problem. 

# In[ ]:


cols_seen = {
    'Which of the following Star Wars films have you seen? Please select all that apply.': 'seen_1',
    'Unnamed: 4': 'seen_2',
    'Unnamed: 5': 'seen_3',
    'Unnamed: 6': 'seen_4',
    'Unnamed: 7': 'seen_5',
    'Unnamed: 8': 'seen_6'    
}

star_wars = star_wars.rename(columns=cols_seen)


# In[ ]:


star_wars.columns[3:9]


# Now, since these columns tell us whether or not fans watched the episode, it makes sense and would be convenient to simplify them to True or False. Right now, their elements look like this:
# 
# - Name of the episode: The fan watched the movie.
# - NaN: Missing value.
# 
# For this project, let's assume NaN to be the case where the fan did not see the movie. Based on that, mapping both values to True or False will work.

# In[ ]:


import numpy as np

seen_notseen = {
    
    'seen_notseen_1': {
        star_wars.iloc[0,3]: True,
        np.NaN: False
    },

    'seen_notseen_2': {
        star_wars.iloc[0,4]: True,
        np.NaN: False
    },

    'seen_notseen_3': {
        star_wars.iloc[0,5]: True,
        np.NaN: False
    },
    
    'seen_notseen_4': {
        star_wars.iloc[0,6]: True,
        np.NaN: False
    },
    
    'seen_notseen_5': {
        star_wars.iloc[0,7]: True,
        np.NaN: False
    },

    'seen_notseen_6': {
        star_wars.iloc[0,8]: True,
        np.NaN: False
    },
}


for movie in range(1,7):
    star_wars['seen_' + str(movie)] = star_wars['seen_' + str(movie)].map(seen_notseen['seen_notseen_' + str(movie)])

star_wars.head()


# ---
# Observing the dataset, notice the columns that contain fan responses as to how they rank each Star Wars episode, from best (1) to worst (6).
# 
# The same way as before, these columns include:
# 
# - 'Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.' (Referring to Episode I).
# - 'Unnamed: 10' (Referring to Episode II).
# - 'Unnamed: 11' (Referring to Episode III).
# - 'Unnamed: 12' (Referring to Episode IV).
# - 'Unnamed: 13' (Referring to Episode V).
# - 'Unnamed: 14' (Referring to Episode VI).
# 
# Let's convert their values to float and rename them with 'ranking_1', 'ranking_2', and so on until 'ranking_6'. 

# In[ ]:


star_wars[star_wars.columns[9:15]] = star_wars[star_wars.columns[9:15]].astype(float)


# In[ ]:


cols_rank = {
    'Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.': 'ranking_1',
    'Unnamed: 10': 'ranking_2',
    'Unnamed: 11': 'ranking_3',
    'Unnamed: 12': 'ranking_4',
    'Unnamed: 13': 'ranking_5',
    'Unnamed: 14': 'ranking_6'    
}

star_wars = star_wars.rename(columns=cols_rank)

star_wars.head()


# # Analysis
# 
# Let's start out by finding the highest-ranked movie based on the mean of each of the ranking columns previously cleaned.

# In[ ]:


cols = ['ranking_1', 'ranking_2', 'ranking_3', 'ranking_4', 'ranking_5', 'ranking_6']
agg_data = {}

av_rank = {}

for col in cols:   
    av_rank[col] = star_wars[col].mean()

agg_data['rank_mean'] = av_rank
    
agg_data


# In[ ]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

rank = sorted(agg_data['rank_mean'].items())

plt.bar(range(len(agg_data['rank_mean'])), [x[1] for x in rank], align='center')
plt.xticks(range(len(agg_data['rank_mean'])), ['Ep I', 'Ep II', 'Ep III', 'Ep IV', 'Ep V', 'Ep VI'])
plt.ylabel('Rank')
plt.title('Average Ranking, by Episode')
plt.show()


# The plot shows that, on average, Episode V was thought as the best among all six. One of the reasons could be that it was the first Star Wars movie ever made, hence its impact on the public. It is followed by Ep. VI, Ep. IV, Ep. I, Ep. II and Ep. III, respectively. 
# 
# An interesting fact is that even though technology for movie FX and visual effects has improved over the years, the first trilogy has a better ranking for fans.
# 
# In addition, for the newer, first trilogy (Ep. I-III) the graph shows how every episode has been getting worse as it came out, according to fans opinion.
# 
# However, all these conclusions are the result of people who voted. In order to see how reliable the ranking is, we need to know how many people voted for each episode. Let's find that out.

# In[ ]:


seen_cols = ['seen_1', 'seen_2', 'seen_3', 'seen_4', 'seen_5', 'seen_6']
num_seen = {}

for col in seen_cols:   
    num_seen[col] = star_wars[col].sum()

agg_data['total_seen'] = num_seen
    
agg_data


# In[ ]:


seen = sorted(agg_data['total_seen'].items())

plt.bar(range(len(agg_data['total_seen'])), [x[1] for x in seen], align='center')
plt.xticks(range(len(agg_data['total_seen'])), ['Ep I', 'Ep II', 'Ep III', 'Ep IV', 'Ep V', 'Ep VI'])
plt.title('Total Number of Viewers, by Episode')
plt.ylabel('# of Viewers')
plt.show()


# We could say the conclusions got from the previous graph that showed ranking were pretty accurate. 
# 
# The best episode (Ep. V) has the highest number of viewers (who ranked it in the survey), along with the two other episodes inside the older, second trilogy. 
# 
# It should also be pointed out that even though the first trilogy got a worse response as each episode came out, the number of viewers also dicreased over time. This makes that conclusion a little more questionable.
# 

# ---
# The previous analysis concerned the survey population as a whole. However, it would be interesting to split the analysis by certain  segments, such as:
# 
# - People who consider themselves Star Wars fans and people who don't.
# - People who consider themselves Star Trek fans and people who don't.
# - Gender, female or male.
# 
# Let's repeat the same analysis for these new segments.

# # Star Wars Fans

# In[ ]:


#Star Wars fans
sw_fans = star_wars[star_wars["Do you consider yourself to be a fan of the Star Wars film franchise?"] == True]
not_sw_fans = star_wars[star_wars["Do you consider yourself to be a fan of the Star Wars film franchise?"] == False]

#Repeat analysis, now for both segments.
agg_data_fans = {}
agg_data_not_fans = {}

#Average Ranking
cols = ['ranking_1', 'ranking_2', 'ranking_3', 'ranking_4', 'ranking_5', 'ranking_6']
av_rank_fans = {}
av_rank_not_fans = {}

for col in cols:   
    av_rank[col] = sw_fans[col].mean()
    av_rank_not_fans[col] = not_sw_fans[col].mean()
    
agg_data_fans['rank_mean'] = av_rank
agg_data_not_fans['rank_mean'] = av_rank_not_fans

rank_fans = sorted(agg_data_fans['rank_mean'].items())
rank_not_fans = sorted(agg_data_not_fans['rank_mean'].items())


#Total Number of Viewers
seen_cols = ['seen_1', 'seen_2', 'seen_3', 'seen_4', 'seen_5', 'seen_6']
num_seen_fans = {}
num_seen_not_fans = {}

for col in seen_cols:   
    num_seen_fans[col] = sw_fans[col].sum()
    num_seen_not_fans[col] = not_sw_fans[col].sum()

agg_data_fans['total_seen'] = num_seen_fans
agg_data_not_fans['total_seen'] = num_seen_not_fans

seen_fans = sorted(agg_data_fans['total_seen'].items())
seen_not_fans = sorted(agg_data_not_fans['total_seen'].items())


#Plots
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.bar(range(len(agg_data_fans['rank_mean'])), [x[1] for x in rank_fans], align='center')
ax1.set_xticklabels(['', 'Ep I', 'Ep II', 'Ep III', 'Ep IV', 'Ep V', 'Ep VI'], rotation = 45)
ax1.set_ylabel('Rank')
ax1.set_title('Average Ranking, by Episode\nSTAR WARS FANS')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.yaxis.set_ticks_position('left')

ax2.bar(range(len(agg_data_not_fans['rank_mean'])), [x[1] for x in rank_not_fans], align='center')
ax2.set_xticklabels(['', 'Ep I', 'Ep II', 'Ep III', 'Ep IV', 'Ep V', 'Ep VI'], rotation = 45)
ax2.set_ylabel('Rank')
ax2.set_title('Average Ranking, by Episode\nREST')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.yaxis.set_ticks_position('left')

ax3.bar(range(len(agg_data_fans['total_seen'])), [x[1] for x in seen_fans], align='center')
ax3.set_xticklabels(['', 'Ep I', 'Ep II', 'Ep III', 'Ep IV', 'Ep V', 'Ep VI'], rotation = 45)
ax3.set_ylabel('# of Viewers')
ax3.set_title('Total Number of Viewers, by Episode\nSTAR WARS FANS')
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.yaxis.set_ticks_position('left')

ax4.bar(range(len(agg_data_fans['total_seen'])), [x[1] for x in seen_not_fans], align='center')
ax4.set_xticklabels(['', 'Ep I', 'Ep II', 'Ep III', 'Ep IV', 'Ep V', 'Ep VI'], rotation = 45)
ax4.set_ylabel('# of Viewers')
ax4.set_title('Total Number of Viewers, by Episode\nREST')
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.yaxis.set_ticks_position('left')

plt.tight_layout()
plt.show()


# # Star Trek Fans

# In[ ]:


#Star Trek fans
st_fans = star_wars[star_wars["Do you consider yourself to be a fan of the Star Trek franchise?"] == 'Yes']
not_st_fans = star_wars[star_wars["Do you consider yourself to be a fan of the Star Trek franchise?"] == 'No']

#Repeat analysis, now for both segments.
agg_data_fans = {}
agg_data_not_fans = {}

#Average Ranking
cols = ['ranking_1', 'ranking_2', 'ranking_3', 'ranking_4', 'ranking_5', 'ranking_6']
av_rank_fans = {}
av_rank_not_fans = {}

for col in cols:   
    av_rank[col] = st_fans[col].mean()
    av_rank_not_fans[col] = not_st_fans[col].mean()
    
agg_data_fans['rank_mean'] = av_rank
agg_data_not_fans['rank_mean'] = av_rank_not_fans

rank_fans = sorted(agg_data_fans['rank_mean'].items())
rank_not_fans = sorted(agg_data_not_fans['rank_mean'].items())


#Total Number of Viewers
seen_cols = ['seen_1', 'seen_2', 'seen_3', 'seen_4', 'seen_5', 'seen_6']
num_seen_fans = {}
num_seen_not_fans = {}

for col in seen_cols:   
    num_seen_fans[col] = st_fans[col].sum()
    num_seen_not_fans[col] = not_st_fans[col].sum()

agg_data_fans['total_seen'] = num_seen_fans
agg_data_not_fans['total_seen'] = num_seen_not_fans

seen_fans = sorted(agg_data_fans['total_seen'].items())
seen_not_fans = sorted(agg_data_not_fans['total_seen'].items())


#Plots
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.bar(range(len(agg_data_fans['rank_mean'])), [x[1] for x in rank_fans], align='center')
ax1.set_xticklabels(['', 'Ep I', 'Ep II', 'Ep III', 'Ep IV', 'Ep V', 'Ep VI'], rotation = 45)
ax1.set_ylabel('Rank')
ax1.set_title('Average Ranking, by Episode\nSTAR TREK FANS')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.yaxis.set_ticks_position('left')

ax2.bar(range(len(agg_data_not_fans['rank_mean'])), [x[1] for x in rank_not_fans], align='center')
ax2.set_xticklabels(['', 'Ep I', 'Ep II', 'Ep III', 'Ep IV', 'Ep V', 'Ep VI'], rotation = 45)
ax2.set_ylabel('Rank')
ax2.set_title('Average Ranking, by Episode\nREST')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.yaxis.set_ticks_position('left')

ax3.bar(range(len(agg_data_fans['total_seen'])), [x[1] for x in seen_fans], align='center')
ax3.set_xticklabels(['', 'Ep I', 'Ep II', 'Ep III', 'Ep IV', 'Ep V', 'Ep VI'], rotation = 45)
ax3.set_ylabel('# of Viewers')
ax3.set_title('Total Number of Viewers, by Episode\nSTAR TREK FANS')
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.yaxis.set_ticks_position('left')

ax4.bar(range(len(agg_data_fans['total_seen'])), [x[1] for x in seen_not_fans], align='center')
ax4.set_xticklabels(['', 'Ep I', 'Ep II', 'Ep III', 'Ep IV', 'Ep V', 'Ep VI'], rotation = 45)
ax4.set_ylabel('# of Viewers')
ax4.set_title('Total Number of Viewers, by Episode\nREST')
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.yaxis.set_ticks_position('left')

plt.tight_layout()
plt.show()


# Looks like both Star Wars and Star Trek fans/not fans results follow a very similar pattern. A priori, this tells their respective fans/not fans think similarly about Star Wars movies.
# 
# Even though the proportion is pretty much equal for both segments in all graphs, the number of Star Wars fans that viewed the movies is higher than the number of Star Trek fans, as well as there were less people who did **not** consider themselves **Star Wars** fans that watched the episodes than those who did **not** consider themselves **Star Trek** fans. This was something to expect.
# 
# An interesting fact is that in all cases Episode V was ranked the best and was also the most viewed, confirming conclusions got from the whole survey population results. Also, while both Star Wars and Star Trek fans watched the episodes way more than those who are not fans, they liked the first trilogy less than not-fans, which doesn't happen with the second. 
# 
# This shows the first trilogy is subjectively worse from the eyes of people who could be assumed to know Star Wars better, since they consider themselves fans.

# # Gender

# In[ ]:


#Gender
males = star_wars[star_wars["Gender"] == "Male"]
females = star_wars[star_wars["Gender"] == "Female"]

#Repeat analysis, now for both segments.
agg_data_male = {}
agg_data_not_female = {}

#Average Ranking
cols = ['ranking_1', 'ranking_2', 'ranking_3', 'ranking_4', 'ranking_5', 'ranking_6']
av_rank_male = {}
av_rank_female = {}

for col in cols:   
    av_rank_male[col] = males[col].mean()
    av_rank_female[col] = females[col].mean()
    
agg_data_male['rank_mean'] = av_rank_male
agg_data_not_female['rank_mean'] = av_rank_female

rank_male = sorted(agg_data_male['rank_mean'].items())
rank_female = sorted(agg_data_not_female['rank_mean'].items())


#Total Number of Viewers
seen_cols = ['seen_1', 'seen_2', 'seen_3', 'seen_4', 'seen_5', 'seen_6']
num_seen_male = {}
num_seen_not_female = {}

for col in seen_cols:   
    num_seen_male[col] = males[col].sum()
    num_seen_not_female[col] = females[col].sum()

agg_data_male['total_seen'] = num_seen_male
agg_data_not_female['total_seen'] = num_seen_not_female

seen_male = sorted(agg_data_male['total_seen'].items())
seen_female = sorted(agg_data_not_female['total_seen'].items())


#Plots
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.bar(range(len(agg_data_fans['rank_mean'])), [x[1] for x in rank_male], align='center')
ax1.set_xticklabels(['', 'Ep I', 'Ep II', 'Ep III', 'Ep IV', 'Ep V', 'Ep VI'], rotation = 45)
ax1.set_ylabel('Rank')
ax1.set_title('Average Ranking, by Episode\nMALE')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.yaxis.set_ticks_position('left')

ax2.bar(range(len(agg_data_not_fans['rank_mean'])), [x[1] for x in rank_female], align='center')
ax2.set_xticklabels(['', 'Ep I', 'Ep II', 'Ep III', 'Ep IV', 'Ep V', 'Ep VI'], rotation = 45)
ax2.set_ylabel('Rank')
ax2.set_title('Average Ranking, by Episode\nFEMALE')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.yaxis.set_ticks_position('left')

ax3.bar(range(len(agg_data_fans['total_seen'])), [x[1] for x in seen_male], align='center')
ax3.set_xticklabels(['', 'Ep I', 'Ep II', 'Ep III', 'Ep IV', 'Ep V', 'Ep VI'], rotation = 45)
ax3.set_ylabel('# of Viewers')
ax3.set_title('Total Number of Viewers, by Episode\nMALE')
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.yaxis.set_ticks_position('left')

ax4.bar(range(len(agg_data_fans['total_seen'])), [x[1] for x in seen_female], align='center')
ax4.set_xticklabels(['', 'Ep I', 'Ep II', 'Ep III', 'Ep IV', 'Ep V', 'Ep VI'], rotation = 45)
ax4.set_ylabel('# of Viewers')
ax4.set_title('Total Number of Viewers, by Episode\nFEMALE')
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.yaxis.set_ticks_position('left')

plt.tight_layout()
plt.show()


# The gender graphs look very similar to the previous bar plots showing Star Wars/Star Trek fans contrast, again showing Episode V as the best and most viewed of all six.
# 
# It should also be pointed out that while more males watched the first trilogy they liked them less than females did.
# 
# This suggests that, for both Star Wars and Star Trek, a big part of those who consider themselves fans might be males, whereas those who don't might be to females, since the corresponding graphs look similar.
# 
# Let's find out the exact proportion. 

# In[ ]:


#Star Wars
count_female_sw = females['Do you consider yourself to be a fan of the Star Wars film franchise?'].value_counts()
count_male_sw = males['Do you consider yourself to be a fan of the Star Wars film franchise?'].value_counts()

#Star Trek
count_female_st = females['Do you consider yourself to be a fan of the Star Trek franchise?'].value_counts()
count_male_st = males['Do you consider yourself to be a fan of the Star Trek franchise?'].value_counts()

female_sw_perc = count_female_sw[1]/sum(count_female_sw)
male_sw_perc = count_male_sw[1]/sum(count_male_sw)
female_st_perc = count_female_st[1]/sum(count_female_st)
male_st_perc = count_male_st[1]/sum(count_male_st)


# In[ ]:


print("\nPercentage of Male Star Wars Fans: ", male_sw_perc*100, "%")
print("Percentage of Female Star Wars Fans: ", female_sw_perc*100, "%\n")
print("Percentage of Male Star Trek Fans: ", male_st_perc*100, "%")
print("Percentage of Female Star Trek Fans: ", female_st_perc*100, "%\n")


# Indeed, whoever said yes to being a fan of both franchises in the survey was most likely a male, thus the pattern showed in the previous graphs. 
# 
# However, while this might suggest this conclusion, the relation between gender and being/not being a fan is not trivial. Even though most fans are males the percentages above still are very close to each other, especially in the Star Wars case, just around 10% apart.

# In[ ]:




