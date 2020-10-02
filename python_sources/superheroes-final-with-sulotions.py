#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
from hashlib import md5
from IPython.display import Image
from matplotlib import pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None  # default='warn'


# In[ ]:


import os
print(os.listdir("../input"))
print(os.listdir("../input/superheroes-comics-and-characters/"))
print(os.listdir("../input/superheroes-stats-n-info/"))


# # Read data from files

# In[ ]:


characters_stats = pd.read_csv("../input/superheroes-stats-n-info/superheroes_stats.csv")
characters_stats.sample(1)


# In[ ]:


superheroes_power_matrix = pd.read_csv("../input/superheroes-stats-n-info/superheroes_power_matrix.csv",index_col="Name")
superheroes_power_matrix.sample(1)


# In[ ]:


characters_info = pd.read_csv("../input/superheroes-stats-n-info/superheroes_info.csv",index_col=0,parse_dates=["FirstAppearance","Year"])
characters_info.sample(1)


# In[ ]:


marvel_comics = pd.read_csv("../input/superheroes-comics-and-characters/comics.csv")
marvel_characters = pd.read_csv("../input/superheroes-comics-and-characters/characters.csv")
marvel_characters_to_comics = pd.read_csv("../input/superheroes-comics-and-characters/charactersToComics.csv")
comics_and_characters = marvel_comics.merge(marvel_characters_to_comics).merge(marvel_characters)

comics_and_characters.sample(1)


# In[ ]:


# marvel_characters_to_comics.groupby("comicID").count()


# ## Easy

# In[ ]:


# e-01
# question: Good versus Evil - Which group has more combined power?
# difficulty: easy
# datasets: characters_stats.csv


# In[ ]:


# danniel
# characters_stats.groupby(characters_stats["Alignment"]).sum()
total_powers_by_aligment = characters_stats[["Alignment","Total"]].groupby(characters_stats["Alignment"]).sum()
total_powers_by_aligment = total_powers_by_aligment.sort_values(by="Total",ascending=False)
total_powers_by_aligment.plot(kind='bar')


# In[ ]:


# nimrod
alignment_based_power = characters_stats.groupby('Alignment')['Total'].sum()
alignment_based_power.plot(kind='bar')


# In[ ]:


# e-02
# question:   Which alignment (good\bad) has higher avg speed?
# difficulty: easy
# datasets: characters_stats


# In[ ]:


# danniel

# remove nneutral 
good_n_bad_characters_stats = characters_stats[characters_stats['Alignment'].isin(["good","bad"])]
good_n_bad_characters_stats
avg_speed_by_aligment = good_n_bad_characters_stats[["Alignment","Speed"]].groupby(characters_stats["Alignment"]).mean()
avg_speed_by_aligment = avg_speed_by_aligment.sort_values(by="Speed",ascending=False)
avg_speed_by_aligment
avg_speed_by_aligment.plot.bar()


# In[ ]:


# nimrod
alignment_based_avg_speed = characters_stats.groupby('Alignment')['Speed'].mean()
alignment_based_avg_speed.plot(kind='bar')
# datasets: characters_stats


# In[ ]:


# e-03
# question: How many superheros have more Intelligence then strength?
# difficulty: easy
# datasets: characters_stats


# In[ ]:


# nimrod
len(characters_stats[characters_stats['Intelligence'] > characters_stats['Strength']])


# In[ ]:


# e-04
# question: Show the distribution of Total (all the powers combined) for the good and the evil.
# difficulty: easy
# datasets: charcters_stats


# # TODO: clarify the question

# In[ ]:


# nimrod
stats_with_data = characters_stats.dropna()

mean_total = stats_with_data['Total'].mean()
max_total = stats_with_data['Total'].max()
bins=range(0,int(max_total + mean_total),20)

plt.hist(stats_with_data["Total"],bins,histtype="bar",rwidth=1.2,color='#0ff0ff')
plt.xlabel('Total') #set the xlabel name
plt.ylabel('Count') #set the ylabel name
plt.plot()
plt.axvline(mean_total,linestyle='dashed',color='red')
plt.show()


# In[ ]:


# Omer
plt.subplots(figsize = (20,10))
plt.title('Attack by Type1')
sns.violinplot(x = "Alignment", y = "Total",data = characters_stats)
plt.show()


# In[ ]:


# e-05
# question: How many comics with 7 or more character published each year?
# difficulty: easy
# datasets: comics_and_characters


# # TODO: move to hard and solve

# In[ ]:


# Omer 
# Answer 1
grouped_up_comics = comics_and_characters.groupby('comicID').count()
grouped_up_comics.loc[grouped_up_comics['characterID']>=7].count()


# In[ ]:


# Omer
# Answer 2
comics_series = comics_and_characters['comicID'].value_counts()#.groupby('comicID').count()
len(comics_series[comics_series >= 7])


# In[ ]:


# e-06

# question: How has more characters DC or Marvel?
# difficulty: easy
# datasets: comics,characters,comics_to_characters


# In[ ]:


# Nimrod
marvel_and_dc_characters_only = characters_info[characters_info.Publisher.isin(["Marvel","DC"])]
# marvel_and_dc_characters_only.sample(5)
universe_based_count = marvel_and_dc_characters_only.groupby('Publisher')['Publisher'].count()
universe_based_count.plot(kind='bar')


# In[ ]:


# e-07
# question: Who has higher representation of female heros DC or Marvel?
# difficulty: easy
# datasets: characters_info


# In[ ]:


# TODO: update question


# In[ ]:


# Nimrod
marvel_and_dc_characters_only = characters_info[characters_info.Publisher.isin(["Marvel","DC"])]
universe_based_woman_percentage = 100 * marvel_and_dc_characters_only[marvel_and_dc_characters_only['Gender'] == 'Female'].groupby('Publisher')['Publisher'].count() / marvel_and_dc_characters_only.groupby('Publisher')['Publisher'].count()
universe_based_woman_percentage.plot(kind='barh')


# In[ ]:


#hen
marvel_and_dc_characters_only = characters_info[characters_info.Publisher.isin(["Marvel","DC"])]
marvel_vs_dc_woman=marvel_and_dc_characters_only[marvel_and_dc_characters_only['Gender']=='Female'].groupby('Publisher')['Publisher'].count()
if (marvel_vs_dc_woman.DC > marvel_vs_dc_woman.Marvel):
    print ("DC") 
elif (marvel_vs_dc_woman.DC < marvel_vs_dc_woman.Marvel):
    print ("MARVEL")
else:
    print ("EQUAL")
    
marvel_vs_dc_woman.plot(kind='bar')


# In[ ]:


# e-08
# question: Who has higher representation of black skined heros DC or Marvel?
# difficulty: easy
# datasets: characters_info


# # TODO: change to hair color

# In[ ]:


param_to_compare = "EyeColor"
param_to_compare = "HairColor"
characters_with_info = characters_info[~characters_info[param_to_compare].isna()]
# len(characters_with_info)
characters_with_info[param_to_compare].value_counts().sort_values(ascending=False).head(5)


# In[ ]:


# Nimrod
universe_based_black_percentage = 100 * characters_info[characters_info['SkinColor'].isin(['Black'])].groupby('Publisher')['Publisher'].count() / characters_info.groupby('Publisher')['Publisher'].count()
universe_based_black_percentage.plot(kind='barh')

### Not enough information to answer this question, not a very good graph :) ###


# In[ ]:


# e-09
### dup ###
# question: Show how common is each trait in 'superheroes_power_matrix.csv'.
# difficulty: easy
# datasets: superheroes_power_matrix


# # TODO: Remvoe Dup

# In[ ]:


# e-10
# question: Show the hight distrebution for the characters of 'Marvel Comics' (from 'characters_info.csv').
# difficulty: easy
# datasets: characters_info


# In[ ]:


# Nimrod
character_with_data = characters_info.dropna()
mean_total = character_with_data['Height'].mean()
max_total = character_with_data['Height'].max()
bins=range(0,int(max_total + mean_total),20)

plt.hist(character_with_data["Height"],bins,histtype="bar",rwidth=1.2,color='#0ff0ff')
plt.xlabel('Total') #set the xlabel name
plt.ylabel('Count') #set the ylabel name
plt.plot()
plt.axvline(mean_total,linestyle='dashed',color='red')
plt.show()


# In[ ]:


# e-11
# question: Show the distrebution of apperences.
# difficulty: easy
# datasets: characters_info.csv


# In[ ]:


# Nimrod
total_aperences = marvel_characters_to_comics['characterID'].value_counts()
mean_total = total_aperences.mean()
max_total = total_aperences.max()
bins=range(0,int(max_total + mean_total),20)
plt.hist(total_aperences,bins,histtype="bar",rwidth=1.2,color='#0ff0ff')
plt.xlabel('Apperences') #set the xlabel name
plt.ylabel('Count') #set the ylabel name
plt.plot()
plt.axvline(mean_total,linestyle='dashed',color='red')
plt.show()


# In[ ]:


# e-12
# question: Show the distrebution of eye colors.
# difficulty: easy
# datasets: characters_info.csv


# In[ ]:


eye_color=characters_info.groupby('EyeColor')['EyeColor'].count()
eye_color.plot(kind='bar')


# In[ ]:


# Nimrod
characters_info['EyeColor'].value_counts().plot.barh()


# In[ ]:


# Nimrod
characters_info['EyeColor'].value_counts().head(10).plot.pie()


# In[ ]:


# e-13
# question: How many characters apperred only once?
# difficulty: easy
# datasets: characters_info.csv


# In[ ]:


comic_aperences = comics_and_characters['characterID'].value_counts()
len(comic_aperences[comic_aperences == 1])


# In[ ]:


# e-14
# question: How many characters died in thair first apperance (have one apperance and are deceased)?
# difficulty: easy
# datasets: characters_info.csv


# In[ ]:


#hen
died_in_first_apperance=characters_info[(characters_info['Appearances']==1) & (characters_info['Status']=='Deceased')] 
print (len(died_in_first_apperance))


# In[ ]:


# nimrod
len(characters_info[(characters_info['Appearances'] == 1.0) & (characters_info['Status'] == 'Deceased')])


# In[ ]:


# e-15
# question:   Display a pie chart of the 10 most common hair styles
# difficulty: easy
# datasets: characters_info


# In[ ]:


# Eitan
marvel_chars_df = characters_info.replace("No Hair", "Bald") #this transformation needs to be added to the data
marvel_chars_df = characters_info.replace("-", "Unknown") #this transformation needs to be added to the data
most_common_hair_styles = pd.value_counts(marvel_chars_df["HairColor"]).head(10)
most_common_hair_styles.plot.pie()


# In[ ]:


# e-16
# question: Display the average height
# difficulty: easy
# datasets: characters_info


# In[ ]:


# ?
marvel_chars_df
pd.DataFrame.mean(marvel_chars_df["Height"])


# In[ ]:


# e-17
### dup ###
# find the comic with most characters. display the comics name and the name of the characters
# difficulty: easy


# In[ ]:


# Nimrod
#find the comic with the most participants
best_comic = comics_and_characters["title"].value_counts().idxmax()
print(best_comic)
comics_and_characters[comics_and_characters['title'] == best_comic]['name'].head()


# In[ ]:


# e-18
# the oldest character of both universes
# difficulty: easy
pd.DataFrame.min(characters_info["Year"])


# In[ ]:


# Nimrod
dc = characters_info[characters_info['Publisher'] == 'DC']
marvel = characters_info[characters_info['Publisher'] == 'Marvel']
marvel.sample()
print(dc[['Name','Year']].loc[dc['Year'].idxmin()])
print(marvel[['Name','Year']].loc[marvel['Year'].idxmin])


# In[ ]:


# e-19
# we want to build the master group to fight evil, kind of an avengers 2.0, but only better,
# lets select the captain, the one with the most total stats  (obviously his Alignment must be good to fight evil)
# level: easy


# In[ ]:


# Nimrod
good_characters = characters_stats[characters_stats['Alignment'] == 'good']
good_characters.loc[good_characters['Total'].idxmax()][['Name','Total']]


# In[ ]:


# danniel
good_characters = characters_stats[characters_stats['Alignment'] == 'good']
max_total = good_characters['Total'].max()
good_characters[good_characters['Total']==max_total]


# In[ ]:


# e-20
# People will pay big money for original vintage comic books, retrive all first issue comic books
# level: easy
# datasets: comics_characters


# In[ ]:


# Nimrod
marvel_comics[marvel_comics['issueNumber'] == 1.0].head()


# In[ ]:


# e-21
# On the other hand, long lasting series are great as well :), retrive the comic book with the biggest issue number
#level: easy
# datasets: comics_characters


# In[ ]:


# Nimrod
long_lasting_series = marvel_comics.loc[marvel_comics['issueNumber'].idxmax()]
long_lasting_series


# In[ ]:


# e-22
# It's the holiday season, and to celebrate marvel usually comes out with holiday special comic books, 
# retrive all  holiday special comic books (the word 'Holiday' will appeer in the title)
# level: easy


# In[ ]:


# Nimrod
marvel_comics[marvel_comics.title.str.contains('Holiday')].head() 


# In[ ]:


# e-23
# What's the mean intelligence of the superheroes who have a 'True' value in the power matrix and the same for the superheroes who have a 'False' value?

# difficulty: easy


# In[ ]:


# ?
characters_intelligence = characters_stats.merge(superheroes_power_matrix, on='Name')[['Name', 'Intelligence_x', 'Intelligence_y']].rename(columns={'Intelligence_x': 'Intelligence Score', 'Intelligence_y': 'Is Intelligent'})
characters_intelligence.groupby('Is Intelligent').mean()


# ## Medium

# In[ ]:


# m-01
# question: Show 5 top comics with top participants on a plot bar.
# difficulty: medium
# datasets: comics,characters,comics_to_characters


# In[ ]:


# Nimrod
comics_and_characters['title'].value_counts().head(5).plot(kind='barh')


# In[ ]:


# m-02
# question: Unmatched rivals - show for each super hero the number of vilans that stronger then him/her
# difficulty: medium-hard
# datasets: charcters_stats.csv


# # TODO: solve

# In[ ]:


# m-03
# question: Weak point - for each vilan, show his weakest characteristic.
# difficulty: medium
# datasets: characters_stats


# In[ ]:


# Nimrod
charactaristics = ['Intelligence','Strength','Speed','Durability','Power','Combat']
villans = characters_stats[characters_stats['Alignment'] == 'bad']
villans['weakest'] = villans[charactaristics].idxmin(axis=1)
villans[['Name','weakest']].sample(10)


# In[ ]:


# m-04
# question: Who can beat me? - for each vilan, show how many superheros can defeat them (compare by total score)
# difficulty: medium
# datasets: characters_stats


# In[ ]:


#hen
superheros_number = []
superhearos= characters_stats[characters_stats['Alignment'] == 'good'][['Name','Total']]
villans= characters_stats[characters_stats['Alignment'] == 'bad'][['Name','Total']]
for villan_power in villans['Total']:
    number=len(superhearos.loc[superhearos['Total']>villan_power])
    superheros_number.append(number)
    
villans['superheros_num']=superheros_number

print (villans[['Name','superheros_num']])


# In[ ]:


# m-05
# question: Display box plot summarizing the next statistics:
# Height, Weight, Intelligence, Strength, Speed, Durability, Power, Combat
# difficulty: medium
# datasets: characters_info, characters_stats


# In[ ]:


# Nimrod
sns.boxplot(data=characters_stats[['Intelligence','Strength','Speed','Durability','Power','Combat']])
plt.ylim(0,150)  #change the scale of y axix
plt.show()


# In[ ]:


# m-06
# find the comics with most participants and display all of the participants
# difficulty: medium


# # TODO: solve

# In[ ]:


# m-07
# A great team needs great diversity, and to be great at everything, get the best hero at each statistical category
# level: easy - medium


# In[ ]:


# Nimrod
good_characters = characters_stats[characters_stats['Alignment'] == 'good']
stats = ['Intelligence','Strength','Speed','Durability','Power','Combat']
max_stats_rows = []
for stat in stats:
    max_stats_rows.append(good_characters.loc[good_characters[stat].idxmax()][['Name',stat]])
pd.concat(max_stats_rows)


# In[ ]:


# m-08
# Is your strngth and intelligence related?.
# Show a scatter chart where the x axis is stength, and the y axis is intelligence, scatter heros and villans as two different color dots
# level: easy - medium


# In[ ]:


# Nimrod
good = characters_stats[characters_stats['Alignment'] == 'good'] #fire contains all fire pokemons
bad = characters_stats[characters_stats['Alignment'] == 'bad']  #all water pokemins
plt.scatter(good.Strength.head(100),good.Intelligence.head(100),color='B',label='Good',marker="*",s=50) #scatter plot
plt.scatter(bad.Strength.head(100),bad.Intelligence.head(100),color='R',label="Bad",s=25)
plt.xlabel("Stength")
plt.ylabel("Intelligence")
plt.legend()
plt.plot()
fig=plt.gcf()  #get the current figure using .gcf()
fig.set_size_inches(12,6) #set the size for the figure
plt.show()


# In[ ]:


# m-09
# To truly be a great superhero, you can't be a one trick pony, you need to posess multipule abilities. Create a series of every superhero and how many different abilities they posess, in descending order
# level: medium


# In[ ]:


# Nimrod
ability_count = [ int(sum([row[c] for c in superheroes_power_matrix.columns])) for index, row in superheroes_power_matrix.iterrows() ]
superheroes_power_matrix['Ability Count'] = ability_count
superheroes_power_matrix['Ability Count'].sort_values(ascending=False).head()


# In[ ]:


# m-10
# Create a serires that counts the number of comic book appeerences for each hero
# Bonus: show the top 10 heros in a pie chart

#level: easy - medium


# In[ ]:


# Nimrod
superhero_comic_performences = pd.merge(marvel_characters_to_comics,marvel_characters)['name'].value_counts()
top_10 = superhero_comic_performences.nlargest(10)
labels = top_10.index.tolist()
sizes = top_10.values
colors = ['Y', 'B', '#00ff00', 'C', 'R', 'G', 'silver', 'white', 'M','gray']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.title("Top 10 comic performences")
plt.plot()
fig=plt.gcf()
fig.set_size_inches(7,7)
plt.show()


# In[ ]:


# m-11
# Pick any hero from the previous question and list all the comic book titles that he appeared in
#level: medium


# In[ ]:


# Nimrod
hero_name = 'Iron Man'
all_character_comics = pd.merge(marvel_characters_to_comics,marvel_characters)
all_ca_comics_id = all_character_comics[all_character_comics['name'] == hero_name]['comicID']
marvel_comics[marvel_comics['comicID'].isin(all_ca_comics_id)]['title'].head()


# In[ ]:


# m-12
# It's the holiday season once again, since we already have a list of all holiday comics, 
# retrive all heros who have participated in a holiday comic book
# level: easy - medium


# In[ ]:


# Nimrod
marvel_characters_to_comics[marvel_characters_to_comics['comicID'] == 17429]
holiday_comics = marvel_comics[marvel_comics.title.str.contains('Holiday')]
holiday_character_to_comic = marvel_characters_to_comics[marvel_characters_to_comics['comicID'].isin(holiday_comics['comicID'])]
pd.merge(holiday_character_to_comic,marvel_characters)


# In[ ]:


# m-13


# We saw that the characters with the 'False' intelligence do have a lower intelligence score than the 'True' ones. That means that the 2 different datasets we based our analysis on have a similar evalutaion of characters intelligence in general.
# 
# * Can you find characters that one dataset classifies them as intelligent and the other one classifies them as not intelligent?
# 
# * How many characters like that are there? (since 1 dataset is boolean and the other is numeric, assume that 100 score equals 'True' and 0 score equals 'False')
# 
# * What that might say on those 2 datasets?

# In[ ]:


# difficulty: medium


# In[ ]:


# Adir
first_mismatch = characters_intelligence.loc[(characters_intelligence['Intelligence Score'] == 100) & (~characters_intelligence['Is Intelligent'])]
second_mismatch = characters_intelligence.loc[(characters_intelligence['Intelligence Score'] == 0) & (characters_intelligence['Is Intelligent'])]
mismatch = pd.concat([first_mismatch, second_mismatch])
mismatch.head(10)
# len(mismatch) - the answer


# * Yes - solution above.
# * 8 - solution above.
# * That means that those 2 datasets sometimes classifies characters differently and we, as analysts, should pay attention to this fact.

# In[ ]:


# m-14
# show the distribution of BMI for all characters with height and weight data. show the distribution according to BMI categories
# difficulty: medium
characters_info
height_n_weight_info = characters_info[["Name","Height","Weight","AdditionalData"]]
height_n_weight_info.dropna(inplace=True)
height_n_weight_info.drop_duplicates(inplace=True)
height_n_weight_info["BMI"] = height_n_weight_info.Weight/np.power(height_n_weight_info.Height/100,2)
height_n_weight_info["WeightStatus"] = pd.cut(height_n_weight_info['BMI'], [0, 18.5, 25,30,1000],labels=["Underweight","Normal weight","Overweight","Obesity"])
height_n_weight_info.WeightStatus.value_counts().plot.bar()


# ## Hard

# In[ ]:


# h-01
# question: Show pairs of characters that always appear together. rank them by number of appearances
# difficulty: hard - very hard
# datasets: comics,characters,comics_to_characters



# In[ ]:


# danniel
comics_character_list = comics_and_characters[["title","name"]]
# fill_val = 0, since NaN is incomparable
comics_character_matrix = comics_character_list.pivot_table(index="name",columns="title",aggfunc=len,fill_value=0)
comics_character_matrix.sample(5)


# In[ ]:


comics_character_matrix["ComicsArrayHash"] = ""
comics_character_matrix.sample(5)
import hashlib
stop_after = 5
curr_id = 0
# len(comics_character_matrix.index)

for curr_character in comics_character_matrix.index:
    comics_array = comics_character_matrix.loc[curr_character]
    comics_array = comics_array.values.tobytes()
    
    digest = hashlib.sha224(comics_array).hexdigest()
    comics_character_matrix.at[curr_character,"ComicsArrayHash"] = digest
#     print (comics_array.values)
#     curr_id += 1
    
#     if curr_id == stop_after:
#         break

    
    
comics_character_matrix["ComicsArrayHash"].sample(5)


# In[ ]:


loyal_characters_ids = pd.DataFrame(comics_character_matrix.ComicsArrayHash.value_counts())
# loyal_characters_ids = loyal_characters_ids[loyal_characters_ids.ComicsArrayHash > 1]
loyal_characters_ids.sample(10)
# appear_together = trns.pivot_table(index="ComicsArrayHash",)


# In[ ]:


trns = pd.DataFrame(comics_character_matrix["ComicsArrayHash"])
trns.ComicsArrayHash.value_counts().sort_values(ascending=False)
# trns.ComicsArrayHash
# trns = trns
# trns.sample(5)


# In[ ]:


# first try
number_of_columns = len(comics_character_matrix.columns)
loyal_duos = pd.DataFrame(columns=["Appearances","DuoName"])

stop_after = number_of_columns
stop_after = 3

for first_idx in range(number_of_columns):
    if first_idx == stop_after:
        break
    
    for second_idx in range(first_idx+1,number_of_columns):
        first_charcter_name = comics_character_matrix.columns[first_idx]
        second_charcter_name = comics_character_matrix.columns[second_idx]
        are_equals = comics_character_matrix[first_charcter_name].equals(comics_character_matrix[second_charcter_name])
        
        if (are_equals):
            duo_name = first_charcter_name,"-",second_charcter_name
            num_of_appearances = len(comics_character_matrix[comics_character_matrix[first_charcter_name]!=0])
            new_data = [{"DuoName":duo_name,"Appearances":num_of_appearances}]
            loyal_duos = loyal_duos.append(new_data,ignore_index=True)
#             print(duo_name)

loyal_duos       
loyal_duos.sort_values(by=["Appearances"],ascending =False)


# In[ ]:


# Nimrod
comics_to_duos = pd.merge(marvel_characters_to_comics,marvel_characters_to_comics,on='comicID')
comics_to_duos = comics_to_duos[comics_to_duos['characterID_x'] > comics_to_duos['characterID_y']]
unique_duos = comics_to_duos.groupby(['characterID_x','characterID_y']).size().reset_index().rename(columns={0:'apperences_together'})
heros_together = []
for index, duo in unique_duos.iterrows():
    x_apperences = len(marvel_characters_to_comics[marvel_characters_to_comics['characterID'] == duo['characterID_x']])
    y_apperences = len(marvel_characters_to_comics[marvel_characters_to_comics['characterID'] == duo['characterID_y']])
    xy_apperences = duo['apperences_together']
    if x_apperences == y_apperences == xy_apperences:
        heros_together.append({'x':duo['characterID_x'],'y':duo['characterID_y'],'apperences':xy_apperences})
heros_together_df = pd.DataFrame(heros_together)
heros_together_df= pd.merge(heros_together_df,marvel_characters,left_on='x',right_on='characterID')
heros_together_df= pd.merge(heros_together_df,marvel_characters,left_on='y',right_on='characterID')
heros_together_df[['apperences','name_x','name_y']].sort_values(by=['apperences'],ascending=False).sample(5)


# In[ ]:


# h-02
# question: Unmatched rivals - show for each super hero , all the names of the  vilans that stronger then him/her
# difficulty: hard
# datasets: characters_stats.csv

# BONUS
# question: Unmatched rivals - find an informative way to visualize the results you got


# In[ ]:


# danniel
characters_stats = pd.read_csv("../input/superheroes-stats-n-info/superheroes_stats.csv",index_col="Name")

good_characters = pd.DataFrame(characters_stats[characters_stats['Alignment'] == 'good']["Total"])
good_characters.dropna(inplace=True)
good_characters.drop_duplicates(inplace=True)
bad_characters = pd.DataFrame(characters_stats[characters_stats['Alignment'] == 'bad']["Total"])
bad_characters.dropna(inplace=True)
bad_characters.drop_duplicates(inplace=True)

good_characters = good_characters.sort_values(by="Total",ascending=False).head(7)
bad_characters = bad_characters.sort_values(by="Total",ascending=False).head(7)

unmatched_rivals = pd.DataFrame(columns=["Good","Bad","IsUnmatched"])
for curr_good in good_characters.index:
    good_char_total = good_characters.loc[curr_good]["Total"]
#     print (curr_good,":", good_char_total)
    for curr_bad in bad_characters.index:
        bad_char_total = bad_characters.loc[curr_bad]["Total"]
        is_unmatched = "Yes" if np.bool((bad_char_total > good_char_total)) else "No"
#         print ("\t",curr_bad,":", bad_char_total, "unmatched=",is_unmatched)
        new_data = [{"Good":curr_good,"Bad":curr_bad,"IsUnmatched":is_unmatched}]
        unmatched_rivals = unmatched_rivals.append(new_data,ignore_index=True)
 
unmatched_rivals.head(100)

def concat_characters(in_str):
    result = in_str.str.cat(sep=", ")
    return result
    

u_matrix = unmatched_rivals.pivot_table(index="Good",columns="IsUnmatched",values="Bad",aggfunc=concat_characters)
result = pd.DataFrame(u_matrix.Yes)
result


# In[ ]:


# h-03
# show 5 top rare and common abilites
# difficulty: hard


# In[ ]:


# danniel
superheroes_power_matrix

shp_trans = superheroes_power_matrix.transpose()
shp_trans
shp_trans["AbilityNum"] = (shp_trans[shp_trans.columns]==True).sum(axis=1)
abilities_sorted_by_freq = shp_trans["AbilityNum"].sort_values(ascending=False)
abilities_sorted_by_freq

common = abilities_sorted_by_freq[:5]
common

rare = abilities_sorted_by_freq[-5:]
rare
 
common_and_rare = common.append(rare)
common_and_rare


# In[ ]:


# h-04
# Two of the most iconic marvel superheros, Iron Man and Captain America, appeer together quite offten. 
# see if you can get the ammount of comic books they both appear in

# level: medium - hard


# In[ ]:


# Nimrod
all_character_comics = pd.merge(marvel_characters_to_comics,marvel_characters)
marvel_comics
all_ca_comics_id = all_character_comics[all_character_comics['name'] == 'Captain America']['comicID']
all_ca_comics = marvel_comics[marvel_comics['comicID'].isin(all_ca_comics_id)]

all_im_comics_id = all_character_comics[all_character_comics['name'] == 'Iron Man']['comicID']
all_im_comics = marvel_comics[marvel_comics['comicID'].isin(all_im_comics_id)]

len(pd.merge(all_im_comics,all_ca_comics,on='comicID', how='inner'))


# In[ ]:


# h-05
# Now that we know how many comic books both of those guys have appeared together at, are they the best power duo in the marvel universe?.
# craete a series with a multi index of 2 superheros(name1,name2) and count for each of them the ammount of comic books they have been in together in, order by that ammount in a descending order
# level: really hard :)


# In[ ]:


# Nimrod
marvel_characters_to_comics
comics_to_duos = pd.merge(marvel_characters_to_comics,marvel_characters_to_comics,on='comicID')
comics_to_duos = comics_to_duos[comics_to_duos['characterID_x'] > comics_to_duos['characterID_y']] # remove acuurences where the 2 heros are the same, 


# and removes duplicates, because hero1,hero2 == hero2,hero1
comic_duos = pd.merge(comics_to_duos,marvel_characters,left_on='characterID_x',right_on='characterID').drop(['characterID','characterID_x'],axis=1).rename(columns={'name':'name1'})
comic_duos = pd.merge(comic_duos,marvel_characters,left_on='characterID_y',right_on='characterID').drop(['characterID','characterID_y'],axis=1).rename(columns={'name':'name2'})

# comic_duos = comic_duos.set_index(['name1','name2'])
# comic_duos = comic_duos.groupby(level=[0,1]).size().sort_values(ascending=False)
# top_duos = comic_duos[:5]


comic_duos["DouName"] = comic_duos.name1 + "-" + comic_duos.name2
top_duos = comic_duos.DouName.value_counts()[:5]

top_duos.plot.barh()


# # The End
