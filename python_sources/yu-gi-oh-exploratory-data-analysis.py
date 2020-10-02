#!/usr/bin/env python
# coding: utf-8

# In this notebook I hope to explore data about Monster cards within the Yu-Gi-Oh! Trading Card Game. In addition to looking at the overall picture, I will be interested to find if there are any trends amongst certain attributes and monster types. In a later notebook, I hope to build Machine Learning models to make predicitons about a card's properties.
# 
# The data used here was originally imported from the [YGOHub](https://www.ygohub.com) API, which seems to contain lots of the information I will need for this project. Since Kaggle doesn't allow importing of this kind, I did it locally & made a dataset to upload [here](https://www.kaggle.com/xagor1/ygo-data).
# 
# During this process, I ran into a few problems. 
# 
# Firstly, cards with & or # in their name were not correctly listed in the API, so are missing. This only covers about 35 cards, <1% of the total, so I don't expect it to have too drastic an effect on the trends in the data, or the power of predictive models. These missing entries can possibly be obtained from the YGOPrices API, or another dataset I found her on Kaggle. However, both require cleaning, and modifications to bring them up to the detail contained here. Progress so far on this matter can be [found here](https://www.kaggle.com/xagor1/cleaning-yu-gi-oh-data/notebook). Due to the low number of cards involved, I'm not planning on expending too much effort on the problem.
# 
# Secondly, it's likely slightly out of date, and won't contain all the most recent releases, since the API doesn't seem to have been updated in a little while. This will only get worse as new cards are released.
# 
# Next, some columns, like releases, require further cleaning to be useful.
# 
# In the end, I have nearly 6000 monster entries, which is more than enough to work with for now.

# # Setting Up
# 
# Before doing any Exploratory Data Analysis (EDA), we need to load the data. I have also transformed the values in some columns to account for unusual values, like '?'.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm

import os
print(os.listdir("../input"))


# In[ ]:


YGO_df=pd.read_csv('../input/YGO_Cards_v2.csv',encoding = "ISO-8859-1")
YGO_df.rename(columns={'Unnamed: 0':'Name'},inplace=True)
YGO_df.head()


# In[ ]:


YGO_df['attack'].replace('?',np.nan,inplace=True)
YGO_df['attack'].replace('X000',np.nan,inplace=True)
YGO_df['attack'].replace('---',np.nan,inplace=True)
YGO_df['defense'].replace('?',np.nan,inplace=True)
YGO_df['defense'].replace('X000',np.nan,inplace=True)
YGO_df['defense'].replace('---',np.nan,inplace=True)
YGO_df['number'].replace('None',np.nan,inplace=True)


# In[ ]:


YGO_df.dtypes


# In[ ]:


YGO_df.attack=pd.to_numeric(YGO_df['attack'])
YGO_df.defense=pd.to_numeric(YGO_df['defense'])
YGO_df.number=pd.to_numeric(YGO_df['number'])


# In[ ]:


YGO_df.columns


# # Preliminary EDA
# 
# To start, I want to do some very coarse grained analysis. This will simply look value counts and proportions across various categories, to get a feel for the data.
# 
# First of all, what data do we actually have?
# 
# * Name: The name of the card. This could potentially be useful for building a predictive model, if used with NLP. For example, a lot of Warrior cards might contain Warrior in their name, or a similar word, like Samurai or Knight.
# * Attack: The attack of the card. Varies between 0 and 5,000.
# * Attribute: The attribute of the card. Covers the four classical elements, Earth, Wind, Water and Fire, along with Light, Dark and Divine.
# * Defense: The defense of the card. Varies between 0 and 5,000.
# * has_materials: Whether the card needs materials for their summon (Xyz, Synchro etc)
# * has_name_condition: Whether the card has a special condition related to its name.
# * is_extra_deck: Whether the card is found in the Extra deck. This is a catchall for Fusion, Xyz, Synchro etc
# * is_fusion: Whether the card is a fusion or not.
# * is_link: Whether the card is a Link monster or not.
# * is_pendulum: Whether the card is a pendulum or not.
# * is_synchro: Whether the card is a Synchro or not.
# * is_xyz: Whether the card is an Xyz or not.
# * link_markers: The position of a card's link markers.
# * link_number: The card's link number.
# * materials: If a card does need materials to summon, these are the materials required.
# * monster_types: The type(s) of card. Such as Normal, Effect, Fusion, Flip etc.
# * name_condition: What the card's name condition is.
# * number: The passcode on the card.
# * pendulum_left: The value of the left pendulum scale.
# * pendulum_right: The value of the right pendulum scale.
# * pendulum_text: The pendulum effect of the card (if it has one)
# * releases: When a card was released.
# * Type: The Type of the card, such as Machine, Warrior etc.
# * stars: How many Levels or Ranks a card has.
# * text: The card's effect or flavour text.
# 
# The easiest place to start, is the 3 numerical values most obvious when looking at a card, the attack, defense and stars (level / rank). These give you a very rough feeling for the power of the card, without considering its effect. Of course, without the effect considered, you don't get the full picture.
# 
# Since attack & defense both contain NANs due to unusual characters, I will encode these as -1 for now. A value of zero could also work, but I want a way to tell them apart from the true zero cases. Link monsters for example do not have a defense stat. For stars, I will fill the NANs with zeros, since they literally do not have stars, and it's an unused value.
# 
# Other numerical fearures worth considering might include the Link Number, the Pendulum scales and the Number (or Passcode), but I suspect the latter is meaningless. The majority of monsters are not Link, so I will encode NANs as zero. For the pendulum scales, scales of zero do actually exist, so I will encode them as -1. In both cases, these will then be excluded from my plots. For number, a few monsters do genuinely not have a passcode, so I will encode those values as zero.
# 
# Before doing some graphical EDA, we can take a very quick look at summary statistics for Attack, Defense & Stars. Whilst thier overall range is the same, in general, Attack is slightly higher than Defense, with mean and median values of ~1500, compared to 1200. This is likely because the game generally favours aggressive over defensive play. These values are close to baseline values found for monsters with a level of less than 4. Given that you generally need monsters with a level of 4 or under to start playing the game, it's no surprise that the mean & media level is approximately 4.

# In[ ]:


YGO_df.attack.fillna(-1,inplace=True)
YGO_df.defense.fillna(-1,inplace=True)
YGO_df.stars.fillna(0,inplace=True)
YGO_df.link_number.fillna(0,inplace=True)
YGO_df.pendulum_left.fillna(-1,inplace=True)
YGO_df.pendulum_right.fillna(-1,inplace=True)
YGO_df.number.fillna(0,inplace=True)


# In[ ]:


YGO_df[['attack','defense','stars']].describe()


# # Attack
# 
# First of all, let's look at Attack. If we just look at individual value counts, we see that the most common attack value is actually zero. I knew that monsters with powerful or unusual effects often had zero attack to counteract this fact, but didn't realise it was the most common value in the game.
# 
# The next several most common values all fall below 2000, so within a range you would expect for Level 1-4 cards. 1000, 1800, 1600, 1500, 1200, and 800 are all values that feel familiar to me as a long time Yu-Gi-Oh! player, and can imagine as baselines for the different levels, or for certain types of effects.
# 
# The value of 2000 follows next, which is very common for level 5s, for example. Most of the remaining common values fall below 2000, with a few exceptions, 2400, 2500 and 3000. 2400 is very common for tribute monsters with effects, such as the Monarchs, and 2500 and 3000 are iconic values within the history of the game.
# 
# I was slightly surprised by quite how common my -1 placeholder was.
# 
# The rarest values are either non-hundreds (i.e. XX50, XX10) or above 3000. The former is because it's unusual to see non-rounded numbers in the game, and the latter is because values above 3000 are usually reserved for difficult to summon 'boss' monsters.

# In[ ]:


#Plot of value counts for Attack
_=plt.style.use('fivethirtyeight')
ax=YGO_df.attack.value_counts().plot.bar(figsize=(25,10),rot=90,
                                                   title='Value counts for Monster Attack')
_=ax.set(xlabel='Attack',ylabel='Count')


# Trends are slightly easier to see if we instead plot a histogram of the attack values. Most monsters have 2000 or below attack, with values getting rarer the larger they get. The peak between 1500 and 2000 is likely because of the overlap between Level 1-4 monsters with high attack, and level 5-6 monsters with low attack.

# In[ ]:


#Histogram of attack
plt.figure(figsize=(16,8))
_=plt.hist(YGO_df['attack'])
_=plt.xlabel('Attack',fontsize=25)
_=plt.ylabel('Frequency',fontsize=25)
_=plt.title('Attack Frequency in Yu-Gi-Oh!')
plt.show()


# How does attack vary in relation to other stats? Naively, you would expect that as attack increases, so too does defense. However, if you know Yu-Gi-Oh!, you'd know that often a high value in one stat is offset by a low value in the other, unless it's a particularly powerful card. Since there is a lot of overlap between data points, rather than just show a regular scatterplot, the points have been colour coded based on the density at that value.
# 
# There is a roughly linear relationship between attack and defense, with the largest density of points approximately along attack = defense. However, many points also lie outside this range, with a huge spread of values, covering most possible combinations up to 3,000 attack/defense. The data outside this range is sparser, with values above 3,000 (The attack of Blue Eyes White Dragon), traditionally reserved for special and extra powerful monsters.
# 
# It's also noticeable that many points lie along the axes, where one value is zero, and the other can reach as high as 5,000. There are particularly dense regions around 0 attack with 2000 defense, and 1,500 to 2,000 attack, with 0 defense. This is likely due to common design choices seen for level 4 and under monsters. It's not unusual for these monsters to completely forego one stat for a strong value in the other. Meaning a monster that can protect you, but can't attack the opponent, or a strong attacker, that is worthless on defense.
# 
# The densest points lie below 2,000 attack/ defense, particularly around 0/0 and 1600/1000-1200. As mentioned before, many monsters with powerful effects have no natural attack or defense, to counterbalance this. The latter values just feel familiar as a Yu-Gi-Oh! player, with defense usually skewing slightly lower than attack.
# 
# Another denser region is seen around 2400/2000, which is a very common value for level 5/6 monsters, such as Jinzo, or the Monarchs.

# In[ ]:


#Function for mapping density to colours
def makeColours( vals ):
    colours = np.zeros( (len(vals),3) )
    norm = Normalize( vmin=vals.min(), vmax=vals.max() )

    #Can put any colormap you like here.
    colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]

    return colours


# In[ ]:


#Scatter Plot of Attack vs Defense
densObj = kde([YGO_df.attack,YGO_df.defense])

colours = makeColours( densObj.evaluate([YGO_df.attack,YGO_df.defense]) )

plt.figure(figsize=(12,12))
_=plt.scatter('attack','defense',data=YGO_df,color=colours,s=50)
_=plt.xlabel('Attack',fontsize=25)
_=plt.ylabel('Defense',fontsize=25)
_=plt.title('Attack Vs Defense')
plt.show()


# Similarly, a scatterplot of Attack vs Stars (Level or Rank) shows an approximately linear relationship between the two, with a large spread towards the axis. The instances at zero attack, even up to 12 stars, are likely monsters with powerful effects whilst those at zero stars cover Link monsters and other special cases. Unusually high values below 4 stars are likely from Xyz monsters, or massively detrimental effects, so it would be useful to separate these out.
# 
# The density of values seems a bit more spread out than attack / defense, but does make the roughly linear relationship clearer. The densest points exist at Level 4, which is not surprising, given the importance of that Level to actually playing the game.

# In[ ]:


#Scatterplot of attack vs Stars
densObj = kde([YGO_df.attack,YGO_df.stars])
colours = makeColours( densObj.evaluate([YGO_df.attack,YGO_df.stars]) )

plt.figure(figsize=(12,12))
_=plt.scatter('attack','stars',data=YGO_df,color=colours,s=50)
_=plt.xlabel('Attack',fontsize=25)
_=plt.ylabel('Stars',fontsize=25)
_=plt.title('Attack Vs Stars (Level/Rank)')
plt.show()


# In general, my assumptions turned out to be true. For most star values in the game, generally Xyz monsters have a higher attack than non-Xyz. 12 Star monsters are fairly rare and unusual, so it's not a surprise that non-Xyz are actually larger there. Similarly, 0 Star monsters are nearly exclusively Link Monsters, so of course cannot be Xyz. The one unusual set of results are the 1 Star monsters with above 2000 attack. These are Meklord monsters, who have strong effects and stats, despite being level 1, but come with a restrictive summoning condition.

# In[ ]:


#Above colour-coded to separate Xyz from not-Xyz
sns.set(font_scale=2)
g=sns.pairplot(x_vars=['attack'], y_vars=['stars'], data=YGO_df, hue="is_xyz",
               height=12,plot_kws={"s": 100})
_=g._legend.set_title('Xyz?')
new_labels = ['No', 'Yes']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
_=g.set(xlabel='Attack',ylabel='Stars',title='Attack Vs Stars (Xyz separated)')


# In the previous analysis, we could see Link monsters clearly separated from other monsters, since they don't have a Level. Instead, it might be more instructive to look at how their Link Number varies with attack. My initial assumption would be that higher Link Numbers mean higher attack. This is roughly speaking true, but as always there are some outliers with zero attack. This montser type is still in its relative infancy, so there's probably much more development to come in the future.

# In[ ]:


#Scatter Plot of Attack vs Link Number
plt.figure(figsize=(12,12))
_=plt.scatter('attack','link_number',data=YGO_df,s=50)
_=plt.xlabel('Attack',fontsize=25)
_=plt.ylabel('Link Number',fontsize=25)
_=plt.ylim([0.5,5.5])
_=plt.xlim([-100,3100])
_=plt.title('Attack Vs Link Number')
plt.show()


# Similarly, we can look at how Attack varies with pendulum scale. Even though both Left & Right values exist, both are the same, so we only need to consider one of them. This value can vary between 0 and 13, and as with Link number, I suspect larger values would correlate to larger attack. I think a scale of zero is a particularly rare case, so the relationship might not hold there.
# 
# As it turns out, there is no clear relationship between Attack and Pendulum Scale.

# In[ ]:


#Scatter plot of attack against pendulum scale
plt.figure(figsize=(12,12))
_=plt.scatter('attack','pendulum_left',data=YGO_df,s=50)
_=plt.xlabel('Attack',fontsize=25)
_=plt.ylabel('Pendulum Scale',fontsize=25)
_=plt.title('Attack Vs Pendulum Scale')
_=plt.ylim([-0.5,14])
_=plt.xlim([-100,4100])
plt.show()


# The passcode is an eight digit number included on nearly every Yu-Gi-Oh! card as a sort of serial number. Once upon a time, they were used as a method to obtain copies of the card in the Video Game versions. However, I don't think this has been the case now for many years.
# 
# As expected there is no meaningful relationship between these two values, with the resulting scatterplot simply looking like the distribution of attack values.

# In[ ]:


#Scatterplot of attack vs number
densObj = kde([YGO_df.attack,YGO_df.number])
colours = makeColours( densObj.evaluate([YGO_df.attack,YGO_df.number]) )

plt.figure(figsize=(12,12))
_=plt.scatter('attack','number',data=YGO_df,color=colours)
_=plt.xlabel('Attack',fontsize=25)
_=plt.ylabel('Passcode',fontsize=25)
_=plt.title('Attack Vs Passcode')
plt.show()


# Ultimately, I want to try and build predictive models for the Type & Attribute of a Yu-Gi-Oh! card based on other features, like their attack and defense. For this to be possible, there must be a difference between trends observed across these categories. 
# 
# Fortunately, this seems to be true, but the effect fairly small when considering attribute. Since summary statistics are only slightly different across each category, I have included both numbers and graphical representations.
# 
# All attributes have the same minimum attack value of zero, but not all reach the maximum of 5000. Fire, Water and Wind only reach 3,800, 3,300 and 4,000 respectively. So it could be possible to assign the very rare values above this range to the other Attributes. 
# 
# On average, Dark, Light and Fire seem to have the strongest monsters in terms of attack, and Earth the weakest. This is reflected in the mean, median and quartile values. Although this difference is only by about 100 or 200 points, so hardly a huge difference. The fact Light & Dark have some of the strongest monsters is likely because the cards belonging to the heroes and villains of the series tend to belong to these, like Yugi's Dark Magician.
# 
# Fire, despite being apparently the smallest attribute comes close in terms of attack power, and actually has the largest 1st quartile value. This is likely because Fire is often envisaged as explosive and powerful. Since Earth is generally weaker than the others, it also means it has more outliers in the data, around 4,000 attack and above.
# 
# Divine is a tiny category, consisting of the God cards, which are generally speaking outside the realm of normal cards, and can probably be ignored for now.

# In[ ]:


YGO_df.groupby(['attribute']).describe()[['attack']]


# In[ ]:


#Plot of attack wrt Attribute
fig, ax = plt.subplots(figsize=(12, 12))
my_pal = {"EARTH": "brown", "WATER": "blue", "WIND":"green","LIGHT":"yellow",
          "DARK":"purple","FIRE":"red","DIVINE":'gold'}
g=sns.boxplot(y='attribute',x='attack',data=YGO_df,palette=my_pal)
_=g.set(ylabel='Attribute',xlabel='Attack',title="Attack Variation with Attribute")


# There are many more Types to consider than Attributes, with 25 recorded here. This results in a slightly crowded boxplot, but one where there are much clearer differences between the Types. Since Divine Beast is such a narrow category, it can probably be excluded from this analysis. Similarly, Creator God is just a single card (Although this database has also encoded a generic Token as this type).
# 
# Dragons for example tend to have higher Attack than all other monsters, likely because many of the legendary cards within the game happen to be dragons, such as the Blue Eyes White Dragon, or the 5 Dragons of 5Ds. The 2nd strongest is Wyrm, a relatively recently introduce Type, which is basically a variation on Dragons. They are in fact also the least populous type in the game.
# 
# Plant monsters on the other hand tend to be weaker than other Monsters, with the lowest 1st quartile and mean. Some like Fiend have a fairly wide distribution of values, whilst others like Beast-Warrior or Sea Serpent are narrower.
# 
# Since there are many Types to consider, I won't give an account of each in detail, except to say that at first glance, it seems like Attack would be more useful when trying to predict Attribute, than Type.

# In[ ]:


YGO_df.groupby(['Type']).describe()[['attack']]


# In[ ]:


#Plot of attack wrt Type
fig, ax = plt.subplots(figsize=(12, 12))
g=sns.boxplot(y='Type',x='attack',data=YGO_df)
_=g.set(ylabel='Type',xlabel='Attack',title='Attack Variation with Type')


# # Defense
# 
# From the earlier analysis, we already know that in general Defense is slightly lower than Attack, and that the two are approximately positively correlated. We would therefore expect other relationships like Level vs Defense to be similar.
# 
# As with Attack, it appears that Defense is mostly concentrated below 2,000, the usual cut-off for Level 4 monsters. The most common value this time is 1,000, rather than zero. This is because there are many monsters, regardless of level which have X/1000 for their stats. The 2nd most common value is zero, which as mentioned earlier, is due to powerful or interesting effects being counterbalanced by weak stats.
# 
# Historically, 2,000 was the cut-off defense for Level 4 monsters without effects, but this has gradually crept up over time, with effect monsters now found with this high defense. There are also contributions from Level 5/6 monsters, which commonly have 2,000 defense.
# 
# The next common values are all below 2,000, and will feel familiar to Yu-Gi-Oh! players, even if they aren't sure why, such as 1,200, 800 and 1,500. It's a while before we see any values above 2,000, and in fact monsters without a Defense stat are actually more common. The first we find is 2,500, which is historically tied to boss monsters like the Blue Eyes White Dragon, and Attack values of 3,000.
# 
# As with Attack, non-hundreds values, and values above 3,000 are very rare.

# In[ ]:


#Plot of value counts for Defense
ax=YGO_df.defense.value_counts().plot.bar(figsize=(25,10),rot=90,
                                                   title='Value counts for Monster Defense')
_=ax.set(xlabel='Defense',ylabel='Count')


# Overall trends look similar to those found for Attack, but skewed slightly lower. This time, values peak between 1,000 and 1,500, and decline from there on. The lower peak between 0 and 500 can be attributed to the larger number of zero defense monsters.

# In[ ]:


#Histogram of defense
plt.figure(figsize=(16,8))
_=plt.hist(YGO_df['defense'])
_=plt.xlabel('Defense',fontsize=25)
_=plt.ylabel('Frequency',fontsize=25)
_=plt.title('Defense Frequency in Yu-Gi-Oh!')
plt.show()


# As with attack, the relationship between Stars and Defense is roughly linear, with the higher Star monsters having higher Defense. There is however a lot of spread amongst the monsters, with many having zero defense, even all the way up to 12 stars. This is partly powerful monsters with zero attack & defense, and partly high attack monsters that completely forgo defense. We can also clearly see the largest density of monsters is Level 4, with about 1,000 defense.
# 
# The dense points near the origin is because I encoded monsters without Level as 0, and these are nearly exclusively Link monsters, which also don't have a Defense value.
# 
# As before, it's also likely that outliers are due to Xyz monsters, although I know some of them are due to peculiar regular monsters.

# In[ ]:


#Scatterplot of defense vs Stars
densObj = kde([YGO_df.defense,YGO_df.stars])
colours = makeColours( densObj.evaluate([YGO_df.defense,YGO_df.stars]) )

plt.figure(figsize=(12,12))
_=plt.scatter('defense','stars',data=YGO_df,color=colours,s=50)
_=plt.xlabel('Defense',fontsize=25)
_=plt.ylabel('Stars',fontsize=25)
_=plt.title('Defense Vs Stars (Level/Rank)')
plt.show()


# The distinction between Xyz and non-Xyz monsters is far less clear with Defense than Attack, partly because particularly high defense values will not have a detrimental effect on the game. It might be difficult for your opponent to defeat a Level 5/6 with 3,000 defense, but you can't actually use it yourself to defeat your opponent. Famous examples of such monsters include Big Shield Gardna, and Labyrinth Wall.

# In[ ]:


#Above colour-coded to separate Xyz from not-Xyz
sns.set(font_scale=2)
g=sns.pairplot(x_vars=['defense'], y_vars=['stars'], data=YGO_df, hue="is_xyz",
               height=12,plot_kws={"s": 100})
_=g._legend.set_title('Xyz?')
new_labels = ['No', 'Yes']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
_=g.set(xlabel='Defense',ylabel='Stars',title='Defense Vs Stars (Xyz separated)')


# As with attack, there is not really much relationship between Defense and Pendulum scale. Most scale values can be associated with nearly any Defense value, barring the very rare scales at the upper and lower end (0 and >10).

# In[ ]:


#Scatter plot of defense against pendulum scale
plt.figure(figsize=(12,12))
_=plt.scatter('defense','pendulum_left',data=YGO_df,s=50)
_=plt.xlabel('Defense',fontsize=25)
_=plt.ylabel('Pendulum Scale',fontsize=25)
_=plt.title('Defense Vs Pendulum Scale')
_=plt.ylim([-0.5,14])
_=plt.xlim([-100,4100])
plt.show()


# Also as expected, Defense and Passcode are completely unrelated. All this plot does is show us the distribution of Defense values, making it extra clear that values close to 1,000 are very common.

# In[ ]:


#Scatterplot of defense vs number
densObj = kde([YGO_df.defense,YGO_df.number])
colours = makeColours( densObj.evaluate([YGO_df.defense,YGO_df.number]) )

plt.figure(figsize=(12,12))
_=plt.scatter('defense','number',data=YGO_df,color=colours)
_=plt.xlabel('Defense',fontsize=25)
_=plt.ylabel('Passcode',fontsize=25)
_=plt.title('Defense Vs Passcode')
plt.show()


# We'll round things out by taking a look again at the relationship between Defense and Attribute / Type. Earlier we found that Attack was a weak indicator of Attribute, but would be more successful at mapping Types, like Dragon.
# 
# Defense appears to differ more significantly between Attributes than Attack, as can be seen from a cursory glance over the boxplots. Fire monsters tend to have lower Defense than all others, probably related to the fact they have higher Attack to compensate. Whilst Light tend to have higher Defense than other monsters, something also found for Attack. Wind monsters have the narrowest range of common defense values, whilst Fire appears largest.
# 
# In my mind, I associated Earth monsters with particularly high defense, so was surprised to find Water monsters tend to have larger defense (ignoring outliers).
# 
# Overall, I expect Defense to be slightly more useful at predicting Attribute than Attack.

# In[ ]:


YGO_df.groupby(['attribute']).describe()[['defense']]


# In[ ]:


#Plot of defense wrt Attribute
fig, ax = plt.subplots(figsize=(12, 12))
g=sns.boxplot(y='attribute',x='defense',data=YGO_df,palette=my_pal)
_=g.set(ylabel='Attribute',xlabel='Defense',title="Defense Variation with Attribute")


# We saw earlier that there were variaton between Type with Attack, so it would be interesting to see if the same holds true for Defense. Unsurprisingly, both Dragon and Wyrm monsters are also found to have the highest defenses. So if a monster has high stats in both categories, it's quite likely to be one of these two.
# 
# There are two quite interesting entries in this data, with behaviour we didn't see for Attack, namely Cyberse and Zombie. Both of these have a huge number of monsters with no Defense. With Cyberse this is because many of them are Link monsters, who physically do not have a Defense stat. For Zombies, this is for historical reasons, linked all the way back to the original manga/anime. When Zombies first appeared, they were revived versions of other monsters, brought back from the dead, but with zero defense. This was even a plot point related to how to defeat them. Many zombies have kept up with this tradition, and can still be found with zero defense.
# 
# There's probably a lot more to say about these relationships, but I will just note a few more obvious ones, like the fact Plants and Insects tend to have lower defense, and Fairies show one of the widest spread of possible values.

# In[ ]:


YGO_df.groupby(['Type']).describe()[['defense']]


# In[ ]:


#Plot of defense wrt Type
fig, ax = plt.subplots(figsize=(12, 12))
g=sns.boxplot(y='Type',x='defense',data=YGO_df)
_=g.set(ylabel='Type',xlabel='Defense',title='Defense Variation with Type')


# # Stars (Level & Rank) 
# Some trends within Stars, which encapsulates both Level and Rank (for Xyz), has already been seen within the other data. Higher Star monsters tend to be more powerful, but since this covers both Effect and Attack/Defense you sometimes find high Star monsters with low stats, even going as far as 0/0 for the highest levels.
# 
# Since the only monsters you can summon without fulfilling special conditions are those are Level 4 or lower, it would make sense that most monsters are found to be Level 4 or lower. It comes as no surprise that 4 Star monsters are the most common, since they are an integral part of most decks. This is compounded by the fact that many Xyz monsters are Rank 4, due to the prevalence of Level 4 monsters.
# 
# Lower Stars, from 3 to 1 are less common, since they tend to be weaker than 4 Stars and therefore easier to defeat. I was slightly surprised to learn that the ordering goes 4>3>2>1 though. However, several higher Star monsters are actually more common than 1 Stars. 5 and 6 star monsters usually require 1 tribute to summon, and are therefore harder to summon, and less common than 4 Stars. The fact there are slightly more 6 than 5, is probably just the game favouring even numbers. 
# 
# Monsters with more than 6 Stars usually require 2 tributes to summon, so are once again, less common. The fact there appear to be quite a lot more 8 Stars than 7, is probably once again due to even numbers being favoured. Anything above 8 is especially rare, since it usually includes especially powerful Synchro, Fusion and Xyz monsters, or the God cards.

# In[ ]:


#Plot of value counts for Stars
ax=YGO_df.stars.value_counts().plot.bar(figsize=(20,10),rot=90,
                                                   title='Value counts for Monster Stars')
_=ax.set(xlabel='Stars',ylabel='Count')


# As with other stats, I had naively assumed that Pendulum scale might be related to the Stars of a monster. However, as you can see from the below plot, every level can take just about any Pendulum scale.

# In[ ]:


#Scatter plot of Stars against pendulum scale
plt.figure(figsize=(12,12))
_=plt.scatter('stars','pendulum_left',data=YGO_df,s=50)
_=plt.xlabel('Stars',fontsize=25)
_=plt.ylabel('Pendulum Scale',fontsize=25)
_=plt.title('Stars Vs Pendulum Scale')
_=plt.ylim([-0.5,14])
#_=plt.xlim([-100,4100])
plt.show()


# Once again, Passcode is meaningless in relation to Stars, and the plot just shows us the prevalence of 3 and 4 Star monsters.

# In[ ]:


#Scatterplot of Stars vs number
densObj = kde([YGO_df.stars,YGO_df.number])
colours = makeColours( densObj.evaluate([YGO_df.stars,YGO_df.number]) )

plt.figure(figsize=(12,12))
_=plt.scatter('stars','number',data=YGO_df,color=colours)
_=plt.xlabel('Stars',fontsize=25)
_=plt.ylabel('Passcode',fontsize=25)
_=plt.title('Stars Vs Passcode')
plt.show()


# The number of possible Star values is much less than the number of possible Attack and Defense values, so I don't expect to see as much noticeable variation between Types and Attributes as before.
# 
# This is especially stark for Attribute, where Earth, Water and Wind are distributed approximately the same, and only marginally different from Dark, Light and Fire, which themselves are approximately the same. We might be able to say that a high Level monster is more likely to be Dark, Light or Fire, but that's about it.

# In[ ]:


YGO_df.groupby(['attribute']).describe()[['stars']]


# In[ ]:


#Plot of Stars wrt Attribute
fig, ax = plt.subplots(figsize=(12, 12))
g=sns.boxplot(y='attribute',x='stars',data=YGO_df,palette=my_pal)
_=g.set(ylabel='Attribute',xlabel='Stars',title="Stars Variation with Attribute")


# Type also shows far less variation with Stars compared to Attack and Defense, but still allows us to pick out interesting trends. It is no surprise that Dragons and Wyrms are generally speaking the largest, since this was also true of Attack and defense. Most types have most of their monsters as 3-5 Stars, so it's easy to pick out those that don't.
# 
# For example, Beast and Aqua comparatively rarely go above Level 4, whilst Insects and Plants are more commonly found at Level 2 than other Types. The fact Cyberse go all the way to zero, is because they are mostly Link monsters, which do not have a Level. Others, like Fairy, Dinosaur, Machine, Warrior and Fiend are slightly more likely to have Level 6 monsters.
# 
# I expect that Level will have some use when trying to predict the Type of a card, but the small number of possible values, and similarity between types means I doubt it will contribute as much as Defense.

# In[ ]:


YGO_df.groupby(['Type']).describe()[['stars']]


# In[ ]:


#Plot of Stars wrt Type
fig, ax = plt.subplots(figsize=(12, 12))
g=sns.boxplot(y='Type',x='stars',data=YGO_df)
_=g.set(ylabel='Type',xlabel='Stars',title='Stars Variation with Type')


# # Attribute & Other Categories
# 
# I won't bother making any extra plots of Passcode, because as we saw earlier, it's a fairly meaningless feature. Any differences are likely due to random chance than any design intent. This leaves us with several categorical variables to compare against Attribute and Type. Some of these already exist in the data as is, others will need to be extracted from 'monster_types'.
# 
# First, let's check what we have using value_counts. Possible categories incude: Effect, Normal, Xyz, Synchro, Fusion, Pendulum, Tuner, Link, Fusion, Flip, Ritual, Gemini, Spirit, Union, Toon, Token, and apparently a few blank entries. Of these, Synchro, Fusion, Pendulum, Xyz and Link already exist in the database, along with a collective group for the extra deck monsters. This leaves us with Effect, Normal, Tuner, Flip, Ritual, Gemini, Spirit, Union, Toon and Token.
# 
# Token and the blank entries may need closer inspection, since Tokens aren't real cards, and something must be wrong with the blank entries. Cases where cards are simply listed as Xyz or Fusion, without Effect, refer to monsters without effects, such as Gagagigo the Risen.
# 
# The 3 blank entries are just mistakes, and should actually be Effect. This should be easy enough to fix when making new feature columns.
# 
# To address all this, I have added 10 new columns, in the format 'is_X' based on partial string matching from the 'monster_type' column.
# 
# Now, I will explore how Type & Attribute vary given these categories. For example, do Dragons have more Synchro monsters than other types?

# In[ ]:


YGO_df.monster_types.value_counts()


# In[ ]:


YGO_df[YGO_df.monster_types=='[]']


# In[ ]:


#Not the best way to do this, but don't feel like generating names from a loop
YGO_df['is_effect']=False
for idx, row in YGO_df.iterrows():
    if 'Effect' in row['monster_types']:
        YGO_df.loc[idx, 'is_effect'] = True

for idx, row in YGO_df.iterrows():
    if '[]' in row['monster_types']:
        YGO_df.loc[idx, 'is_effect'] = True
        
YGO_df['is_normal']=False        
for idx, row in YGO_df.iterrows():
    if 'Normal' in row['monster_types']:
        YGO_df.loc[idx, 'is_normal'] = True

YGO_df['is_tuner']=False
for idx, row in YGO_df.iterrows():
    if 'Tuner' in row['monster_types']:
        YGO_df.loc[idx, 'is_tuner'] = True

YGO_df['is_flip']=False
for idx, row in YGO_df.iterrows():
    if 'Flip' in row['monster_types']:
        YGO_df.loc[idx, 'is_flip'] = True
        
YGO_df['is_gemini']=False
for idx, row in YGO_df.iterrows():
    if 'Gemini' in row['monster_types']:
        YGO_df.loc[idx, 'is_gemini'] = True
        
YGO_df['is_ritual']=False
for idx, row in YGO_df.iterrows():
    if 'Ritual' in row['monster_types']:
        YGO_df.loc[idx, 'is_ritual'] = True
        
YGO_df['is_spirit']=False
for idx, row in YGO_df.iterrows():
    if 'Spirit' in row['monster_types']:
        YGO_df.loc[idx, 'is_spirit'] = True
        
YGO_df['is_union']=False
for idx, row in YGO_df.iterrows():
    if 'Union' in row['monster_types']:
        YGO_df.loc[idx, 'is_union'] = True
        
YGO_df['is_toon']=False
for idx, row in YGO_df.iterrows():
    if 'Toon' in row['monster_types']:
        YGO_df.loc[idx, 'is_toon'] = True
        
YGO_df['is_token']=False
for idx, row in YGO_df.iterrows():
    if 'Token' in row['monster_types']:
        YGO_df.loc[idx, 'is_token'] = True


# To see how Attribute & Type varied between the different monster categories, I plotted value counts for the True cases in each category, as shown below. First I will cover Attribute across all the categories, followed by Type. It's likely easier to see trends in the former, due to the smaller number of categories involved.
# 
# Before looking at the plots, I first got the value counts for each sub-category, converted them to percentages for that sub-category, and calculated the change relative to overall percentage.

# In[ ]:


#List of categories to consider

type_list=['is_normal','is_effect','is_flip','is_gemini','is_union','is_spirit','is_toon','is_tuner',
           'is_token','is_ritual','is_fusion','is_synchro','is_xyz','is_link','is_pendulum']

#Make a dictionary of value counts for each attribute, grouped by sub-categories
d={}
for i in range(len(type_list)):
    d["Att_{0}".format(type_list[i])]=YGO_df.groupby(type_list[i]).attribute.value_counts()[1]

#Convert to a dataframe,and add the Total columns
Attribute_df=pd.DataFrame(d)
Attribute_df['Total']=YGO_df.attribute.value_counts()
Attribute_df['Total_Perc']=round(Attribute_df.Total/sum(Attribute_df.Total)*100,1)
Attribute_df.fillna(0,inplace=True)

#Add a % column for each
for i in range(len(type_list)):
    Attribute_df["Perc_{0}".format(type_list[i])]=round(Attribute_df.iloc[:,i]/np.nansum(
        Attribute_df.iloc[:,i])*100,1)
    
#Add a change relative to Total, called Delta
for i in range(len(type_list)):
    Attribute_df["Delta_{0}".format(type_list[i])]=round(Attribute_df["Perc_{0}".format(type_list[i])]
    -Attribute_df['Total_Perc'],1)

Attribute_df


# Before looking at individual sub-groups, it would be useful to look at the overall distribution for all monsters. We can see that Dark is the most populous Attribute, but only slightly beating Earth, shortly followed by Light. These 3 attributes make up ~75% of all monsters, with the remaining 25% spread relatively evenly between Water, Wind and Fire, with Fire at the bottom. Since it's a tiny category, Divine will be ignored.

# In[ ]:


#Let's also output some %'s'
print('Overall Attribute distribution')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Total_Perc.values[i])+'%')
    
#Attribute count
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Total,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of All Monsters")


# The first sub-group considered is Normal Monsters, which are those without effects. Compared to the full distribution, the proportion of Earth monsters is much higher (+8.9%), and that of Light monsters is much lower (-10%), with the rest changing by smaller amounts. This results in Earth and Dark, and Water and Light switching positions in the overall ordering. Wind and Fire remain at the bottom.
# 
# So, if you know a monster is Normal, it could help with identifying Earth or Light monsters.

# In[ ]:


#Let's also output some %'s'
print('Normal Attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_normal.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_normal.values[i])+'%)')
    
#Attribute count
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_normal,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Normal Monsters")


# Most monsters in the game can be categorised as Effect monsters, those which do have effects, so its no surprise that the trends observed are exactly the same as for the full set of monsters. Only Earth and Light show a change in proportion of >1 %. It's unlikely much extra information can be gained via this category.

# In[ ]:


#Effect monsters
print('Effect monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_effect.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_effect.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_effect,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Effect Monsters")


# Flip monsters are a subset of effect monsters, who have an effect that activates when flipped from face-down, to face-up. Historically, it was not granted its own category, with this change relatively recent. The small number of results shown here is therefore likely only a measure of Flip monsters since the change. As such, I might want to be wary of using this category. As with Normal monsters, Earth attribute is overrepresented compared to the total, and Light underrepresented. Fire has barely any Flip monsters, so it's highly unlikely that an unknown Flip monster would be Fire.

# In[ ]:


#Flip monsters
print('Flip monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_flip.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_flip.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_flip,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Flip Monsters")


# Gemini monsters are a small category of monsters who start off as Normal monsters, but gain effects by performing a second summon, known as a Gemini summon. Even though there is only a tiny number of these monsters, the attribute distribution is significantly different from overall. Whilst Dark is still the most common, Water is the 2nd most, and Fire actually ties with Earth. Unusually, Light is relatively rare. Whilst it won't help identify many monsters, I expect knowing something is a Gemini monster will be a useful indicator. Only Dark and Wind do not show a significant change in distribution compared to the total.

# In[ ]:


#Gemini monsters
print('Gemini monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_gemini.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_gemini.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_gemini,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Gemini Monsters")


# Union monsters are a special class of monsters that can transform themselves into Equip cards in order to power up other monsters. This often means symbiotic monsters, transforming robots, or partners teaming up. Like Gemini before, it's a very rare category, but is noticeably different from the general distribution. There are significantly more Light monsters, which comes at the expense of Dark and Wind. Other Attributes are relative unchanged.

# In[ ]:


#Union monsters
print('Union monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_union.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_union.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_union,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Union Monsters")


# Spirits are a class of monsters based on mythological sources, like Japanese Folklore. They are nearly all characterised by the fact that they activate a special effect when summoned, then return to the hand at the end of the turn. There might only be a handful of Spirits, but the distribution of monsters is one of the most unusual seen so far. Wind is the most common Attribute, followed by Fire and Water. The 3 Attributes normally the rarest are now the most common, and the 3 most common are now the 3 rarest.
# 
# Since Spirit is a tiny class, I don't expect its inclusion to help identify many monsters, but it expect it to help with Wind.

# In[ ]:


#Spirit monsters
print('Spirit monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_spirit.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_spirit.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_spirit,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Spirit Monsters")


# Toon monsters are the signature cards of Pegasus, who in the original anime and manga created Duel Monsters (the in-series name for Yu-Gi-Oh!). They are usually Toon versions of other monsters, gaining the ability to attack directly, but needing Toon World to be played (usually). This is another very rare class, with an unusual distribution. None are Wind, so if we know a card is a Toon, then we know it won't be Wind (but this may change one day). Dark makes up nearly 50% of Toons, with all other attributes declining, or staying about the same as in the general population.
# 
# I expect this will be useful, mainly for the exclusion of Wind.

# In[ ]:


#Toon monsters
print('Toon monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_toon.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_toon.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_toon,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Toon Monsters")


# Tuners are a special class of monsters, which are needed in order to summon Synchro monsters. By combining a Tuner with one or more other monsters, you can summon a Synchro monster with the total number of Stars amongst the materials. They are usually paired with effects, but there are unusual cases, like Formula Synchron, that is both Synchro and Tuner.
# 
# Overall trends are similar to the general population, Light, Dark and Earth are still the top 3, and Water, Wind and Fire the bottom. Light aside, the top 3 all see a decline in their overall share, along with Water, whilst Fire and Wind appear slightly more often. I expect including this feature to only have a minor impact on the results.

# In[ ]:


#Tuner monsters
print('Tuner monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_tuner.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_tuner.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_tuner,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Tuner Monsters")


# Tokens are not proper cards, so I'm just including this here because the YGOHub API had entries for them.

# In[ ]:


#Token monsters
print('Token monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_token.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_token.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_token,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Token Monsters")


# Rituals are a special type of Monster with a Blue border, which can only be summoned using a Ritual spell card, and by fulfilling the conditions on the card. Since you usually lose lots of resources to play them (2 cards minimum), they have tended to not be very popular to use. However, more powerful effects, and ways to mitigate their costs, like replacement effects, have seen them see play more recently.
# 
# Both Light and Water types appear far more than in the general population, and are the two most most common attribute for rituals. I suspect the surge in Water is at least due to the Gishki archetype, which is based around Ritual summons. This is at the expense of all other types, except for Dark, which remains unchanged.

# In[ ]:


#Ritual monsters
print('Ritual monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_ritual.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_ritual.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_ritual,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Ritual Monsters")


# Fusion monsters have a Purple border and were historically summoned by fusing multiple monsters together with Polymerization. Over time, other methods were introduced, like Contact Fusion (or similar), where the fusion card is not needed, or special fusion cards for certain themes, like Heroes.
# 
# The distribution of Attributes amongst fusions is similar to the general population, but with Fire and Dark more common, and the rest less. I'm not sure this will be a particularly useful feature.

# In[ ]:


#Fusion monsters
print('Fusion monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_fusion.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_fusion.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_fusion,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Fusion Monsters")


# Synchro monsters have a White border, and as mentioned earlier must be summoned using a Tuner and other materials. Overall trends are surprisingly similar to the general population, but with a decline for Earth and Water, and a boost for Wind types. I suspect the prevalence of Wind types is because the signature card of the hero of 5Ds (the series which introduced Synchros) is a wind card, Stardust Dragon, which gains many upgraded forms.

# In[ ]:


#Synchro monsters
print('Synchro monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_synchro.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_synchro.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_synchro,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Synchro Monsters")


# Xyz monsters have a Black border, and can be summoned by combining 2 or more monsters of the same Level, to create an Xyz of that Rank. The cards then go underneath the Xyz monster as materials, which can be used for various effects. Like Fusion and Synchro, I expect it to look broadly similar to the general population, but with slight differences due to the fact the Hero of ZeXal uses Utopia, a light monster.
# 
# This assumption turns out to be correct, there are more Light Xyz than anything else, and it shows a large increase relative to the general population. I'm not quite sure why there's such a drop in Earth though.

# In[ ]:


#Xyz monsters
print('Xyz monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_xyz.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_xyz.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_xyz,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Xyz Monsters")


# Pendulum monsters are a special class of monsters introduced in Arc-V, which are both Spell cards and Monster cards at the same time. They have effects based on which variety of card they are played as, and have a new feature known as the Pendulum scale, which can be used to summon many monsters to the field at once.
# 
# In this case, there is an increase in the number of Dark monsters, at the expense of other Attributes, mainly I suspect due to the Odd-Eyes and D/D/D monsters. I'm not sure why there are so few Light monsters.

# In[ ]:


#Pendulum monsters
print('Pendulum monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_pendulum.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_pendulum.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_pendulum,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Pendulum Monsters")


# The final class of monsters is Link monsters, which have a different shade of Blue border, and do not have any Stars or Defense. Instead they have a Link Number and Link markers. They were introduced in V-Rains, as a mechanic change to try and slow the game down. The game had become too reliant on summoning from the Extra deck, and Links were a way to address this. Now, players can only normally summon 1 card from the extra deck, but if it's a Link monsters, they can also make additional summons in the spaces that the Link Markers point to. I'm not sure if this actually did slow the game down, but it's certainly different from when I last played personally.
# 
# There is a big boost to Light monsters, mostly at the expense of Water and Wind. I don't know enough about this era of the game to guess as to why there are so many Light Link monsters.

# In[ ]:


#Link monsters
print('Link monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_link.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_link.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_link,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Link Monsters")


# # Type & Other Categories
# 
# Much like Attribute, we can also see how Type varies between these 15 categories. However, since there are far more possible Types than Attributes, it will probably be more difficult to spot underlying trends. I willinstead just focus on those which show a significant increase compared to the general population, or those which are notably absent from the sub-category. I expect we will have far more cases where there are gaps in the data, due to those monsters simply not existing (yet).

# In[ ]:


#Make a dictionary of value counts for each type, grouped by sub-categories
d={}
for i in range(len(type_list)):
    d["Att_{0}".format(type_list[i])]=YGO_df.groupby(type_list[i]).Type.value_counts()[1]

#Convert to a dataframe,and add the Total columns
Type_df=pd.DataFrame(d)
Type_df['Total']=YGO_df.Type.value_counts()
Type_df['Total_Perc']=round(Type_df.Total/sum(Type_df.Total)*100,1)

Type_df.fillna(0,inplace=True)

#Add a % column for each
for i in range(len(type_list)):
    Type_df["Perc_{0}".format(type_list[i])]=round(Type_df.iloc[:,i]/np.nansum(
        Type_df.iloc[:,i])*100,1)
    
#Add a change relative to Total, called Delta
for i in range(len(type_list)):
    Type_df["Delta_{0}".format(type_list[i])]=round(Type_df["Perc_{0}".format(type_list[i])]
    -Type_df['Total_Perc'],1)
    
Type_df


# When looking at the overall distribution, there is a clear top 5: Warrior, Machine, Fiend, Spellcaster and Dragon.
# 
# None of these are particularly surprising, since many themes (Hero, Six Samurai, Noble Knights) are Warriors, and many human-like cards are Warriors. There are also many machine based themes, and several important anime characters have used them. I knew Spellcaster and Dragon would be high up, but had expected them to be the top 2, with the conflict between the Types going all the way back to Dark Magician vs Blue Eyes White Dragon. Fiend rounds out this list as the general category for 'evil' cards.
# 
# Fairy and Beast are close behind, as the counterpart to Fiends, and an alternative to Warrior for more generic cards.
# 
# At the bottom, we have Types like Cyberse, Wyrm, Fish and Sea Serpent. The former two of which are understandable, due to being relatively new. Whereas the latter two have existed from the start of the game, and simply don't exist in huge numbers.

# In[ ]:


#Type overall
print('Overall Type distribution')
for i in range(len(Type_df.index)):
    print(str(Type_df.index[i])+': '+str(Type_df.Total_Perc.values[i])+'%')
    
#Attribute count
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Type_df.index,y=Type_df.Total)
_=g.set(xlabel='Type',ylabel='Frequency',title="Type Distribution of All Monsters")


# Normal monsters have a broadly speaking similar distribution to the full dataset, with most Types within 2% of the distribution in the full set. The only cases where this isn't true are Warrior, Dragon and Machine, which all show decreases in their share, and Aqua where their proportion of the data actually increases. In fact, Dragon even leaves the top 5 and is replaced by Aqua. In the former case, this is likely because they're important types, so have tended towards effect monsters for longer (Most Normals are from the start of the game). For Aqua, this is likely just because of lots of bad normal Aqua monsters in the first few sets. 

# In[ ]:


#Normal
print('Normal Type distribution (Change from Total)')
for i in range(len(Type_df.index)):
    print(str(Type_df.index[i])+': '+str(Type_df.Perc_is_normal.values[i])+'%'
         + ' ('+str(Type_df.Delta_is_normal.values[i])+'%)')
    
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Type_df.index,y=Type_df.Att_is_normal)
_=g.set(xlabel='Type',ylabel='Frequency',title="Type Distribution of Normal Monsters")


# Since most monsters are Effect monsters, it's unsurprising that the distribution of Types amongst them is nearly the same as the full monster set. The ordering remains the same, and most Types change their share of the overall distribution by <1%.

# In[ ]:


#Effect
print('Effect Type distribution (Change from Total)')
for i in range(len(Type_df.index)):
    print(str(Type_df.index[i])+': '+str(Type_df.Perc_is_effect.values[i])+'%'
         + ' ('+str(Type_df.Delta_is_effect.values[i])+'%)')
    
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Type_df.index,y=Type_df.Att_is_effect)
_=g.set(xlabel='Type',ylabel='Frequency',title="Type Distribution of Effect Monsters")


# As expected, several types are absent from the Flip data, but this is partly because the large number of historical Flip monsters have not been updated to this category. Beast-Warrior, Cyberse, Dinosaur, Fish, Psychic and Wyrm are all absent, with others like Aqua or Pyro only having 1 occurence. Most types have 2 or fewer occurences. Interestingly Spellcaster is now the most represented, followed by Insect.
# 
# At the very least, is_flip can be used to filter out certain Types from predictions.

# In[ ]:


#Flip
print('Flip Type distribution (Change from Total)')
for i in range(len(Type_df.index)):
    print(str(Type_df.index[i])+': '+str(Type_df.Perc_is_flip.values[i])+'%'
         + ' ('+str(Type_df.Delta_is_flip.values[i])+'%)')
    
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Type_df.index,y=Type_df.Att_is_flip)
_=g.set(xlabel='Type',ylabel='Frequency',title="Type Distribution of Flip Monsters")


# Gemini is a very rare category, so it comes as no surprise that many Types are absent (mostly the same as for Flip), or only occur once. Warrior, Fiend and Dragon remain in the Top 5 Types, joined by Aqua and a tie between Spellcaster and Zombie. As with Flip, this is probably best used to filter out certain Types.

# In[ ]:


#Gemini
print('Gemini Type distribution (Change from Total)')
for i in range(len(Type_df.index)):
    print(str(Type_df.index[i])+': '+str(Type_df.Perc_is_gemini.values[i])+'%'
         + ' ('+str(Type_df.Delta_is_gemini.values[i])+'%)')
    
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Type_df.index,y=Type_df.Att_is_gemini)
_=g.set(xlabel='Type',ylabel='Frequency',title="Type Distribution of Gemini Monsters")


# Most types absent. 50% machine. Fits with robot idea.
# More later!

# In[ ]:


#Union
print('Union Type distribution (Change from Total)')
for i in range(len(Type_df.index)):
    print(str(Type_df.index[i])+': '+str(Type_df.Perc_is_union.values[i])+'%'
         + ' ('+str(Type_df.Delta_is_union.values[i])+'%)')
    
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Type_df.index,y=Type_df.Att_is_union)
_=g.set(xlabel='Type',ylabel='Frequency',title="Type Distribution of Union Monsters")


# In[ ]:




