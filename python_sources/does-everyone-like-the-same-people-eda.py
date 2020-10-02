#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import widgets
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import HTML
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


#toggle code off and on (taken stackoverflow)
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')


# ## **WHO IS MOST SUCCESSFUL AT SPEED DATING?**
#  ![](https://img.evbuc.com/https%3A%2F%2Fcdn.evbuc.com%2Fimages%2F26423962%2F74297917129%2F1%2Foriginal.jpg?w=800&auto=compress&rect=0%2C40%2C300%2C150&s=4c47ad9c65f91faa35daf7a44b1bb878)
# # **Introduction**
# 
# This analysis is intended to look into the dynamics that drive decisions during speed dating events. Questions such as are women more successful than men? Are more attractive people more successful? Can we predict who will be most sucessful? are considered. This analysis also looks into whether all daters act similarly, or if there are different strategies at play?
# 
# # **Data Summary**
# 
# Data has been provided based on 21 speed dating nights run from 2002-2004. Participants had a series of dates lasting 4 minutes. They were then asked if they would like to see their date again, and asked to rate their date on some key attributes. 
#  
# These attributes were:
# * Attractiveness
# * Sincerity 
# * Intelligence
# * Fun 
# * Ambition
# * Shared Interests
# * How much did they like them overall
#  
# Note: All the code is hidden, but when shown should be in line with the relevant text.

# # **Information on Daters**
# 
# A questionnaire was given to every dater before the event asking for information on them and their interests. Below is a list of the information gathered, the name in brackets the name used in the data set.
# 
# * Gender 0-female, 1-male (gender)
# * Age (age)
# * Categorised field of study (field_cd)
# * Race (race)
# * Importance (out of 10) placed on having same race as match (imprace)
# * Importance (out of 10) placed on having the same religion as match (imprelig)
# * Goal of attending the speed dating event (goal)
# * How often does the dater go out 1-most,7-least (go_out)
# * How interested in the following categories the dater is out of 10
# * Shopping (shopping)
# * Playing sports/ athletics (sports)
# * Watching sports (tvsports)
# * Exercise (exercise)
# * Dining out (dining)
# * Museums/ galleries (museums)
# * Art (art)
# * Hiking/ Camping (camping)
# * Gaming (gaming)
# * Dancing/ Clubbing (clubbing)
# * Reading (reading)
# * TV (tv)
# * Theater (theater)
# * Movies (movies)
# * Concerts (concerts)
# * Music (music)
# * Shopping (shopping)
# * Yoga/ meditation (yoga)
# * How happy the daters expected to be with their dates out of 10 (exphappy)
# * How many matches the daters expected to receive (expnum)
# * How attractive they consider themselves to be out of 10 (attr3_1)
# * How sincere they consider themselves to be out of 10 (sinc3_1)
# * How fun they consider themselves to be out of 10 (fun3_1)
# * How intelligent they consider themselves to be out of 10 (intel3_1)
# * How fun they consider themselves to be out of 10 (amb3_1)

# In[ ]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
get_ipython().run_line_magic('matplotlib', 'inline')

#show more columns and rows by default
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#import data
data = pd.read_csv('../input/speed-dating-experiment/Speed Dating Data.csv',encoding="ISO-8859-1")
data.set_index('iid',inplace=True)
#data.head(1)


# In[ ]:


#create a table for each idd. which unchanging features (i.e. drop information related to individual dates)
iid_lookup = data[['gender','age','field_cd','race','imprace','imprelig','goal','date','go_out','sports','tvsports','exercise','dining','museums','art','hiking','gaming','clubbing','reading','tv','theater','movies','concerts','music','shopping','yoga','exphappy','expnum','attr3_1','sinc3_1','fun3_1','intel3_1','amb3_1']]
iid_lookup = iid_lookup.groupby('iid').mean() #have to group on some criteria, just taken mean because it's easy
#iid_lookup.head(1)


# # **Information on dates**
#  
# After each date the participants were asked to fill in a scorecard on each date. This involved the information listed below.
# 
# * Daters id (iid)
# * Partners id (pid)
# * Match? 1-match, 0-no match (match)
# * Dater's decision (dec)
# * Partner's decision (dec_o)
# * Wave of speed dating (wave
# * Calculated correlation between interests (int_corr)
# * Whether the couple are the same race (samerace)
# * Attraction score (out of 10) given to partner (attr)
# * Sincerity score (out of 10) given to partner (sinc)
# * Intelligence score (out of 10) given to partner (intel)
# * Fun score (out of 10) given to partner (fun)
# * Ambition score (out of 10) given to partner (amb)
# * Perception (out of 10) of shared interests with partner (shar)
# * Overall like score (out of 10) for partner (like)
# * Estimate of probability (out of 10) partner will say yes to them (prop)
# * Met before? 1=yes, 2=no (met)
# * Attraction score (out of 10) given by partner (attr_o)
# * Sincerity score (out of 10) given by partner (sinc_o)
# * Intelligence score (out of 10) given by partner (intel_o)
# * Fun score (out of 10) given by partner (fun_o)
# * Ambition score (out of 10) given by partner (amb_o)
# * Overall like score (out of 10) given by partner (like_o)
# * Partner's estimate of probability (out of 10) they will say yes (prop_o)
# * Met before as assessed by partner? 1=yes, 2=no (met_o)

# In[ ]:


#create a cut down table for each invididual date
list_of_dates = data[['pid','match','dec','dec_o','wave','int_corr','samerace','attr','sinc','intel','fun','amb','shar','like','prob','met','attr_o','sinc_o','intel_o','fun_o','amb_o','shar_o','like_o','prob_o','met_o']].copy()
#list_of_dates.head(1)
#list_of_dates


# In[ ]:


#add a new column summerising the sucess of each date
list_of_dates['decs'] = 2*list_of_dates['dec_o']+1*list_of_dates['dec']
#create colormap (1=neither positive,2=they positive only, 3=partner postive only, 4=match)
cust = ["#E0E0E0","#E0E0E0","#404040", "#FF0000"]
my_cmap = ListedColormap(sns.color_palette(cust).as_hex())


# In[ ]:


#create a heatmap summerising how each date went
waves=[]
for i in range(21):
    #create a cutdown table for each wave
    date_decision = list_of_dates[list_of_dates['wave']==(i+1)][['pid','decs']]
    #rearrange the table to the right format
    wave = pd.pivot_table(date_decision,values='decs',index=['iid'],columns=['pid'])
    waves.append(wave)


# # **How each date night went**
# 
# The graphic below summarises the success of each date night. You can use the slider to look through each wave (set of dates). 
# 
# Throughout this analysis we are going to call the person we are focusing on the dater, and their date the partner.
# 
# In the graphic below red symbolises a match (i.e. they both wanted to meet again), and black symbolises the dater wanted to meet again, however the partner didn't.
# 
# The men are making the decisions in the top right hand box, and the women in the bottom left.
# 
# IID is a unique identifier for each dater.

# In[ ]:


#create a slider running through the different waves, display as heatmap.
def wave_display(i):
    sns.heatmap(waves[(i-1)],cmap=my_cmap,cbar=False, square=True)
    plt.xlabel('IID decision making partner')
    plt.ylabel('IID decision recieving partner')
    plt.title('wave {}'.format(i))
interact(wave_display,i=(widgets.IntSlider(value=9,description='Wave Num.',min=1,max=21,step=1,continuous_update=False)));


# We are going to take wave 9 as an example, and look at information dynamics that stand out:
# 
# * It seems men want to meet again more often that women (there are a lot more blacks in the top right than bottom left).
# * Dater 208 was the most popular dater; every man wanted to meet up with her again.
# * 13 women had more than 50 percent of their partners want to meet them again, whereas only 4 men were that popular.
# 
# So it appears from this snapshot, that women are more fussy, but also more popular. Next we will look if this is true overall.

# In[ ]:


#find the average of how each data did on their dates.
daters_means = list_of_dates.groupby('iid').mean()
daters_means.drop(['pid','samerace','int_corr','decs'],axis=1,inplace=True) #meaningless here
#daters_means.head(1)


# In[ ]:


#create a table for followup info
follow_up = data[['you_call','them_cal','date_3','numdat_3','num_in_3']]
follow_up = follow_up.groupby('iid').mean()
#follow_up.head(1)


# In[ ]:


#join all the tables together again. This is so all the information on each dater is together
joined = iid_lookup.join(daters_means)
#joined.head(20)


# # **Analysis of daters decisions**
# 
# The graphs below show the distribution of success for each dater. For this analysis, we have defined successful as the fraction of their dates that would like to see them again.
# 
# Intersting observations from these graphs:
# * The median for number of positive decisions received is 40%. 
# * More women receiving a higher number of positive results than men. 
# * The woman mentioned earlier (from wave 9) was the only person who had 100% success with their dates. 
# * Quite a few daters received no positive decisions, more of these were men, which again suggests women have been more successful.
# 
# This means the trend has continued, and women are more successful overall than men during these speed dating events.
# 

# In[ ]:


#plot figures showing how daters did overall and seperated by sex. This is for the dec_o column, which is decision made about person.
plt.figure(figsize=(15,5))

#total
plt.subplot(1,3,1)
sns.distplot(joined['dec_o'],kde=False,bins=10)
plt.xlabel('fraction of positive decisions recieved total')
plt.ylabel('number of people total')
plt.tight_layout()
plt.ylim(0,110)

#women
plt.subplot(1,3,2)
sns.distplot(joined[joined['gender']==0]['dec_o'],label='women',kde=False,bins=10,hist_kws=dict(alpha=0.3),color='red')
plt.xlabel('fraction of positive decisions recieved by women ')
plt.ylabel('number of women')
plt.tight_layout()
plt.ylim(0,110)

plt.subplot(1,3,3)
sns.distplot(joined[joined['gender']==1]['dec_o'],label='men',kde=False,bins=10,hist_kws=dict(alpha=0.3),color='green')
#plt.legend()
plt.xlabel('fraction of positive decisions recieved by men')
plt.ylabel('number of men')
plt.ylim(0,110);


# This next set of graphs show the fraction of their partners each dater wants to see again.
# 
# Key insights:
# * Women are pickier and want to see a smaller fraction of their dates again than men. 
# * A lot more women don't want to see any of their dates again than men.
# 
# The differences between success of men and women is likely stongly related to the fact that men want to meet more of their dates again than women.

# In[ ]:


#plot figures showing how daters did overall and seperated by sex. This is for the dec column, which is decision made by the person.
plt.figure(figsize=(15,5))

#total
plt.subplot(1,3,1)
sns.distplot(joined['dec'],kde=False,bins=10)
plt.xlabel('fraction of positive decicions given')
plt.ylabel('number of people total')
plt.tight_layout()
plt.ylim(0,100)

#women
plt.subplot(1,3,2)
sns.distplot(joined[joined['gender']==0]['dec'],label='women',kde=False,bins=10,hist_kws=dict(alpha=0.3),color='red')
plt.xlabel('fraction of positive decisions given by women')
plt.ylabel('number of women')
plt.tight_layout()
plt.ylim(0,100)

#men
plt.subplot(1,3,3)
sns.distplot(joined[joined['gender']==1]['dec'],label='men',kde=False,bins=10,hist_kws=dict(alpha=0.3),color='green')
#plt.legend()
plt.xlabel('fraction of positive decisions given by men')
plt.ylabel('number of women')
plt.ylim(0,100);


# In[ ]:


list_of_dates.groupby('dec').mean()['dec_o'];


# These next graphs show the fraction of dates that result in a match for each person.
# 
# Key insights:
# * It a lot of people receive no matches (20%). This means that even though most people want to see at least one date (>95%) again, and most people are chosen to be seen again (0.96%), these do nessisarily align.
# * For the people who want to see their date again, 39% of their partners want to see them again, but for those who don't want to see their date again 44% of their partners do. It seems liking your date makes them less likely to like you.
# * The median number of matches is 1 in 9, and the distributions for men and women are very similar. 
# * There is an interesting peak for women at 0.45 of their dates resulting in a match. 
# * The person with the most matches was a woman (IID=19) with 0.9 of her dates resulting in a match. She wanted to see all of her partners again, which means one of her dates was not successful. So while she got the most matches, she was not the most desired women overall.

# In[ ]:


#graphs showing distribution of matches
plt.figure(figsize=(15,5))

#total
plt.subplot(1,3,1)
sns.distplot(joined['match'],kde=False,bins=10)
plt.xlabel('fraction of matches total')
plt.ylabel('number of people total')
plt.tight_layout()
plt.ylim(0,175)


plt.subplot(1,3,2)
sns.distplot(joined[joined['gender']==0]['match'],label='women',kde=False,bins=10,hist_kws=dict(alpha=0.3),color='red')
plt.xlabel('fraction of matches for women ')
plt.ylabel('number of women')
plt.tight_layout()
plt.ylim(0,175)

plt.subplot(1,3,3)
sns.distplot(joined[joined['gender']==1]['match'],label='men',kde=False,bins=10,hist_kws=dict(alpha=0.3),color='green')
#plt.legend()
plt.xlabel('fraction of matches for men')
plt.ylabel('number of men')
plt.ylim(0,175);


# # **Does self rating match partner rating?**
# 
# The daters were rated by their partners, but also rated themselves in the initial questionaire. First we are going to look if there is a correlation between the self ratings and ratings given. The graphs below are a scatter graph showing self rating vs average partner rating for attraction, sincerity, fun, intelligence and ambition. The colour relates to how many of that person's dates wanted to see them again; darker means the dater was more successful.
# 
# Key insights:
# * From the figures below we can see there is less correlation between self ratings and ratings by partners than you might expect. 
# * The most correlated is attraction, with a correlation coefficient of 0.29. From the graph there's quite a spread in ratings. For instance, some people that rate themselves 8 only got an average of 3 from their dates. One person who rated themselves 4 was rated an average of 8 by their dates. We can see the most successful people were judged most attractive as is expected. 
# * The ratings for fun follow a similar pattern as attraction.
# * The sincerity rating has no correlation at all between self rating and average rating, this may be because it's a hard thing to judge in a short amount of time. Some people who rated themselves 2 actually got rated around 7.5 by their partners. 
# * There is also no correlation between self rating on intelligence and rating by partners, this is probably for similar reasons.
# 

# In[ ]:


corr = joined.corr()


# In[ ]:


plt.figure(figsize=(20,5));
plt.subplot(1,5,1);
sns.scatterplot(joined['attr3_1'],joined['attr_o'],hue=joined['dec_o'],cmap='coolwarm',s=50,legend=False);
plt.xlabel('self rated');
plt.ylabel('average rating by partner');
plt.title('attraction corr= {:0.2f}'.format(corr['attr3_1']['attr_o']));
plt.tight_layout();
plt.subplot(1,5,2);
sns.scatterplot(joined['sinc3_1'],joined['sinc_o'],hue=joined['dec_o'],cmap='coolwarm',s=50,legend=False);
plt.xlabel('self rated');
plt.ylabel('average rating by partner');
plt.title('sinserity corr= {:0.2f}'.format(corr['sinc3_1']['sinc_o']));
plt.tight_layout();
plt.subplot(1,5,3);
sns.scatterplot(joined['fun3_1'],joined['fun_o'],hue=joined['dec_o'],cmap='coolwarm',s=50,legend=False);
plt.xlabel('self rated');
plt.ylabel('average rating by partner');
plt.title('fun corr= {:0.2f}'.format(corr['fun3_1']['fun_o']));
plt.tight_layout();
plt.subplot(1,5,4);
sns.scatterplot(joined['intel3_1'],joined['intel_o'],hue=joined['dec_o'],cmap='coolwarm',s=50,legend=False);
plt.xlabel('self rated');
plt.ylabel('average rating by partner');
plt.title('intellegence corr= {:0.2f}'.format(corr['intel3_1']['intel_o']));
plt.tight_layout();
plt.subplot(1,5,5);
sns.scatterplot(joined['amb3_1'],joined['amb_o'],hue=joined['dec_o'],cmap='Wistia',s=50,legend=False);
plt.xlabel('self rated');
plt.ylabel('average rating by partner');
plt.title('ambition corr= {:0.2f}'.format(corr['amb3_1']['amb_o']));
plt.tight_layout();


# ## **Which attributes predict success?**
# As we said earlier, are going to use the fraction of daters' partners that wanted to see them again to define success in speed dating. 
# 
# 
# The sidewards scrolling table below shows the corrolation statistic; it's sorted so the most correlated comes first. Some intersting attributes are then plotted below to visualise this corrolation. The points on the graph are coloured by sex. As we have seen above, men and women approach speed dating differently, so this is an important aspect to investigate further.
# 
# Key insights:
# * The most correlated attribute to success is a high average attraction score given by partners (0.79). This is to be expected, as people are likely to find the people they want to see again attractive. Also more attractive people are likely to give a better first impression, which is important in speed dating.
# 
# * Self rated attraction score is only correlated at 0.19, which is much lower, but still slightly correlated. From the graphs below we can see that people who rated themselves lower than 5 in attractiveness are less successful in speed dating, but above this there does not appear to be a strong pattern. We saw above that self rated attraction is not necessarily related to how their dates perceived them.
# 
# * The most corrolated attribute not rated by the partners is the expected number of daters the partner expected to get. This is interesting because it suggests that the daters themselves knew objectively how popular they would be at speed dating, this is better correlated than self rated attraction. (0.24)
# 
# * As we saw before, gender is correlated with how many of their partners want to see them again. Men want to see more women again, than women want to see men. (-0.22)
# 
# * The fraction of their partners the dater themselves want to see again is inversely correlated with the fraction of partners who want to meet them again (-0.26). We touched on this earlier. It is likely this is largely a consequence of the gender differences, however the correlation is higher than the correlation for gender.
# 
# * How often the dater goes out is inversely correlated with dating success.However, given that the lowest number represents goes out most, this means the people who go out more often have more dating success.(-0.23)

# In[ ]:


predictive_atts = corr['dec_o'].sort_values(ascending=False)
pd.DataFrame(predictive_atts[abs(predictive_atts)>0.1]).T.round(2)


# In[ ]:


plt.figure(figsize=(20,5));
plt.subplot(1,4,1);
sns.scatterplot(joined['attr_o'],joined['dec_o'],hue=joined['gender']);
plt.legend(['female','male']);
plt.title('average attraction given by partners = {:0.2f}'.format(corr['attr_o']['dec_o']));
plt.tight_layout();
plt.subplot(1,4,2);
sns.scatterplot(joined['attr3_1'],joined['dec_o'],hue=joined['gender']);
plt.legend(['female','male']);
plt.title('self rated attraction corrolation = {:0.2f}'.format(corr['attr3_1']['dec_o']));
plt.tight_layout();
plt.subplot(1,4,3);
sns.scatterplot(joined['expnum'],joined['dec_o'],hue=joined['gender']);
plt.legend(['female','male']);
plt.title('expected number of matches corrolation = {:0.2f}'.format(corr['expnum']['dec_o']));
plt.tight_layout();
plt.subplot(1,4,4);
sns.scatterplot(joined['exercise'],joined['dec_o'],hue=joined['gender']);
plt.legend(['female','male']);
plt.title('exercise interest corrolation = {:0.2f}'.format(corr['expnum']['dec_o']));
plt.tight_layout();


# # **Which Attributes correlated with fraction of partners daters want to see again?**
# 
# As we saw earlier some daters want to meet most of their dates and some want to meet none. This could be due to the quality of their dates, but is more likely due to the characteristics of the dater themselves. This section looks at which attributes might lead to picky daters.
# 
# The table below shows the highest correlated attributes.The first few are average ratings of their dates, so we would expect these to be high in daters that would like to see most of their dates again, and low in daters that don't.
# 
# Expected number of dates is correlated with this (0.27), just as it was with the partners decisions. This is interesting, and shows the dater has some insight into how many people they are likely to like. There is also a correlation between the average probability they think their dates will want to see them again (0.24), which shows an expectation that the people they like will like them. 
# 
# Gaming is corrolated (0.15), but this is likely to be because of the gender differences in interest in gaming. We will look into this more later.
# 

# In[ ]:


predictive_atts = corr['dec'].sort_values(ascending=False)
pd.DataFrame(predictive_atts[abs(predictive_atts)>0.1]).T.round(2)


# # ** Which attributes are more correlated with gender than speed dating?**
# 
# As we have seen above, some attributes may appear to be correlated with speed dating characteristics, however they are more correlated with gender, which itself affects how people approach speed dating.
# 
# * Below are the attributes most correlated with gender, perhaps predictably the most correlated is sports (0.25); other typically male pastimes also have some correlation, gaming (0.23) and tv sports (0.15). Negatively correlated are the more typical female interests of shopping (-0.33), theater (-0.33), yoga (-0.24).
# 
# * More relevant to the speed dating events there is some correlation between gender and how attractive the dater rates their partners (0.24), and the fraction of dates they want to see again (0.21). 
# * We can see that men also expect to be happier with their dates than women (0.19). 
# * Women put less importance on having the same religion (-0.20) and race as their partner (-0.13).
# 

# In[ ]:


predictive_atts = corr['gender'].sort_values(ascending=False)
pd.DataFrame(predictive_atts[abs(predictive_atts)>0.1]).T.round(2)


# # ** Which attributes are correlated?**
# 
# This section is going to look at what attributes overall are most correlated and what information does this tell us about the dates and daters.
# 
# * The most correlated attributes are art and museums (0.87), this means people who like museums are more likely to art and people who don't like museums are less likely to like art.
# 
# * The next most correlated is average fun and like scores as rated by partners (0.84). This means people who are seen more fun on average are more likely to be liked by their partners. 
# * The average attraction score is also highly correlated with these, as is the fraction of dates that were successful. This is all to be expected; people who like their dates, find them fun and attractive want to see them again. 
# * What is less expected is how strongly correlated the perception of shared interests was with all of these attributes. Are people who are more fun more likely to share interests with their dates? Or is this just a perception because they like their date?
# 
# * The expected number of dates and self rated attractiveness are correlated with a measure of 0.51, this is interesting. Some of this is likely to be true impact of being an attractive person who has a history of romantic success, but it also could be that self rating on both of these measures is highly dependent on self confidence.
# 
# * The first time there is a correlation between self rating and partner rating is attraction at 0.29, which we have already discussed above.

# In[ ]:


#find the attributes that are most corrolated 

best_corrs=[]
#run through attributes in corrolation matrix
for ind in list(corr.index.values):
    for i in range(20):
    #sort the column for this atribute by the corrolation and take the top value (0 position always==1)
        att, cor =corr.sort_values(by=ind,ascending=False)[ind].iloc[[(i+1)]].to_string().split()
        best_corrs.append([ind,att,cor])

    
#sort final list by corolation 
best_corrs.sort(key=lambda x:x[2],reverse=True)
#take top 40 values (note each row is dublicated so only take every other row)
best_corrs = pd.DataFrame(best_corrs).round(2)
(best_corrs[0:600:2]).T.round(2)


# In[ ]:


plt.figure(figsize=(20,5));
plt.subplot(1,4,1);
sns.scatterplot(joined['attr_o'],joined['like_o'],hue=joined['gender']);
plt.legend(['female','male']);
plt.title((('average attraction and like scores given by partners \n, {:0.2f}')).format(corr['attr_o']['like_o']));
plt.tight_layout();
plt.subplot(1,4,2);
sns.scatterplot(joined['attr_o'],joined['fun_o'],hue=joined['gender']);
plt.legend(['female','male']);
plt.title('average attraction and fun scores given by partners,\n {:0.2f}'.format(corr['attr_o']['fun_o']));
plt.tight_layout();
plt.subplot(1,4,3);
sns.scatterplot(joined['fun_o'],joined['shar_o'],hue=joined['gender']);
plt.legend(['female','male']);
plt.title('average fun and shared interest scores given \n by partners, {:0.2f}'.format(corr['fun_o']['shar_o']));
plt.tight_layout();
plt.subplot(1,4,4);
sns.scatterplot(joined['expnum'],joined['attr3_1'],hue=joined['gender']);
plt.legend(['female','male']);
plt.title('expected number of matches and self rated \n attractiveness, {:0.2f}'.format(corr['expnum']['attr3_1']));
plt.tight_layout();


# # ** Can we predict which people will be most successful?**
# 
# In this section, we are going to see if we can predict the success of each dater. As we said earlier, we have defined this as the fraction of partners who want to see the dater again.
# 
# First we will try using all the information, including average attractive scores given by partners. This should be relatively easy as it includes how much their partners like them and find them attractive on average.
# 
# Then we will just use the information given on the initial questionaire. This will be a much tougher challenge. This uses information on interests and self ratings, as well as age and gender.
# 
# (running the models has created an output that I haven't managed to hide. So just ignore the black boxes)

# I have split the data into a training and test set. The training set trains the neural network and the test set is used to analyse the success of the model. The test set is 10% of the total data set (53 daters).

# In[ ]:


#lose nas. unfortunetly have to lose expnum column as a lot left this blank
joined_na=joined.drop(['expnum'],axis=1)
joined_na.dropna(inplace=True)


# In[ ]:


#set up data
#inputs
X = joined_na[['gender','age','field_cd','race','imprace','imprelig','goal','date','go_out','sports','tvsports','exercise','dining','museums','art','hiking','gaming','clubbing','reading','tv','theater','movies','concerts','music','shopping','yoga','exphappy','wave','attr3_1','sinc3_1','intel3_1','fun3_1','amb3_1']]
#outputs
Y = joined_na['dec_o']
#split data 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,shuffle=False)


# In[ ]:


#scale data, so no one variable skews fitting
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#scaler.fit(X_train)
#X_train = pd.DataFrame(data=scaler.transform(X_train))
#X_test = pd.DataFrame(data=scaler.transform(X_test),columns=X_test.columns)


# In[ ]:


import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
#create feature columns loop taken from stackover flow
my_columns=[]
import pandas.api.types as ptypes

feat_cols = []

for col in X_train.columns:
  if ptypes.is_string_dtype(X_train[col]): #is_string_dtype is pandas function
    feat_cols.append(tf.feature_column.categorical_column_with_hash_bucket(col,                            
       hash_bucket_size= len(X_train[col].unique())))
  elif ptypes.is_numeric_dtype(X_train[col]): #is_numeric_dtype is pandas function

    feat_cols.append(tf.feature_column.numeric_column(col))


# In[ ]:


#set up tensorflow
#define input, and output
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=Y_train,batch_size=10,num_epochs=100,shuffle=False)
#set up estimator
model = tf.estimator.DNNRegressor(hidden_units=[10,10,10],feature_columns=feat_cols)
#train
model.train(input_fn=input_func,steps=2500)
#predcit function
predict_input_func = tf.estimator.inputs.pandas_input_fn(
x=X_test,batch_size=10,num_epochs=1,shuffle=False)
#get outputs
pred_gen= model.predict(predict_input_func)
predictions = list(pred_gen)
final_preds =[]
for pred in predictions:
    final_preds.append(pred['predictions'])

#measure error
from sklearn.metrics import mean_squared_error 
#mean_squared_error(Y_test,final_preds)**0.5


# In[ ]:


#plt.plot(range(0,len(Y_test)),Y_test)
#plt.plot(final_preds)


# In[ ]:


#set up data
#inputs
X = joined_na.drop(['match','dec','dec_o'],axis=1)
#outputs
Y = joined_na['dec_o']
#split data 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,shuffle=False)


# In[ ]:


import tensorflow as tf
#create feature columns loop taken from stackover flow
my_columns=[]
import pandas.api.types as ptypes

feat_cols = []

for col in X_train.columns:
  if ptypes.is_string_dtype(X_train[col]): #is_string_dtype is pandas function
    feat_cols.append(tf.feature_column.categorical_column_with_hash_bucket(col,                            
       hash_bucket_size= len(X_train[col].unique())))
  elif ptypes.is_numeric_dtype(X_train[col]): #is_numeric_dtype is pandas function
    feat_cols.append(tf.feature_column.numeric_column(col))


# In[ ]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=Y_train,batch_size=10,num_epochs=100,shuffle=False)
model = tf.estimator.DNNRegressor(hidden_units=[10,10,10,10],feature_columns=feat_cols)
model.train(input_fn=input_func,steps=2500)
predict_input_func = tf.estimator.inputs.pandas_input_fn(
x=X_test,batch_size=10,num_epochs=1,shuffle=False)
pred_gen= model.predict(predict_input_func)
predictions = list(pred_gen)

final_preds_2 =[]
for pred in predictions:
    final_preds_2.append(pred['predictions'])

from sklearn.metrics import mean_squared_error 
#mean_squared_error(Y_test,final_preds_2)**0.5


# This graphs below show the difference in predictions between the two data sets, the success of 53 daters has been predicted. 
# 
# * The top graph shows although there is room for improvement, a reasonable job has been done predicting the success when the ratings given by partners are fed in. 
# * The second graph shows the prediction of dating success when no partner ratings are given; the neural networks seems to be predicting around 0.5 for each dater, I think this highlights the predictions are not much better than chance.
# 
# But these graphs have the most corrolated variable avaliable to the predictor also plotted. For the top graph this is partner rated attraction, for the bottom graph it's self rated attraction. 

# In[ ]:


#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#init_notebook_mode(connected=True) 
plt.subplot(2,1,1)
plt.plot(range(0,len(Y_test)),Y_test,label='actual success',c='black')
plt.plot(final_preds_2,label='including partner rating')
plt.plot(range(0,len(Y_test)),X_test['attr_o']/10,ls='--',lw=0.4,label='partner rated attraction')
plt.xlabel('dater')
plt.legend(bbox_to_anchor=(1,1))

plt.subplot(2,1,2)
plt.plot(range(0,len(Y_test)),Y_test,label='actual success',c='black')
plt.plot(final_preds,label='excluding partner rating')
plt.plot(range(0,len(Y_test)),X_test['attr3_1']/10,ls='--',lw=0.4,label='self rated attraction')
plt.ylabel('fraction of partners who want to see them again')
plt.legend(bbox_to_anchor=(1,1))
plt.xlabel('dater');


# # ** Do the least successful dates like each other or the most successful daters?**
# 
# One important question left outstanding is whether everyone like the same successful daters, or if some of the least successful daters are drawn together. We often think of a spark as a magical connection between two well suited individuals. This section is going to look at whether there seems to be any truth to this, at least in the context of speed dating.
# 
# The graphic below is a repeat of the graphic from the beginning of this analysis. Matches are shown in red, and one way positive decisions are shown in black. In this case the daters are sorted by success, with the most successful at the top, and left. Again we are going to focus on wave 9 for analysis.
# 
# * We can see that all the postive decisions are skewed towards the top. This implies daters typically like the most successful partners. 
# * The matches are skewed towards the top right, which means the most successful people seem to like each other.
# * The top left is black, which means the least successful people also like the most successful people, but they are not typically liked back.
# 
# This implies there is no magic spark. Everyone is likes simular people in speed dating. 

# In[ ]:


wave_sort=[]
for i in range(21):
    temp_wave_sort = (waves[i].join(joined['dec_o']).sort_values(by='dec_o',ascending=False).T.join(joined['dec_o']).sort_values(by='dec_o',ascending=False).T)
    wave_sort.append(temp_wave_sort)
#wave_sort[0]


# In[ ]:


def wave_display_2(i):
    sns.heatmap(wave_sort[(i-1)],cmap=my_cmap,cbar=False, square=True)
    plt.xlabel('IID decision making partner')
    plt.ylabel('IID decision recieving partner')
    plt.title('wave {}'.format(i))
interact(wave_display_2,i=(widgets.IntSlider(value=9,description='Wave Num.',min=1,max=21,step=1,continuous_update=False)));


# In[ ]:


#add the overall sucess of each dater to the dates list
success_table=daters_means[['dec_o']]
success_table.rename({'dec_o':'success'},axis=1,inplace=True)
dates=list_of_dates[['pid','match','dec']]
dates=dates.join(success_table)
dates.rename({'success':'suc'},axis=1,inplace=True)
dates=dates.join(success_table,on='pid')
dates.rename({'success':'suc_o'},axis=1,inplace=True)


# Below we have plotted a data point for each date. It shows the daters success score, vs their parners and is coloured by the outcome for each date.
# 
# * The first graph is coloured by when the dater wants to meet again, orange is a postive decision. We can see no matter the dater success level the most successful partners have orange dots. We have over 8000 dates,so this is not true for every single date, but it holds true overall.

# In[ ]:


plt.figure(figsize=(20,5));
plt.subplot(1,3,1)
sns.scatterplot(dates['suc'],dates['suc_o'],hue=dates['dec'])
plt.title('Daters decision, shown for overall successfulness of each partner');
plt.ylabel('Partners overall successfulness')
plt.xlabel('Daters overall successfulness')
plt.tight_layout();
plt.legend(title='Dater wants to see again',title_fontsize='large',frameon=True,facecolor='white',framealpha=1.0,loc=5,ncol=4,borderpad=0.1,shadow=True)
plt.subplot(1,3,2)
sns.scatterplot(dates['suc'],dates['suc_o'],hue=dates['match'])
plt.legend(['no match','match']);
plt.title('Match, shown for overall successfulness of each partner');
plt.ylabel('Partners overall successfulness')
plt.xlabel('Daters overall successfulness')
plt.tight_layout();
plt.legend(title='Match',frameon=True,title_fontsize='large',facecolor='white',framealpha=1.0,loc=5,ncol=4,borderpad=0.1,shadow=True);
#plt.subplot(1,3,3)
#sns.scatterplot(dates[dates['dec']==1]['suc'],dates[dates['dec']==1]['suc_o'],hue=dates['gender'])
#plt.legend(['no match','match']);
#plt.title('Match, shown for overall successfulness of each partner');
#plt.ylabel('Partners overall successfulness')
#plt.xlabel('Daters overall successfulness')
#plt.tight_layout();
#plt.legend(title='Match',frameon=True,title_fontsize='large',facecolor='white',framealpha=1.0,loc=5,ncol=4,borderpad=0.1,shadow=True);
plt.subplot(1,3,3)
sns.scatterplot(dates[dates['match']==1]['suc'],dates[dates['match']==1]['suc_o'])
#b= sns.scatterplot(dates[dates['dec']==1]['suc'],dates[dates['dec']==1]['suc_o'])
#plt.legend(['not see again','see again']);
#plt.title('Matches');
#plt.ylabel('Partners overall successfulness')
#plt.xlabel('Daters overall successfulness')
#plt.tight_layout();
#plt.legend(title='Partner wants to see again',title_fontsize='x-large',frameon=True,facecolor='white',framealpha=1.0,loc=5,ncol=4,borderpad=0.1,shadow=True);


# The figure below shows the corrolation between these variables. This has been shown to look at the corrolation between the daters decision, their success overall and their partners success overall.
# 
# We can see there is a corrolation of 0.47 between the daters decision and partners successfulness.

# In[ ]:


sns.heatmap(dates.corr(),annot=True,linecolor='white',linewidths=2.0,square=True,mask=np.triu(np.ones(5)).astype(np.bool),cmap='Reds');


# ## **Conclusions**
# We have learnt many things from this speed dating data.
# Some key things:
# * Men want to meet a higher proportion of their dates again.
# * This leads to women being more successful overall (success if defined by what fraction of partners want to meet again).
# * The best predictor of dating success is attraction score as rated by partners. Which is probably to be exoected. 
# * Self rated attraction is a much lower predictor of dating success. 
# * The best self rated predictor of success is how many matches the dater excepts to get; people who exepct to get more matches are more successful. 
# 
# (Sorry I need to finish this off)
# 

# 

# In[ ]:




