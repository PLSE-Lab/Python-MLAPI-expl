#!/usr/bin/env python
# coding: utf-8

#  **Hello!** I've been studying Data Science for a while now, and this is my first Kernel submission. In it, I'll make a brief analysis of the data given, developing parallels between the worldwide values, and values for Brazil (where I was born). **Feedback and commentaries are most welcome!**
#   
#  First of all, I'll import the default libraries, and create a subroutine called *cleanOuterAxis*, that will make the visualizations cleaner.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

def cleanOuterAxis(outerAxis):
    plt.subplots_adjust(top=0.85,hspace=0.6)
    outerAxis.spines['top'].set_color('none')
    outerAxis.spines['bottom'].set_color('none')
    outerAxis.spines['left'].set_color('none')
    outerAxis.spines['right'].set_color('none')
    outerAxis.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


#  Now, let us create a object for holding the worldwide data, and another one for holding only Brazil's info. Also, let's see their beginnings.

# In[ ]:


worldData=pd.read_csv('../input/master.csv')
brazilData=worldData[worldData['country']=='Brazil']


# In[ ]:


worldData.head()


# In[ ]:


brazilData.head()


#  The first thing I wanted to look at was at which age most suicides happen, and also, how it changes according to each generation. By itself this information might be interesting enough, however, there is a motivation for knowing such value, which I will explain soon enough.
#  
#  Here are the results, beginning with worldwide data, and then with Brazil data:

# In[ ]:


fig=plt.figure(figsize=(16, 20))
outerAxis=fig.add_subplot(1,1,1)
axisArray=[fig.add_subplot(4,2,1),fig.add_subplot(4,2,2),
            fig.add_subplot(4,2,3),fig.add_subplot(4,2,4),
            fig.add_subplot(4,2,5),fig.add_subplot(4,2,6),
            fig.add_subplot(4,2,(7,8))]

outerAxis.set_xlabel('Age interval',fontsize=18); outerAxis.xaxis.set_label_coords(0.5,-0.2)
outerAxis.set_ylabel('Number of suicides',fontsize=18); outerAxis.yaxis.set_label_coords(-0.075,0.5)
fig.suptitle("Number of suicides according to age and generation for worldwide data", fontsize=18)
order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']
hueOrder=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z']

i=0
for generation in hueOrder: 
    sns.barplot(data=worldData[worldData['generation']==generation].groupby([worldData['age']]).sum().reset_index(level=['age']),
                x='age',
                y='suicides_no',
                #hue='generation',
                ax=axisArray[i],
                order=order,
                hue_order=hueOrder,
                ci=None)
    axisArray[i].set_title(str(generation), fontsize=16)
    i+=1

sns.barplot(data=worldData['suicides_no'].groupby([worldData['age'],worldData['generation']]).sum().reset_index(level=['generation','age']),
            x='age',
            y='suicides_no',
            hue='generation',
            ax=axisArray[6],
            order=order,
            hue_order=hueOrder,
            ci=None)
axisArray[6].set_title("World data", fontsize=16)

for axis in axisArray:
    axis.set_xlabel('')
    axis.set_ylabel('')
    sns.despine(ax=axis)
 
cleanOuterAxis(outerAxis)


# In[ ]:


fig=plt.figure(figsize=(16, 20))
outerAxis=fig.add_subplot(1,1,1)
axisArray=[fig.add_subplot(4,2,1),fig.add_subplot(4,2,2),
            fig.add_subplot(4,2,3),fig.add_subplot(4,2,4),
            fig.add_subplot(4,2,5),fig.add_subplot(4,2,6),
            fig.add_subplot(4,2,(7,8))]

outerAxis.set_xlabel('Age interval',fontsize=18); outerAxis.xaxis.set_label_coords(0.5,-0.2)
outerAxis.set_ylabel('Number of suicides',fontsize=18); outerAxis.yaxis.set_label_coords(-0.075,0.5)
fig.suptitle("Number of suicides according to age and generation for Brazil data", fontsize=18)
order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']
hueOrder=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z']

i=0
for generation in hueOrder: 
    sns.barplot(data=brazilData[brazilData['generation']==generation].groupby([brazilData['age']]).sum().reset_index(level=['age']),
                x='age',
                y='suicides_no',
                #hue='generation',
                ax=axisArray[i],
                order=order,
                hue_order=hueOrder,
                ci=None)
    axisArray[i].set_title(str(generation), fontsize=16)
    i+=1

sns.barplot(data=brazilData['suicides_no'].groupby([brazilData['age'],brazilData['generation']]).sum().reset_index(level=['generation','age']),
            x='age',
            y='suicides_no',
            hue='generation',
            ax=axisArray[6],
            order=order,
            hue_order=hueOrder,
            ci=None)
axisArray[6].set_title("World data", fontsize=16)

for axis in axisArray:
    axis.set_xlabel('')
    axis.set_ylabel('')
    sns.despine(ax=axis)
 
cleanOuterAxis(outerAxis)


#  The first thing to notice is that our the suicide data is skewed with some generations: older generations, such as the G.I. and Silent Generations doesn't have any data regarding suicides that happened with people with 35 years old or less (alternatively, we may believe that within these generations younger people just didn't commit suicide, but I find that *very* unlikely). Also, younger generations such as the Millenials and Generation Z don't have data concerning people with 35 years old or more, since no people of such generations have gotten to such age yet.
#  
#  Then, our best bet at guessing the age range where suicides are most common (without some major bias) are the Boomer and X generations. For them, suicides happen most commonly between 35 and 54 years. This also is valid for Brazil only data, as can be seen by analyzing the visualization for Brazil only data. Henceforth, let's call such interval the "Critical Range".
#  
#  As I said before, there is a motivation behind knowing such interval: knowing that most suicides happen after 35 years of age, we can analyze the total number of suicides for each generation, and then infer how that number might change in the future.
#  
#   Using the following visualization we can proceed with our study.

# In[ ]:


fig=plt.figure(figsize=(16, 9))
outerAxis=fig.add_subplot(1,1,1)
outerAxis.set_xlabel('Generations (from oldest to newest)',fontsize=18); outerAxis.xaxis.set_label_coords(0.5,-0.2)
outerAxis.set_ylabel('Number of suicides',fontsize=18); outerAxis.yaxis.set_label_coords(-0.075,0.5)
axisArray=[fig.add_subplot(2,2,1),fig.add_subplot(2,2,2),fig.add_subplot(2,2,(3,4))]
fig.suptitle("Suicide rate according to generation", fontsize=18)

order=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z']

sns.barplot(data=brazilData['suicides_no'].groupby(brazilData['generation']).sum().reset_index(level='generation'),
            x='generation',
            y='suicides_no',
            ax=axisArray[0],
            order=order,
            ci=None)
axisArray[0].set_title("Brazil data", fontsize=16)

sns.barplot(data=worldData['suicides_no'].groupby(worldData['generation']).sum().reset_index(level='generation'),
            x='generation',
            y='suicides_no',
            ax=axisArray[1],
            order=order,
            ci=None)
axisArray[1].set_title("Worldwide data", fontsize=16)

df=pd.concat([brazilData['suicides_no'].groupby(brazilData['generation']).sum().reindex(order).rename('Brazil'),
              worldData['suicides_no'].groupby(worldData['generation']).sum().reindex(order).rename('World')],
             axis='columns',
             sort=False).reset_index(level='generation')
sns.barplot(data=df.melt(id_vars='generation',value_vars=['Brazil','World'],var_name='scope',value_name='suicides_no'),
            x='generation',
            y='suicides_no',
            hue='scope',
            ax=axisArray[2],
            ci=None)
axisArray[2].set_title("Comparision", fontsize=16)

for axis in axisArray:
    axis.set_xlabel('')
    axis.set_ylabel('')
    sns.despine(ax=axis)

cleanOuterAxis(outerAxis)


# The main observations that can be made from these graphs are:
# 
# * a) The suicide rate for both scopes was growing steadily until Generation X, after which it apparently started to decay;
# * b) Comparing both scopes, it's visible that Brazil's G.I. and Silent generations had slightly lower suicidal tendencies than the rest of the world, however, generations X, Y (aka Millenials) and Z have slightly bigger suicidal tendencies.
# 
#  Notice that I said the suicide rates *apparently* started to decay after Generation X. The reason for such is because Generation X people are mostly older than 54 years by now, and as we've seen, between 35 and 54 years is when most suidides happen. That is to say, most of Generation X got through the Critical Range already, so it's unlikely that the X Generation will become the second generation with most suicides, surpassing the Silent generation.
#  
#  However, notice that Millenials did **not** go through the Critical Range, and yet, they are the fourth most suicidal generation already. This suggests that it's likely Millenials, with their fame as the Depression, Anxiety, and Suicide Generation, might become one of the generations with most suicides. 
#  
#  Alternatively, we might suggest that Millenials might have a earlier Critical Range. By doing so, it's possible to believe that older Millenials will not have greater suicidal tendencies when compared to younger ones, but we also might be left thinking if Generation Z will have a still earlier Critical Range.
#  
#   What I'm trying to point out here is that a first analysis of the data might suggest that suicide rates are dwindling recently, but it might be too early to ignore suicide as a significant problem in today's world.
#   
#   Finally, there's a mild suggestion that in Brazil's sample younger generations have slightly higher suicidal tendencies, at least when comparing to the world data. Still, since it's so small, I do not think such difference is relevant.
#   
#   Anotherparameter I want to study is the suicide rate difference accoding to sex, which is represented in the following visualization:

# In[ ]:


fig=plt.figure(figsize=(16, 9))
outerAxis=fig.add_subplot(1,1,1)
axisArray=[fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)]
outerAxis.set_xlabel('Age interval',fontsize=18); outerAxis.xaxis.set_label_coords(0.5,-0.2)
outerAxis.set_ylabel('Number of suicides',fontsize=18); outerAxis.yaxis.set_label_coords(-0.075,0.5)
fig.suptitle("Number of suicides according to age and sex", fontsize=18)
order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']

sns.barplot(data=brazilData['suicides_no'].groupby([brazilData['age'],brazilData['sex']]).sum().reset_index(level=['sex','age']),
            x='age',
            y='suicides_no',
            hue='sex',
            ax=axisArray[0],
            order=order,
            ci=None)
axisArray[0].set_title("Brazil data", fontsize=16)

sns.barplot(data=worldData['suicides_no'].groupby([worldData['age'],worldData['sex']]).sum().reset_index(level=['sex','age']),
            x='age',
            y='suicides_no',
            hue='sex',
            ax=axisArray[1],
            order=order,
            ci=None)
axisArray[1].set_title("Worldwide data", fontsize=16)

for axis in axisArray:
    axis.set_xlabel('')
    axis.set_ylabel('')
    sns.despine(ax=axis)
    
cleanOuterAxis(outerAxis)


#  As said before, when considering all generations, the most common age interval for suicides to happen is the 35 to 54 years range, aka the Critical Range. Also, a new information we can adquire from this visualization is that men are much more likely to commit suicide than women, both worldwide and in Brazil only.
#  
#  Trying to find a explanation for this, I found [this](https://science.howstuffworks.com/life/inside-the-mind/emotions/men-or-women-happier.htm) How Stuff Works' post, which indicates that women are mostly happier than men, however, as both grow older men show a greater happiness than women. I'm pointing this out mostly because it might be an explanation as to why for people with more than 75 years old there's a lot less difference in the suicide rate. Notice that by saying this I'm disconsidering teenagers, where the difference is a lot smaller, and the only argument I can give for doing so is that teenagers are mostly shielded from sex differences, which begin to be more important (and apparent) after 15 years of age.
#  
#   As a final parameter to analyze, I would like to see how suicide rates are changing over the years, disconsidering each generation particularities.

# In[ ]:


fig=plt.figure(figsize=(16, 9))
outerAxis=fig.add_subplot(1,1,1)
axisArray=[fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)]
outerAxis.set_xlabel('Year',fontsize=18); outerAxis.xaxis.set_label_coords(0.5,-0.2)
outerAxis.set_ylabel('Number of suicides',fontsize=18); outerAxis.yaxis.set_label_coords(-0.075,0.5)
fig.suptitle("Number of suicides versus time", fontsize=18)

sns.lineplot(data=brazilData['suicides_no'].groupby(brazilData['year']).sum().reset_index(level='year'),
            x='year',
            y='suicides_no',
             ax=axisArray[0])
axisArray[0].set_title("Brazil data", fontsize=18)

sns.lineplot(data=worldData['suicides_no'].groupby(worldData['year']).sum().reset_index(level='year'),
            x='year',
            y='suicides_no',
            ax=axisArray[1])
axisArray[1].set_title("Worldwide data", fontsize=18)

for axis in axisArray:
    axis.set_xlabel('')
    axis.set_ylabel('')
    sns.despine(ax=axis)
    
cleanOuterAxis(outerAxis)


#  The first thing that catches the eye is that apparently the global suicide rate crashed to almost zero in the last years. I'm not sure if I missed something in my code, which might be the case, but I'll believe that the rest of the plot is correct.
#  
#  With that out of the way, we can see that recently the suicides rate really are dwindling, albeit not in a dramatic manner. Still, as I pointed out before, it might be still too soon to assume that in the future years the suicide rate will keep going down, since Millenials and Generation Z still need to go though their Critical Ranges. Also, for Brazil in particular, the suicide rates have kept increasing in the last years, despite the worldwide trend being to show a decrease.
#  
#  

#  And that closes my first (decent) Kernel here on Kaggle. I still got a lot more to learn, of course, but I believe that this first study, as modest as it might be, was very important for me getting at least started, I believe. Once again, feedback is most welcome!
