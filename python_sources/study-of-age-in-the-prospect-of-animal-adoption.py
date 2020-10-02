#!/usr/bin/env python
# coding: utf-8

# **Hello!**
# 
# In this kernel I will play around with different forms of data visualisation and look for possible insigths into the data. 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sea 
import matplotlib.pyplot as pl
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv('../input/train/train.csv')
#data.head(5)



# In this kernel I will only use the data from the training file(train.csv) to show some uses of data visualization, proper data crunshing will be done on a later stage. First I will set a function to print bar graph from raw data.

# In[ ]:


def BarGraph(data,Names):
    DataSize = len(data[0])
    for val in data:
        if(len(val)>DataSize):
            DataSize = len(val)
    if(len(data) != len(Names)):
        print("Error: number of sets different from number of names",len(data),len(Names))
        return
    Width=1/len(data)-0.01
    fig=pl.figure(figsize=[14,6.5])
    ind = np.arange(len(data[0]))
    
    Colors=['SkyBlue','coral','mediumseagreen','chocolate','gold','seagreen']
    NColors=len(Colors)
    recs=[]
    i=0;
    for vals in data:
        recs+=[pl.bar(ind + i*Width, vals, Width, 
                    color=Colors[i%NColors], label=Names[i])]
        i+=1
    
    pl.xticks(range(DataSize));
    pl.legend(loc=0)
    for rec in recs:
        for rect in rec:
            height = rect.get_height()
            pl.text(rect.get_x() + rect.get_width()*0.5, 1.01*height,
                '%.1f'%(height), ha='center', va='bottom')


# For the analyse I will start by dividing the set between cats and dogs to analise how their adoption speed varies between the species using bar graph function I set previouly.

# In[ ]:


# check by type

cats=data[data.Type==2]
dogs=data[data.Type==1]

CatsTotal=[]
for i in range(5):
    curr_=sum(sum([cats.AdoptionSpeed == i]))
    CatsTotal+=[curr_/len(cats)*100]

DogsTotal=[]
for i in range(5):
    curr_=sum(sum([dogs.AdoptionSpeed == i]))
    DogsTotal+=[curr_/len(dogs)*100]

BarGraph([CatsTotal,DogsTotal],['Cats','Dogs'])
pl.ylabel('% of adopted animals by type')
pl.xlabel('Time period')
pl.title('% of animals adopted on a period of time')
pl.show()


# As we can see Cats tend to be adopted on an earlyer stage than dogs.
# 
# Next I will visualize the adoption speed by gender and type again by using bar graphs. Spoiler, there is something midly interesting there.

# In[ ]:


# check by gender
FCats=cats[cats.Gender==2]
MCats=cats[cats.Gender==1]

FDogs=dogs[dogs.Gender==2]
MDogs=dogs[dogs.Gender==1]

CatsMTotal=[]
for i in range(5):
    curr_=sum(sum([MCats.AdoptionSpeed == i]))
    CatsMTotal+=[curr_/len(MCats)*100]

CatsFTotal=[]
for i in range(5):
    curr_=sum(sum([FCats.AdoptionSpeed == i]))
    CatsFTotal+=[curr_/len(FCats)*100]
    
DogsFTotal=[]
for i in range(5):
    curr_=sum(sum([FDogs.AdoptionSpeed == i]))
    DogsFTotal+=[curr_/len(FDogs)*100]

DogsMTotal=[]
for i in range(5):
    curr_=sum(sum([MDogs.AdoptionSpeed == i]))
    DogsMTotal+=[curr_/len(MDogs)*100]

BarGraph([CatsMTotal,CatsFTotal,DogsMTotal,DogsFTotal],['Cats M','Cats F','Dogs M','Dogs F'])
pl.ylabel('% of adopted animals by gender')
pl.xlabel('Time period')
pl.title('% of animals adopted on a period of time by gender and type')


Fem = data[data.Gender==2]
Mal = data[data.Gender==1]
Group = data[data.Gender==3]

FemTotal=[]
for i in range(5):
    curr_=sum(sum([Fem.AdoptionSpeed == i]))
    FemTotal+=[curr_/len(Fem)*100]

MalTotal=[]
for i in range(5):
    curr_=sum(sum([Mal.AdoptionSpeed == i]))
    MalTotal+=[curr_/len(Mal)*100]
    
GroupTotal=[]
for i in range(5):
    curr_=sum(sum([Group.AdoptionSpeed == i]))
    GroupTotal+=[curr_/len(Group)*100]

BarGraph([FemTotal,MalTotal,GroupTotal],['Female','Malee','Group'])
pl.ylabel('% of adopted animals by Gender')
pl.xlabel('Time period')
pl.title('% of animals adopted on a period of time')


#BarGraph([DogsTotal,CatsTotal,FemTotal,MalTotal],['Dogs','Cats','Female','Malee'])
#pl.ylabel('% of adopted animals by Gender')
#pl.xlabel('Time period')
#pl.title('% of animals adopted on a period of time')
pl.show()


# Now, as I sad before there is something that I find midly interesting on this graphs, and that thing is that male animals are generaly faster to adopt than females except for female cats adopted on the same day as they are put to adoption (although not something statistically significant). This is interesting, but to better visualize of fast a animal can be adopted the visualizaion of the # of animlas up to that period can be more interesting as it gives the comulative odd of adoption.

# In[ ]:


# check by gender
FCats=cats[cats.Gender==2]
MCats=cats[cats.Gender==1]
FDogs=dogs[dogs.Gender==2]
MDogs=dogs[dogs.Gender==1]

CatsMTotal=[]
curr_=0
for i in range(5):
    curr_+=sum(sum([MCats.AdoptionSpeed == i]))
    CatsMTotal+=[curr_/len(MCats)*100]

CatsFTotal=[]
curr_=0
for i in range(5):
    curr_+=sum(sum([FCats.AdoptionSpeed == i]))
    CatsFTotal+=[curr_/len(FCats)*100]
    
DogsFTotal=[]
curr_=0
for i in range(5):
    curr_+=sum(sum([FDogs.AdoptionSpeed == i]))
    DogsFTotal+=[curr_/len(FDogs)*100]

DogsMTotal=[]
curr_=0
for i in range(5):
    curr_+=sum(sum([MDogs.AdoptionSpeed == i]))
    DogsMTotal+=[curr_/len(MDogs)*100]

BarGraph([CatsMTotal,CatsFTotal,DogsMTotal,DogsFTotal],['Cats M','Cats F','Dogs M','Dogs F'])
pl.ylabel('comulative % of adopted animals by gender')
pl.xlabel('Time period')
pl.title('% of animals adopted on a period of time by gender and type')


Fem = data[data.Gender==2]
Mal = data[data.Gender==1]
Group = data[data.Gender==3]

FemTotal=[]
curr_=0
for i in range(5):
    curr_+=sum(sum([Fem.AdoptionSpeed == i]))
    FemTotal+=[curr_/len(Fem)*100]

MalTotal=[]
curr_=0
for i in range(5):
    curr_+=sum(sum([Mal.AdoptionSpeed == i]))
    MalTotal+=[curr_/len(Mal)*100]
    
GroupTotal=[]
curr_=0
for i in range(5):
    curr_+=sum(sum([Group.AdoptionSpeed == i]))
    GroupTotal+=[curr_/len(Group)*100]

BarGraph([FemTotal,MalTotal,GroupTotal],['Female','Malee','Group'])
pl.ylabel('comulative% of adopted animals by Gender')
pl.xlabel('Time period')
pl.title('% of animals adopted on a period of time')
pl.show()


# So far we got some interesting information, but to look for the more relevant information we should study the animal age.
# Now, if an old animal is less likely to be adopted it might get to a point where it always getting to old to be statically adopted, ie., it will end up never adopted. Lets study it and see what we can find by using a box plot.

# In[ ]:


Adop0 = data[data.AdoptionSpeed == 0]
Adop1 = data[data.AdoptionSpeed == 1]
Adop2 = data[data.AdoptionSpeed == 2]
Adop3 = data[data.AdoptionSpeed == 3]
Adop4 = data[data.AdoptionSpeed == 4]


fig=pl.figure(figsize=[14,6.5])
pl.boxplot([Adop0.Age,
        Adop1.Age,
        Adop2.Age,
        Adop3.Age,
        Adop4.Age])

pl.show()
#pl.plot(Adop0.AdoptionSpeed,Adop0.Age,'*',
#        Adop1.AdoptionSpeed,Adop1.Age,'*',
#        Adop2.AdoptionSpeed,Adop2.Age,'*',
#        Adop3.AdoptionSpeed,Adop3.Age,'*',
#        Adop4.AdoptionSpeed,Adop4.Age,'*')
#pl.show()


# What we can conclude with the box plot of this data is that most of the animals are very young and that skew the box graphs making hard to visualize what is happening.
# A better way to visualize is by using violin plots, has it add information of the density that a pure box plot doesn't, also, by grouping the animals by age groups and swaping the independent and dependent variable on the graphs..

# In[ ]:


AgeData = []
AgeData+=[data[data.Age<=3]]
age_labels=['<3']
for n in range(3,60,6):
    Agen = data[data.Age>n]
    AgeData+=[Agen[Agen.Age<=n+3]]
    age_labels+=[str(n+3)]
              
AgeData += [data[data.Age>60]]
age_labels+=['60+']

AgeDataViolin = [ Age.AdoptionSpeed.values for Age in AgeData]

fig = pl.figure(figsize=[14,6])
pl.violinplot(AgeDataViolin,showmeans=True,showmedians=True)
pl.setp(fig.axes,xticks=[y + 1 for y in range(len(AgeData))],
         xticklabels=age_labels)
pl.ylabel('Adoption Speed')
pl.xlabel('Animal Age')
pl.title('Distribution of an animal adoption speed by its age')
pl.show()


# Now, this is interesting. the younger the animal is the most likely it will be to be adopted on an early stage up to around 18 months (1.5 years), after that the values seem a bit more random. I believe that the following factors can contribute for this distribution:
# * young animals are cutter and therefor universally desirable (less than 6 months),
# * animals that loose their cutness of youth but are energetic and not trained get in a vale of desirability (6 to 24 months),
# * older animals are already trained and are typicaly less energetic, which might be desirable by some people(18 to 36 months),
# * animals that are to old to be cute but are not so old that people feel sorry for them (36 to 54 months)
# * people can feel sorry by older animals, specially the one with stories of abandonment or injuries (older animals).
# This could be studied further by looking at the pictures of the animals and evaluait them on how cute or sorry they look, but that isn't for this kernel.
# 
# This information can also be more clearly visualised on a heat map.

# In[ ]:


info = np.array([[sum(Age.AdoptionSpeed == 0), 
                 sum(Age.AdoptionSpeed == 1), 
                 sum(Age.AdoptionSpeed == 2), 
                 sum(Age.AdoptionSpeed == 3), 
                 sum(Age.AdoptionSpeed == 4)] for Age in AgeData])
HeatData=[];
for row in info:
    HeatData+=[row/(sum(row)) *100]

f, ax = pl.subplots(figsize=(9, 6))
sea.heatmap(HeatData, annot=True, linewidths=.5, ax=ax)

pl.xlabel('Adoption Speed')
pl.ylabel('Animal Age')
pl.title('Percentage of an animal adoption speed by its age')

pl.show()


# In[ ]:


CatsAgeData = []
CatsAgeData+=[cats[cats.Age<=3]]
DogsAgeData = []
DogsAgeData+=[dogs[dogs.Age<=3]]
age_labels=['<3']
for n in range(3,60,6):
    Agenc = cats[cats.Age>n]
    CatsAgeData+=[Agenc[Agenc.Age<=n+3]]
    Agend = dogs[dogs.Age>n]
    DogsAgeData+=[Agend[Agend.Age<=n+3]]
    age_labels+=[str(n+3)]  
CatsAgeData += [cats[cats.Age>60]]
DogsAgeData += [dogs[dogs.Age>60]]
age_labels+=['60+']     

infoc = np.array([[sum(Age.AdoptionSpeed == 0), 
                 sum(Age.AdoptionSpeed == 1), 
                 sum(Age.AdoptionSpeed == 2), 
                 sum(Age.AdoptionSpeed == 3), 
                 sum(Age.AdoptionSpeed == 4)] for Age in CatsAgeData])
HeatDataCats=[];
for row in infoc:
    HeatDataCats+=[row/(sum(row)) *100]
    
infod = np.array([[sum(Age.AdoptionSpeed == 0), 
                 sum(Age.AdoptionSpeed == 1), 
                 sum(Age.AdoptionSpeed == 2), 
                 sum(Age.AdoptionSpeed == 3), 
                 sum(Age.AdoptionSpeed == 4)] for Age in DogsAgeData])
HeatDataDogs=[];
for row in infod:
    HeatDataDogs+=[row/(sum(row)) *100]
    
f, (ax1,ax2) = pl.subplots(nrows=1,ncols=2, figsize=(15, 6))    
ax1=sea.heatmap(HeatDataCats, annot=True, linewidths=.5, ax=ax1)
ax1.set_xlabel('Adoption Speed')
ax1.set_ylabel('Animal Age')
ax1.set_yticklabels(age_labels)
ax1.set_title('Cats')
ax2=sea.heatmap(HeatDataDogs, annot=True, linewidths=.5, ax=ax2)
ax2.set_xlabel('Adoption Speed')
ax2.set_ylabel('Animal Age')
ax2.set_title('Dogs')
ax2.set_yticklabels(age_labels)
pl.show()


# We can see some similarities between the heat maps of both animals, but for older animals the reduce number of specimens makes infering from the older animals unrealible.
# 
# This is all for this kernel for now.
# 
# Thank you for reading! :)
