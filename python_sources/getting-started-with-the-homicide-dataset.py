#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime 
import scipy.stats as stats
import math
import warnings
warnings.simplefilter('ignore')

df = pd.read_csv("../input/Serial Killers Data.csv",encoding = "ISO-8859-1")
df.head()


# Preprocessing

# In[ ]:


print(df.shape)


# In[ ]:


del_list = ['PerfIQ','VerbalIQ','LocAbduct','Nickname','IQ2','Chem','Fired','GetToCrime','XYY','Killed',
           'Combat','BedWet','AgeEvent','IQ1','PsychologicalDiagnosis','Otherinformation','News',
           'DadSA','DadStable','LivChild','LocKilling','PoliceGroupie','MomSA','ParMarSta','Fire',
           'AppliedCop','Attrac','LiveWith','Educ','FamEvent','MomStable','Mental','Animal',
           'HeadInj','PartSex']

for i in del_list:
    del df[i]


# In[ ]:


df.KillMethod.unique()


# In[ ]:


Kill_Methods = []
for i in df['KillMethod']:
    if ',' in str(i):
        i.split(',')
        for x in i:
            Kill_Methods.append(x)
    elif i == 'liveabortions':
        Kill_Methods.append(27)
    else:
        try:
            Kill_Methods.append(int(i))
        except:
            continue

Kill_Methods[:] = [int(m) for m in Kill_Methods if m != ',']
#Returns all unique values after splitting observations with multiple methods
set(Kill_Methods)


# In[ ]:


methods_dict = {0:'Unknown', 1: 'Bludgeon' , 2:'Gun', 3:'Poison',
              4:'Stabbing', 5:'Strangulation', 6:'Pills', 7:'Bomb',
             8:'Suffication',9:'Gassed',10:'Drown',11:'Fire',12:'Starved/Neglect',13:'Shaken',
             14:'Axed',15:'Hanging',18:'RanOver',21:'AlcoholPoisoning',22:'DrugOverdose',
             25:'WithdrewTreatment',27:'LiveAbortions'}


# In[ ]:


len(Kill_Methods)


# In[ ]:


ethnicities = ['White','Black','Hispanic','Asian','NativeAmerican','Aboriginal']
ethnicity_dict = {}
gender_dict = {}
for x,y in zip(range(1,7,1),ethnicities):
    ethnicity_dict[x] = y

Gender = ['Male','Female']
for a,b in zip(range(1,3,1),Gender):
    gender_dict[a] = b
    
birthcat_dict = {}
BirthOrder = ['Oldest','Middle','Youngest','Only']
for c,d in zip(range(1,5,1),BirthOrder):
    birthcat_dict[c] = d


# For locations, some murderers killed in multiple locations. Although the 'City' column has a 'multiplelocations' value to account for this, some of them were still missed. Below are some examples.

# In[ ]:


for i in df['City'].iloc[:30]:
    if ',' in str(i):
        print(i)


# In[ ]:


multiple_cities = [x for x in df['City'] if ',' in str(x)]
print(len(multiple_cities),len(df['City'].loc[df.City == 'Multiplecities']))


# There are a total of 1310 people whose murders likely spanned across multiple cities in addition to the 82 that were correctly labeled. Unfortunately, we do not know the actual cities where those 82 "Multiplecities" murders took place.  

# In[ ]:


low_counts = df.City.value_counts()
print(sum(low_counts[low_counts < 2]),sum(low_counts[low_counts > 1]),len(multiple_cities))


# Almost half of the city values have only one observation, and it appears that 524 values are possibly unique given that no pre processing has been done to account for what information the 'multiple_cities' variable can give us. 

# In[ ]:


def city_totals(str_city):
    Total_City = [x for x in multiple_cities if str(str_city) in x in multiple_cities]
    Total_City = len(Total_City)
    Total_City += low_counts[str_city]
    return Total_City


city_totals("Detroit"),low_counts["Detroit"]


# In[ ]:


print([x for x in multiple_cities if 'Detroit' in x in multiple_cities])


# From this function, we picked up an additional 13 homicides that took place in the city of Detroit.

# **Location Preprocessing**

# In[ ]:


df.state.value_counts(ascending=False).iloc[:25].plot(kind='barh').invert_yaxis()
plt.title("Serial Homicide Offenders by State")


# The distribution of serial killers by state is very similar to the total population by state. Since the dataset includes observations outside of the U.S., London, Ontario, and Gauteng are represented in the top 25 most common regions in the dataset. Additionally, some serial killers murdered people in multiple regions, which can be seen by looking at the bottom portion of the dataset.

# In[ ]:


print("There are {} unique regions in the dataset. \n\n\n{}".format(len(set(df['state'])),df['state'].value_counts(ascending=True)[:20]))


# In[ ]:


df.City.value_counts().iloc[:30].plot(kind="barh").invert_yaxis()


# In[ ]:


Actual_City_Totals = {}
Top_Cities = ['Chicago','Houston','LosAngeles','Washington','Philadelphia','Miami','London','NewOrleans',
             'Jacksonville','Richmond','Moscow','NYC-Brooklyn','Phoenix','Detroit','Baltimore','LasVegas',
             'Atlanta','KansasCity','Tampa','NYC-Manhattan','Indianapolis','FortWorth',
             'SanFrancisco','St.Petersburg','Oakland','St.Louis','OklahomaCity','Memphis','Birmingham',
             'Boston','Johannesburg','Cleveland','Portland','Austin','Dallas','Nashville',
              'Cincinnati','Louisville','Sydney','Vienna','Vancouver','CapeTown','Milwaukee','Seattle',
             'Columbus','Wichita','Sacramento','Milan']

for x in Top_Cities:
    Actual_City_Totals[x] = city_totals(x)


# In[ ]:



import operator
sorted_city_values = sorted(Actual_City_Totals.items(), key=operator.itemgetter(1),reverse=True)
sorted_city_values


# **Visualization**

# In[ ]:


plt.style.use(['fivethirtyeight'])
sns.countplot(df['Sex'].map(gender_dict),palette='Blues_d')
plt.title("Gender Makeup")


# In[ ]:


schoolprob_dict = {0:'No',1:'Yes'}
teased_dict = {0: 'No',1:'Yes'}
sns.countplot(df['SchoolProb'].map(schoolprob_dict),hue=df['Teased'].map(teased_dict))
plt.title("Did the individual have problems in school?")


# In[ ]:


alc_dict = {1: "Yes",0:"No"}
sns.countplot(df['Killerabusealcohol'].map(alc_dict))
plt.title("Did the killer abuse alcohol?")


# In[ ]:


drug_dict = {1:"Yes",0:"No"}
sns.countplot(df['Killerabusedrugs'].map(drug_dict))
plt.title("Did the killer abuse drugs?")


# Many of the individuals who were not teased also did not have any problems in school; however, the category has many missing values. Similarly, for the observations that are not missing, there is an above average occurance of alcohol and drug abuse.

# In[ ]:


Kill_Methods = pd.Series(Kill_Methods).dropna().map(methods_dict)
Kill_Methods.value_counts().sort_values(ascending=True).plot(kind='barh')
plt.title("Methods of killing")


# In[ ]:


df['Race'].map(ethnicity_dict).value_counts().plot(kind='pie')


# In[ ]:


df.Race.isnull().sum()


# In[ ]:


df['BirthCat'].map(birthcat_dict).value_counts().plot(kind='pie')


# In[ ]:



df.BirthCat.isnull().sum()


# In[ ]:


sns.distplot(df['YearsBetweenFirstandLast'].dropna(),rug=True)


# In[ ]:


df.plot(
    x="YearsBetweenFirstandLast",
    y="NumVics",
    kind="scatter"
)


# In[ ]:


df['NumVics'].dropna().describe()


# In[ ]:


df.loc[df['NumVics']<150].plot(
    x="YearsBetweenFirstandLast",
    y="NumVics",
    kind="scatter"
)

#sns.regplot(x='YearsBetweenFirstandLast',y='NumVics',data=df.loc[df['NumVics'] < 150])


# In[ ]:


sns.distplot(df['YearFinal'].dropna(),rug=True)


# Most of the individuals' last murder took place after the 19th century. Let's zoom in from 1900-now to get a better since of the distribution.

# In[ ]:


sns.distplot(df['YearFinal'].dropna(),rug=True).set(xlim=(1900, 2016))


# **Differences between those who had their first kill as minors and those who were adults**

# In[ ]:


sns.distplot(df['Age1stKill'].dropna(),rug=True)


# In[ ]:


sns.distplot(df['AgeLastKill'].dropna(),rug=True)


# In[ ]:


print("Age at first kill: \n{} \n\n\nAge at last kill: \n{} \n\n{} individuals were minors when they killed for the first time.".format(df['Age1stKill'].dropna().describe(),
                                                                                                                                         df['AgeLastKill'].dropna().describe(),
                                                                                                                                         len(df.loc[df['Age1stKill']<18])))


# In[ ]:


Minor_dict = {1:'Minor',0:'Adult'}
df['Minor'] = np.where(df['Age1stKill'] < 18,1,0)
facetgrid = sns.FacetGrid(df,hue='Minor',size=6)
facetgrid.map(sns.kdeplot,'NumVics',shade=True)
facetgrid.set(xlim=(0,df['NumVics'].max()))
facetgrid.add_legend()


# In[ ]:


print("Test Statistic:\n{} \n\n\nDescriptive Statistics:\n{}".format(stats.ttest_ind(a= df['NumVics'].dropna().loc[df['Minor']==1],
                b= df['NumVics'].dropna().loc[df['Minor']==0],
                equal_var=False), df['NumVics'].dropna().groupby(df['Minor'].map(Minor_dict)).describe()))


# In[ ]:


print("Test Statistic:\n{} \n\n\nDescriptive Statistics:\n{}".format(stats.ttest_ind(a= df['YearsBetweenFirstandLast'].dropna().loc[df['Minor']==1],
                b= df['YearsBetweenFirstandLast'].dropna().loc[df['Minor']==0],
                equal_var=False), df['YearsBetweenFirstandLast'].dropna().groupby(df['Minor'].map(Minor_dict)).describe()))


# In[ ]:


facetgrid = sns.FacetGrid(df,hue='Minor',size=6)
facetgrid.map(sns.kdeplot,'YearsBetweenFirstandLast',shade=True)
facetgrid.set(xlim=(0,df['YearsBetweenFirstandLast'].max()))
facetgrid.add_legend()


# There is no difference in the amount of people killed with respect to whether or not the killer was a minor when the spree started. A strong difference in means was found in how long the spree lasts for killers who started as minors versus those who started as adults. 

#  **Are there any patterns based on the gender of the victims?**

# In[ ]:


VicSex_dict = {1.0:"Men",2.0:"Women",3.0:"Both"}
df['NumVics'].dropna().loc[df['VicSex'] != 9].groupby(df['VicSex'].map(VicSex_dict)).describe()


# In[ ]:


stats.ttest_ind(a= df['NumVics'].dropna().loc[df['VicSex']==3.0],
                b=df['NumVics'].dropna().loc[df['VicSex']!=3.0],
                equal_var=False)


# In[ ]:


print(len(df['NumVics'].dropna().loc[df['VicSex']==3.0]),df['NumVics'].dropna().mean())


# In[ ]:


stats.t.ppf(q=0.025, 
            df=1423)


# In[ ]:


print(
stats.ttest_ind(a= df['NumVics'].dropna().loc[df['VicSex']==3.0],
                b= df['NumVics'].dropna().loc[df['VicSex']==2.0],
                equal_var=False),

stats.ttest_ind(a= df['NumVics'].dropna().loc[df['VicSex']==3.0],
                b= df['NumVics'].dropna().loc[df['VicSex']==2.0],
                equal_var=False))


# There's a clear numerical difference in the means of the sample that killed both men and women and the samples for those who only killed one gender. The hypothesis test tells us that we could expect to see a difference this large in the sample mean of the group that killed both genders versus the rest of the population by chance 1.8% of the time. When testing that group against the other two sub groups, a statistically significant result was also shown.
# 
# <br>
# 
# </br>
# Although the results of group that killed both men and women were not surprising, I was surprised to find such large max values for those who only men and those who killed only women. Ezrebet Bathory was the one female who killed over 150 women (200 according to this dataset), and Giulia Tofana is the female responsible for the max value in the group of people who only killed males.

# In[ ]:


df.loc[(df['NumVics']>150) & ((df['VicSex']==2) | (df['VicSex']==1))]


# In[ ]:


print(
stats.ttest_ind(a= df['NumVics'].dropna().loc[df['VicSex']==2.0],
                b= df['NumVics'].dropna().loc[df['VicSex']==1.0],
                equal_var=False))


# There is no clear difference in the number of victims for those who kill only males versus those who only kill females.

# In[ ]:


print(
stats.ttest_ind(a= df['AgeSeries'].dropna().loc[df['VicSex']==2.0],
                b= df['AgeSeries'].dropna().loc[df['VicSex']==1.0],
                equal_var=False))


# In[ ]:


Facetgrid = sns.FacetGrid(df,hue='VicSex',size=6)
Facetgrid.map(sns.kdeplot,'AgeSeries',shade=True)
Facetgrid.set(xlim=(0,df['AgeSeries'].max()))
Facetgrid.add_legend()


# Those who killed females exclusively are slightly older at the beginning of the series compared to those who exclusively killed men.

# In[ ]:




