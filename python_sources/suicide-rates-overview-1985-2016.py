#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()

# Disabling warnings
import warnings
warnings.simplefilter("ignore")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## We are going to find answers of the issues below;
# * Global Suicides(per 100K)-trend over time 1985-2015
# * Global Suicides(per 100K) by Continent(? not now)
# * Global Suicides(per 100k) by Gender and trend over time 1985-2015
# * Population-gdp_per_capita Plot
# * Correlation between GDP(per Capita) and suicides per 100k
# * Generation hue Gender Counter(? not now)
# * which age of people suicide a most
# * which generation of people suicide a most

# ## First we are exploring our data;

# In[ ]:


suicides = pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv")


# In[ ]:


suicides.tail()


# In[ ]:


suicides.info()


# In[ ]:


suicides.isnull().any()


# In[ ]:


suicides.isnull().sum()


# ### suicides['column name'].fillna(0,inplace = True) 
# #### if we have Nan values and we want to get rid of them;
# - we can fill NAN values with 0,
# - we can fill NAN values with the mean of the column or any number we chose,
# - we can drop NAN lines if neccessary etc.

# In[ ]:


scc = suicides.copy()#for more manipulation we take a copy of dataset.


# In[ ]:


scc.drop(['HDI for year'],axis = 1,inplace = True) ## Due to we do not need HDI for year column we drop it anyway.


# ## the number of the suicide rates by countries below;

# # Box plot

# In[ ]:


plt.figure(figsize = (14,5)) # Shape of the figure/graphic
best_20_countries = scc.sort_values(by= 'suicides_no',ascending = True) #Making the data best suitable our graphic
sns.boxplot(x='country', y = 'suicides_no',data = best_20_countries); #Visualization
plt.xticks(rotation = 90);# rotate the country names for clearly see in the table


# In[ ]:


last_20_countries = scc.sort_values(by= 'suicides_no',ascending = True) #Making the data best suitable our graphic
sns.boxplot(x='country', y = 'suicides_no',data = last_20_countries[-500:]); #Visualization
plt.xticks(rotation = 90);# rotate the country names for clearly see in the table


# # Bar plot

# In[ ]:


plt.figure(figsize = (14,5))
sns.barplot(x='year',y ='suicides_no',data = scc );#Suicide rates from 1985 to 2016. hue = "sex"
plt.xticks(rotation=90);

#For making much clear of the axis and table names
plt.xlabel('Year 1985-2016')
plt.ylabel('Sucides No')
plt.title('Total Sucides From 1985-2016');

#scc.groupby("year").sum().suicides_no


# ## please pay attention the rate in 2016 and look again the table below carefully. Thats why sometimes the data hides the real information. 
# ## Always remember and check the tables with cross check or add third part values...

# In[ ]:


sns.barplot(x='year',y ='suicides/100k pop',data = scc );#Suicide rates per 100k population from 1985 to 2016.
plt.xticks(rotation=90);
#For making much clear of the axis and table names
plt.xlabel('Year 1985-2016')
plt.ylabel('Sucides per 100k pop')
plt.title('Sucides per 100k Population From 1985-2016');


# # Point plot

# In[ ]:


plt.figure(figsize = (12,6))

sns.pointplot(x='year',y =scc['suicides_no']/20,data = scc,color = 'lime',alpha = 0.1);
sns.pointplot(x='year',y ='suicides/100k pop',data = scc,color = 'blue',alpha = 0.1);
plt.xticks(rotation=90);
#For making much clear of the axis and table names
plt.xlabel('Year 1985-2016')
plt.ylabel('Sucides num-Suicides per 100k pop')
plt.title('Sucides per 100k Population From 1985-2016');


plt.text(35,0,'Sucides number',color='lime',fontsize = 12,style = 'italic')
plt.text(35,0.8,'Suicides 100kpop',color='blue',fontsize = 12,style = 'italic');
plt.grid()


# # Joint Plot

# ## Our Dataset is not suitable for jointplot

# In[ ]:


g = sns.jointplot(scc['year'],scc.suicides_no, kind="kde", size=7,ratio = 3,color = 'red')
# plt.savefig('graph.png')
# plt.show()


# # Pie Chart

# In[ ]:


scc.groupby('generation').sum()


# In[ ]:


labels = scc.groupby('generation').sum().index
colors = ['grey','blue','red','yellow','green','brown']#["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
explode = [0.05,0.05,0.05,0.05,0.05,0.05]
sizes = scc.groupby('generation').sum()['suicides/100k pop']
# sizes = scc.groupby('generation').sum()['suicides_no']

# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')#colors=colors,
plt.title('Suicides 100k pop - Generation',color = 'blue',fontsize = 15);


# In[ ]:


scc.corr()


# # Heatmap

# In[ ]:


f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(scc.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax)#
plt.show()


# In[ ]:


current_palette = sns.color_palette("husl", 8)
sns.palplot(current_palette)


# ### 2-) Global Suicides(per 100K) by Continent

# In[ ]:


ASIA = ['Afghanistan',
'Bangladesh',
'Bhutan',
'Brunei',
'Burma',
'Cambodia',
'China',
'East Timor',
'Hong Kong',
'India',
'Indonesia',
'Iran',
'Japan',
'Republic of Korea',
'Laos',
'Macau',
'Malaysia',
'Maldives',
'Mongolia',
'Nepal',
'Pakistan',
'Philippines',
'Singapore',
'Sri Lanka',
'Taiwan',
'Thailand',
'Vietnam']

C_W_OF_IND_STATES=['Armenia',
'Azerbaijan',
'Belarus',
'Georgia',
'Kazakhstan',
'Kyrgyzstan',
'Moldova',
'Russian Federation',
'Tajikistan',
'Turkmenistan',
'Ukraine',
'Uzbekistan']
EASTERN_EUROPE=['Albania','Bosnia and Herzegovina','Bulgaria','Croatia','Czech Republic','Hungary','Macedonia','Poland','Romania']
EASTERN_EUROPE+=['Serbia','Slovakia','Slovenia']
LATIN_AMER_CARIB=['Anguilla',
'Antigua and Barbuda',
'Argentina',
'Aruba',
'Bahamas',
'Barbados',
'Belize',
'Bolivia',
'Brazil',
'British Virgin Is.',
'Cayman Islands',
'Chile',
'Colombia',
'Costa Rica',
'Cuba',
'Dominica',
'Dominican Republic',
'Ecuador',
'El Salvador',
'French Guiana',
'Grenada',
'Guadeloupe',
'Guatemala',
'Guyana',
'Haiti',
'Honduras',
'Jamaica',
'Martinique',
'Mexico',
'Montserrat',
'Netherlands Antilles',
'Nicaragua',
'Panama',
'Paraguay',
'Peru',
'Puerto Rico',
'Saint Kitts and Nevis',
'Saint Lucia',
'Saint Vincent and Grenadines',
'Suriname',
'Trinidad and Tobago',
'Turks and Caicos Is',
'Uruguay',
'Venezuela',
'Virgin Islands']

NEAR_EAST=['Bahrain',
'Cyprus',
'Gaza Strip',
'Iraq',
'Israel',
'Jordan',
'Kuwait',
'Lebanon',
'Oman',
'Qatar',
'Saudi Arabia',
'Syria',
'Turkey',
'United Arab Emirates',
'West Bank',
'Yemen']

NORTHERN_AFRICA=['Algeria',
'Egypt',
'Libya',
'Morocco',
'Tunisia',
'Western Sahara']
NORTHERN_AMERICA=['Bermuda',
'Canada',
'Greenland',
'St Pierre and Miquelon',
'United States']

OCEANIA=['American Samoa',
'Australia',
'Cook Islands',
'Fiji',
'French Polynesia',
'Guam',
'Kiribati',
'Marshall Islands',
'Micronesia, Fed. St.',
'Nauru',
'New Caledonia',
'New Zealand',
'N. Mariana Islands',
'Palau',
'Papua New Guinea',
'Samoa',
'Solomon Islands',
'Tonga',
'Tuvalu',
'Vanuatu',
'Wallis and Futuna']

SUB_SAHARAN_AFRICA=['Angola',
'Benin',
'Botswana',
'Burkina Faso',
'Burundi',
'Cameroon',
'Cape Verde',
'Central African Rep.',
'Chad',
'Comoros',
'Congo, Dem. Rep.',
'Congo, Repub. of the',
'Cote dIvoire',
'Djibouti',
'Equatorial Guinea',
'Eritrea',
'Ethiopia',
'Gabon',
'Gambia, The',
'Ghana',
'Guinea',
'Guinea-Bissau',
'Kenya',
'Lesotho',
'Liberia',
'Madagascar',
'Malawi',
'Mali',
'Mauritania',
'Mauritius',
'Mayotte',
'Mozambique',
'Namibia',
'Niger',
'Nigeria',
'Reunion',
'Rwanda',
'Saint Helena',
'Sao Tome & Principe',
'Senegal',
'Seychelles',
'Sierra Leone',
'Somalia',
'South Africa',
'Sudan',
'Swaziland',
'Tanzania',
'Togo',
'Uganda',
'Zambia',
'Zimbabwe']
WESTERN_EUROPE=['Andorra',
'Austria',
'Belgium',
'Denmark',
'Faroe Islands',
'Finland',
'France',
'Germany',
'Gibraltar',
'Greece',
'Guernsey',
'Iceland',
'Ireland',
'Isle of Man',
'Italy',
'Jersey',
'Liechtenstein',
'Luxembourg',
'Malta',
'Monaco',
'Netherlands',
'Norway',
'Portugal',
'San Marino',
'Spain',
'Sweden',
'Switzerland',
'United Kingdom']
def GetConti(counry):
    if counry in ASIA:
        return "ASIA"
    elif counry in C_W_OF_IND_STATES:
        return "C_W_OF_IND_STATES"
    elif counry in EASTERN_EUROPE:
        return "EASTERN_EUROPE"
    elif counry in LATIN_AMER_CARIB:
        return "LATIN_AMER_CARIB"
    elif counry in NEAR_EAST:
        return "NEAR_EAST"
    elif counry in NORTHERN_AFRICA:
        return "NORTHERN_AFRICA"
    elif counry in NORTHERN_AMERICA:
        return "NORTHERN_AMERICA"
    elif counry in OCEANIA:
        return "OCEANIA"
    elif counry in SUB_SAHARAN_AFRICA:
        return "SUB_SAHARAN_AFRICA"
    elif counry in WESTERN_EUROPE:
        return "WESTERN_EUROPE"
    else:
        return "other"
country=scc["country"]
country=pd.DataFrame(country)
# list(country["country"])
df1 = pd.DataFrame({"Country": list(country["country"])})
df1['Continent'] = df1['Country'].apply(lambda x: GetConti(x))
scc["continent"]=df1["Continent"]
scc[scc["continent"]=="other"]["country"]


# In[ ]:


continent_list=list(scc['continent'].unique())
suicides_100k_pop = []
for i in continent_list:
    x = scc[scc['continent']==i]
    rate = sum(x['suicides/100k pop'])/len(x)
    suicides_100k_pop.append(rate)
data1 = pd.DataFrame({'Continent_list': continent_list,'suicides/100k pop':suicides_100k_pop})

plt.figure(figsize = (15,15))
plt.subplot(2,2,1)
sns.barplot(x=scc.groupby('continent')['suicides/100k pop'].mean().index,y=scc.groupby('continent')['suicides/100k pop'].mean().values)
plt.title("Global Suicides(per 100K) by Continent")
plt.ylabel("Suicide per 100K")
plt.xlabel("Continents")
plt.xticks(rotation=90)

plt.subplot(2,2,2)
labels =data1.Continent_list
colors = ['grey','blue','red','yellow','green',"orange", "darkblue","purple","maroon","gold"]
explode = [0,0,0,0,0,0,0,0,0,0]
sizes = data1['suicides/100k pop']
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Global Suicides(per 100K) rate of Continents',color = 'blue',fontsize = 15)
plt.show()


# In[ ]:


scc.info()


# ### 3-) Global Suicides(per 100k) by Gender and trend over time 1985-2015

# In[ ]:


scc.groupby('sex')['suicides/100k pop'].sum()


# # Lineplot

# In[ ]:


sns.lineplot(x="year", y='suicides/100k pop',data = scc);


# In[ ]:


sns.lineplot(x='year',y='suicides/100k pop' ,hue = 'sex',data = scc); #hue = "generation",


# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='year', y='suicides_no', data=scc, hue='sex');


# ### 4-) Population-gdp_per_capita Plot

# In[ ]:


plt.figure(figsize=(12,18))
sns.scatterplot(x='population',y = 'country',hue = 'gdp_per_capita ($)',data = scc);


# ### 5-) Correlation between GDP(per Capita) and suicides per 100k
# 

# In[ ]:


scc.head()


# In[ ]:


sns.pairplot(scc,kind='reg');#hue="sex"


# In[ ]:


sns.jointplot(x='suicides/100k pop',y = 'gdp_per_capita ($)',data = scc,kind = 'reg');


# In[ ]:


plt.figure(figsize=(10,8))
sns.scatterplot(x='suicides/100k pop',y='gdp_per_capita ($)', hue ="age",data=scc);


# In[ ]:


plt.figure(figsize=(10,8))
sns.scatterplot(x='suicides/100k pop',y='gdp_per_capita ($)',hue= "generation", data=scc);#gedp and Sucides rate dispersion over generations.


# ### 6-) Generation hue Gender Counter(later)

# In[ ]:



gen = scc.groupby('generation').sum()
gen


# ### 7-) which age of people suicide a most

# In[ ]:


sns.barplot(x="age",y = 'suicides_no',data=scc);
plt.xticks(rotation=90);


# In[ ]:


plt.figure(figsize=(10,7))
sns.barplot(x="sex",y = 'suicides_no',hue = 'age',data=scc);


# ## Mid aged(35-54) peoples from both genders suicide at most and the 55-74 aged group is the second. 

# ### 8-) which generation of people suicide a most

# In[ ]:


sns.barplot(x="generation",y = 'suicides_no',data=scc);
plt.xticks(rotation=90);


# In[ ]:


plt.figure(figsize=(10,7))
sns.barplot(x="sex",y = 'suicides_no',hue = 'generation',data=scc);


# ### For both genders Boomers suicide at most and Silent generation is the second.

# In[ ]:


scc.groupby('generation').sum()


# In[ ]:


plt.figure(figsize=(12,14))
sns.catplot(x='generation', y="suicides/100k pop", kind="boxen",data=scc);
plt.xticks(rotation=90);


# In[ ]:


plt.figure(figsize=(12,14))
sns.catplot(x='generation', y="suicides/100k pop", kind="box",data=scc);
plt.xticks(rotation=90);


# In[ ]:


# scc.groupby('sex').sum()
# scc.sex.value_counts()


# In[ ]:


sns.countplot(x=scc['generation'], data=scc); #split = True,hue = "sex", kind="count",
plt.xticks(rotation=90);


# In[ ]:


# plt.figure(figsize=(12,14))
sns.catplot(x='sex', y="suicides_no", kind="violin", data=scc); #split = True,hue = "sex",
plt.xticks(rotation=90);

