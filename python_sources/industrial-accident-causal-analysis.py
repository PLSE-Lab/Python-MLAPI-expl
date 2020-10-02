#!/usr/bin/env python
# coding: utf-8

# <h2 style="text-align:center;font-size:200%;;">Industrial Accident Causal Analysis </h2>
# <h3  style="text-align:center;">Keywords : <span class="label label-success">Manufacturing</span> <span class="label label-success">EDA</span> <span class="label label-success">Visualization</span> <span class="label label-success">Feature Engineering</span> <span class="label label-success">NLP</span> <span class="label label-success">Causal Analysis</span></h3>

# ## Table of Contents<a id='top'></a>
# >1. [Overview](#1.-Overview)  
# >    * [Project Detail](#Project-Detail)
# >    * [Goal of this notebook](#Goal-of-this-notebook)
# >1. [Import libraries](#2.-Import-libraries)
# >1. [Load the dataset](#3.-Load-the-dataset)
# >1. [Pre-processing](#4.-Pre-processing)
# >    * [NLP Pre-processing](#NLP-Pre-processing)
# >1. [EDA](#5.-EDA)  
# >    * [Univariate Analysis](#Univariate-Analysis)
# >    * [Multivariate Analysis](#Multivariate-Analysis)
# >    * [NLP Analysis](#NLP-Analysis)
# >1. [Modeling](#6.-Modeling)
# >    * [Feature Engineering](#Feature-Engineering)
# >    * [Case1 : Accident Level](#Case1-:-Accident-Level)
# >    * [Case2 : Potential Accident Level](#Case2-:-Potential-Accident-Level)
# >1. [Conclusion](#7.-Conclusion)
# >    * [Task Submission](#Task-Submission)
# >1. [References](#8.-References)

# # 1. Overview
# ## Project Detail
# >In [this dataset](https://www.kaggle.com/ihmstefanini/industrial-safety-and-health-analytics-database), the information about accidents in 12 manufacturing plants in 3 countries are given by a brazilian company, [IHM Stefanini](https://en.ihm.com.br/). We need to use this dataset to understand why accidents occur, and discover clues to reduce tragedic accidents.<br/>
# ><p>Dataset columns are below:</p>
# ><ul>
# ><li><b>Date</b> : timestamp or time/date information</li>
# ><li><b>Countries</b> : which country the accident occurred (<b>anonymized</b>)</li>
# ><li><b>Local</b> : the city where the manufacturing plant is located (<b>anonymized</b>)</li>
# ><li><b>Industry sector</b> : which sector the plant belongs to</li>
# ><li><b>Accident level</b> : from I to VI, it registers how severe was the accident (I means not severe but VI means very severe)</li>
# ><li><b>Potential Accident Level</b> : Depending on the Accident Level, the database also registers how severe the accident could have been (due to other factors involved in the accident)</li>
# ><li><b>Genre</b> : if the person is male of female</li>
# ><li><b>Employee or Third Party</b> : if the injured person is an employee or a third party</li>
# ><li><b>Critical Risk</b> : some description of the risk involved in the accident</li>
# ><li><b>Description</b> : Detailed description of how the accident happened</li>
# ></ul>
# 
# ## Goal of this notebook
# >* Practice Pre-processing technique
#     * Time-related feature extraction
#     * NLP pre-precessing(lower-casing, lemmatizing, stemming and removing stopwords)
# >* Practice EDA technique
# >* Practice visualising technique(especially using bokeh via holoviews)
# >* Practice feature enginieering technique
# >    * Time-related features
# >    * NLP features(TF-IDF)
# >* Practice modeling technique
# >    * LightGBM(+ plotting the tree)
# >* Causal analysis skill

# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# # 2. Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import tokenize,stem
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import lightgbm as lgb
import nltk
from nltk.util import ngrams
from wordcloud import WordCloud, STOPWORDS


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# # 3. Load the dataset

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv("/kaggle/input/industrial-safety-and-health-analytics-database/IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv")
df.head(3)


# In[ ]:


df.drop("Unnamed: 0", axis=1, inplace=True)
df.rename(columns={'Data':'Date', 'Countries':'Country', 'Genre':'Gender', 'Employee or Third Party':'Employee type'}, inplace=True)
df.head(3)


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# # 4. Pre-processing

# >Accidents may increase or decrease throughout the year or month, so I added datetime features such as year,month and day.

# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].apply(lambda x : x.year)
df['Month'] = df['Date'].apply(lambda x : x.month)
df['Day'] = df['Date'].apply(lambda x : x.day)
df['Weekday'] = df['Date'].apply(lambda x : x.day_name())
df['WeekofYear'] = df['Date'].apply(lambda x : x.weekofyear)
df.head(3)


# ### Seasonal variable
# ><div class="alert alert-success" role="alert">
# >Accordin to <a href='https://www.kaggle.com/ihmstefanini/industrial-safety-and-health-analytics-database/discussion/54113'>this discussion</a>, countries where the dataset was collected is anonymized but they are all located in South America. So in this analysis, let's assume the dataset was collected in Brazil.<br/>
# >It is said in <a href='https://seasonsyear.com/Brazil'>this web page</a> that Brazil has four climatological seasons as below.
# ><ul>
# >    <li><b>Spring</b> : September to November</li>
# >    <li><b>Summer</b> : December to February</li>
# >    <li><b>Autumn</b> : March to May</li>
# >    <li><b>Winter</b> : June to August</li>
# ></ul>
# >We can create seasonal variable based on month variable.
# ></div>

# >function to convert month variable into seasons

# In[ ]:


def month2seasons(x):
    if x in [9, 10, 11]:
        season = 'Spring'
    elif x in [12, 1, 2]:
        season = 'Summer'
    elif x in [3, 4, 5]:
        season = 'Autumn'
    elif x in [6, 7, 8]:
        season = 'Winter'
    return season


# In[ ]:


df['Season'] = df['Month'].apply(month2seasons)
df.head(3)


# ## NLP Pre-processing
# >Description column contains the details of why accidents happend. So I tried to add new features by using this important information with NLP technique.

# >In addition to the predifined stopwords in WORDCLOUD, I defined handmade-stopwords list by inspecting the documents in 'Description' column.

# In[ ]:


STOPWORDS.update(["cm", "kg", "mr", "wa" ,"nv", "ore", "da", "pm", "am", "cx"])
print(STOPWORDS)


# >NLP preprocessing pipeline is a little complicated, so I made preprocessing function.

# In[ ]:


def nlp_preprocesser(row):
    sentence = row.Description
    #convert all characters to lowercase
    lowered = sentence.lower()
    tok = tokenize.word_tokenize(lowered)

    #lemmatizing & stemming
    lemmatizer = stem.WordNetLemmatizer()
    lem = [lemmatizer.lemmatize(i) for i in tok if i not in STOPWORDS]
    stemmer = stem.PorterStemmer()
    stems = [stemmer.stem(i) for i in lem if i not in STOPWORDS]

    #remove non-alphabetical characters like '(', '.' or '!'
    alphas = [i for i in stems if i.isalpha() and (i not in STOPWORDS)]
    return " ".join(alphas)


# >convert text into applicable format by lower-casing, tokenizing, lemmatizing and stemming.

# In[ ]:


df['Description_processed'] = df.apply(nlp_preprocesser, axis=1)
df.head(3)


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# # 5. EDA

# ## Univariate Analysis

# ### Country

# In[ ]:


country_cnt = np.round(df['Country'].value_counts(normalize=True) * 100)
hv.Bars(country_cnt).opts(title="Country Count", color="green", xlabel="Countries", ylabel="Percentage", yformatter='%d%%')                .opts(opts.Bars(width=500, height=300,tools=['hover'],show_grid=True))            * hv.Text('Country_01', 15, f"{int(country_cnt.loc['Country_01'])}%")            * hv.Text('Country_02', 15, f"{int(country_cnt.loc['Country_02'])}%")            * hv.Text('Country_03', 15, f"{int(country_cnt.loc['Country_03'])}%")


# ### Local

# In[ ]:


local_cnt = np.round(df['Local'].value_counts(normalize=True) * 100)
hv.Bars(local_cnt).opts(title="Local Count", color="green", xlabel="Locals", ylabel="Percentage", yformatter='%d%%')                .opts(opts.Bars(width=700, height=300,tools=['hover'],show_grid=True))


# ### Industry Sector

# In[ ]:


sector_cnt = np.round(df['Industry Sector'].value_counts(normalize=True) * 100)
hv.Bars(sector_cnt).opts(title="Industry Sector Count", color="green", xlabel="Sectors", ylabel="Percentage", yformatter='%d%%')                .opts(opts.Bars(width=500, height=300,tools=['hover'],show_grid=True))                * hv.Text('Mining', 15, f"{int(sector_cnt.loc['Mining'])}%")                * hv.Text('Metals', 15, f"{int(sector_cnt.loc['Metals'])}%")                * hv.Text('Others', 15, f"{int(sector_cnt.loc['Others'])}%")


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ### Accident Levels
# ><div class="alert alert-success" role="alert">
# ><ul>
# >    <li>The number of accidents decreases as the Accident Level increases.</li>
# >    <li>The number of accidents increases as the Potential Accident Level decreases.</li>
# ></ul>
# ></div>

# In[ ]:


ac_level_cnt = np.round(df['Accident Level'].value_counts(normalize=True) * 100)
pot_ac_level_cnt = np.round(df['Potential Accident Level'].value_counts(normalize=True) * 100, decimals=1)
ac_pot = pd.concat([ac_level_cnt, pot_ac_level_cnt], axis=1,sort=False).fillna(0).rename(columns={'Accident Level':'Accident', 'Potential Accident Level':'Potential'})
ac_pot = pd.melt(ac_pot.reset_index(), ['index']).rename(columns={'index':'Severity', 'variable':'Levels'})
hv.Bars(ac_pot, ['Severity', 'Levels'], 'value').opts(opts.Bars(title="Accident Levels Count", width=700, height=300,tools=['hover'],                                                                show_grid=True,xrotation=45, ylabel="Percentage", yformatter='%d%%'))


# ### Gender
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>As a characteristic of the industry, the proportion of men is overwhelmingly.</li>
# ></ul>
# ></div>

# In[ ]:


gender_cnt = np.round(df['Gender'].value_counts(normalize=True) * 100)
hv.Bars(gender_cnt).opts(title="Gender Count", color="green", xlabel="Gender", ylabel="Percentage", yformatter='%d%%')                .opts(opts.Bars(width=500, height=300,tools=['hover'],show_grid=True))


# ### Employee type
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>The large number of Third Party employee type indicates the difference of employement system in gender or industry sector.</li>
# ></ul>
# ></div>

# In[ ]:


emp_type_cnt = np.round(df['Employee type'].value_counts(normalize=True) * 100)
hv.Bars(emp_type_cnt).opts(title="Employee type Count", color="green", xlabel="Employee Type", ylabel="Percentage", yformatter='%d%%')                .opts(opts.Bars(width=500, height=300,tools=['hover'],show_grid=True))


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ### Critical Risks
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>Because most part of the Critical Risks are classified as 'Others', it is thought that there are too many risks to classify precisely</li>
# ><li>And it is also thought that it takes so many time to analyze risks and reasons why the accidents occur.</li>
# ></ul>
# ></div>

# In[ ]:


cr_risk_cnt = np.round(df['Critical Risk'].value_counts(normalize=True) * 100)
hv.Bars(cr_risk_cnt[::-1]).opts(title="Critical Risk Count", color="green", xlabel="Critical Risks", ylabel="Percentage", xformatter='%d%%')                .opts(opts.Bars(width=600, height=600,tools=['hover'],show_grid=True,invert_axes=True))


# ### Calendar
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>It seems that the number of accidents decreased in latter of the year / month.</li>
# ><li>The number of accidents increased during the middle of the week and declined since the middle of the week.</li>
# ></ul>
# ></div>

# In[ ]:


year_cnt = np.round(df['Year'].value_counts(normalize=True,sort=False) * 100)
y = hv.Bars(year_cnt).opts(title="Year Count", color="green", xlabel="Years")
month_cnt = np.round(df['Month'].value_counts(normalize=True,sort=False) * 100)
m = hv.Bars(month_cnt).opts(title="Month Count", color="skyblue", xlabel="Months") * hv.Curve(month_cnt).opts(color='red', line_width=3)
day_cnt = np.round(df['Day'].value_counts(normalize=True,sort=False) * 100)
d = hv.Bars(day_cnt).opts(title="Day Count", color="skyblue", xlabel="Days") * hv.Curve(day_cnt).opts(color='red', line_width=3)
weekday_cnt = pd.DataFrame(np.round(df['Weekday'].value_counts(normalize=True,sort=False) * 100))
weekday_cnt['week_num'] = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(i) for i in weekday_cnt.index]
weekday_cnt.sort_values('week_num', inplace=True)
w = hv.Bars((weekday_cnt.index, weekday_cnt.Weekday)).opts(title="Weekday Count", color="green", xlabel="Weekdays") * hv.Curve(weekday_cnt['Weekday']).opts(color='red', line_width=3)
(y + m + d + w).opts(opts.Bars(width=400, height=300,tools=['hover'],show_grid=True, ylabel="Percentage", yformatter='%d%%')).cols(2)


# ### Season
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>The number of accidents increased in Summer and Autumn.</li>
# ><li>It is thought that the occurrence of accidents is related to the climate(especially tempeature).</li>
# ></ul>
# ></div>

# In[ ]:


season_cnt = pd.DataFrame(np.round(df['Season'].value_counts(normalize=True,sort=False) * 100).reset_index())
season_cnt['season_order'] = season_cnt['index'].apply(lambda x: ['Spring','Summer','Autumn','Winter'].index(x))
season_cnt.sort_values('season_order', inplace=True)
season_cnt.index = season_cnt['index']
season_cnt.drop(['index','season_order'], axis=1, inplace=True)
hv.Bars(season_cnt).opts(title="Season Count", color="green", xlabel="Season", ylabel="Percentage", yformatter='%d%%')                .opts(opts.Bars(width=500, height=300,tools=['hover'],show_grid=True))


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ## Multivariate Analysis

# ### Industry Sector by Countries
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>We can see that there are major industries by countries.</li><br/>
# ><li><b>Country_01</b> : Mining</li>
# ><li><b>Country_02</b> : Metals</li>
# ><li><b>Country_03</b> : Others</li>
# ></ul>
# ></div>

# In[ ]:


f = lambda x : np.round(x/x.sum() * 100)
con_sector = df.groupby(['Country','Industry Sector'])['Industry Sector'].count().unstack().apply(f, axis=1)
hv.Bars(pd.melt(con_sector.reset_index(), ['Country']), ['Country', 'Industry Sector'], 'value').opts(opts.Bars(title="Industry Sector by Countries Count", width=800, height=300,tools=['hover'],                                                                show_grid=True,xrotation=0, ylabel="Percentage", yformatter='%d%%'))


# ###  Employee type by Gender
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>Ratio of employee types by gender is not different in each gender.</li>
# ><li>The proportion of female with Third Party(Remote) is slightly higher than that of males.</li>
# ><li>It is thought that this is because <u>males have more on-site work and females often do work far away from relatively safe sites</u>.</li>
# ></ul>
# ></div>

# In[ ]:


f = lambda x : np.round(x/x.sum() * 100)
em_gen = df.groupby(['Gender','Employee type'])['Employee type'].count().unstack().apply(f, axis=1)
hv.Bars(pd.melt(em_gen.reset_index(), ['Gender']), ['Gender','Employee type'], 'value').opts(opts.Bars(title="Employee type by Gender Count", width=800, height=300,tools=['hover'],                                                                show_grid=True,xrotation=0, ylabel="Percentage", yformatter='%d%%'))


# ### Industry Sector by Gender
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>There are differences mainly in metals and mining between males and females.</li>
# ><li>Same as employee type above, it is thought that <u>this is due to different safety level by industry sector</u>.</li>
# ></ul>
# ></div>

# In[ ]:


f = lambda x : np.round(x/x.sum() * 100)
em_gen = df.groupby(['Gender','Industry Sector'])['Industry Sector'].count().unstack().apply(f, axis=1)
hv.Bars(pd.melt(em_gen.reset_index(), ['Gender']), ['Gender','Industry Sector'], 'value').opts(opts.Bars(title="Industry Sector by Gender Count", width=800, height=300,tools=['hover'],                                                                show_grid=True,xrotation=0, ylabel="Percentage", yformatter='%d%%'))


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ### Accident Levels by Gender
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>In terms of <b>accident levels</b>, there are many mild risks at general accident level, but many serious risks at potential accident level.
#     <p>It can be said that <u>many potential accidents are overlooked and potentially high-risk accidents are possible</u>.</p></li>
# ><li>In terms of <b>gender</b>, the general trend is the same, but males have a higher accident levels than females.
#     <p>Same as discussion above, it is thought that <u>this is due to different safety level by industry sector</u>.</p></li>
# ></ul>
# ></div>

# In[ ]:


f = lambda x : np.round(x/x.sum() * 100)
ac_gen = df.groupby(['Gender','Accident Level'])['Accident Level'].count().unstack().apply(f, axis=1)
ac = hv.Bars(pd.melt(ac_gen.reset_index(), ['Gender']), ['Gender','Accident Level'], 'value').opts(opts.Bars(title="Accident Level by Gender Count"))
pot_ac_gen = df.groupby(['Gender','Potential Accident Level'])['Potential Accident Level'].count().unstack().apply(f, axis=1)
pot_ac = hv.Bars(pd.melt(pot_ac_gen.reset_index(), ['Gender']), ['Gender','Potential Accident Level'], 'value').opts(opts.Bars(title="Potential Accident Level by Gender Count"))
(ac + pot_ac).opts(opts.Bars(width=400, height=300,tools=['hover'],show_grid=True,xrotation=0, ylabel="Percentage", yformatter='%d%%'))


# ### Accident Levels by Employee type
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>For both accident levels, the incidence of Employee is higher at low accident levels, but <u>the incidence of Third parties seems to be slightly higher at <b>high accident levels</b></u>.</li>
# ></ul>
# ></div>

# In[ ]:


f = lambda x : np.round(x/x.sum() * 100)
ac_em = df.groupby(['Employee type','Accident Level'])['Accident Level'].count().unstack().apply(f, axis=1)
ac = hv.Bars(pd.melt(ac_em.reset_index(), ['Employee type']), ['Employee type','Accident Level'], 'value').opts(opts.Bars(title="Accident Level by Employee type Count"))
pot_ac_em = df.groupby(['Employee type','Potential Accident Level'])['Potential Accident Level'].count().unstack().apply(f, axis=1)
pot_ac = hv.Bars(pd.melt(pot_ac_em.reset_index(), ['Employee type']), ['Employee type','Potential Accident Level'], 'value').opts(opts.Bars(title="Potential Accident Level by Employee type Count"))
(ac + pot_ac).opts(opts.Bars(width=400, height=300,tools=['hover'],show_grid=True,xrotation=0, ylabel="Percentage", yformatter='%d%%',fontsize={'title':9}))


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ### Accident Levels by Month
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>Both of the two accident level have the tendency that non-severe levels decreased throughout the year, <u>but severe levels did not changed much, and some of these levels increased slightly in the second half of the year</u>.</li>
# ><li>The fact above seems to be related to <b>the skill level of the employees</b>, and <u>while their experiences can reduce minor mistakes, sometimes they can make serious mistakes accidentally</u>.</li>
# ></ul>
# ></div>

# In[ ]:


f = lambda x : np.round(x/x.sum() * 100)
ac_mo = df.groupby(['Month','Accident Level'])['Accident Level'].count().unstack().apply(f, axis=1).fillna(0)
ac = hv.Curve(ac_mo['I'], label='I') * hv.Curve(ac_mo['II'], label='II') * hv.Curve(ac_mo['III'], label='III') * hv.Curve(ac_mo['IV'], label='IV') * hv.Curve(ac_mo['V'], label='V')        .opts(opts.Curve(title="Accident Level by Month Count"))
pot_ac_mo = df.groupby(['Month','Potential Accident Level'])['Potential Accident Level'].count().unstack().apply(f, axis=1).fillna(0)
pot_ac = hv.Curve(pot_ac_mo['I'], label='I') * hv.Curve(pot_ac_mo['II'], label='II') * hv.Curve(pot_ac_mo['III'], label='III') * hv.Curve(pot_ac_mo['IV'], label='IV')        * hv.Curve(pot_ac_mo['V'], label='V') * hv.Curve(pot_ac_mo['VI'], label='VI').opts(opts.Curve(title="Potential Accident Level by Month Count"))
(ac+pot_ac).opts(opts.Curve(width=800, height=300,tools=['hover'],show_grid=True, ylabel="Percentage", yformatter='%d%%')).cols(1)


# ### Accident Levels by Weekday
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>Both of the two accident level is thought that non-severe levels decreased in the first and the last of the week, but severe levels did not changed much.</li>
# ><li>It can be said that <u>employees' experiences against work can reduce minor mistakes</u>.</li>
# ></ul>
# ></div>

# In[ ]:


f = lambda x : np.round(x/x.sum() * 100)
ac_weekday = df.groupby(['Weekday','Accident Level'])['Accident Level'].count().unstack().apply(f, axis=1).fillna(0)
ac_weekday['week_num'] = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(i) for i in ac_weekday.index]
ac_weekday.sort_values('week_num', inplace=True)
ac_weekday.drop('week_num', axis=1, inplace=True)
ac = hv.Curve(ac_weekday['I'], label='I') * hv.Curve(ac_weekday['II'], label='II') * hv.Curve(ac_weekday['III'], label='III') * hv.Curve(ac_weekday['IV'], label='IV') * hv.Curve(ac_weekday['V'], label='V')        .opts(opts.Curve(title="Accident Level by Weekday Count"))
pot_ac_weekday = df.groupby(['Weekday','Potential Accident Level'])['Potential Accident Level'].count().unstack().apply(f, axis=0).fillna(0)
pot_ac_weekday['week_num'] = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(i) for i in pot_ac_weekday.index]
pot_ac_weekday.sort_values('week_num', inplace=True)
pot_ac_weekday.drop('week_num', axis=1, inplace=True)
pot_ac = hv.Curve(pot_ac_weekday['I'], label='I') * hv.Curve(pot_ac_weekday['II'], label='II') * hv.Curve(pot_ac_weekday['III'], label='III') * hv.Curve(pot_ac_weekday['IV'], label='IV')        * hv.Curve(pot_ac_weekday['V'], label='V') * hv.Curve(pot_ac_weekday['VI'], label='VI').opts(opts.Curve(title="Potential Accident Level by Weekday Count"))
(ac+pot_ac).opts(opts.Curve(width=800, height=300,tools=['hover'],show_grid=True, ylabel="Percentage", yformatter='%d%%')).cols(1)


# ### Accident Levels by Season
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>As same as accident levels by month, both of the two accident level have the tendency that non-severe levels decreased throughout the year, <u>but severe levels did not changed much, and some of these levels increased slightly in the second half of the year</u>.</li>
# ></ul>
# ></div>

# In[ ]:


f = lambda x : np.round(x/x.sum() * 100)
ac_season = df.groupby(['Season','Accident Level'])['Accident Level'].count().unstack().apply(f, axis=1).fillna(0)
ac_season['season_num'] = [['Spring', 'Summer', 'Autumn', 'Winter'].index(i) for i in ac_season.index]
ac_season.sort_values('season_num', inplace=True)
ac_season.drop('season_num', axis=1, inplace=True)
ac = hv.Curve(ac_season['I'], label='I') * hv.Curve(ac_season['II'], label='II') * hv.Curve(ac_season['III'], label='III') * hv.Curve(ac_season['IV'], label='IV') * hv.Curve(ac_season['V'], label='V')        .opts(opts.Curve(title="Accident Level by Season Count"))
pot_ac_season = df.groupby(['Season','Potential Accident Level'])['Potential Accident Level'].count().unstack().apply(f, axis=0).fillna(0)
pot_ac_season['season_num'] = [['Spring', 'Summer', 'Autumn', 'Winter'].index(i) for i in pot_ac_season.index]
pot_ac_season.sort_values('season_num', inplace=True)
pot_ac_season.drop('season_num', axis=1, inplace=True)
pot_ac = hv.Curve(pot_ac_season['I'], label='I') * hv.Curve(pot_ac_season['II'], label='II') * hv.Curve(pot_ac_season['III'], label='III') * hv.Curve(pot_ac_season['IV'], label='IV')        * hv.Curve(pot_ac_season['V'], label='V') * hv.Curve(pot_ac_season['VI'], label='VI').opts(opts.Curve(title="Potential Accident Level by Season Count"))
(ac+pot_ac).opts(opts.Curve(width=800, height=300,tools=['hover'],show_grid=True, ylabel="Percentage", yformatter='%d%%')).cols(1)


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ## NLP Analysis
# >Description about accidents is important to understand the cause of accidents, so we need to discover characteristical words or phrases indicating situation when accidents occured.

# >function to calculate ngram under several conditions

# In[ ]:


def ngram_func(ngram, trg='', trg_value=''):
    #trg_value is list-object
    if (trg == '') or (trg_value == ''):
        string_filterd =  df['Description_processed'].sum().split()
    else:
        string_filterd =  df[df[trg].isin(trg_value)]['Description_processed'].sum().split()
    dic = nltk.FreqDist(nltk.ngrams(string_filterd, ngram)).most_common(30)
    ngram_df = pd.DataFrame(dic, columns=['ngram','count'])
    ngram_df.index = [' '.join(i) for i in ngram_df.ngram]
    ngram_df.drop('ngram',axis=1, inplace=True)
    return ngram_df


# ### Unigram
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>There are several words which is related to <b>hands</b>. For example <u>left, hand, right and finger</u>.</li>
# ><li>Moreover there are several words which is related to <b>movement of something</b>. For example <u>hit, remov, fall and move</u>.</li>
# ></ul>
# ></div>

# In[ ]:


hv.Bars(ngram_func(1)[::-1]).opts(title="Unigram Count top-30", color="red", xlabel="Unigrams", ylabel="Count")                .opts(opts.Bars(width=600, height=600,tools=['hover'],show_grid=True,invert_axes=True))


# ### Bigram
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>There are so many phrases which is related to <b>hands</b>. For example <u>left hand, right hand, finger left, finger right, middl finger and ring finger</u>.</li>
# ><li>There are also some phrases which is related to other body parts. For example <u>left foot and right reg</u>.</li>
# ></ul>
# ></div>

# In[ ]:


hv.Bars(ngram_func(2)[::-1]).opts(title="Bigram Count top-30", color="yellow", xlabel="Bigrams", ylabel="Count")                .opts(opts.Bars(width=600, height=600,tools=['hover'],show_grid=True,invert_axes=True))


# ### Trigram
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>Like Unigram and Bigram, there are also many phrases which is related to <b>hands or other body parts</b>, but concreteness seems to increase.</li>
# ><li>For example <u>one hand glove, left arm uniform and wear safeti uniform</u>.</li>
# ></ul>
# ></div>

# In[ ]:


hv.Bars(ngram_func(3)[::-1]).opts(title="Trigram Count top-30", color="blue", xlabel="Trigrams", ylabel="Count")                .opts(opts.Bars(width=600, height=600,tools=['hover'],show_grid=True,invert_axes=True))


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ### Ngram with Gender

# In[ ]:


uni_ma=hv.Bars(ngram_func(1, 'Gender', ['Male'])[0:15][::-1]).opts(title="Unigram with Male", color="red", xlabel="Unigrams", ylabel="Count")
uni_fe=hv.Bars(ngram_func(1, 'Gender', ['Female'])[0:15][::-1]).opts(title="Unigram with Female", color="red", xlabel="Unigrams", ylabel="Count")

bi_ma=hv.Bars(ngram_func(2, 'Gender', ['Male'])[0:15][::-1]).opts(title="Bigram with Male", color="yellow", xlabel="Bigrams", ylabel="Count")
bi_fe=hv.Bars(ngram_func(2, 'Gender', ['Female'])[0:15][::-1]).opts(title="Bigram with Female", color="yellow", xlabel="Bigrams", ylabel="Count")

tri_ma=hv.Bars(ngram_func(3, 'Gender', ['Male'])[0:15][::-1]).opts(title="Trigram with Male", color="blue", xlabel="Trigrams", ylabel="Count")
tri_fe=hv.Bars(ngram_func(3, 'Gender', ['Female'])[0:15][::-1]).opts(title="Trigram with Female", color="blue", xlabel="Trigrams", ylabel="Count")
                

(uni_ma + uni_fe + bi_ma + bi_fe + tri_ma + tri_fe).opts(opts.Bars(width=400, height=300,tools=['hover'],show_grid=True,invert_axes=True, shared_axes=False)).opts(shared_axes=False).cols(2)


# ### Ngram with Accident Level
# >Classifing accident levels into two part, Low Accident Level(I,II) and High Accident Level(III,IV,V).

# In[ ]:


uni_ac_lo=hv.Bars(ngram_func(1, 'Accident Level', ['I','II'])[0:15][::-1]).opts(title="Unigram with High Accident Level", color="red", xlabel="Unigrams", ylabel="Count")
uni_ac_hi=hv.Bars(ngram_func(1, 'Accident Level', ['III','IV','V'])[0:15][::-1]).opts(title="Unigram with Low Accident Level", color="red", xlabel="Unigrams", ylabel="Count")

bi_ac_lo=hv.Bars(ngram_func(2, 'Accident Level', ['I','II'])[0:15][::-1]).opts(title="Bigram with High Accident Level", color="yellow", xlabel="Bigrams", ylabel="Count")
bi_ac_hi=hv.Bars(ngram_func(2, 'Accident Level', ['III','IV','V'])[0:15][::-1]).opts(title="Bigram with Low Accident Level", color="yellow", xlabel="Bigrams", ylabel="Count")

tri_ac_lo=hv.Bars(ngram_func(3, 'Accident Level', ['I','II'])[0:15][::-1]).opts(title="Trigram with High Accident Level", color="blue", xlabel="Trigrams", ylabel="Count")
tri_ac_hi=hv.Bars(ngram_func(3, 'Accident Level', ['III','IV','V'])[0:15][::-1]).opts(title="Trigram with Low Accident Level", color="blue", xlabel="Trigrams", ylabel="Count")
                
(uni_ac_lo + uni_ac_hi + bi_ac_lo + bi_ac_hi + tri_ac_lo + tri_ac_hi).opts(opts.Bars(width=400, height=300,tools=['hover'],show_grid=True,invert_axes=True, shared_axes=False)).opts(shared_axes=False).cols(2)


# ### Ngram with Industry Sector

# In[ ]:


uni_mine=hv.Bars(ngram_func(1, 'Industry Sector', ['Mining'])[0:15][::-1]).opts(title="Unigram with Mining Sector", color="red", xlabel="Unigrams", ylabel="Count")
uni_metal=hv.Bars(ngram_func(1, 'Industry Sector', ['Metals'])[0:15][::-1]).opts(title="Unigram with Metal Sector", color="red", xlabel="Unigrams", ylabel="Count")
uni_others=hv.Bars(ngram_func(1, 'Industry Sector', ['Others'])[0:15][::-1]).opts(title="Unigram with Other Sector", color="red", xlabel="Unigrams", ylabel="Count")

bi_mine=hv.Bars(ngram_func(2, 'Industry Sector', ['Mining'])[0:15][::-1]).opts(title="Bigram with Mining Sector", color="yellow", xlabel="Bigrams", ylabel="Count")
bi_metal=hv.Bars(ngram_func(2, 'Industry Sector', ['Metals'])[0:15][::-1]).opts(title="Bigram with Metal Sector", color="yellow", xlabel="Bigrams", ylabel="Count")
bi_others=hv.Bars(ngram_func(2, 'Industry Sector', ['Others'])[0:15][::-1]).opts(title="Bigram with Other Sector", color="yellow", xlabel="Bigrams", ylabel="Count")

tri_mine=hv.Bars(ngram_func(3, 'Industry Sector', ['Mining'])[0:15][::-1]).opts(title="Trigram with Mining Sector", color="blue", xlabel="Trigrams", ylabel="Count")
tri_metal=hv.Bars(ngram_func(3, 'Industry Sector', ['Metals'])[0:15][::-1]).opts(title="Trigram with Metal Sector", color="blue", xlabel="Trigrams", ylabel="Count")
tri_others=hv.Bars(ngram_func(3, 'Industry Sector', ['Others'])[0:15][::-1]).opts(title="Trigram with Other Sector", color="blue", xlabel="Trigrams", ylabel="Count")

(uni_mine + uni_metal + uni_others + bi_mine + bi_metal + bi_others + tri_mine + tri_metal + tri_others)            .opts(opts.Bars(width=265, height=300,tools=['hover'],show_grid=True,invert_axes=True, shared_axes=False,fontsize={'title':6.5,'labels':7,'yticks':8.5})).opts(shared_axes=False).cols(3)


# ### Ngram with Employee type

# In[ ]:


uni_emp=hv.Bars(ngram_func(1, 'Employee type', ['Employee'])[0:15][::-1]).opts(title="Unigram with Employee", color="red", xlabel="Unigrams", ylabel="Count")
uni_third=hv.Bars(ngram_func(1, 'Employee type', ['Third Party','Third Party (Remote)'])[0:15][::-1]).opts(title="Unigram with Third Party", color="red", xlabel="Unigrams", ylabel="Count")

bi_emp=hv.Bars(ngram_func(2, 'Employee type', ['Employee'])[0:15][::-1]).opts(title="Bigram with Employee", color="yellow", xlabel="Bigrams", ylabel="Count")
bi_third=hv.Bars(ngram_func(2, 'Employee type', ['Third Party','Third Party (Remote)'])[0:15][::-1]).opts(title="Bigram with Third Party", color="yellow", xlabel="Bigrams", ylabel="Count")

tri_emp=hv.Bars(ngram_func(3, 'Employee type', ['Employee'])[0:15][::-1]).opts(title="Trigram with Employee", color="blue", xlabel="Trigrams", ylabel="Count")
tri_third=hv.Bars(ngram_func(3, 'Employee type', ['Third Party','Third Party (Remote)'])[0:15][::-1]).opts(title="Trigram with Third Party", color="blue", xlabel="Trigrams", ylabel="Count")

(uni_emp + uni_third+ bi_emp + bi_third + tri_emp + tri_third).opts(opts.Bars(width=400, height=300,tools=['hover'],show_grid=True,invert_axes=True, shared_axes=False)).opts(shared_axes=False).cols(2)


# ### WordCloud
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>As same as Ngram analysis above, there are many hand-related and movement-related words.</li>
# ><li><b>Hand-related</b> : left, right, hand, finger and glove</li>
# ><li><b>Movement-related</b> : fall, hit, carri, lift and slip</li>
# ></ul>
# ></div>

# In[ ]:


wordcloud = WordCloud(width = 1500, height = 800, random_state=0, background_color='black', colormap='rainbow',                       min_font_size=5, max_words=300, collocations=False, min_word_length=3, stopwords = STOPWORDS).generate(" ".join(df['Description_processed'].values))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# # 6. Modeling
# ><div class="alert alert-success" role="alert">
# >Objectives:
# ><ul>
# ><li>Presumption of cause of accidents</li>
# ><li>Surveying a factor that increases severity of accidents</li>
# ></ul>
# >Building the model which classify the severity of accidents, we can understand the factor related to the causality of accidents.<br/>
# >So, two models were built based on those cases below.
# ><ul>
# ><li><a href='#ac'>Case1 : Accident Level</a></li>
# ><li><a href='#pac'>Case2 : Potential Accident Level</a></li>
# ></ul>
# ></div>

# ## Feature Engineering

# ### TFIDF Feature

# In[ ]:


feature_df = pd.DataFrame()
for i in [1,2,3]:
    vec_tfidf = TfidfVectorizer(max_features=10, norm='l2', stop_words='english', lowercase=True, use_idf=True, ngram_range=(i,i))
    X = vec_tfidf.fit_transform(df['Description_processed']).toarray()
    tfs = pd.DataFrame(X, columns=["TFIDF_" + n for n in vec_tfidf.get_feature_names()])
    feature_df = pd.concat([feature_df, tfs], axis=1)
feature_df = pd.concat([df, feature_df], axis=1)
feature_df.head(3)


# ### Label Encoding

# In[ ]:


feature_df['Country'] = LabelEncoder().fit_transform(feature_df['Country']).astype(np.int8)
feature_df['Local'] = LabelEncoder().fit_transform(feature_df['Local']).astype(np.int8)
feature_df['Industry Sector'] = LabelEncoder().fit_transform(feature_df['Industry Sector']).astype(np.int8)
feature_df['Accident Level'] = LabelEncoder().fit_transform(feature_df['Accident Level']).astype(np.int8)
feature_df['Potential Accident Level'] = LabelEncoder().fit_transform(feature_df['Potential Accident Level']).astype(np.int8)
feature_df['Gender'] = LabelEncoder().fit_transform(feature_df['Gender']).astype(np.int8)
feature_df['Employee type'] = LabelEncoder().fit_transform(feature_df['Employee type']).astype(np.int8)
feature_df['Critical Risk'] = LabelEncoder().fit_transform(feature_df['Critical Risk']).astype(np.int8)
feature_df['Weekday'] = LabelEncoder().fit_transform(feature_df['Weekday']).astype(np.int8)
feature_df['Season'] = LabelEncoder().fit_transform(feature_df['Season']).astype(np.int8)
feature_df.drop(['Date','Description', 'Description_processed'],axis=1,inplace=True)
feature_df.head(3)


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ## Case1 : Accident Level<a id='ac'></a>
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>There are many time-series features with high importance such as <b>Day, Weekday and Month</b>, and it is thought that <u>the occurrence of accidents and the accident level will change  easily depending on the time</u>.</li>
# ><li>Since there are many TFIDF features with high importance related to a part of the body, and in particular many features are related to the hands such as <b>hand, left and right</b>, so it is considered that <u>mistakes in manual work are related to the occurrence and severity of accidents</u>.</li>
# ></ul>
# ></div>

# In[ ]:


y_series = feature_df['Accident Level']
x_df = feature_df.drop(['Accident Level'], axis=1) 
X_train, X_valid, Y_train, Y_valid = train_test_split(x_df, y_series, test_size=0.2, random_state=0, stratify=y_series)

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_valid = lgb.Dataset(X_valid, Y_valid, reference=lgb_train)


# In[ ]:


params = {
    'task' : 'train',
    'boosting' : 'gbdt',
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',
    'num_leaves': 200,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 5
}
gbm_ac = lgb.train(params,
            lgb_train,
            num_boost_round=100,
            valid_sets=lgb_valid,
            early_stopping_rounds=100)


# In[ ]:


feature_imp_ac = pd.DataFrame()
feature_imp_ac['feature'] = gbm_ac.feature_name()
feature_imp_ac['importance'] = gbm_ac.feature_importance()
hv.Bars(feature_imp_ac.sort_values(by='importance', ascending=False)[::-1]).opts(title="Feature Importance in Accident Level", color="purple", xlabel="Features", ylabel="Importance", invert_axes=True)                            .opts(opts.Bars(width=700, height=700, tools=['hover'], show_grid=True))


# In[ ]:


t = lgb.plot_tree(gbm_ac, figsize=(20, 20), precision=1, tree_index=0, show_info=['split_gain'])
plt.title('Visulalization of Tree in Accident Level')
plt.show()


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ## Case2 : Potential Accident Level<a id='pac'></a>
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>Similar to the model of Accident Level above, the features of time series and TFIDF features of hands are highly important, but the TFIDF features seem to be slightly more important.</li>
# ></ul>
# ></div>

# In[ ]:


_feature_df = feature_df[~feature_df['Potential Accident Level'].isin([5])]
y_series = _feature_df['Potential Accident Level']
x_df = _feature_df.drop(['Potential Accident Level'], axis=1) 
X_train, X_valid, Y_train, Y_valid = train_test_split(x_df, y_series, test_size=0.2, random_state=0, stratify=y_series)

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_valid = lgb.Dataset(X_valid, Y_valid, reference=lgb_train)


# In[ ]:


params = {
    'task' : 'train',
    'boosting' : 'gbdt',
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',
    'num_leaves': 200,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 5
}
gbm_pac = lgb.train(params,
            lgb_train,
            num_boost_round=100,
            valid_sets=lgb_valid,
            early_stopping_rounds=100)


# In[ ]:


feature_imp_pac = pd.DataFrame()
feature_imp_pac['feature'] = gbm_pac.feature_name()
feature_imp_pac['importance'] = gbm_pac.feature_importance()
hv.Bars(feature_imp_pac.sort_values(by='importance', ascending=False)[::-1]).opts(title="Feature Importance in Potential Accident Level", color="purple", xlabel="Features", ylabel="Importance", invert_axes=True)                            .opts(opts.Bars(width=700, height=700, tools=['hover'], show_grid=True))


# In[ ]:


t = lgb.plot_tree(gbm_pac, figsize=(20, 20), precision=1, tree_index=0, show_info=['split_gain'])
plt.title('Visulalization of Tree in Potential Accident Level')
plt.show()


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# # 7. Conclusion
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>In this project, we discovered that the main causes of accidents are <b>mistakes in hand-operation and time-related factor</b>.</li>
# ><li>To reduce the occurrences of accidents, <u>more stringent safety standards in hand-operation will be needed in period when many accidents occur</u>.</li>
# ></ul>
# ><ul>
# ><li>I realized that the detail information of accidents like 'Description' are so useful to analyze the cause.</li>
# ><li>With more detailed information such as <b>machining data(ex. CNC, Current, Voltage) in plants, weather information, employee's personal data(ex. age, experience in the industry sector, work performance
# )</b>, we can clarify the cause of accidents more correctly.</li>
# ></ul>
# ></div>

# ## Task Submission
# >Through the EDA & Modeling above, we can answer [several tasks](https://www.kaggle.com/ihmstefanini/industrial-safety-and-health-analytics-database/tasks).

# ### Gender mostly involved in accidents
# ><div class="alert alert-info" role="alert">
# >In <a href='https://www.kaggle.com/ihmstefanini/industrial-safety-and-health-analytics-database/tasks?taskId=240'>this task</a>, the question is : <b>Which gender is mostly involved in accidents at these plants?</b><br/>
# >Answer : <b>Male</b><br/><br/>
# ><ul>
# ><li>Though the staffs of the manufacturing plants are mostly males, <u>EDA shows that <b>males</b> are likely involved in accidents(95%)</u>.</li>
# ><li>And males are tend to get involved in accidents with higher risk levels than females.</li>
# ></ul>
# ></div>

# In[ ]:


f1 = lambda x : np.round(x/len(df) * 100)
gender_cnt = df.groupby(['Gender'])['Accident Level'].count().apply(f1)
g = hv.Bars(pd.melt(gender_cnt.reset_index(), ['Gender']), ['Gender'], 'value').opts(opts.Bars(title="Gender Count", color='green'))

f2 = lambda x : np.round(x/x.sum() * 100)
ac_gen = df.groupby(['Gender','Accident Level'])['Accident Level'].count().unstack().apply(f2, axis=1)
ac = hv.Bars(pd.melt(ac_gen.reset_index(), ['Gender']), ['Gender','Accident Level'], 'value').opts(opts.Bars(title="Accident Level by Gender Count"))

(g + ac).opts(opts.Bars(width=400, height=300,tools=['hover'],show_grid=True,xrotation=0, ylabel="Percentage", yformatter='%d%%', shared_axes=True))


# ### Third Parties Or Employees?
# ><div class="alert alert-info" role="alert">
# >In <a href='https://www.kaggle.com/ihmstefanini/industrial-safety-and-health-analytics-database/tasks?taskId=241'>this task</a>, the question is : <b>Are third parties usually involved in these accidents or it is mainly the employees?</b><br/>
# >Answer : <b>Yes, they are. Third parties more likely get involved in accidents.</b><br/><br/>
# ><ul>
# ><li>Comparing employee's accidents count with third parties' accidents count, <u>EDA shows that <b>third parties</b> are likely involved in accidents(58%)</u>.</li>
# ><li>And third parties are slightly tend to get involved in accidents with higer risk levels than employee.</li>
# ></ul>
# ></div>

# In[ ]:


df_em_tmp = df.copy()
df_em_tmp.loc[df_em_tmp['Employee type'].isin(['Third Party','Third Party (Remote)']), 'Employee type'] = 'Third Party(+Remote)'
f1 = lambda x : np.round(x/len(df_em_tmp) * 100)
emp_type_cnt = df_em_tmp.groupby(['Employee type'])['Accident Level'].count().apply(f1)
g = hv.Bars(pd.melt(emp_type_cnt.reset_index(), ['Employee type']), ['Employee type'], 'value').opts(opts.Bars(title="Employee type Count", color='green'))


f2 = lambda x : np.round(x/x.sum() * 100)
ac_em = df_em_tmp.groupby(['Employee type','Accident Level'])['Accident Level'].count().unstack().apply(f2, axis=1)
ac = hv.Bars(pd.melt(ac_em.reset_index(), ['Employee type']), ['Employee type','Accident Level'], 'value').opts(opts.Bars(title="Accident Level by Employee type Count"))
(g + ac).opts(opts.Bars(width=400, height=300,tools=['hover'],show_grid=True,xrotation=0, ylabel="Percentage", yformatter='%d%%',fontsize={'title':9}))


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ### Main cause of accidents
# ><div class="alert alert-info" role="alert">
# >In <a href='https://www.kaggle.com/ihmstefanini/industrial-safety-and-health-analytics-database/tasks?taskId=242'>this task</a>, the question is : <b>What usually causes these accidents?</b><br/>
# >Answer : <b>Mistakes in hands operations.</b><br/><br/>
# ><ul>
# ><li>According to Ngram analysis, we can say that <u>operations related to hands are mainly the causes of accidents</u>.</li>
# ><li>According to the modeling to classify accident levels shows that in addition to hands-operation, time-related features also affect to the occurrence of accidents.</li>
# ></ul>
# ></div>

# In[ ]:


uni=hv.Bars(ngram_func(1)[0:15][::-1]).opts(title="Unigram Count", color="red", xlabel="Unigrams", ylabel="Count")
bi=hv.Bars(ngram_func(2)[0:15][::-1]).opts(title="Bigram Count", color="yellow", xlabel="Bigrams", ylabel="Count")
tri=hv.Bars(ngram_func(3)[0:15][::-1]).opts(title="Trigram Count", color="blue", xlabel="Trigrams", ylabel="Count")
(uni + bi + tri).opts(opts.Bars(width=265, height=300,tools=['hover'],show_grid=True,invert_axes=True, shared_axes=False)).opts(shared_axes=False)


# In[ ]:


ac=hv.Bars(feature_imp_ac.sort_values(by='importance', ascending=False)[0:15][::-1]).opts(title="Feature Importance in Accident Level")
pac=hv.Bars(feature_imp_pac.sort_values(by='importance', ascending=False)[0:15][::-1]).opts(title="Feature Importance in Potential Accident Level")
(ac + pac).opts(opts.Bars(width=400, height=300, tools=['hover'], show_grid=True,color="purple", xlabel="Features", ylabel="Importance", invert_axes=True,fontsize={'title':9})).opts(shared_axes=False)


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# # 8. References
# >* **Good EDA Notebook**  
# >https://www.kaggle.com/schorsi/industrial-safety-totw  
# >https://www.kaggle.com/schorsi/industrial-safety-totw-part-2
# >* **Pandas value_counts() tips**  
# >https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html
# >* **Holoviews plot tips**  
# >http://holoviews.org/user_guide/Customizing_Plots.html
# >* **NLP Pre-processing tutorial**  
# >http://haya14busa.com/python-nltk-natural-language-processing/
# >* **WORDCLOUD example**  
# >https://towardsdatascience.com/simple-wordcloud-in-python-2ae54a9f58e5  
# >* **LightGBM Parameters**  
# >https://lightgbm.readthedocs.io/en/latest/Parameters.html
# >* **LightGBM Tree Visulizing**  
# >https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_tree.html

# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>
