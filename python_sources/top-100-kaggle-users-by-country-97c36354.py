#!/usr/bin/env python
# coding: utf-8

# The purpose of this script is to show that it would be nice to have **country** and **city** data available in public Meta Kaggle database. To get such info for this notebook I did web scrapping on Kaggle for only top-100 Kaggle users. I hope Kaggle admins wouldn't be angry to me for that.
# 
# There are some ideas of how to use it:<br>
# 1. For each competition calculate distribution by country. e.g. recent one from TalkingData may involve more Data Scientist from China.<br>
# 2. Analyze Teams to understand why they are connected. Probably because they are living in one city.<br>
# 3. See what type of competitions involve people from which country.<br>
# and etc.
# 
# This is my first Python Notebook on Kaggle and there were some troubles to run my code from local Notebook because of libraries. If you know some best practice to improve code, it would be nice to share.

# ## Load top-100 users to the DataFrame ##

# In[ ]:


import pandas as pd
import sqlite3

# Connect to the DB
con = sqlite3.connect('../input/database.sqlite')

# Exec select query
top100users = pd.read_sql_query("""
SELECT u.Id, u.UserName, u.DisplayName, cast(u.Ranking as INTEGER) as Ranking
FROM Users u
WHERE u.Ranking <= 100
ORDER BY u.Ranking""", con)

# Show Top-5
top100users.head(5)


# ## Web scraping code to get countries ##

# It doesn't work here, that is why I've commented it and would be using dictionary with preloaded data instead later. You could copy it and run locally.

# In[ ]:


#from bs4 import BeautifulSoup
#import re
#import requests
#from lxml import html

#geo_info = []
#for num in range(top100users.shape[0]):
#    response = requests.get('http://www.kaggle.com/{}'.format(top100users.loc[num, 'UserName']))
#    raw_text = str(BeautifulSoup(response.text, "lxml")).replace('"', '')
#    print re.findall('(country:[A-Za-z\s\n]+)', raw_text)[0][8:]
#    geo_info.append([top100users.loc[num, 'UserName'], 
#                     re.findall('(country:[A-Za-z\s\n]+)', raw_text)[0][8:]])
#top100users = pd.merge(top100users, pd.DataFrame(geo_info, columns=['UserName', 'Country']), how='left', on='UserName')
   


# Do some Country cleaning:<br>
#  1. Translate alpha-2 code to ordinal country names;<br>
#  2. Map some countries to the standard names;<br>
#  3. Capitalize letters.<br>
# <br>
# This is also commented because we don't have results from previous step.

# In[ ]:


#import pycountry as pc

## Generate dict to translate Alpha2 Code to Country Name
#alpha2_code = dict()
#for country in list(pc.countries):
#    alpha2_code[country.alpha2] = country.name

## Dict to mapping some non-standard names
#extraCountry = {'Russia':'Russian Federation',
#                'USA':'United States'}

## Apply mappings
#top100users.loc[top100users['Country'].isin(extraCountry.keys()), 'Country'] = map(lambda x: extraCountry[x], top100users.loc[top100users['Country'].isin(extraCountry.keys()), 'Country'])
#top100users.loc[top100users['Country'].isin(alpha2_code.keys()), 'Country'] = map(lambda x: alpha2_code[x], top100users.loc[top100users['Country'].isin(alpha2_code.keys()), 'Country'])
#top100users.loc[:, 'Country'] = map(lambda x: x.title(), top100users['Country'])
#top100users.head(5)


# Prepare dictionary to use it further instead of Kaggle scrapping.

# In[ ]:


#top100users[['Id', 'Country']].set_index('Id').to_dict()


# ## Define country dictionary for top-100 users ##

# In[ ]:


dict_country = {808: 'Russian Federation',  1455: 'Austria',  1483: 'Iran',  2036: 'Netherlands',  2242: 'United Kingdom',
  3090: 'Russian Federation',  3230: 'Japan',  4398: 'United States',  5309: 'Germany',  5635: 'Poland',  5642: 'Spain',
  6388: u'Japan',  6603: 'Japan',  7756: 'United States',  9766: 'United States',  9974: 'The Netherlands',
  10171: 'United States',  12260: 'Finland',  12584: 'United States',  16398: 'Null',  17379: 'Singapore',
  18102: 'Hungary',  18396: 'United States',  18785: 'United States',  19099: 'Russian Federation',  19298: u'Italy',
  19605: 'United States',  24266: 'Brazil',  27805: 'Russian Federation',  29346: 'Uae',  29756: u'Netherlands',
  31529: 'Croatia',  33467: 'Ukraine',  34304: 'Israel',  38113: 'Netherlands',  41959: 'India',  42188: 'Germany',
  42832: 'Null',  43621: 'United States',  48625: 'United States',  54836: u'Brazil',  56101: 'France',
  58838: 'South Korea',  59561: 'United States',  64274: u'United States',  64626: 'United States',  68889: u'Israel',
  70574: u'United States',  71388: u'India',  73703: 'Switzerland',  77226: 'United States',  85195: 'United States',
  90001: 'Singapore',  90646: 'Spain',  93420: 'United States',  94510: 'Turkey',  98575: 'United States',  99029: 'Turkey',
  100236: u'United States',  102203: u'China',  104698: u'United Kingdom',  105240: 'United States',  106249: 'Brazil',
  111066: 'Japan',  111640: u'Greece',  113389: 'India',  114032: 'United States',  114978: 'The Netherlands',
  116125: 'France',  147404: 'United States',  149229: 'Russian Federation',  149814: 'Null',  150865: 'United States',
  160106: u'Japan',  161151: u'Russian Federation',  163663: 'United States',  168767: 'Null',  170170: 'China',
  189197: 'Canada',  194108: 'United States',  200451: 'Russian Federation',  210078: u'Germany',  217312: u'Canada',
  218203: 'Japan',  221419: 'Null',  226693: u'Russian Federation',  254602: 'United States',  263583: u'Mexico',
  266958: 'Slovakia',  269623: 'Russian Federation',  275512: 'Israel',  275730: u'Germany',  278920: 'United States',
  300713: 'India',  312728: 'United States',  338701: 'Canada',  356943: u'India',  384014: 'Germany',
  405318: 'Lithuania',  582611: 'Belgium'}

population = {'Aruba':103889,'Andorra':70473,'Afghanistan':32526562,'Angola':25021974,'Albania':2889167,'Arab World':392022276,'United Arab Emirates':9156963,'Argentina':43416755,'Armenia':3017712,'American Samoa':55538,'Antigua and Barbuda':91818,'Australia':23781169,'Austria':8611088,'Azerbaijan':9651349,'Burundi':11178921,'Belgium':11285721,'Benin':10879829,'Burkina Faso':18105570,'Bangladesh':160995642,'Bulgaria':7177991,'Bahrain':1377237,'BahamasThe':388019,'Bosnia and Herzegovina':3810416,'Belarus':9513000,'Belize':359287,'Bermuda':65235,'Bolivia':10724705,'Brazil':207847528,'Barbados':284215,'Brunei Darussalam':423188,'Bhutan':774830,'Botswana':2262485,'Central African Republic':4900274,'Canada':35851774,'Central Europe and the Baltics':103318638,'Switzerland':8286976,'Channel Islands':163692,'Chile':17948141,'China':1371220000,'Cote Ivoire':22701556,'Cameroon':23344179,'CongoRep':4620330,'Colombia':48228704,'Comoros':788474,'Cabo Verde':520502,'Costa Rica':4807850,'Caribbean small states':7048966,'Cuba':11389562,'Curacao':158040,'Cayman Islands':59967,'Cyprus':1165300,'Czech Republic':10551219,'Germany':81413145,'Djibouti':887861,'Dominica':72680,'Denmark':5676002,'Dominican Republic':10528391,'Algeria':39666519,'East Asia  Pacific excluding high income':2035129646,'Early-demographic dividend':312270331737425,'East Asia  Pacific':2279186469,'Europe  Central Asia excluding high income':411338238,'Europe  Central Asia':907944124,'Ecuador':16144363,'EgyptArab Rep':91508084,'Euro area':339425073,'Spain':46418269,'Estonia':1311998,'Ethiopia':99390750,'European Union':509668361,'Fragile and conflict affected situations':485609230,'Finland':5482013,'Fiji':892145,'France':66808385,'Faroe Islands':48199,'MicronesiaFed Sts':104460,'Gabon':1725292,'United Kingdom':65138232,'Georgia':3679000,'Ghana':27409893,'Gibraltar':32217,'Guinea':12608590,'GambiaThe':1990924,'Guinea-Bissau':1844325,'Equatorial Guinea':845060,'Greece':10823732,'Grenada':106825,'Greenland':56114,'Guatemala':16342897,'Guam':169885,'Guyana':767085,'High income':1187189841,'Hong Kong SARChina':7305700,'Honduras':8075060,'Heavily indebted poor countries HIPC':721104105,'Croatia':4224404,'Haiti':10711067,'Hungary':9844686,'IBRD only':454258068837425,'IDA  IBRD total':618363483037425,'IDA total':1641054142,'IDA blend':585760189,'Indonesia':257563815,'IDA only':1055293953,'Isle of Man':87780,'India':1311050527,'Ireland':4640703,'IranIslamic Rep':79109272,'Iraq':36423395,'Iceland':330823,'Israel':8380400,'Italy':60802085,'Jamaica':2725941,'Jordan':7594547,'Japan':126958472,'Kazakhstan':17544126,'Kenya':46050302,'Kyrgyz Republic':5957000,'Cambodia':15577899,'Kiribati':112423,'St Kitts and Nevis':55572,'South Korea':50617045,'Kosovo':1797151,'Kuwait':3892115,'Latin America  Caribbean excluding high income':561948237,'Lao PDR':6802023,'Lebanon':5850743,'Liberia':4503438,'Libya':6278438,'St Lucia':184999,'Latin America  Caribbean':632959079,'Least developed countries: UN classification':954218054,'Low income':638286288,'Liechtenstein':37531,'Sri Lanka':20966000,'Lower middle income':2927414098,'Low  middle income':611602644137425,'Lesotho':2135022,'Late-demographic dividend':2249154750,'Lithuania':2910199,'Luxembourg':569676,'Latvia':1978440,'Macao SARChina':587606,'St Martin French part':31754,'Morocco':34377511,'Monaco':37731,'Moldova':3554150,'Madagascar':24235390,'Maldives':409163,'Middle East  North Africa':424065257,'Mexico':127017224,'Marshall Islands':52993,'Middle income':547774015337425,'MacedoniaFYR':2078453,'Mali':17599694,'Malta':431333,'Myanmar':53897154,'Middle East  North Africa excluding high income':362560941,'Montenegro':622388,'Mongolia':2959134,'Northern Mariana Islands':55070,'Mozambique':27977863,'Mauritania':4067564,'Mauritius':1262605,'Malawi':17215232,'Malaysia':30331007,'North America':357335829,'Namibia':2458830,'New Caledonia':273000,'Niger':19899120,'Nigeria':182201962,'Nicaragua':6082032,'Netherlands':16936520,'Norway':5195921,'Nepal':28513700,'Nauru':10222,'New Zealand':4595700,'OECD members':1280996600,'Oman':4490541,'Other small states':28650005,'Pakistan':188924874,'Panama':3929141,'Peru':31376670,'Philippines':100699395,'Palau':21291,'Papua New Guinea':7619321,'Poland':37999494,'Pre-demographic dividend':850271023,'Puerto Rico':3474182,'KoreaDem Peoples Rep':25155317,'Portugal':10348648,'Paraguay':6639123,'Pacific island small states':2351091,'Post-demographic dividend':1097737383,'French Polynesia':282764,'Qatar':2235355,'Romania':19832389,'Russian Federation':144096812,'Rwanda':11609666,'South Asia':1744161298,'Saudi Arabia':31540372,'Sudan':40234882,'Senegal':15129273,'Singapore':5535002,'Solomon Islands':583591,'Sierra Leone':6453184,'El Salvador':6126583,'San Marino':31781,'Somalia':10787104,'Serbia':7098247,'Sub-Saharan Africa excluding high income':100088808137425,'South Sudan':12339812,'Sub-Saharan Africa':100098098137425,'Small states':38050062,'Sao Tome and Principe':190344,'Suriname':542975,'Slovak Republic':5424050,'Slovenia':2063768,'Sweden':9798871,'Swaziland':1286970,'Sint Maarten Dutch part':38817,'Seychelles':92900,'Syrian Arab Republic':18502413,'Turks and Caicos Islands':34339,'Chad':14037472,'East Asia  Pacific IDA  IBRD countries':2009929013,'Europe  Central Asia IDA  IBRD countries':453562136,'Togo':7304578,'Thailand':67959359,'Tajikistan':8481855,'Turkmenistan':5373502,'Latin America  the Caribbean IDA  IBRD countries':616862604,'Middle East  North Africa IDA  IBRD countries':358138798,'Timor-Leste':1245015,'Tonga':106170,'South Asia IDA  IBRD':1744161298,'Sub-Saharan Africa IDA  IBRD countries':100098098137425,'Trinidad and Tobago':1360088,'Tunisia':11107800,'Turkey':78665830,'Tuvalu':9916,'Tanzania':53470420,'Uganda':39032383,'Ukraine':45198200,'Upper middle income':255032605537425,'Uruguay':3431555,'United States':321418820,'Uzbekistan':31299500,'St Vincent and the Grenadines':109462,'VenezuelaRB':31108083,'British Virgin Islands':30117,'Virgin Islands US':103574,'Vietnam':91703800,'Vanuatu':264652,'West Bank and Gaza':4422143,'World':734663303737425,'Samoa':193228,'YemenRep':26832215,'South Africa':549569203742456,'CongoDem Rep':77266814,'Zambia':16211767,'Zimbabwe':15602751}


# ## Create country list ##

# In[ ]:


# Merge Country dictionary with the top100users
top100usersCC = pd.merge(top100users, pd.DataFrame(list(dict_country.items()), columns=['Id','Country']), how='left', on='Id')
#top100usersCCPP = pd.merge(top100usersCC, pd.DataFrame(list(population.items()), columns=['Id','Country']), how='left', on='Country')
# Count countries
top100usersCC.loc[:, 'CountryCount'] = top100usersCC.groupby('Country')['Country'].transform('count')
# Create list with coounts ordered by desc
topCountries = top100usersCC[['Country', 'CountryCount']].drop_duplicates('Country', keep='first')                                                        .sort_values('CountryCount', ascending=False)       .reset_index(drop=True)                                                   

topCountriesPP = pd.merge(topCountries,  pd.DataFrame(list(population.items()),  columns=['Country', 'population']), how='left', on='Country')
topCountriesCP = topCountriesPP.where(topCountriesPP['population']>0, topCountriesPP[['CountryCount']]/topCountriesPP[['CountryCount']])
topCountriesCP


# ## Plot pie chart to show percentage ##

# In[ ]:


import matplotlib

matplotlib.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

pieCountry = topCountries.set_index('Country')
pd.Series(pieCountry['CountryCount']).plot.pie(figsize=(13, 13), autopct='%0.1f')


# ## Plot countries on the Wolrd map ##

# In[ ]:


import plotly.offline as py
py.offline.init_notebook_mode()

data = [ dict(
        type = 'choropleth',
        locations = topCountries['Country'],
        z = topCountries['CountryCount'],
        locationmode = 'country names',
        text = topCountries['Country'],
        colorscale = [[0,"rgb(153, 241, 243)"],[0.2,"rgb(16, 64, 143)"],[1,"rgb(0, 0, 0)"]],
        autocolorscale = False,
        marker = dict(
            line = dict(color = 'rgb(58,100,69)', width = 0.6)),
            colorbar = dict(autotick = True, tickprefix = '', title = '# of Kagglers')
            )
       ]

layout = dict(
    title = 'Top100 Kagglers distributed by Country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
        type = 'equirectangular'
        ),
    margin = dict(b = 0, t = 0, l = 0, r = 0)
            )
    )

fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='d3-world-map')


# Please fill free to fork/copy/enhance this script.
