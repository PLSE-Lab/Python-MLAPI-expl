#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Analysis

# In[ ]:


from IPython.display import HTML
from IPython.display import display
baseCodeHide="""
<style>
.button {
    background-color: #008CBA;;
    border: none;
    color: white;
    padding: 8px 22px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
}
</style>
 <script>
   // Assume 3 input cells. Manage from here.
   var divTag0 = document.getElementsByClassName("input")[0]
   var displaySetting0 = divTag0.style.display;
   divTag0.style.display = 'none';
   
   var divTag1 = document.getElementsByClassName("input")[4]
   var displaySetting1 = divTag1.style.display;
   divTag1.style.display = 'block';
   divTag1.style.display = 'none';
   
   var divTag2 = document.getElementsByClassName("input")[3]
   var displaySetting2 = divTag2.style.display;
   divTag2.style.display = 'block';
   divTag2.style.display = 'none';
 
    function toggleInput(i) { 
      var divTag = document.getElementsByClassName("input")[i]
      var displaySetting = divTag.style.display;
     
      if (displaySetting == 'block') { 
         divTag.style.display = 'none';
       }
      else { 
         divTag.style.display = 'block';
       } 
  }  
  </script>
  <!-- <button onclick="javascript:toggleInput(0)" class="button">Show Code</button> -->
"""
h=HTML(baseCodeHide)


display(h)


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Birth Death Rate Trend

# In order to plot the Birht/Date Rate on the Map I need the Alpha-3 code. I therefore have to turn country names into alpha-3 code exploiting a list I have partially manually filled. This may have introduced errors and so the plotted data might not be exact. If you find any error please leave a comment.

# In[ ]:


bdrates = pd.read_csv("../input/birth_death_growth_rates.csv")
bdrates.head()


# In[ ]:



display(HTML("<button onclick=\"javascript:toggleInput(4)\" class=\"button\">Show Country Dictionary</button>"))


# In[ ]:



countries={'Afghanistan':'AFG',
'Aland Islands':'ALA','Albania':'ALB','Algeria':'DZA','American Samoa':'ASM','Andorra':'AND','Angola':'AGO','Anguilla':'AIA','Antigua and Barbuda':'ATG',
'Antarctica':'ATA','Argentina':'ARG','Armenia':'ARM','Aruba':'ABW','Australia':'AUS','Austria':'AUT','Azerbaijan':'AZE','Azerbaidjan':'AZE',
'Bahrain':'BHR','Bahamas The':'BHS','Bangladesh':'BGD','Barbados':'BRB','Belarus':'BLR',
'Belgium':'BEL','Belize':'BLZ','Benin':'BEN','Bermuda':'BMU','Bhutan':'BTN','Bolivia':'BOL','Bosnia and Herzegovina':'BIH','Bosnia-Herzegovina':'BIH',
'Botswana':'BWA','Bouvet Island':'BVT','Brazil':'BRA','British Virgin Islands':'VGB','British Indian Ocean Territory':'IOT',
'Brunei':'BRN','Brunei Darussalam':'BRN','Bulgaria':'BGR','Burkina Faso':'BFA','Burma':'MMR',
'Burundi':'BDI','Cabo Verde':'CPV','Cape Verde':'CPV','Cambodia':'KHM','Cameroon':'CMR',
'Canada':'CAN','Cayman Islands':'CYM','Central African Republic':'CAF','Chad':'TCD','Chile':'CHL',
'Christmas Island':'CHR','China':'CHN','Colombia':'COL','Comoros':'COM','Congo (Kinshasa)':'COG','Cook Islands':'COK','Costa Rica':'CRI','Cote d\'Ivoire':'CIV',
"Ivory Coast (Cote D'Ivoire)":'CIV','Croatia':'HRV','Cuba':'CUB','Curacao':'CUW','Cyprus':'CYP',
'Czech Republic':'CZE','Denmark':'DNK','Djibouti':'DJI','Dominica':'DMA','Dominican Republic':'DOM',
'Ecuador':'ECU','Egypt':'EGY','El Salvador':'SLV','Equatorial Guinea':'GNQ','Eritrea':'ERI','Estonia':'EST',
'Ethiopia':'ETH','Falkland Islands (Islas Malvinas)':'FLK','Falkland Islands':'FLK','Faroe Islands':'FRO',
'Fiji':'FJI','Finland':'FIN','France':'FRA','French Polynesia':'PYF','Gabon':'GAB',
'Gambia The':'GMB','Georgia':'GEO','Germany':'DEU','Ghana':'GHA','Gibraltar':'GIB',
'Greece':'GRC','Greenland':'GRL','Grenada':'GRD','Guam':'GUM','Guatemala':'GTM',
'Guernsey':'GGY','Guinea-Bissau':'GNB','Guinea':'GIN','Guyana':'GUY','French Guyana':'GUY','Haiti':'HTI',
'Honduras':'HND','Heard and McDonald Islands':'HMD','Hong Kong':'HKG','Hungary':'HUN','Iceland':'ISL',
'India':'IND','Indonesia':'IDN','Iran':'IRN','Iraq':'IRQ','Ireland':'IRL','Isle of Man':'IMN',
'Israel':'ISR','Italy':'ITA','Jamaica':'JAM','Japan':'JPN','Jersey':'JEY','Jordan':'JOR',
'Kazakhstan':'KAZ','Kenya':'KEN','Kiribati':'KIR','Korea North':'KOR','Korea South':'PRK',
'South Korea':'PRK','North Korea':'KOR','Kosovo':'KSV','Kuwait':'KWT','Kyrgyzstan':'KGZ',
'Laos':'LAO','Latvia':'LVA','Lebanon':'LBN','Lesotho':'LSO','Liberia':'LBR','Libya':'LBY','Liechtenstein':'LIE',
'Lithuania':'LTU','Luxembourg':'LUX','Macau':'MAC','Macedonia':'MKD','Madagascar':'MDG',
'Malawi':'MWI','Malaysia':'MYS','Maldives':'MDV','Mali':'MLI','Malta':'MLT','Marshall Islands':'MHL',
'Martinique (French)':'MTQ','Mauritania':'MRT','Mauritius':'MUS','Mexico':'MEX','Micronesia, Federated States of':'FSM',
'Moldova':'MDA','Moldavia':'MDA','Monaco':'MCO','Mongolia':'MNG','Montenegro':'MNE','Montserrat':'MSR',
'Morocco':'MAR','Mozambique':'MOZ','Myanmar':'MMR','Namibia':'NAM','Nepal':'NPL','Netherlands':'NLD',
'Netherlands Antilles':'ANT','New Caledonia':'NCL','New Caledonia (French)':'NCL','New Zealand':'NZL','Nicaragua':'NIC',
'Nigeria':'NGA','Niger':'NER','Niue':'NIU','Northern Mariana Islands':'MNP','Norway':'NOR','Oman':'OMN',
'Pakistan':'PAK','Palau':'PLW','Panama':'PAN','Papua New Guinea':'PNG','Paraguay':'PRY','Peru':'PER',
'Philippines':'PHL','Pitcairn Island':'PCN','Poland':'POL','Polynesia (French)':'PYF','Portugal':'PRT',
'Puerto Rico':'PRI','Qatar':'QAT','Reunion (French)':'REU','Romania':'ROU','Russia':'RUS','Russian Federation':'RUS',
'Rwanda':'RWA','Saint Kitts and Nevis':'KNA','Saint Lucia':'LCA','Saint Martin':'MAF','Saint Pierre and Miquelon':'SPM',
'Saint Vincent and the Grenadines':'VCT','Saint Vincent & Grenadines':'VCT','S. Georgia & S. Sandwich Isls.':'SGS','Samoa':'WSM',
'San Marino':'SMR','Saint Helena':'SHN','Sao Tome and Principe':'STP','Saudi Arabia':'SAU','Senegal':'SEN',
'Serbia':'SRB','Seychelles':'SYC','Sierra Leone':'SLE','Singapore':'SGP','Sint Maarten':'SXM',
'Slovakia':'SVK','Slovak Republic':'SVK','Slovenia':'SVN','Solomon Islands':'SLB','Somalia':'SOM','South Africa':'ZAF',
'South Sudan':'SSD','Spain':'ESP','Sri Lanka':'LKA','Sudan':'SDN','Suriname':'SUR',
'Swaziland':'SWZ','Sweden':'SWE','Switzerland':'CHE','Syria':'SYR','Taiwan':'TWN','Tajikistan':'TJK',
'Tadjikistan':'TJK','Tanzania':'TZA','Thailand':'THA','Timor-Leste':'TLS','Togo':'TGO','Tonga':'TON',
'Trinidad and Tobago':'TTO','Tunisia':'TUN','Turkey':'TUR','Turkmenistan':'TKM','Tuvalu':'TUV',
'Uganda':'UGA','Ukraine':'UKR','United Arab Emirates':'ARE','United Kingdom':'GBR','United States':'USA',
'U.S. Minor Outlying Islands':'UMI','Uruguay':'URY','Uzbekistan':'UZB','Vanuatu':'VUT',
'Vatican City State':'VAT','Venezuela':'VEN','Vietnam':'VNM','Virgin Islands':'VGB',
'Virgin Islands U.S.':'VIR','Virgin Islands British':'VGB','West Bank':'WBG','Yemen':'YEM',
'Zaire':'ZAR','Zambia':'ZMB','Zimbabwe':'ZWE','Saint Barthelemy':'BLM','Nauru':'NRU',
'Western Sahara':'ESH','Wallis and Futuna':'WLF','Gaza Strip':'GZA','Micronesia Federated States of':'FSM',
          'Czechia':'CZE','Congo (Brazzaville)':'COG','Turks and Caicos Islands':'TCA'}


# In[ ]:


codes = [countries[country] if country != 'I prefer not to say' else None for country in bdrates['country_name']]


# In[ ]:


bdrates['code']=codes


# In[ ]:


sbd = bdrates[['code','crude_birth_rate','year']]
sbdgroup36 = sbd[sbd['year']==2036].drop('year',axis=1).groupby('code').mean()
sbdgroup16 = sbd[sbd['year']==2016].drop('year',axis=1).groupby('code').mean()
sbdgroup00 = sbd[sbd['year']==2000].drop('year',axis=1).groupby('code').mean()
sbdgroup90 = sbd[sbd['year']==1990].drop('year',axis=1).groupby('code').mean()


# In[ ]:


gdata = [ dict(
        type = 'choropleth',
        locations = sbdgroup90.index,
        z = sbdgroup90['crude_birth_rate'],
        text = sbdgroup90.index,
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            tickprefix = '#',
            title = '# Devs'),
      ) ]

layout = dict(
    title = 'Birth Death Rate in 1990',
    geo = dict(
            projection = dict(
                type = 'Mercator'
            ),
            showframe=False
            )
)




figure = dict( data=gdata, layout=layout )
iplot(figure)


# In[ ]:


gdata = [ dict(
        type = 'choropleth',
        locations = sbdgroup00.index,
        z = sbdgroup00['crude_birth_rate'],
        text = sbdgroup00.index,
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            tickprefix = '#',
            title = '# Devs'),
      ) ]

layout = dict(
    title = 'Birth Death Rate in 2000',
    geo = dict(
            projection = dict(
                type = 'Mercator'
            ),
            showframe=False
            )
)




figure = dict( data=gdata, layout=layout )
iplot(figure)


# In[ ]:


gdata = [ dict(
        type = 'choropleth',
        locations = sbdgroup16.index,
        z = sbdgroup16['crude_birth_rate'],
        text = sbdgroup16.index,
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            tickprefix = '#',
            title = '# Devs'),
      ) ]

layout = dict(
    title = 'Birth Death Rate in 2016',
    geo = dict(
            projection = dict(
                type = 'Mercator'
            ),
            showframe=False
            )
)




figure = dict( data=gdata, layout=layout )
iplot(figure)


# In[ ]:


gdata = [ dict(
        type = 'choropleth',
        locations = sbdgroup36.index,
        z = sbdgroup36['crude_birth_rate'],
        text = sbdgroup36.index,
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            tickprefix = '#',
            title = '# Devs'),
      ) ]

layout = dict(
    title = 'Birth Death Rate in 2036',
    geo = dict(
            projection = dict(
                type = 'Mercator'
            ),
            showframe=False
            )
)




figure = dict( data=gdata, layout=layout )
iplot(figure)

