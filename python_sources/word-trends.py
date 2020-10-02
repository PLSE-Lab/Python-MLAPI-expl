#!/usr/bin/env python
# coding: utf-8

# # UN General Debates
# 
# First of all, import packages and data

# In[ ]:


import pandas as pd
import re
import nltk
import nltk.stem
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from string import punctuation

#to plot inside the document
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


debates = pd.read_csv("../input/un-general-debates.csv")


# ## Description

# Have a first look at the data.

# In[ ]:


debates.head()


# In[ ]:


debates.describe(include="all")


# We have 7507 speaches, ranging from 1970 to 2015.
# 
# They were spoken by representatives of 199 countries.

# In[ ]:


debates[["year", "country"]].groupby("year").count().plot(kind="bar")


# It seems like more an more countries are speakers at the General Debates each year, from 1970 (70 countries) to at least 2006 (193 countries).
# 
# This is explained, at first by the end of series of decolonization in the 70s and 80s (more countries break of from their empires to become independent countries), then by the break-up of the Soviet Union after 1989.

# ## Text Preparation

# Put all text to lower case to avoid problems. Certain characters are not recognized because of the encoding too.

# In[ ]:


debates['text'] = debates['text'].str.lower().map(lambda x: re.sub('\W+',' ', x))


# Now let's transform the text to lists

# In[ ]:


debates['token'] = debates['text'].apply(word_tokenize)


# We need to remove the punctuation and english stopwords to only keep the essence of the text.

# In[ ]:


stop_words = set(stopwords.words('english'))
# I noticed that "'s" is not included in stopwords, while I think it doesn't bring much meaning in a text, so I'll add it to the set to remove from the cleaned tokens.
stop_words.add("'s")
stop_words.add("'")
stop_words.add("-")
stop_words.add("'")
debates['clean'] = debates['token'].apply(lambda x: [w for w in x if not w in stop_words and not w in punctuation])


# The last piece of preparation I want to do is now stemming (so we do not have different variations of same words separated).

# In[ ]:


stemmer = nltk.stem.PorterStemmer()


# In[ ]:


debates['stems'] = [[format(stemmer.stem(token)) for token in speech] for speech in debates['clean']]


# In[ ]:


debates.head()


# It would be interesting to combine the speeches of all speakers of a session and plot the 25 most used words.
# 
# This will give us an idea of what the world was interested/worried about on each year.

# In[ ]:


all_per_year = debates.groupby('session').agg({'year': 'mean', 'clean': 'sum'})


# In[ ]:


for i, row in all_per_year.iterrows():
    sess = dict(nltk.FreqDist(row['clean']))
    sort_sess = sorted(sess.items(), key=lambda x: x[1], reverse=True)[0:25]
    plt.bar(range(len(sort_sess)), [val[1] for val in sort_sess], align='center')
    plt.xticks(range(len(sort_sess)), [val[0] for val in sort_sess])
    plt.xticks(rotation=90)
    plt.title("25 most used words in %d's session" % row['year'])
    plt.show()


# Interestingly, the constant most used words are "United" and "Nations". "General" and "Assembly" are both often part of the most cited. _**Are the UN General Assembly sessions a bit self-centered?...**_
# 
# 2013 and 2014 are the only two years when words other than UN, countries or international made it in the top 3. This word is **"Development"**. Development has been gradually growing towards the first positions since the 1990s
# 
# The words "Peace" and "People" are always present in the top list.

# Now, I'm just going to count and combine global and yearly occurrences of all words in all speeches.

# In[ ]:


freqs = {}
for i, speech in debates.iterrows():
    year = speech['year']
    for token in speech['stems']:
        if token not in freqs:
            freqs[token] = {"total_freq":1, year:1}
        else:
            freqs[token]["total_freq"] += 1
            if not freqs[token].get(year):
                freqs[token][year] = 1
            else:
                freqs[token][year] += 1


# Now transform this dictionary of dictionaries into a dataframe

# In[ ]:


freqs_df = pd.DataFrame.from_dict(freqs, orient='index')
freqs_df['word'] = freqs_df.index


# In[ ]:


# Example of data for the stem of the word "peace"
freqs_df[freqs_df.index == "peac"]


# Make the dataframe a little more presentable (order columns chronologically, sort rows by total_freq).

# In[ ]:


new_cols = ["total_freq", "word"] + sorted(freqs_df.columns.tolist()[1:-1])
freqs_df = freqs_df[new_cols]

freqs_df = freqs_df.sort_values('total_freq', ascending=False)

freqs_df.head()


# In[ ]:


freqs_df.shape


# In[ ]:


freqs_df.tail(30)


# We're working with 35,026 stems (an improvement from 84,256 unique words befor using the stemming method). Though looking at the less used words makes me realize that much more cleaning would be appropriate to get rid of some concatenated words or even strange characters...

# ## Word trends

# Let's plot the most used, to see their evolution over time.

# In[ ]:


freqs_df.iloc[0:5, 1:47].transpose().iloc[1:].plot(title="Most common words")


# In[ ]:


freqs_df[freqs_df['word'].isin(['peac', 'war', 'securit', 'cold', 'conflict', 'aggression'])].iloc[:, 1:47].transpose().iloc[1:].plot(title = "War and Peace")


# Peace and Security are main concerns of the United Nations throughout the years.
# 
# It's very interesting to see the use of the word peace decline so drastically around the fall of the Berlin Wall (concurrently to a peak in the use of "aggression", for some reason I cannot explain), then kick back to the same level right afterward... Since the mid 90s, I am deeply sorry to see it decrease so much...
# 
# I do not see any particular trend over the words war or conflict. I do, though, see a surge on the use of "cold" at the end of the 80s, gradually descending after this. I would think that this has to do with the Cold War, which was ending at the time.

# In[ ]:


freqs_df[freqs_df['word'].isin(['economi', 'wealth', 'crisi', 'growth', 'inflat', 'trade', 'poverti', 'rich', 'recession', 'income'])].iloc[:, 1:47].transpose().iloc[1:].plot(title="Economy")


# Here, we see the spikes of the 73 and 75 Oil Crises, the 1980s and the 1990 recession, what I believe to be the 1998 Ruble Crisis, and the 2008 crisis.
# 
# Poverty has changed in level of prevalence between 1985 and 2000, increasing drastically.

# In[ ]:


freqs_df[freqs_df['word'].isin(['environment', 'sustain', 'green', 'energi', 'ecolog', 'warm', 'temperatur', 'pollution', 'planet'])].iloc[:, 1:47].transpose().iloc[1:].plot(title="Environment")


# Interesting trends here:
# 
# - Energy was mentionned a lot in the 70s and 80s because of the Oil Crises, and reappeared in the 2000s in the context of the environmentalism for different reasons.
# 
# - There was an interest for the environment (and pollution) in the 70s, which fell into oblivion (probably because the world was struggling with energy issues at the time) until the end 80s. This huge spike then slowly degraded.
# 
# - Sustainability is a notion that appeared in the 90s, and has since been growing especially since the 2010s.
# 
# In general, the later years show an increase in worries over environmental issues.

# In[ ]:


freqs_df[freqs_df['word'].isin(['peopl', 'inequaliti', 'refuge', 'humanitarian', 'immigr', 'freedom', 'right'])].iloc[:, 1:47].transpose().iloc[1:].plot(title="People")


# Peoples are always a subject of interest in UN debates. The concern over Freedom, however, has declined after the fall of the Eastern Block (though it reappeared  in the early 2010s, I can't seem to recall the context).
# 
# Humanitarian aid seems to have exploded in the 90s with famines in Somalia, North Korea, Sudan, Ethiopia.

# In[ ]:


freqs_df[freqs_df['word'].isin(['democraci', 'republ', 'dictat', 'sovereign', 'politic', 'vote'])].iloc[:, 1:47].transpose().iloc[1:].plot(title="Politics")


# The end of decolonization was a period when Sovereignty was an important concept. It is still a concern since, during the UN General Assembly, in a more relative measure.
# 
# The end of the Cold War brought a spike in the use of the word Democracy, which is now a constantly used word in debates. It's interesting to see how much less that word was brandished before that period...
# 
# Republic comes very often with great variations. I believe it must be because so many countries are called "Republic of ...", and speakers use the complete names of countries when refering to one another.

# In[ ]:


freqs_df[freqs_df['word'].isin(['violenc', 'unrest', 'genocid', 'atroc', 'kill', 'death'])].iloc[:, 1:47].transpose().iloc[1:].plot(title="Violence")


# I had to remove the word Terrorism out of the plot (see below), because the difference of scale made the other words unnoticeable...
# 
# Both the 70s and early 2000s had their violent terrrorism episodes. The word Violence, unhappily, rarely decreases in use...
# 
# Here we can notice the genocides of Cambodia (1975-79), Guatemala (1981), Bosnia-Herzegovine (1992), Rwanda (1994), Sudan (2003).

# In[ ]:


freqs_df[freqs_df['word'].isin(['terror'])].iloc[:, 1:47].transpose().iloc[1:].plot(title="Terrorism")


# In[ ]:


freqs_df[freqs_df['word'].isin(['health', 'disease', 'famin', 'drought', 'hiv', 'aid', 'research'])].iloc[:, 1:47].transpose().iloc[1:].plot(title="Health")


# Health is definitively a growing concern. Unhappily, research is not.
# 
# HIV started being mentioned in the UN roughly 10 years after AIDS (though I believe this has to be taken with a grain of salt as word is probably mistaken for some sor of aids, helps).
# 
# Cases of droughts always entails famine episodes.

# ## Countries mentioned

# I want to find out which countries speakers mention, and will show trends in time, as well as  look at whether certain countries speak more about specific other countries. List found [here](http://www.nationsonline.org/oneworld/country_code_list.htm), and tweaked here and there for better results.

# In[ ]:


#IPython.load_extensions('usability/hide_input/main');
countries = dict((k, v.lower()) for k,v in {
    'AFG': 'Afghanistan', 
    'ALA': 'Aland Islands', 
    'ALB': 'Albania', 
    'DZA': 'Algeria', 
    'ASM': 'American Samoa', 
    'AND': 'Andorra', 
    'AGO': 'Angola', 
    'AIA': 'Anguilla', 
    'ATA': 'Antarctica', 
    'ATG': 'Antigua and Barbuda', 
    'ARG': 'Argentina', 
    'ARM': 'Armenia', 
    'ABW': 'Aruba', 
    'AUS': 'Australia', 
    'AUT': 'Austria', 
    'AZE': 'Azerbaijan', 
    'BHS': 'Bahamas', 
    'BHR': 'Bahrain', 
    'BGD': 'Bangladesh', 
    'BRB': 'Barbados', 
    'BLR': 'Belarus', 
    'BEL': 'Belgium', 
    'BLZ': 'Belize', 
    'BEN': 'Benin', 
    'BMU': 'Bermuda', 
    'BTN': 'Bhutan', 
    'BOL': 'Bolivia', 
    'BIH': 'Bosnia and Herzegovina', 
    'BWA': 'Botswana', 
    'BVT': 'Bouvet Island', 
    'BRA': 'Brazil', 
    'VGB': 'Virgin Islands', 
    'IOT': 'British Indian Ocean Territory', 
    'BRN': 'Brunei', 
    'BGR': 'Bulgaria', 
    'BFA': 'Burkina Faso', 
    'BDI': 'Burundi', 
    'KHM': 'Cambodia', 
    'CMR': 'Cameroon', 
    'CAN': 'Canada', 
    'CPV': 'Cape Verde', 
    'CYM': 'Cayman Islands', 
    'CAF': 'Central Africa', 
    'TCD': 'Chad', 
    'CHL': 'Chile', 
    'CHN': 'China', 
    'HKG': 'Hong Kong', 
    'MAC': 'Macao', 
    'CXR': 'Christmas Island', 
    'CCK': 'Cocos Islands', 
    'COL': 'Colombia', 
    'COM': 'Comoros', 
    'COG': 'Congo', 
    'COD': 'Democratic Republic of Congo', 
    'COK': 'Cook Islands', 
    'CRI': 'Costa Rica', 
    'CIV': "Cote d'Ivoire", 
    'HRV': 'Croatia', 
    'CUB': 'Cuba', 
    'CYP': 'Cyprus', 
    'CZE': 'Czech Republic', 
    'DNK': 'Denmark', 
    'DJI': 'Djibouti', 
    'DMA': 'Dominica', 
    'DOM': 'Dominican Republic', 
    'ECU': 'Ecuador', 
    'EGY': 'Egypt', 
    'SLV': 'El Salvador', 
    'GNQ': 'Equatorial Guinea', 
    'ERI': 'Eritrea', 
    'EST': 'Estonia', 
    'ETH': 'Ethiopia', 
    'FLK': 'Falkland', 
    'FRO': 'Faroe', 
    'FJI': 'Fiji', 
    'FIN': 'Finland', 
    'FRA': 'France', 
    'GUF': 'French Guiana', 
    'PYF': 'French Polynesia', 
    'ATF': 'French Southern Territories', 
    'GAB': 'Gabon', 
    'GMB': 'Gambia', 
    'GEO': 'Georgia', 
    'DEU': 'Germany', 
    'GHA': 'Ghana', 
    'GIB': 'Gibraltar', 
    'GRC': 'Greece', 
    'GRL': 'Greenland', 
    'GRD': 'Grenada', 
    'GLP': 'Guadeloupe', 
    'GUM': 'Guam', 
    'GTM': 'Guatemala', 
    'GGY': 'Guernsey', 
    'GIN': 'Guinea', 
    'GNB': 'Guinea-Bissau', 
    'GUY': 'Guyana', 
    'HTI': 'Haiti', 
    'HMD': 'Heard and Mcdonald Islands', 
    'VAT': 'Vatican', 
    'HND': 'Honduras', 
    'HUN': 'Hungary', 
    'ISL': 'Iceland', 
    'IND': 'India', 
    'IDN': 'Indonesia', 
    'IRN': 'Iran', 
    'IRQ': 'Iraq', 
    'IRL': 'Ireland', 
    'IMN': 'Isle of Man', 
    'ISR': 'Israel', 
    'ITA': 'Italy', 
    'JAM': 'Jamaica', 
    'JPN': 'Japan', 
    'JEY': 'Jersey', 
    'JOR': 'Jordan', 
    'KAZ': 'Kazakhstan', 
    'KEN': 'Kenya', 
    'KIR': 'Kiribati', 
    'PRK': 'North Korea', 
    'KOR': 'South Korea', 
    'KWT': 'Kuwait', 
    'KGZ': 'Kyrgyzstan', 
    'LAO': 'Lao', 
    'LVA': 'Latvia', 
    'LBN': 'Lebanon', 
    'LSO': 'Lesotho', 
    'LBR': 'Liberia', 
    'LBY': 'Libya', 
    'LIE': 'Liechtenstein', 
    'LTU': 'Lithuania', 
    'LUX': 'Luxembourg', 
    'MKD': 'Macedonia', 
    'MDG': 'Madagascar', 
    'MWI': 'Malawi', 
    'MYS': 'Malaysia', 
    'MDV': 'Maldives', 
    'MLI': 'Mali', 
    'MLT': 'Malta', 
    'MHL': 'Marshall Islands', 
    'MTQ': 'Martinique', 
    'MRT': 'Mauritania', 
    'MUS': 'Mauritius', 
    'MYT': 'Mayotte', 
    'MEX': 'Mexico', 
    'FSM': 'Micronesia', 
    'MDA': 'Moldova', 
    'MCO': 'Monaco', 
    'MNG': 'Mongolia', 
    'MNE': 'Montenegro', 
    'MSR': 'Montserrat', 
    'MAR': 'Morocco', 
    'MOZ': 'Mozambique', 
    'MMR': 'Myanmar', 
    'NAM': 'Namibia', 
    'NRU': 'Nauru', 
    'NPL': 'Nepal', 
    'NLD': 'Netherlands', 
    'ANT': 'Netherlands Antilles', 
    'NCL': 'New Caledonia', 
    'NZL': 'New Zealand', 
    'NIC': 'Nicaragua', 
    'NER': 'Niger', 
    'NGA': 'Nigeria', 
    'NIU': 'Niue', 
    'NFK': 'Norfolk Island', 
    'MNP': 'Northern Mariana Islands', 
    'NOR': 'Norway', 
    'OMN': 'Oman', 
    'PAK': 'Pakistan', 
    'PLW': 'Palau', 
    'PSE': 'Palestine', 
    'PAN': 'Panama', 
    'PNG': 'Papua New Guinea', 
    'PRY': 'Paraguay', 
    'PER': 'Peru', 
    'PHL': 'Philippines', 
    'PCN': 'Pitcairn', 
    'POL': 'Poland', 
    'PRT': 'Portugal', 
    'PRI': 'Puerto Rico', 
    'QAT': 'Qatar', 
    'REU': 'Reunion', 
    'ROU': 'Romania', 
    'RUS': 'Russia', 
    'RWA': 'Rwanda', 
    'BLM': 'Saint-Barthelemy', 
    'SHN': 'Saint Helena', 
    'KNA': 'Saint Kitts', 
    'LCA': 'Saint Lucia', 
    'MAF': 'Saint-Martin', 
    'SPM': 'Saint Pierre and Miquelon', 
    'VCT': 'Saint Vincent and Grenadines', 
    'WSM': 'Samoa', 
    'SMR': 'San Marino', 
    'STP': 'Sao Tome and Principe', 
    'SAU': 'Saudi Arabia', 
    'SEN': 'Senegal', 
    'SRB': 'Serbia', 
    'SYC': 'Seychelles', 
    'SLE': 'Sierra Leone', 
    'SGP': 'Singapore', 
    'SVK': 'Slovakia', 
    'SVN': 'Slovenia', 
    'SLB': 'Solomon Islands', 
    'SOM': 'Somalia', 
    'ZAF': 'South Africa', 
    'SGS': 'South Georgia and the South Sandwich Islands', 
    'SSD': 'South Sudan', 
    'ESP': 'Spain', 
    'LKA': 'Sri Lanka', 
    'SDN': 'Sudan', 
    'SUR': 'Suriname', 
    'SJM': 'Svalbard', 
    'SWZ': 'Swaziland', 
    'SWE': 'Sweden', 
    'CHE': 'Switzerland', 
    'SYR': 'Syria', 
    'TWN': 'Taiwan', 
    'TJK': 'Tajikistan', 
    'TZA': 'Tanzania', 
    'THA': 'Thailand', 
    'TLS': 'Timor', 
    'TGO': 'Togo', 
    'TKL': 'Tokelau', 
    'TON': 'Tonga', 
    'TTO': 'Trinidad', 
    'TUN': 'Tunisia', 
    'TUR': 'Turkey', 
    'TKM': 'Turkmenistan', 
    'TCA': 'Turks and Caicos Islands', 
    'TUV': 'Tuvalu', 
    'UGA': 'Uganda', 
    'UKR': 'Ukraine', 
    'ARE': 'United Arab Emirates', 
    'GBR': 'United Kingdom', 
    'USA': 'United States', 
    'UMI': 'US Minor Outlying Islands', 
    'URY': 'Uruguay', 
    'UZB': 'Uzbekistan', 
    'VUT': 'Vanuatu', 
    'VEN': 'Venezuela', 
    'VNM': 'Viet Nam', 
    'VIR': 'Virgin Islands', 
    'WLF': 'Wallis and Futuna', 
    'ESH': 'Western Sahara', 
    'YEM': 'Yemen', 
    'ZMB': 'Zambia', 
    'ZWE': 'Zimbabwe'
}.items())


# In[ ]:


debates['countries_mentioned'] = debates['token'].apply(lambda token: {x:token.count(x) for x in token if x in countries.values()})


# I'll now save this in a table for study and display

# In[ ]:


country_mentions = pd.concat([debates[["year", "country"]],
                              debates['countries_mentioned'].apply(pd.Series)], axis=1).dropna(axis=1, how='all')
country_mentions['country'] = country_mentions['country'].apply(lambda x: countries.get(x))
country_mentions.head()


# This table is a little too long and large to make sense out of it. Let's group it by country to get the speaking country info.

# In[ ]:


country_mentions_by_country = country_mentions.groupby("country")[country_mentions.columns[2:]].sum()


# Trying to plot a sankey diagram of countries mentioning each other.

# In[ ]:


# First need to melt country_mentions_by_country to long form
sankey_data = country_mentions_by_country.unstack().reset_index()
sankey_data.columns = ['source','target','value']
sankey_data = sankey_data.sort_values(by='value', ascending=False)
sankey_data.head()


# In[ ]:


import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
init_notebook_mode()

data = dict(
    type='sankey',
    domain = dict(
      x =  [0,1],
      y =  [0,1]
    ),
    orientation = "h",
    valueformat = ".0f",
    valuesuffix = "TWh"   
  )

layout =  dict(
    title = "Which countries mention which in the UN's General Assembly\n(1970-2015)",
    font = dict(
      size = 10
    )
)

data_trace = dict(
    type='sankey',
    width = 1118,
    height = 772,
    domain = dict(
      x =  [0,1],
      y =  [0,1]
    ),
    orientation = "h",
    valueformat = ".0f",
    valuesuffix = "TWh",
    node = dict(
      pad = 15,
      thickness = 15,
      line = dict(
        color = "black",
        width = 0.5
      ),
      label =  sankey_data['target'],
      color =  "black"
  ),
    link = dict(
      source =  sankey_data['source'],
      target =  sankey_data['target'],
      value =  sankey_data['value'],
      label =  sankey_data['source']
  ))

fig = dict(data=[data_trace], layout=layout)
iplot(fig, validate=False)


# In[ ]:


from ipysankeywidget import SankeyWidget

sankey_data.columns = ['source','target','value']
sankey_data = sankey_data.sort_values(by='value', ascending=False)
links=sankey_data[0:200].dropna()[['source','target','value']].to_dict(orient='records')

SankeyWidget(value={'links': links},
             width=800, height=800,margins=dict(top=0, bottom=0))

