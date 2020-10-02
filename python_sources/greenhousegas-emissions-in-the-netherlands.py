#!/usr/bin/env python
# coding: utf-8

# # Broeikasgasuitstoot in Nederland 
# 
# ## Introductie
# In mijn zoektocht naar meer langetermijn, en fact-based journalistiek betreffende het klimaatprobleem, heb ik de uitstoot van broeikasgassen in Nederland geanalyseerd. 
# 
# 
# ## Bronnen
# Dit notebook bevat een analyse van de data van het Centraal Bureau voor Statistiek (CBS) betreffende uitstoot van broeikasgassen, gerapporteerd volgens IPCC richtlijne, per jaar, sinds 1990. 
# 
# Deze data heb ik wat bewerkt om makkelijker inzichten eruit te kunnen halen. Zo heb ik de relatieve kleine bijdragers aan het broeikaseffect samengevoegd tot een de categorie 'overig', en heb ik de data gestructureerd zodat uitstoot niet dubbel gerapporteerd wordt in de dataset. 
# 
# ## Global warming potential
# De data bevat de uitstoot van de drie belangrijkste broeikasgassen - CO2, CH4 en NO2. Voor het bepalen van de totale CO2-equivalente uitstoot heb ik global warming potential (GPW) waarden gebruikt volgens de laatste richtlijnen van het IPCC (zie wikipedia, GPW). Deze waarden zijn een maatstaf voor de relatieve bijdrage van een gas aan het broeikas effect - zo is methaan een potenter broeikasgas dan CO2. 
# 
# ## Belangrijkste inzichten 
# De belangrijkste inzichten zijn onder andere de relatieve bijdrage van alle sectoren, en de trends hierin. Zo zien we bijvoorbeeld dat: 
# 
# - ...landbouw een vergelijkbare bijdrage levert als de opwekking van electrictiteit;
# - ...afvalverwerking en watervoorziening de belangrijkste bijdrage heeft geleverd aan de reductie over de afgelopen 30 jaar;
# - ...en dat de uitstoot van de meeste andere activiteiten gelijk is gebleven, of gestegen. 
# 
# ## Taal en packages 
# Ik heb Python gebruikt voor deze analyse, en de volgende packages: 
# 
# - Pandas - een package voor efficiente verwerking van grote datasets. 
# - Seaborn - een package voor het maken van mooiere plots
# - Numpy - een efficiente data structuur voor lineare algebra

# In[ ]:


# import modules
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', palette='muted')


# ## Data verwerking 
# De data is afkomstig van het CBS. Hieronder doe ik een aantal dingen om deze data leesbaarder te maken, en makkelijker te verwerken met pandas. 

# In[ ]:


# read data 
df = pd.read_csv("../input/IPCC_emissions.csv", sep=";")
keys = pd.read_csv("../input/IPCC_key_description.csv", sep=";")

# Weghalen van overbodige kolommen in key description 
keys_clean = keys.drop(['Description', 'L1.1','L2.1', 'L3.1', 'L4.1'], axis=1)

# Weghalen van spaties 
keys_clean['Key'] = keys_clean['Key'].str.strip()
df['Bronnen'] = df['Bronnen'].str.strip()

# Samenvoegen van de dataset en de metadata
df = df.rename(columns={'Bronnen': 'Key'})
df = df.merge(keys_clean, on='Key')

# Hierachy niveau toevoegen
df['Hierarchy key'] = df['Hierarchy key'].apply(str)
df['Level'] = df['Hierarchy key'].apply(len)
df.tail()

# Hernoemen uitstoot kolomnamen 
df = df.rename(columns={'CO2_1': 'CO2', 'CH4_2': 'CH4', 'N2O_3': 'N2O'})

# Toevoegen van CO2 equivalente uitstoot voor 20 jaar horizon en 100 jaar
GPW = pd.DataFrame(data=[[1, 86, 268], [1, 36, 298]], 
                   columns=['CO2', 'CH4', 'N2O'], 
                   index=['20yr', '100yr'])
for index, row in GPW.iterrows():
    df['Emissie_' + index] = df['CO2'] * row['CO2'] + df['CH4'] * row['CH4'] + df['N2O'] * row['N2O']
    df['CH4_' + index] = df['CH4'] * row['CH4'] 
    df['N2O_' + index] = df['N2O'] * row['N2O']

# Jaren weergeven als datetime format, voor makkelijker plotten
df['Jaar'] = df['Perioden'].str[:4]
df['Jaar'] = pd.to_datetime(df['Jaar'])
df.set_index('Jaar', inplace=True)

# Voeg level 4 toe voor particulier huishouden
df.loc[df['L3'] == 'Particulier huishouden', 'L4'] = 'Particulier huishouden'
df.loc[df['L3'] == 'Particulier huishouden', 'Level'] = 4

# Alles behalve L4 wegdoen, zodat uitstoot nergens dubbel staat
df = df.loc[df['Level'] == 4]

# Namen opschonen
old = df['L3'].unique()[3]
new = old[8:]
df = df.replace(old, new)
old = df['L4'].unique()[[0,1,2,3,4,6,8]]
new = [old[0][3:], old[1][4:], old[2][3:], old[3][6:], old[4][3:], old[5][2:], old[6][3:]]
df = df.replace(old, new)

# Alternatieve level 4 classificatie toevoegen, met 'overig' categorie 
emission_2017 = df.loc['2017'].sort_values(by='Emissie_20yr', ascending=False)[['L4', 'Emissie_20yr']]
df['L4_clean'] = df['L4'].replace(emission_2017['L4'][7:].values, 'Overig')

# Wegdoen van kolommen die niet meer relevant zijn 
df = df.drop(['Perioden', 'ID', 'Hierarchy key', 'Level'], axis=1)

# Opgeschoonde dataset opslaan 
df.to_csv('Emission_Netherlands_clean.csv', sep=';')


# ## Uitstoot van de verschillende broeikasgassen 
# In de data wordt uitstoot van methaan (CO2), methaan (CH4) en distikstofoxide (N2O) gerapporteerd. Om te kijken wat de relatieve bijdrage hiervaan is aan het broeikaseffect, moeten we deze waarden corrigeren voor hun global warming potential (GPW). Hieronder staat in tabelvorm de relatieve bijdragen aan het broeikas effect van de verschillende gassen, voor de twee GPW horizons (20 jaar en 100 jaar). 
# 
# We zien dat de bijdrage van methaan zeer significant is, met name voor de komende 20 jaar. 

# In[ ]:


# Totale itstoot per broeikasgas 2016
tot = df['2016'].sum()
d = np.array([[tot['CO2'], tot['CO2']], 
              [tot['CH4_20yr'], tot['CH4_100yr']], 
              [tot['N2O_20yr'], tot['N2O_100yr']]])
d = pd.DataFrame(data=d,
                 index=['CO2', 'CH4', 'N2O'],
                 columns=['20 jaar', '100 jaar'])

# Relatieve uitstoot per broeikasgas 2016
_ = d.div(d.sum()).T.plot(kind='bar', stacked=True)


# ## Uitstoot op classificatie niveau 3 en 4 
# De data van het CBS kent labels toe aan uitstoot, op verschillende niveaus. De meest interessante zijn niveau 3 en 4. Een voorbeeld van een niveau 3 categorie is 'vervoer'. Een voorbeeld van een niveau 4 categorie, die onder de niveau 3 categorie ' vervoer' valt, is 'wegverkeer'. 
# 
# Hieronder is de uitstoot van Nederland sinds 1990 geplot, uitgaande van een GPW horizon van 20 jaar, voor zowel niveau 3 als 4. We zien dat niveau 4 ons de meest relevantie inzichten geeft, al is het wel een goed idee om alle kleinere niveau 4 categorien samen te voegen tot overig (wat we boven al gedaan hebben). 
# 

# In[ ]:


# Pivot tabellen maken
L2 = pd.pivot_table(data=df, 
                    values='Emissie_20yr', 
                    index='Jaar', 
                    columns='L2', 
                    aggfunc='sum')
L3 = pd.pivot_table(data=df, 
                    values='Emissie_20yr', 
                    index='Jaar', 
                    columns='L3', 
                    aggfunc='sum')
L4 = pd.pivot_table(data=df, 
                    values='Emissie_20yr', 
                    index='Jaar', 
                    columns='L4', 
                    aggfunc='sum')

# Plot
fig, axs = plt.subplots(1,2, figsize=(20, 10))
_ = L3.plot.area(ax=axs[0])
_ = L4.plot.area(ax=axs[1])


# ## Uitstoot 2016 voor niveau 4 categorieen en verschillende GPW horizon 
# Nu we weten dat niveau 4 de beste inzichten verschaft, is het  interessant om de uitstoot te bekijken in 2016, voor de verschillende categorieen, en de verschillende GPW horizonnen.
# 
# We zien dat de keuze van de horizon veel impact heeft op de relatieve bijdragen van landbouw en watervoorziening, wat aangeeft dat hier voorval veel methaan (CH4) en distikstofoxide (N2O) bij vrijkomt. 
# 
# Ook is het interessant om te zien dat landbouw zo'n grote bijdrage levert, vergelijkbaar met elektriciteitsproductie. 

# In[ ]:


# Plot uitstoot in 2016; per categorie, voor 20 jaar en 100 jaar GPW horizon
fig_2016, axs = plt.subplots(2, 1, figsize=(5, 10))
kwargs = dict(autopct='%.2f', fontsize=10, kind='pie')
_ = df.loc['2016'].groupby(['L4_clean'])['Emissie_20yr'].sum().plot(ax=axs[0], **kwargs)
_ = df.loc['2016'].groupby(['L4_clean'])['Emissie_100yr'].sum().plot(ax=axs[1], **kwargs)
_ = axs[0].set_ylabel('')
_ = axs[1].set_ylabel('')
_ = axs[0].set_title('Emissie 2016\n' 
                     + 'GPW 20 jaar (CO2: 1, CH4: 34, N2O: 298)')
_ = axs[1].set_title('Emissie 2016\n'
                     + 'GPW 100 jaar (CO2: 1, CH4: 86, N2O: 268')


# ## Uitstoot over tijd 
# En volgende stap is om bovenstaande doorsnijdingen niet alleen voor 2016 te laten zien, maar over tijd, om ze de ontwikkelingen hiervan te kunnen zien, zowel cumulatief als voor elke sector afzonderlijk. 
# 
# Een van de meest opvallende inzichten hier is de sterke afname van uitstoot in afvalverwerking en watervoorziening. 

# In[ ]:


# Plot CO2 equivalente uitstoot; per categorie en cumulatief
fig_overtime, axs = plt.subplots(2,2, figsize=(20, 20))
kwargs = dict(fontsize=10)
L4 = df.groupby(['Jaar','L4_clean']).agg({'Emissie_20yr':'sum', 'Emissie_100yr':'sum'}).unstack()
_ = L4['Emissie_20yr'].plot(ax=axs[0, 0], legend=False, **kwargs)
_ = L4['Emissie_100yr'].plot(ax=axs[0, 1], **kwargs)
_ = L4['Emissie_20yr'].plot.area(ax=axs[1, 0], legend=False, **kwargs)
_ = L4['Emissie_100yr'].plot.area(ax=axs[1, 1], **kwargs)

# Zet limieten, titels, en weergave van de assen
for i in range(2):
    _ = axs[0, i].set_ylim(top=1e5)
    _ = axs[1, i].set_ylim(top=4e5) 
    _ = axs[i, 0].set_title('CO2-equivalente uitstoot [mln kg]\n'
                            + 'GPW 20 jaar (CO2: 1, CH4: 86, N2O: 268)')
    _ = axs[i, 1].set_title('CO2-equivalente uitstoot [mln kg]\n' 
                            + 'GPW 100 jaar (CO2: 1, CH4: 34, N2O: 298)')
    for j in range(2):
        axs[i, j].set_yticklabels(['{:,}'.format(int(x)) for x in axs[i, j].get_yticks().tolist()])

