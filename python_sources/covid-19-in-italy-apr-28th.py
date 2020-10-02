#!/usr/bin/env python
# coding: utf-8

# <a id="Index"><font size=22>Index</font></a>

# * <a href="#Dataset">Dataset</a>
# * <a href="#RegionSummary">Italy</a><br>
#   * <a href="#RegionSummary">Summary table by Region</a>
#   * <a href="#ItalyDailyNewCases">Daily new cases</a>
#   * <a href="#ItalyCumulativeDeaths">Cumulative deaths</a>
#   * <a href="#ItalyDailyDeaths">Daily deaths</a>
#   * <a href="#ItalyDeathPctTrend">Daily deaths trend</a>
#   * <a href="#ItalyIntCareProgress">Intensive Care Progress</a>
#   * <a href="#ItalyDailyIntCare">Daily Intensive Care</a>
#   * <a href="#ItalyNewCasesProgress">New Cases Progress</a>
# * <a href="#LombardiaCumulativeDeaths">Lombardia</a><br>
#   * <a href="#LombardiaCumulativeDeaths">Cumulative deaths</a>
#   * <a href="#LombardiaDailyDeaths">Daily deaths</a>
#   * <a href="#LombardiaDeathPctTrend">Daily deaths trend</a>
#   * <a href="#LombardiaIntCareProgress">Intensive Care Progress</a>
#   * <a href="#LombardiaDailyIntCare">Daily Intensive Care</a>
#   * <a href="#LombardiaNewCasesProgress">New Cases Progress</a>
# * <a href="#LazioCumulativeDeaths">Lazio</a><br>
#   * <a href="#LazioCumulativeDeaths">Cumulative deaths</a>
#   * <a href="#LazioDailyDeaths">Daily deaths</a>
#   * <a href="#LazioDeathPctTrend">Daily deaths trend</a>
#   * <a href="#LazioIntCareProgress">Intensive Care Progress</a>
#   * <a href="#LazioDailyIntCare">Daily Intensive Care</a>
#   * <a href="#LazioNewCasesProgress">New Cases Progress</a>
# * <a href="#ComparisonHealedDeaths">Comparison Italy Deaths vs Healed</a> 

# <a id="Dataset"><font size=22>Dataset</font></a><br>
# <a href="#Index">Back to Index</a>

# **Last Dataset Update**: 28/04/2020 17:00:00 (ref. https://github.com/pcm-dpc/COVID-19)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

from pandas import read_excel

my_sheet = 'dpc-covid19-ita-regioni'
file_name = '/kaggle/input/dpccovid19itaregioni-20200428/dpc-covid19-ita-regioni.xlsx' 
df = read_excel(file_name, sheet_name = my_sheet)
df.head()


# <a id="RegionSummary"><font size=22>Italy summary by region</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


ricoverati_con_sintomi = df.groupby("denominazione_regione")["ricoverati_con_sintomi"].max()
terapia_intensiva = df.groupby("denominazione_regione")["terapia_intensiva"].max()
totale_ospedalizzati = df.groupby("denominazione_regione")["totale_ospedalizzati"].max()
isolamento_domiciliare = df.groupby("denominazione_regione")["isolamento_domiciliare"].max()
#totale_attualmente_positivi = df.groupby("denominazione_regione")["totale_attualmente_positivi"].max()
totale_attualmente_positivi = df.groupby("denominazione_regione")["totale_positivi"].max()
#nuovi_attualmente_positivi = df.groupby("denominazione_regione")["nuovi_attualmente_positivi"].max()
nuovi_attualmente_positivi = df.groupby("denominazione_regione")["nuovi_positivi"].max()
dimessi_guariti = df.groupby("denominazione_regione")["dimessi_guariti"].max()
deceduti = df.groupby("denominazione_regione")["deceduti"].max()
totale_casi = df.groupby("denominazione_regione")["totale_casi"].max()
tamponi = df.groupby("denominazione_regione")["tamponi"].max()


#df2['x'] = italia.groupby("denominazione_regione")["ricoverati_con_sintomi"].max()
italia = pd.concat([ricoverati_con_sintomi, terapia_intensiva, totale_ospedalizzati, isolamento_domiciliare, 
                   totale_attualmente_positivi,nuovi_attualmente_positivi,dimessi_guariti,deceduti,
                   totale_casi, tamponi], axis=1)

#italia.sum(axis=0)
italia['Total'] = italia.sum(axis=1)
italia.loc['Total']= italia.sum()
italia.to_excel('italia.xlsx', sheet_name='data')

core = pd.DataFrame(data=italia)
core = core.sort_values(['totale_casi'], ascending=[False])
#core = core[['totale_casi', 'nuovi_attualmente_positivi', 'terapia_intensiva', 'deceduti', 'dimessi_guariti', 'tamponi', 'Total']]
core = core[['totale_casi', 'nuovi_positivi', 'terapia_intensiva', 'deceduti', 'dimessi_guariti', 'tamponi', 'Total']]
core.to_excel('core.xlsx', sheet_name='data')
italia


# <a id="ItalyNewCases"><font size=22>Italy new cases</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


df['tempo'] = df['data'].map( lambda d: pd.to_datetime(d).timetuple().tm_yday )
de = df.groupby("tempo")["nuovi_positivi"].sum()
#de = df.groupby("tempo")["nuovi_attualmente_positivi"].sum()
ax1 = plt.axes()
ax1.xaxis.label.set_visible(False)
de.plot(figsize=(10,6), kind='bar', title="ITALY - NEW CASES")


# <a id="ItalyDailyNewCases"><font size=22>Daily new cases</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


nc = df.groupby("tempo")["nuovi_positivi"].sum()
#nc = df.groupby("tempo")["nuovi_attualmente_positivi"].sum()
nc = nc.to_frame()
nc['daily'] = nc['nuovi_positivi'].diff()
#nc['daily'] = nc['nuovi_attualmente_positivi'].diff()
nc_daily = nc['daily'].to_frame()
nc.tail(10).sort_index(ascending=False, axis=0)


# In[ ]:


nc_daily_graph = nc_daily['daily'] 
plt.axes().xaxis.label.set_visible(False)
nc_daily_graph.plot(figsize=(10,6), kind='bar', title="ITALY - DAILY NEW CASES")


# <a id="ItalyCumulativeDeaths"><font size=22>Italy Cumulative Deaths</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


df['tempo'] = df['data'].map( lambda d: pd.to_datetime(d).timetuple().tm_yday )
de = df.groupby("tempo")["deceduti"].sum()
ax1 = plt.axes()
ax1.xaxis.label.set_visible(False)
de.plot(figsize=(10,6), kind='bar', title="ITALY - CUMULATIVE DEATHS")


# <a id="ItalyDailyDeaths"><font size=22>Italy daily Deaths</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


de = df.groupby("tempo")["deceduti"].sum()
de = de.to_frame()
de['daily'] = de['deceduti'].diff()
de_daily = de['daily'].to_frame()
de.tail(10).sort_index(ascending=False, axis=0)


# In[ ]:


de_daily.plot(figsize=(10,6), kind='bar', title="ITALY - DAILY DEATHS")


# <a id="ItalyDeathPctTrend"><font size=22>Italy daily deaths trend</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


de_daily["pct"] = round(de_daily.pct_change(), 3)*100
de_daily["avg"] = round(de_daily["pct"].expanding().mean(), 3)
de_daily_pct = de_daily["pct"]
de_daily_avg = de_daily["avg"]
de_daily.tail(10).sort_index(ascending=False, axis=0)


# In[ ]:


plt.axes().xaxis.label.set_visible(False)
de_daily_avg.plot(figsize=(10,6), kind='bar', title="ITALY - DAILY DEATHS TREND")


# <a id="ItalyIntCareProgress"><font size=22>Italy intensive care progress</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


plt.axes().xaxis.label.set_visible(False)
ti = df.groupby("tempo")["terapia_intensiva"].sum()
ti.plot(figsize=(10,6), kind='bar', title="ITALY - INTENSIVE CARE PROGRESS")


# <a id="ItalyDailyIntCare"><font size=22>Italy daily Intensive Care</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


ti = ti.to_frame()
ti['daily'] = ti['terapia_intensiva'].diff()
ti_daily = ti['daily'].to_frame()
ti.tail(10).sort_index(ascending=False, axis=0)


# In[ ]:


ti_daily.plot(figsize=(10,6), kind='bar', title="ITALY - DAILY INTENSIVE CARE")


# <a id="ItalyNewCasesProgress"><font size=22>Italy New Cases Progress</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


plt.axes().xaxis.label.set_visible(False)
it_np = df.groupby("tempo")["nuovi_positivi"].sum()
it_np.plot(figsize=(10,6), kind='bar', title="ITALY - NEW CASES PROGRESS")


# <a id="LombardiaCumulativeDeaths"><font size=22>Lombardia Cumulative deaths</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


lombardia = df[df['denominazione_regione']=='Lombardia']
lombardia_de = lombardia.groupby("tempo")["deceduti"].sum()
lombardia_de.tail(10).sort_index(ascending=False, axis=0).to_frame()


# In[ ]:


plt.axes().xaxis.label.set_visible(False)
lombardia_de.plot(figsize=(10,6), kind='bar', title="LOMBARDIA - CUMULATIVE DEATHS")


# <a id="LombardiaDailyDeaths"><font size=22>Lombardia daily deaths</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


lombardia_de['daily'] = lombardia['deceduti'].diff()
lombardia_de_daily = lombardia_de['daily'].to_frame()
lombardia_de_daily.tail(10).sort_index(ascending=False, axis=0)


# In[ ]:


lombardia_de_daily.plot(figsize=(10,6), kind='bar', title="LOMBARDIA - DAILY DEATHS")


# <a id="LombardiaDeathPctTrend"><font size=22>Lombardia daily deaths trend</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


lombardia_de_daily["pct"] = round(lombardia_de_daily.pct_change(), 3)*100
lombardia_de_daily["avg"] = round(lombardia_de_daily["pct"].expanding().mean(), 3)
lombardia_de_daily_pct = lombardia_de_daily["pct"]
lombardia_de_daily_avg = lombardia_de_daily["avg"]
lombardia_de_daily.tail(10).sort_index(ascending=False, axis=0)


# In[ ]:


plt.axes().xaxis.label.set_visible(False)
lombardia_de_daily_avg.plot(figsize=(10,6), kind='bar', title="LOMBARDIA - DAILY DEATHS TREND")


# <a id="LombardiaIntCareProgress"><font size=22>Lombardia Intensive Care Progress</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


plt.axes().xaxis.label.set_visible(False)
lombardiati = lombardia.groupby("tempo")["terapia_intensiva"].sum()
lombardiati.plot(figsize=(10,6), kind='bar', title="LOMBARDIA - INTENSIVE CARE PROGRESS")


# <a id="LombardiaDailyIntCare"><font size=22>Lombardia Daily Intensive Care</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


ti_lo = lombardia.groupby("tempo")["terapia_intensiva"].sum()
ti_lo = ti_lo.to_frame()
ti_lo['daily'] = ti_lo['terapia_intensiva'].diff()
ti_lo_daily = ti_lo['daily'].to_frame()
ti_lo.tail(10).sort_index(ascending=False, axis=0)


# In[ ]:


ti_lo_daily.plot(figsize=(10,6), kind='bar', title="LOMBARDIA - DAILY INTENSIVE CARE")


# <a id="LombardiaNewCasesProgress"><font size=22>Lombardia New Cases Progress</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


plt.axes().xaxis.label.set_visible(False)
lombardia_np = lombardia.groupby("tempo")["nuovi_positivi"].sum()
lombardia_np.plot(figsize=(10,6), kind='bar', title="LOMBARDIA - NEW CASES PROGRESS")


# <a id="LazioCumulativeDeaths"><font size=22>Lazio Cumulative deaths</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


lazio = df[df['denominazione_regione']=='Lazio']
lazio_de = lazio.groupby("tempo")["deceduti"].sum()
lazio_de.tail(10).sort_index(ascending=False, axis=0).to_frame()


# In[ ]:


plt.axes().xaxis.label.set_visible(False)
lazio_de.plot(figsize=(10,6), kind='bar', title="LAZIO - CUMULATIVE DEATHS")


# <a id="LazioDailyDeaths"><font size=22>Lazio daily deaths</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


lazio_de['daily'] = lazio['deceduti'].diff()
lazio_de_daily = lazio_de['daily'].to_frame()
lazio_de_daily.tail(10).sort_index(ascending=False, axis=0)


# In[ ]:


lazio_de_daily.plot(figsize=(10,6), kind='bar', title="LAZIO - DAILY DEATHS")


# <a id="LazioDeathPctTrend"><font size=22>Lazio daily deaths trend</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


lazio_de_daily["pct"] = round(lazio_de_daily.pct_change(), 3)*100
lazio_de_daily["avg"] = round(lazio_de_daily["pct"].expanding().mean(), 3)
lazio_de_daily_pct = lazio_de_daily["pct"]
lazio_de_daily_avg = lazio_de_daily["avg"]
lazio_de_daily.tail(10).sort_index(ascending=False, axis=0)


# In[ ]:


plt.axes().xaxis.label.set_visible(False)
lazio_de_daily_avg.plot(figsize=(10,6), kind='bar', title="LAZIO - DAILY DEATHS TREND")


# <a id="LazioIntCareProgress"><font size=22>Lazio Intensive Care Progress</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


plt.axes().xaxis.label.set_visible(False)
lazioti = lazio.groupby("tempo")["terapia_intensiva"].sum()
lazioti.plot(figsize=(10,6), kind='bar', title="LAZIO - INTENSIVE CARE PROGRESS")


# <a id="LazioDailyIntCare"><font size=22>Lazio Daily Intensive Care</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


ti_la = lazio.groupby("tempo")["terapia_intensiva"].sum()
ti_la = ti_la.to_frame()
ti_la['daily'] = ti_la['terapia_intensiva'].diff()
ti_la_daily = ti_la['daily'].to_frame()
ti_la.tail(10).sort_index(ascending=False, axis=0)


# In[ ]:


ti_la_daily.plot(figsize=(10,6), kind='bar', title="LAZIO - DAILY INTENSIVE CARE")


# <a id="LazioNewCasesProgress"><font size=22>Lazio New Cases Progress</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


plt.axes().xaxis.label.set_visible(False)
lazio_np = lazio.groupby("tempo")["nuovi_positivi"].sum()
lazio_np.plot(figsize=(10,6), kind='bar', title="LAZIO - NEW CASES PROGRESS")


# <a id="ComparisonHealedDeaths"><font size=22>Comparison Italy Deaths vs Healed</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


he = df.groupby("tempo")["dimessi_guariti"].sum().to_frame()
he['healed'] = he.diff()

de = df.groupby("tempo")["deceduti"].sum().to_frame()
de['deaths'] = de.diff()

he_de = pd.concat([he, de], axis=1)
he_de = he_de.drop('dimessi_guariti', axis=1)
he_de = he_de.drop('deceduti', axis=1)
he_de['diff'] = he_de['healed'] - de['deaths']
he_de.sort_values(by="tempo", ascending=False).head(10)


# In[ ]:


he_de.plot(figsize=(14,6), kind='bar', title="ITALY - HEALED vs DEATHS")


# In[ ]:


he_de['diff'].plot(figsize=(14,6), kind='bar', title="ITALY - HEALED vs DEATHS")


# <a id="RegionalScore"><font size=22>Regional score</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


score = core.head(11)
score = score.drop(score.index[0])
score


# In[ ]:


#lombardia 10040000
#piemonte 4376000
#Emilia Romagna 4453000
#veneto 4905000
#toscana 3737000
#liguria 1557000
#lazio 5897000
#marche 1532000
#campania 5827000
#trentino 541000

#Friuli Venezia Giulia 1216000
#Puglia 4048000
pop = [10040000, 4376000, 4453000, 4905000, 3737000, 1532000, 5897000, 1532000, 5827000, 541000]

f = score
f['POPULATION'] = pop
a = score['totale_casi']/score['POPULATION']*100
f['a'] = a
b = score['dimessi_guariti']/score['totale_casi']*100
f['b'] = b
c = score['deceduti']/score['totale_casi']*100
f['c'] = c

f['SCORE'] = 100 - a + b - c

f = f.sort_values(['SCORE'], ascending=[False])
f


# From best to worst situation
# (China after crisis: 192.1)

# In[ ]:


final = score['SCORE']
s = round(final.to_frame().sort_values(['SCORE'], ascending=[False]), 2)
s


# In[ ]:


s.plot(figsize=(10,6), kind='bar', title="REGIONAL SCORE")


# <a id="RegionalScoreDayByDay"><font size=22>Regional score (day by day)</font></a><br>
# <a href="#Index">Back to Index</a>

# In[ ]:


lombardia_score = df[df['denominazione_regione']=='Lombardia']
lombardia_score = lombardia_score[['totale_casi', 'nuovi_positivi', 'terapia_intensiva', 'deceduti', 'dimessi_guariti', 'tamponi']]

a = lombardia_score['totale_casi']/10040000*100
lombardia_score['a'] = a
b = lombardia_score['dimessi_guariti']/lombardia_score['totale_casi']*100
lombardia_score['b'] = b
c = lombardia_score['deceduti']/lombardia_score['totale_casi']*100
lombardia_score['c'] = c

lombardia_score['SCORE'] = 100 - a + b - c

lombardia_score = lombardia_score.sort_values(['SCORE'], ascending=[False])
#lombardia_score = lombardia_score[['a', 'b', 'c', 'SCORE']]
lombardia_score.reset_index(inplace = True)
lombardia_score = lombardia_score['SCORE'].to_frame()
lombardia_score.head(15)


# In[ ]:


veneto_score = df[df['denominazione_regione']=='Veneto']
veneto_score = veneto_score[['totale_casi', 'nuovi_positivi', 'terapia_intensiva', 'deceduti', 'dimessi_guariti', 'tamponi']]

a = veneto_score['totale_casi']/4905000*100
veneto_score['a'] = a
b = veneto_score['dimessi_guariti']/veneto_score['totale_casi']*100
veneto_score['b'] = b
c = veneto_score['deceduti']/veneto_score['totale_casi']*100
veneto_score['c'] = c

veneto_score['SCORE'] = 100 - a + b - c

veneto_score = veneto_score.sort_values(['SCORE'], ascending=[False])
veneto_score = veneto_score[['a', 'b', 'c', 'SCORE']]
veneto_score.reset_index(inplace = True)
veneto_score = veneto_score['SCORE'].to_frame()
veneto_score.head(15)


romagna_score = df[df['denominazione_regione']=='Emilia-Romagna']
romagna_score = romagna_score[['totale_casi', 'nuovi_positivi', 'terapia_intensiva', 'deceduti', 'dimessi_guariti', 'tamponi']]

a = romagna_score['totale_casi']/4453000*100
romagna_score['a'] = a
b = romagna_score['dimessi_guariti']/romagna_score['totale_casi']*100
romagna_score['b'] = b
c = romagna_score['deceduti']/romagna_score['totale_casi']*100
romagna_score['c'] = c

romagna_score['SCORE'] = 100 - a + b - c

romagna_score = romagna_score.sort_values(['SCORE'], ascending=[False])
romagna_score = romagna_score[['a', 'b', 'c', 'SCORE']]
romagna_score.reset_index(inplace = True)
romagna_score = romagna_score['SCORE'].to_frame()
romagna_score.head(15)




frames = [lombardia_score, veneto_score, romagna_score]
final_score = pd.concat(frames, axis=1)
final_score.columns = ['LOMBARDIA','VENETO','EMILIA']
final_score.head(15)

