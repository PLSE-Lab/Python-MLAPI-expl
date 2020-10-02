#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pandas_highcharts.core import serialize
from pandas_highcharts.display import display_charts
from IPython.display import display
import numpy as np
pd.options.display.max_columns = None
#pd.options.display.max_rows = None
fileAll = "../input/all.csv"
filePOP = "../input/india-districts-census-2011.csv"
fileAssam = "../input/Assam.csv"
fileBihar = "../input/Bihar.csv"
fileAndPr = "../input/AndhraPradesh.csv"
fileKrntk = "../input/karnataka.csv"
fileKerala = "../input/Kerala.csv"
fileMH  = "../input/Maharashtra.csv"
fileODS = "../input/Odisha.csv"
filePNB = "../input/Punjab.csv"
fileRSN = "../input/Rajasthan.csv"
fileUP  = "../input/UttarPradesh.csv"
fileWB = "../input/westbengal.csv"
dataAll= pd.read_csv(fileAll, encoding='utf-8')
dataAssam = pd.read_csv(fileAssam , encoding='utf-8')
dataBihar = pd.read_csv(fileBihar , encoding='utf-8')
dataAP= pd.read_csv(fileAndPr , encoding='utf-8')
dataKarnatka = pd.read_csv(fileKrntk , encoding='utf-8')
dataKerala = pd.read_csv(fileKerala , encoding='utf-8')
dataMH = pd.read_csv(fileMH  , encoding='utf-8')
dataODS = pd.read_csv(fileODS  , encoding='utf-8')
dataPNB = pd.read_csv(filePNB  , encoding='utf-8')
dataRSN = pd.read_csv(fileRSN  , encoding='utf-8')
dataUP = pd.read_csv(fileUP  , encoding='utf-8')
dataWB = pd.read_csv(fileWB  , encoding='utf-8')
dataPOP = pd.read_csv(filePOP  , encoding='utf-8')

# Any results you write to the current directory are saved as output.
# GDP Manupilation
MH = dataMH.transpose().reset_index()
MH.columns = MH.iloc[1]
MH.columns = MH.columns+'|'+MH.iloc[0,:]
MH = MH.rename(columns={'Description|Year':'District'})
MH.insert(0,'State', 'Maharashtra')
finalMH = MH.drop([0, 1])

BR = dataBihar.transpose().reset_index()
BR.columns = BR.iloc[1]
BR.columns = BR.columns+'|'+BR.iloc[0,:]
BR = BR.rename(columns={'Description|Year':'District'})
BR.insert(0,'State', 'Bihar')
finalBR = BR.drop([0, 1])

ASM = dataAssam.transpose().reset_index()
ASM.columns = ASM.iloc[1]
ASM.columns = ASM.columns+'|'+ASM.iloc[0,:]
ASM = ASM.rename(columns={'Description|Year':'District'})
ASM.insert(0,'State', 'Assam')
finalASM = ASM.drop([0, 1])

ANP = dataAP.transpose().reset_index()
ANP.columns = ANP.iloc[1]
ANP.columns = ANP.columns+'|'+ANP.iloc[0,:]
ANP = ANP.rename(columns={'Description|Year':'District'})
ANP.insert(0,'State', 'Andhra')
finalANP = ANP.drop([0, 1])

KNT = dataKarnatka.transpose().reset_index()
KNT.columns = KNT.iloc[1]
KNT.columns = KNT.columns+'|'+KNT.iloc[0,:]
KNT = KNT.rename(columns={'Description|Year':'District'})
KNT.insert(0,'State', 'Karnataka')
finalKNT = KNT.drop([0, 1])

KRL = dataKerala.transpose().reset_index()
KRL.columns = KRL.iloc[1]
KRL.columns = KRL.columns+'|'+KRL.iloc[0,:]
KRL = KRL.rename(columns={'Description|Year':'District'})
KRL.insert(0,'State', 'Kerala')
finalKRL = KRL.drop([0, 1])

ODS = dataODS.transpose().reset_index()
ODS.columns = ODS.iloc[1]
ODS.columns = ODS.columns+'|'+ODS.iloc[0,:]
ODS = ODS.rename(columns={'Description|Year':'District'})
ODS.insert(0,'State', 'Orrisa')
finalODS = ODS.drop([0, 1])

PNB = dataPNB.transpose().reset_index()
PNB.columns = PNB.iloc[1]
PNB.columns = PNB.columns+'|'+PNB.iloc[0,:]
PNB = PNB.rename(columns={'Description|Year':'District'})
PNB.insert(0,'State', 'Punjab')
finalPNB = PNB.drop([0, 1])

RSN = dataRSN.transpose().reset_index()
RSN.columns = RSN.iloc[1]
RSN.columns = RSN.columns+'|'+RSN.iloc[0,:]
RSN = RSN.rename(columns={'Description|Year':'District'})
RSN.insert(0,'State', 'Rajasthan')
finalRSN = RSN.drop([0, 1])

UP = dataUP.transpose().reset_index()
UP.columns = UP.iloc[1]
UP.columns = UP.columns+'|'+UP.iloc[0,:]
UP = UP.rename(columns={'Description|Year':'District'})
UP.insert(0,'State', 'UP')
finalUP = UP.drop([0, 1])

WB = dataWB.transpose().reset_index()
WB.columns = WB.iloc[1]
WB.columns = WB.columns+'|'+WB.iloc[0,:]
WB = WB.rename(columns={'Description|Year':'District'})
WB.insert(0,'State', 'WB')
finalWB = WB.drop([0, 1])


# <h1> Population </h1>

# In[ ]:


dataAll['Rural'] = pd.to_numeric(dataAll['Rural'], errors='coerce')
dataAll['Urban'] = pd.to_numeric(dataAll['Urban'], errors='coerce')
dataAll['Scheduled.Caste.population'] = pd.to_numeric(dataAll['Scheduled.Caste.population'], errors='coerce')
dataAll['Percentage...SC.to.total'] = pd.to_numeric(dataAll['Percentage...SC.to.total'], errors='coerce')
pldf = dataAll.groupby(['State'])[['Persons','Males','Females','Growth..1991...2001.','Rural','Urban','Scheduled.Caste.population','Percentage...SC.to.total']].sum()
pldf['Growth 1991-2001'] = pldf['Growth..1991...2001.'].div(dataAll.groupby(['State'])['District'].count())
del pldf['Growth..1991...2001.']
display_charts(pldf , kind="bar", title="Population",figsize = (1000, 700))


# <h1> HouseHold </h1>

# In[ ]:


dataAll['Scheduled.Tribe.population'] =  pd.to_numeric(dataAll['Scheduled.Tribe.population'], errors='coerce')
dataAll['Percentage.to.total.population..ST.'] = pd.to_numeric(dataAll['Percentage.to.total.population..ST.'], errors='coerce')
dataAll['Number.of.households'] = pd.to_numeric(dataAll['Number.of.households'], errors='coerce')
dataAll['Household.size..per.household.'] = pd.to_numeric(dataAll['Household.size..per.household.'], errors='coerce')
dataAll['Sex.ratio..females.per.1000.males.'] = pd.to_numeric(dataAll['Sex.ratio..females.per.1000.males.'], errors='coerce')
dataAll['Sex.ratio..0.6.years.'] = pd.to_numeric(dataAll['Sex.ratio..0.6.years.'], errors='coerce')
dataAll['Scheduled.Tribe.population'] = pd.to_numeric(dataAll['Scheduled.Tribe.population'], errors='coerce')
dataAll['Percentage.to.total.population..ST.'] = pd.to_numeric(dataAll['Percentage.to.total.population..ST.'], errors='coerce')

HHdf = dataAll.groupby(['State'])[['Number.of.households','Household.size..per.household.','Sex.ratio..females.per.1000.males.','Sex.ratio..0.6.years.','Scheduled.Tribe.population','Percentage.to.total.population..ST.']].sum()
HHdf['HH Size Per HouseHold'] = HHdf['Household.size..per.household.'].div(dataAll.groupby(['State'])['District'].count())
HHdf['Sex Ratio (Females) Per 1000 Males'] = HHdf['Sex.ratio..females.per.1000.males.'].div(dataAll.groupby(['State'])['District'].count())
HHdf['Sex Ratio 0-6 Years'] = HHdf['Sex.ratio..0.6.years.'].div(dataAll.groupby(['State'])['District'].count())
HHdf['Percentage to total Population ST'] = HHdf['Percentage.to.total.population..ST.'].div(dataAll.groupby(['State'])['District'].count())

HHdf.drop(['Household.size..per.household.','Sex.ratio..females.per.1000.males.','Sex.ratio..0.6.years.','Percentage.to.total.population..ST.'], axis = 1, inplace = True)
display_charts(HHdf , kind="bar", title="HouseHold",figsize = (1000, 700))


# <h1> Scheduled Tribe population in different states of India</h1>

# In[ ]:


SCdf = dataAll.groupby(['State'])[['Scheduled.Tribe.population']].sum()
display_charts(SCdf , kind="bar", title="State Wise ST populations in India",figsize = (1000, 700))


# <h1> Sex ratio of females per 1000 males in different states of India </h1>

# In[ ]:


SRdf = round(dataAll.groupby(['State'])[['Sex.ratio..females.per.1000.males.']].sum().div(dataAll.groupby(['State'])[['Sex.ratio..females.per.1000.males.']].count())).rename(columns={'Sex.ratio..females.per.1000.males.':'Females Per 1000 Males'})
display_charts(SRdf , kind="bar", title="Sex ratio of females per 1000 males in different states of India",figsize = (1000, 700))


# <h1> Sex ratio 0-6 years In different states of India </h1>

# In[ ]:


Srcdf = round(dataAll.groupby(['State'])[['Sex.ratio..0.6.years.']].sum().div(dataAll.groupby(['State'])[['Sex.ratio..0.6.years.']].count())).rename(columns={'Sex.ratio..0.6.years.':'Sex ratio 0-6 years'})
display_charts(SRdf , kind="bar", title="Sex ratio 0-6 years in different states of India",figsize = (1000, 700))


# <h1>Sex wise Literate population in different states of India </h1>

# In[ ]:


litdf = dataAll.groupby(['State'])[['Persons..literate','Males..Literate','Females..Literate']].sum()
display_charts(litdf , kind="bar", title="Sex wise Literate population in different states of India",figsize = (1000, 700))


# <h1> Sex wise Literacy Rate in different states of India </h1>

# In[ ]:


literacydf = dataAll.groupby(['State'])[['Persons..literacy.rate','Males..Literatacy.Rate','Females..Literacy.Rate']].sum().div(dataAll.groupby(['State'])[['Persons..literacy.rate','Males..Literatacy.Rate','Females..Literacy.Rate']].count())
display_charts(literacydf , kind="area", title="Sex wise Literacy Rate in different states of India",figsize = (1000, 700))


# <h1>Types of  Education Level Attended populations in different states of India</h1>

# In[ ]:


edulvldf = dataAll.groupby(['State'])[['Total.Educated','Data.without.level','Below.Primary','Primary','Middle','Matric.Higher.Secondary.Diploma','Graduate.and.Above']].sum()
display_charts(edulvldf , kind="line", title="Types of  Education Level Attended populations in different states of India",figsize = (1000, 700))


# <h1>Populations of different Age Groups In different states of  India </h1>

# In[ ]:


ageGrpdf = dataAll.groupby(['State'])[['X0...4.years','X5...14.years','X15...59.years','X60.years.and.above..Incl..A.N.S..']].sum().rename(columns={'X0...4.years':'0-4 Years','X5...14.years':'5-14 Years','X15...59.years':'15-59 Years','X60.years.and.above..Incl..A.N.S..':'60 Years & Above'})
display_charts(ageGrpdf , kind="bar", title="Populations of different Age Groups In different states of  India",figsize = (1000, 700))


# <h1>Populations of different Types of  Workers In different states of India </h1>

# In[ ]:


workdf = dataAll.groupby(['State'])[['Total.workers','Main.workers','Marginal.workers','Non.workers']].sum()
display_charts(workdf , kind="area", title="Populations of different Types of  Workers In different states of India ",figsize = (1000, 700))


# <h1> Different types of Scheduled Caste Populations in India </h1>

# In[ ]:


dataAll['SC.1.Population'] = pd.to_numeric(dataAll['SC.1.Population'], errors='coerce')
dataAll['SC.1.Population'] = pd.to_numeric(dataAll['SC.1.Population'], errors='coerce')
sc1df = dataAll.groupby(['SC.1.Name'])[['SC.1.Population']].sum()
display_charts(sc1df , kind="bar", title="Different types of Scheduled Caste Populations in India",figsize = (1000, 700))


# <h1> Different types of Religion's population in India </h1>

# In[ ]:


reldf = dataAll.groupby(['Religeon.1.Name'])[['Religeon.1.Population']].sum()
display_charts(reldf , kind="bar", title="Different types of Religion population in India ",figsize = (1000, 700))


# <h1> Different types of Scheduled Tribes population In India </h1>

# In[ ]:


dataAll['ST.1.Population'] = pd.to_numeric(dataAll['ST.1.Population'], errors='coerce') 
scdf = dataAll.groupby(['ST.1.Name'])[['ST.1.Population']].sum()
display_charts(scdf , kind="bar", title="Different types of Scheduled Tribes population In India",figsize = (1000, 700))


# <h1> Town Wise  Population of India </h1>

# In[ ]:


dataAll['Imp.Town.1.Population'] = pd.to_numeric(dataAll['Imp.Town.1.Population'], errors='coerce') 
twdf = dataAll.groupby(['Imp.Town.1.Name'])[['Imp.Town.1.Population']].sum()
display_charts(twdf , kind="line", title="Town Wise Populations In India",figsize = (1000, 700))


# <h1> State wise total Inhabited Village In India</h1>

# In[ ]:


dataAll['Total.Inhabited.Villages'] = pd.to_numeric(dataAll['Total.Inhabited.Villages'], errors='coerce') 
tivdf = dataAll.groupby(['State'])[['Total.Inhabited.Villages']].sum()
display_charts(tivdf , kind="bar", title="State wise total Inhabited Village In India",figsize = (1000, 700))


# <h1> State wise population having Drinking Water Facilities In India </h1>

# In[ ]:


dataAll['Drinking.water.facilities'] = pd.to_numeric(dataAll['Drinking.water.facilities'], errors='coerce')
dataAll['Safe.Drinking.water'] = pd.to_numeric(dataAll['Safe.Drinking.water'], errors='coerce') 
dwfdf = dataAll.groupby(['State'])[['Drinking.water.facilities','Safe.Drinking.water']].sum()
display_charts(dwfdf , kind="area", title="State wise population having Drinking Water Facilities In India",figsize = (1000, 700))


# <h1>State wise populations having Electric Power Supply In India </h1>

# In[ ]:


dataAll['Electricity..Power.Supply.'] = pd.to_numeric(dataAll['Electricity..Power.Supply.'], errors='coerce')
dataAll['Electricity..domestic.'] = pd.to_numeric(dataAll['Electricity..domestic.'], errors='coerce')
dataAll['Electricity..Agriculture.'] = pd.to_numeric(dataAll['Electricity..Agriculture.'], errors='coerce')
epsdf = dataAll.groupby(['State'])[['Electricity..Power.Supply.','Electricity..domestic.','Electricity..Agriculture.']].sum()
display_charts(epsdf , kind="bar", title="State wise populations having Electric Power Supply In India",figsize = (1000, 700))


# <h1> State Wise Housing Percentage In India <h1>

# In[ ]:


dfHousing = dataAll.groupby(['State'])[['Permanent.House','Semi.permanent.House','Temporary.House']].sum().div(dataAll.groupby(['State'])[['Permanent.House','Semi.permanent.House','Temporary.House']].count())
display_charts(dfHousing , kind="bar", title="State Wise Housing Percentage In India " ,figsize = (1000, 700))


# <h1> State Wise Yearly GDP In India </h1>

# In[ ]:


dfbr = pd.melt(dataBihar, id_vars=["Year", "Description"], var_name=["District name"])
dfbr['State Name'] = 'BIHAR'

dfasm = pd.melt(dataAssam, id_vars=["Year", "Description"], var_name=["District name"])
dfasm['State Name'] = 'ASSAM'

dfanp = pd.melt(dataAP,id_vars=["Year", "Description"], var_name=["District name"])
dfanp['State Name'] = 'ANDHRA PRADESH'

dfkrnt = pd.melt(dataKarnatka,id_vars=["Year", "Description"], var_name=["District name"])
dfkrnt['State Name'] = 'KARNATAKA'

dfkerala = pd.melt(dataKerala,id_vars=["Year", "Description"], var_name=["District name"])
dfkerala['State Name'] = 'KERALA'

dfMH = pd.melt(dataMH,id_vars=["Year", "Description"], var_name=["District name"])
dfMH['State Name'] = 'MAHARASHTRA'

dfODS = pd.melt(dataODS,id_vars=["Year", "Description"], var_name=["District name"])
dfODS['State Name'] = 'ORISSA'

dfPNB = pd.melt(dataPNB,id_vars=["Year", "Description"], var_name=["District name"])
dfPNB['State Name'] = 'PUNJAB'

dfRSN = pd.melt(dataRSN,id_vars=["Year", "Description"], var_name=["District name"])
dfRSN['State Name'] = 'RAJASTHAN'

dfUP = pd.melt(dataUP,id_vars=["Year", "Description"], var_name=["District name"])
dfUP['State Name'] = 'UTTAR PRADESH'

dfWB = pd.melt(dataWB,id_vars=["Year", "Description"], var_name=["District name"])
dfWB['State Name'] = 'WEST BENGAL'

dfGDP = pd.DataFrame(dfbr.append([dfasm,dfanp,dfkrnt,dfkerala,dfMH,dfODS,dfPNB,dfRSN,dfUP,dfWB]))
dfGDP['value'] = dfGDP['value'].replace(',','', regex=True).astype(float)
x = pd.DataFrame(dfGDP.groupby(['State Name','Year']).apply(lambda x: x[x['Description'] == 'GDP (in Rs. Cr.)']['value'].sum()))
display_charts(x , kind="bar", title="State Wise Yearly India GDP",figsize = (1000, 700))


# <h1> District Wise Yearly GDP Growth Rate Percentage In India </h1>

# In[ ]:


disdf = pd.DataFrame(dfGDP.groupby(['State Name','Year']).apply(lambda x: x[x['Description'] == 'Growth Rate % (YoY)']['value'].sum()).div(dfGDP.groupby(['State Name','Year']).apply(lambda x: x[x['Description'] == 'Growth Rate % (YoY)']['value'].count())))
display_charts(disdf , kind="bar", title="GDP Growth Rate Percent",figsize = (1000, 700))


# <h1>State Wise Per Capita Income </h1>

# In[ ]:


dfGDP = pd.DataFrame(dfbr.append([dfasm,dfanp,dfkrnt,dfkerala,dfMH,dfODS,dfPNB,dfRSN,dfUP,dfWB]))
dfGDP['value'] = dfGDP['value'].replace(',','', regex=True).astype(float)
dfGDP = dfGDP[dfGDP.Description != 'Growth Rate % (YoY)']
PerCapdf = pd.merge(dfGDP[['State Name','Year','value','District name']],dataPOP[['State Name','District name','Population']],on=['State Name','District name'])
PerCapdf['Per Capita Income'] = PerCapdf['value'].div(PerCapdf['Population']).multiply(10000000)
PerCapIncome =  pd.DataFrame((PerCapdf.groupby(['State Name','Year'])['Per Capita Income'].sum()).div(PerCapdf.groupby(['State Name','Year'])['Per Capita Income'].count()))
display_charts(PerCapIncome , kind="bar", title="State Wise Per Capita Income " ,figsize = (1000, 700))


# <h1> State Wise BPL Populations In India </h1>

# In[ ]:


PerCapdf['Population BPL'] = 0
PerCapdf.loc[PerCapdf['Per Capita Income'] <=27000, 'Population BPL'] = PerCapdf['Population']
PerBPL = pd.DataFrame((PerCapdf.groupby(['State Name','Year'])['Population BPL'].sum()))
display_charts(PerBPL , kind="bar", title="State Wise BPL Population In India " ,figsize = (1000, 700))


# <h1> State Wise GDP data </h1>

# In[ ]:


dfFinalGDP = pd.DataFrame(finalMH.append([finalBR,finalASM,finalANP,finalKNT,finalKRL,finalODS,finalPNB,finalRSN,finalUP,finalWB]))
dfFinalGDP = dfFinalGDP.replace(np.NaN, 'NA', regex=True)
dfFinalGDP


# <h1> Socio Economic Data </h1>

# In[ ]:


dataAll['District'] = dataAll['District'].str.split('(').str[0].str.split('*').str[0].str.replace('District' , '').str.strip()
dataCombined = pd.merge(dataAll,dfFinalGDP,how='left',on=['State','District']) 
dataCombined[['PerCapita Income (2004-05)','PerCapita Income (2005-06)','PerCapita Income (2006-07)','PerCapita Income (2007-08)','PerCapita Income (2008-09)','PerCapita Income (2009-10)','PerCapita Income (2010-11)','PerCapita Income (2011-12)','PerCapita Income (2012-13)']] = dataCombined[['GDP (in Rs. Cr.)|2004-05','GDP (in Rs. Cr.)|2005-06','Growth Rate % (YoY)|2006-07','GDP (in Rs. Cr.)|2007-08','GDP (in Rs. Cr.)|2008-09','GDP (in Rs. Cr.)|2009-10','GDP (in Rs. Cr.)|2010-11','GDP (in Rs. Cr.)|2011-12','GDP (in Rs. Cr.)|2012-13']].replace(',','', regex=True).replace('NA',0, regex=True).astype(float).div(dataCombined['Persons'].values,axis=0).multiply(10000000).round(2)
dataCombined = dataCombined.replace(np.NaN, 'NA', regex=True)
dataCombined


# <h1> Socio Economic Data With Aggregate GDP and Percapita Income</h1>

# In[ ]:


dfSGDP = dfFinalGDP
dfSGDP['GDP'] = dfSGDP[['GDP (in Rs. Cr.)|2004-05','GDP (in Rs. Cr.)|2005-06','GDP (in Rs. Cr.)|2006-07','GDP (in Rs. Cr.)|2007-08','GDP (in Rs. Cr.)|2008-09','GDP (in Rs. Cr.)|2009-10','GDP (in Rs. Cr.)|2010-11','GDP (in Rs. Cr.)|2011-12','GDP (in Rs. Cr.)|2012-13']].replace(',','', regex=True).replace('NA',0, regex=True).astype(float).mean(axis=1).round(2)
dfSGDP['Growth Rate %'] = dfSGDP[['Growth Rate % (YoY)|2005-06','Growth Rate % (YoY)|2006-07','Growth Rate % (YoY)|2007-08','Growth Rate % (YoY)|2008-09','Growth Rate % (YoY)|2009-10','Growth Rate % (YoY)|2010-11','Growth Rate % (YoY)|2011-12','Growth Rate % (YoY)|2012-13']].replace(',','', regex=True).replace('NA',0, regex=True).astype(float).mean(axis=1).round(2)
dfSGDP = dfSGDP[['State','District','GDP','Growth Rate %']]
dataSocio = pd.merge(dataAll,dfSGDP,how='left',on=['State','District'])
dataSocio['PerCapita Income'] =  dataSocio[['GDP']].replace(',','', regex=True).replace('NA',0, regex=True).astype(float).div(dataSocio['Persons'].values,axis=0).multiply(10000000).round(2)
dataSocio['BPL Population'] = 0
dataSocio.loc[dataSocio['PerCapita Income'].between(0.01,27000), 'BPL Population'] = dataSocio['Persons']
dataSocio =  dataSocio.replace(np.NaN, 'NA', regex=True)
dataSocio.loc[dataSocio['PerCapita Income'] == 'NA', 'BPL Population'] = "NA"
dataSocio

