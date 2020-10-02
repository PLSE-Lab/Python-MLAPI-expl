#!/usr/bin/env python
# coding: utf-8

# # **Yet Another COVID-19 EDA/Visualization Dashboard...**
# ## by AY
# This notebook is a visualization dashboard for the dynamics/kinetics of the COVID-19 virus using the [2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository by Johns Hopkins CSSE](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset).  I will try to refresh the data daily as I don't think Kaggle will dynamically update it.  Really quick and dirty just to see what's in the data, this code is not elegant.  Mostly just doing this to learn how to use Plotly and delve into some signal/stochastic processes topics.
# 
# Currently Includes:
# 
# * [Time-series Dynamics curves](#dynamics-graphs); **Position** *(Current Number of Cases)*, **Speed** *(New Cases/Day)* and **Acceleration** *(\[New Cases/Day\]/Day)*/the change in New Cases reported per day.  Might add more country plots to this notebook later but feel free to fork it and tailor it towards your own region.  NB Tried to drop cruise ship passengers from the list (exclude any Province/State with 'Princess' in the name).
# 
# * [Net Cases curves](#net-cases) (i.e. Reported-(Fatalities+Recovered)).
# 
# * [Correlation heatmaps](#corrMaps) to show the similarity between each country's case dynamics.
# 
# * [Singular Spectrum Analysis](#SSA)- Decomposes a curve into a bunch of constituent curves through SVD magic and other things that went over my head as explained in this [Kaggle notebook by jdarcy](https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition).  So far just one SSA for [Ontario, Canada](#SSA-Ontario)
# 
# ToDo:
# 
# * Make it look better
# * More Dimensionality reduction would be cool.  Still looking into whether PCA will actually yield anything meaningful.  These should be stationary right?
# * Some sort of timeseries predictive modelling if this goes on long enough.  I hope not though.
#  
# 
# Good luck and stay safe.
# 

# In[ ]:


get_ipython().system('pip install pyts')
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 
#NOTEBOOK WHILE KERNEL IS RUNNING
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import cufflinks as cf
cf.go_offline()
import os
import warnings
from pyts.decomposition import SingularSpectrumAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


# <a id="section-one"></a>
# # Virus Dynamics
# When the derivatives (speed and acceleration) start getting smaller or negative, that's a good thing...except in Recovered Cases.
# 

# ## Dataset Initialization
# *generateMultiregionTotal*: This gets applied to a Pandas GroupBy object in order to generate country-wide totals for countries with multiple regions in the dataset

# In[ ]:


def generateMultiregionTotal(groupDf):
    if(len(groupDf.index)>1):
        countryTotal=groupDf.sum(numeric_only=True).astype(np.int64)
        countryTotal.name="{}".format(groupDf.name)
        countryTotal["Country/Region"]=groupDf.name
        countryTotal["Province/State"]=groupDf.name
        pseudoDf=pd.DataFrame(countryTotal).T

        pseudoDf=pseudoDf.set_index(['Country/Region',pd.Index([countryTotal.name])])
        groupDf=pd.concat([groupDf,pseudoDf])
#     print(groupDf)
    return groupDf


# In[ ]:


def graphLegendFormatting(trace):
    tokens=(trace.name).split(",")
    region=tokens[-1].replace(")","").replace("'","")
    if(len(tokens)>2):
        quantity=tokens[0].replace("(","").replace("'","")
        trace.name="({}.) {}".format(quantity[:3],region)
    else:
        trace.name=region
    return trace


# *generateKinetics*: Restructures data and calculates derivatives

# In[ ]:


def generateKinetics(filename='/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv'):
    df_raw=pd.read_csv(filename)
    df_raw=df_raw.set_index([
        df_raw["Country/Region"],
        df_raw["Province/State"].fillna(df_raw["Country/Region"])
                                               ],drop=True)
    countries=df_raw.groupby(level=0,group_keys=False)
    df_raw=countries.apply(generateMultiregionTotal)
    df_raw=df_raw.drop(['Country/Region', 'Lat', 'Long','Province/State'],axis=1)
    df_raw.columns=pd.to_datetime(df_raw.columns)
    df_raw=df_raw.reindex(sorted(df_raw.columns), axis=1)
    df_position=df_raw[~df_raw.index.get_level_values("Province/State").str.contains("|".join(["Princess","Recovered"]),na=False)]
    df_speed = df_position.diff(axis=1).fillna(0)
    df_accel = df_speed.diff(axis=1).fillna(0)
    kineticDf=pd.concat({"Position":df_position,"Speed":df_speed,"Acceleration":df_accel})
    return kineticDf


# In[ ]:


def generatePlot(masterDf,countriesOfInterest="Canada",regionsOfInterest=[],dataToPlot={"metric":"Reported Cases","quantity":"Position"}):
    if(type(countriesOfInterest)==list):
        countryLevelStats=pd.concat([masterDf.groupby("Country/Region").get_group(country) for country in countriesOfInterest])
    else:
        countryLevelStats=masterDf.groupby("Country/Region").get_group(countriesOfInterest)
    #Format chart labels depending on how many countries requested
    if(type(countriesOfInterest)==list):        
        formatList=['{}' for country in countriesOfInterest]
        countryLabels=", ".join(formatList)
        countryLabels=countryLabels.format(*countriesOfInterest)
    else:
        countryLabels=countriesOfInterest
    #Determine which data to plot
#     print(countryLevelStats.index)
    if(dataToPlot["metric"]==None and dataToPlot["quantity"]==None):
        plotDf=countryLevelStats.loc[:,:]
        title="All Available Data in {}".format(countryLabels)
    elif(dataToPlot["metric"]!=None and dataToPlot["quantity"]==None):
        plotDf=countryLevelStats.loc[dataToPlot["metric"]]
        title="All Available {} Data in {}".format(dataToPlot["metric"],countryLabels)
    else:
        plotDf=countryLevelStats.loc[dataToPlot["metric"],dataToPlot["quantity"]]
        title="{} of {} in {}".format(dataToPlot["quantity"],dataToPlot["metric"],countryLabels)  
#     if(len(regionsOfInterest)>0):
#         print(plotDf.index)
#         plotDf.query("Province/State=={}".format(regionsOfInterest))
#         print(plotDf)
    plot=plotDf.T.iplot(asFigure=True,layout=cf.Layout(height=700,title=title,xaxis=dict(title="Date"),yaxis=dict(title="Cases"),legend=dict(font=dict(size=10),orientation="h",xanchor='center',x=0.5,y=-0.3)))
    plot.for_each_trace(fn=graphLegendFormatting)
    return plot


# In[ ]:


def generateCorrMatrix(masterDf,dataToPlot={"metric":"Reported Cases","quantity":"Position"},plotDim=625):
    correlationMatrix=masterDf.loc[dataToPlot["metric"],dataToPlot["quantity"]].droplevel(0).T.astype(int).corr().dropna(axis=1,how="all").dropna(axis=0,how="all")
    corrPlot=correlationMatrix.iplot(asFigure=True,kind="heatmap",colorscale="RdBu",layout=cf.Layout(title="Correlation Matrix: {} of {}".format(dataToPlot["quantity"],dataToPlot["metric"]),height=plotDim,width=plotDim))
    return corrPlot   


# Store the CSV files into a Pandas dataframe.

# In[ ]:


reportedKinetics = generateKinetics()
fatalitiesKinetics = generateKinetics(filename='/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
recoveryKinetics = generateKinetics(filename='/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
masterDf = pd.concat({"Reported Cases":reportedKinetics,"Fatalities":fatalitiesKinetics,"Recovered Cases":recoveryKinetics})


# <a id="dynamics-graphs"></a>
# # Dynamics Graphs

# Comparison of Position between Canada and Sweden (i.e. # of Reported Cases)

# In[ ]:


countriesOfInterest=["Canada","Sweden"]
parameters = {"metric":"Reported Cases","quantity":"Position"}
casesPosition = generatePlot(masterDf,countriesOfInterest=countriesOfInterest,dataToPlot=parameters)
iplot(casesPosition)


# In[ ]:


countriesOfInterest=["Canada","Australia"]
parameters = {"metric":"Reported Cases","quantity":"Position"}
casesPosition = generatePlot(masterDf,countriesOfInterest=countriesOfInterest,dataToPlot=parameters)
iplot(casesPosition)


# Speed of New Cases in Canada (# of New Cases Reported/Day).

# In[ ]:


countriesOfInterest=["Sweden","Canada"]
parameters = {"metric":"Reported Cases","quantity":"Speed"}
casesSpeed = generatePlot(masterDf,countriesOfInterest=countriesOfInterest,dataToPlot=parameters)
iplot(casesSpeed)


# Comparison of Acceleration of Cases between Canada and Sweden. (Change in # of New Cases Reported/Day).

# In[ ]:


countriesOfInterest=["Sweden","Canada"]
parameters = {"metric":"Reported Cases","quantity":"Acceleration"}
casesAccel = generatePlot(masterDf,countriesOfInterest=countriesOfInterest,dataToPlot=parameters)
iplot(casesAccel)


# In[ ]:


countriesOfInterest=["Sweden","Canada"]
parameters = {"metric":"Reported Cases","quantity":None}
casesAllData = generatePlot(masterDf,countriesOfInterest=countriesOfInterest,dataToPlot=parameters)
iplot(casesAllData)


# In[ ]:


countriesOfInterest=["Sweden","Canada"]
parameters = {"metric":"Fatalities","quantity":None}
fatalitiesAllData = generatePlot(masterDf,countriesOfInterest=countriesOfInterest,dataToPlot=parameters)
iplot(fatalitiesAllData)


# In[ ]:


countriesOfInterest=["Sweden","Canada"]
parameters = {"metric":"Recovered Cases","quantity":None}
fatalitiesAllData = generatePlot(masterDf,countriesOfInterest=countriesOfInterest,dataToPlot=parameters)
iplot(fatalitiesAllData)


# <a id="net-cases"></a>
# Net Cases (Reported - (Fatalities+Recovered))

# In[ ]:


countriesOfInterest=["Canada","US"]
parameters = {"metric":"Recovered Cases","quantity":"Position"}
netDf=masterDf.loc["Reported Cases",parameters["quantity"]].subtract(masterDf.loc["Fatalities",parameters["quantity"]].add(masterDf.loc["Recovered Cases",parameters["quantity"]]))
plotDf=netDf.loc[countriesOfInterest].dropna()

if(type(countriesOfInterest)==list):        
    formatList=['{}' for country in countriesOfInterest]
    countryLabels=", ".join(formatList)
    countryLabels=countryLabels.format(*countriesOfInterest)
    title="Net Cases in {}".format(countryLabels)
else:
    countryLabels=countriesOfInterest
    title="Net Cases in {}".format(countryLabels)
    
netPlot=plotDf.T.iplot(asFigure=True, xTitle="Date",yTitle="Cases",title=title)
netPlot.for_each_trace(fn=graphLegendFormatting)
iplot(netPlot)


# <a id="corrMaps"></a>
# # Correlation Heatmaps
# You'll need to zoom in for a detailed view but these basically shows the similarity between each country's curves, measured between negative or 0 (Not similar at all, Red) to 1 (Very Similar, Blue)  Notice the bands/distinction clustered around the Chinese provinces/regions.
# 

# In[ ]:


parameters={"metric":"Reported Cases","quantity":"Position"}
reportedPosition_corrPlot=generateCorrMatrix(masterDf,dataToPlot=parameters,plotDim=625)
iplot(reportedPosition_corrPlot)


# In[ ]:


parameters={"metric":"Reported Cases","quantity":"Speed"}
reportedSpeed_corrPlot=generateCorrMatrix(masterDf,dataToPlot=parameters,plotDim=625)
iplot(reportedSpeed_corrPlot)


# In[ ]:


parameters={"metric":"Reported Cases","quantity":"Acceleration"}
reportedAccel_corrPlot=generateCorrMatrix(masterDf,dataToPlot=parameters,plotDim=625)
iplot(reportedAccel_corrPlot)


# In[ ]:


parameters={"metric":"Fatalities","quantity":"Position"}
fatalitiesAccel_corrPlot=generateCorrMatrix(masterDf,dataToPlot=parameters,plotDim=625)
iplot(fatalitiesAccel_corrPlot)


# In[ ]:


parameters={"metric":"Recovered Cases","quantity":"Position"}
recoveredPosition_corrPlot=generateCorrMatrix(masterDf,dataToPlot=parameters,plotDim=625)
iplot(recoveredPosition_corrPlot)


# <a id="SSA"></a>
# # Singular Spectrum Analysis
# 
# Singular Spectrum Analysis is a decomposition technique that breaks a signal into 3 groups of parts: trend, periodicity and noise which allows for them to be inspected individually. 
# * *Trend* takes care of the function's overall increase/decrease, 
# * *Periodicity* group contains sinusoidal-ish functions which account for any repeating patterns in the waveform 
# * *Noise* accounts for minor variations away from the trend pattern that would otherwise be imposed by the Trend and Periodicity group waveforms.

# <a id="SSA-Ontario"></a>
# SSA for Reported Cases curve in Ontario, Canada.
# Component curves are the constituent *trend*, *period* and *noise* groups waveforms.
# * Component 0: Trend
# * Components 1-5 (arguable): Periodicity
# * Component 6: Noise
# 
# I'm not really sure what to do with these, just wanted to see how it works.
# 

# In[ ]:


def plotSSA(masterDf,parameters={"metric":"Position","quantity":"Reported Cases"},countryOfInterest=["Canada"],regionOfInterest=None,windowSize=7,n_components=None):
    if(regionOfInterest==None):
        regionOfInterest=countryOfInterest
    if(n_components==None):
        n_components=windowSize
        
    regionDf=masterDf.loc[parameters["quantity"],parameters["metric"],countryOfInterest,regionOfInterest]
    ssa=SingularSpectrumAnalysis(window_size=windowSize)
    decomposedWaveforms=ssa.fit_transform(regionDf)
    compDf=pd.DataFrame(decomposedWaveforms)
    compDf.columns=regionDf.columns
    compDf=compDf.rename(index=lambda s: "Component {}".format(s))
    reconstructedSeries=compDf.iloc[:n_components,:].sum(axis=0)
    reconstructedSeries.name="Reconstructed from SSA"
    regionDf=regionDf.append(reconstructedSeries)
    regionDf=regionDf.append(compDf)
    title="Singular Spectrum Analysis for {} of {} in {}".format(parameters["metric"],parameters["quantity"],regionOfInterest[0])
    aio_plot=regionDf.T.iplot(asFigure=True, xTitle="Date",yTitle="Component Magnitude",title=title)
    return aio_plot


# In[ ]:


countryOfInterest=regionOfInterest=["Canada"]
parameters={"metric":"Speed","quantity":"Reported Cases"}
aio_plot=plotSSA(masterDf,parameters=parameters,countryOfInterest=countryOfInterest)
iplot(aio_plot)

