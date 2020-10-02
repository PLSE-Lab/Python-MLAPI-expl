# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

indicators = pd.read_csv("../input/Indicators.csv")
country = pd.read_csv("../input/Country.csv")
country_notes = pd.read_csv("../input/CountryNotes.csv")

india_co2 = indicators[(indicators.CountryName=='India')&(indicators.IndicatorCode=='EN.CO2.TRAN.ZS')]
phil_co2  = indicators[(indicators.CountryName=='Philippines')&(indicators.IndicatorCode=='EN.CO2.TRAN.ZS')]
china_co2  = indicators[(indicators.CountryName=='China')&(indicators.IndicatorCode=='EN.CO2.TRAN.ZS')]
fig = plt.figure()

plt.plot(india_co2.Year,india_co2.Value,'o-',label='India',color="red")
plt.plot(phil_co2.Year,phil_co2.Value,'o-',label='Philippines',color="green")
plt.plot(china_co2.Year,china_co2.Value,'o-',label='China',color="blue")
plt.legend(bbox_to_anchor=(0.85, 1), loc=2, borderaxespad=0.)
plt.xlabel('Years',  fontsize=12)
plt.ylabel('(% of total fuel combustion)',  fontsize=12)
plt.title('CO2 Emissions from Transport by Country', fontsize=14)

fig.savefig('CO2_emissions.png',format='png', dpi=50)

in_bldg  = indicators[(indicators.CountryName=='India')&(indicators.IndicatorCode=='EN.CO2.BLDG.ZS')]
in_tran  = indicators[(indicators.CountryName=='India')&(indicators.IndicatorCode=='EN.CO2.TRAN.ZS')]
in_etot  = indicators[(indicators.CountryName=='India')&(indicators.IndicatorCode=='EN.CO2.ETOT.ZS')]
in_manf  = indicators[(indicators.CountryName=='India')&(indicators.IndicatorCode=='EN.CO2.MANF.ZS')]
in_othx  = indicators[(indicators.CountryName=='India')&(indicators.IndicatorCode=='EN.CO2.OTHX.ZS')]

fig = plt.figure()
plt.plot(in_tran.Year,in_tran.Value,'o-',label='Transport',color="red")
plt.plot(in_othx.Year,in_othx.Value,'o-',label='Other Sectors',color="green")
plt.plot(in_manf.Year,in_manf.Value,'o-',label='Manufacturing',color="blue")
plt.plot(in_etot.Year,in_etot.Value,'o-',label='Electricity and Heat Production',color="yellow")
plt.plot(in_bldg.Year,in_bldg.Value,'o-',label='Residential Buildings and Commercial and Public Services',color="orange")
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.xlabel('Years',  fontsize=14)
plt.ylabel('(% of total fuel combustion)',  fontsize=14)
plt.title('CO2 Emissions from India', fontsize=14)

fig.savefig('CO2_emissions_from_India.png',format='png', dpi=400)