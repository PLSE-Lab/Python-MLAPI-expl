# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#Pull in CSV file and subset Latin American countries of interest
ue = pd.read_csv("../input/API_ILO_country_YU.csv")

countries = ['Chile','Argentina','Bolivia','Ecuador','Paraguay']
years = [col for col in ue.columns if col.isdigit()]

  #Create reshaped df based on original 'ue' df
reshaped_ue = pd.DataFrame(columns=['Country','Year','UE'])
reshaped_ue['Country'] = np.repeat(ue['Country Name'],len(years))

reshaped_ue['Year'] = np.reshape(list(list(map(int,years)) for i in \
range(len(ue['Country Name']))),len(years)*len(ue['Country Name']))

reshaped_ue.index = range(len(reshaped_ue['Country']))
for x in range(len(ue['Country Name'])):
    for year in years:
        add_index = int(year)-2010
        reshaped_ue.set_value(x*5+add_index,'UE',round(ue[year][x],2))
        
# Subset the new df to only countries of interest (LatAm)
reshaped_ue.latam = [x in countries for x in reshaped_ue.Country]
latam_ue = reshaped_ue.loc[(reshaped_ue.latam),['Country','Year','UE']]
latam_ue.UE = latam_ue.UE.astype(np.float32)
print(latam_ue)

latam_table = pd.pivot_table(data=latam_ue,
                             index=['Year'],
                             columns=['Country'],
                             values=['UE'],
                             aggfunc='mean')
print(latam_table)
tab = sns.heatmap(data=latam_table['UE'],vmin=0,annot=True,fmt='2.2f')
plt.title('Latin American Unemployment')
ticks = plt.setp(tab.get_xticklabels(),rotation=45)

plt.show()






