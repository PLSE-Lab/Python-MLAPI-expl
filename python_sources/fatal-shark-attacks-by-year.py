import pandas as pd
import re
import numpy as np
import matplotlib.pylab as pylab
pylab.style.use('ggplot')

GSAF = "../input/attacks.csv"

AllData = pd.read_csv(GSAF, encoding = 'ISO-8859-1')
AllData['Date'] = AllData['Date'].astype(str)


def find_year(date): #This function tries to extract the year from the dates column
    try:
        matches = [int(y) for y in list(re.findall(r'.*([1-3][0-9]{3})', date))]
        return int(np.mean(matches)) #Some date values containa  range of years
    except:
        return 0

AllData['Year'] = AllData['Date'].apply(find_year)

Fatal_Attacks = AllData[AllData['Fatal (Y/N)'] == 'Y']

startyear, endyear = 1960, 2017
Fatal_by_year = Fatal_Attacks['Year'].value_counts().sort_index(ascending = True).ix[startyear:endyear]#Fatal Attacks by year
Fatal_by_year.plot(kind = 'bar', title = "Fatal shark attacks by year")
pylab.savefig("output.png")