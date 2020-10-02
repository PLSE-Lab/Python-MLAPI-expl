"""
IDENTIFY AND CHART THE 30 NATIONS WITH POPULATION OVER ONE MILLION WHOSE GDPs/PerCapita
GREW THE MOST ON AVERAGE IN THE THREE DECADES THAT ELAPSED FROM 1980 TO 2010
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


df1=pd.read_csv('Country.csv',index_col='wbid')
df2=df1[['country','SES','year','gdppc']]
df3=df2.loc[df2['year'].isin([1980,1990,2000,2010])]
df4=df3.sort_values(['country','year'])

IncreasingGDPNationsData=[]
marker=0
while marker<len(df4):
    c=df4[marker:marker+4]
    j=0
    incr=True
    while j<3:
        if float(c[j:j+1]['gdppc'])>=float(c[j+1:j+2]['gdppc']):
            incr=False
        j+=1
    if incr: #change form of data to list of lists
        four_years_gdps=list(c.get_value(c.index[0],'gdppc'))
        IncreasingGDPNationsData.append([c.index[0],four_years_gdps])
    marker+=4

nationsList1=[]  
for elem in IncreasingGDPNationsData:
    nationData=[elem[0]]
    nationData.append([int(gdpnum) for gdpnum in elem[1]])
    spot=list(df1.index).index(elem[0])
    name=list(df1['country'])[spot]
    nationData.append(name)
    nationsList1.append(nationData)
"""
#for item in nationsList1:
#    print(item)
***************************************
nat gdp1980 gdp1990 gdp2000 gdp2010
['ALB', [2347, 4557, 5470, 9927]]
['ARG', [8206, 10833, 14924, 18794]]
['AUS', [14412, 28572, 35244, 41363]]
['AUT', [13759, 31265, 38812, 43174]]
['BGD', [549, 1290, 1645, 2451]]...
*************************************
"""

nationsList2=[]
for d in nationsList1:
    nationData=d
    growth_rates_per_decade=["%.3f"%(d[1][i+1]/(d[1][i])) for i in range(3)]    
    ave_dec_growth="%.3f"%(sum([float(k) for k in growth_rates_per_decade])/3)
    nationData.append(growth_rates_per_decade)
    nationData.append(ave_dec_growth)
    nationsList2.append(nationData)    
"""
for item in nationsList2:
    print(item)
***************************************
nat gdp1980 gdp1990 gdp2000 gdp2010 gworth1 growth2 growth3   averageGrowth
['ALB', [2347, 4557, 5470, 9927], ['1.942', '1.200', '1.815'], '1.239']
['ARG', [8206, 10833, 14924, 18794], ['1.320', '1.378', '1.259'], '0.989']
['AUS', [14412, 28572, 35244, 41363], ['1.983', '1.234', '1.174'], '1.098']
['AUT', [13759, 31265, 38812, 43174], ['2.272', '1.241', '1.112'], '1.156']
['BGD', [549, 1290, 1645, 2451], ['2.350', '1.275', '1.490'], '1.279']...
***************************************
"""  

nationsList3=sorted(nationsList2,key=lambda x:x[-1],reverse=True)
"""
for item in nationsList3:
    print(item)
***************************************
nat gdp1980 gdp1990 gdp2000 gdp2010 gworth1 growth2 growth3   averageGrowth
['MAC', [2522, 34392, 37281, 98614], 'Macao', ['13.637', '1.084', '2.645'], '5.789']
['LUX', [10436, 54263, 78687, 89727], 'Luxembourg', ['5.200', '1.450', '1.140'], '2.597']
['BWA', [1765, 8099, 10361, 13119], 'Botswana', ['4.589', '1.279', '1.266'], '2.378']
['SGP', [9058, 34298, 51636, 72017], 'Singapore', ['3.786', '1.506', '1.395'], '2.229']
['CHN', [1061, 1526, 3700, 9525], 'China', ['1.438', '2.425', '2.574'], '2.146']...
***************************************
"""

nationsList4=[[d[2],"%.1f"%((float(d[4])-1)*100)] for d in nationsList3]
"""
for item in nationsList4:
    print(item)
***************************************
   nat      avg%Grwth/Decade
['Macao', '478.9']
['Luxembourg', '159.7']
['Botswana', '137.8']
['Singapore', '122.9']
['China', '114.6']...
***************************************
"""

#remove countries with population under one-million...
lowPops=['Macao','Luxembourg','Maldives','Iceland','Fiji']#...
nationsList5=[nation for nation in nationsList4 if nation[0] not in lowPops]
"""
for country in nationsList5:
    print(country)
***************************************   
['Botswana', '137.8']
['Singapore', '122.9']
['China', '114.6']
['South Korea', '104.1']
['Malaysia', '90.3']...
***************************************
"""

ThirtyFastestGrowingNationsSince1980=nationsList5[:30]
"""
for nat in ThirtyFastestGrowingNationsSince1980:
    print(nat)
***************************************
['Botswana', '137.8']
['Singapore', '122.9']
['China', '114.6']
['South Korea', '104.1']
['Malaysia', '90.3']
['Ireland', '85.1']
['Egypt', '83.3']
['Thailand', '82.1']
['Vietnam', '81.4']
['Norway', '75.6']
['Hong Kong', '74.8']
['Turkey', '74.0']
['Iran', '73.4']
['Indonesia', '71.6']
['Dominican Republic', '71.0']
['Bangladesh', '70.5']
['Sri Lanka', '68.7']
['India', '68.6']
['Hungary', '67.9']
['Pakistan', '66.8']
['Lao', '66.4']
['Albania', '65.2']
['Spain', '64.3']
['Tanzania', '62.6']
['Portugal', '61.5']
['Czech Republic', '58.5']
['Poland', '56.7']
['Switzerland', '56.7']
['Croatia', '56.3']
['Greece', '55.4']
***************************************
"""

nations=[n[0] for n in ThirtyFastestGrowingNationsSince1980]
percents=[float(n[1]) for n in ThirtyFastestGrowingNationsSince1980]

x_pos=np.arange(len(nations))
plt.bar(x_pos,percents,alpha=.8)
plt.xticks(x_pos,nations,rotation='vertical')
plt.ylabel('% Avg GDP/PerCap Growth Per Decade 1980-2010')
plt.title('30 Fastest Growing Nations since 1980')
fig_size=plt.rcParams["figure.figsize"]
fig_size[0],fig_size[1]=12,8
print(fig_size) #--> [6.0,4.0]
#plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.show()
