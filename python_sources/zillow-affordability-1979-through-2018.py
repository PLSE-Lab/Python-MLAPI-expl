import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy

#this file from https://www.zillow.com/research/data/ but has to be saved as UTF csv file format, in Excel or notepad
#to read in 
# uploaded to https://www.kaggle.com/janiscorona/zillow-19792018-affordability
data =pd.read_csv('../input/Affordability_Income_2018Q3.csv')

data.head(10)

headerNames = data.columns.values.tolist()
print(headerNames, 
      '\n\nNumber of Columns:', len(headerNames))

print("There are ", len(data['RegionID']), " observations in this dataframe.\n")

Top10 = data.head(11)
print(Top10)

Top10 = data.head(11)
print(Top10)
type(Top10)

Name = Top10['RegionName']
print(Name)
type(Name) #Dataframe

Last40Springs = Top10.loc[:,'1979-03':'2018-09':20]

Forty = pd.concat([Name, Last40Springs], axis=1)
print(Forty)
type(Forty)

Last = Last40Springs.T

print(Last)

x = Last.index

print(x)

USA = Last[0]
print(USA)
USA.plot()
plt.title('map')

plt.plot(Last[0],x)
plt.title('Spring 1979 to 2018 in Five Year Increments')

#plot all 11 regions and US as a line graph with the dates on the x-axis, and each y-axis the region value by date
plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow','black','grey','purple','orange','tan','silver','gold'])

plt.plot(x, Last[0])
plt.plot(x, Last[1])
plt.plot(x, Last[2])
plt.plot(x, Last[3])
plt.plot(x, Last[4])
plt.plot(x, Last[5])
plt.plot(x, Last[6])




plt.legend(['USA', 'NYC', 'LA', 'Chicago','Dallas','Philly','Houston'], loc='upper left')
plt.title('Six US Cities and USA Home Affordability Income March 1979-2018')
plt.show()

