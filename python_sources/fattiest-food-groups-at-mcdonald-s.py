import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#'Declare empty lists used later
percentFat = []
category = []
zero = 0
ten = 0
twenty = 0
thirty = 0
forty = 0
fifty = 0
zero_groups = []
ten_groups = []
twenty_groups = []
thirty_groups = []
fourty_groups = []
fifty_groups = []
allcats = []

breakfast = []
beefAndPork = []
ChickenAndFish = []
Salads = []
SnacksAndSides = []
Desserts = []
Beverages = []
CoffeeAndTea = []
SmoothiesAndShakes = []



#import csv
menu = pd.read_csv('menu.csv')
calories = menu['Calories']
caloriesFromFat = menu['Calories from Fat']
tempcats = menu['Category']
tempitems = menu['Item']
fattyitems = []
fattycats = []

        
#calculate percentage fat values
for i in range(0, len(menu)):
    if int(caloriesFromFat[i]) != 0 and int(calories[i]) != 0:
        temp = np.round((int(caloriesFromFat[i]) / int(calories[i])*100), decimals = 1)
        if temp >= 0 and temp < 10:
             zero= zero + 1
             zero_groups.append(tempcats[i])
        elif temp >= 10 and temp < 20:
            ten = ten + 1
            ten_groups.append(tempcats[i])
        elif temp >= 20 and temp < 30:
            twenty = twenty + 1
            twenty_groups.append(tempcats[i])
        elif temp >= 30 and temp < 40:
            thirty = thirty + 1
            thirty_groups.append(tempcats[i])
        elif temp >= 40 and temp < 50:
            forty = forty + 1
            fourty_groups.append(tempcats[i])
        elif temp >= 50 and temp < 60:
            fifty = fifty + 1
            fifty_groups.append(tempcats[i])
            fattyitems.append(tempitems[i])
            if tempcats[i] == 'Breakfast':
                breakfast.append(tempitems[i])
            if tempcats[i] == 'Beef & Pork':
                beefAndPork.append(tempitems[i])
            if tempcats[i] == 'Chicken & Fish':
                ChickenAndFish.append(tempitems[i])
            if tempcats[i] == 'Salads':
                Salads.append(tempitems[i])
            if tempcats[i] == 'Snacks & Sides':
                SnacksAndSides.append(tempitems[i])
            if tempcats[i] == 'Desserts':
                Desserts.append(tempitems[i])
            if tempcats[i] == 'Beverages':
                Beverages.append(tempitems[i])
            if tempcats[i] == 'Coffee & Tea':
                CoffeeAndTea.append(tempitems[i])
            if tempcats[i] == 'Smoothies & Shakes':
                SmoothiesAndShakes.append(tempitems[i])
        percentFat.append(temp)
#sorts values in increasing order        
percentFat = (sorted(percentFat,key=int))
tempcats = menu['Category']

for item in tempcats:
    if item not in allcats:
        allcats.append(item)
        
for i in range(0,len(menu)):
    if int(caloriesFromFat[i]) != 0 and int(calories[i]) != 0:
        category.append(tempcats[i])
#    elif int(caloriesFromFat[i]) == 0 or int(calories[i]) == 0:
#        print("Found a zero at: ", int(i))
category = np.arange(len(category))
np.set_printoptions(precision=1)

#Charts
labels = 'Breakfast','Beef & Pork','Chicken & Fish','Salads','Snacks & Sides','Coffee & Tea'
sizes = [len(breakfast),len(beefAndPork), len(ChickenAndFish), len(Salads), len(SnacksAndSides), len(CoffeeAndTea)]
explode = [0,0,0,0,0,0,0,0,]
fig1, ax1 = plt.subplots()
ax1.pie(sizes,labels=labels,startangle=90,shadow=True,autopct='%1.1f%%')
ax1.axis('equal')
plt.title('Percentage of Foods Where At Least 50% of Calories Come from Fat')
plt.show()






               
