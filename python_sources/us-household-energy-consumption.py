import csv
import matplotlib.pyplot as plot
import numpy as np

all_us_stats = []
location = r'../input/all_energy_statistics.csv'
with open(location,'r', encoding='utf8')as f:
    reader = csv.reader(f)
    for line in reader:
        if(line[0] == 'United States'):
            all_us_stats.append(line)

household_energy_consumption_yearly = []
household_energy_consumption_amount = []
for place, type, year, measure, amount, empty, total in all_us_stats:
    if(type == "Electricity - Consumption by households"):
        household_energy_consumption_yearly.append(int(year))
        household_energy_consumption_amount.append(int(amount))
x = list(household_energy_consumption_yearly)
y = list(household_energy_consumption_amount)
xy = list(zip(x,y))
x_sorted = []
y_sorted = []
for year, amount in sorted(xy):
    x_sorted.append(year)
    y_sorted.append(amount)
plot.title("United States Household Energy Consumption")
plot.ylabel("Kilowatts")
plot.bar(x_sorted, y_sorted)
plot.show()