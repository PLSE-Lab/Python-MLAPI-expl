import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from collections import Counter
import squarify

# import file data into a DataFrame
df_ParkingCitations = DataFrame(pd.read_csv("../input/parking-citations.csv", nrows=1312199, low_memory=False))
df_ParkingCitations['Issue Date'] = pd.to_datetime(df_ParkingCitations['Issue Date'])
df_ParkingCitations['Year'], df_ParkingCitations['Month'] = df_ParkingCitations['Issue Date'].dt.year, \
                                                            df_ParkingCitations['Issue Date'].dt.strftime('%B')

# Plot the count of the tickets each year.
year_Count = Counter(df_ParkingCitations['Year'])
sns.barplot(list(year_Count.keys()), list(year_Count.values()))
plt.xlabel('Year')
plt.ylabel('# of Parking Tickets issued')
plt.title('# of parking tickets issued')
plt.show()
plt.gcf().clear()

# Plot the count for each month in 2016
month_Count = Counter(df_ParkingCitations.loc[df_ParkingCitations["Year"] == 2016]["Month"])
sns.barplot(list(month_Count.values()), list(month_Count.keys()))
# squarify.plot(list(month_Count.values()), label=list(month_Count.keys()))
plt.xlabel("Count of the parking tickets")
plt.ylabel("Months")
plt.title("Count of tickets issued each month in 2016")
plt.show()
plt.gcf().clear()

# Plot the count of all state plates in 2016
plate_count = Counter(df_ParkingCitations.loc[df_ParkingCitations["Year"] == 2016]["RP State Plate"])
plt.pie(plate_count.values(), labels=plate_count.keys(), autopct='%1.2f%%')
fontP = FontProperties()
fontP.set_size('small')
plt.legend(ncol=8, prop=fontP)
plt.title("State plates on which the ticket was issued")
plt.show()
plt.gcf().clear()

# Plot the count of the Other State Plates in 2016
other_plate_count = Counter(df_ParkingCitations.loc[(df_ParkingCitations["Year"] == 2016) &
                                                    (df_ParkingCitations["RP State Plate"] != 'CA')]["RP State Plate"])
sns.barplot(list(other_plate_count.keys()), list(other_plate_count.values()), color="crimson")
plt.xticks(rotation=90)
plt.title("Parking Citations for State Plates other than CA")
plt.xlabel("States")
plt.ylabel("Count")
plt.show()
plt.gcf().clear()

# Plot the types of tickets in 2016
violations = Counter(df_ParkingCitations.loc[df_ParkingCitations["Year"] == 2016]["Violation Description"])
cmap = matplotlib.cm.coolwarm
minCount = min(violations.values())
maxCount = max(violations.values())
norm = matplotlib.colors.Normalize(vmin=minCount, vmax=maxCount)
colors = [cmap(norm(value)) for value in violations.values()]
squarify.plot(violations.values(), label=violations.keys(), value=violations.values(), color=colors)
plt.title("Parking Violations")
plt.show()
plt.gcf().clear()

# Plot Tickets Issued to Non-CA State Plates in 2016
df_violations_non_ca = df_ParkingCitations.loc[(df_ParkingCitations["Year"] == 2016)
                                               & (df_ParkingCitations["RP State Plate"] != "CA")]
violation_non_ca = Counter(df_violations_non_ca["Violation Description"])
violation_non_ca = {k: v for k, v in violation_non_ca.items() if v > 1000}
# print(violation_non_ca)
sns.barplot(list(violation_non_ca.keys()), list(violation_non_ca.values()))
plt.title("Violations by Non-CA State Plates")
plt.xlabel("Violations")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
plt.gcf().clear()

# Plot Fine Amount for violations by NON-CA Plates in 2016
df_violations_non_ca = df_violations_non_ca.loc[df_violations_non_ca["Violation Description"].isin(violation_non_ca.keys())]
fineCollectedNonCA = df_violations_non_ca.groupby(["Violation Description"])["Fine amount"].agg('sum')
# print(fineCollectedNonCA)
fineCollectedNonCA.plot.bar()
plt.xticks(rotation=45)
plt.xlabel("Violations")
plt.ylabel("Fine Amount")
plt.title("Fine amount for violations by NON-CA Plates")
plt.show()
plt.gcf().clear()

# Plot Tickets Issued to CA State Plates in 2016
df_violations_ca = df_ParkingCitations.loc[(df_ParkingCitations["Year"] == 2016)
                                           & (df_ParkingCitations["RP State Plate"] == "CA")]
violation_ca = Counter(df_violations_ca["Violation Description"])
# print(violation_ca)
violation_ca = {k: v for k, v in violation_ca.items() if v > 5000}
sns.barplot(list(violation_ca.keys()), list(violation_ca.values()))
plt.title("Violations by CA state plates")
plt.xlabel("Violations")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()
plt.gcf().clear()

# Plot Fine Amount for violations by CA Plates in 2016
df_violations_ca = df_violations_ca.loc[df_violations_ca["Violation Description"].isin(violation_ca.keys())]
fineCollectedCA = df_violations_ca.groupby(["Violation Description"])["Fine amount"].agg('sum')
# print(fineCollectedCA)
fineCollectedCA.plot.bar()
plt.xticks(rotation=45)
plt.ylim([300000, 25000000])
plt.xlabel("Violations")
plt.ylabel("Fine Collected")
plt.title("Fine collected for violations by CA Plates")
plt.show()
plt.gcf().clear()

df_no_parking = df_ParkingCitations.loc[(df_ParkingCitations["Year"] == 2016)
                                        & (df_ParkingCitations["Violation Description"] == "NO PARK/STREET CLEAN")]
df_no_parking_location = Counter(df_no_parking["Location"])
df_no_parking_location = {k: v for k, v in df_no_parking_location.items() if v > 50}
squarify.plot(df_no_parking_location.values(), label=df_no_parking_location.keys(), value=df_no_parking_location.values())
plt.show()
print(df_no_parking_location)
