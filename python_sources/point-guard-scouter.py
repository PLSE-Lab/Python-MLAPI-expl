import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math 

from subprocess import check_output

goodCut = 0.05

raw_data = pd.read_csv('../input/players_stats.csv')
stats = pd.DataFrame(raw_data)
driveRatings = pd.DataFrame(raw_data, columns = ["Name","Age","FG%","OREB","BMI"])

index = driveRatings.set_index("Name")

hollinger = []
 
x = 1

for x in range(len(driveRatings["Name"])):
        
    tempStore = (stats.at[x,'FG%'])
    hollinger.append(int(tempStore))

print(hollinger)