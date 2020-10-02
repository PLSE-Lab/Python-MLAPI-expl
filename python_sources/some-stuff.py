import numpy as np
import pandas as pd

from geopy.distance import vincenty
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

plt.show()

newport_ri = (41.49008, -71.312796)
cleveland_oh = (41.499498, -81.695391)
print("Distances...\n")
print(vincenty(newport_ri, cleveland_oh).miles)
print("ok..\n")

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print ("hello world...\n")
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)