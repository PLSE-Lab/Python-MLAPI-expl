import pandas as pd
import numpy as np
teams = pd.read_csv("../input/Teams.csv")
tourneySeeds = pd.read_csv("../input/TourneySeeds.csv")
tourneySlots = pd.read_csv("../input/TourneySlots.csv")

import os
files = os.listdir('../input/predictions')
# print (files)

files = files[:10]

for f in files:
    df = pd.read_csv(os.path.join("../input/predictions", f))
    print (df.head())

