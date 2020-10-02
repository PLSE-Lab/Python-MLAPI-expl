import pandas as pd
import numpy as np


competitions = pd.read_csv('../input/Competitions.csv',)

# ge competiions
competitions = competitions[competitions['CompetitionHostSegmentId'] == 7]

print(competitions)