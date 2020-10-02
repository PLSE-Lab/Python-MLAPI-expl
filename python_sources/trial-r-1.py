import os

for file in os.listdir("../input/"):
    print(file)

import pandas as pd
df=pd.read_csv('../input/train.csv')

df.head()