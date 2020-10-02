import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train_data = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
survived_passengrers_df=train_data[train_data['Survived']==1]
train=survived_passengrers_df[['Survived','Sex']]

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)