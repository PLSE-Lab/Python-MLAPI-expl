import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#data processing: filling in missing datas with median and removing unwanted columns
train.dtypes
#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)