import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
training = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(training.head())

print("\n\nSummary statistics of training data")
print(training.describe())

#Any files you save will be available in the output tab below
training.to_csv('copy_of_the_training_data.csv', index=False)