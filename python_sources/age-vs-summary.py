
import pandas as pd #dataframes
import numpy as np  #scientific
import matplotlib.pyplot as plt #graphs

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)
subplots_fig = plt.figure() 
ax1 = subplots_fig.add_subplot(121) 
y=train.Age.fillna(0)
ax1.hist(y) 
ax1.set_title("Histogram of Titanic Age Distribution") 
ax1.set_ylabel('Count of Passengers',fontsize=10) 
ax1.set_xlabel('Age of Distribution',fontsize=10)