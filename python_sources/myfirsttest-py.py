import pandas as pd
import numpy as np
import csv as csv

import matplotlib as mat
import matplotlib.pyplot as plt
import seaborn as sns
import time

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())
sns.plt.show()

sns.set(font_scale=1)

pd.options.display.mpl_style = 'default'
#mat.pyplot.style.use='default'
train.hist(bins=10,figsize=(9,7),grid=False)

g = sns.FacetGrid(train, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",color="purple")

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)