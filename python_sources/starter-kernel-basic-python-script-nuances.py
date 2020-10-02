
####################################################################
# Load modules
import pandas as pd # Data manipulation
import matplotlib.pyplot as plt # Data visualization
import seaborn as sns

####################################################################
# Read in the data
# Data files are located at "../input/"
# Click on the "Input" tab in the pane to the right to inspect the data files.
recipes = pd.read_csv('../input/epi_r.csv')

####################################################################
# Explore the data
# How are the ratings distributed?
sns.distplot(recipes["rating"])
plt.savefig('rating_distribution.png')
# Note: You can find anything you save in the "Output" tab of the kernel.

####################################################################
# Start your analysis here!
# Click on "Run" to execute your code. You can run your kernel as many times as you like as you add and tweak code.
# Some ideas:

# 1) Knowing the most common ingredients in the dataset could help you keep you fridge stocked. What are they?
# 2) Which ingredients are most likely to be in highly rated recipes?



####################################################################