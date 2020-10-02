# As always, first some libraries
# Numpy handles matrices
import numpy as np
# Pandas handles data
import pandas as pd
# Matplotlib is a plotting library
import matplotlib.pyplot as plt
#Set matplotlib to render immediately
#%matplotlib inline
# Seaborn is a plotting library built on top of matplotlib that can handle some more advanced plotting
import seaborn as sns

# Define colors for seaborn
five_thirty_eight = [
    "#30a2da",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b",
]
# Tell seaborn to use the 538 colors
sns.set(palette=five_thirty_eight)

# Load data with pandas
df = pd.read_csv('balanced_bank.csv',index_col=0)

# Display first five rows for a rough overview
df.head()

# Count missing values per column
df.isnull().sum()