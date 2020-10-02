# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use("fivethirtyeight")

# Reading file
df = pd.read_csv("../input/cities_r2.csv")

# Convert Dataframe into list
def to_list(x):
    lst = []
    for line in df[x]:
        lst.append(line)
    return lst

# Add up
pop_male = sum(to_list("population_male"))
pop_female = sum(to_list("population_female"))

# Graphing
plt.pie([pop_male,pop_female],labels=["Male","Female"],startangle=90,autopct="%1.1f%%",colors=["red","green"])
plt.title("Share Male/Female")
plt.show()