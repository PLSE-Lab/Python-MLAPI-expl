import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_path = "../input/attacks.csv"

# pull all the data from the csv to the dataframe
df = pd.read_csv(df_path, encoding = 'ISO-8859-1')

# Lets see all the available columns in the dataset
print(df.columns.values)


top10countries = df.Country.value_counts().head(10)
countries = top10countries.reset_index()
countries.columns = ['Location', 'Counts']
sns.barplot(countries.Location, countries.Counts, palette="Set3")
plt.xticks(rotation=90)
plt.show()