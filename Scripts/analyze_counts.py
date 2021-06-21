import pandas as pd
modulesDF = pd.read_csv('kernels_competitions.csv', index_col=False, header=0)
modulesDF["values"].value_counts().to_csv("values_count_competitions.csv")