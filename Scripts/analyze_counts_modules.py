import pandas as pd
modulesDF = pd.read_csv('results_modules.txt', index_col=False, header=0)
modulesDF["values"].value_counts().to_csv("values_count_modules.csv")