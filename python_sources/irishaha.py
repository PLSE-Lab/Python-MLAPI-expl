import pandas as pd

import warnings

warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

iris = pd.read_csv("../input/Iris.csv") # the iris dataset is now a Pandas DataFrame

iris.head()
