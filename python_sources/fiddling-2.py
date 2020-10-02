# Some fancy Python:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.DataFrame({"A":[1,2,3,4,5,6],"B":[4,7,1,2,3,4]})

plt.figure()
sns.pairplot(data=df)
plt.savefig("x.png")