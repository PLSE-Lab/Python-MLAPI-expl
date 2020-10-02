import matplotlib.pyplot as plt
import pandas as pd

survey = pd.read_csv("../input/Survey.csv")

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns

completed = survey.iloc[[s=="Complete" for s in survey["Status"]],:]

sns.set(style="white", color_codes=True)
sns.countplot(data=completed, x="How often do you use Jupyter Notebook?")
plt.savefig("JupyterNotebookUse.png")